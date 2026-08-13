"""Export leakage-free, search-critical pairwise policy preferences.

The exporter screens a training-only level pool with two frozen checkpoints at
one bounded-search budget.  For solve/no-solve disagreements it traces the
first deterministic search divergence, proves both competing actions optimal
with the exact oracle, and emits one weak local preference for the branch used
by the model that solved within the budget.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import torch

from ..conformance import _normalized_hashable
from ..dataset.schema import build_record
from ..dataset.training_pool import load_excluded_signatures
from ..environment import Environment
from ..oracle import Oracle, StateAnalysis
from ..search.config import SearchConfig
from ..search.graph_search import BlocksortAdapter, GraphSearch
from ..search.seeding import derive_trial_seed, level_search_identity
from ..serialization import level_from_dict
from ..signature import static_level_signature
from ..solution import deserialize_action
from ..training.checkpoint import (
    configs_from_checkpoint,
    load_checkpoint,
    model_from_checkpoint,
)
from ..training.experiment_identity import hash_canonical_value
from ..training.transaction import (
    atomic_write_json,
    atomic_write_text,
    sha256_file,
)
from .retention_search_trace import (
    _resolve_device,
    _selection_divergence_detail,
)
from .search_trace import _run_traced_search


SCHEMA_VERSION = 1
SEMANTICS = "training_only_bounded_search_trace_preferences_v1"
PREFERENCE_SEMANTICS = "local_weak_pairwise_bounded_search_preference_v1"


@dataclass(frozen=True)
class TraceRankingPoolConfig:
    training_pool: str
    incumbent_checkpoint: str
    candidate_checkpoint: str
    output_dir: str
    exclusion_files: tuple[str, ...]
    budget: int = 95
    seed: int = 8247
    c_puct: float = 1.5
    inference_batch_size: int = 8
    oracle_max_nodes: int = 200_000
    oracle_time_limit_seconds: float | None = None
    device: str = "cuda"

    def validate(self) -> None:
        for label, raw in (
                ("training pool", self.training_pool),
                ("incumbent checkpoint", self.incumbent_checkpoint),
                ("candidate checkpoint", self.candidate_checkpoint)):
            if not Path(raw).is_file():
                raise ValueError(f"{label} does not exist: {raw}")
        if not self.exclusion_files:
            raise ValueError("at least one evaluation exclusion file is required")
        for raw in self.exclusion_files:
            if not Path(raw).is_file():
                raise ValueError(f"exclusion file does not exist: {raw}")
        if self.budget <= 0 or self.inference_batch_size <= 0:
            raise ValueError("budget and inference batch size must be positive")
        if self.c_puct <= 0 or self.oracle_max_nodes <= 0:
            raise ValueError("c_puct and oracle max nodes must be positive")
        if (self.oracle_time_limit_seconds is not None
                and self.oracle_time_limit_seconds <= 0):
            raise ValueError("oracle time limit must be positive when supplied")


def _load_training_levels(
    path: str | Path,
) -> tuple[dict[str, dict[str, Any]], dict[tuple[str, str], dict[str, Any]]]:
    """Load one level representative and every exact record by state identity."""
    levels: dict[str, dict[str, Any]] = {}
    exact_states: dict[tuple[str, str], dict[str, Any]] = {}
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            record = json.loads(line)
            level = level_from_dict(record["level"])
            signature = static_level_signature(level)
            if record.get("static_level_signature") != signature:
                raise ValueError(
                    f"training-pool line {line_number} has inconsistent identity")
            prior = levels.setdefault(signature, record)
            if prior["level"] != record["level"]:
                raise ValueError("one training signature has conflicting levels")
            state_key = record.get("state_key")
            if isinstance(state_key, str):
                exact_states[(signature, state_key)] = record
    if not levels:
        raise ValueError("training pool is empty")
    return levels, exact_states


def _search_outcome(adapter, state, config: SearchConfig) -> dict[str, Any]:
    result = GraphSearch(adapter, config).run(state)
    return {
        "solved": bool(result.solved),
        "solution_length": result.solution_length,
        "first_solution_simulation": result.first_solution_simulation,
        "termination_reason": result.termination_reason,
    }


def _reconstruct_divergence_state(
    env: Environment,
    initial_state,
    trace: dict[str, Any],
    divergence: dict[str, Any],
):
    row = trace["timeline"][int(divergence["simulation"]) - 1]
    state = initial_state
    for locator in row["path_locators"][:int(
            divergence["divergence_depth"])]:
        state = env.apply_action(state, deserialize_action(state, locator))
    shared = divergence.get("shared_selection_node")
    if shared is None:
        raise ValueError("divergence lacks a shared selection node")
    node_key = shared["node_key"]
    expected_state_key = node_key[1] \
        if isinstance(node_key, (list, tuple)) else node_key
    if env.canonical_key(state) != expected_state_key:
        raise RuntimeError("reconstructed divergence state has the wrong key")
    return state


def _analysis_by_action(
    analysis: StateAnalysis,
) -> dict[tuple[Any, ...], Any]:
    return {
        _normalized_hashable(item.serialized): item
        for item in analysis.actions
    }


def _preference_rejection_reason(
    analysis: StateAnalysis,
    preferred_action: dict[str, Any],
    competing_action: dict[str, Any],
) -> str | None:
    if not (
            analysis.exact and analysis.solvable
            and analysis.all_successors_exact):
        return "oracle_incomplete"
    by_action = _analysis_by_action(analysis)
    preferred = by_action.get(_normalized_hashable(preferred_action))
    competing = by_action.get(_normalized_hashable(competing_action))
    if preferred is None or competing is None:
        return "divergent_action_not_legal"
    if not preferred.optimal:
        return "successful_branch_not_oracle_optimal"
    if not competing.optimal:
        return "competing_branch_not_oracle_optimal"
    return None


def _build_preference_record(
    *,
    analysis: StateAnalysis,
    state,
    source_level_id: str,
    direction: str,
    preferred_action: dict[str, Any],
    competing_action: dict[str, Any],
    divergence: dict[str, Any],
    outcomes: dict[str, dict[str, Any]],
    checkpoint_sha256: dict[str, str],
    budget: int,
    trace_seed: int,
    difficulty_stratum: str | None,
) -> dict[str, Any]:
    record = build_record(
        analysis,
        state,
        level_id=source_level_id,
        provenance={
            "sampling": "first-bounded-search-divergence",
            "budget": budget,
            "trace_seed": trace_seed,
        },
    )
    if record is None:
        raise ValueError("exact eligible analysis did not produce a record")
    action_indices = {
        _normalized_hashable(action): index
        for index, action in enumerate(record["legal_actions"])
    }
    preferred_index = action_indices[_normalized_hashable(preferred_action)]
    competing_index = action_indices[_normalized_hashable(competing_action)]
    successful_role = "incumbent" if direction == "incumbent_only" \
        else "candidate"
    unsuccessful_role = "candidate" if successful_role == "incumbent" \
        else "incumbent"
    shared = divergence["shared_selection_node"]
    record["target_source"] = "exact_oracle_with_trace_preference"
    record["trace_preference"] = {
        "schema_version": SCHEMA_VERSION,
        "semantics": PREFERENCE_SEMANTICS,
        "scope": "first_divergent_action_pair_only",
        "strength": "weak_local_preference_margin_set_by_experiment",
        "target_budget": budget,
        "successful_role": successful_role,
        "unsuccessful_role": unsuccessful_role,
        "preferred_action": preferred_action,
        "competing_action": competing_action,
        "preferred_action_index": preferred_index,
        "competing_action_index": competing_index,
        "both_actions_full_exact_oracle_optimal": True,
        "first_divergence_simulation": divergence["simulation"],
        "first_divergence_depth": divergence["divergence_depth"],
        "first_divergence_kind": divergence["kind"],
        "successful_outcome": outcomes[successful_role],
        "unsuccessful_outcome": outcomes[unsuccessful_role],
        "successful_checkpoint_sha256": checkpoint_sha256[successful_role],
        "unsuccessful_checkpoint_sha256": checkpoint_sha256[unsuccessful_role],
        "prior_l1_distance": shared["prior_l1_distance"],
        "max_prior_absolute_delta": shared["max_prior_absolute_delta"],
        "node_value_cost_delta_candidate_minus_incumbent":
            shared["node_value_cost_delta"],
        "difficulty_stratum": difficulty_stratum,
        "trace_seed": trace_seed,
    }
    return record


def _canonical_jsonl(records: Iterable[dict[str, Any]]) -> str:
    return "".join(
        json.dumps(
            record, sort_keys=True, separators=(",", ":"),
            ensure_ascii=False) + "\n"
        for record in records
    )


def build_trace_ranking_pool(
    cfg: TraceRankingPoolConfig,
) -> dict[str, Any]:
    cfg.validate()
    levels, exact_states = _load_training_levels(cfg.training_pool)
    excluded, exclusion_sources = load_excluded_signatures(
        cfg.exclusion_files)
    overlap = sorted(set(levels) & excluded)
    if overlap:
        raise RuntimeError(
            f"training trace pool overlaps {len(overlap)} evaluation levels")

    checkpoints = {
        "incumbent": load_checkpoint(
            cfg.incumbent_checkpoint, map_location="cpu"),
        "candidate": load_checkpoint(
            cfg.candidate_checkpoint, map_location="cpu"),
    }
    checkpoint_sha = {
        "incumbent": sha256_file(cfg.incumbent_checkpoint),
        "candidate": sha256_file(cfg.candidate_checkpoint),
    }
    checkpoint_configs = {
        role: configs_from_checkpoint(checkpoint)
        for role, checkpoint in checkpoints.items()
    }
    if tuple(
            item.to_dict() for item in checkpoint_configs["incumbent"]
    ) != tuple(item.to_dict() for item in checkpoint_configs["candidate"]):
        raise ValueError("checkpoint model configurations differ")

    device = _resolve_device(cfg.device)
    encoding, _model_config, value_norm = checkpoint_configs["incumbent"]
    models = {
        role: model_from_checkpoint(checkpoint, map_location=device)
        for role, checkpoint in checkpoints.items()
    }
    env = Environment()
    adapters = {
        role: BlocksortAdapter(env, model, encoding, value_norm, device)
        for role, model in models.items()
    }
    identity_payload = {
        "schema_version": SCHEMA_VERSION,
        "semantics": SEMANTICS,
        "config": asdict(cfg),
        "inputs": {
            "training_pool_sha256": sha256_file(cfg.training_pool),
            "incumbent_checkpoint_sha256": checkpoint_sha["incumbent"],
            "candidate_checkpoint_sha256": checkpoint_sha["candidate"],
            "evaluation_exclusion_sources": exclusion_sources,
        },
        "contract": {
            "preference_scope": "first_divergent_action_pair_only",
            "oracle_requirement": "both_actions_full_exact_optimal",
            "directionality": "symmetric_by_within_budget_success_role",
            "retention_levels_used_for_training": False,
            "evaluation_outcomes_used": False,
        },
    }
    fingerprint = hash_canonical_value(identity_payload)
    # Normalize tuples (notably exclusion_files) exactly as persisted JSON so
    # an interrupted run can compare and resume its immutable identity.
    identity = json.loads(json.dumps({
        **identity_payload, "fingerprint": fingerprint}))
    output = Path(cfg.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    experiment_path = output / "experiment.json"
    if experiment_path.exists():
        if json.loads(experiment_path.read_text(encoding="utf-8")) != identity:
            raise RuntimeError("output directory belongs to another experiment")
    else:
        atomic_write_json(experiment_path, identity)

    oracle_queries: list[dict[str, Any]] = []
    oracle = Oracle(
        env,
        max_nodes=cfg.oracle_max_nodes,
        time_limit_seconds=cfg.oracle_time_limit_seconds,
        search_observer=oracle_queries.append,
    )
    counters: Counter[str] = Counter()
    direction_counts: Counter[str] = Counter()
    eligible_direction_counts: Counter[str] = Counter()
    eligible_band_counts: Counter[str] = Counter()
    screening_rows = []
    evidence_rows = []
    records = []
    search_config = SearchConfig(
        simulations=cfg.budget,
        c_puct=cfg.c_puct,
        temperature=0.0,
        inference_batch_size=cfg.inference_batch_size,
        value_normalization_constant=getattr(value_norm, "constant", 20.0),
        seed=cfg.seed,
    )

    for index, signature in enumerate(sorted(levels), start=1):
        source = levels[signature]
        level = level_from_dict(source["level"])
        state = env.initial_state(level)
        search_identity = level_search_identity(env, level)
        trial_seed = derive_trial_seed(
            cfg.seed,
            trial_index=0,
            level_identity=search_identity,
            evaluation_context="trace_ranking_pool.screen_v1",
        )
        config = SearchConfig(**{
            **search_config.__dict__, "seed": trial_seed})
        outcomes = {
            role: _search_outcome(adapter, state, config)
            for role, adapter in adapters.items()
        }
        direction = None
        if outcomes["incumbent"]["solved"] \
                != outcomes["candidate"]["solved"]:
            direction = "incumbent_only" \
                if outcomes["incumbent"]["solved"] else "candidate_only"
            direction_counts[direction] += 1
            counters["discordant_levels"] += 1
        screening_rows.append({
            "static_level_signature": signature,
            "difficulty_stratum": source.get(
                "training_anchor", {}).get("difficulty_stratum"),
            "trace_seed": trial_seed,
            "direction": direction,
            "outcomes": outcomes,
        })
        if direction is None:
            counters["concordant_levels"] += 1
            continue

        traces = {
            role: _run_traced_search(adapter, state, config=config)
            for role, adapter in adapters.items()
        }
        for role in ("incumbent", "candidate"):
            if traces[role]["final"]["solved"] != outcomes[role]["solved"]:
                raise RuntimeError("traced search did not reproduce screening")
        divergence = _selection_divergence_detail(
            traces["incumbent"], traces["candidate"])
        evidence: dict[str, Any] = {
            "static_level_signature": signature,
            "direction": direction,
            "difficulty_stratum": source.get(
                "training_anchor", {}).get("difficulty_stratum"),
            "trace_seed": trial_seed,
            "outcomes": outcomes,
            "first_selection_divergence": divergence,
        }
        if divergence is None or divergence.get("shared_selection_node") is None:
            evidence["eligibility"] = "no_shared_selection_divergence"
            counters[evidence["eligibility"]] += 1
            evidence_rows.append(evidence)
            continue
        divergence_state = _reconstruct_divergence_state(
            env, state, traces["incumbent"], divergence)
        successful_role = "incumbent" if direction == "incumbent_only" \
            else "candidate"
        unsuccessful_role = "candidate" if successful_role == "incumbent" \
            else "incumbent"
        shared = divergence["shared_selection_node"]
        preferred_action = shared[f"{successful_role}_selected_locator"]
        competing_action = shared[f"{unsuccessful_role}_selected_locator"]
        state_identity = (signature, env.canonical_key(divergence_state))
        precomputed = exact_states.get(state_identity)
        analysis = oracle.analyze(divergence_state)
        rejection = _preference_rejection_reason(
            analysis, preferred_action, competing_action)
        evidence.update({
            "divergence_state_key": state_identity[1],
            "divergence_state_was_precomputed_exact_training_record":
                precomputed is not None,
            "oracle": {
                "exact": analysis.exact,
                "solvable": analysis.solvable,
                "all_successors_exact": analysis.all_successors_exact,
                "value": analysis.value,
                "preferred_action_optimal": (
                    _analysis_by_action(analysis).get(
                        _normalized_hashable(preferred_action)).optimal
                    if _analysis_by_action(analysis).get(
                        _normalized_hashable(preferred_action)) is not None
                    else None),
                "competing_action_optimal": (
                    _analysis_by_action(analysis).get(
                        _normalized_hashable(competing_action)).optimal
                    if _analysis_by_action(analysis).get(
                        _normalized_hashable(competing_action)) is not None
                    else None),
            },
            "eligibility": "eligible" if rejection is None else rejection,
        })
        counters[evidence["eligibility"]] += 1
        evidence_rows.append(evidence)
        if rejection is not None:
            continue
        record = _build_preference_record(
            analysis=analysis,
            state=divergence_state,
            source_level_id=str(source.get("level_id", signature)),
            direction=direction,
            preferred_action=preferred_action,
            competing_action=competing_action,
            divergence=divergence,
            outcomes=outcomes,
            checkpoint_sha256=checkpoint_sha,
            budget=cfg.budget,
            trace_seed=trial_seed,
            difficulty_stratum=evidence["difficulty_stratum"],
        )
        records.append(record)
        eligible_direction_counts[direction] += 1
        eligible_band_counts[str(evidence["difficulty_stratum"])] += 1
        print(
            f"eligible trace preference {len(records)} from level "
            f"{index}/{len(levels)} ({direction})",
            flush=True,
        )

    records.sort(key=lambda record: (
        record["static_level_signature"], record["state_key"]))
    dataset_path = output / "trace_preferences.jsonl"
    atomic_write_text(dataset_path, _canonical_jsonl(records))
    atomic_write_json(output / "screening_rows.json", screening_rows)
    atomic_write_json(output / "discordance_evidence.json", evidence_rows)
    atomic_write_json(output / "oracle_queries.json", oracle_queries)
    summary = {
        "schema_version": SCHEMA_VERSION,
        "semantics": SEMANTICS,
        "experiment_fingerprint": fingerprint,
        "status": "frozen_training_only_trace_preference_dataset",
        "training_level_count": len(levels),
        "training_exact_state_record_count": len(exact_states),
        "evaluation_excluded_signature_count": len(excluded),
        "evaluation_overlap_count": 0,
        "screening": {
            "budget": cfg.budget,
            "temperature": 0.0,
            "c_puct": cfg.c_puct,
            "inference_batch_size": cfg.inference_batch_size,
            "stochasticity": "none_seed_inert_at_temperature_zero",
            "direction_counts": dict(direction_counts),
            "counters": dict(counters),
        },
        "eligible_preference_count": len(records),
        "eligible_direction_counts": dict(eligible_direction_counts),
        "eligible_difficulty_stratum_counts": dict(eligible_band_counts),
        "oracle_query_count": len(oracle_queries),
        "dataset": {
            "path": str(dataset_path),
            "sha256": sha256_file(dataset_path),
            "canonical_sha256": hash_canonical_value(records),
        },
        "evaluation_policy": {
            "retention_and_final_pool_identities_used_for_exclusion_only": True,
            "evaluation_search_outcomes_used": False,
            "evaluation_states_exported": False,
            "sealed_final_test_rerun": False,
        },
    }
    atomic_write_json(output / "summary.json", summary)
    return summary


def _parse_args(argv: list[str] | None = None) -> TraceRankingPoolConfig:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-pool", required=True)
    parser.add_argument("--incumbent-checkpoint", required=True)
    parser.add_argument("--candidate-checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--exclusion-file", dest="exclusion_files", action="append",
        required=True)
    parser.add_argument("--budget", type=int, default=95)
    parser.add_argument("--seed", type=int, default=8247)
    parser.add_argument("--c-puct", type=float, default=1.5)
    parser.add_argument("--inference-batch-size", type=int, default=8)
    parser.add_argument("--oracle-max-nodes", type=int, default=200_000)
    parser.add_argument("--oracle-time-limit-seconds", type=float)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args(argv)
    args.exclusion_files = tuple(args.exclusion_files)
    return TraceRankingPoolConfig(**vars(args))


def main(argv: list[str] | None = None) -> int:
    result = build_trace_ranking_pool(_parse_args(argv))
    print(json.dumps({
        "training_level_count": result["training_level_count"],
        "evaluation_overlap_count": result["evaluation_overlap_count"],
        "screening": result["screening"],
        "eligible_preference_count": result["eligible_preference_count"],
        "eligible_direction_counts": result["eligible_direction_counts"],
        "eligible_difficulty_stratum_counts":
            result["eligible_difficulty_stratum_counts"],
        "dataset": result["dataset"],
        "evaluation_policy": result["evaluation_policy"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
