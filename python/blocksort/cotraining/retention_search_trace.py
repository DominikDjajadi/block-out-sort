"""Deterministically trace paired retention failures across a dense budget curve."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch

from ..environment import Environment
from ..search.config import SearchConfig
from ..search.graph_search import BlocksortAdapter, GraphSearch
from ..search.seeding import derive_trial_seed, level_search_identity
from ..serialization import level_from_dict
from ..training.checkpoint import (
    configs_from_checkpoint,
    load_checkpoint,
    model_from_checkpoint,
)
from ..training.experiment_identity import hash_canonical_value
from ..training.transaction import atomic_write_json, sha256_file
from .replay_anchor_sweep import (
    _load_champion_retention_cache,
    _load_json,
)
from .retention import load_retention_pool
from .search_trace import _paired_report, _run_traced_search


SCHEMA_VERSION = 1
SEMANTICS = "deterministic_retention_discordance_trace_v1"
SUPPORTED_SWEEP_SEMANTICS = frozenset((
    "matched_baseline_difficulty_replay_anchor_sweep_v1",
    "matched_fixed_champion_anchor_distillation_sweep_v1",
))


@dataclass(frozen=True)
class RetentionSearchTraceConfig:
    sweep_run: str
    arm: str
    round_number: int
    output_dir: str
    difficulty_stratum: str = "first_solved_95_through_112"
    trace_budget: int = 95
    dense_budgets: tuple[int, ...] = (
        64, 72, 80, 88, 95, 104, 112, 128, 160)
    device: str = "cuda"

    def validate(self) -> None:
        root = Path(self.sweep_run)
        if not root.is_dir() or not (root / "experiment.json").is_file():
            raise ValueError("sweep run lacks experiment metadata")
        if not self.arm:
            raise ValueError("arm must not be empty")
        if self.round_number <= 0:
            raise ValueError("round number must be positive")
        if self.trace_budget not in self.dense_budgets:
            raise ValueError("trace budget must be in dense budgets")
        if (tuple(sorted(set(self.dense_budgets))) != self.dense_budgets
                or any(value <= 0 for value in self.dense_budgets)):
            raise ValueError(
                "dense budgets must be unique increasing positive integers")


def _resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    return torch.device(name)


def _paired_outcomes(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {str(row["static_level_signature"]): row for row in rows}


def _selection_divergence_detail(
    incumbent: dict[str, Any],
    candidate: dict[str, Any],
) -> dict[str, Any] | None:
    for left, right in zip(incumbent["timeline"], candidate["timeline"]):
        if left["path_locators"] == right["path_locators"]:
            continue
        common = 0
        for left_locator, right_locator in zip(
                left["path_locators"], right["path_locators"]):
            if left_locator != right_locator:
                break
            common += 1
        left_steps = left.get("selection_trace", [])
        right_steps = right.get("selection_trace", [])
        left_step = left_steps[common] if common < len(left_steps) else None
        right_step = right_steps[common] if common < len(right_steps) else None
        comparison = None
        if left_step is not None and right_step is not None:
            if left_step["node_key"] != right_step["node_key"]:
                raise RuntimeError(
                    "paired searches disagree on the common divergence node")
            if left_step["locators"] != right_step["locators"]:
                raise RuntimeError(
                    "paired searches expose different legal actions at a "
                    "shared state")
            prior_delta = [
                float(candidate_prior) - float(incumbent_prior)
                for incumbent_prior, candidate_prior in zip(
                    left_step["priors"], right_step["priors"])
            ]
            comparison = {
                "node_key": left_step["node_key"],
                "node_value_cost_delta": (
                    float(right_step["node_value_cost"])
                    - float(left_step["node_value_cost"])),
                "prior_l1_distance": sum(abs(value) for value in prior_delta),
                "max_prior_absolute_delta": max(
                    (abs(value) for value in prior_delta), default=0.0),
                "prior_delta_candidate_minus_incumbent": prior_delta,
                "incumbent_selected_edge":
                    left_step["selected_edge_index"],
                "candidate_selected_edge":
                    right_step["selected_edge_index"],
                "incumbent_selected_locator":
                    left_step["selected_locator"],
                "candidate_selected_locator":
                    right_step["selected_locator"],
                "incumbent": left_step,
                "candidate": right_step,
            }
        return {
            "simulation": int(left["simulation"]),
            "divergence_depth": common,
            "kind": "root_selection" if common == 0 else "deeper_selection",
            "shared_selection_node": comparison,
        }
    return None


def _root_expansion_order(trace: dict[str, Any]) -> list[dict[str, int]]:
    first: dict[int, int] = {}
    for row in trace["timeline"]:
        index = row.get("root_edge_index")
        if index is not None and int(index) not in first:
            first[int(index)] = int(row["simulation"])
    return [
        {"edge_index": index, "first_selected_simulation": simulation}
        for index, simulation in sorted(first.items(), key=lambda item: item[1])
    ]


def _dense_curve(
    adapter,
    state,
    *,
    budgets: tuple[int, ...],
    c_puct: float,
    value_constant: float,
    seed: int,
) -> dict[str, Any]:
    outcomes = {}
    first_solution_by_budget = {}
    for budget in budgets:
        result = GraphSearch(adapter, SearchConfig(
            simulations=budget,
            c_puct=c_puct,
            temperature=0.0,
            value_normalization_constant=value_constant,
            seed=seed,
        )).run(state)
        outcomes[str(budget)] = {
            "solved": bool(result.solved),
            "solution_length": result.solution_length,
            "first_solution_simulation": result.first_solution_simulation,
            "termination_reason": result.termination_reason,
        }
        first_solution_by_budget[str(budget)] = (
            int(result.first_solution_simulation)
            if result.first_solution_simulation is not None else None)
    return {
        "outcomes": outcomes,
        "first_solution_simulation_by_budget": first_solution_by_budget,
    }


def _delay_classification(
    direction: str,
    incumbent_curve: dict[str, Any],
    candidate_curve: dict[str, Any],
    *,
    trace_budget: int = 95,
) -> dict[str, Any]:
    incumbent_solved = [
        (int(budget), row["first_solution_simulation"])
        for budget, row in incumbent_curve["outcomes"].items()
        if row["solved"]]
    candidate_solved = [
        (int(budget), row["first_solution_simulation"])
        for budget, row in candidate_curve["outcomes"].items()
        if row["solved"]]
    incumbent_first = min(incumbent_solved, default=None)
    candidate_first = min(candidate_solved, default=None)
    if direction == "incumbent_only":
        later_candidate = [
            row for row in candidate_solved if row[0] > trace_budget]
        earlier_candidate = [
            row for row in candidate_solved if row[0] < trace_budget]
        if later_candidate:
            kind = "candidate_search_delay"
            delay = min(later_candidate)[0] - trace_budget
        elif earlier_candidate:
            kind = "candidate_nonmonotonic_budget_failure"
            delay = None
        else:
            kind = "candidate_absent_through_max_budget"
            delay = None
    else:
        later_incumbent = [
            row for row in incumbent_solved if row[0] > trace_budget]
        earlier_incumbent = [
            row for row in incumbent_solved if row[0] < trace_budget]
        if later_incumbent:
            kind = "incumbent_search_delay"
            delay = min(later_incumbent)[0] - trace_budget
        elif earlier_incumbent:
            kind = "incumbent_nonmonotonic_budget_failure"
            delay = None
        else:
            kind = "incumbent_absent_through_max_budget"
            delay = None
    return {
        "kind": kind,
        "incumbent_first_solved_budget_and_simulation": incumbent_first,
        "candidate_first_solved_budget_and_simulation": candidate_first,
        "delayed_side_additional_budget": delay,
    }


def run_retention_search_trace(
    cfg: RetentionSearchTraceConfig,
) -> dict[str, Any]:
    cfg.validate()
    sweep = Path(cfg.sweep_run)
    experiment = _load_json(sweep / "experiment.json")
    if experiment.get("semantics") not in SUPPORTED_SWEEP_SEMANTICS:
        raise ValueError("unsupported sweep semantics")
    experiment_config = experiment["config"]
    source = Path(experiment_config["source_run"])
    source_config = _load_json(source / "config.json")
    champion_path = Path(experiment_config["champion_checkpoint"])
    round_dir = sweep / cfg.arm / f"round_{cfg.round_number:03d}"
    candidate_path = round_dir / "candidate.pt"
    retention_rows_path = round_dir / "retention_rows.json"
    retention_report_path = round_dir / "retention.json"
    for label, path in (
            ("champion checkpoint", champion_path),
            ("candidate checkpoint", candidate_path),
            ("retention rows", retention_rows_path),
            ("retention report", retention_report_path)):
        if not path.is_file():
            raise ValueError(f"{label} does not exist: {path}")

    retention_records, _all_signatures, retention_manifest = \
        load_retention_pool(
            experiment_config["retention_dataset"],
            per_band=int(source_config["learner_retention_per_band"]),
        )
    budgets = [int(value) for value in
               source_config["learner_retention_budgets"]]
    if cfg.trace_budget not in budgets:
        raise ValueError("trace budget was not in the source retention check")
    retention_seed = int(source_config["seed"])
    champion_sha = sha256_file(champion_path)
    reference_rows = _load_champion_retention_cache(
        source,
        champion_sha256=champion_sha,
        selected=retention_records,
        budgets=budgets,
        seed=retention_seed,
    )
    candidate_payload = _load_json(retention_rows_path)
    if (candidate_payload.get("checkpoint_sha256")
            != sha256_file(candidate_path)):
        raise RuntimeError("candidate retention rows target another checkpoint")
    candidate_rows = candidate_payload["rows"]
    reference = _paired_outcomes(reference_rows)
    candidate = _paired_outcomes(candidate_rows)
    by_signature = {
        row["static_level_signature"]: row for row in retention_records}
    discordant = []
    budget_key = str(cfg.trace_budget)
    for signature in sorted(reference):
        if reference[signature]["difficulty_stratum"] \
                != cfg.difficulty_stratum:
            continue
        left = bool(reference[signature]["budgets"][budget_key]["solved"])
        right = bool(candidate[signature]["budgets"][budget_key]["solved"])
        if left == right:
            continue
        discordant.append({
            "static_level_signature": signature,
            "direction": "incumbent_only" if left else "candidate_only",
            "incumbent_solved": left,
            "candidate_solved": right,
        })
    if not discordant:
        raise ValueError("selected retention cell has no discordant outcomes")

    device = _resolve_device(cfg.device)
    loaded = {
        "incumbent": load_checkpoint(champion_path, map_location="cpu"),
        "candidate": load_checkpoint(candidate_path, map_location="cpu"),
    }
    configs = {
        role: configs_from_checkpoint(checkpoint)
        for role, checkpoint in loaded.items()
    }
    if tuple(config.to_dict() for config in configs["incumbent"]) != tuple(
            config.to_dict() for config in configs["candidate"]):
        raise ValueError("checkpoint model configurations differ")
    encoding, _model_config, value_norm = configs["incumbent"]
    models = {
        role: model_from_checkpoint(checkpoint, map_location=device)
        for role, checkpoint in loaded.items()
    }
    env = Environment()
    adapters = {
        role: BlocksortAdapter(
            env, model, encoding, value_norm, device)
        for role, model in models.items()
    }
    c_puct = 1.5
    fingerprint_payload = {
        "schema_version": SCHEMA_VERSION,
        "semantics": SEMANTICS,
        "config": asdict(cfg),
        "inputs": {
            "sweep_experiment_sha256": sha256_file(
                sweep / "experiment.json"),
            "champion_checkpoint_sha256": champion_sha,
            "candidate_checkpoint_sha256": sha256_file(candidate_path),
            "retention_rows_sha256": sha256_file(retention_rows_path),
            "retention_report_sha256": sha256_file(retention_report_path),
            "retention_dataset_sha256": sha256_file(
                experiment_config["retention_dataset"]),
        },
        "search": {
            "temperature": 0.0,
            "c_puct": c_puct,
            "inference_batch_size": 8,
            "stochasticity": "none_seed_inert_at_temperature_zero",
        },
        "discordant": discordant,
    }
    fingerprint = hash_canonical_value(fingerprint_payload)
    output = Path(cfg.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    experiment_path = output / "experiment.json"
    identity = {**fingerprint_payload, "fingerprint": fingerprint}
    if experiment_path.exists():
        if _load_json(experiment_path) != identity:
            raise RuntimeError("output directory belongs to another trace")
    else:
        atomic_write_json(experiment_path, identity)
    levels_dir = output / "levels"
    levels_dir.mkdir(parents=True, exist_ok=True)

    summary_rows = []
    original_trial_index = budgets.index(cfg.trace_budget)
    for index, item in enumerate(discordant, start=1):
        signature = item["static_level_signature"]
        path = levels_dir / f"{signature}.json"
        if path.exists():
            level_result = _load_json(path)
            if level_result.get("experiment_fingerprint") != fingerprint:
                raise RuntimeError("cached level trace has incompatible identity")
        else:
            level_record = by_signature[signature]
            level = level_from_dict(level_record["level"])
            state = env.initial_state(level)
            search_identity = level_search_identity(env, level)
            trial_seed = derive_trial_seed(
                retention_seed,
                trial_index=original_trial_index,
                level_identity=search_identity,
                evaluation_context="shadow_learner_retention_v1",
            )
            traces = {}
            curves = {}
            for role in ("incumbent", "candidate"):
                trace_config = SearchConfig(
                    simulations=cfg.trace_budget,
                    c_puct=c_puct,
                    temperature=0.0,
                    value_normalization_constant=getattr(
                        value_norm, "constant", 20.0),
                    seed=trial_seed,
                )
                traces[role] = _run_traced_search(
                    adapters[role], state, config=trace_config)
                curves[role] = _dense_curve(
                    adapters[role],
                    state,
                    budgets=cfg.dense_budgets,
                    c_puct=c_puct,
                    value_constant=getattr(value_norm, "constant", 20.0),
                    seed=trial_seed,
                )
            if (traces["incumbent"]["final"]["solved"]
                    != item["incumbent_solved"]
                    or traces["candidate"]["final"]["solved"]
                    != item["candidate_solved"]):
                raise RuntimeError(
                    "trace failed to reproduce saved retention outcome")
            pair = _paired_report(traces["incumbent"], traces["candidate"])
            divergence = _selection_divergence_detail(
                traces["incumbent"], traces["candidate"])
            delay = _delay_classification(
                item["direction"], curves["incumbent"], curves["candidate"],
                trace_budget=cfg.trace_budget)
            level_result = {
                "schema_version": SCHEMA_VERSION,
                "semantics": SEMANTICS,
                "experiment_fingerprint": fingerprint,
                **item,
                "search_identity": search_identity,
                "trace_seed": trial_seed,
                "dense_budgets": list(cfg.dense_budgets),
                "dense_curve": curves,
                "delay_classification": delay,
                "first_selection_divergence": divergence,
                "root_expansion_order": {
                    role: _root_expansion_order(trace)
                    for role, trace in traces.items()
                },
                "paired": pair,
                "traces": traces,
            }
            atomic_write_json(path, level_result)
        divergence = level_result["first_selection_divergence"]
        shared = (
            divergence.get("shared_selection_node")
            if divergence is not None else None)
        summary_rows.append({
            "static_level_signature": signature,
            "direction": item["direction"],
            "delay_classification": level_result["delay_classification"],
            "first_divergence_simulation": (
                divergence["simulation"] if divergence else None),
            "first_divergence_depth": (
                divergence["divergence_depth"] if divergence else None),
            "first_divergence_kind": (
                divergence["kind"] if divergence else None),
            "divergence_prior_l1": (
                shared["prior_l1_distance"] if shared else None),
            "divergence_max_prior_delta": (
                shared["max_prior_absolute_delta"] if shared else None),
            "shared_state_value_delta": (
                shared["node_value_cost_delta"] if shared else None),
            "trace_file": str(path),
            "trace_file_sha256": sha256_file(path),
        })
        print(
            f"traced {index}/{len(discordant)} retention discordances",
            flush=True,
        )

    direction_counts = Counter(row["direction"] for row in summary_rows)
    delay_counts = Counter(
        row["delay_classification"]["kind"] for row in summary_rows)
    divergence_counts = Counter(
        row["first_divergence_kind"] for row in summary_rows)
    prior_l1 = [
        float(row["divergence_prior_l1"])
        for row in summary_rows
        if row["divergence_prior_l1"] is not None
    ]
    value_deltas = [
        abs(float(row["shared_state_value_delta"]))
        for row in summary_rows
        if row["shared_state_value_delta"] is not None
    ]
    result = {
        "schema_version": SCHEMA_VERSION,
        "semantics": SEMANTICS,
        "experiment_fingerprint": fingerprint,
        "source": {
            "sweep_run": cfg.sweep_run,
            "arm": cfg.arm,
            "round": cfg.round_number,
            "difficulty_stratum": cfg.difficulty_stratum,
            "trace_budget": cfg.trace_budget,
            "retention_manifest": retention_manifest,
        },
        "discordant_level_count": len(summary_rows),
        "direction_counts": dict(direction_counts),
        "delay_classification_counts": dict(delay_counts),
        "first_divergence_kind_counts": dict(divergence_counts),
        "mean_divergence_prior_l1": (
            sum(prior_l1) / len(prior_l1) if prior_l1 else None),
        "max_shared_state_value_absolute_delta": max(
            value_deltas, default=0.0),
        "levels": summary_rows,
        "final_test_status": "sealed_not_loaded_or_evaluated",
    }
    atomic_write_json(output / "summary.json", result)
    return result


def _parse_args(argv: list[str] | None = None) -> RetentionSearchTraceConfig:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sweep-run", required=True)
    parser.add_argument("--arm", required=True)
    parser.add_argument("--round", dest="round_number", type=int, required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--difficulty-stratum",
        default="first_solved_95_through_112")
    parser.add_argument("--trace-budget", type=int, default=95)
    parser.add_argument(
        "--dense-budgets", type=int, nargs="+",
        default=[64, 72, 80, 88, 95, 104, 112, 128, 160])
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args(argv)
    args.dense_budgets = tuple(args.dense_budgets)
    return RetentionSearchTraceConfig(**vars(args))


def main(argv: list[str] | None = None) -> int:
    result = run_retention_search_trace(_parse_args(argv))
    print(json.dumps({
        key: result[key]
        for key in (
            "discordant_level_count", "direction_counts",
            "delay_classification_counts", "first_divergence_kind_counts",
            "mean_divergence_prior_l1",
            "max_shared_state_value_absolute_delta",
            "final_test_status")
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
