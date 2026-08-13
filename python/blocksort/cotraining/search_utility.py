"""Cache bounded-search utility labels among oracle-optimal actions.

This is an offline experiment tool, not part of live co-training. It evaluates
each fully exact optimal successor with a frozen teacher and writes utilities
aligned with the record's legal actions. Training can then add a pairwise loss
that prefers optimal actions whose successors are easier for the actual
bounded graph search.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch

from ..dataset.schema import LABEL_FULL_EXACT, deserialize_state
from ..environment import Environment
from ..expert_iteration.records import dedup_key
from ..expert_iteration.train import POLICY_SEARCH_UTILITY_FIELD
from ..search.config import SearchConfig
from ..search.graph_search import BlocksortAdapter, GraphSearch
from ..search.seeding import derive_trial_seed
from ..serialization import level_from_dict
from ..training.action_encoding import decode_action, normalized_action_index
from ..training.checkpoint import (
    configs_from_checkpoint,
    load_checkpoint,
    model_from_checkpoint,
)
from ..training.dataset import load_records
from ..training.transaction import atomic_write_json, atomic_write_text, sha256_file


SCHEMA_VERSION = 2
SEMANTICS = "frozen_teacher_optimal_successor_budget_utility_v2"


@dataclass(frozen=True)
class SearchUtilityConfig:
    sample_jsonl: str
    teacher_checkpoint: str
    output_dir: str
    budgets: tuple[int, ...] = (4, 8, 16)
    budget_weights: tuple[float, ...] = (0.2, 0.3, 0.5)
    c_puct: float = 1.5
    inference_batch_size: int = 8
    virtual_loss: float = 1.0
    seed: int = 2_065
    checkpoint_interval: int = 25
    device: str = "cpu"
    max_unique_states: int | None = None
    preferred_iteration: int | None = None

    def validate(self) -> None:
        for label, path in (
                ("sample JSONL", self.sample_jsonl),
                ("teacher checkpoint", self.teacher_checkpoint)):
            if not Path(path).is_file():
                raise ValueError(f"{label} does not exist: {path}")
        if not self.budgets:
            raise ValueError("at least one search-utility budget is required")
        if len(self.budgets) != len(self.budget_weights):
            raise ValueError("search-utility budgets and weights must align")
        if any(isinstance(value, bool) or not isinstance(value, int)
               or value <= 0 for value in self.budgets):
            raise ValueError("search-utility budgets must be positive integers")
        if tuple(sorted(set(self.budgets))) != self.budgets:
            raise ValueError(
                "search-utility budgets must be strictly increasing")
        if any(not math.isfinite(value) or value < 0
               for value in self.budget_weights):
            raise ValueError(
                "search-utility budget weights must be finite and non-negative")
        if not math.isclose(sum(self.budget_weights), 1.0, abs_tol=1e-9):
            raise ValueError("search-utility budget weights must sum to 1")
        if not math.isfinite(self.c_puct) or self.c_puct < 0:
            raise ValueError("c_puct must be finite and non-negative")
        if (isinstance(self.inference_batch_size, bool)
                or not isinstance(self.inference_batch_size, int)
                or self.inference_batch_size <= 0):
            raise ValueError("inference batch size must be a positive integer")
        if not math.isfinite(self.virtual_loss) or self.virtual_loss < 0:
            raise ValueError("virtual loss must be finite and non-negative")
        if isinstance(self.seed, bool) or not isinstance(self.seed, int):
            raise ValueError("seed must be an integer")
        if (isinstance(self.checkpoint_interval, bool)
                or not isinstance(self.checkpoint_interval, int)
                or self.checkpoint_interval <= 0):
            raise ValueError("checkpoint interval must be a positive integer")
        if self.max_unique_states is not None and (
                isinstance(self.max_unique_states, bool)
                or not isinstance(self.max_unique_states, int)
                or self.max_unique_states <= 0):
            raise ValueError("max unique states must be positive or None")
        if self.preferred_iteration is not None and (
                isinstance(self.preferred_iteration, bool)
                or not isinstance(self.preferred_iteration, int)
                or self.preferred_iteration < 0):
            raise ValueError("preferred iteration must be non-negative or None")


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _canonical_sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _sample_text(records: list[dict[str, Any]]) -> str:
    return "".join(_canonical_json(record) + "\n" for record in records)


def _record_label_key(record: dict[str, Any]) -> str:
    return _canonical_sha256({
        "dedup_key": dedup_key(record),
        "legal_actions": record.get("legal_actions"),
        "optimal_support": [
            float(probability) > 0
            for probability in record.get("policy_target", [])
        ],
    })


def _optimal_indices(record: dict[str, Any]) -> list[int]:
    if record.get("label_kind") != LABEL_FULL_EXACT:
        return []
    if not (
            bool(record.get("value_exact", False))
            and bool(record.get("policy_exact", False))
            and bool(record.get("optimal_actions_complete", False))
            and bool(record.get("action_values_complete", False))):
        return []
    target = record.get("policy_target")
    legal_actions = record.get("legal_actions")
    if not isinstance(target, list) or not isinstance(legal_actions, list):
        return []
    if len(target) != len(legal_actions):
        raise ValueError(
            f"misaligned exact record: {record.get('state_key')}")
    return [index for index, probability in enumerate(target)
            if float(probability) > 0]


def strict_preference_pairs(utilities: list[float | None]) -> int:
    values = [float(value) for value in utilities if value is not None]
    return sum(left > right + 1e-8 for left in values for right in values)


def _resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA requested but unavailable")
    return device


def _identity(cfg: SearchUtilityConfig) -> dict[str, Any]:
    inputs = {
        "sample_sha256": sha256_file(cfg.sample_jsonl),
        "teacher_checkpoint_sha256": sha256_file(cfg.teacher_checkpoint),
    }
    semantic_config = {
        key: value for key, value in asdict(cfg).items()
        if key not in ("sample_jsonl", "teacher_checkpoint", "output_dir",
                       "checkpoint_interval", "device")
    }
    payload = {
        "schema_version": SCHEMA_VERSION,
        "semantics": SEMANTICS,
        "inputs": inputs,
        "semantic_config": semantic_config,
    }
    return {**payload, "fingerprint": _canonical_sha256(payload)}


def _evaluate_record(
    record: dict[str, Any],
    *,
    cfg: SearchUtilityConfig,
    env: Environment,
    adapter: BlocksortAdapter,
    encoding_config,
    value_normalization_constant: float,
) -> dict[str, Any]:
    optimal_indices = _optimal_indices(record)
    utilities: list[float | None] = [None] * len(record["legal_actions"])
    action_rows: list[dict[str, Any]] = []
    if len(optimal_indices) < 2:
        return {
            "utilities": utilities,
            "actions": action_rows,
            "strict_preference_pairs": 0,
        }
    level = level_from_dict(record["level"])
    state = deserialize_state(level, record["state"])
    record_identity = _record_label_key(record)
    for action_offset, index in enumerate(optimal_indices):
        action_index = normalized_action_index(
            record["legal_actions"][index], encoding_config)
        action = decode_action(env, state, action_index, encoding_config)
        successor = env.apply_action(state, action)
        budget_rows: dict[str, dict[str, Any]] = {}
        utility = 0.0
        if env.is_terminal(successor):
            first_solution_simulation = 0
            maximum_result = None
        else:
            maximum_budget = cfg.budgets[-1]
            maximum_seed = derive_trial_seed(
                cfg.seed,
                trial_index=action_offset,
                level_identity=record_identity,
                evaluation_context="search_utility.optimal_successor.max",
            )
            maximum_result = GraphSearch(adapter, SearchConfig(
                simulations=maximum_budget,
                inference_batch_size=cfg.inference_batch_size,
                virtual_loss=cfg.virtual_loss,
                c_puct=cfg.c_puct,
                temperature=0.0,
                value_normalization_constant=value_normalization_constant,
                seed=maximum_seed,
            )).run(successor)
            first_solution_simulation = (
                maximum_result.first_solution_simulation)
            if maximum_result.solved and not maximum_result.solution_verified:
                raise RuntimeError(
                    "search utility accepted an unverified solution")
            if (first_solution_simulation is not None
                    and not maximum_result.solved):
                raise RuntimeError(
                    "search reported a first solution that failed final "
                    "verification")
        for budget_offset, (budget, weight) in enumerate(
                zip(cfg.budgets, cfg.budget_weights)):
            # A max-budget run has the same complete inference batches as every
            # prefix budget divisible by inference_batch_size. A smaller final
            # partial batch can select different leaves, so rerun only those
            # exceptional prefix budgets exactly.
            requires_exact_prefix_run = (
                not env.is_terminal(successor)
                and budget != cfg.budgets[-1]
                and budget % cfg.inference_batch_size != 0)
            if requires_exact_prefix_run:
                trial_seed = derive_trial_seed(
                    cfg.seed,
                    trial_index=(action_offset * len(cfg.budgets)
                                 + budget_offset),
                    level_identity=record_identity,
                    evaluation_context="search_utility.optimal_successor.prefix",
                )
                result = GraphSearch(adapter, SearchConfig(
                    simulations=budget,
                    inference_batch_size=cfg.inference_batch_size,
                    virtual_loss=cfg.virtual_loss,
                    c_puct=cfg.c_puct,
                    temperature=0.0,
                    value_normalization_constant=value_normalization_constant,
                    seed=trial_seed,
                )).run(successor)
                solved = bool(result.solved)
                verified = bool(result.solution_verified)
                solution_length = result.solution_length
                termination_reason = result.termination_reason
                if solved and not verified:
                    raise RuntimeError(
                        "search utility accepted an unverified solution")
            else:
                solved = (
                    first_solution_simulation is not None
                    and first_solution_simulation <= budget)
                verified = solved
                solution_length = (
                    maximum_result.solution_length
                    if solved and maximum_result is not None else 0
                    if solved else None)
                termination_reason = (
                    "solved" if solved else "budget_exhausted")
            if solved:
                utility += float(weight)
            budget_rows[str(budget)] = {
                "solved": solved,
                "solution_verified": verified,
                "solution_length": solution_length,
                "termination_reason": termination_reason,
            }
        utilities[index] = utility
        action_rows.append({
            "legal_action_index": index,
            "utility": utility,
            "budgets": budget_rows,
        })
    return {
        "utilities": utilities,
        "actions": action_rows,
        "strict_preference_pairs": strict_preference_pairs(utilities),
    }


def run_search_utility_labeling(
    cfg: SearchUtilityConfig,
) -> dict[str, Any]:
    cfg.validate()
    root = Path(cfg.output_dir)
    root.mkdir(parents=True, exist_ok=True)
    identity = _identity(cfg)
    identity_path = root / "experiment.json"
    if identity_path.exists():
        persisted_identity = json.loads(
            identity_path.read_text(encoding="utf-8"))
        if persisted_identity != identity:
            raise RuntimeError(
                "search-utility output directory belongs to another experiment")
    else:
        atomic_write_json(identity_path, identity)

    cache_path = root / "cache.json"
    cache: dict[str, Any] = {}
    if cache_path.exists():
        persisted_cache = json.loads(cache_path.read_text(encoding="utf-8"))
        if persisted_cache.get("fingerprint") != identity["fingerprint"]:
            raise RuntimeError("search-utility cache fingerprint mismatch")
        cache = dict(persisted_cache.get("records", {}))

    records = load_records(cfg.sample_jsonl)
    all_unique_eligible: dict[str, dict[str, Any]] = {}
    for record in records:
        if len(_optimal_indices(record)) >= 2:
            all_unique_eligible.setdefault(_record_label_key(record), record)
    ordered_eligible = sorted(
        all_unique_eligible.items(),
        key=lambda item: (
            -int(
                cfg.preferred_iteration is not None
                and int(item[1].get("generation_iteration", 0))
                == cfg.preferred_iteration),
            -float(item[1].get("optimal_remaining_moves", 0)),
            item[0],
        ),
    )
    if cfg.max_unique_states is not None:
        ordered_eligible = ordered_eligible[:cfg.max_unique_states]
    unique_eligible = dict(ordered_eligible)

    checkpoint = load_checkpoint(cfg.teacher_checkpoint, map_location="cpu")
    encoding, _model_config, value_norm = configs_from_checkpoint(checkpoint)
    device = _resolve_device(cfg.device)
    model = model_from_checkpoint(checkpoint, map_location=device)
    model.eval()
    env = Environment()
    adapter = BlocksortAdapter(env, model, encoding, value_norm, device)
    total = len(unique_eligible)
    for position, (key, record) in enumerate(unique_eligible.items(), start=1):
        if key not in cache:
            cache[key] = _evaluate_record(
                record,
                cfg=cfg,
                env=env,
                adapter=adapter,
                encoding_config=encoding,
                value_normalization_constant=float(
                    getattr(value_norm, "constant", 20.0)),
            )
        if position % cfg.checkpoint_interval == 0 or position == total:
            atomic_write_json(cache_path, {
                "schema_version": SCHEMA_VERSION,
                "fingerprint": identity["fingerprint"],
                "records": cache,
            })
            print(
                f"search utility: {position}/{total} unique exact states",
                flush=True,
            )

    augmented: list[dict[str, Any]] = []
    for record in records:
        result = dict(record)
        key = _record_label_key(record)
        cached = cache.get(key)
        if cached is not None:
            result[POLICY_SEARCH_UTILITY_FIELD] = cached["utilities"]
        augmented.append(result)
    output_text = _sample_text(augmented)
    output_path = root / "sample.jsonl"
    if output_path.exists():
        if output_path.read_text(encoding="utf-8") != output_text:
            raise RuntimeError(
                "persisted search-utility sample differs from reconstruction")
    else:
        atomic_write_text(output_path, output_text)

    utility_histogram: Counter[str] = Counter()
    strict_records = strict_pairs = action_labels = 0
    for result in cache.values():
        pairs = int(result["strict_preference_pairs"])
        strict_records += pairs > 0
        strict_pairs += pairs
        for action in result["actions"]:
            action_labels += 1
            utility_histogram[f"{float(action['utility']):.12g}"] += 1
    summary = {
        "schema_version": SCHEMA_VERSION,
        "semantics": SEMANTICS,
        "experiment_fingerprint": identity["fingerprint"],
        "inputs": {
            **identity["inputs"],
            "sample_jsonl": cfg.sample_jsonl,
            "teacher_checkpoint": cfg.teacher_checkpoint,
        },
        "search": identity["semantic_config"],
        "output": {
            "sample_jsonl": str(output_path),
            "sample_sha256": hashlib.sha256(
                output_text.encode("utf-8")).hexdigest(),
            "utility_field": POLICY_SEARCH_UTILITY_FIELD,
        },
        "coverage": {
            "records": len(records),
            "unique_multi_optimal_full_exact_states_available":
                len(all_unique_eligible),
            "unique_multi_optimal_full_exact_states_selected": total,
            "unique_states_with_strict_preferences": strict_records,
            "optimal_successor_action_labels": action_labels,
            "strict_preference_pairs": strict_pairs,
            "utility_histogram": dict(sorted(utility_histogram.items())),
        },
        "methodology": {
            "eligible_actions": "complete_full_exact_optimal_support_only",
            "utility": "weighted_solve_indicator_across_successor_budgets",
            "ties": "abstain",
            "teacher_frozen": True,
            "root_noise": False,
            "selection": (
                "preferred_iteration_then_descending_exact_depth_then_hash"),
            "preferred_iteration": cfg.preferred_iteration,
            "max_unique_states": cfg.max_unique_states,
            "final_test_evaluated": False,
        },
    }
    atomic_write_json(root / "summary.json", summary)
    return summary


def _comma_ints(value: str) -> tuple[int, ...]:
    try:
        return tuple(int(part) for part in value.split(","))
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "expected comma-separated integers") from exc


def _comma_floats(value: str) -> tuple[float, ...]:
    try:
        return tuple(float(part) for part in value.split(","))
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "expected comma-separated numbers") from exc


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Cache bounded-search utilities for fully exact optimal successors."))
    parser.add_argument("--sample-jsonl", required=True)
    parser.add_argument("--teacher-checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--budgets", type=_comma_ints, default=(4, 8, 16))
    parser.add_argument(
        "--budget-weights", type=_comma_floats, default=(0.2, 0.3, 0.5))
    parser.add_argument("--c-puct", type=float, default=1.5)
    parser.add_argument("--inference-batch-size", type=int, default=8)
    parser.add_argument("--virtual-loss", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=2_065)
    parser.add_argument("--checkpoint-interval", type=int, default=25)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--max-unique-states", type=int, default=None)
    parser.add_argument("--preferred-iteration", type=int, default=None)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    summary = run_search_utility_labeling(SearchUtilityConfig(
        sample_jsonl=args.sample_jsonl,
        teacher_checkpoint=args.teacher_checkpoint,
        output_dir=args.output_dir,
        budgets=args.budgets,
        budget_weights=args.budget_weights,
        c_puct=args.c_puct,
        inference_batch_size=args.inference_batch_size,
        virtual_loss=args.virtual_loss,
        seed=args.seed,
        checkpoint_interval=args.checkpoint_interval,
        device=args.device,
        max_unique_states=args.max_unique_states,
        preferred_iteration=args.preferred_iteration,
    ))
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
