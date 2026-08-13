"""Paired multi-seed dense-budget evaluation on a fresh dev pool."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
from pathlib import Path
from typing import Any, Mapping

from ..environment import Environment
from ..serialization import level_from_dict
from ..signature import static_level_signature
from ..training.transaction import atomic_write_json, sha256_file
from .eval_split import canonical_eval_pool_sha256, load_eval_pool_records


SCHEMA_VERSION = 1
SEMANTICS = "paired_multiseed_midbudget_dev_evaluation_v1"


def _canonical_hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(
        value, sort_keys=True, separators=(",", ":"),
        ensure_ascii=False).encode("utf-8")).hexdigest()


def _quantile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    position = fraction * (len(ordered) - 1)
    lower, upper = math.floor(position), math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _resolve(source: Path, value: str) -> Path:
    path = Path(value)
    if not path.is_absolute():
        path = source.parent / path
    return path.resolve()


def _load_contract(path: str | Path) -> tuple[Path, dict[str, Any]]:
    source = Path(path).resolve()
    contract = json.loads(source.read_text(encoding="utf-8"))
    if contract.get("schema_version") != SCHEMA_VERSION \
            or contract.get("contract_id") != "midbudget_multiseed_eval_v1" \
            or contract.get("status") != "frozen_before_model_evaluation":
        raise ValueError("unsupported mid-budget evaluation contract")
    for name in ("baseline_checkpoint", "candidate_checkpoint", "pool",
                 "pool_seal", "implementation"):
        item = contract[name]
        resolved = _resolve(source, item["path"])
        if not resolved.is_file() or sha256_file(resolved) != item["sha256"]:
            raise ValueError(f"evaluation contract input mismatch: {name}")
        item["resolved_path"] = str(resolved)
    if Path(contract["implementation"]["resolved_path"]) != \
            Path(__file__).resolve():
        raise ValueError("evaluation implementation path mismatch")
    seal = json.loads(Path(
        contract["pool_seal"]["resolved_path"]).read_text(encoding="utf-8"))
    stored = seal.get("seal_sha256")
    payload = {key: value for key, value in seal.items() if key != "seal_sha256"}
    if stored != _canonical_hash(payload) \
            or seal.get("candidate_checkpoint_loaded") is not False \
            or seal.get("excluded_signature_overlap_count") != 0:
        raise ValueError("development pool seal is invalid")
    records = load_eval_pool_records(contract["pool"]["resolved_path"])
    if len(records) != seal["record_count"] \
            or canonical_eval_pool_sha256(records) != \
            seal["canonical_pool_sha256"]:
        raise ValueError("development pool canonical identity mismatch")
    contract["source_path"] = str(source)
    return source, contract


def _evaluate_checkpoint(
    checkpoint_path: str, levels, *, budgets: list[int], trial_seeds: list[int],
    c_puct: float, inference_batch_size: int, virtual_loss: float,
    device: str, role: str, partial_path: Path, evaluation_identity: str,
) -> list[dict[str, Any]]:
    import torch

    from ..search.config import SearchConfig
    from ..search.graph_search import BlocksortAdapter, GraphSearch
    from ..search.seeding import derive_trial_seed, level_search_identity
    from ..training.checkpoint import (
        configs_from_checkpoint, load_checkpoint, model_from_checkpoint)

    env = Environment()
    checkpoint = load_checkpoint(checkpoint_path, map_location="cpu")
    encoding, _model_config, value_norm = configs_from_checkpoint(checkpoint)
    torch_device = torch.device(device)
    model = model_from_checkpoint(checkpoint, map_location=torch_device)
    model.eval()
    adapter = BlocksortAdapter(env, model, encoding, value_norm, torch_device)
    rows: list[dict[str, Any]] = []
    if partial_path.exists():
        partial = json.loads(partial_path.read_text(encoding="utf-8"))
        if partial.get("evaluation_identity") != evaluation_identity \
                or partial.get("role") != role:
            raise ValueError("incompatible evaluation partial")
        rows = partial["rows"]
        for index, row in enumerate(rows):
            if row["static_level_signature"] != \
                    static_level_signature(levels[index]):
                raise ValueError("evaluation partial level mismatch")
        print(f"resuming {role}: {len(rows)}/{len(levels)}", flush=True)
    for index in range(len(rows), len(levels)):
        level = levels[index]
        state = env.initial_state(level)
        level_identity = level_search_identity(env, level)
        budget_results = {}
        for budget_index, budget in enumerate(budgets):
            trials = []
            for trial_index, trial_root in enumerate(trial_seeds):
                seed = derive_trial_seed(
                    trial_root, trial_index=budget_index,
                    level_identity=level_identity,
                    evaluation_context=(
                        f"midbudget_multiseed_v1/trial={trial_index}"))
                config = SearchConfig(
                    simulations=budget, c_puct=c_puct, temperature=0.0,
                    inference_batch_size=inference_batch_size,
                    virtual_loss=virtual_loss,
                    value_normalization_constant=getattr(
                        value_norm, "constant", 20.0), seed=seed)
                result = GraphSearch(adapter, config).run(state)
                trials.append({
                    "trial_index": trial_index,
                    "seed": seed,
                    "solved": bool(result.solved),
                    "solution_length": result.solution_length,
                    "termination_reason": result.termination_reason,
                })
            budget_results[str(budget)] = {
                "trials": trials,
                "solve_rate": sum(item["solved"] for item in trials)
                    / len(trials),
            }
        rows.append({
            "index": index,
            "static_level_signature": static_level_signature(level),
            "budgets": budget_results,
        })
        atomic_write_json(partial_path, {
            "schema_version": SCHEMA_VERSION,
            "evaluation_identity": evaluation_identity,
            "role": role,
            "rows": rows,
        })
        if (index + 1) % 10 == 0 or index + 1 == len(levels):
            print(f"{role}: {index + 1}/{len(levels)}", flush=True)
    return rows


def _bootstrap_interval(
    deltas: list[float], *, replicates: int, seed: int,
) -> dict[str, float]:
    rng = random.Random(seed)
    count = len(deltas)
    samples = [sum(
        deltas[rng.randrange(count)] for _ in range(count)) / count
        for _ in range(replicates)]
    return {
        "lower_95": _quantile(samples, 0.025),
        "upper_95": _quantile(samples, 0.975),
        "bootstrap_mean": sum(samples) / len(samples),
    }


def summarize(
    baseline_rows: list[Mapping[str, Any]],
    candidate_rows: list[Mapping[str, Any]], *, budgets: list[int],
    trial_count: int, bootstrap_replicates: int, bootstrap_seed: int,
) -> dict[str, Any]:
    baseline = {row["static_level_signature"]: row for row in baseline_rows}
    candidate = {row["static_level_signature"]: row for row in candidate_rows}
    if set(baseline) != set(candidate) or not baseline:
        raise ValueError("paired evaluation levels do not match")
    signatures = sorted(baseline)
    per_budget = {}
    curve_level_deltas = {signature: [] for signature in signatures}
    for budget_index, budget in enumerate(budgets):
        level_deltas = []
        baseline_solved = candidate_solved = 0
        wins = losses = ties = 0
        for signature in signatures:
            base_item = baseline[signature]["budgets"][str(budget)]
            cand_item = candidate[signature]["budgets"][str(budget)]
            base_rate = float(base_item["solve_rate"])
            cand_rate = float(cand_item["solve_rate"])
            delta = cand_rate - base_rate
            level_deltas.append(delta)
            curve_level_deltas[signature].append(delta)
            baseline_solved += sum(
                bool(item["solved"]) for item in base_item["trials"])
            candidate_solved += sum(
                bool(item["solved"]) for item in cand_item["trials"])
            wins += delta > 0
            losses += delta < 0
            ties += delta == 0
        total_trials = len(signatures) * trial_count
        delta = sum(level_deltas) / len(level_deltas)
        interval = _bootstrap_interval(
            level_deltas, replicates=bootstrap_replicates,
            seed=bootstrap_seed + budget_index)
        per_budget[str(budget)] = {
            "levels": len(signatures), "trials_per_level": trial_count,
            "baseline_solved_trials": baseline_solved,
            "candidate_solved_trials": candidate_solved,
            "total_trials": total_trials,
            "baseline_solve_rate": baseline_solved / total_trials,
            "candidate_solve_rate": candidate_solved / total_trials,
            "solve_rate_delta": delta,
            "paired_level_wins": wins, "paired_level_losses": losses,
            "paired_level_ties": ties,
            "paired_bootstrap_95_interval": interval,
            "direction": "improved" if delta > 0 else
                "regressed" if delta < 0 else "tied",
            "negative_delta_supported": interval["upper_95"] < 0,
            "positive_delta_supported": interval["lower_95"] > 0,
        }
    aggregate_level_deltas = [
        sum(curve_level_deltas[signature]) / len(budgets)
        for signature in signatures]
    aggregate_delta = sum(aggregate_level_deltas) / len(signatures)
    aggregate_interval = _bootstrap_interval(
        aggregate_level_deltas, replicates=bootstrap_replicates,
        seed=bootstrap_seed + len(budgets))
    at_95 = per_budget["95"]
    return {
        "levels": len(signatures), "budgets": budgets,
        "trials_per_level_per_budget": trial_count,
        "per_budget": per_budget,
        "equal_budget_weighted_curve_delta": aggregate_delta,
        "aggregate_paired_bootstrap_95_interval": aggregate_interval,
        "budget_95_exploratory_assessment": {
            "delta": at_95["solve_rate_delta"],
            "direction_reproduced": at_95["solve_rate_delta"] < 0,
            "negative_delta_statistically_supported":
                at_95["negative_delta_supported"],
            "interpretation": (
                "midbudget_weakness_reproduced_with_negative_interval"
                if at_95["negative_delta_supported"] else
                "negative_direction_only_not_resolved"
                if at_95["solve_rate_delta"] < 0 else
                "midbudget_weakness_not_reproduced"),
        },
    }


def run_evaluation(
    *, contract_path: str | Path, output_dir: str | Path,
    device: str = "cuda",
) -> dict[str, Any]:
    contract_source, contract = _load_contract(contract_path)
    records = load_eval_pool_records(contract["pool"]["resolved_path"])
    levels = [level_from_dict(record["level"]) for record in records]
    evaluation = contract["evaluation"]
    identity_payload = {
        "semantics": SEMANTICS,
        "implementation_sha256": sha256_file(Path(__file__)),
        "contract_sha256": sha256_file(contract_source),
        "pool_sha256": contract["pool"]["sha256"],
        "baseline_sha256": contract["baseline_checkpoint"]["sha256"],
        "candidate_sha256": contract["candidate_checkpoint"]["sha256"],
        "evaluation": evaluation,
    }
    identity = _canonical_hash(identity_payload)
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    baseline_rows = _evaluate_checkpoint(
        contract["baseline_checkpoint"]["resolved_path"], levels,
        budgets=evaluation["budgets"],
        trial_seeds=evaluation["trial_seeds"],
        c_puct=evaluation["c_puct"],
        inference_batch_size=evaluation["inference_batch_size"],
        virtual_loss=evaluation["virtual_loss"], device=device,
        role="baseline", partial_path=destination / "baseline.partial.json",
        evaluation_identity=identity)
    candidate_rows = _evaluate_checkpoint(
        contract["candidate_checkpoint"]["resolved_path"], levels,
        budgets=evaluation["budgets"],
        trial_seeds=evaluation["trial_seeds"],
        c_puct=evaluation["c_puct"],
        inference_batch_size=evaluation["inference_batch_size"],
        virtual_loss=evaluation["virtual_loss"], device=device,
        role="candidate", partial_path=destination / "candidate.partial.json",
        evaluation_identity=identity)
    summary = summarize(
        baseline_rows, candidate_rows, budgets=evaluation["budgets"],
        trial_count=len(evaluation["trial_seeds"]),
        bootstrap_replicates=evaluation["bootstrap_replicates"],
        bootstrap_seed=evaluation["bootstrap_seed"])
    result = {
        "schema_version": SCHEMA_VERSION, "semantics": SEMANTICS,
        "evaluation_identity": identity, "identity": identity_payload,
        "exploratory_only": True,
        "sealed_final_test_loaded": False,
        "summary": summary,
        "baseline_rows": baseline_rows, "candidate_rows": candidate_rows,
    }
    result_path = destination / "result.json"
    atomic_write_json(result_path, result)
    atomic_write_json(destination / "summary.json", {
        "schema_version": SCHEMA_VERSION, "semantics": SEMANTICS,
        "evaluation_identity": identity,
        "result_sha256": sha256_file(result_path),
        "exploratory_only": True,
        "sealed_final_test_loaded": False,
        "summary": summary,
    })
    return result


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cuda")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    result = run_evaluation(
        contract_path=args.contract, output_dir=args.output_dir,
        device=args.device)
    print(json.dumps(result["summary"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
