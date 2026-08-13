"""Budget-sweep evaluation summaries for protagonist checkpoints."""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from numbers import Real
from pathlib import Path
from typing import Any, Mapping

from ..cotraining.eval_split import (
    load_eval_pool_records, load_eval_split_manifest)
from ..environment import Environment
from ..oracle import Oracle
from ..serialization import level_from_dict
from ..training.transaction import atomic_write_json, sha256_file


def _validated_budgets(budgets: list[int]) -> list[int]:
    if not budgets:
        raise ValueError("budgets must contain at least one value")
    result: list[int] = []
    for budget in budgets:
        if isinstance(budget, bool) or not isinstance(budget, int) or budget <= 0:
            raise ValueError("budgets must contain only positive integers")
        if budget in result:
            raise ValueError("budgets must not contain duplicates")
        result.append(budget)
    return result


def _budget_metrics(report: Mapping[str, Any], budget: int) -> dict[str, Any]:
    budgets = report.get("budgets")
    if not isinstance(budgets, Mapping) or str(budget) not in budgets:
        raise ValueError(f"evaluation report is missing budget {budget}")
    item = budgets[str(budget)]
    total = item.get("total_evaluated_count")
    known = item.get("search_optimal_classification_count")
    confirmed = item.get("search_confirmed_optimal_count")
    regret_known = item.get("search_exact_regret_count")
    solved = item.get("search_solved_count")
    if not all(isinstance(value, int)
               for value in (total, known, confirmed, regret_known, solved)):
        raise ValueError(f"budget {budget} is missing required count fields")
    if any(isinstance(value, bool) or value < 0
           for value in (total, known, confirmed, regret_known, solved)):
        raise ValueError(f"budget {budget} has invalid count fields")
    if (known > total or regret_known > total or confirmed > known
            or solved > total):
        raise ValueError(f"budget {budget} has inconsistent count fields")
    unknown = item.get("search_unknown_classification_count", total - known)
    if (isinstance(unknown, bool) or not isinstance(unknown, int)
            or unknown != total - known):
        raise ValueError(
            f"budget {budget} has inconsistent unknown classification count")

    def checked_ratio(field: str, numerator: int) -> float | None:
        value = item.get(field)
        expected = numerator / total if total else None
        if expected is None:
            if value is not None:
                raise ValueError(
                    f"budget {budget} field {field} must be None for an empty set")
            return None
        if (isinstance(value, bool) or not isinstance(value, Real)
                or not math.isfinite(float(value))
                or not math.isclose(float(value), expected,
                                    rel_tol=1e-9, abs_tol=1e-12)):
            raise ValueError(
                f"budget {budget} field {field} does not match its counts")
        return float(value)

    classification_coverage = checked_ratio(
        "search_optimal_classification_coverage", known)
    confirmed_rate = checked_ratio(
        "search_confirmed_optimal_rate", confirmed)
    solve_rate_total = checked_ratio(
        "search_solve_rate_total", solved)
    exact_regret_coverage = checked_ratio(
        "search_exact_regret_coverage", regret_known)
    return {
        "budget": budget,
        "total_level_count": total,
        "classification_known_count": known,
        "classification_coverage": classification_coverage,
        "confirmed_optimal_count": confirmed,
        "confirmed_optimal_rate_total": confirmed_rate,
        "confirmed_optimal_rate_classified":
            (confirmed / known if known else None),
        "solved_count": solved,
        "solve_rate_total": solve_rate_total,
        "timeout_unknown_count": unknown,
        "exact_regret_count": regret_known,
        "exact_regret_coverage": exact_regret_coverage,
        "mean_regret": item.get("search_mean_regret"),
    }


def summarize_budget_sweep(
    report: Mapping[str, Any],
    *,
    budgets: list[int],
    bucket_reports: Mapping[str, Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Return a stable machine-readable summary for a multi-budget report."""
    budgets = _validated_budgets(budgets)
    per_budget = {str(budget): _budget_metrics(report, budget)
                  for budget in budgets}
    buckets: dict[str, Any] = {}
    for bucket, bucket_report in sorted((bucket_reports or {}).items()):
        buckets[bucket] = {
            str(budget): _budget_metrics(bucket_report, budget)
            for budget in budgets
        }
    for budget in budgets:
        total = sum(
            bucket_summary[str(budget)]["total_level_count"]
            for bucket_summary in buckets.values())
        if buckets and total != per_budget[str(budget)]["total_level_count"]:
            raise ValueError(
                f"bucket breakdown totals {total} do not match budget {budget} "
                f"total {per_budget[str(budget)]['total_level_count']}")
        for field in (
                "classification_known_count", "confirmed_optimal_count",
                "solved_count", "timeout_unknown_count",
                "exact_regret_count"):
            bucket_total = sum(
                bucket_summary[str(budget)][field]
                for bucket_summary in buckets.values())
            if buckets and bucket_total != per_budget[str(budget)][field]:
                raise ValueError(
                    f"bucket breakdown {field} total {bucket_total} does not "
                    f"match budget {budget} total "
                    f"{per_budget[str(budget)][field]}")
    return {
        "schema_version": 1,
        "budgets": list(budgets),
        "per_budget": per_budget,
        "bucket_breakdown": buckets,
    }


def select_evaluation_records(
    pool_path: str | Path,
    *,
    split_manifest_path: str | Path | None = None,
    split_role: str | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any] | None]:
    """Select an explicitly named immutable split role without test peeking."""
    records = load_eval_pool_records(pool_path)
    if split_manifest_path is None:
        raise ValueError(
            "budget-sweep evaluation requires an immutable split manifest")
    if split_role not in ("promotion_validation", "final_test"):
        raise ValueError(
            "an evaluation split manifest requires an explicit split_role of "
            "promotion_validation or final_test")
    manifest = load_eval_split_manifest(split_manifest_path, pool_path)
    by_signature = {
        record["static_level_signature"]: record for record in records}
    selected = [
        by_signature[item["signature"]] for item in manifest[split_role]]
    return selected, {
        "manifest": str(split_manifest_path),
        "evaluation_split_fingerprint":
            manifest["evaluation_split_fingerprint"],
        "pool_sha256": manifest["pool"]["sha256"],
        "split_seed": manifest["split_config"]["split_seed"],
        "role": split_role,
        "selected_count": len(selected),
    }


def evaluate_checkpoint_budget_sweep(
    checkpoint_path: str | Path,
    pool_path: str | Path,
    *,
    budgets: list[int],
    split_manifest_path: str | Path | None = None,
    split_role: str | None = None,
    oracle_max_nodes: int = 20_000,
    oracle_time_limit_seconds: float | None = 3.0,
    device: str = "cpu",
    seed: int = 0,
) -> dict[str, Any]:
    import torch
    from ..training.checkpoint import (
        configs_from_checkpoint, load_checkpoint, model_from_checkpoint)
    from .evaluate import evaluate_checkpoint

    budgets = _validated_budgets(budgets)
    env = Environment()
    records, evaluation_split = select_evaluation_records(
        pool_path, split_manifest_path=split_manifest_path,
        split_role=split_role)
    states = [env.initial_state(level_from_dict(record["level"]))
              for record in records]
    checkpoint = load_checkpoint(checkpoint_path, map_location="cpu")
    enc, _model_cfg, value_norm = configs_from_checkpoint(checkpoint)
    torch_device = torch.device(device)
    model = model_from_checkpoint(checkpoint, map_location=torch_device)
    oracle = Oracle(
        env, max_nodes=oracle_max_nodes,
        time_limit_seconds=oracle_time_limit_seconds)
    full = evaluate_checkpoint(
        env, model, enc, value_norm, states, budgets=budgets, oracle=oracle,
        device=torch_device, c_puct=1.5, seed=seed)

    by_bucket: dict[str, list[int]] = defaultdict(list)
    for index, record in enumerate(records):
        bucket = record.get("generation_bucket")
        if isinstance(bucket, str) and bucket:
            by_bucket[bucket].append(index)
    bucket_reports = {}
    for bucket, indexes in by_bucket.items():
        bucket_states = [states[index] for index in indexes]
        bucket_reports[bucket] = evaluate_checkpoint(
            env, model, enc, value_norm, bucket_states, budgets=budgets,
            oracle=oracle, device=torch_device, c_puct=1.5, seed=seed)

    return {
        "schema_version": 1,
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "eval_pool": str(pool_path),
        "evaluation_split": evaluation_split,
        "evaluation_config": {
            "budgets": list(budgets),
            "oracle_max_nodes": oracle_max_nodes,
            "oracle_time_limit_seconds": oracle_time_limit_seconds,
            "device": device,
            "seed": seed,
        },
        "raw_evaluation": full,
        "summary": summarize_budget_sweep(
            full, budgets=budgets, bucket_reports=bucket_reports),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate a protagonist checkpoint across search budgets.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--eval-levels-dataset", required=True)
    parser.add_argument("--eval-split-manifest", required=True)
    parser.add_argument(
        "--split-role",
        choices=("promotion_validation", "final_test"),
        required=True,
        help="use final_test only once after model and settings are locked")
    parser.add_argument("--budgets", type=int, nargs="+", required=True)
    parser.add_argument("--output", default=None)
    parser.add_argument("--oracle-max-nodes", type=int, default=20_000)
    parser.add_argument("--oracle-time-limit-seconds", type=float, default=3.0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=0)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if bool(args.eval_split_manifest) != bool(args.split_role):
        parser.error(
            "--eval-split-manifest and --split-role must be provided together")
    if args.split_role == "final_test" and not args.output:
        parser.error(
            "final_test evaluation requires --output for a durable report")
    if (args.split_role == "final_test" and args.output
            and Path(args.output).exists()):
        raise FileExistsError(
            f"final-test report already exists and will not be overwritten: "
            f"{args.output}")
    result = evaluate_checkpoint_budget_sweep(
        args.checkpoint,
        args.eval_levels_dataset,
        budgets=args.budgets,
        split_manifest_path=args.eval_split_manifest,
        split_role=args.split_role,
        oracle_max_nodes=args.oracle_max_nodes,
        oracle_time_limit_seconds=(
            None if args.oracle_time_limit_seconds <= 0
            else args.oracle_time_limit_seconds),
        device=args.device,
        seed=args.seed,
    )
    if args.output:
        atomic_write_json(args.output, result)
    print(json.dumps(result["summary"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
