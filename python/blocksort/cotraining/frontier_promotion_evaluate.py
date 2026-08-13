"""Preregistered paired evaluation for a frontier promotion challenge."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
from collections import Counter
from pathlib import Path
from typing import Any, Mapping

from ..environment import Environment
from ..serialization import level_from_dict
from ..signature import static_level_signature
from ..training.transaction import atomic_write_json, sha256_file
from .eval_split import load_eval_split_manifest, load_eval_pool_records


SCHEMA_VERSION = 1
SEMANTICS = "preregistered_paired_frontier_promotion_v1"


def _canonical_hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(
        value, sort_keys=True, separators=(",", ":"),
        ensure_ascii=False).encode("utf-8")).hexdigest()


def _quantile(values: list[float], fraction: float) -> float:
    if not values:
        raise ValueError("cannot compute a quantile of no values")
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = fraction * (len(ordered) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def paired_bootstrap_lower_bound(
    level_deltas: list[float], *, confidence_level: float,
    replicates: int, seed: int,
) -> tuple[float, list[float]]:
    """Return a deterministic one-sided percentile lower confidence bound."""
    if not level_deltas:
        raise ValueError("paired bootstrap requires level deltas")
    if not 0.0 < confidence_level < 1.0:
        raise ValueError("confidence_level must be in (0, 1)")
    if isinstance(replicates, bool) or not isinstance(replicates, int) \
            or replicates <= 0:
        raise ValueError("replicates must be a positive integer")
    rng = random.Random(seed)
    count = len(level_deltas)
    samples = []
    for _ in range(replicates):
        samples.append(sum(
            level_deltas[rng.randrange(count)] for _ in range(count)) / count)
    return _quantile(samples, 1.0 - confidence_level), samples


def summarize_paired_promotion(
    champion_rows: list[Mapping[str, Any]],
    candidate_rows: list[Mapping[str, Any]],
    *, budgets: list[int], weights: list[float],
    minimum_delta: float, maximum_per_budget_regression: float,
    bootstrap_confidence: float, bootstrap_replicates: int,
    bootstrap_seed: int,
) -> dict[str, Any]:
    if len(champion_rows) != len(candidate_rows) or not champion_rows:
        raise ValueError("paired evaluations must have equal nonzero length")
    if len(budgets) != len(weights) or not math.isclose(sum(weights), 1.0):
        raise ValueError("budgets and normalized weights must align")
    champion_by_signature = {
        row["static_level_signature"]: row for row in champion_rows}
    candidate_by_signature = {
        row["static_level_signature"]: row for row in candidate_rows}
    if set(champion_by_signature) != set(candidate_by_signature):
        raise ValueError("paired evaluation signatures do not match")

    per_budget: dict[str, Any] = {}
    champion_score = candidate_score = 0.0
    for budget, weight in zip(budgets, weights):
        champion_solved = candidate_solved = 0
        champion_only = candidate_only = both = 0
        for signature in sorted(champion_by_signature):
            champion = bool(
                champion_by_signature[signature]["budgets"][str(budget)][
                    "solved"])
            candidate = bool(
                candidate_by_signature[signature]["budgets"][str(budget)][
                    "solved"])
            champion_solved += champion
            candidate_solved += candidate
            both += champion and candidate
            champion_only += champion and not candidate
            candidate_only += candidate and not champion
        total = len(champion_rows)
        champion_rate = champion_solved / total
        candidate_rate = candidate_solved / total
        delta = candidate_rate - champion_rate
        champion_score += weight * champion_rate
        candidate_score += weight * candidate_rate
        per_budget[str(budget)] = {
            "weight": weight,
            "levels": total,
            "champion_solved": champion_solved,
            "candidate_solved": candidate_solved,
            "both_solved": both,
            "champion_only": champion_only,
            "candidate_only": candidate_only,
            "champion_solve_rate": champion_rate,
            "candidate_solve_rate": candidate_rate,
            "solve_rate_delta": delta,
            "regression_guard_passed":
                delta >= -maximum_per_budget_regression,
        }

    level_deltas = []
    for signature in sorted(champion_by_signature):
        delta = 0.0
        for budget, weight in zip(budgets, weights):
            champion = bool(
                champion_by_signature[signature]["budgets"][str(budget)][
                    "solved"])
            candidate = bool(
                candidate_by_signature[signature]["budgets"][str(budget)][
                    "solved"])
            delta += weight * (int(candidate) - int(champion))
        level_deltas.append(delta)
    lower, bootstrap_samples = paired_bootstrap_lower_bound(
        level_deltas, confidence_level=bootstrap_confidence,
        replicates=bootstrap_replicates, seed=bootstrap_seed)
    weighted_delta = candidate_score - champion_score
    conditions = {
        "minimum_weighted_delta": weighted_delta >= minimum_delta,
        "per_budget_regression_guards": all(
            item["regression_guard_passed"]
            for item in per_budget.values()),
        "bootstrap_lower_bound_strictly_positive": lower > 0.0,
    }
    return {
        "levels": len(champion_rows),
        "budgets": budgets,
        "weights": weights,
        "champion_weighted_solve_rate": champion_score,
        "candidate_weighted_solve_rate": candidate_score,
        "weighted_solve_rate_delta": weighted_delta,
        "per_budget": per_budget,
        "paired_level_delta_distribution": dict(Counter(level_deltas)),
        "paired_bootstrap": {
            "method": "one_sided_percentile_by_level_v1",
            "confidence_level": bootstrap_confidence,
            "replicates": bootstrap_replicates,
            "seed": bootstrap_seed,
            "lower_confidence_bound": lower,
            "bootstrap_mean": sum(bootstrap_samples) / len(bootstrap_samples),
        },
        "conditions": conditions,
        "promoted": all(conditions.values()),
        "decision": "promote" if all(conditions.values()) else "do_not_promote",
    }


def _resolve_contract_path(contract_path: Path, value: str) -> Path:
    path = Path(value)
    if not path.is_absolute():
        path = contract_path.parent / path
    return path.resolve()


def _load_contract(path: str | Path) -> tuple[Path, dict[str, Any]]:
    source = Path(path).resolve()
    contract = json.loads(source.read_text(encoding="utf-8"))
    if contract.get("schema_version") != 2 \
            or contract.get("contract_id") != "research_frontier_promotion_v2":
        raise ValueError("unsupported promotion contract")
    if contract.get("status") != "frozen_before_pool_generation":
        raise ValueError("promotion contract is not frozen")
    for role in ("champion", "candidate"):
        item = contract["checkpoints"][role]
        resolved = _resolve_contract_path(source, item["path"])
        if not resolved.is_file() or sha256_file(resolved) != item["sha256"]:
            raise ValueError(f"{role} checkpoint identity mismatch")
        item["resolved_path"] = str(resolved)
    return source, contract


def _evaluate_checkpoint(
    checkpoint_path: str, levels, *, budgets: list[int], c_puct: float,
    inference_batch_size: int, virtual_loss: float, evaluation_seed: int,
    device: str, role: str, partial_path: Path,
    evaluation_identity: str, progress_interval: int = 25,
    evaluation_context: str = "frontier_promotion.challenge_v1",
) -> list[dict[str, Any]]:
    import torch

    from ..search.config import SearchConfig
    from ..search.graph_search import BlocksortAdapter, GraphSearch
    from ..search.seeding import derive_trial_seed, level_search_identity
    from ..training.checkpoint import (
        configs_from_checkpoint, load_checkpoint, model_from_checkpoint)

    env = Environment()
    checkpoint = load_checkpoint(checkpoint_path, map_location="cpu")
    encoding, _model_cfg, value_norm = configs_from_checkpoint(checkpoint)
    torch_device = torch.device(device)
    model = model_from_checkpoint(checkpoint, map_location=torch_device)
    model.eval()
    adapter = BlocksortAdapter(
        env, model, encoding, value_norm, torch_device)
    rows: list[dict[str, Any]] = []
    if partial_path.exists():
        partial = json.loads(partial_path.read_text(encoding="utf-8"))
        if partial.get("evaluation_identity") != evaluation_identity \
                or partial.get("role") != role:
            raise ValueError(f"incompatible partial evaluation: {partial_path}")
        rows = partial.get("rows", [])
        if not isinstance(rows, list) or len(rows) > len(levels):
            raise ValueError(f"invalid partial evaluation: {partial_path}")
        for index, row in enumerate(rows):
            if row.get("static_level_signature") != \
                    static_level_signature(levels[index]):
                raise ValueError("partial evaluation level mismatch")
        print(f"resuming {role} at {len(rows)}/{len(levels)} levels", flush=True)

    for index in range(len(rows), len(levels)):
        level = levels[index]
        state = env.initial_state(level)
        search_identity = level_search_identity(env, level)
        budget_results = {}
        for trial_index, budget in enumerate(budgets):
            trial_seed = derive_trial_seed(
                evaluation_seed, trial_index=trial_index,
                level_identity=search_identity,
                evaluation_context=evaluation_context)
            cfg = SearchConfig(
                simulations=budget, c_puct=c_puct, temperature=0.0,
                inference_batch_size=inference_batch_size,
                virtual_loss=virtual_loss,
                value_normalization_constant=getattr(
                    value_norm, "constant", 20.0), seed=trial_seed)
            result = GraphSearch(adapter, cfg).run(state)
            budget_results[str(budget)] = {
                "solved": bool(result.solved),
                "solution_length": result.solution_length,
                "solution_verified": bool(result.solution_verified),
                "termination_reason": result.termination_reason,
                "simulations": result.stats.simulations,
                "unique_states": result.stats.unique_states,
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
        if (index + 1) % progress_interval == 0 or index + 1 == len(levels):
            print(f"{role}: {index + 1}/{len(levels)} levels", flush=True)
    return rows


def run_frontier_promotion_challenge(
    *, contract_path: str | Path, pool_path: str | Path,
    split_manifest_path: str | Path, output_dir: str | Path,
    device: str = "cuda",
) -> dict[str, Any]:
    contract_source, contract = _load_contract(contract_path)
    pool_source = Path(pool_path).resolve()
    split_source = Path(split_manifest_path).resolve()
    pool_requirement = contract["pool_requirements"]
    manifest = load_eval_split_manifest(
        split_source, pool_source,
        expected_split_seed=pool_requirement["split_seed"],
        expected_validation_count=
            pool_requirement["promotion_validation_count"])
    if manifest["split_config"]["test_count"] != \
            pool_requirement["sealed_final_test_count"]:
        raise ValueError("sealed final-test count does not match contract")
    records = load_eval_pool_records(pool_source)
    by_signature = {
        record["static_level_signature"]: record for record in records}
    validation_signatures = [
        item["signature"] for item in manifest["promotion_validation"]]
    # The final-test entries are deliberately not converted to states.
    validation_records = [by_signature[value] for value in validation_signatures]
    levels = [level_from_dict(record["level"])
              for record in validation_records]
    evaluation = contract["evaluation"]
    identity_payload = {
        "semantics": SEMANTICS,
        "evaluation_implementation_sha256": sha256_file(Path(__file__)),
        "contract_sha256": sha256_file(contract_source),
        "pool_file_sha256": sha256_file(pool_source),
        "pool_canonical_sha256": manifest["pool"]["sha256"],
        "split_manifest_sha256": sha256_file(split_source),
        "evaluation_split_fingerprint":
            manifest["evaluation_split_fingerprint"],
        "champion_checkpoint_sha256":
            contract["checkpoints"]["champion"]["sha256"],
        "candidate_checkpoint_sha256":
            contract["checkpoints"]["candidate"]["sha256"],
        "evaluation": evaluation,
    }
    evaluation_identity = _canonical_hash(identity_payload)
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    champion_rows = _evaluate_checkpoint(
        contract["checkpoints"]["champion"]["resolved_path"], levels,
        budgets=evaluation["budgets"], c_puct=evaluation["c_puct"],
        inference_batch_size=evaluation["inference_batch_size"],
        virtual_loss=evaluation["virtual_loss"],
        evaluation_seed=evaluation["evaluation_seed"], device=device,
        role="champion", partial_path=destination / "champion.partial.json",
        evaluation_identity=evaluation_identity)
    candidate_rows = _evaluate_checkpoint(
        contract["checkpoints"]["candidate"]["resolved_path"], levels,
        budgets=evaluation["budgets"], c_puct=evaluation["c_puct"],
        inference_batch_size=evaluation["inference_batch_size"],
        virtual_loss=evaluation["virtual_loss"],
        evaluation_seed=evaluation["evaluation_seed"], device=device,
        role="candidate", partial_path=destination / "candidate.partial.json",
        evaluation_identity=evaluation_identity)
    rule = contract["promotion_rule"]
    bootstrap = rule["paired_bootstrap"]
    summary = summarize_paired_promotion(
        champion_rows, candidate_rows, budgets=evaluation["budgets"],
        weights=evaluation["weights"],
        minimum_delta=rule["minimum_weighted_solve_rate_delta"],
        maximum_per_budget_regression=
            rule["maximum_absolute_solve_rate_regression_per_budget"],
        bootstrap_confidence=bootstrap["confidence_level"],
        bootstrap_replicates=bootstrap["replicates"],
        bootstrap_seed=bootstrap["seed"])
    stratum_counts = Counter(
        record.get("protagonist_filter", {}).get(
            "difficulty_stratum", "unknown")
        for record in validation_records)
    result = {
        "schema_version": SCHEMA_VERSION,
        "semantics": SEMANTICS,
        "evaluation_identity": evaluation_identity,
        "identity": identity_payload,
        "promotion_validation_stratum_counts": dict(stratum_counts),
        "sealed_final_test": {
            "count": manifest["split_config"]["test_count"],
            "status": "sealed_not_evaluated",
            "states_constructed": 0,
        },
        "summary": summary,
        "champion_rows": champion_rows,
        "candidate_rows": candidate_rows,
    }
    result_path = destination / "result.json"
    atomic_write_json(result_path, result)
    atomic_write_json(destination / "verdict.json", {
        "schema_version": SCHEMA_VERSION,
        "semantics": SEMANTICS,
        "evaluation_identity": evaluation_identity,
        "result": str(result_path),
        "result_sha256": sha256_file(result_path),
        "summary": summary,
        "sealed_final_test_status": "sealed_not_evaluated",
    })
    return result


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", required=True)
    parser.add_argument("--pool", required=True)
    parser.add_argument("--split-manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cuda")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    result = run_frontier_promotion_challenge(
        contract_path=args.contract, pool_path=args.pool,
        split_manifest_path=args.split_manifest,
        output_dir=args.output_dir, device=args.device)
    print(json.dumps(result["summary"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
