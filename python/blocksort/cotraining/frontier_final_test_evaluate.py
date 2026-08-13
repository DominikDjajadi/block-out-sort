"""One-shot evaluation on the sealed research-frontier final-test split."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

from ..serialization import level_from_dict
from ..training.transaction import atomic_write_json, sha256_file
from .eval_split import load_eval_pool_records, load_eval_split_manifest
from .frontier_promotion_evaluate import (
    _evaluate_checkpoint,
    summarize_paired_promotion,
)


SCHEMA_VERSION = 1
SEMANTICS = "sealed_frontier_final_test_one_shot_v1"
EVALUATION_CONTEXT = "frontier_final_test.one_shot_v1"


def _canonical_hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(
        value, sort_keys=True, separators=(",", ":"),
        ensure_ascii=False).encode("utf-8")).hexdigest()


def _resolve(source: Path, value: str) -> Path:
    path = Path(value)
    if not path.is_absolute():
        path = source.parent / path
    return path.resolve()


def _load_contract(path: str | Path) -> tuple[Path, dict[str, Any]]:
    source = Path(path).resolve()
    contract = json.loads(source.read_text(encoding="utf-8"))
    if contract.get("schema_version") != SCHEMA_VERSION \
            or contract.get("contract_id") != \
            "research_frontier_final_test_v1":
        raise ValueError("unsupported final-test contract")
    if contract.get("status") != "frozen_before_final_test_evaluation":
        raise ValueError("final-test contract is not frozen")
    for role in ("baseline", "candidate"):
        item = contract["checkpoints"][role]
        resolved = _resolve(source, item["path"])
        if not resolved.is_file() or sha256_file(resolved) != item["sha256"]:
            raise ValueError(f"{role} checkpoint identity mismatch")
        item["resolved_path"] = str(resolved)
    implementation = contract["implementation"]
    implementation_path = _resolve(source, implementation["path"])
    if implementation_path != Path(__file__).resolve() \
            or sha256_file(implementation_path) != implementation["sha256"]:
        raise ValueError("final-test implementation identity mismatch")
    return source, contract


def run_frontier_final_test(
    *, contract_path: str | Path, pool_path: str | Path,
    split_manifest_path: str | Path, output_dir: str | Path,
    device: str = "cuda",
) -> dict[str, Any]:
    contract_source, contract = _load_contract(contract_path)
    pool_source = Path(pool_path).resolve()
    split_source = Path(split_manifest_path).resolve()
    requirement = contract["pool_requirements"]
    manifest = load_eval_split_manifest(
        split_source, pool_source,
        expected_split_seed=requirement["split_seed"],
        expected_validation_count=requirement["promotion_validation_count"])
    if sha256_file(pool_source) != requirement["pool_file_sha256"] \
            or manifest["pool"]["sha256"] != \
            requirement["pool_canonical_sha256"]:
        raise ValueError("final-test pool identity mismatch")
    if sha256_file(split_source) != \
            requirement["split_manifest_file_sha256"] \
            or manifest["manifest_sha256"] != \
            requirement["split_manifest_canonical_sha256"] \
            or manifest["evaluation_split_fingerprint"] != \
            requirement["evaluation_split_fingerprint"]:
        raise ValueError("final-test split identity mismatch")
    if manifest["split_config"]["test_count"] != \
            requirement["final_test_count"]:
        raise ValueError("final-test count does not match contract")

    evaluation = contract["evaluation"]
    identity_payload = {
        "semantics": SEMANTICS,
        "evaluation_context": EVALUATION_CONTEXT,
        "evaluation_implementation_sha256": sha256_file(Path(__file__)),
        "contract_sha256": sha256_file(contract_source),
        "pool_file_sha256": sha256_file(pool_source),
        "pool_canonical_sha256": manifest["pool"]["sha256"],
        "split_manifest_sha256": sha256_file(split_source),
        "evaluation_split_fingerprint":
            manifest["evaluation_split_fingerprint"],
        "baseline_checkpoint_sha256":
            contract["checkpoints"]["baseline"]["sha256"],
        "candidate_checkpoint_sha256":
            contract["checkpoints"]["candidate"]["sha256"],
        "evaluation": evaluation,
    }
    evaluation_identity = _canonical_hash(identity_payload)
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    result_path = destination / "result.json"
    verdict_path = destination / "verdict.json"
    if result_path.exists() or verdict_path.exists():
        raise RuntimeError(
            "the sealed final test has already been evaluated; refusing rerun")

    started_path = destination / "evaluation_started.json"
    started = {
        "schema_version": SCHEMA_VERSION,
        "semantics": SEMANTICS,
        "evaluation_identity": evaluation_identity,
        "contract_sha256": sha256_file(contract_source),
        "status": "started_or_resuming",
    }
    if started_path.exists():
        observed = json.loads(started_path.read_text(encoding="utf-8"))
        if observed != started:
            raise ValueError("incompatible prior final-test start seal")
    else:
        # The one-shot identity is committed before final-test records are
        # selected or converted into level objects. Interrupted work may only
        # resume this exact identity via its partial result files.
        atomic_write_json(started_path, started)

    records = load_eval_pool_records(pool_source)
    by_signature = {
        record["static_level_signature"]: record for record in records}
    final_signatures = [
        item["signature"] for item in manifest["final_test"]]
    final_records = [by_signature[value] for value in final_signatures]
    levels = [level_from_dict(record["level"]) for record in final_records]

    baseline_rows = _evaluate_checkpoint(
        contract["checkpoints"]["baseline"]["resolved_path"], levels,
        budgets=evaluation["budgets"], c_puct=evaluation["c_puct"],
        inference_batch_size=evaluation["inference_batch_size"],
        virtual_loss=evaluation["virtual_loss"],
        evaluation_seed=evaluation["evaluation_seed"], device=device,
        role="baseline", partial_path=destination / "baseline.partial.json",
        evaluation_identity=evaluation_identity,
        evaluation_context=EVALUATION_CONTEXT)
    candidate_rows = _evaluate_checkpoint(
        contract["checkpoints"]["candidate"]["resolved_path"], levels,
        budgets=evaluation["budgets"], c_puct=evaluation["c_puct"],
        inference_batch_size=evaluation["inference_batch_size"],
        virtual_loss=evaluation["virtual_loss"],
        evaluation_seed=evaluation["evaluation_seed"], device=device,
        role="candidate", partial_path=destination / "candidate.partial.json",
        evaluation_identity=evaluation_identity,
        evaluation_context=EVALUATION_CONTEXT)
    rule = contract["confirmation_rule"]
    bootstrap = rule["paired_bootstrap"]
    summary = summarize_paired_promotion(
        baseline_rows, candidate_rows, budgets=evaluation["budgets"],
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
        for record in final_records)
    confirmatory_pass = bool(summary["promoted"])
    result = {
        "schema_version": SCHEMA_VERSION,
        "semantics": SEMANTICS,
        "evaluation_identity": evaluation_identity,
        "identity": identity_payload,
        "final_test": {
            "count": len(final_records),
            "status": "evaluated_once",
            "promotion_validation_states_constructed": 0,
            "difficulty_stratum_counts": dict(stratum_counts),
        },
        "summary": summary,
        "confirmatory_pass": confirmatory_pass,
        "interpretation": (
            "generalization_confirmed_under_preregistered_rule"
            if confirmatory_pass else
            "generalization_not_confirmed_under_preregistered_rule"),
        "baseline_rows": baseline_rows,
        "candidate_rows": candidate_rows,
    }
    atomic_write_json(result_path, result)
    atomic_write_json(verdict_path, {
        "schema_version": SCHEMA_VERSION,
        "semantics": SEMANTICS,
        "evaluation_identity": evaluation_identity,
        "result": str(result_path),
        "result_sha256": sha256_file(result_path),
        "final_test_status": "evaluated_once",
        "confirmatory_pass": confirmatory_pass,
        "interpretation": result["interpretation"],
        "summary": summary,
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
    result = run_frontier_final_test(
        contract_path=args.contract, pool_path=args.pool,
        split_manifest_path=args.split_manifest,
        output_dir=args.output_dir, device=args.device)
    print(json.dumps({
        "evaluation_identity": result["evaluation_identity"],
        "confirmatory_pass": result["confirmatory_pass"],
        "interpretation": result["interpretation"],
        "summary": result["summary"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
