"""Resumable full-pool confirmation for shadow-learner retention monitors."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from ..serialization import level_from_dict
from ..training.experiment_identity import hash_canonical_value
from ..training.transaction import atomic_write_json, sha256_file
from .frontier_promotion_evaluate import _evaluate_checkpoint
from .retention import load_retention_pool, summarize_retention


def _checkpoint_identity(path: Path, role: str) -> dict[str, Any]:
    return {
        "role": role,
        "path": str(path),
        "sha256": sha256_file(path),
    }


def _retention_rows(
    rows: list[dict[str, Any]],
    bands: dict[str, str],
) -> list[dict[str, Any]]:
    return [{
        "static_level_signature": row["static_level_signature"],
        "difficulty_stratum": bands[row["static_level_signature"]],
        "budgets": {
            budget: {"solved": bool(outcome["solved"])}
            for budget, outcome in row["budgets"].items()
        },
    } for row in rows]


def run_confirmation(
    *,
    champion_checkpoint: str | Path,
    round5_checkpoint: str | Path,
    round10_checkpoint: str | Path,
    pool_path: str | Path,
    output_dir: str | Path,
    device: str = "cuda",
) -> dict[str, Any]:
    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    paths = {
        "champion": Path(champion_checkpoint).resolve(),
        "round_005": Path(round5_checkpoint).resolve(),
        "round_010": Path(round10_checkpoint).resolve(),
    }
    pool = Path(pool_path).resolve()
    selected, all_signatures, selection = load_retention_pool(
        pool, per_band=50)
    if len(selected) != 200 or len(all_signatures) != 200:
        raise ValueError("full retention confirmation requires 200 levels")
    budgets = [64, 95, 128]
    contract = {
        "schema_version": 1,
        "contract_id": "full_pool_retention_confirmation_v1",
        "status": "frozen_before_evaluation",
        "pool": {
            "path": str(pool),
            "sha256": sha256_file(pool),
            "level_count": len(selected),
            "band_counts": selection["source_band_counts"],
        },
        "checkpoints": {
            role: _checkpoint_identity(path, role)
            for role, path in paths.items()
        },
        "budgets": budgets,
        "evaluation_seed": 8242,
        "c_puct": 1.5,
        "maximum_allowed_regression": 0.05,
        "comparison_roles": ["round_005", "round_010"],
        "semantics": (
            "paired_deterministic_full_50_per_baseline_difficulty_band_v1"),
    }
    contract_path = output / "contract.json"
    if contract_path.exists():
        import json
        if json.loads(contract_path.read_text(encoding="utf-8")) != contract:
            raise ValueError("persisted retention contract differs")
    else:
        atomic_write_json(contract_path, contract)
    evaluation_identity = hash_canonical_value(contract)
    levels = [level_from_dict(record["level"]) for record in selected]
    bands = {
        record["static_level_signature"]: record["difficulty_stratum"]
        for record in selected
    }
    evaluated = {}
    for role, path in paths.items():
        evaluated[role] = _evaluate_checkpoint(
            str(path), levels, budgets=budgets, c_puct=1.5,
            inference_batch_size=1, virtual_loss=1.0,
            evaluation_seed=8242, device=device, role=role,
            partial_path=output / f"{role}.partial.json",
            evaluation_identity=evaluation_identity,
            progress_interval=10,
            evaluation_context="full_pool_retention_confirmation_v1",
        )
    champion = _retention_rows(evaluated["champion"], bands)
    summaries = {}
    for role in ("round_005", "round_010"):
        summaries[role] = summarize_retention(
            champion, _retention_rows(evaluated[role], bands),
            budgets=budgets, max_regression=0.05)
    result = {
        "schema_version": 1,
        "contract_sha256": evaluation_identity,
        "pool_level_count": len(selected),
        "levels_per_band": 50,
        "budgets": budgets,
        "comparisons_to_champion": summaries,
    }
    atomic_write_json(output / "analysis.json", result)
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--champion-checkpoint", required=True)
    parser.add_argument("--round5-checkpoint", required=True)
    parser.add_argument("--round10-checkpoint", required=True)
    parser.add_argument("--pool", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args(argv)
    result = run_confirmation(
        champion_checkpoint=args.champion_checkpoint,
        round5_checkpoint=args.round5_checkpoint,
        round10_checkpoint=args.round10_checkpoint,
        pool_path=args.pool,
        output_dir=args.output_dir,
        device=args.device,
    )
    for role, report in result["comparisons_to_champion"].items():
        print(
            f"{role}: retention_passed={report['passed']} "
            f"failures={len(report['failures'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
