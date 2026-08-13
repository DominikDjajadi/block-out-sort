"""Controlled policy-label ablations with sealed final-test roles.

The workflow keeps the immutable final-test role sealed:

1. ``export-promotion`` extracts only promotion-validation levels.
2. ``prepare`` creates control/treatment datasets and shared manifests while
   rejecting any overlap with the complete frozen pool.
3. ``prepare-matched`` optionally builds base/path/full arms in which the path
   and full augmentations label exactly the same states.
4. ``run`` trains matched seeds and evaluates every arm on identical
   promotion-only exact policy records.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import statistics
import sys
from pathlib import Path
from typing import Any, Optional

import torch
from torch.utils.data import DataLoader

from ..cotraining.eval_split import (
    evaluation_split_identity,
    load_eval_pool_records,
    load_eval_split_manifest,
)
from ..dataset.combine import combine_records
from ..dataset.schema import LABEL_EXACT_PATH_POLICY, LABEL_FULL_EXACT
from .dataset import PolicyValueDataset, collate_batch, load_records
from .losses import masked_policy_probs
from .predict import load_model_bundle
from .splits import group_key
from .train import build_parser as build_training_parser
from .train import resolve_device, run_training
from .transaction import atomic_write_json, atomic_write_text, sha256_file

ABLATION_SCHEMA_VERSION = 1


def _write_jsonl(records: list[dict[str, Any]], path: Path) -> None:
    atomic_write_text(
        path,
        "".join(json.dumps(record, sort_keys=True) + "\n" for record in records),
    )


def _refuse_existing(paths: list[Path], overwrite: bool) -> None:
    existing = [path for path in paths if path.exists()]
    if existing and not overwrite:
        raise FileExistsError(
            "ablation artifact already exists; pass --overwrite: "
            + ", ".join(str(path) for path in existing))


def export_promotion_levels(
    pool_path: str | Path,
    split_path: str | Path,
    output: str | Path,
    *,
    report: str | Path | None = None,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Export promotion-validation levels without exporting final-test levels."""
    pool_path = Path(pool_path)
    split_path = Path(split_path)
    output = Path(output)
    report_path = Path(report) if report else output.with_name(
        f"{output.stem}_report.json")
    _refuse_existing([output, report_path], overwrite)

    records = load_eval_pool_records(pool_path)
    manifest = load_eval_split_manifest(split_path, pool_path)
    by_signature = {
        record["static_level_signature"]: record for record in records}
    promotion_signatures = [
        item["signature"] for item in manifest["promotion_validation"]]
    final_signatures = {
        item["signature"] for item in manifest["final_test"]}
    if set(promotion_signatures) & final_signatures:
        raise RuntimeError("frozen evaluation roles overlap")
    levels = [by_signature[signature]["level"] for signature in promotion_signatures]
    atomic_write_json(output, levels)
    document = {
        "schema_version": ABLATION_SCHEMA_VERSION,
        "role": "promotion_validation",
        "source_pool": {"path": str(pool_path), "sha256": sha256_file(pool_path)},
        "source_split": {"path": str(split_path), "sha256": sha256_file(split_path)},
        "evaluation_identity": evaluation_split_identity(manifest, eval_limit=None),
        "promotion_count": len(levels),
        "promotion_signature_hash": manifest["validation_signature_hash"],
        "final_test_count": len(final_signatures),
        "final_test_status": "sealed_not_exported",
        "output": {"path": str(output), "sha256": sha256_file(output)},
    }
    atomic_write_json(report_path, document)
    return {**document, "report": str(report_path)}


def export_path_training_levels(
    path_dataset: str | Path,
    frozen_pool: str | Path,
    output: str | Path,
    *,
    report: str | Path | None = None,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Export the exact training levels represented by path-policy records."""
    path_dataset = Path(path_dataset)
    frozen_pool = Path(frozen_pool)
    output = Path(output)
    report_path = Path(report) if report else output.with_name(
        f"{output.stem}_report.json")
    _refuse_existing([output, report_path], overwrite)

    records = load_records(path_dataset)
    if not records:
        raise ValueError("path dataset must be nonempty")
    if any(record.get("label_kind") != LABEL_EXACT_PATH_POLICY
           for record in records):
        raise ValueError("training-level export requires exact-path-policy records")
    signatures = [group_key(record) for record in records]
    if len(signatures) != len(set(signatures)):
        raise ValueError("path dataset must contain exactly one state per level")
    frozen_signatures = {
        record["static_level_signature"]
        for record in load_eval_pool_records(frozen_pool)}
    overlap = set(signatures) & frozen_signatures
    if overlap:
        raise ValueError(
            f"path dataset overlaps {len(overlap)} frozen evaluation levels")

    ordered = sorted(records, key=lambda record: group_key(record))
    atomic_write_json(output, [record["level"] for record in ordered])
    document = {
        "schema_version": ABLATION_SCHEMA_VERSION,
        "role": "training_only_matched_full_exact_source",
        "source_path_records": {
            "path": str(path_dataset), "sha256": sha256_file(path_dataset)},
        "frozen_pool": {
            "path": str(frozen_pool), "sha256": sha256_file(frozen_pool)},
        "level_count": len(ordered),
        "unique_static_signature_count": len(set(signatures)),
        "frozen_overlap_count": 0,
        "output": {"path": str(output), "sha256": sha256_file(output)},
        "final_test_status": "sealed_not_exported_or_evaluated",
    }
    atomic_write_json(report_path, document)
    return {**document, "report": str(report_path)}


def _training_manifest(signatures: set[str]) -> dict[str, Any]:
    return {
        "version": 1,
        "seed": 0,
        "group_key": "static_level_signature",
        "ratios": {"train": 1.0, "validation": 0.0, "test": 0.0},
        "train_levels": sorted(signatures),
        "validation_levels": [],
        "test_levels": [],
    }


def prepare_ablation(
    *,
    base_dataset: str | Path,
    path_dataset: str | Path,
    promotion_dataset: str | Path,
    frozen_pool: str | Path,
    frozen_split: str | Path,
    output_dir: str | Path,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Create immutable arm datasets and manifests with leakage checks."""
    root = Path(output_dir)
    control_path = root / "data" / "control.jsonl"
    treatment_path = root / "data" / "treatment.jsonl"
    promotion_path = root / "data" / "promotion.jsonl"
    train_manifest_path = root / "training_manifest.json"
    eval_manifest_path = root / "promotion_manifest.json"
    report_path = root / "prepared.json"
    targets = [control_path, treatment_path, promotion_path,
               train_manifest_path, eval_manifest_path, report_path]
    _refuse_existing(targets, overwrite)

    base_records = load_records(base_dataset)
    path_records = load_records(path_dataset)
    promotion_records = load_records(promotion_dataset)
    if not base_records or not path_records or not promotion_records:
        raise ValueError("base, path, and promotion datasets must all be nonempty")
    if any(record.get("label_kind", LABEL_FULL_EXACT) != LABEL_FULL_EXACT
           for record in base_records):
        raise ValueError("control base must contain only full-exact records")
    if any(record.get("label_kind") != LABEL_EXACT_PATH_POLICY
           for record in path_records):
        raise ValueError("path augmentation must contain only exact-path-policy records")
    if any(record.get("label_kind") != LABEL_EXACT_PATH_POLICY
           for record in promotion_records):
        raise ValueError(
            "promotion evaluation must contain only exact-path-policy records")

    pool_records = load_eval_pool_records(frozen_pool)
    split = load_eval_split_manifest(frozen_split, frozen_pool)
    promotion_expected = {
        item["signature"] for item in split["promotion_validation"]}
    final_signatures = {item["signature"] for item in split["final_test"]}
    promotion_observed = {group_key(record) for record in promotion_records}
    if promotion_observed != promotion_expected:
        missing = len(promotion_expected - promotion_observed)
        unexpected = len(promotion_observed - promotion_expected)
        raise ValueError(
            "promotion labels must cover the frozen promotion role exactly: "
            f"missing={missing}, unexpected={unexpected}")
    frozen_signatures = {
        record["static_level_signature"] for record in pool_records}
    training_signatures = {
        group_key(record) for record in base_records + path_records}
    overlap = training_signatures & frozen_signatures
    if overlap:
        raise ValueError(
            f"training data overlaps {len(overlap)} frozen evaluation levels")
    if promotion_observed & final_signatures:
        raise ValueError("promotion labels contain sealed final-test levels")

    control_records, control_stats = combine_records([base_dataset])
    treatment_records, treatment_stats = combine_records(
        [base_dataset, path_dataset])
    promotion_clean, promotion_stats = combine_records([promotion_dataset])
    _write_jsonl(control_records, control_path)
    _write_jsonl(treatment_records, treatment_path)
    _write_jsonl(promotion_clean, promotion_path)
    atomic_write_json(
        train_manifest_path, _training_manifest(training_signatures))
    atomic_write_json(eval_manifest_path, {
        "version": 1,
        "seed": 0,
        "group_key": "static_level_signature",
        "ratios": {"train": 0.0, "validation": 0.0, "test": 1.0},
        "train_levels": [],
        "validation_levels": [],
        "test_levels": sorted(promotion_observed),
    })
    document = {
        "schema_version": ABLATION_SCHEMA_VERSION,
        "semantics": "paired_full_exact_vs_full_plus_exact_path_policy_v1",
        "control": {
            "path": str(control_path), "sha256": sha256_file(control_path),
            **control_stats,
        },
        "treatment": {
            "path": str(treatment_path), "sha256": sha256_file(treatment_path),
            **treatment_stats,
        },
        "promotion": {
            "path": str(promotion_path), "sha256": sha256_file(promotion_path),
            **promotion_stats,
        },
        "training_manifest": {
            "path": str(train_manifest_path),
            "sha256": sha256_file(train_manifest_path),
        },
        "promotion_manifest": {
            "path": str(eval_manifest_path),
            "sha256": sha256_file(eval_manifest_path),
        },
        "evaluation_identity": evaluation_split_identity(split, eval_limit=None),
        "final_test_status": "sealed_not_labelled_or_evaluated",
    }
    atomic_write_json(report_path, document)
    return {**document, "prepared_report": str(report_path)}


def prepare_matched_ablation(
    *,
    base_dataset: str | Path,
    path_dataset: str | Path,
    full_dataset: str | Path,
    promotion_dataset: str | Path,
    frozen_pool: str | Path,
    frozen_split: str | Path,
    output_dir: str | Path,
    expected_matched_count: int | None = None,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Prepare base/path/full arms with identical augmentation states."""
    root = Path(output_dir)
    arm_paths = {
        "base": root / "data" / "base.jsonl",
        "path": root / "data" / "path.jsonl",
        "full": root / "data" / "full.jsonl",
    }
    promotion_path = root / "data" / "promotion.jsonl"
    train_manifest_path = root / "training_manifest.json"
    eval_manifest_path = root / "promotion_manifest.json"
    report_path = root / "prepared.json"
    _refuse_existing([
        *arm_paths.values(), promotion_path, train_manifest_path,
        eval_manifest_path, report_path,
    ], overwrite)

    base_records = load_records(base_dataset)
    path_records = load_records(path_dataset)
    full_records = load_records(full_dataset)
    promotion_records = load_records(promotion_dataset)
    if not all((base_records, path_records, full_records, promotion_records)):
        raise ValueError("all matched-ablation datasets must be nonempty")
    if any(record.get("label_kind", LABEL_FULL_EXACT) != LABEL_FULL_EXACT
           for record in base_records + full_records):
        raise ValueError("base and matched-control records must be full-exact")
    if any(record.get("label_kind") != LABEL_EXACT_PATH_POLICY
           for record in path_records + promotion_records):
        raise ValueError("path and promotion records must be exact-path-policy")

    path_by_state = {record["state_key"]: record for record in path_records}
    full_by_state = {record["state_key"]: record for record in full_records}
    if len(path_by_state) != len(path_records):
        raise ValueError("path augmentation contains duplicate state keys")
    if len(full_by_state) != len(full_records):
        raise ValueError("full augmentation contains duplicate state keys")
    if path_by_state.keys() != full_by_state.keys():
        missing = len(path_by_state.keys() - full_by_state.keys())
        unexpected = len(full_by_state.keys() - path_by_state.keys())
        raise ValueError(
            "path and full augmentations must label exactly the same states: "
            f"missing_full={missing}, unexpected_full={unexpected}")
    if expected_matched_count is not None and len(path_records) != expected_matched_count:
        raise ValueError(
            f"expected {expected_matched_count} matched records, "
            f"found {len(path_records)}")

    for state_key, path_record in path_by_state.items():
        full_record = full_by_state[state_key]
        if group_key(path_record) != group_key(full_record):
            raise ValueError(f"matched state has conflicting level identity: {state_key}")
        if (int(path_record["optimal_remaining_moves"])
                != int(full_record["optimal_remaining_moves"])):
            raise ValueError(f"matched state has conflicting exact value: {state_key}")
        full_optimal = {
            json.dumps(action, sort_keys=True)
            for action in full_record["optimal_actions"]}
        path_optimal = {
            json.dumps(action, sort_keys=True)
            for action in path_record["optimal_actions"]}
        if not path_optimal or not path_optimal <= full_optimal:
            raise ValueError(
                f"path proof is inconsistent with full-exact actions: {state_key}")

    pool_records = load_eval_pool_records(frozen_pool)
    split = load_eval_split_manifest(frozen_split, frozen_pool)
    promotion_expected = {
        item["signature"] for item in split["promotion_validation"]}
    final_signatures = {item["signature"] for item in split["final_test"]}
    promotion_observed = {group_key(record) for record in promotion_records}
    if promotion_observed != promotion_expected:
        raise ValueError("promotion labels do not exactly match the frozen role")
    frozen_signatures = {
        record["static_level_signature"] for record in pool_records}
    training_signatures = {
        group_key(record)
        for record in base_records + path_records + full_records}
    overlap = training_signatures & frozen_signatures
    if overlap:
        raise ValueError(
            f"training data overlaps {len(overlap)} frozen evaluation levels")
    if promotion_observed & final_signatures:
        raise ValueError("promotion labels contain sealed final-test levels")

    combined: dict[str, tuple[list[dict[str, Any]], dict[str, Any]]] = {
        "base": combine_records([base_dataset]),
        "path": combine_records([base_dataset, path_dataset]),
        "full": combine_records([base_dataset, full_dataset]),
    }
    expected_arm_size = len(base_records) + len(path_records)
    for arm, (records, _stats) in combined.items():
        expected = len(base_records) if arm == "base" else expected_arm_size
        if len(records) != expected:
            raise ValueError(
                f"{arm} arm contains unexpected overlaps: "
                f"expected={expected}, observed={len(records)}")
        _write_jsonl(records, arm_paths[arm])
    promotion_clean, promotion_stats = combine_records([promotion_dataset])
    _write_jsonl(promotion_clean, promotion_path)
    atomic_write_json(train_manifest_path, _training_manifest(training_signatures))
    atomic_write_json(eval_manifest_path, {
        "version": 1,
        "seed": 0,
        "group_key": "static_level_signature",
        "ratios": {"train": 0.0, "validation": 0.0, "test": 1.0},
        "train_levels": [],
        "validation_levels": [],
        "test_levels": sorted(promotion_observed),
    })

    document: dict[str, Any] = {
        "schema_version": ABLATION_SCHEMA_VERSION,
        "semantics": "matched_base_vs_path_vs_full_exact_policy_v1",
        "arm_order": ["base", "path", "full"],
        "baseline_arm": "base",
        "matched_augmentation_count": len(path_records),
        "matched_state_key_hash": hashlib.sha256(
            "\n".join(sorted(path_by_state)).encode("utf-8")).hexdigest(),
        "promotion": {
            "path": str(promotion_path), "sha256": sha256_file(promotion_path),
            **promotion_stats,
        },
        "training_manifest": {
            "path": str(train_manifest_path),
            "sha256": sha256_file(train_manifest_path),
        },
        "promotion_manifest": {
            "path": str(eval_manifest_path),
            "sha256": sha256_file(eval_manifest_path),
        },
        "evaluation_identity": evaluation_split_identity(split, eval_limit=None),
        "final_test_status": "sealed_not_labelled_or_evaluated",
    }
    for arm, (_records, stats) in combined.items():
        document[arm] = {
            "path": str(arm_paths[arm]), "sha256": sha256_file(arm_paths[arm]),
            **stats,
        }
    atomic_write_json(report_path, document)
    return {**document, "prepared_report": str(report_path)}


def _verify_prepared_artifacts(prepared: dict[str, Any]) -> None:
    """Reject missing, replaced, or role-unsafe prepared inputs."""
    if prepared.get("final_test_status") != "sealed_not_labelled_or_evaluated":
        raise ValueError("prepared report does not attest a sealed final test")
    arm_order = prepared.get("arm_order", ["control", "treatment"])
    if (not isinstance(arm_order, list) or not arm_order
            or any(not isinstance(arm, str) for arm in arm_order)):
        raise ValueError("prepared report has an invalid arm order")
    for name in (*arm_order, "promotion", "training_manifest",
                 "promotion_manifest"):
        artifact = prepared.get(name)
        if not isinstance(artifact, dict):
            raise ValueError(f"prepared report is missing {name} metadata")
        path = Path(str(artifact.get("path", "")))
        expected = artifact.get("sha256")
        if not path.is_file():
            raise FileNotFoundError(f"prepared {name} artifact is missing: {path}")
        actual = sha256_file(path)
        if not expected or actual != expected:
            raise ValueError(
                f"prepared {name} artifact hash mismatch: "
                f"expected={expected}, actual={actual}")


@torch.no_grad()
def evaluate_policy_checkpoint(
    checkpoint: str | Path,
    records_path: str | Path,
    *,
    device: str = "auto",
    batch_size: int = 256,
) -> dict[str, Any]:
    """Return per-state conservative verified-action policy measurements."""
    resolved = resolve_device(device)
    model, encoding, value_norm, checkpoint_data = load_model_bundle(
        checkpoint, resolved)
    records = load_records(records_path)
    dataset = PolicyValueDataset(
        records, encoding_config=encoding, value_norm=value_norm)
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_batch)
    rows: list[dict[str, Any]] = []
    cursor = 0
    model.eval()
    for batch in loader:
        board = batch["board"].to(resolved)
        glob = batch["global_features"].to(resolved)
        legal = batch["legal_action_mask"].to(resolved)
        target = batch["policy_target"].to(resolved)
        regret = batch["action_regret"].to(resolved)
        known = batch["action_regret_known_mask"].to(resolved)
        logits, _value = model(board, glob)
        probs = masked_policy_probs(logits, legal)
        selected = torch.where(
            legal > 0, logits, torch.full_like(logits, float("-inf"))).argmax(-1)
        raw_selected = logits.argmax(-1)
        verified = (known > 0) & (regret == 0) & (legal > 0)
        for index in range(logits.shape[0]):
            record = records[cursor]
            sel = int(selected[index].item())
            target_nonzero = target[index] > 0
            ce = float(-(target[index][target_nonzero] * torch.log(
                probs[index][target_nonzero].clamp_min(1e-12))).sum().item())
            rows.append({
                "static_level_signature": record["static_level_signature"],
                "state_key": record["state_key"],
                "level_id": str(record.get("level_id", "")),
                "rows": int(record["level"]["rows"]),
                "cols": int(record["level"]["cols"]),
                "optimal_remaining_moves": int(record["optimal_remaining_moves"]),
                "legal_action_count": len(record["legal_actions"]),
                "label_kind": record.get("label_kind", LABEL_FULL_EXACT),
                "verified_top1": int(bool(verified[index, sel])),
                "verified_target_mass": float(
                    (probs[index] * target_nonzero.float()).sum().item()),
                "policy_cross_entropy": ce,
                "raw_top1_legal": int(bool(
                    legal[index, int(raw_selected[index].item())] > 0)),
            })
            cursor += 1
    if cursor != len(records):
        raise RuntimeError("evaluation row count mismatch")
    count = len(rows)
    return {
        "schema_version": ABLATION_SCHEMA_VERSION,
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": sha256_file(checkpoint),
        "checkpoint_epoch": int(checkpoint_data["epoch"]),
        "records": str(records_path),
        "records_sha256": sha256_file(records_path),
        "count": count,
        "metrics": {
            "verified_top1_acc": sum(row["verified_top1"] for row in rows) / count,
            "verified_target_mass": sum(
                row["verified_target_mass"] for row in rows) / count,
            "policy_cross_entropy": sum(
                row["policy_cross_entropy"] for row in rows) / count,
            "raw_top1_legal_rate": sum(
                row["raw_top1_legal"] for row in rows) / count,
        },
        "rows": rows,
        "final_test_status": "sealed_not_evaluated",
    }


def _paired_report(evaluations: list[dict[str, Any]]) -> dict[str, Any]:
    by_seed: dict[int, dict[str, dict[str, Any]]] = {}
    for evaluation in evaluations:
        by_seed.setdefault(int(evaluation["seed"]), {})[
            str(evaluation["arm"])] = evaluation
    seed_rows: list[dict[str, Any]] = []
    all_deltas: list[dict[str, float | int]] = []
    for seed, arms in sorted(by_seed.items()):
        if set(arms) != {"control", "treatment"}:
            raise ValueError(f"seed {seed} does not contain both ablation arms")
        control = arms["control"]
        treatment = arms["treatment"]
        control_rows = {
            (row["static_level_signature"], row["state_key"]): row
            for row in control["rows"]}
        treatment_rows = {
            (row["static_level_signature"], row["state_key"]): row
            for row in treatment["rows"]}
        if control_rows.keys() != treatment_rows.keys():
            raise ValueError(f"seed {seed} arms evaluated different states")
        deltas = []
        for key in sorted(control_rows):
            c = control_rows[key]
            t = treatment_rows[key]
            row = {
                "hit": int(t["verified_top1"]) - int(c["verified_top1"]),
                "mass": float(t["verified_target_mass"]) - float(
                    c["verified_target_mass"]),
                "cross_entropy": float(t["policy_cross_entropy"]) - float(
                    c["policy_cross_entropy"]),
            }
            deltas.append(row)
            all_deltas.append(row)
        n = len(deltas)
        seed_rows.append({
            "seed": seed,
            "states": n,
            "control": control["metrics"],
            "treatment": treatment["metrics"],
            "paired": {
                "wins": sum(row["hit"] > 0 for row in deltas),
                "losses": sum(row["hit"] < 0 for row in deltas),
                "ties": sum(row["hit"] == 0 for row in deltas),
                "verified_top1_delta": sum(row["hit"] for row in deltas) / n,
                "verified_target_mass_delta": sum(
                    row["mass"] for row in deltas) / n,
                "policy_cross_entropy_delta": sum(
                    row["cross_entropy"] for row in deltas) / n,
            },
        })
    top1_deltas = [row["paired"]["verified_top1_delta"] for row in seed_rows]
    mass_deltas = [row["paired"]["verified_target_mass_delta"] for row in seed_rows]
    ce_deltas = [row["paired"]["policy_cross_entropy_delta"] for row in seed_rows]
    return {
        "schema_version": ABLATION_SCHEMA_VERSION,
        "semantics": "paired_promotion_policy_ablation_v1",
        "seed_count": len(seed_rows),
        "per_seed": seed_rows,
        "aggregate": {
            "verified_top1_delta_mean": statistics.fmean(top1_deltas),
            "verified_target_mass_delta_mean": statistics.fmean(mass_deltas),
            "policy_cross_entropy_delta_mean": statistics.fmean(ce_deltas),
            "wins": sum(row["hit"] > 0 for row in all_deltas),
            "losses": sum(row["hit"] < 0 for row in all_deltas),
            "ties": sum(row["hit"] == 0 for row in all_deltas),
        },
        "interpretation": {
            "positive_top1_or_mass_favors": "treatment",
            "negative_cross_entropy_favors": "treatment",
        },
        "final_test_status": "sealed_not_evaluated",
    }


def _comparison(reference: dict[str, Any], candidate: dict[str, Any]) -> dict[str, Any]:
    reference_rows = {
        (row["static_level_signature"], row["state_key"]): row
        for row in reference["rows"]}
    candidate_rows = {
        (row["static_level_signature"], row["state_key"]): row
        for row in candidate["rows"]}
    if reference_rows.keys() != candidate_rows.keys():
        raise ValueError("ablation arms evaluated different states")
    deltas = []
    for key in sorted(reference_rows):
        ref = reference_rows[key]
        cand = candidate_rows[key]
        deltas.append({
            "hit": int(cand["verified_top1"]) - int(ref["verified_top1"]),
            "mass": float(cand["verified_target_mass"])
            - float(ref["verified_target_mass"]),
            "cross_entropy": float(cand["policy_cross_entropy"])
            - float(ref["policy_cross_entropy"]),
        })
    count = len(deltas)
    return {
        "states": count,
        "wins": sum(row["hit"] > 0 for row in deltas),
        "losses": sum(row["hit"] < 0 for row in deltas),
        "ties": sum(row["hit"] == 0 for row in deltas),
        "verified_top1_delta": sum(row["hit"] for row in deltas) / count,
        "verified_target_mass_delta": sum(row["mass"] for row in deltas) / count,
        "policy_cross_entropy_delta": sum(
            row["cross_entropy"] for row in deltas) / count,
    }


def _three_arm_report(evaluations: list[dict[str, Any]]) -> dict[str, Any]:
    """Compare matched base, path-policy, and full-exact arms."""
    by_seed: dict[int, dict[str, dict[str, Any]]] = {}
    for evaluation in evaluations:
        by_seed.setdefault(int(evaluation["seed"]), {})[
            str(evaluation["arm"])] = evaluation
    specifications = {
        "path_vs_base": ("base", "path"),
        "full_vs_base": ("base", "full"),
        "path_vs_full": ("full", "path"),
    }
    seed_rows: list[dict[str, Any]] = []
    for seed, arms in sorted(by_seed.items()):
        if set(arms) != {"base", "path", "full"}:
            raise ValueError(f"seed {seed} does not contain all three matched arms")
        seed_rows.append({
            "seed": seed,
            "arms": {arm: arms[arm]["metrics"] for arm in ("base", "path", "full")},
            "comparisons": {
                name: _comparison(arms[reference], arms[candidate])
                for name, (reference, candidate) in specifications.items()
            },
        })
    aggregates: dict[str, Any] = {}
    for name in specifications:
        comparisons = [row["comparisons"][name] for row in seed_rows]
        aggregates[name] = {
            "verified_top1_delta_mean": statistics.fmean(
                row["verified_top1_delta"] for row in comparisons),
            "verified_target_mass_delta_mean": statistics.fmean(
                row["verified_target_mass_delta"] for row in comparisons),
            "policy_cross_entropy_delta_mean": statistics.fmean(
                row["policy_cross_entropy_delta"] for row in comparisons),
            "wins": sum(row["wins"] for row in comparisons),
            "losses": sum(row["losses"] for row in comparisons),
            "ties": sum(row["ties"] for row in comparisons),
        }
    return {
        "schema_version": ABLATION_SCHEMA_VERSION,
        "semantics": "matched_three_arm_promotion_policy_ablation_v1",
        "seed_count": len(seed_rows),
        "per_seed": seed_rows,
        "aggregate": aggregates,
        "comparison_direction": "candidate_minus_reference",
        "interpretation": {
            "positive_top1_or_mass_favors": "candidate",
            "negative_cross_entropy_favors": "candidate",
        },
        "final_test_status": "sealed_not_evaluated",
    }


def _prune_run_checkpoints(run_dir: Path) -> dict[str, Any]:
    """Keep authoritative active/best shards; remove redundant epoch shards."""
    root = run_dir.resolve()
    state_path = root / "run_state.json"
    if not state_path.is_file():
        raise FileNotFoundError(f"cannot prune run without state: {state_path}")
    state = json.loads(state_path.read_text(encoding="utf-8"))
    checkpoint_dir = (root / "checkpoints").resolve()
    if checkpoint_dir.parent != root:
        raise RuntimeError("unsafe checkpoint directory")
    keep = set()
    for role, relative in (
        ("active", state.get("active_checkpoint")),
        ("best", state.get("best_checkpoint")),
    ):
        if not relative:
            if role == "active":
                raise ValueError("run state is missing its active checkpoint")
            continue
        resolved = (root / relative).resolve()
        if (resolved.parent != checkpoint_dir
                or not re.fullmatch(r"epoch_\d+\.pt", resolved.name)):
            raise RuntimeError(f"unsafe {role} checkpoint path: {resolved}")
        if not resolved.is_file():
            raise FileNotFoundError(f"{role} checkpoint is missing: {resolved}")
        keep.add(resolved)
    initialization = checkpoint_dir / "epoch_000.pt"
    if initialization.is_file():
        keep.add(initialization.resolve())
    removed = 0
    removed_bytes = 0
    for path in checkpoint_dir.glob("epoch_*.pt"):
        resolved = path.resolve()
        if resolved.parent != checkpoint_dir:
            raise RuntimeError(f"unsafe checkpoint prune target: {resolved}")
        if resolved in keep:
            continue
        removed_bytes += path.stat().st_size
        path.unlink()
        removed += 1
    return {
        "policy": (
            "keep_initial_active_and_best_shards_plus_root_mirrors"),
        "removed_checkpoint_shards": removed,
        "removed_bytes": removed_bytes,
        "kept_checkpoint_shards": len([
            path for path in checkpoint_dir.glob("epoch_*.pt")]),
    }


def compact_supervised_runs(run_dirs: list[str | Path]) -> dict[str, Any]:
    """Compact explicitly named transactional supervised runs."""
    if not run_dirs:
        raise ValueError("at least one supervised run directory is required")
    rows = []
    for run_dir in run_dirs:
        path = Path(run_dir)
        rows.append({"run_dir": str(path), **_prune_run_checkpoints(path)})
    return {
        "runs": rows,
        "removed_checkpoint_shards": sum(
            row["removed_checkpoint_shards"] for row in rows),
        "removed_bytes": sum(row["removed_bytes"] for row in rows),
    }


def run_ablation(
    prepared_report: str | Path,
    *,
    seeds: list[int],
    epochs: int,
    batch_size: int,
    learning_rate: float,
    channels: int,
    residual_blocks: int,
    device: str,
) -> dict[str, Any]:
    prepared_report = Path(prepared_report)
    prepared = json.loads(prepared_report.read_text(encoding="utf-8"))
    _verify_prepared_artifacts(prepared)
    root = prepared_report.parent
    arm_order = prepared.get("arm_order", ["control", "treatment"])
    evaluations: list[dict[str, Any]] = []
    training_parser = build_training_parser()
    for seed in seeds:
        for arm in arm_order:
            dataset = prepared[arm]["path"]
            run_dir = root / "runs" / f"seed_{seed}" / arm
            state_path = run_dir / "run_state.json"
            argv = [
                "--dataset", dataset,
                "--output-dir", str(run_dir),
                "--split-manifest", prepared["training_manifest"]["path"],
                "--seed", str(seed),
                "--epochs", str(epochs),
                "--batch-size", str(batch_size),
                "--learning-rate", str(learning_rate),
                "--policy-loss-weight", "1",
                "--value-loss-weight", "0",
                "--channels", str(channels),
                "--residual-blocks", str(residual_blocks),
                "--device", device,
                "--train-ratio", "1",
                "--val-ratio", "0",
                "--test-ratio", "0",
            ]
            if state_path.is_file():
                state = json.loads(state_path.read_text(encoding="utf-8"))
                completed = int(state.get("completed_epochs", 0))
                if completed < epochs:
                    argv.extend(["--resume", str(run_dir / "last.pt")])
                elif completed > epochs:
                    raise ValueError(
                        f"{run_dir} already exceeds requested epochs")
            if not state_path.is_file() or int(json.loads(
                    state_path.read_text(encoding="utf-8")).get(
                        "completed_epochs", 0)) < epochs:
                run_training(training_parser.parse_args(argv))
            evaluation = evaluate_policy_checkpoint(
                run_dir / "last.pt", prepared["promotion"]["path"],
                device=device)
            evaluation.update({"seed": seed, "arm": arm})
            eval_path = run_dir / "promotion_evaluation.json"
            atomic_write_json(eval_path, evaluation)
            evaluation["checkpoint_retention"] = _prune_run_checkpoints(run_dir)
            atomic_write_json(eval_path, evaluation)
            evaluations.append(evaluation)
    if arm_order == ["control", "treatment"]:
        report = _paired_report(evaluations)
    elif arm_order == ["base", "path", "full"]:
        report = _three_arm_report(evaluations)
    else:
        raise ValueError(f"unsupported ablation arm order: {arm_order}")
    report.update({
        "prepared_report": str(prepared_report),
        "prepared_report_sha256": sha256_file(prepared_report),
        "seeds": seeds,
        "training": {
            "epochs": epochs, "batch_size": batch_size,
            "learning_rate": learning_rate, "channels": channels,
            "residual_blocks": residual_blocks,
            "policy_loss_weight": 1.0, "value_loss_weight": 0.0,
            "device": device,
        },
    })
    atomic_write_json(root / "paired_report.json", report)
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    export = sub.add_parser("export-promotion")
    export.add_argument("--pool", required=True)
    export.add_argument("--split", required=True)
    export.add_argument("--output", required=True)
    export.add_argument("--report")
    export.add_argument("--overwrite", action="store_true")

    export_training = sub.add_parser("export-path-training-levels")
    export_training.add_argument("--path-dataset", required=True)
    export_training.add_argument("--frozen-pool", required=True)
    export_training.add_argument("--output", required=True)
    export_training.add_argument("--report")
    export_training.add_argument("--overwrite", action="store_true")

    prepare = sub.add_parser("prepare")
    prepare.add_argument("--base-dataset", required=True)
    prepare.add_argument("--path-dataset", required=True)
    prepare.add_argument("--promotion-dataset", required=True)
    prepare.add_argument("--frozen-pool", required=True)
    prepare.add_argument("--frozen-split", required=True)
    prepare.add_argument("--output-dir", required=True)
    prepare.add_argument("--overwrite", action="store_true")

    matched = sub.add_parser("prepare-matched")
    matched.add_argument("--base-dataset", required=True)
    matched.add_argument("--path-dataset", required=True)
    matched.add_argument("--full-dataset", required=True)
    matched.add_argument("--promotion-dataset", required=True)
    matched.add_argument("--frozen-pool", required=True)
    matched.add_argument("--frozen-split", required=True)
    matched.add_argument("--output-dir", required=True)
    matched.add_argument("--expected-matched-count", type=int)
    matched.add_argument("--overwrite", action="store_true")

    run = sub.add_parser("run")
    run.add_argument("--prepared-report", required=True)
    run.add_argument("--seeds", nargs="+", type=int, default=[2051, 2052, 2053])
    run.add_argument("--epochs", type=int, default=30)
    run.add_argument("--batch-size", type=int, default=128)
    run.add_argument("--learning-rate", type=float, default=1e-3)
    run.add_argument("--channels", type=int, default=128)
    run.add_argument("--residual-blocks", type=int, default=6)
    run.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")

    compact = sub.add_parser("compact-checkpoints")
    compact.add_argument("--run-dir", nargs="+", required=True)
    return parser


def main(argv: Optional[list[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "export-promotion":
        result = export_promotion_levels(
            args.pool, args.split, args.output, report=args.report,
            overwrite=args.overwrite)
    elif args.command == "export-path-training-levels":
        result = export_path_training_levels(
            args.path_dataset, args.frozen_pool, args.output,
            report=args.report, overwrite=args.overwrite)
    elif args.command == "prepare":
        result = prepare_ablation(
            base_dataset=args.base_dataset, path_dataset=args.path_dataset,
            promotion_dataset=args.promotion_dataset,
            frozen_pool=args.frozen_pool, frozen_split=args.frozen_split,
            output_dir=args.output_dir, overwrite=args.overwrite)
    elif args.command == "prepare-matched":
        result = prepare_matched_ablation(
            base_dataset=args.base_dataset, path_dataset=args.path_dataset,
            full_dataset=args.full_dataset,
            promotion_dataset=args.promotion_dataset,
            frozen_pool=args.frozen_pool, frozen_split=args.frozen_split,
            output_dir=args.output_dir,
            expected_matched_count=args.expected_matched_count,
            overwrite=args.overwrite)
    elif args.command == "run":
        result = run_ablation(
            args.prepared_report, seeds=args.seeds, epochs=args.epochs,
            batch_size=args.batch_size, learning_rate=args.learning_rate,
            channels=args.channels, residual_blocks=args.residual_blocks,
            device=args.device)
    else:
        result = compact_supervised_runs(args.run_dir)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
