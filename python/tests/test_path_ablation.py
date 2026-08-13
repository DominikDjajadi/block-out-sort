from __future__ import annotations

import json

import pytest

from blocksort import Environment, Oracle, level_from_dict
from blocksort.cotraining.eval_split import create_eval_split_manifest
from blocksort.dataset.generate import write_jsonl
from blocksort.dataset.schema import build_exact_path_record, build_record
from blocksort.serialization import level_to_dict
from blocksort.signature import static_level_signature
from blocksort.solver import solve_astar
from blocksort.training import path_ablation
from blocksort.training.path_ablation import (
    _paired_report,
    _three_arm_report,
    export_path_training_levels,
    export_promotion_levels,
    prepare_ablation,
    prepare_matched_ablation,
)


def _level(index: int):
    return level_from_dict({
        "name": f"level-{index}", "cols": 4, "rows": 4,
        "blocks": [{"color": "red", "cells": [[1, 1 + index % 2]]}],
        "exits": [{
            "edge": "top", "start": 1 + index % 2,
            "length": 1, "color": "red",
        }],
        "holes": [[3, index % 3]],
    })


def _pool(tmp_path, count=4):
    records = []
    for index in range(count):
        level = _level(index)
        records.append({
            "level_id": f"heldout-{index}",
            "static_level_signature": static_level_signature(level),
            "level": level_to_dict(level),
        })
    path = tmp_path / "pool.jsonl"
    path.write_text(
        "".join(json.dumps(record) + "\n" for record in records),
        encoding="utf-8")
    split = tmp_path / "split.json"
    manifest = create_eval_split_manifest(
        path, split, validation_count=2, split_seed=17)
    return path, split, manifest, records


def _full_record(level, name):
    env = Environment()
    state = env.initial_state(level)
    record = build_record(Oracle(env).analyze(state), state, level_id=name)
    assert record is not None
    return record


def _path_record(level, name):
    env = Environment()
    state = env.initial_state(level)
    record = build_exact_path_record(
        solve_astar(env, state), state, env, level_id=name)
    assert record is not None
    return record


def test_export_promotion_never_exports_final_test(tmp_path):
    pool, split, manifest, records = _pool(tmp_path)
    output = tmp_path / "promotion.json"
    report = export_promotion_levels(pool, split, output)
    exported = json.loads(output.read_text(encoding="utf-8"))
    exported_signatures = {
        static_level_signature(level_from_dict(level)) for level in exported}
    expected = {item["signature"] for item in manifest["promotion_validation"]}
    final = {item["signature"] for item in manifest["final_test"]}
    assert exported_signatures == expected
    assert not exported_signatures & final
    assert report["final_test_status"] == "sealed_not_exported"


def test_export_path_training_levels_is_unique_and_frozen_disjoint(tmp_path):
    pool, _split, _manifest, _records = _pool(tmp_path)
    path_records = [_path_record(_level(10), "a"), _path_record(_level(11), "b")]
    source = tmp_path / "path.jsonl"
    output = tmp_path / "levels.json"
    write_jsonl(path_records, source)

    report = export_path_training_levels(source, pool, output)

    levels = json.loads(output.read_text(encoding="utf-8"))
    assert len(levels) == 2
    assert report["unique_static_signature_count"] == 2
    assert report["frozen_overlap_count"] == 0


def test_prepare_uses_shared_training_manifest_and_exact_promotion_role(tmp_path):
    pool, split, manifest, pool_records = _pool(tmp_path)
    promotion_signatures = {
        item["signature"] for item in manifest["promotion_validation"]}
    by_signature = {
        record["static_level_signature"]: record for record in pool_records}
    promotion = [
        _path_record(
            level_from_dict(by_signature[signature]["level"]), signature)
        for signature in sorted(promotion_signatures)]
    base = [_full_record(_level(10), "base")]
    path = [_path_record(_level(11), "path")]
    base_file = tmp_path / "base.jsonl"
    path_file = tmp_path / "path.jsonl"
    promotion_file = tmp_path / "promotion.jsonl"
    write_jsonl(base, base_file)
    write_jsonl(path, path_file)
    write_jsonl(promotion, promotion_file)

    prepared = prepare_ablation(
        base_dataset=base_file, path_dataset=path_file,
        promotion_dataset=promotion_file, frozen_pool=pool,
        frozen_split=split, output_dir=tmp_path / "ablation")

    training_manifest = json.loads(
        (tmp_path / "ablation" / "training_manifest.json").read_text())
    assert set(training_manifest["train_levels"]) == {
        base[0]["static_level_signature"], path[0]["static_level_signature"]}
    assert prepared["control"]["output_records"] == 1
    assert prepared["treatment"]["output_records"] == 2
    assert prepared["promotion"]["output_records"] == 2
    assert prepared["final_test_status"] == "sealed_not_labelled_or_evaluated"


def test_prepare_rejects_training_overlap_with_frozen_pool(tmp_path):
    pool, split, manifest, pool_records = _pool(tmp_path)
    promotion_signatures = {
        item["signature"] for item in manifest["promotion_validation"]}
    by_signature = {
        record["static_level_signature"]: record for record in pool_records}
    promotion = [
        _path_record(
            level_from_dict(by_signature[signature]["level"]), signature)
        for signature in sorted(promotion_signatures)]
    frozen_training = [_full_record(
        level_from_dict(pool_records[0]["level"]), "leak")]
    path = [_path_record(_level(20), "path")]
    base_file = tmp_path / "base.jsonl"
    path_file = tmp_path / "path.jsonl"
    promotion_file = tmp_path / "promotion.jsonl"
    write_jsonl(frozen_training, base_file)
    write_jsonl(path, path_file)
    write_jsonl(promotion, promotion_file)
    with pytest.raises(ValueError, match="overlaps"):
        prepare_ablation(
            base_dataset=base_file, path_dataset=path_file,
            promotion_dataset=promotion_file, frozen_pool=pool,
            frozen_split=split, output_dir=tmp_path / "bad")


def test_prepare_matched_requires_and_builds_identical_augmentation_states(tmp_path):
    pool, split, manifest, pool_records = _pool(tmp_path)
    by_signature = {
        record["static_level_signature"]: record for record in pool_records}
    promotion = [
        _path_record(
            level_from_dict(by_signature[item["signature"]]["level"]),
            item["signature"])
        for item in manifest["promotion_validation"]]
    matched_level = _level(11)
    base = [_full_record(_level(10), "base")]
    path = [_path_record(matched_level, "matched")]
    full = [_full_record(matched_level, "matched")]
    files = {}
    for name, records in {
        "base": base, "path": path, "full": full, "promotion": promotion,
    }.items():
        files[name] = tmp_path / f"{name}.jsonl"
        write_jsonl(records, files[name])

    prepared = prepare_matched_ablation(
        base_dataset=files["base"], path_dataset=files["path"],
        full_dataset=files["full"], promotion_dataset=files["promotion"],
        frozen_pool=pool, frozen_split=split,
        output_dir=tmp_path / "matched", expected_matched_count=1)

    assert prepared["arm_order"] == ["base", "path", "full"]
    assert prepared["matched_augmentation_count"] == 1
    assert prepared["base"]["output_records"] == 1
    assert prepared["path"]["output_records"] == 2
    assert prepared["full"]["output_records"] == 2
    path_ablation._verify_prepared_artifacts(prepared)

    different_full = [_full_record(_level(10), "different")]
    write_jsonl(different_full, files["full"])
    with pytest.raises(ValueError, match="exactly the same states"):
        prepare_matched_ablation(
            base_dataset=files["base"], path_dataset=files["path"],
            full_dataset=files["full"], promotion_dataset=files["promotion"],
            frozen_pool=pool, frozen_split=split,
            output_dir=tmp_path / "mismatch")


def test_prepared_artifact_verification_detects_replacement(tmp_path):
    artifact_names = (
        "control", "treatment", "promotion",
        "training_manifest", "promotion_manifest",
    )
    prepared = {"final_test_status": "sealed_not_labelled_or_evaluated"}
    for name in artifact_names:
        path = tmp_path / f"{name}.json"
        path.write_text(name, encoding="utf-8")
        prepared[name] = {
            "path": str(path),
            "sha256": path_ablation.sha256_file(path),
        }

    path_ablation._verify_prepared_artifacts(prepared)
    (tmp_path / "promotion.json").write_text("replaced", encoding="utf-8")

    with pytest.raises(ValueError, match="promotion artifact hash mismatch"):
        path_ablation._verify_prepared_artifacts(prepared)


def test_checkpoint_pruning_keeps_active_and_best_shards(tmp_path):
    checkpoint_dir = tmp_path / "checkpoints"
    checkpoint_dir.mkdir()
    sizes = {}
    for epoch in range(1, 5):
        path = checkpoint_dir / f"epoch_{epoch:03d}.pt"
        path.write_bytes(bytes([epoch]) * epoch)
        sizes[epoch] = path.stat().st_size
    (tmp_path / "run_state.json").write_text(json.dumps({
        "active_checkpoint": "checkpoints/epoch_004.pt",
        "best_checkpoint": "checkpoints/epoch_002.pt",
    }), encoding="utf-8")

    result = path_ablation._prune_run_checkpoints(tmp_path)

    assert result["removed_checkpoint_shards"] == 2
    assert result["removed_bytes"] == sizes[1] + sizes[3]
    assert sorted(path.name for path in checkpoint_dir.glob("*.pt")) == [
        "epoch_002.pt", "epoch_004.pt"]


def test_checkpoint_pruning_preserves_inherited_initialization(tmp_path):
    checkpoint_dir = tmp_path / "checkpoints"
    checkpoint_dir.mkdir()
    for epoch in range(3):
        (checkpoint_dir / f"epoch_{epoch:03d}.pt").write_bytes(bytes([epoch]))
    (tmp_path / "run_state.json").write_text(json.dumps({
        "active_checkpoint": "checkpoints/epoch_002.pt",
        "best_checkpoint": "checkpoints/epoch_002.pt",
    }), encoding="utf-8")

    path_ablation._prune_run_checkpoints(tmp_path)

    assert sorted(path.name for path in checkpoint_dir.glob("*.pt")) == [
        "epoch_000.pt", "epoch_002.pt"]


def test_checkpoint_pruning_rejects_state_paths_outside_checkpoint_dir(tmp_path):
    (tmp_path / "checkpoints").mkdir()
    outside = tmp_path / "outside.pt"
    outside.write_bytes(b"checkpoint")
    (tmp_path / "run_state.json").write_text(json.dumps({
        "active_checkpoint": "outside.pt",
    }), encoding="utf-8")

    with pytest.raises(RuntimeError, match="unsafe active checkpoint"):
        path_ablation._prune_run_checkpoints(tmp_path)

    assert outside.is_file()


def test_paired_report_counts_state_level_wins_and_losses():
    def evaluation(seed, arm, hits, masses):
        rows = [{
            "static_level_signature": f"sig-{index}", "state_key": "state",
            "verified_top1": hit, "verified_target_mass": masses[index],
            "policy_cross_entropy": 2.0 - masses[index],
        } for index, hit in enumerate(hits)]
        return {
            "seed": seed, "arm": arm, "rows": rows,
            "metrics": {"verified_top1_acc": sum(hits) / len(hits)},
        }

    report = _paired_report([
        evaluation(1, "control", [0, 1], [0.2, 0.7]),
        evaluation(1, "treatment", [1, 0], [0.4, 0.6]),
        evaluation(2, "control", [0, 0], [0.1, 0.2]),
        evaluation(2, "treatment", [1, 0], [0.3, 0.2]),
    ])
    assert report["aggregate"]["wins"] == 2
    assert report["aggregate"]["losses"] == 1
    assert report["aggregate"]["ties"] == 1
    assert report["final_test_status"] == "sealed_not_evaluated"


def test_three_arm_report_exposes_all_causal_comparisons():
    def evaluation(arm, hits, mass):
        return {
            "seed": 7, "arm": arm,
            "metrics": {"verified_top1_acc": sum(hits) / len(hits)},
            "rows": [{
                "static_level_signature": f"sig-{index}",
                "state_key": "state", "verified_top1": hit,
                "verified_target_mass": mass[index],
                "policy_cross_entropy": 1.0 - mass[index],
            } for index, hit in enumerate(hits)],
        }

    report = _three_arm_report([
        evaluation("base", [0, 0], [0.2, 0.2]),
        evaluation("path", [1, 0], [0.5, 0.3]),
        evaluation("full", [1, 1], [0.6, 0.6]),
    ])

    assert set(report["aggregate"]) == {
        "path_vs_base", "full_vs_base", "path_vs_full"}
    assert report["aggregate"]["path_vs_base"]["wins"] == 1
    assert report["aggregate"]["path_vs_full"]["losses"] == 1
