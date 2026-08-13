"""Fixed held-out promotion-validation/final-test split contracts."""

from __future__ import annotations

import json
import hashlib
import random
from dataclasses import replace
from pathlib import Path

import pytest

import blocksort.cotraining.hard_eval_pool as hard_eval_pool_mod
from blocksort.cotraining.config import CoTrainingConfig
from blocksort.cotraining.eval_split import (
    EVAL_SPLIT_ALGORITHM,
    HARD_EVAL_POOL_SCHEMA_VERSION,
    EvaluationSplitError,
    create_eval_split_manifest,
    evaluation_split_identity,
    load_eval_split_manifest,
    validate_common_evaluation_split,
)
from blocksort.cotraining.loop import CoTraining
from blocksort.cotraining.hard_eval_pool import (
    HardPoolConfig, generate_hard_eval_pool)
from blocksort.serialization import level_from_dict
from blocksort.serialization import level_to_dict
from blocksort.signature import static_level_signature


def _records(count: int = 8):
    records = []
    for index in range(count):
        size = 3 + index
        level = {
            "name": f"heldout-{index}",
            "cols": size,
            "rows": size,
            "blocks": [{"color": "red", "cells": [[1, 1]]}],
            "exits": [
                {"edge": "left", "start": 1, "length": 1, "color": "red"}
            ],
        }
        signature = static_level_signature(level_from_dict(level))
        records.append({
            "level_id": f"level-{index}",
            "static_level_signature": signature,
            "level": level,
        })
    return records


def _write_pool(path: Path, records) -> Path:
    path.write_text(
        "".join(json.dumps(record, sort_keys=True) + "\n"
                for record in records),
        encoding="utf-8",
    )
    return path


def _hard_record(index: int = 0, *, bucket: str = "in_distribution_hard"):
    record = _records(index + 1)[index]
    level_hash = hashlib.sha256(json.dumps(
        level_to_dict(level_from_dict(record["level"])),
        sort_keys=True, separators=(",", ":"),
        ensure_ascii=False).encode("utf-8")).hexdigest()
    return {
        **record,
        "hard_eval_pool_schema_version": HARD_EVAL_POOL_SCHEMA_VERSION,
        "canonical_level_sha256": level_hash,
        "generation_bucket": bucket,
        "generation_seed": 100 + index,
        "generation_parameters": {
            "rows": record["level"]["rows"],
            "cols": record["level"]["cols"],
            "color_count": 1,
            "density": 0.5,
            "mutation_budget": 8,
            "global_seed": 1729,
        },
        "oracle_validation": {
            "max_nodes": 200000,
            "exact": True,
            "solvable": True,
            "optimal_remaining_moves": 1,
        },
        "protagonist_filter": {
            "enabled": True,
            "difficulty_band": [0.2, 0.8],
            "solve_rate": 0.4,
            "retained": True,
        },
    }


def _make(tmp_path, *, records=None, seed=1729, name="eval_split.json"):
    pool = _write_pool(tmp_path / f"{name}.jsonl", records or _records())
    output = tmp_path / name
    manifest = create_eval_split_manifest(
        pool, output, validation_count=4, split_seed=seed)
    return pool, output, manifest


def _role_signatures(manifest):
    return (
        tuple(item["signature"] for item in manifest["promotion_validation"]),
        tuple(item["signature"] for item in manifest["final_test"]),
    )


def test_training_seed_does_not_affect_eval_split_membership(tmp_path):
    pool, path, manifest = _make(tmp_path)
    config_2028 = CoTrainingConfig(
        seed=2028, eval_levels_dataset=str(pool),
        eval_split_manifest=str(path), eval_split_seed=1729)
    config_2029 = CoTrainingConfig(
        seed=2029, eval_levels_dataset=str(pool),
        eval_split_manifest=str(path), eval_split_seed=1729)
    loaded_a = load_eval_split_manifest(
        config_2028.eval_split_manifest, config_2028.eval_levels_dataset,
        expected_split_seed=config_2028.eval_split_seed)
    loaded_b = load_eval_split_manifest(
        config_2029.eval_split_manifest, config_2029.eval_levels_dataset,
        expected_split_seed=config_2029.eval_split_seed)
    assert _role_signatures(loaded_a) == _role_signatures(loaded_b)
    assert (loaded_a["evaluation_split_fingerprint"]
            == loaded_b["evaluation_split_fingerprint"]
            == manifest["evaluation_split_fingerprint"])


def test_eval_split_seed_changes_fingerprint(tmp_path):
    records = _records()
    pool_a, _, first = _make(
        tmp_path, records=records, seed=1729, name="split-a.json")
    pool_b = _write_pool(tmp_path / "split-b.json.jsonl", records)
    second = create_eval_split_manifest(
        pool_b, tmp_path / "split-b.json", validation_count=4, split_seed=1730)
    assert first["evaluation_split_fingerprint"] != \
        second["evaluation_split_fingerprint"]
    assert _role_signatures(first) != _role_signatures(second)


def test_eval_split_is_input_order_independent(tmp_path):
    records = _records()
    shuffled = list(records)
    random.Random(99).shuffle(shuffled)
    pool_a = _write_pool(tmp_path / "ordered.jsonl", records)
    pool_b = _write_pool(tmp_path / "shuffled.jsonl", shuffled)
    first = create_eval_split_manifest(
        pool_a, tmp_path / "ordered.json", validation_count=4, split_seed=1729)
    second = create_eval_split_manifest(
        pool_b, tmp_path / "shuffled.json", validation_count=4, split_seed=1729)
    assert first["pool"]["sha256"] == second["pool"]["sha256"]
    assert first["evaluation_split_fingerprint"] == \
        second["evaluation_split_fingerprint"]
    assert _role_signatures(first) == _role_signatures(second)


def test_eval_split_is_disjoint_and_complete(tmp_path):
    _, _, manifest = _make(tmp_path)
    validation, test = map(set, _role_signatures(manifest))
    all_signatures = {
        record["static_level_signature"] for record in _records()}
    assert not validation & test
    assert validation | test == all_signatures
    assert manifest["split_algorithm"] == EVAL_SPLIT_ALGORITHM


def test_eval_split_rejects_duplicate_signatures(tmp_path):
    records = _records()
    records.append(dict(records[0]))
    pool = _write_pool(tmp_path / "duplicate.jsonl", records)
    with pytest.raises(EvaluationSplitError, match="duplicate held-out signature"):
        create_eval_split_manifest(
            pool, tmp_path / "split.json", validation_count=4)


def test_hard_eval_pool_manifest_preserves_bucket_metadata(tmp_path):
    records = [_hard_record(index, bucket=("hard" if index < 4 else "ood"))
               for index in range(8)]
    pool = _write_pool(tmp_path / "hard-pool.jsonl", records)
    manifest = create_eval_split_manifest(
        pool, tmp_path / "hard-split.json", validation_count=4)
    loaded = load_eval_split_manifest(tmp_path / "hard-split.json", pool)

    assert manifest["pool"]["record_count"] == 8
    assert loaded["evaluation_split_fingerprint"] == \
        manifest["evaluation_split_fingerprint"]


def test_hard_eval_pool_malformed_record_is_rejected(tmp_path):
    record = _hard_record()
    del record["generation_bucket"]
    pool = _write_pool(tmp_path / "bad-hard-pool.jsonl", [record, _hard_record(1)])

    with pytest.raises(EvaluationSplitError, match="generation_bucket"):
        create_eval_split_manifest(
            pool, tmp_path / "bad-hard-split.json", validation_count=1)


def test_adversarial_bucket_rejects_random_generator_provenance(tmp_path):
    record = _hard_record(bucket="adversarial_designer_hard")
    pool = _write_pool(
        tmp_path / "false-adversarial.jsonl", [record, _hard_record(1)])

    with pytest.raises(EvaluationSplitError, match="trained_designer"):
        create_eval_split_manifest(
            pool, tmp_path / "false-adversarial.json", validation_count=1)


def test_generated_hard_pool_hash_matches_manifest_identity(tmp_path):
    pool = tmp_path / "generated-hard-pool.jsonl"
    summary = generate_hard_eval_pool(HardPoolConfig(
        output=str(pool), total_count=3,
        buckets=("in_distribution_hard",), seed=77,
        rows=3, cols=3, color_count=1,
        density_min=0.25, density_max=0.35,
        mutation_budget_min=1, mutation_budget_max=2,
        oracle_validation_budget=50_000,
    ))
    manifest = create_eval_split_manifest(
        pool, tmp_path / "generated-hard-split.json", validation_count=1)

    assert summary["pool_sha256"] == manifest["pool"]["sha256"]


def test_hard_pool_excludes_levels_from_prior_files(tmp_path):
    baseline = tmp_path / "baseline.jsonl"
    common = HardPoolConfig(
        output=str(baseline), total_count=1,
        buckets=("in_distribution_hard",), seed=77,
        rows=3, cols=3, color_count=1,
        density_min=0.25, density_max=0.35,
        mutation_budget_min=1, mutation_budget_max=2,
        oracle_validation_budget=50_000,
    )
    generate_hard_eval_pool(common)
    excluded = tmp_path / "excluded.json"
    baseline_record = json.loads(baseline.read_text(encoding="utf-8"))
    excluded.write_text(
        json.dumps([baseline_record["level"]]), encoding="utf-8")

    holdout = tmp_path / "holdout.jsonl"
    summary = generate_hard_eval_pool(replace(
        common,
        output=str(holdout),
        exclude_level_files=(str(excluded),),
    ))
    holdout_record = json.loads(holdout.read_text(encoding="utf-8"))

    assert summary["excluded_rejected"] >= 1
    assert summary["excluded_unique_signature_count"] == 1
    assert (
        holdout_record["static_level_signature"]
        != baseline_record["static_level_signature"]
    )


def test_hard_pool_excludes_levels_wrapped_in_training_records(tmp_path):
    baseline = tmp_path / "baseline.jsonl"
    common = HardPoolConfig(
        output=str(baseline), total_count=1,
        buckets=("in_distribution_hard",), seed=77,
        rows=3, cols=3, color_count=1,
        density_min=0.25, density_max=0.35,
        mutation_budget_min=1, mutation_budget_max=2,
        oracle_validation_budget=50_000,
    )
    generate_hard_eval_pool(common)
    baseline_record = json.loads(baseline.read_text(encoding="utf-8"))
    training_data = tmp_path / "training.jsonl"
    training_data.write_text(json.dumps({
        "level": baseline_record["level"],
        "state": {"blocks": []},
        "policy_target": [],
    }), encoding="utf-8")

    holdout = tmp_path / "holdout.jsonl"
    summary = generate_hard_eval_pool(replace(
        common,
        output=str(holdout),
        exclude_level_files=(str(training_data),),
    ))
    holdout_record = json.loads(holdout.read_text(encoding="utf-8"))

    assert summary["excluded_rejected"] >= 1
    assert summary["excluded_unique_signature_count"] == 1
    assert (
        holdout_record["static_level_signature"]
        != baseline_record["static_level_signature"]
    )


def test_hard_pool_interruption_resumes_deterministically(
    tmp_path, monkeypatch, capsys,
):
    common = HardPoolConfig(
        output=str(tmp_path / "unused.jsonl"), total_count=3,
        buckets=("in_distribution_hard",), seed=77,
        rows=3, cols=3, color_count=1,
        density_min=0.25, density_max=0.35,
        mutation_budget_min=1, mutation_budget_max=2,
        oracle_validation_budget=50_000,
        checkpoint_every_attempts=1,
    )
    baseline = tmp_path / "baseline.jsonl"
    baseline_summary = generate_hard_eval_pool(
        replace(common, output=str(baseline)))

    interrupted = tmp_path / "interrupted.jsonl"
    interrupted_cfg = replace(common, output=str(interrupted))
    real_atomic_write_json = hard_eval_pool_mod.atomic_write_json
    writes = 0

    def interrupt_after_first_checkpoint(path, value):
        nonlocal writes
        real_atomic_write_json(path, value)
        writes += 1
        if writes == 1:
            raise RuntimeError("injected interruption")

    monkeypatch.setattr(
        hard_eval_pool_mod, "atomic_write_json",
        interrupt_after_first_checkpoint)
    with pytest.raises(RuntimeError, match="injected interruption"):
        generate_hard_eval_pool(interrupted_cfg)
    partial = interrupted.with_name(interrupted.name + ".partial.json")
    assert partial.is_file()
    assert not interrupted.exists()

    monkeypatch.setattr(
        hard_eval_pool_mod, "atomic_write_json", real_atomic_write_json)
    with pytest.raises(RuntimeError, match="different generation settings"):
        generate_hard_eval_pool(replace(interrupted_cfg, seed=78))

    resumed_summary = generate_hard_eval_pool(interrupted_cfg)

    assert resumed_summary["resumed_from_partial"] is True
    assert resumed_summary["pool_sha256"] == baseline_summary["pool_sha256"]
    assert interrupted.read_bytes() == baseline.read_bytes()
    assert not partial.exists()
    assert "resuming hard-pool generation" in capsys.readouterr().out


def test_adversarial_hard_bucket_requires_a_real_designer(tmp_path):
    with pytest.raises(ValueError, match="requires --designer-checkpoint"):
        generate_hard_eval_pool(HardPoolConfig(
            output=str(tmp_path / "hard.jsonl"), total_count=2,
            buckets=("adversarial_designer_hard",),
        ))


@pytest.mark.parametrize("time_limit", [-1.0, float("nan"), float("inf")])
def test_hard_pool_rejects_invalid_oracle_time_limit(tmp_path, time_limit):
    with pytest.raises(
            ValueError, match="oracle_validation_time_limit_seconds"):
        generate_hard_eval_pool(HardPoolConfig(
            output=str(tmp_path / "hard.jsonl"), total_count=1,
            buckets=("in_distribution_hard",),
            oracle_validation_time_limit_seconds=time_limit,
        ))


def test_eval_split_identity_changes_when_hard_pool_metadata_changes(tmp_path):
    records_a = [_hard_record(index) for index in range(8)]
    records_b = [dict(record) for record in records_a]
    records_b[0] = {
        **records_b[0],
        "generation_parameters": {
            **records_b[0]["generation_parameters"],
            "mutation_budget": 9,
        },
    }
    pool_a = _write_pool(tmp_path / "hard-a.jsonl", records_a)
    pool_b = _write_pool(tmp_path / "hard-b.jsonl", records_b)
    manifest_a = create_eval_split_manifest(
        pool_a, tmp_path / "hard-a.json", validation_count=4)
    manifest_b = create_eval_split_manifest(
        pool_b, tmp_path / "hard-b.json", validation_count=4)

    assert manifest_a["pool"]["sha256"] != manifest_b["pool"]["sha256"]
    assert manifest_a["evaluation_split_fingerprint"] != \
        manifest_b["evaluation_split_fingerprint"]


def test_eval_split_detects_pool_content_change(tmp_path):
    pool, path, _ = _make(tmp_path)
    records = _records()
    records[0]["level"]["blocks"][0]["cells"] = [[1, 2]]
    _write_pool(pool, records)
    with pytest.raises(EvaluationSplitError, match="pool content hash differs"):
        load_eval_split_manifest(path, pool, expected_split_seed=1729)


def test_eval_split_detects_manifest_tampering(tmp_path):
    pool, path, _ = _make(tmp_path)
    data = json.loads(path.read_text(encoding="utf-8"))
    data["final_test"].append(data["promotion_validation"].pop())
    path.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(EvaluationSplitError, match="integrity hash mismatch"):
        load_eval_split_manifest(path, pool)


def test_eval_split_missing_manifest_never_regenerates(tmp_path):
    pool = _write_pool(tmp_path / "pool.jsonl", _records())
    missing = tmp_path / "missing.json"
    with pytest.raises(FileNotFoundError, match="will not be regenerated"):
        load_eval_split_manifest(missing, pool)
    assert not missing.exists()


def test_resume_with_missing_eval_split_manifest_fails_before_mutation(tmp_path):
    pool = _write_pool(tmp_path / "pool.jsonl", _records())
    missing = tmp_path / "missing.json"
    output = tmp_path / "run"
    output.mkdir()
    run_state = output / "run_state.json"
    run_state.write_text('{"completed_rounds":[]}', encoding="utf-8")
    before = run_state.read_bytes()
    config = CoTrainingConfig(
        protagonist_checkpoint="unused.pt",
        designer_checkpoint="unused.pt",
        base_dataset=str(pool),
        output_dir=str(output),
        eval_levels_dataset=str(pool),
        eval_split_manifest=str(missing),
    )
    with pytest.raises(FileNotFoundError, match="will not be regenerated"):
        CoTraining(config).run()
    assert run_state.read_bytes() == before
    assert not missing.exists()


def test_cross_run_evaluation_limit_does_not_change_roles_but_blocks_aggregation(
    tmp_path,
):
    _, _, manifest = _make(tmp_path)
    full = _role_signatures(manifest)
    identity_2 = evaluation_split_identity(manifest, eval_limit=2)
    identity_4 = evaluation_split_identity(manifest, eval_limit=4)
    assert _role_signatures(manifest) == full
    assert (identity_2["evaluation_split_fingerprint"]
            == identity_4["evaluation_split_fingerprint"])
    assert validate_common_evaluation_split([identity_2, dict(identity_2)]) \
        == identity_2
    with pytest.raises(EvaluationSplitError, match="eval_limit"):
        validate_common_evaluation_split([identity_2, identity_4])


def test_external_eval_loader_constructs_only_validation_states(
        tmp_path, monkeypatch):
    pool, path, manifest = _make(tmp_path)
    config = CoTrainingConfig(
        seed=2028, eval_levels_dataset=str(pool),
        eval_split_manifest=str(path), eval_split_seed=1729)
    runner = CoTraining(config)
    constructed_signatures = []
    initial_state = runner.env.initial_state

    def tracked_initial_state(level):
        constructed_signatures.append(static_level_signature(level))
        return initial_state(level)

    monkeypatch.setattr(runner.env, "initial_state", tracked_initial_state)
    validation, signatures = runner._external_validation_states(pool, manifest)
    validation_signatures = tuple(
        static_level_signature(state.level) for state in validation)
    assert validation_signatures == _role_signatures(manifest)[0]
    assert tuple(constructed_signatures) == validation_signatures
    assert set(_role_signatures(manifest)[0]) | \
        set(_role_signatures(manifest)[1]) == signatures


def test_eval_split_config_requires_pool_and_manifest_together():
    with pytest.raises(ValueError, match="must be provided together"):
        CoTrainingConfig(eval_levels_dataset="pool.jsonl")
    with pytest.raises(ValueError, match="must be provided together"):
        CoTrainingConfig(eval_split_manifest="split.json")
