"""Focused regression tests for content-based run identity."""

from __future__ import annotations

import copy
from dataclasses import replace
import json
from pathlib import Path
import shutil

import pytest

from blocksort.cotraining.config import CoTrainingConfig, CurriculumState
from blocksort.cotraining.eval_split import (
    EvaluationSplitError, create_eval_split_manifest)
from blocksort.cotraining.loop import (
    CoTraining, _cotraining_experiment_spec)
from blocksort.designer.checkpoint import save_designer
from blocksort.designer.model import DesignerModelConfig, DesignerNet
from blocksort.expert_iteration.config import ExpertIterationConfig
from blocksort.expert_iteration.iterate import (
    ExpertIteration, _expert_iteration_experiment_spec)
from blocksort.final_benchmark.run import (
    _final_benchmark_spec, build_parser as build_final_parser)
from blocksort.final_benchmark.common import checkpoint_identity
from blocksort.training.checkpoint import save_checkpoint
from blocksort.training.config import (
    EncodingConfig, ModelConfig, ValueNormConfig)
from blocksort.training.dataset import load_records
from blocksort.training.experiment_identity import (
    EXPERIMENT_SPEC_FILE, ContinuationHorizonError, ExperimentIdentityError,
    ExperimentSpecIntegrityError, LegacyRunMigrationError,
    MissingRunStateError,
    build_experiment_spec, compare_experiment_specs,
    fingerprint_experiment_spec,
    load_legacy_migration_spec, load_persisted_experiment_spec,
    persist_legacy_migration_manifest, validate_field_classification,
    validate_or_initialize_experiment)
from blocksort.training.model import PolicyValueNet
from blocksort.training.splits import SplitRatios, make_split, save_manifest
from blocksort.training.transaction import (
    atomic_write_json, relative_to_run, sha256_file)
from blocksort.training import experiment_identity as identity_mod


REPO = Path(__file__).resolve().parents[2]
SOURCE_DATASET = REPO / "data" / "training" / "pv_smoke.jsonl"


def _protagonist(path: Path, seed: int) -> Path:
    import torch

    torch.manual_seed(seed)
    encoding = EncodingConfig()
    model_config = ModelConfig(
        channels=4, residual_blocks=1, value_hidden_size=8)
    save_checkpoint(
        path, model=PolicyValueNet(encoding, model_config), optimizer=None,
        scheduler=None, epoch=0, best_val_metric=None,
        encoding_config=encoding, model_config=model_config,
        value_norm=ValueNormConfig(), seed=seed, dataset_version=1,
        split_identity=None)
    return path


def _designer(path: Path, seed: int) -> Path:
    import torch

    torch.manual_seed(seed)
    encoding = EncodingConfig()
    model_config = DesignerModelConfig(
        channels=4, residual_blocks=1, hidden_size=8)
    save_designer(
        path, model=DesignerNet(encoding, model_config),
        encoding_config=encoding, model_config=model_config, seed=seed)
    return path


@pytest.fixture()
def identity_inputs(tmp_path):
    dataset = tmp_path / "dataset.jsonl"
    shutil.copyfile(SOURCE_DATASET, dataset)
    evaluation = tmp_path / "evaluation.jsonl"
    unique = {}
    for line in SOURCE_DATASET.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        unique.setdefault(record["static_level_signature"], record)
        if len(unique) == 4:
            break
    evaluation.write_text(
        "".join(json.dumps(record) + "\n" for record in unique.values()),
        encoding="utf-8")
    evaluation_split = tmp_path / "evaluation_split.json"
    create_eval_split_manifest(
        evaluation, evaluation_split, validation_count=2, split_seed=1729)
    return {
        "dataset": dataset,
        "evaluation": evaluation,
        "evaluation_split": evaluation_split,
        "protagonist_a": _protagonist(tmp_path / "protagonist_a.pt", 1),
        "protagonist_b": _protagonist(tmp_path / "protagonist_b.pt", 2),
        "designer_a": _designer(tmp_path / "designer_a.pt", 3),
        "designer_b": _designer(tmp_path / "designer_b.pt", 4),
    }


def _co_config(tmp_path, inputs, **changes):
    config = CoTrainingConfig(
        protagonist_checkpoint=str(inputs["protagonist_a"]),
        designer_checkpoint=str(inputs["designer_a"]),
        base_dataset=str(inputs["dataset"]),
        eval_levels_dataset=str(inputs["evaluation"]),
        eval_split_manifest=str(inputs["evaluation_split"]),
        eval_split_seed=1729,
        output_dir=str(tmp_path / "co_run"),
        rounds=1, levels_per_round=1, eval_budgets=(1, 4, 8, 32),
        promotion_budget=4,
        initial_curriculum=CurriculumState(
            rows=5, cols=5, color_count=2, density=0.3,
            mutation_budget=2))
    return replace(config, **changes)


def _expert_config(tmp_path, inputs, **changes):
    config = ExpertIterationConfig(
        initial_checkpoint=str(inputs["protagonist_a"]),
        base_dataset=str(inputs["dataset"]),
        output_dir=str(tmp_path / "expert_run"),
        iterations=1, levels_per_iteration=1,
        eval_budgets=(1, 4, 8, 32), promotion_budget=4)
    return replace(config, **changes)


def _initialize_pipeline_spec(root: Path, spec: dict) -> str:
    fingerprint, initialized = validate_or_initialize_experiment(
        root, spec, run_state=None)
    assert initialized
    run_state = {
        "experiment_fingerprint": fingerprint,
        "completed_rounds": [],
        "completed_iterations": [],
    }
    if spec.get("derived", {}).get("evaluation_split") is not None:
        run_state["evaluation_split"] = spec["derived"]["evaluation_split"]
    atomic_write_json(root / "run_state.json", run_state)
    return fingerprint


def test_canonical_identity_accepts_relocation_and_budget_reordering(
        tmp_path, identity_inputs):
    config = _co_config(tmp_path, identity_inputs)
    records = load_records(config.base_dataset)
    original = _cotraining_experiment_spec(
        replace(config, device="cpu"), records, resolved_device="cpu")

    moved = tmp_path / "moved.jsonl"
    shutil.copyfile(identity_inputs["dataset"], moved)
    requested = _cotraining_experiment_spec(
        replace(
            config, base_dataset=str(moved),
            eval_budgets=(32, 8, 4, 1),
            device="cpu"),
        load_records(moved), resolved_device="cpu")

    assert compare_experiment_specs(original, requested) == []
    assert fingerprint_experiment_spec(original) == \
        fingerprint_experiment_spec(requested)


def test_cotraining_label_strategy_is_identity_bound_and_legacy_is_explicit(
        tmp_path, identity_inputs):
    config = _co_config(tmp_path, identity_inputs)
    records = load_records(config.base_dataset)
    path_strategy = _cotraining_experiment_spec(config, records)
    legacy_strategy = _cotraining_experiment_spec(
        replace(config, label_mode="hybrid"), records)
    search_only = _cotraining_experiment_spec(
        replace(config, label_mode="search_only"), records)

    assert path_strategy["software_semantics"]["teacher_labeling_policy"] == \
        "full_exact_cached_path_then_neural_search_v1"
    assert "teacher_labeling_policy" not in legacy_strategy["software_semantics"]
    assert search_only["software_semantics"]["teacher_labeling_policy"] == \
        "neural_search_only_no_astar_v2"
    assert "semantic_config.label_mode" in {
        item.field for item in compare_experiment_specs(
            path_strategy, legacy_strategy)}


def test_cotraining_identity_includes_fixed_evaluation_split(
        tmp_path, identity_inputs):
    config = _co_config(tmp_path, identity_inputs)
    records = load_records(config.base_dataset)
    original = _cotraining_experiment_spec(config, records)

    alternate_manifest = tmp_path / "alternate_eval_split.json"
    create_eval_split_manifest(
        identity_inputs["evaluation"], alternate_manifest,
        validation_count=2, split_seed=1730)
    changed = replace(
        config,
        eval_split_manifest=str(alternate_manifest),
        eval_split_seed=1730,
    )
    alternate = _cotraining_experiment_spec(changed, records)
    differences = {
        difference.field for difference in
        compare_experiment_specs(original, alternate)}
    assert "derived.evaluation_split.split_seed" in differences
    assert "inputs.evaluation_split_manifest.sha256" in differences
    assert "derived.evaluation_split.evaluation_split_fingerprint" in differences
    assert fingerprint_experiment_spec(original) != \
        fingerprint_experiment_spec(alternate)


def test_cotraining_identity_binds_imported_replay_snapshots(
        tmp_path, identity_inputs):
    protagonist_replay = tmp_path / "protagonist-replay.jsonl"
    designer_replay = tmp_path / "designer-replay.jsonl"
    protagonist_replay.write_text('{"record":1}\n', encoding="utf-8")
    designer_replay.write_text('{"level":1}\n', encoding="utf-8")
    config = _co_config(
        tmp_path,
        identity_inputs,
        initial_protagonist_replay=str(protagonist_replay),
        initial_designer_replay=str(designer_replay),
    )
    records = load_records(config.base_dataset)
    original = _cotraining_experiment_spec(config, records)

    protagonist_replay.write_text('{"record":2}\n', encoding="utf-8")
    changed = _cotraining_experiment_spec(config, records)
    differences = {
        difference.field for difference in
        compare_experiment_specs(original, changed)}

    assert "inputs.initial_protagonist_replay.sha256" in differences
    assert (
        original["inputs"]["initial_designer_replay"]["sha256"]
        == changed["inputs"]["initial_designer_replay"]["sha256"]
    )


def test_cotraining_identity_binds_compatible_imported_base_split(
        tmp_path, identity_inputs):
    config = _co_config(tmp_path, identity_inputs)
    records = load_records(config.base_dataset)
    keys = sorted({
        record.get("static_level_signature") or record["level_id"]
        for record in records
    })
    split_path = tmp_path / "base-split.json"
    save_manifest(
        make_split(
            keys,
            ratios=SplitRatios(train=0.8, validation=0.1, test=0.1),
            seed=123,
        ),
        split_path,
    )
    imported = _cotraining_experiment_spec(
        replace(config, initial_base_split=str(split_path)),
        records,
    )
    generated = _cotraining_experiment_spec(config, records)

    assert imported["inputs"]["initial_base_split"]["sha256"] == \
        sha256_file(split_path)
    assert (
        imported["derived"]["split_manifest_sha256"]
        != generated["derived"]["split_manifest_sha256"]
    )


def test_cotraining_initialize_only_does_not_change_experiment_identity(
        tmp_path, identity_inputs):
    config = _co_config(tmp_path, identity_inputs)
    records = load_records(config.base_dataset)

    regular = _cotraining_experiment_spec(config, records)
    setup = _cotraining_experiment_spec(
        replace(config, initialize_only=True),
        records,
    )

    assert compare_experiment_specs(regular, setup) == []
    assert fingerprint_experiment_spec(regular) == \
        fingerprint_experiment_spec(setup)


def test_cotraining_identity_includes_budget_sweep_semantics(
        tmp_path, identity_inputs):
    config = _co_config(
        tmp_path,
        identity_inputs,
        promotion_metric="weighted_budget_sweep_confirmed_optimal_rate",
        promotion_budgets=(1, 2, 4),
        promotion_budget_weights=(0.5, 0.3, 0.2),
        eval_budgets=(1, 2, 4, 8),
    )
    records = load_records(config.base_dataset)
    original = _cotraining_experiment_spec(config, records)
    changed = _cotraining_experiment_spec(
        replace(config, promotion_budget_weights=(0.4, 0.4, 0.2)),
        records,
    )
    differences = {
        difference.field for difference in
        compare_experiment_specs(original, changed)}

    assert "semantic_config.promotion_budget_weights" in differences
    assert fingerprint_experiment_spec(original) != \
        fingerprint_experiment_spec(changed)


def test_cotraining_identity_includes_policy_target_profile(
        tmp_path, identity_inputs):
    config = _co_config(tmp_path, identity_inputs)
    records = load_records(config.base_dataset)
    original = _cotraining_experiment_spec(config, records)
    changed = _cotraining_experiment_spec(
        replace(config, policy_target_profile="recorded"),
        records,
    )
    differences = {
        difference.field for difference in
        compare_experiment_specs(original, changed)}

    assert "semantic_config.policy_target_profile" in differences
    assert fingerprint_experiment_spec(original) != \
        fingerprint_experiment_spec(changed)


def test_cotraining_identity_records_solve_promotion_and_yield_policy(
        tmp_path, identity_inputs):
    config = _co_config(
        tmp_path,
        identity_inputs,
        promotion_metric="weighted_budget_sweep_solve_rate",
        promotion_budgets=(4, 8, 16),
        promotion_budget_weights=(0.2, 0.3, 0.5),
        eval_budgets=(1, 2, 4, 8, 16),
    )
    spec = _cotraining_experiment_spec(
        config, load_records(config.base_dataset))

    assert spec["software_semantics"]["curriculum_adaptation_policy"] == \
        "frontier_yield_aware_v1"
    assert spec["software_semantics"]["full_level_solve_promotion_policy"] == \
        "completed_trajectory_over_total_evaluated_v1"
    assert spec["software_semantics"]["loss_aggregation_policy"] == \
        "global_supervision_mass_weighted_v1"
    assert spec["software_semantics"]["replay_source_weighting_policy"] == \
        "weighted_sampling_source_weighted_policy_loss_v2"


def test_cotraining_reports_all_semantic_and_input_differences_before_writes(
        tmp_path, identity_inputs):
    config = _co_config(tmp_path, identity_inputs)
    root = Path(config.output_dir)
    spec = _cotraining_experiment_spec(
        config, load_records(config.base_dataset))
    _initialize_pipeline_spec(root, spec)
    sentinels = {}
    for relative in ("best.pt", "replay/manifest.json", "round_001/report.json"):
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(f"sentinel:{relative}".encode())
        sentinels[path] = (sha256_file(path), path.stat().st_mtime_ns)
    state_before = (
        sha256_file(root / "run_state.json"),
        (root / "run_state.json").stat().st_mtime_ns)

    changed = replace(
        config,
        base_dataset=str(identity_inputs["evaluation"]),
        protagonist_checkpoint=str(identity_inputs["protagonist_b"]),
        designer_checkpoint=str(identity_inputs["designer_b"]),
        seed=config.seed + 1,
        promotion_budget=8,
        eval_budgets=(1, 4, 8, 32, 64),
        initial_curriculum=replace(
            config.initial_curriculum, rows=6, cols=6, color_count=3,
            mutation_budget=4))
    with pytest.raises(ExperimentIdentityError) as caught:
        CoTraining(changed).run()

    message = str(caught.value)
    for expected in (
            "inputs.base_dataset.sha256",
            "inputs.initial_protagonist.sha256",
            "inputs.initial_designer.sha256",
            "semantic_config.seed",
            "semantic_config.promotion_budget",
            "semantic_config.eval_budgets",
            "semantic_config.initial_curriculum.rows",
            "semantic_config.initial_curriculum.cols",
            "semantic_config.initial_curriculum.color_count",
            "semantic_config.initial_curriculum.mutation_budget",
            "new --output-dir"):
        assert expected in message
    assert state_before == (
        sha256_file(root / "run_state.json"),
        (root / "run_state.json").stat().st_mtime_ns)
    assert sentinels == {
        path: (sha256_file(path), path.stat().st_mtime_ns)
        for path in sentinels}


def test_cotraining_detects_same_path_dataset_and_evaluation_changes(
        tmp_path, identity_inputs):
    config = _co_config(tmp_path, identity_inputs)
    root = Path(config.output_dir)
    _initialize_pipeline_spec(
        root, _cotraining_experiment_spec(
            config, load_records(config.base_dataset)))
    state_hash = sha256_file(root / "run_state.json")

    with identity_inputs["dataset"].open("a", encoding="utf-8") as handle:
        handle.write("\n")
    rows = [
        json.loads(line) for line in
        identity_inputs["evaluation"].read_text(encoding="utf-8").splitlines()
        if line.strip()]
    rows[0]["level"]["name"] = "mutated-pool-content"
    identity_inputs["evaluation"].write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    with pytest.raises(EvaluationSplitError) as caught:
        CoTraining(config).run()
    message = str(caught.value)
    assert "pool content hash differs" in message
    assert sha256_file(root / "run_state.json") == state_hash


@pytest.mark.parametrize("pipeline", ["cotraining", "expert_iteration"])
def test_identified_missing_run_state_with_progress_fails_without_writes(
        tmp_path, identity_inputs, pipeline):
    if pipeline == "cotraining":
        config = _co_config(tmp_path, identity_inputs, rounds=0)
        root = Path(config.output_dir)
        spec = _cotraining_experiment_spec(
            config, load_records(config.base_dataset))
        progress = root / "round_001" / "report.prepared.json"
        runner = CoTraining(config)
    else:
        config = _expert_config(tmp_path, identity_inputs, iterations=0)
        root = Path(config.output_dir)
        spec = _expert_iteration_experiment_spec(
            config, load_records(config.base_dataset))
        progress = root / "iter_001" / "report.prepared.json"
        runner = ExpertIteration(config)
    validate_or_initialize_experiment(root, spec, run_state=None)
    atomic_write_json(root / "config.json", config.to_dict())
    progress.parent.mkdir(parents=True)
    progress.write_bytes(b"committed-progress-sentinel")
    replay = root / "replay" / "committed_001.jsonl"
    replay.parent.mkdir()
    replay.write_bytes(b"replay-sentinel")
    before = {
        path: (path.read_bytes(), path.stat().st_mtime_ns)
        for path in (
            root / EXPERIMENT_SPEC_FILE, root / "config.json",
            progress, replay)}

    with pytest.raises(MissingRunStateError, match="run_state.json is missing"):
        runner.run()
    assert before == {
        path: (path.read_bytes(), path.stat().st_mtime_ns)
        for path in before}
    assert not (root / "run_state.json").exists()


def test_identified_setup_only_expert_initialization_recovers(
        tmp_path, identity_inputs):
    config = _expert_config(tmp_path, identity_inputs, iterations=0)
    root = Path(config.output_dir)
    spec = _expert_iteration_experiment_spec(
        config, load_records(config.base_dataset))
    validate_or_initialize_experiment(root, spec, run_state=None)
    atomic_write_json(root / "config.json", config.to_dict())
    initial = root / "protagonist" / "initial.pt"
    initial.parent.mkdir()
    shutil.copyfile(identity_inputs["protagonist_a"], initial)

    result = ExpertIteration(config).run()
    assert result["run_state"]["completed_iterations"] == []
    assert (root / "run_state.json").is_file()


@pytest.mark.parametrize("pipeline", ["cotraining", "expert_iteration"])
@pytest.mark.parametrize("corruption", ["checkpoint", "replay", "report"])
def test_existing_run_corrupt_artifact_causes_no_write(
        tmp_path, identity_inputs, pipeline, corruption):
    if pipeline == "cotraining":
        base_config = _co_config(tmp_path, identity_inputs, rounds=0)
        requested_config = replace(base_config, rounds=1)
        root = Path(base_config.output_dir)
        spec = _cotraining_experiment_spec(
            base_config, load_records(base_config.base_dataset))
        completed_key = "completed_rounds"
        source_key = "active_protagonist_source_round"
        commit_number_key = "round"
        runner = CoTraining(requested_config)
    else:
        base_config = _expert_config(tmp_path, identity_inputs, iterations=0)
        requested_config = replace(base_config, iterations=1)
        root = Path(base_config.output_dir)
        spec = _expert_iteration_experiment_spec(
            base_config, load_records(base_config.base_dataset))
        completed_key = "completed_iterations"
        source_key = "active_protagonist_source_iteration"
        commit_number_key = "iteration"
        runner = ExpertIteration(requested_config)
    fingerprint, _ = validate_or_initialize_experiment(
        root, spec, run_state=None)
    active = root / "protagonist" / "committed_001.pt"
    active.parent.mkdir()
    shutil.copyfile(identity_inputs["protagonist_a"], active)
    replay = root / "replay" / "committed_001.jsonl"
    replay.parent.mkdir()
    replay.write_bytes(b"")
    prepared = root / (
        "round_001/report.prepared.json"
        if pipeline == "cotraining"
        else "iter_001/report.prepared.json")
    prepared.parent.mkdir()
    atomic_write_json(prepared, {"prepared": True})
    state = {
        "schema_version": 2,
        "experiment_fingerprint": fingerprint,
        completed_key: [1],
        "active_protagonist_checkpoint": relative_to_run(active, root),
        "active_protagonist_sha256": sha256_file(active),
        source_key: 1,
        "active_replay_snapshot": relative_to_run(replay, root),
        "active_replay_sha256": sha256_file(replay),
        "commits": [{
            commit_number_key: 1,
            "prepared_report": relative_to_run(prepared, root),
            "prepared_report_sha256": sha256_file(prepared),
        }],
    }
    if pipeline == "cotraining":
        level_replay = root / "level_replay" / "committed_001.jsonl"
        level_replay.parent.mkdir()
        level_replay.write_bytes(b"")
        state.update({
            "designer_checkpoint": str(identity_inputs["designer_a"]),
            "active_level_replay_snapshot":
                relative_to_run(level_replay, root),
            "active_level_replay_sha256": sha256_file(level_replay),
        })
    atomic_write_json(root / "run_state.json", state)
    config_path = root / "config.json"
    config_path.write_bytes(b"config-must-remain-byte-identical")

    if corruption == "checkpoint":
        active.write_bytes(active.read_bytes() + b"corrupt")
    elif corruption == "replay":
        replay.write_bytes(b"corrupt")
    else:
        prepared.unlink()
    watched = [
        root / EXPERIMENT_SPEC_FILE, root / "run_state.json", config_path,
        active, replay]
    watched = [path for path in watched if path.exists()]
    before = {
        path: (path.read_bytes(), path.stat().st_mtime_ns)
        for path in watched}

    with pytest.raises((ExperimentSpecIntegrityError, RuntimeError)):
        runner.run()
    assert before == {
        path: (path.read_bytes(), path.stat().st_mtime_ns)
        for path in watched}
    assert not (root / "best.pt").exists()


@pytest.mark.parametrize(
    "change, field",
    [
        ({"seed": 43}, "semantic_config.seed"),
        ({"promotion_budget": 8}, "semantic_config.promotion_budget"),
        ({"eval_budgets": (1, 4, 8, 64)}, "semantic_config.eval_budgets"),
        ({"label_mode": "hybrid"}, "semantic_config.label_mode"),
    ])
def test_expert_iteration_rejects_changed_identity_before_recovery(
        tmp_path, identity_inputs, change, field):
    config = _expert_config(tmp_path, identity_inputs)
    root = Path(config.output_dir)
    _initialize_pipeline_spec(
        root, _expert_iteration_experiment_spec(
            config, load_records(config.base_dataset)))
    best = root / "best.pt"
    best.write_bytes(b"do-not-repair")
    before = (sha256_file(best), best.stat().st_mtime_ns)

    with pytest.raises(ExperimentIdentityError, match=field):
        ExpertIteration(replace(config, **change)).run()
    assert before == (sha256_file(best), best.stat().st_mtime_ns)


def test_expert_iteration_rejects_dataset_and_checkpoint_changes(
        tmp_path, identity_inputs):
    config = _expert_config(tmp_path, identity_inputs)
    root = Path(config.output_dir)
    _initialize_pipeline_spec(
        root, _expert_iteration_experiment_spec(
            config, load_records(config.base_dataset)))
    changed = replace(
        config,
        base_dataset=str(identity_inputs["evaluation"]),
        initial_checkpoint=str(identity_inputs["protagonist_b"]))
    with pytest.raises(ExperimentIdentityError) as caught:
        ExpertIteration(changed).run()
    assert "inputs.base_dataset.sha256" in str(caught.value)
    assert "inputs.initial_protagonist.sha256" in str(caught.value)


def test_spec_and_run_state_tampering_are_integrity_failures(
        tmp_path, identity_inputs):
    config = _expert_config(tmp_path, identity_inputs)
    root = Path(config.output_dir)
    spec = _expert_iteration_experiment_spec(
        config, load_records(config.base_dataset))
    fingerprint = _initialize_pipeline_spec(root, spec)

    state = json.loads((root / "run_state.json").read_text(encoding="utf-8"))
    state["experiment_fingerprint"] = "0" * 64
    atomic_write_json(root / "run_state.json", state)
    with pytest.raises(ExperimentSpecIntegrityError, match="run-state"):
        validate_or_initialize_experiment(root, spec, run_state=state)

    state["experiment_fingerprint"] = fingerprint
    atomic_write_json(root / "run_state.json", state)
    document_path = root / EXPERIMENT_SPEC_FILE
    document = json.loads(document_path.read_text(encoding="utf-8"))
    document["spec"]["semantic_config"]["seed"] += 1
    atomic_write_json(document_path, document)
    with pytest.raises(ExperimentSpecIntegrityError, match="integrity"):
        load_persisted_experiment_spec(root)


def test_replay_is_not_loaded_under_a_different_state_fingerprint(
        tmp_path, identity_inputs, monkeypatch):
    config = _co_config(tmp_path, identity_inputs)
    root = Path(config.output_dir)
    spec = _cotraining_experiment_spec(
        config, load_records(config.base_dataset))
    _initialize_pipeline_spec(root, spec)
    state = json.loads((root / "run_state.json").read_text(encoding="utf-8"))
    state["experiment_fingerprint"] = "f" * 64
    atomic_write_json(root / "run_state.json", state)
    loaded = {"value": False}

    def forbidden_load(*_args, **_kwargs):
        loaded["value"] = True
        raise AssertionError("replay load must follow identity validation")

    monkeypatch.setattr(
        "blocksort.cotraining.loop.ReplayBuffer.load", forbidden_load)
    with pytest.raises(ExperimentSpecIntegrityError, match="run-state"):
        CoTraining(config).run()
    assert not loaded["value"]


@pytest.mark.parametrize("pipeline", ["cotraining", "expert_iteration"])
def test_lower_continuation_horizon_rejected_before_mutation(
        tmp_path, identity_inputs, pipeline):
    if pipeline == "cotraining":
        config = _co_config(tmp_path, identity_inputs, rounds=5)
        root = Path(config.output_dir)
        spec = _cotraining_experiment_spec(
            config, load_records(config.base_dataset))
        fingerprint, _ = validate_or_initialize_experiment(
            root, spec, run_state=None)
        state = {
            "experiment_fingerprint": fingerprint,
            "completed_rounds": [1, 2, 3, 4, 5],
            "active_protagonist_source_round": 5,
            "evaluation_split": spec["derived"]["evaluation_split"],
        }
        runner = CoTraining(replace(config, rounds=2))
        expected = "requested rounds 2"
    else:
        config = _expert_config(tmp_path, identity_inputs, iterations=5)
        root = Path(config.output_dir)
        spec = _expert_iteration_experiment_spec(
            config, load_records(config.base_dataset))
        fingerprint, _ = validate_or_initialize_experiment(
            root, spec, run_state=None)
        state = {
            "experiment_fingerprint": fingerprint,
            "completed_iterations": [1, 2, 3, 4, 5],
            "active_protagonist_source_iteration": 5,
        }
        runner = ExpertIteration(replace(config, iterations=2))
        expected = "requested iterations 2"
    atomic_write_json(root / "run_state.json", state)
    config_path = root / "config.json"
    config_path.write_bytes(b"truthful-config-sentinel")
    before = {
        path: (path.read_bytes(), path.stat().st_mtime_ns)
        for path in (root / "run_state.json", config_path)}

    with pytest.raises(ContinuationHorizonError, match=expected):
        runner.run()
    assert before == {
        path: (path.read_bytes(), path.stat().st_mtime_ns)
        for path in before}


def test_resolved_device_is_strict_identity_and_requested_form_is_diagnostic(
        tmp_path, identity_inputs):
    config = _expert_config(tmp_path, identity_inputs)
    records = load_records(config.base_dataset)
    cpu_auto = _expert_iteration_experiment_spec(
        replace(config, device="auto"), records, resolved_device="cpu")
    cpu_explicit = _expert_iteration_experiment_spec(
        replace(config, device="cpu"), records, resolved_device="cpu")
    cuda = _expert_iteration_experiment_spec(
        replace(config, device="cuda"), records, resolved_device="cuda")
    cuda_auto = _expert_iteration_experiment_spec(
        replace(config, device="auto"), records, resolved_device="cuda")

    assert compare_experiment_specs(cpu_auto, cpu_explicit) == []
    assert compare_experiment_specs(cuda_auto, cuda) == []
    differences = compare_experiment_specs(cpu_auto, cuda)
    fields = {item.field for item in differences}
    assert "software_semantics.runtime.resolved_device" in fields
    assert fields <= {
        "software_semantics.runtime.resolved_device",
        "software_semantics.runtime.cuda_version",
        "software_semantics.runtime.cudnn_version",
    }
    assert cpu_auto["software_semantics"]["runtime"] == {
        "requested_device": "auto",
        "resolved_device": "cpu",
        "torch_version": identity_mod.torch.__version__,
        "cuda_version": None,
        "cudnn_version": None,
        "deterministic_algorithms":
            identity_mod.torch.are_deterministic_algorithms_enabled(),
    }
    root = tmp_path / "device_run"
    fingerprint, _ = validate_or_initialize_experiment(
        root, cpu_auto, run_state=None)
    state = {"experiment_fingerprint": fingerprint}
    atomic_write_json(root / "run_state.json", state)
    sentinel = root / "best.pt"
    sentinel.write_bytes(b"do-not-touch")
    before = {
        path: (path.read_bytes(), path.stat().st_mtime_ns)
        for path in (root / "run_state.json", sentinel)}
    with pytest.raises(
            ExperimentIdentityError,
            match="software_semantics.runtime.resolved_device"):
        validate_or_initialize_experiment(root, cuda, run_state=state)
    assert before == {
        path: (path.read_bytes(), path.stat().st_mtime_ns)
        for path in before}

    version_changed = copy.deepcopy(cpu_auto)
    version_changed["software_semantics"]["runtime"]["torch_version"] = "other"
    deterministic_changed = copy.deepcopy(cpu_auto)
    deterministic_changed["software_semantics"]["runtime"][
        "deterministic_algorithms"] = not cpu_auto["software_semantics"][
            "runtime"]["deterministic_algorithms"]
    assert [item.field for item in compare_experiment_specs(
        cpu_auto, version_changed)] == [
            "software_semantics.runtime.torch_version"]
    assert [item.field for item in compare_experiment_specs(
        cpu_auto, deterministic_changed)] == [
            "software_semantics.runtime.deterministic_algorithms"]
    for changed_spec, field in (
            (version_changed, "runtime.torch_version"),
            (deterministic_changed, "runtime.deterministic_algorithms")):
        with pytest.raises(ExperimentIdentityError, match=field):
            validate_or_initialize_experiment(
                root, changed_spec, run_state=state)
    assert before == {
        path: (path.read_bytes(), path.stat().st_mtime_ns)
        for path in before}


def test_field_classification_rejects_gaps_and_overlaps():
    with pytest.raises(ValueError, match="unclassified=.*new_field"):
        validate_field_classification(
            {"seed", "new_field"}, {"semantic": {"seed"}})
    with pytest.raises(ValueError, match="multiple_categories"):
        validate_field_classification(
            {"seed"}, {
                "semantic": {"seed"},
                "operational": {"seed"},
            })


def test_legacy_migration_is_explicit_and_conservative(
        tmp_path, identity_inputs):
    config = _expert_config(tmp_path, identity_inputs)
    spec = _expert_iteration_experiment_spec(
        config, load_records(config.base_dataset))

    safe = tmp_path / "safe_legacy"
    safe.mkdir()
    atomic_write_json(safe / "config.json", config.to_dict())
    fingerprint, migrated = validate_or_initialize_experiment(
        safe, spec, run_state=None, legacy_spec=spec)
    assert migrated
    assert load_persisted_experiment_spec(safe)[1] == fingerprint

    ambiguous = tmp_path / "ambiguous_legacy"
    ambiguous.mkdir()
    (ambiguous / "best.pt").write_bytes(b"unknown provenance")
    with pytest.raises(LegacyRunMigrationError, match="cannot be safely migrated"):
        validate_or_initialize_experiment(
            ambiguous, spec, run_state=None)


def test_legacy_pipeline_rejects_replaced_external_inputs_without_writes(
        tmp_path, identity_inputs):
    config = _co_config(tmp_path, identity_inputs, rounds=0)
    root = Path(config.output_dir)
    root.mkdir()
    atomic_write_json(root / "config.json", config.to_dict())
    atomic_write_json(root / "run_state.json", {"completed_rounds": []})
    before = {
        path: (path.read_bytes(), path.stat().st_mtime_ns)
        for path in (root / "config.json", root / "run_state.json")}

    with identity_inputs["dataset"].open("a", encoding="utf-8") as handle:
        handle.write("\n")
    with identity_inputs["evaluation"].open("a", encoding="utf-8") as handle:
        handle.write("\n")
    changed_cli = replace(
        config,
        protagonist_checkpoint=str(identity_inputs["protagonist_b"]))
    with pytest.raises(ExperimentSpecIntegrityError) as caught:
        CoTraining(changed_cli).run()
    message = str(caught.value)
    assert "no persisted fixed evaluation split" in message
    assert before == {
        path: (path.read_bytes(), path.stat().st_mtime_ns)
        for path in before}
    assert not (root / EXPERIMENT_SPEC_FILE).exists()


def test_verified_legacy_manifest_can_use_committed_internal_checkpoint(
        tmp_path, identity_inputs):
    config = _expert_config(tmp_path, identity_inputs, iterations=0)
    root = Path(config.output_dir)
    internal = root / "protagonist" / "initial.pt"
    internal.parent.mkdir(parents=True)
    shutil.copyfile(identity_inputs["protagonist_a"], internal)
    historical_spec = _expert_iteration_experiment_spec(
        config, load_records(config.base_dataset),
        initial_identity_path=str(internal), resolved_device="cpu")
    requested = _expert_iteration_experiment_spec(
        replace(config, device="cpu"), load_records(config.base_dataset),
        resolved_device="cpu")
    assert compare_experiment_specs(historical_spec, requested) == []
    atomic_write_json(root / "config.json", config.to_dict())
    persist_legacy_migration_manifest(root, historical_spec)

    result = ExpertIteration(replace(config, device="cpu")).run()
    persisted, fingerprint = load_persisted_experiment_spec(root)
    assert persisted == historical_spec
    assert result["run_state"]["experiment_fingerprint"] == fingerprint


def test_legacy_migration_manifest_is_verified_against_current_invocation(
        tmp_path, identity_inputs):
    config = _expert_config(tmp_path, identity_inputs, iterations=0)
    historical = _expert_iteration_experiment_spec(
        config, load_records(config.base_dataset), resolved_device="cpu")
    root = Path(config.output_dir)
    root.mkdir()
    atomic_write_json(root / "config.json", config.to_dict())
    persist_legacy_migration_manifest(root, historical)

    with pytest.raises(ExperimentIdentityError, match="initial_protagonist.sha256"):
        ExpertIteration(replace(
            config, initial_checkpoint=str(identity_inputs["protagonist_b"]),
            device="cpu")).run()
    assert not (root / EXPERIMENT_SPEC_FILE).exists()


def test_legacy_state_migration_recovers_if_spec_publish_crashes(
        tmp_path, identity_inputs, monkeypatch):
    config = _expert_config(tmp_path, identity_inputs)
    spec = _expert_iteration_experiment_spec(
        config, load_records(config.base_dataset))
    root = tmp_path / "legacy_crash"
    root.mkdir()
    atomic_write_json(root / "config.json", config.to_dict())
    original_state = {"completed_iterations": []}
    atomic_write_json(root / "run_state.json", original_state)

    real_persist = identity_mod.persist_experiment_spec

    def crash_before_spec(*_args, **_kwargs):
        raise RuntimeError("injected spec publish crash")

    monkeypatch.setattr(
        identity_mod, "persist_experiment_spec", crash_before_spec)
    with pytest.raises(RuntimeError, match="injected"):
        validate_or_initialize_experiment(
            root, spec, run_state=original_state, legacy_spec=spec)
    migrated_state = json.loads(
        (root / "run_state.json").read_text(encoding="utf-8"))
    assert migrated_state["experiment_fingerprint"] == \
        fingerprint_experiment_spec(spec)
    assert not (root / EXPERIMENT_SPEC_FILE).exists()

    monkeypatch.setattr(identity_mod, "persist_experiment_spec", real_persist)
    fingerprint, migrated = validate_or_initialize_experiment(
        root, spec, run_state=migrated_state, legacy_spec=spec)
    assert migrated
    assert fingerprint == migrated_state["experiment_fingerprint"]


def test_final_benchmark_records_source_fingerprint_and_rejects_changed_model(
        tmp_path, identity_inputs):
    source_root = tmp_path / "source_run"
    source_spec = build_experiment_spec(
        pipeline="expert_iteration", semantic_config={"seed": 9}, inputs={},
        software_semantics={"fixture_version": 1})
    source_fingerprint, _ = validate_or_initialize_experiment(
        source_root, source_spec, run_state=None)
    source_checkpoint = _protagonist(source_root / "best.pt", 9)
    atomic_write_json(source_root / "run_state.json", {
        "experiment_fingerprint": source_fingerprint,
        "active_protagonist_checkpoint": "best.pt",
        "active_protagonist_sha256": sha256_file(source_checkpoint),
        "active_protagonist_source_iteration": 3,
    })

    output = tmp_path / "final"
    args = build_final_parser().parse_args([
        "--output-dir", str(output),
        "--phases", "report",
        "--supervised", str(source_checkpoint),
        "--expert-iteration", str(source_checkpoint),
        "--adversarial-designer", str(identity_inputs["designer_a"]),
        "--pretrained-designer", str(identity_inputs["designer_b"]),
        "--base-dataset", str(identity_inputs["dataset"]),
        "--handcrafted-dataset", str(identity_inputs["evaluation"]),
        "--prior-cotraining-dir", str(tmp_path / "missing_prior"),
    ])
    original = _final_benchmark_spec(args)
    assert original["inputs"]["expert_iteration"][
        "source_experiment_fingerprint"] == source_fingerprint
    validate_or_initialize_experiment(output, original, run_state=None)

    state_path = source_root / "run_state.json"
    saved_state = state_path.read_bytes()
    state_path.unlink()
    with pytest.raises(
            ExperimentSpecIntegrityError,
            match="missing authoritative state"):
        _final_benchmark_spec(args)
    state_path.write_bytes(saved_state)

    shutil.copyfile(identity_inputs["protagonist_b"], source_checkpoint)
    promoted_state = json.loads(saved_state)
    promoted_state["active_protagonist_sha256"] = sha256_file(source_checkpoint)
    promoted_state["active_protagonist_source_iteration"] = 4
    atomic_write_json(state_path, promoted_state)
    changed = _final_benchmark_spec(args)
    with pytest.raises(
            ExperimentIdentityError,
            match="inputs.expert_iteration.sha256"):
        validate_or_initialize_experiment(output, changed, run_state=None)


def test_checkpoint_identity_requires_authoritative_identified_source_state(
        tmp_path, identity_inputs):
    root = tmp_path / "identified_co"
    spec = build_experiment_spec(
        pipeline="cotraining", semantic_config={}, inputs={},
        software_semantics={"fixture_version": 1})
    fingerprint, _ = validate_or_initialize_experiment(
        root, spec, run_state=None)
    active = _protagonist(root / "protagonist" / "round_001.pt", 31)
    best = root / "best.pt"
    shutil.copyfile(active, best)

    with pytest.raises(
            ExperimentSpecIntegrityError,
            match="missing authoritative state"):
        checkpoint_identity(str(best))
    atomic_write_json(root / "run_state.json", {
        "experiment_fingerprint": fingerprint,
        "active_protagonist_checkpoint": "protagonist/round_001.pt",
        "active_protagonist_sha256": sha256_file(active),
        "active_protagonist_source_round": 1,
    })
    accepted = checkpoint_identity(str(best))
    assert accepted["source_kind"] == "identified_run"
    assert accepted["committed_role"] == "active"
    assert accepted["committed_progress"] == 1

    best.write_bytes(best.read_bytes() + b"stale")
    with pytest.raises(
            ExperimentSpecIntegrityError,
            match="not the authoritative active checkpoint"):
        checkpoint_identity(str(best))


def test_checkpoint_identity_validates_supervised_roles_and_standalone(
        tmp_path):
    root = tmp_path / "supervised"
    spec = build_experiment_spec(
        pipeline="supervised_protagonist", semantic_config={}, inputs={},
        software_semantics={"fixture_version": 1})
    fingerprint, _ = validate_or_initialize_experiment(
        root, spec, run_state=None)
    active = _protagonist(root / "checkpoints" / "epoch_002.pt", 41)
    best_committed = _protagonist(
        root / "checkpoints" / "epoch_001.pt", 42)
    last = root / "last.pt"
    best = root / "best.pt"
    shutil.copyfile(active, last)
    shutil.copyfile(best_committed, best)
    atomic_write_json(root / "run_state.json", {
        "schema_version": 1,
        "experiment_fingerprint": fingerprint,
        "completed_epochs": 2,
        "active_checkpoint": "checkpoints/epoch_002.pt",
        "active_checkpoint_sha256": sha256_file(active),
        "best_checkpoint": "checkpoints/epoch_001.pt",
        "best_checkpoint_sha256": sha256_file(best_committed),
    })
    assert checkpoint_identity(str(last))["committed_role"] == "active"
    assert checkpoint_identity(str(best))["committed_role"] == "best"
    last.write_bytes(best.read_bytes())
    with pytest.raises(
            ExperimentSpecIntegrityError,
            match="last.pt does not match"):
        checkpoint_identity(str(last))

    standalone = _protagonist(tmp_path / "standalone.pt", 43)
    standalone_identity = checkpoint_identity(str(standalone))
    assert standalone_identity["source_kind"] == "standalone_checkpoint"
    assert standalone_identity["source_experiment_fingerprint"] is None


def test_checkpoint_identity_rejects_unproven_identified_designer(
        tmp_path):
    root = tmp_path / "designer"
    spec = build_experiment_spec(
        pipeline="designer_training", semantic_config={}, inputs={},
        software_semantics={"fixture_version": 1})
    validate_or_initialize_experiment(root, spec, run_state=None)
    checkpoint = _designer(root / "best.pt", 51)
    with pytest.raises(
            ExperimentSpecIntegrityError,
            match="provenance"):
        checkpoint_identity(str(checkpoint))

    proven_root = tmp_path / "proven_designer"
    proven_fingerprint, _ = validate_or_initialize_experiment(
        proven_root, spec, run_state=None)
    encoding = EncodingConfig()
    model_config = DesignerModelConfig(
        channels=4, residual_blocks=1, hidden_size=8)
    proven = proven_root / "best.pt"
    save_designer(
        proven, model=DesignerNet(encoding, model_config),
        encoding_config=encoding, model_config=model_config, seed=52,
        metadata={"experiment_fingerprint": proven_fingerprint})
    atomic_write_json(proven_root / "summary.json", {
        "best_checkpoint": str(proven),
        "best_checkpoint_sha256": sha256_file(proven),
        "last_checkpoint": str(proven),
        "last_checkpoint_sha256": sha256_file(proven),
        "encoding_fingerprint": identity_mod.hash_canonical_value({
            "encoding_config": encoding.to_dict(),
            "model_config": model_config.to_dict(),
        }),
    })
    assert checkpoint_identity(str(proven))["committed_role"] == "best"
