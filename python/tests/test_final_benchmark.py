"""Focused correctness tests for final benchmark/report contracts."""

from __future__ import annotations

from types import SimpleNamespace
import json
import shutil

import pytest
import torch

from blocksort import level_from_dict
from blocksort.final_benchmark import (
    ablations, common, harder, report, run as final_run)
from blocksort.cotraining.config import CoTrainingConfig
from blocksort.designer.checkpoint import save_designer
from blocksort.designer.config import GeneratorConfig
from blocksort.designer.model import DesignerModelConfig, DesignerNet
from blocksort.training.checkpoint import save_checkpoint
from blocksort.training.config import (
    EncodingConfig, ModelConfig, ValueNormConfig)
from blocksort.training.experiment_identity import (
    ExperimentSpecIntegrityError, build_experiment_spec,
    validate_or_initialize_experiment)
from blocksort.training.model import PolicyValueNet
from blocksort.training.transaction import atomic_write_json, sha256_file


def _identified_protagonist_run(
    root, *, pipeline="cotraining", seed=11,
):
    spec = build_experiment_spec(
        pipeline=pipeline, semantic_config={"seed": seed}, inputs={},
        software_semantics={"fixture_version": 1})
    fingerprint, _ = validate_or_initialize_experiment(
        root, spec, run_state=None)
    active = root / "protagonist" / "committed.pt"
    torch.manual_seed(seed)
    encoding = EncodingConfig()
    model_config = ModelConfig(
        channels=4, residual_blocks=1, value_hidden_size=8)
    save_checkpoint(
        active, model=PolicyValueNet(encoding, model_config), optimizer=None,
        scheduler=None, epoch=0, best_val_metric=None,
        encoding_config=encoding, model_config=model_config,
        value_norm=ValueNormConfig(), seed=seed, dataset_version=1,
        split_identity=None)
    mirror = root / "best.pt"
    shutil.copyfile(active, mirror)
    progress_field = (
        "active_protagonist_source_round"
        if pipeline == "cotraining"
        else "active_protagonist_source_iteration")
    atomic_write_json(root / "run_state.json", {
        "experiment_fingerprint": fingerprint,
        "active_protagonist_checkpoint": "protagonist/committed.pt",
        "active_protagonist_sha256": sha256_file(active),
        progress_field: 1,
    })
    return mirror, active


def _solver_fixture():
    search = {}
    scores = {
        "A": {1: 0.60, 8: 0.80, 32: 0.75},
        "B": {1: 0.70, 8: 0.72, 32: 0.78},
    }
    for model, budgets in scores.items():
        for budget, score in budgets.items():
            search[f"{model}@{budget}"] = {
                "optimal_acc": score,
                "confirmed_optimal_rate": score,
                "mean_regret": 0.0,
                "solve_rate": score,
                "solution_length_gap_common": 0.0,
                "solution_length_gap_each": 0.0,
                "runtime_ms_mean": 1.0,
                "nodes_expanded_mean": 1.0,
                "transposition_hits_mean": 0.0,
            }
    return {
        "states": 10, "common_solved_count": 5, "search": search,
        "raw_policy": {}, "astar": {},
    }


def test_solver_conclusion_uses_one_common_budget():
    by_model, _conclusion = report._solver_conclusion(_solver_fixture())
    assert by_model == {"A": 0.75, "B": 0.78}


def test_best_observed_is_diagnostic_and_delta_uses_common_budget():
    comparison = report.solver_comparison(_solver_fixture())
    assert comparison["comparison_budget"] == 32
    assert comparison["models"]["A"] == {
        "common_budget_score": 0.75,
        "available_budgets": [1, 8, 32],
        "best_observed_score": 0.80,
        "best_observed_budget": 8,
    }
    assert comparison["common_budget_deltas"]["B-A"] == pytest.approx(0.03)


def test_missing_common_budget_fails_without_substitution():
    fixture = _solver_fixture()
    del fixture["search"]["B@32"]
    with pytest.raises(
            ValueError, match=r"budget 32.*model 'B'.*\[1, 8\]"):
        report.solver_comparison(fixture, comparison_budget=32)


def test_budget_selection_is_numeric_not_lexicographic():
    fixture = _solver_fixture()
    fixture["budgets"] = ["8", 1, "32"]
    fixture["search"] = {
        key: fixture["search"][key]
        for key in ("A@8", "B@32", "A@32", "B@8", "A@1", "B@1")
    }
    assert report.solver_comparison(fixture)["comparison_budget"] == 32


def test_nonnumeric_budget_key_fails_clearly():
    fixture = _solver_fixture()
    fixture["search"]["A@large"] = fixture["search"].pop("A@32")
    with pytest.raises(ValueError, match="nonnumeric budget"):
        report.solver_comparison(fixture)


def test_pretrained_designer_uses_its_own_encoding(
        tmp_path, monkeypatch):
    enc_a, enc_b = object(), object()
    bundles = iter([
        SimpleNamespace(
            model="model-a", encoding=enc_a,
            checkpoint_path="a.pt", checkpoint_sha256="a",
            provenance=lambda: {"checkpoint_sha256": "a"}),
        SimpleNamespace(
            model="model-b", encoding=enc_b,
            checkpoint_path="b.pt", checkpoint_sha256="b",
            provenance=lambda: {"checkpoint_sha256": "b"}),
    ])
    seen = []
    monkeypatch.setattr(harder, "resolve_device", lambda name: "cpu")
    monkeypatch.setattr(
        harder, "Protagonist",
        lambda *args, **kwargs: SimpleNamespace(
            model=object(), enc=object(), value_norm=object()))
    monkeypatch.setattr(
        harder, "load_designer_bundle", lambda *args: next(bundles))
    monkeypatch.setattr(
        harder, "designer_generation_identity",
        lambda bundle, *args, **kwargs: bundle.provenance()["checkpoint_sha256"])
    monkeypatch.setattr(harder, "_handcrafted", lambda *args: [])
    monkeypatch.setattr(harder, "_random_group", lambda *args, **kwargs: [])
    monkeypatch.setattr(harder, "_prior_cotraining", lambda *args: [])
    monkeypatch.setattr(
        harder, "designer_levels",
        lambda env, bundle, *args, **kwargs:
            seen.append(bundle.encoding) or [])
    monkeypatch.setattr(
        harder, "BoundedProtagonist",
        lambda *args, **kwargs: SimpleNamespace(
            solve=lambda *a, **kw: SimpleNamespace(solved=False)))
    harder.build_harder_benchmark(
        str(tmp_path), protagonist_checkpoint="p.pt",
        adversarial_designer_checkpoint="a.pt",
        pretrained_designer_checkpoint="b.pt",
        handcrafted_dataset="data.jsonl", prior_cotraining_dir=None,
        count=1)
    assert seen == [enc_a, enc_b]


def test_equal_shape_different_encoding_schema_has_distinct_identity():
    encoding_a = EncodingConfig()
    encoding_b = EncodingConfig(colors=tuple(reversed(encoding_a.colors)))
    assert encoding_a.num_board_channels == encoding_b.num_board_channels
    assert common.designer_encoding_fingerprint(encoding_a) != \
        common.designer_encoding_fingerprint(encoding_b)


def test_loaded_designer_bundles_retain_checkpoint_specific_encodings(tmp_path):
    encoding_a = EncodingConfig()
    encoding_b = EncodingConfig(colors=tuple(reversed(encoding_a.colors)))
    model_config = DesignerModelConfig(
        channels=4, residual_blocks=1, hidden_size=8)
    paths = [tmp_path / "a.pt", tmp_path / "b.pt"]
    for path, encoding in zip(paths, (encoding_a, encoding_b)):
        save_designer(
            path, model=DesignerNet(encoding, model_config),
            encoding_config=encoding, model_config=model_config, seed=1)
    bundle_a = common.load_designer_bundle(
        str(paths[0]), torch.device("cpu"))
    bundle_b = common.load_designer_bundle(
        str(paths[1]), torch.device("cpu"))
    assert bundle_a.encoding.colors == encoding_a.colors
    assert bundle_b.encoding.colors == encoding_b.colors
    assert bundle_a.encoding_fingerprint != bundle_b.encoding_fingerprint


def test_benchmark_repeated_trial_seeds_are_level_order_independent(
    monkeypatch,
):
    def make_level(name, col):
        return level_from_dict({
            "name": name,
            "cols": 4,
            "rows": 4,
            "blocks": [{"color": "red", "cells": [[1, col]]}],
            "exits": [
                {"edge": "left", "start": 1, "length": 1, "color": "red"}
            ],
        })

    level_a = make_level("a", 1)
    level_b = make_level("b", 2)
    calls = []

    class Bounded:
        def solve(self, level, *, seed=0):
            cell = next(iter(level.blocks[0].cells))
            calls.append(((cell.r, cell.c), seed))
            return SimpleNamespace(solved=(seed % 2 == 0))

    monkeypatch.setattr(harder, "resolve_device", lambda _: "cpu")
    monkeypatch.setattr(
        harder, "Protagonist",
        lambda *args, **kwargs: SimpleNamespace(
            model=object(), enc=object(), value_norm=object()))
    monkeypatch.setattr(harder, "BoundedProtagonist",
                        lambda *args, **kwargs: Bounded())

    def run(groups):
        calls.clear()
        monkeypatch.setattr(harder, "load_benchmark_groups", lambda _: groups)
        harder.saturation_check(
            "unused", protagonist_checkpoint="unused.pt",
            trials=4, per_group=5, seed=31)
        by_level = {}
        for identity, trial_seed in calls:
            by_level.setdefault(identity, []).append(trial_seed)
        return by_level

    forward = run({"g1": [level_a], "g2": [level_b]})
    reverse = run({"g2": [level_b], "g1": [level_a]})
    reordered = run({"g1": [level_b, level_a]})
    assert forward == reverse
    assert reordered[(1, 1)] == forward[(1, 1)]
    assert reordered[(1, 2)] == forward[(1, 2)]
    assert len(set(forward[(1, 1)])) == 4
    assert forward[(1, 1)] != forward[(1, 2)]


def test_generation_identity_includes_encoding_fingerprint():
    generator = GeneratorConfig(rows=5, cols=5, color_count=3, density=0.5)
    base = dict(
        checkpoint_sha256="same-checkpoint",
        checkpoint_path="designer.pt")
    bundle_a = SimpleNamespace(**base, encoding_fingerprint="encoding-a")
    bundle_b = SimpleNamespace(**base, encoding_fingerprint="encoding-b")
    identity_a = common.designer_generation_identity(
        bundle_a, generator, mutation_budget=10, count=4, seed=7)
    identity_b = common.designer_generation_identity(
        bundle_b, generator, mutation_budget=10, count=4, seed=7)
    assert identity_a != identity_b


def test_skipped_forgetting_does_not_crash_ablation(monkeypatch, tmp_path):
    monkeypatch.setattr(ablations, "run_cotraining", lambda cfg: {
        "run_state": {"history": [{
            "accepted": 0, "frontier_acceptance_rate": 0.0,
            "mean_solve_rate": 0.0, "label_exact": 0, "label_search": 0,
            "promoted": False, "promotion_score_candidate": 0.0,
            "curriculum_adjustment": {"direction": "disabled"},
            "forgetting": {"skipped": True, "reason": "disabled"},
        }]}
    })
    result = ablations._cotrain_variant(
        "x", root=tmp_path, base_cfg=CoTrainingConfig())
    assert result["per_round"][0]["forgetting"]["status"] == "skipped"


def test_forgetting_statuses_keep_zero_skipped_and_unavailable_distinct():
    normalized = ablations.normalize_forgetting({
        "zero": {"delta": 0.0, "baseline": 1.0, "candidate": 1.0},
        "skip": {"status": "skipped", "delta": None, "reason": "disabled"},
        "missing": {"delta": None, "baseline": None, "candidate": None},
    })
    assert normalized["groups"]["zero"]["status"] == "measured"
    assert normalized["groups"]["zero"]["delta"] == 0.0
    assert normalized["groups"]["skip"]["status"] == "skipped"
    assert normalized["groups"]["missing"]["status"] == "unavailable"


def test_malformed_forgetting_fails_clearly():
    with pytest.raises(ValueError, match="missing delta/status"):
        ablations.normalize_forgetting({"bad": {"baseline": 1.0}})
    with pytest.raises(ValueError, match="unknown status"):
        ablations.normalize_forgetting({
            "bad": {"status": "mystery", "delta": None}})


def test_benchmark_manifest_declares_group_provenance(
        tmp_path, monkeypatch):
    enc = object()
    bundle = SimpleNamespace(
        model="model", encoding=enc,
        checkpoint_path="a.pt", checkpoint_sha256="a",
        provenance=lambda: {"checkpoint_sha256": "a"})
    monkeypatch.setattr(harder, "resolve_device", lambda name: "cpu")
    monkeypatch.setattr(
        harder, "Protagonist",
        lambda *args, **kwargs: SimpleNamespace(
            model=object(), enc=object(), value_norm=object()))
    monkeypatch.setattr(
        harder, "load_designer_bundle", lambda *args: bundle)
    monkeypatch.setattr(
        harder, "designer_generation_identity",
        lambda *args, **kwargs: "identity")
    monkeypatch.setattr(harder, "_handcrafted", lambda *args: [])
    heldout = [
        level_from_dict({
            "name": f"heldout-{col}", "cols": 4 + col, "rows": 4 + col,
            "blocks": [{"color": "red", "cells": [[1, col]]}],
            "exits": [
                {"edge": "left", "start": 1, "length": 1, "color": "red"}
            ],
        })
        for col in (1, 2)
    ]
    monkeypatch.setattr(
        harder, "_random_group", lambda *args, **kwargs: heldout)
    monkeypatch.setattr(harder, "_prior_cotraining", lambda *args: [])
    monkeypatch.setattr(harder, "designer_levels", lambda *args, **kwargs: [])
    monkeypatch.setattr(
        harder, "BoundedProtagonist",
        lambda *args, **kwargs: SimpleNamespace(
            solve=lambda *a, **kw: SimpleNamespace(solved=False)))
    result = harder.build_harder_benchmark(
        str(tmp_path), protagonist_checkpoint="p.pt",
        adversarial_designer_checkpoint="a.pt",
        pretrained_designer_checkpoint=None,
        handcrafted_dataset="data.jsonl", prior_cotraining_dir=None,
        count=1)
    assert "group_provenance" in result["manifest"]
    assert result["manifest"]["schema_version"] == 2
    assert (tmp_path / "eval_split.json").is_file()
    split = json.loads((tmp_path / "eval_split.json").read_text())
    assert split["split_config"] == {
        "split_seed": 1729, "validation_count": 1, "test_count": 1}
    assert result["manifest"]["evaluation_split"][
        "evaluation_split_fingerprint"] == \
        split["evaluation_split_fingerprint"]


def _provenance_manifest(overlap=0, eligible=True):
    return {"group_provenance": {
        "handcrafted": {
            "evaluation_role": "retention",
            "source_kind": "training-derived",
            "held_out_eligible": False,
            "disjointness_verified": False,
            "overlap_count": 1,
        },
        "prior_cotraining": {
            "evaluation_role": "retention",
            "source_kind": "replay-derived",
            "held_out_eligible": False,
            "disjointness_verified": False,
            "overlap_count": 1,
        },
        "generated": {
            "evaluation_role": "held_out_final",
            "source_kind": "adversarially-generated",
            "held_out_eligible": eligible,
            "disjointness_verified": True,
            "overlap_count": overlap,
        },
    }}


def test_held_out_aggregate_excludes_training_and_replay_groups():
    saturation = {"groups": {
        "handcrafted": {"levels": 2, "bounded_solve_rate": 1.0},
        "prior_cotraining": {"levels": 2, "bounded_solve_rate": 0.0},
        "generated": {"levels": 2, "bounded_solve_rate": 0.5},
    }}
    summary = report.benchmark_provenance_summary(
        _provenance_manifest(), saturation)
    assert summary["aggregates"]["retention"]["members"] == [
        "handcrafted", "prior_cotraining"]
    assert summary["aggregates"]["held_out_final"]["members"] == ["generated"]
    assert summary["aggregates"]["held_out_final"]["score"] == 0.5


def test_held_out_overlap_is_rejected_and_empty_aggregate_is_unavailable():
    with pytest.raises(ValueError, match="without verified zero-overlap"):
        report.benchmark_provenance_summary(
            _provenance_manifest(overlap=1), {"groups": {}})
    summary = report.benchmark_provenance_summary(
        _provenance_manifest(eligible=False), {"groups": {}})
    held_out = summary["aggregates"]["held_out_final"]
    assert held_out["status"] == "unavailable"
    assert held_out["score"] is None


def test_machine_and_human_reports_label_controlled_budget_and_provenance(
        tmp_path):
    benchmark = tmp_path / "benchmark"
    benchmark.mkdir()
    manifest = _provenance_manifest()
    manifest["group_sizes"] = {
        "handcrafted": 2, "prior_cotraining": 2, "generated": 2}
    manifest["evaluation_split"] = {
        "evaluation_split_fingerprint": "a" * 64,
        "evaluation_split_manifest_sha256": "b" * 64,
        "evaluation_pool_sha256": "c" * 64,
        "split_algorithm": "stable_signature_sha256_rank_v1",
        "split_seed": 1729,
        "promotion_validation_count": 42,
        "final_test_count": 42,
        "validation_signature_hash": "d" * 64,
        "test_signature_hash": "e" * 64,
        "eval_limit": 24,
        "evaluation_semantics_version": 5,
    }
    (benchmark / "benchmark_manifest.json").write_text(
        json.dumps(manifest), encoding="utf-8")
    saturation = {"groups": {
        "handcrafted": {"levels": 2, "bounded_solve_rate": 1.0},
        "prior_cotraining": {"levels": 2, "bounded_solve_rate": 0.0},
        "generated": {"levels": 2, "bounded_solve_rate": 0.5},
    }}
    (tmp_path / "saturation.json").write_text(
        json.dumps(saturation), encoding="utf-8")
    (tmp_path / "solver.json").write_text(
        json.dumps(_solver_fixture()), encoding="utf-8")
    transactional = tmp_path / "cotraining"
    transactional.mkdir()
    state = transactional / "run_state.json"
    checkpoint = transactional / "best.pt"
    state.write_text('{"completed_rounds":[1]}', encoding="utf-8")
    checkpoint.write_bytes(b"authoritative-checkpoint")
    state_before = state.read_bytes()
    checkpoint_before = checkpoint.read_bytes()
    report.build_report(
        tmp_path, defaults={}, args_dict={
            "comparison_budget": 32, "output_dir": str(tmp_path),
            "rounds": 0, "levels_per_round": 0,
            "astar_max_nodes": 1, "seed": 1})
    summary = json.loads((tmp_path / "summary.json").read_text())
    markdown = (tmp_path / "report.md").read_text()
    assert summary["solver_controlled_comparison"]["comparison_budget"] == 32
    assert summary["benchmark_provenance"]["aggregates"][
        "held_out_final"]["members"] == ["generated"]
    assert "Controlled comparison" in markdown
    assert "Best observed diagnostics (not used for ranking/deltas)" in markdown
    assert "| handcrafted | 2 | retention |" in markdown
    assert "a" * 64 in markdown
    assert summary["benchmark_provenance"]["evaluation_split"][
        "final_test_count"] == 42
    assert state.read_bytes() == state_before
    assert checkpoint.read_bytes() == checkpoint_before


def test_legacy_solver_report_remains_readable_but_not_reinterpreted(tmp_path):
    legacy = _solver_fixture()
    for metrics in legacy["search"].values():
        metrics.pop("confirmed_optimal_rate")
    (tmp_path / "solver.json").write_text(
        json.dumps(legacy), encoding="utf-8")
    report.build_report(
        tmp_path, defaults={}, args_dict={
            "comparison_budget": 32, "output_dir": str(tmp_path),
            "rounds": 0, "levels_per_round": 0,
            "astar_max_nodes": 1, "seed": 1})
    summary = json.loads((tmp_path / "summary.json").read_text())
    assert summary["solver_controlled_comparison"] is None
    assert "legacy solver artifact" in summary["solver_conclusion"]


@pytest.mark.parametrize("pipeline", ["cotraining", "expert_iteration"])
def test_internal_benchmark_protagonist_uses_authoritative_source_contract(
        tmp_path, monkeypatch, pipeline):
    run_root = (
        tmp_path / "cotraining"
        if pipeline == "cotraining" else tmp_path / "internal_expert")
    mirror, _active = _identified_protagonist_run(
        run_root, pipeline=pipeline)
    calls = []
    real_validate = common.validate_checkpoint_source

    def tracking_validate(path):
        calls.append(str(path))
        return real_validate(path)

    monkeypatch.setattr(common, "validate_checkpoint_source", tracking_validate)
    loaded = common.Protagonist(str(mirror), torch.device("cpu"))

    assert calls == [str(mirror)]
    assert loaded.source.source_kind == "identified_run"
    assert loaded.source.pipeline == pipeline
    assert loaded.source.committed_role == "active"
    assert loaded.source.checkpoint_sha256 == sha256_file(mirror)


def test_internal_benchmark_cotraining_source_missing_state_stale_and_hash_fail(
        tmp_path):
    run_root = tmp_path / "cotraining"
    mirror, active = _identified_protagonist_run(run_root)
    state_path = run_root / "run_state.json"
    state_bytes = state_path.read_bytes()

    state_path.unlink()
    with pytest.raises(
            ExperimentSpecIntegrityError,
            match="internally produced benchmark source.*missing authoritative"):
        final_run._cotrained_sources(tmp_path)
    state_path.write_bytes(state_bytes)

    mirror.write_bytes(mirror.read_bytes() + b"stale")
    with pytest.raises(
            ExperimentSpecIntegrityError,
            match="not the authoritative active checkpoint"):
        final_run._cotrained_sources(tmp_path)
    shutil.copyfile(active, mirror)

    active.write_bytes(active.read_bytes() + b"corrupt")
    with pytest.raises(
            ExperimentSpecIntegrityError,
            match="active checkpoint integrity failure"):
        final_run._cotrained_sources(tmp_path)


def test_internal_source_generation_is_pinned_and_failure_does_not_mutate(
        tmp_path):
    run_root = tmp_path / "cotraining"
    mirror, active = _identified_protagonist_run(run_root)
    source, designer = final_run._cotrained_sources(tmp_path)
    assert source is not None and designer is None
    pinned = final_run._pin_internal_sources(tmp_path, {
        "cotrained_protagonist": (
            source, "internal_cotraining_phase"),
    })
    sentinel = tmp_path / "solver.json"
    sentinel.write_text('{"preserve": true}', encoding="utf-8")
    before = {
        "pin": (tmp_path / final_run._INTERNAL_SOURCES_FILE).read_bytes(),
        "sentinel": sentinel.read_bytes(),
    }

    # Advance the same identified source directory to a new committed
    # generation. The old benchmark must remain pinned to the original one.
    replacement = run_root / "protagonist" / "replacement.pt"
    torch.manual_seed(99)
    replacement_encoding = EncodingConfig()
    replacement_model_config = ModelConfig(
        channels=4, residual_blocks=1, value_hidden_size=8)
    save_checkpoint(
        replacement,
        model=PolicyValueNet(
            replacement_encoding, replacement_model_config),
        optimizer=None, scheduler=None, epoch=0, best_val_metric=None,
        encoding_config=replacement_encoding,
        model_config=replacement_model_config,
        value_norm=ValueNormConfig(), seed=99, dataset_version=1,
        split_identity=None)
    state = json.loads((run_root / "run_state.json").read_text())
    state["active_protagonist_checkpoint"] = "protagonist/replacement.pt"
    state["active_protagonist_sha256"] = sha256_file(replacement)
    state["active_protagonist_source_round"] = 2
    atomic_write_json(run_root / "run_state.json", state)
    shutil.copyfile(replacement, mirror)

    changed = common.validate_checkpoint_source(mirror)
    assert changed.checkpoint_sha256 != pinned[
        "cotrained_protagonist"]["checkpoint_sha256"]
    with pytest.raises(
            ExperimentSpecIntegrityError,
            match="internally produced source.*changed"):
        final_run._validate_pinned_internal_sources(tmp_path)
    assert (tmp_path / final_run._INTERNAL_SOURCES_FILE).read_bytes() == \
        before["pin"]
    assert sentinel.read_bytes() == before["sentinel"]


def test_standalone_and_identified_source_records_have_equivalent_schema(
        tmp_path):
    mirror, _active = _identified_protagonist_run(tmp_path / "identified")
    standalone = tmp_path / "standalone.pt"
    shutil.copyfile(mirror, standalone)

    identified = common.validate_checkpoint_source(mirror).to_dict()
    independent = common.validate_checkpoint_source(standalone).to_dict()

    assert set(identified) == set(independent)
    assert identified["source_kind"] == "identified_run"
    assert identified["source_pipeline"] == "cotraining"
    assert identified["authority_sha256"]
    assert independent["source_kind"] == "standalone_checkpoint"
    assert independent["experiment_fingerprint"] is None
    assert independent["committed_role"] is None


def test_invalid_identified_source_fails_before_benchmark_directory_creation(
        tmp_path):
    source_root = tmp_path / "source"
    spec = build_experiment_spec(
        pipeline="expert_iteration", semantic_config={}, inputs={},
        software_semantics={"fixture_version": 1})
    validate_or_initialize_experiment(source_root, spec, run_state=None)
    checkpoint = source_root / "best.pt"
    checkpoint.write_bytes(b"not-authoritative")
    output = tmp_path / "benchmark"

    with pytest.raises(
            ExperimentSpecIntegrityError,
            match="missing authoritative state"):
        harder.build_harder_benchmark(
            str(output),
            protagonist_checkpoint=str(checkpoint),
            adversarial_designer_checkpoint=str(checkpoint),
            pretrained_designer_checkpoint=None,
            handcrafted_dataset=str(tmp_path / "unused.jsonl"),
            prior_cotraining_dir=None,
            count=1,
        )
    assert not output.exists()
