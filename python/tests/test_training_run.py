"""Integration tests for the end-to-end training loop (run_training)."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from blocksort.training.checkpoint import load_checkpoint
from blocksort.training.experiment_identity import (
    ContinuationHorizonError, ExperimentSpecIntegrityError,
    UnsupportedResumeError)
from blocksort.training.train import build_parser, run_training
from blocksort.training import train as train_mod

REPO_ROOT = Path(__file__).resolve().parents[2]
DATASET = REPO_ROOT / "data" / "training" / "pv_examples.jsonl"


def _assert_nested_equal(left, right):
    if torch.is_tensor(left):
        assert torch.equal(left, right)
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            _assert_nested_equal(left[key], right[key])
    elif isinstance(left, (list, tuple)):
        assert len(left) == len(right)
        for a, b in zip(left, right):
            _assert_nested_equal(a, b)
    else:
        assert left == right


def _args(output_dir, *, epochs=2, resume=None):
    argv = [
        "--dataset", str(DATASET),
        "--output-dir", str(output_dir),
        "--seed", "0", "--epochs", str(epochs), "--batch-size", "64",
        "--channels", "8", "--residual-blocks", "1", "--value-hidden-size", "16",
        "--device", "cpu",
        # 4 handcrafted levels -> 2 train / 1 val / 1 test (val non-empty).
        "--train-ratio", "0.5", "--val-ratio", "0.25", "--test-ratio", "0.25",
    ]
    if resume:
        argv += ["--resume", str(resume)]
    return build_parser().parse_args(argv)


def test_run_training_produces_checkpoints_and_manifest(tmp_path):
    out = tmp_path / "run"
    summary = run_training(_args(out, epochs=2))
    assert (out / "best.pt").exists()
    assert (out / "last.pt").exists()
    assert (out / "splits.json").exists()
    assert (out / "history.json").exists()
    assert summary["best_epoch"] >= 1
    assert len(summary["history"]) == 2
    # validation split is non-empty so selection used validation data.
    assert all(row["val_loss"] is not None for row in summary["history"])


@pytest.mark.skipif(not torch.cuda.is_available(), reason="no CUDA")
def test_rng_restore_accepts_checkpoint_tensors_mapped_to_cuda():
    records = train_mod.load_records(DATASET)[:2]
    manifest = train_mod.make_split(
        train_mod.collect_level_keys(records),
        ratios=train_mod.SplitRatios(1.0, 0.0, 0.0), seed=1)
    loader = train_mod.build_dataloaders(
        records, manifest, train_mod.EncodingConfig(),
        train_mod.ValueNormConfig(), batch_size=2, seed=1)["train"]
    state = train_mod._capture_rng_state(loader)
    mapped = dict(state)
    mapped["torch_cpu"] = state["torch_cpu"].cuda()
    mapped["torch_cuda"] = [item.cuda() for item in state["torch_cuda"]]
    mapped["train_loader_generator"] = state["train_loader_generator"].cuda()

    train_mod._restore_rng_state(mapped, loader)


def test_supervised_repeated_resume_saves_cumulative_epoch_numbering(tmp_path):
    out = tmp_path / "run"
    first = run_training(_args(out, epochs=2))
    first_last = load_checkpoint(out / "last.pt")

    second = run_training(_args(out, epochs=3, resume=out / "last.pt"))
    second_last = load_checkpoint(out / "last.pt")

    third = run_training(_args(out, epochs=4, resume=out / "last.pt"))
    third_last = load_checkpoint(out / "last.pt")

    assert [first_last["epoch"], second_last["epoch"], third_last["epoch"]] \
        == [2, 3, 4]
    assert [row["epoch"] for row in first["history"]] == [1, 2]
    assert [row["epoch"] for row in second["history"]] == [1, 2, 3]
    assert [row["epoch"] for row in third["history"]] == [1, 2, 3, 4]


def test_supervised_resume_preserves_best_checkpoint_and_early_stopping_state(
        tmp_path, monkeypatch):
    out = tmp_path / "run"
    validation_losses = iter([0.1, 0.9])

    def controlled_evaluate(*args, **kwargs):
        loss = next(validation_losses)
        return {"loss": loss}, loss

    monkeypatch.setattr(train_mod, "evaluate_loader", controlled_evaluate)
    initial_args = _args(out, epochs=1)
    initial_args.early_stopping_patience = 2
    run_training(initial_args)
    original_best = load_checkpoint(out / "best.pt")
    initial_last = load_checkpoint(out / "last.pt")

    resume_args = _args(out, epochs=2, resume=out / "last.pt")
    resume_args.early_stopping_patience = 2
    run_training(resume_args)
    preserved_best = load_checkpoint(out / "best.pt")
    latest = load_checkpoint(out / "last.pt")

    assert preserved_best["best_val_metric"] == pytest.approx(0.1)
    assert preserved_best["best_epoch"] == 1
    for name, value in original_best["model_state"].items():
        assert torch.equal(preserved_best["model_state"][name], value)
    assert latest["epoch"] == 2
    assert latest["best_val_metric"] == pytest.approx(0.1)
    assert latest["best_epoch"] == 1
    assert latest["patience_left"] == 1
    assert any(not torch.equal(latest["model_state"][name], value)
               for name, value in original_best["model_state"].items())
    initial_steps = sorted(float(state["step"]) for state in
                           initial_last["optimizer_state"]["state"].values())
    latest_steps = sorted(float(state["step"]) for state in
                          latest["optimizer_state"]["state"].values())
    assert initial_steps and len(initial_steps) == len(latest_steps)
    assert all(after > before
               for before, after in zip(initial_steps, latest_steps))


def test_supervised_resume_repairs_stale_mirrors_from_authoritative_state(
        tmp_path, monkeypatch):
    out = tmp_path / "run"
    validation_losses = iter([0.1, 0.9])

    def controlled_evaluate(*args, **kwargs):
        loss = next(validation_losses)
        return {"loss": loss}, loss

    monkeypatch.setattr(train_mod, "evaluate_loader", controlled_evaluate)
    initial_args = _args(out, epochs=1)
    initial_args.early_stopping_patience = 2
    run_training(initial_args)

    for name in ("best.pt", "last.pt"):
        path = out / name
        legacy = torch.load(path, weights_only=False)
        legacy.pop("best_epoch")
        legacy.pop("patience_left")
        torch.save(legacy, path)

    resume_args = _args(out, epochs=2, resume=out / "last.pt")
    resume_args.early_stopping_patience = 2
    run_training(resume_args)
    best = load_checkpoint(out / "best.pt")
    latest = load_checkpoint(out / "last.pt")

    assert best["epoch"] == 1
    assert best["best_val_metric"] == pytest.approx(0.1)
    assert latest["epoch"] == 2
    assert latest["best_epoch"] == 1
    # Missing legacy patience safely falls back to the configured full value,
    # then the worse resumed epoch consumes one step.
    assert latest["patience_left"] == 1


def test_split_manifest_reused_across_runs(tmp_path):
    out = tmp_path / "run"
    run_training(_args(out, epochs=1))
    from blocksort.training.splits import load_manifest
    m1 = load_manifest(out / "splits.json")
    # Restarting without explicit resume is prohibited.
    with pytest.raises(UnsupportedResumeError):
        run_training(_args(out, epochs=1))
    m2 = load_manifest(out / "splits.json")
    assert m1 == m2
    assert m1["validation_levels"] and m1["test_levels"]


def test_supervised_resume_rejects_stale_checkpoint_and_lower_target(
        tmp_path):
    out = tmp_path / "run"
    run_training(_args(out, epochs=2))
    state_path = out / "run_state.json"
    before = (state_path.read_bytes(), state_path.stat().st_mtime_ns)

    with pytest.raises(ExperimentSpecIntegrityError, match="branch or rewind"):
        run_training(_args(
            out, epochs=3,
            resume=out / "checkpoints" / "epoch_001.pt"))
    assert before == (state_path.read_bytes(), state_path.stat().st_mtime_ns)

    with pytest.raises(ContinuationHorizonError, match="committed progress 2"):
        run_training(_args(out, epochs=1, resume=out / "last.pt"))
    assert before == (state_path.read_bytes(), state_path.stat().st_mtime_ns)


def test_supervised_resume_equal_target_is_cumulative_no_op(tmp_path):
    out = tmp_path / "run"
    first = run_training(_args(out, epochs=2))
    state_before = (out / "run_state.json").read_bytes()
    second = run_training(_args(out, epochs=2, resume=out / "last.pt"))
    assert second["history"] == first["history"]
    assert (out / "run_state.json").read_bytes() == state_before


def test_supervised_resume_repairs_mirror_after_post_commit_crash(
        tmp_path, monkeypatch):
    out = tmp_path / "run"
    run_training(_args(out, epochs=1))

    def crash(stage):
        if stage == "after_state_commit":
            raise RuntimeError("injected post-commit crash")

    monkeypatch.setattr(train_mod, "_supervised_crash_point", crash)
    with pytest.raises(RuntimeError, match="injected"):
        run_training(_args(out, epochs=2, resume=out / "last.pt"))
    assert load_checkpoint(out / "last.pt")["epoch"] == 1
    assert load_checkpoint(out / "checkpoints" / "epoch_002.pt")["epoch"] == 2

    monkeypatch.setattr(train_mod, "_supervised_crash_point", lambda _stage: None)
    summary = run_training(_args(out, epochs=2, resume=out / "last.pt"))
    assert summary["last_completed_epoch"] == 2
    assert load_checkpoint(out / "last.pt")["epoch"] == 2


def test_supervised_resume_matches_uninterrupted_training_exactly(tmp_path):
    uninterrupted = tmp_path / "uninterrupted"
    resumed = tmp_path / "resumed"
    full = run_training(_args(uninterrupted, epochs=2))
    run_training(_args(resumed, epochs=1))
    split = run_training(_args(
        resumed, epochs=2, resume=resumed / "last.pt"))

    full_checkpoint = load_checkpoint(uninterrupted / "last.pt")
    split_checkpoint = load_checkpoint(resumed / "last.pt")
    assert full["history"] == split["history"]
    for name, value in full_checkpoint["model_state"].items():
        assert torch.equal(value, split_checkpoint["model_state"][name])
    _assert_nested_equal(
        full_checkpoint["optimizer_state"],
        split_checkpoint["optimizer_state"])
