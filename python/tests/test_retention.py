from __future__ import annotations

from pathlib import Path

import pytest

from blocksort.cotraining.config import CoTrainingConfig
from blocksort.cotraining.loop import CoTraining
from blocksort.cotraining.retention import (
    apply_retention_guard,
    load_retention_pool,
    summarize_retention,
)
from blocksort.cotraining.run import build_parser, config_from_args


def _row(signature: str, band: str, solved: dict[int, bool]):
    return {
        "static_level_signature": signature,
        "difficulty_stratum": band,
        "budgets": {
            str(budget): {"solved": outcome}
            for budget, outcome in solved.items()
        },
    }


def test_fresh_midbudget_pool_yields_balanced_stable_retention_slice():
    source = Path(__file__).resolve().parents[2] / (
        "data/eval/midbudget_dev_seed8240_v1.jsonl")
    selected, all_signatures, manifest = load_retention_pool(
        source, per_band=2)
    assert len(all_signatures) == 200
    assert len(selected) == 8
    assert manifest["source_band_counts"] == {
        "first_solved_72_through_88": 50,
        "first_solved_95_through_112": 50,
        "solved_by_64": 50,
        "first_solved_120_or_later_or_unsolved": 50,
    }
    for band in manifest["source_band_counts"]:
        signatures = [
            row["static_level_signature"] for row in selected
            if row["difficulty_stratum"] == band]
        assert signatures == sorted(signatures)
        assert len(signatures) == 2


def test_full_retention_pool_preserves_intentionally_unequal_strata():
    source = Path(__file__).resolve().parents[2] / (
        "data/eval/retention_replication_seed8250_v1.jsonl")
    selected, all_signatures, manifest = load_retention_pool(
        source, per_band=None)
    assert len(all_signatures) == 500
    assert len(selected) == 500
    assert manifest["selection_policy"] == "all_source_levels_v1"
    assert manifest["selected_band_counts"] == {
        "first_solved_120_or_later_or_unsolved": 100,
        "first_solved_72_through_88": 100,
        "first_solved_95_through_112": 200,
        "solved_by_64": 100,
    }


def test_retention_summary_is_paired_and_band_specific():
    reference = [
        _row("easy-a", "easy", {64: True, 95: True}),
        _row("easy-b", "easy", {64: True, 95: True}),
        _row("hard-a", "hard", {64: False, 95: False}),
        _row("hard-b", "hard", {64: False, 95: False}),
    ]
    candidate = [
        _row("easy-a", "easy", {64: False, 95: True}),
        _row("easy-b", "easy", {64: True, 95: True}),
        _row("hard-a", "hard", {64: True, 95: True}),
        _row("hard-b", "hard", {64: False, 95: False}),
    ]
    report = summarize_retention(
        reference, candidate, budgets=(64, 95), max_regression=0.25)
    assert report["passed"] is False
    easy_64 = report["bands"]["easy"]["per_budget"]["64"]
    assert easy_64["solve_rate_delta"] == pytest.approx(-0.5)
    assert easy_64["reference_only_solves"] == 1
    hard_64 = report["bands"]["hard"]["per_budget"]["64"]
    assert hard_64["solve_rate_delta"] == pytest.approx(0.5)
    assert hard_64["candidate_only_solves"] == 1


def test_retention_monitoring_does_not_change_continuation_but_guard_does():
    continuation = {
        "decision": "accept_as_anchor", "accepted": True,
        "milestone": True, "reasons": [],
    }
    report = {"passed": False}
    monitored = apply_retention_guard(
        continuation, report, enforce=False)
    assert monitored["accepted"] is True
    assert monitored["retention_passed"] is False

    enforced = apply_retention_guard(
        continuation, report, enforce=True)
    assert enforced["accepted"] is False
    assert enforced["decision"] == "rollback_to_anchor"
    assert "difficulty_band_retention_regression" in enforced["reasons"]


def test_retention_cli_is_explicit_and_configurable():
    args = build_parser().parse_args([
        "--protagonist-checkpoint", "protagonist.pt",
        "--designer-checkpoint", "designer.pt",
        "--base-dataset", "base.jsonl",
        "--shadow-learner",
        "--learner-retention-dataset", "retention.jsonl",
        "--learner-retention-budgets", "32", "64",
        "--learner-retention-per-band", "12",
        "--learner-retention-use-full-pool",
        "--learner-retention-max-regression", "0.1",
        "--learner-retention-enforce",
    ])
    cfg = config_from_args(args)
    assert cfg.learner_retention_dataset == "retention.jsonl"
    assert cfg.learner_retention_budgets == (32, 64)
    assert cfg.learner_retention_per_band == 12
    assert cfg.learner_retention_use_full_pool is True
    assert cfg.learner_retention_max_regression == pytest.approx(0.1)
    assert cfg.learner_retention_enforce is True


def test_storage_pruning_preserves_current_milestone_and_reports(tmp_path):
    cfg = CoTrainingConfig(
        output_dir=str(tmp_path), shadow_learner_enabled=True,
        learner_milestone_interval=2,
        prune_superseded_round_artifacts=True,
    )
    loop = CoTraining(cfg)
    names = (
        "replay.jsonl", "level_replay.jsonl", "training_sample.jsonl",
        "training_sample_source.jsonl", "candidate.pt", "report.json",
    )
    for number in (1, 2, 3):
        directory = tmp_path / f"round_{number:03d}"
        directory.mkdir()
        for name in names:
            (directory / name).write_text(name, encoding="utf-8")
    state = {
        "completed_rounds": [1, 2, 3],
        "commits": [
            {"round": 1, "promoted": False},
            {"round": 2, "promoted": False},
            {"round": 3, "promoted": False},
        ],
    }
    event = loop._prune_superseded_round_artifacts(
        state, current_round=3)
    assert event["deleted_file_count"] == 6
    for name in names[:4]:
        assert not (tmp_path / "round_001" / name).exists()
    assert not (tmp_path / "round_002" / "replay.jsonl").exists()
    assert not (tmp_path / "round_002" / "level_replay.jsonl").exists()
    assert (tmp_path / "round_002" / "training_sample.jsonl").is_file()
    for number in (1, 2, 3):
        assert (tmp_path / f"round_{number:03d}" / "candidate.pt").is_file()
        assert (tmp_path / f"round_{number:03d}" / "report.json").is_file()
    assert (tmp_path / "round_003" / "replay.jsonl").is_file()


@pytest.mark.parametrize("kwargs", [
    {"learner_retention_dataset": "pool.jsonl"},
    {"learner_retention_enforce": True},
    {"learner_retention_budgets": ()},
    {"learner_retention_budgets": (64, 64)},
    {"learner_retention_per_band": 0},
    {"learner_retention_use_full_pool": True},
    {"learner_retention_max_regression": 1.1},
])
def test_invalid_retention_config_is_rejected(kwargs):
    with pytest.raises(ValueError):
        CoTrainingConfig(**kwargs)
