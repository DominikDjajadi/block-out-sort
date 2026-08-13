"""Budget-sweep evaluation and promotion contracts."""

from __future__ import annotations

import json

import pytest

from blocksort.cotraining.eval_split import create_eval_split_manifest
from blocksort.expert_iteration.budget_sweep import (
    main, select_evaluation_records, summarize_budget_sweep)
from blocksort.expert_iteration.promotion import (
    validate_budget_sweep_promotion_evidence)
from blocksort.serialization import level_from_dict
from blocksort.signature import static_level_signature


def _report(scores, known=None, total=10, regret_known=None):
    known = known or {budget: total for budget in scores}
    regret_known = regret_known or {
        budget: known[budget] - 1 for budget in scores}
    return {
        "states": total,
        "total_evaluated_count": total,
        "budgets": {
            str(budget): {
                "total_evaluated_count": total,
                "search_optimal_classification_count": known[budget],
                "search_unknown_classification_count": total - known[budget],
                "search_optimal_classification_coverage":
                    known[budget] / total,
                "search_confirmed_optimal_count": int(scores[budget] * total),
                "search_confirmed_optimal_rate": scores[budget],
                "search_solved_count": int(scores[budget] * total),
                "search_solve_rate_total": scores[budget],
                "search_exact_regret_count": regret_known[budget],
                "search_unknown_exact_regret_count":
                    total - regret_known[budget],
                "search_exact_regret_coverage": regret_known[budget] / total,
                "search_mean_regret": 0.5,
            }
            for budget in scores
        },
    }


def _solve_report(solved, total=10):
    return {
        "states": total,
        "total_evaluated_count": total,
        "budgets": {
            str(budget): {
                "total_evaluated_count": total,
                "search_solved_count": count,
                "search_solve_rate_total": count / total,
            }
            for budget, count in solved.items()
        },
    }


def _fixed_split(tmp_path):
    records = []
    for index in range(4):
        size = 3 + index
        level = {
            "name": f"sealed-{index}",
            "cols": size,
            "rows": size,
            "blocks": [{"color": "red", "cells": [[1, 1]]}],
            "exits": [
                {"edge": "left", "start": 1, "length": 1, "color": "red"}
            ],
        }
        signature = static_level_signature(level_from_dict(level))
        records.append({
            "level_id": signature,
            "static_level_signature": signature,
            "generation_bucket": "even" if index % 2 == 0 else "odd",
            "level": level,
        })
    pool = tmp_path / "pool.jsonl"
    pool.write_text(
        "".join(json.dumps(record) + "\n" for record in records),
        encoding="utf-8")
    manifest_path = tmp_path / "split.json"
    manifest = create_eval_split_manifest(
        pool, manifest_path, validation_count=2, split_seed=1729)
    return pool, manifest_path, manifest


def test_budget_sweep_selects_only_explicit_manifest_role(tmp_path):
    pool, manifest_path, manifest = _fixed_split(tmp_path)

    selected, identity = select_evaluation_records(
        pool, split_manifest_path=manifest_path, split_role="final_test")

    assert [record["static_level_signature"] for record in selected] == [
        item["signature"] for item in manifest["final_test"]]
    assert identity["role"] == "final_test"
    assert identity["selected_count"] == 2
    assert (identity["evaluation_split_fingerprint"]
            == manifest["evaluation_split_fingerprint"])


def test_budget_sweep_manifest_requires_explicit_role(tmp_path):
    pool, manifest_path, _ = _fixed_split(tmp_path)

    with pytest.raises(ValueError, match="explicit split_role"):
        select_evaluation_records(
            pool, split_manifest_path=manifest_path)
    with pytest.raises(ValueError, match="immutable split manifest"):
        select_evaluation_records(pool, split_role="final_test")


def test_final_test_cli_requires_new_durable_output(tmp_path):
    common = [
        "--checkpoint", "checkpoint.pt",
        "--eval-levels-dataset", "pool.jsonl",
        "--eval-split-manifest", "split.json",
        "--split-role", "final_test",
        "--budgets", "1",
    ]
    with pytest.raises(SystemExit):
        main(common)

    existing = tmp_path / "final.json"
    existing.write_text("already evaluated", encoding="utf-8")
    with pytest.raises(FileExistsError, match="will not be overwritten"):
        main([*common, "--output", str(existing)])


def test_budget_sweep_cli_requires_explicit_manifest_role():
    with pytest.raises(SystemExit):
        main([
            "--checkpoint", "checkpoint.pt",
            "--eval-levels-dataset", "pool.jsonl",
            "--budgets", "1",
        ])


def test_budget_sweep_summary_reports_denominators_and_bucket_sums():
    report = _report({1: 0.2, 2: 0.4}, known={1: 8, 2: 10}, total=10)
    buckets = {
        "hard": _report(
            {1: 0.2, 2: 0.2}, known={1: 3, 2: 5},
            regret_known={1: 3, 2: 4}, total=5),
        "ood": _report(
            {1: 0.2, 2: 0.6}, known={1: 5, 2: 5},
            regret_known={1: 4, 2: 5}, total=5),
    }
    summary = summarize_budget_sweep(
        report, budgets=[1, 2], bucket_reports=buckets)

    b1 = summary["per_budget"]["1"]
    assert b1["total_level_count"] == 10
    assert b1["classification_known_count"] == 8
    assert b1["classification_coverage"] == pytest.approx(0.8)
    assert b1["confirmed_optimal_count"] == 2
    assert b1["confirmed_optimal_rate_total"] == pytest.approx(0.2)
    assert b1["confirmed_optimal_rate_classified"] == pytest.approx(2 / 8)
    assert b1["solved_count"] == 2
    assert b1["solve_rate_total"] == pytest.approx(0.2)
    assert b1["timeout_unknown_count"] == 2
    assert b1["exact_regret_coverage"] == pytest.approx(0.7)
    assert sum(
        item["1"]["total_level_count"]
        for item in summary["bucket_breakdown"].values()) == 10


def test_budget_sweep_summary_rejects_bucket_total_mismatch():
    with pytest.raises(ValueError, match="bucket breakdown totals"):
        summarize_budget_sweep(
            _report({1: 0.2}, total=10),
            budgets=[1],
            bucket_reports={"hard": _report({1: 2 / 9}, total=9)},
        )


def test_budget_sweep_summary_rejects_duplicate_budgets():
    with pytest.raises(ValueError, match="duplicates"):
        summarize_budget_sweep(
            _report({1: 0.2}), budgets=[1, 1])


def test_weighted_budget_sweep_promotion_accepts_strict_improvement():
    incumbent = _report({1: 0.2, 2: 0.4, 4: 0.4})
    candidate = _report({1: 0.4, 2: 0.4, 4: 0.6})
    evidence = validate_budget_sweep_promotion_evidence(
        incumbent, candidate, budgets=[1, 2, 4], weights=[0.5, 0.3, 0.2])

    assert evidence.candidate_score > evidence.incumbent_score
    assert evidence.report_fields()["promotion_budget_list"] == [1, 2, 4]
    assert evidence.report_fields()["promotion_comparison_count"] == 30
    assert evidence.report_fields()["promotion_per_budget"]["1"]["outcome"] \
        == "win"
    assert evidence.report_fields()["promotion_per_budget"]["2"]["outcome"] \
        == "tie"


def test_weighted_budget_sweep_tie_does_not_clear_margin():
    incumbent = _report({1: 0.4, 2: 0.4})
    candidate = _report({1: 0.4, 2: 0.4})
    evidence = validate_budget_sweep_promotion_evidence(
        incumbent, candidate, budgets=[1, 2], weights=[1, 1])

    assert (evidence.candidate_score > evidence.incumbent_score + 0.0) is False


def test_weighted_solve_rate_promotes_more_completed_levels():
    evidence = validate_budget_sweep_promotion_evidence(
        _solve_report({4: 1, 8: 2, 16: 3}),
        _solve_report({4: 1, 8: 3, 16: 5}),
        budgets=[4, 8, 16],
        weights=[0.2, 0.3, 0.5],
        metric="weighted_budget_sweep_solve_rate",
    )

    fields = evidence.report_fields()
    assert evidence.evidence_kind == "solved"
    assert evidence.incumbent_score == pytest.approx(0.23)
    assert evidence.candidate_score == pytest.approx(0.36)
    assert fields["promotion_prev_solved_count"] == 6
    assert fields["promotion_candidate_solved_count"] == 9
    assert "promotion_prev_confirmed_optimal_count" not in fields
    assert fields["promotion_per_budget"]["16"]["outcome"] == "win"


def test_solve_rate_promotion_rejects_rate_not_matching_total():
    candidate = _solve_report({4: 2, 8: 3})
    candidate["budgets"]["4"]["search_solve_rate_total"] = 0.9
    with pytest.raises(ValueError, match="does not match solved/total"):
        validate_budget_sweep_promotion_evidence(
            _solve_report({4: 1, 8: 2}),
            candidate,
            budgets=[4, 8],
            weights=[0.5, 0.5],
            metric="weighted_budget_sweep_solve_rate",
        )


def test_budget_weight_mismatch_is_rejected():
    with pytest.raises(ValueError, match="one value per"):
        validate_budget_sweep_promotion_evidence(
            _report({1: 0.2, 2: 0.4}),
            _report({1: 0.4, 2: 0.6}),
            budgets=[1, 2],
            weights=[1.0],
        )


def test_confirmed_count_cannot_exceed_classification_count():
    report = _report({1: 0.4, 2: 0.4}, known={1: 2, 2: 2})
    report["budgets"]["1"]["search_confirmed_optimal_count"] = 4
    with pytest.raises(ValueError, match="classification-known"):
        validate_budget_sweep_promotion_evidence(
            report, _report({1: 0.4, 2: 0.4}),
            budgets=[1, 2], weights=[1.0, 1.0])
