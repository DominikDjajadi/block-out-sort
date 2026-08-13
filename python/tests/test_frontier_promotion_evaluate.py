"""Paired frontier promotion decision tests."""

from __future__ import annotations

import pytest

from blocksort.cotraining.frontier_promotion_evaluate import (
    paired_bootstrap_lower_bound, summarize_paired_promotion)


BUDGETS = [20, 34, 57, 95, 160]
WEIGHTS = [0.2] * 5


def _row(signature: str, solved) -> dict:
    return {
        "static_level_signature": signature,
        "budgets": {
            str(budget): {"solved": value}
            for budget, value in zip(BUDGETS, solved)
        },
    }


def test_paired_bootstrap_is_deterministic() -> None:
    first, samples_first = paired_bootstrap_lower_bound(
        [0.0, 0.2, 0.4], confidence_level=0.95,
        replicates=100, seed=7)
    second, samples_second = paired_bootstrap_lower_bound(
        [0.0, 0.2, 0.4], confidence_level=0.95,
        replicates=100, seed=7)
    assert first == second
    assert samples_first == samples_second


def test_summary_promotes_only_when_every_condition_passes() -> None:
    champion = []
    candidate = []
    for index in range(100):
        champion.append(_row(f"s{index}", [False] * 5))
        candidate.append(_row(f"s{index}", [index < 20] * 5))

    result = summarize_paired_promotion(
        champion, candidate, budgets=BUDGETS, weights=WEIGHTS,
        minimum_delta=0.01, maximum_per_budget_regression=0.02,
        bootstrap_confidence=0.95, bootstrap_replicates=1000,
        bootstrap_seed=8)

    assert result["weighted_solve_rate_delta"] == pytest.approx(0.2)
    assert result["promoted"] is True


def test_summary_rejects_per_budget_regression_despite_overall_gain() -> None:
    champion = []
    candidate = []
    for index in range(100):
        champion.append(_row(
            f"s{index}", [index < 10, False, False, False, False]))
        candidate.append(_row(
            f"s{index}", [False, index < 50, index < 50,
                           index < 50, index < 50]))

    result = summarize_paired_promotion(
        champion, candidate, budgets=BUDGETS, weights=WEIGHTS,
        minimum_delta=0.01, maximum_per_budget_regression=0.02,
        bootstrap_confidence=0.95, bootstrap_replicates=1000,
        bootstrap_seed=9)

    assert result["weighted_solve_rate_delta"] > 0.01
    assert result["conditions"]["per_budget_regression_guards"] is False
    assert result["promoted"] is False
