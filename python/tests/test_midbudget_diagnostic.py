"""Fresh mid-budget development-pool and multi-seed evaluation contracts."""

from __future__ import annotations

from blocksort.cotraining.midbudget_dev_pool import (
    STRATA,
    _contains_candidate_key,
    classify_stratum,
)
from blocksort.cotraining.midbudget_multiseed_evaluate import summarize


BUDGETS = [64, 72, 80, 88, 95, 104, 112, 120, 128]


def test_midbudget_strata_cover_broad_first_solve_bands() -> None:
    assert classify_stratum([True] + [False] * 8) == STRATA[0]
    assert classify_stratum([False, False, True] + [False] * 6) == STRATA[1]
    assert classify_stratum([False] * 5 + [True] + [False] * 3) == STRATA[2]
    assert classify_stratum([False] * 8 + [True]) == STRATA[3]
    assert classify_stratum([False] * 9) == STRATA[3]


def test_candidate_keys_are_forbidden_from_generation_contract() -> None:
    assert _contains_candidate_key({"baseline_checkpoint": {}}) is False
    assert _contains_candidate_key({"candidate_checkpoint": {}}) is True
    assert _contains_candidate_key({"nested": [{"learner_sha256": "x"}]}) \
        is True


def _row(signature: str, solved_by_budget: dict[int, list[bool]]) -> dict:
    return {
        "static_level_signature": signature,
        "budgets": {
            str(budget): {
                "trials": [{"solved": value} for value in values],
                "solve_rate": sum(values) / len(values),
            }
            for budget, values in solved_by_budget.items()
        },
    }


def test_multiseed_summary_distinguishes_local_weakness_from_curve_gain() -> None:
    baseline = []
    candidate = []
    for index in range(60):
        base_values = {}
        candidate_values = {}
        for budget in BUDGETS:
            base_values[budget] = [False, False, False]
            if budget == 95:
                base_values[budget] = [index < 30] * 3
                candidate_values[budget] = [index < 10] * 3
            else:
                candidate_values[budget] = [index < 20] * 3
        baseline.append(_row(f"s{index}", base_values))
        candidate.append(_row(f"s{index}", candidate_values))

    result = summarize(
        baseline, candidate, budgets=BUDGETS, trial_count=3,
        bootstrap_replicates=1000, bootstrap_seed=44)

    assert result["equal_budget_weighted_curve_delta"] > 0
    at_95 = result["budget_95_exploratory_assessment"]
    assert at_95["direction_reproduced"] is True
    assert at_95["negative_delta_statistically_supported"] is True
    assert at_95["interpretation"] == \
        "midbudget_weakness_reproduced_with_negative_interval"
