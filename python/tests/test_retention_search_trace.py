from __future__ import annotations

import pytest

from blocksort.cotraining.retention_search_trace import (
    _delay_classification,
    _selection_divergence_detail,
)


def _trace(locator: str, prior: float) -> dict:
    return {
        "timeline": [{
            "simulation": 1,
            "path_locators": [{"action": locator}],
            "selection_trace": [{
                "node_key": "root",
                "node_value_cost": 2.0,
                "locators": [{"action": "a"}, {"action": "b"}],
                "priors": [prior, 1.0 - prior],
                "selected_edge_index": 0 if locator == "a" else 1,
                "selected_locator": {"action": locator},
            }],
        }],
    }


def test_selection_divergence_reports_shared_root_policy_change() -> None:
    result = _selection_divergence_detail(
        _trace("a", 0.51), _trace("b", 0.49))

    assert result is not None
    assert result["simulation"] == 1
    assert result["divergence_depth"] == 0
    assert result["kind"] == "root_selection"
    shared = result["shared_selection_node"]
    assert shared["prior_l1_distance"] == pytest.approx(0.04)
    assert shared["max_prior_absolute_delta"] == pytest.approx(0.02)
    assert shared["node_value_cost_delta"] == 0.0


def test_delay_classification_distinguishes_delay_from_absence() -> None:
    incumbent = {"outcomes": {
        "95": {"solved": True, "first_solution_simulation": 88}}}
    delayed = {"outcomes": {
        "95": {"solved": False, "first_solution_simulation": None},
        "104": {"solved": True, "first_solution_simulation": 101}}}
    absent = {"outcomes": {
        "95": {"solved": False, "first_solution_simulation": None},
        "160": {"solved": False, "first_solution_simulation": None}}}

    assert _delay_classification(
        "incumbent_only", incumbent, delayed)["kind"] \
        == "candidate_search_delay"
    result = _delay_classification("incumbent_only", incumbent, absent)
    assert result["kind"] == "candidate_absent_through_max_budget"
    assert result["delayed_side_additional_budget"] is None
