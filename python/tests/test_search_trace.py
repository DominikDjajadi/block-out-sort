from __future__ import annotations

import pytest

from blocksort.cotraining import search_trace
from blocksort.search.config import SearchConfig
from blocksort.search.graph_search import GraphSearch


class _Adapter:
    def __init__(self, values=None):
        self.transitions = {
            "R": [("a", "A")],
            "A": [("b", "T")],
        }
        self.values = {"R": 2.0, "A": 1.0, **(values or {})}

    def key(self, state):
        return state

    def is_terminal(self, state):
        return state == "T"

    def is_deadlock(self, state):
        return False

    def legal_actions(self, state):
        return [action for action, _ in self.transitions.get(state, [])]

    def apply(self, state, action):
        return next(
            child for candidate, child in self.transitions[state]
            if candidate == action)

    def evaluate(self, state):
        actions = self.legal_actions(state)
        return [1.0 / len(actions)] * len(actions), self.values[state]

    def to_locator(self, _state, action):
        return {"action": action}

    def from_locator(self, _state, locator):
        return locator["action"]


class _BatchAdapter(_Adapter):
    def evaluate_batch_with_legal_actions(self, states, legal_actions):
        return [self.evaluate(state) for state in states]


def test_search_trace_accepts_both_persisted_audit_schemas():
    assert search_trace.AUDIT_SEMANTICS == {
        "multi_candidate_paired_transfer_audit_v2",
        "multi_candidate_paired_transfer_audit_v3",
    }


def test_traced_search_preserves_search_result_and_records_paths():
    config = SearchConfig(simulations=3, seed=17)

    traced = search_trace._run_traced_search(
        _Adapter(), "R", config=config)
    regular = GraphSearch(_Adapter(), config).run("R")

    assert traced["final"]["solved"] == regular.solved
    assert traced["final"]["solution_length"] == regular.solution_length
    assert traced["final"]["visit_counts"] == regular.visit_counts
    assert traced["final"]["action_q_costs"] == pytest.approx(
        regular.action_q_cost)
    assert len(traced["timeline"]) == 3
    assert traced["timeline"][0]["leaf_reason"] == "model_leaf"
    assert traced["timeline"][0]["path_locators"] == [{"action": "a"}]
    root_selection = traced["timeline"][0]["selection_trace"][0]
    assert root_selection["node_key"] == "R"
    assert root_selection["selected_edge_index"] == 0
    assert root_selection["priors"] == [1.0]
    assert root_selection["exploration_u"] == pytest.approx([1.5])
    assert traced["timeline"][0]["leaf_expansion"]["node_key"] == "A"
    assert traced["timeline"][0]["leaf_expansion"]["value_cost"] == 1.0
    assert traced["timeline"][1]["leaf_reason"] == "terminal"
    assert traced["timeline"][1]["best_solution_length"] == 2
    assert search_trace._first_solution_simulation(traced) == 2


def test_batched_trace_preserves_production_search_result():
    config = SearchConfig(simulations=3, seed=17, inference_batch_size=2)

    traced = search_trace._run_traced_search(
        _BatchAdapter(), "R", config=config)
    regular = GraphSearch(_BatchAdapter(), config).run("R")

    assert traced["inference_batch_size"] == 2
    assert traced["final"]["solved"] == regular.solved
    assert traced["final"]["solution_length"] == regular.solution_length
    assert traced["final"]["visit_counts"] == regular.visit_counts
    assert traced["final"]["action_q_costs"] == pytest.approx(
        regular.action_q_cost)
    assert len(traced["timeline"]) == 3


def test_paired_report_finds_first_path_divergence():
    config = SearchConfig(simulations=3, seed=19)
    incumbent = search_trace._run_traced_search(
        _Adapter(), "R", config=config)
    candidate = search_trace._run_traced_search(
        _Adapter({"A": 4.0}), "R", config=config)

    result = search_trace._paired_report(incumbent, candidate)

    assert result["max_root_prior_absolute_delta"] == 0.0
    assert result["first_root_edge_divergence_simulation"] is None
    assert result["first_full_path_divergence_simulation"] is None
    assert result["root_value_cost_delta"] == 0.0
    assert result["incumbent_first_solution_simulation"] == 2
    assert result["candidate_first_solution_simulation"] == 2


def test_common_prefix_stops_at_first_different_action():
    left = [{"action": "a"}, {"action": "b"}, {"action": "c"}]
    right = [{"action": "a"}, {"action": "x"}, {"action": "c"}]

    assert search_trace._common_prefix(left, right) == 1
