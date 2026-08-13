"""Selection/backup/graph tests on a tiny abstract graph (no puzzle, no torch).

A ``MockAdapter`` implements the search adapter protocol over a hand-built
directed graph so the PUCT math, cost backups, cycle rejection, and
transposition sharing can be verified in isolation.
"""

from __future__ import annotations

import pytest

from blocksort.search.config import SearchConfig
from blocksort.search.export import build_search_policy_record
from blocksort.search.graph_search import GraphSearch, NoLegalActionsError
from blocksort.search.seeding import derive_trial_seed


class MockAdapter:
    def __init__(self, transitions, terminals, priors=None, values=None,
                 deadlocks=None):
        self.transitions = transitions          # state -> [(label, next_state)]
        self.terminals = set(terminals)
        self.priors = priors or {}
        self.values = values or {}
        self.deadlocks = set(deadlocks or ())

    def key(self, s):
        return s

    def is_terminal(self, s):
        return s in self.terminals

    def is_deadlock(self, s):
        return s in self.deadlocks

    def legal_actions(self, s):
        return [label for label, _ in self.transitions.get(s, [])]

    def apply(self, s, a):
        for label, nxt in self.transitions[s]:
            if label == a:
                return nxt
        raise KeyError((s, a))

    def evaluate(self, s):
        acts = self.legal_actions(s)
        pr = self.priors.get(s)
        if pr is None:
            pr = [1.0 / len(acts)] * len(acts) if acts else []
        return list(pr), float(self.values.get(s, 1.0))

    def to_locator(self, s, a):
        return {"a": a}

    def from_locator(self, s, loc):
        return loc["a"]


def run(adapter, root, **cfg):
    return GraphSearch(adapter, SearchConfig(**cfg)).run(root)


# --------------------------------------------------------------------------
# Cost / backup sign correctness
# --------------------------------------------------------------------------

def test_backup_cost_sign_one_move_to_terminal():
    ad = MockAdapter({"R": [("a", "T")]}, {"T"}, {"R": [1.0]}, {"R": 5.0})
    res = run(ad, "R", simulations=1, value_normalization_constant=20.0)
    # Reaching a terminal in one move costs exactly 1 (Q = 1 + V(terminal=0)).
    assert res.action_q_cost[0] == pytest.approx(1.0)
    assert res.search_value_cost == pytest.approx(1.0)
    assert res.search_value_normalized == pytest.approx(-1.0 / 20.0)
    assert res.solved and res.solution_length == 1
    assert res.first_solution_simulation == 1


def test_backup_accumulates_steps():
    # R -a-> A -b-> T ; two moves to terminal -> Q(R,a) = 2.
    # Value estimates equal the true cost-to-go so the running mean is exact.
    ad = MockAdapter({"R": [("a", "A")], "A": [("b", "T")]}, {"T"},
                     {"R": [1.0], "A": [1.0]}, {"R": 2.0, "A": 1.0})
    res = run(ad, "R", simulations=5, value_normalization_constant=20.0)
    assert res.action_q_cost[0] == pytest.approx(2.0)
    assert res.action_q_cost == pytest.approx([2.0])
    assert res.solution_length == 2
    assert res.first_solution_simulation == 2


def test_optional_simulation_trace_does_not_change_backup():
    ad = MockAdapter(
        {"R": [("a", "A")], "A": [("b", "T")]},
        {"T"},
        {"R": [1.0], "A": [1.0]},
        {"R": 2.0, "A": 1.0},
    )
    gs = GraphSearch(ad, SearchConfig(simulations=1, seed=7))
    gs._validate_config()
    gs._reset_per_run_state(None)
    root, created = gs.table.get_or_create("R", "R")
    assert created
    gs._expand(root, "R")

    gs.stats.simulations += 1
    trace = gs._simulate(root, "R", trace=True)

    assert {key: trace[key] for key in (
        "root_edge_index", "path_length", "path_locators", "leaf_reason",
        "leaf_cost", "solution_changed", "best_solution_length",
    )} == {
        "root_edge_index": 0,
        "path_length": 1,
        "path_locators": [{"a": "a"}],
        "leaf_reason": "model_leaf",
        "leaf_cost": 1.0,
        "solution_changed": False,
        "best_solution_length": None,
    }
    assert trace["selection_trace"][0]["node_key"] == "R"
    assert trace["leaf_expansion"]["node_key"] == "A"
    assert root.N == [1]
    assert root.Q == pytest.approx([2.0])


# --------------------------------------------------------------------------
# Degenerate roots
# --------------------------------------------------------------------------

def test_single_legal_action():
    ad = MockAdapter({"R": [("only", "T")]}, {"T"})
    res = run(ad, "R", simulations=10)
    assert res.chosen_action == "only"
    assert res.solved and res.solution_length == 1
    assert sum(res.visit_counts) == 10


def test_terminal_root_does_not_crash():
    ad = MockAdapter({}, {"R"})
    res = run(ad, "R", simulations=8)
    assert res.legal_actions == []
    assert res.chosen_action is None
    assert res.solved is False
    assert res.first_solution_simulation is None
    assert res.visit_policy == []
    assert res.termination_reason == "solved"


def test_explicit_deadlock_root_has_no_fake_action():
    ad = MockAdapter({}, set(), deadlocks={"R"})
    ad.evaluate = lambda _: pytest.fail("deadlocks must not be model-evaluated")
    res = run(ad, "R", simulations=8, max_cost=77.0)
    assert res.termination_reason == "deadlock"
    assert res.chosen_action is None
    assert res.legal_actions == []
    assert res.visit_policy == []
    assert res.search_value_cost == 77.0
    assert res.stats.deadlocks == 1


def test_deadlock_reached_below_root_is_a_losing_leaf():
    ad = MockAdapter({"R": [("a", "D")]}, set(), deadlocks={"D"})
    res = run(ad, "R", simulations=3, max_cost=50.0)
    assert res.chosen_action == "a"
    assert res.action_q_cost == [51.0]
    assert res.stats.deadlocks == 1
    assert not res.solved


def test_unclassified_empty_action_state_is_adapter_error():
    ad = MockAdapter({}, set())
    with pytest.raises(NoLegalActionsError, match="terminal=False.*state='R'"):
        run(ad, "R", simulations=1)


def test_search_requires_positive_simulation_budget():
    ad = MockAdapter({"R": [("a", "T")]}, {"T"})
    with pytest.raises(ValueError, match="simulations must be positive"):
        run(ad, "R", simulations=0)


@pytest.mark.parametrize("value", [0, -1, 1.5, True])
def test_search_requires_positive_integer_inference_batch_size(value):
    ad = MockAdapter({"R": [("a", "T")]}, {"T"})
    with pytest.raises(ValueError, match="inference_batch_size"):
        run(ad, "R", simulations=1, inference_batch_size=value)


@pytest.mark.parametrize("value", [-0.1, float("nan"), float("inf")])
def test_search_requires_nonnegative_finite_virtual_loss(value):
    ad = MockAdapter({"R": [("a", "T")]}, {"T"})
    with pytest.raises(ValueError, match="virtual_loss"):
        run(ad, "R", simulations=1, virtual_loss=value)


# --------------------------------------------------------------------------
# Cycle rejection
# --------------------------------------------------------------------------

def test_cycle_is_never_followed():
    # From A the highest-prior action loops back to the ancestor R; search must
    # reject it and reach T through z instead.
    ad = MockAdapter(
        {"R": [("x", "A")], "A": [("y", "R"), ("z", "T")]},
        {"T"},
        {"R": [1.0], "A": [0.9, 0.1]},
        {"R": 1.0, "A": 1.0},
    )
    res = run(ad, "R", simulations=20)
    assert res.stats.cycle_rejections > 0
    assert res.solved and res.solution_length == 2  # R-x->A-z->T


# --------------------------------------------------------------------------
# Transposition sharing
# --------------------------------------------------------------------------

def test_transposition_shared_node():
    # Both root edges reach the same state M -> one shared node.
    ad = MockAdapter(
        {"R": [("a", "M"), ("b", "M")], "M": [("c", "T")]},
        {"T"},
        {"R": [0.5, 0.5], "M": [1.0]},
    )
    gs = GraphSearch(ad, SearchConfig(simulations=20))
    res = gs.run("R")
    assert len(gs.table) == 3              # R, M, T
    assert gs.table.hits >= 1              # M/T reused across simulations
    assert res.stats.transposition_hits >= 1


# --------------------------------------------------------------------------
# Determinism, visit-count consistency, temperature
# --------------------------------------------------------------------------

def _diamond():
    return MockAdapter(
        {"R": [("a", "M"), ("b", "N")], "M": [("c", "T")], "N": [("d", "T")]},
        {"T"}, {"R": [0.7, 0.3], "M": [1.0], "N": [1.0]},
    )


def test_deterministic_fixed_seed():
    r1 = run(_diamond(), "R", simulations=30, seed=123)
    r2 = run(_diamond(), "R", simulations=30, seed=123)
    assert r1.visit_counts == r2.visit_counts


def test_trial_seed_derivation_is_stable_distinct_and_context_sensitive():
    seeds = [
        derive_trial_seed(
            42, trial_index=i, level_identity="level-a",
            evaluation_context="frontier")
        for i in range(5)
    ]
    assert seeds == [
        derive_trial_seed(
            42, trial_index=i, level_identity="level-a",
            evaluation_context="frontier")
        for i in range(5)
    ]
    assert len(set(seeds)) == 5
    assert seeds != [
        derive_trial_seed(
            42, trial_index=i, level_identity="level-b",
            evaluation_context="frontier")
        for i in range(5)
    ]
    assert seeds != [
        derive_trial_seed(
            42, trial_index=i, level_identity="level-a",
            evaluation_context="benchmark")
        for i in range(5)
    ]


def _result_signature(result):
    return {
        "chosen": result.chosen_action,
        "locators": result.legal_action_locators,
        "visits": result.visit_counts,
        "policy": result.visit_policy,
        "q": result.action_q_cost,
        "priors": result.priors,
        "value": result.search_value_cost,
        "pv": result.principal_variation,
        "solution": result.solution_locators,
        "reason": result.termination_reason,
        "stats": (
            result.stats.seed,
            result.stats.simulations,
            result.stats.nodes_expanded,
            result.stats.unique_states,
            result.stats.transposition_hits,
            result.stats.cycle_rejections,
            result.stats.deadlocks,
        ),
    }


def test_graph_search_reuse_matches_fresh_search_and_is_order_independent():
    ad = MockAdapter(
        {
            "A": [("a1", "M"), ("a2", "T")],
            "B": [("b1", "N"), ("b2", "T")],
            "M": [("m", "T")],
            "N": [("n", "T")],
        },
        {"T"},
        {"A": [0.8, 0.2], "B": [0.3, 0.7], "M": [1.0], "N": [1.0]},
    )
    cfg = SearchConfig(
        simulations=30, seed=91, dirichlet_alpha=0.5,
        dirichlet_weight=0.25,
    )
    reused = GraphSearch(ad, cfg)
    a_then = reused.run("A")
    a_snapshot = _result_signature(a_then)
    b_then = reused.run("B")

    fresh_a = GraphSearch(ad, cfg).run("A")
    fresh_b = GraphSearch(ad, cfg).run("B")
    assert a_snapshot == _result_signature(fresh_a)
    assert _result_signature(b_then) == _result_signature(fresh_b)
    # The earlier result is a stable snapshot and the public table is only the
    # most recent run.
    assert _result_signature(a_then) == a_snapshot
    assert "A" not in reused.table

    reverse = GraphSearch(ad, cfg)
    b_reverse = reverse.run("B")
    a_reverse = reverse.run("A")
    assert _result_signature(b_reverse) == _result_signature(fresh_b)
    assert _result_signature(a_reverse) == _result_signature(fresh_a)


def test_graph_search_visit_reset_does_not_accumulate_or_retain_solution():
    ad = MockAdapter({"R": [("a", "T")]}, {"T"})
    gs = GraphSearch(ad, SearchConfig(simulations=7))
    first = gs.run("R")
    second = gs.run("R")
    assert first.visit_counts == second.visit_counts == [7]
    assert first.stats.simulations == second.stats.simulations == 7

    deadlock_adapter = MockAdapter({}, set(), deadlocks={"D"})
    gs.adapter = deadlock_adapter
    deadlocked = gs.run("D")
    assert not deadlocked.solved
    assert deadlocked.solution_locators is None
    assert deadlocked.termination_reason == "deadlock"


def test_graph_search_explicit_trial_seed_is_reproducible_and_observable():
    ad = _diamond()
    cfg = SearchConfig(
        simulations=20, seed=999, dirichlet_alpha=0.5,
        dirichlet_weight=0.4,
    )
    shared = GraphSearch(ad, cfg)
    seed_100_a = shared.run("R", seed=100)
    seed_100_b = shared.run("R", seed=100)
    seed_101 = shared.run("R", seed=101)
    fresh_100 = GraphSearch(ad, cfg).run("R", seed=100)

    assert _result_signature(seed_100_a) == _result_signature(seed_100_b)
    assert _result_signature(seed_100_a) == _result_signature(fresh_100)
    assert seed_100_a.stats.seed == 100
    assert seed_101.stats.seed == 101
    assert seed_100_a.priors != pytest.approx(seed_101.priors)


def test_graph_search_default_seed_remains_an_independent_one_off_default():
    ad = _diamond()
    cfg = SearchConfig(
        simulations=20, seed=321, dirichlet_alpha=0.5,
        dirichlet_weight=0.4,
    )
    shared = GraphSearch(ad, cfg)
    first = shared.run("R")
    second = shared.run("R")
    assert _result_signature(first) == _result_signature(second)
    assert first.stats.seed == second.stats.seed == 321


def test_search_result_snapshot_is_deeply_independent():
    class NestedLocatorAdapter(MockAdapter):
        def to_locator(self, s, a):
            return {"a": a, "metadata": {"cells": [[1, 2], [3, 4]]}}

    ad = NestedLocatorAdapter({"R": [("a", "T")]}, {"T"})
    gs = GraphSearch(ad, SearchConfig(simulations=3))
    first = gs.run("R")

    # Internal mutation after publication cannot reach the result.
    gs.table.get("R").locators[0]["metadata"]["cells"][0][0] = 99
    assert first.legal_action_locators[0]["metadata"]["cells"][0][0] == 1
    assert first.chosen_action_locator["metadata"]["cells"][0][0] == 1
    assert first.principal_variation[0]["metadata"]["cells"][0][0] == 1
    assert first.solution_locators[0]["metadata"]["cells"][0][0] == 1

    # Caller mutation cannot reach another field, the searcher, or a later run.
    first.legal_action_locators[0]["metadata"]["cells"][0][0] = 77
    first.stats.simulations = 999
    assert first.solution_locators[0]["metadata"]["cells"][0][0] == 1
    second = gs.run("R")
    assert second.legal_action_locators[0]["metadata"]["cells"][0][0] == 1
    assert second.stats.simulations == 3


def test_dirichlet_noise_changes_root_priors_only():
    original_root_priors = [0.8, 0.2]
    original_child_priors = [0.6, 0.4]
    ad = MockAdapter(
        {"R": [("a", "M"), ("b", "M")],
         "M": [("c", "T1"), ("d", "T2")]},
        {"T1", "T2"},
        {"R": original_root_priors, "M": original_child_priors},
    )
    gs = GraphSearch(
        ad,
        SearchConfig(
            simulations=1,
            dirichlet_alpha=0.5,
            dirichlet_weight=0.4,
            seed=123,
        ),
    )

    result = gs.run("R")
    child = gs.table.get("M")

    assert result.priors != pytest.approx(original_root_priors)
    assert child is not None and child.expanded
    assert child.priors == pytest.approx(original_child_priors)


def test_visit_count_consistency():
    res = run(_diamond(), "R", simulations=25)
    # Every simulation traverses exactly one root edge.
    assert sum(res.visit_counts) == 25


def test_temperature_zero_picks_max_visit():
    res = run(_diamond(), "R", simulations=40, temperature=0.0)
    best = max(range(len(res.visit_counts)), key=lambda i: res.visit_counts[i])
    assert res.chosen_action == res.legal_actions[best]


def test_temperature_policy_sums_to_one():
    res = run(_diamond(), "R", simulations=40, temperature=1.0)
    assert sum(res.visit_policy) == pytest.approx(1.0)


# --------------------------------------------------------------------------
# Search improves over a deliberately weak policy
# --------------------------------------------------------------------------

def test_search_beats_misleading_prior():
    # Prior strongly favors 'a' (a long 4-move detour); 'b' solves in 1 move.
    ad = MockAdapter(
        {"R": [("a", "A1"), ("b", "T")],
         "A1": [("c", "A2")], "A2": [("d", "A3")], "A3": [("e", "T")]},
        {"T"},
        {"R": [0.95, 0.05]},
        {},  # uninformative values (default 1.0)
    )
    budget1 = run(ad, "R", simulations=1)
    biggish = run(ad, "R", simulations=200)
    assert budget1.chosen_action == "a"          # ~ raw policy argmax
    assert biggish.chosen_action == "b"          # search corrects the prior
    assert biggish.search_value_cost <= 1.0 + 1e-6
    assert biggish.solution_length == 1


# --------------------------------------------------------------------------
# Search-policy serialization
# --------------------------------------------------------------------------

def test_search_policy_record_is_serializable():
    import json
    res = run(_diamond(), "R", simulations=20, temperature=1.0)
    rec = build_search_policy_record("R", res, model_checkpoint="ckpt.pt",
                                     simulations=20, static_signature="sig")
    text = json.dumps(rec)
    back = json.loads(text)
    assert back["state_key"] == "R"
    assert len(back["legal_actions"]) == len(back["visit_counts"])
    assert len(back["visit_policy"]) == len(back["visit_counts"])
    assert back["simulations"] == 20
