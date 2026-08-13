"""Static signature and exact-value oracle tests."""

from __future__ import annotations

import copy

import pytest

from blocksort import (
    Environment,
    Oracle,
    level_from_dict,
    static_level_signature,
)

ENV = Environment()


# ---- static signatures ----

BASE = {
    "name": "sig",
    "cols": 5,
    "rows": 5,
    "blocks": [{"color": "red", "cells": [[2, 1]]}],
    "exits": [{"edge": "top", "start": 1, "length": 1, "color": "red"}],
}


def sig(d):
    return static_level_signature(level_from_dict(d))


def test_same_level_same_signature():
    assert sig(BASE) == sig(copy.deepcopy(BASE))


def test_block_position_does_not_change_signature():
    moved = copy.deepcopy(BASE)
    moved["blocks"][0]["cells"] = [[0, 1]]  # same shape, different position
    assert sig(moved) == sig(BASE)


def test_different_gate_layout_differs():
    d = copy.deepcopy(BASE)
    d["exits"][0]["start"] = 2
    assert sig(d) != sig(BASE)


def test_different_holes_differ():
    d = copy.deepcopy(BASE)
    d["holes"] = [[0, 0]]
    assert sig(d) != sig(BASE)


def test_different_unlock_thresholds_differ():
    d = copy.deepcopy(BASE)
    d["exits"][0]["unlockAt"] = 1
    assert sig(d) != sig(BASE)


def test_locked_region_change_differs():
    d = copy.deepcopy(BASE)
    d["lockedRegions"] = [{"cells": [[3, 3]], "unlockAt": 1}]
    assert sig(d) != sig(BASE)


# ---- oracle ----

SINGLE = level_from_dict({
    "name": "single", "cols": 4, "rows": 4,
    "blocks": [{"color": "red", "cells": [[2, 1]]}],
    "exits": [{"edge": "top", "start": 1, "length": 1, "color": "red"}],
})

TWO = level_from_dict({
    "name": "two", "cols": 5, "rows": 5,
    "blocks": [{"color": "red", "cells": [[0, 0]]},
               {"color": "blue", "cells": [[4, 4]]}],
    "exits": [{"edge": "top", "start": 0, "length": 1, "color": "red"},
              {"edge": "bottom", "start": 4, "length": 1, "color": "blue"}],
})


def test_exact_remaining_value():
    o = Oracle(ENV)
    assert o.optimal_remaining_moves(ENV.initial_state(SINGLE)) == 1
    assert o.optimal_remaining_moves(ENV.initial_state(TWO)) == 2


def test_terminal_value_is_zero():
    o = Oracle(ENV)
    state = ENV.initial_state(SINGLE)
    # exit the only block
    analysis = o.analyze(state)
    exit_action = next(a.action for a in analysis.actions if a.optimal)
    terminal = ENV.apply_action(state, exit_action)
    assert o.optimal_remaining_moves(terminal) == 0


def test_single_optimal_action():
    o = Oracle(ENV)
    state = ENV.initial_state(SINGLE)
    optimal = o.optimal_actions(state)
    assert len(optimal) == 1
    assert optimal[0].exit is True


def test_multiple_optimal_actions():
    o = Oracle(ENV)
    state = ENV.initial_state(TWO)
    optimal = o.optimal_actions(state)
    # Either block may exit first; both are optimal.
    assert len(optimal) == 2
    assert all(a.exit for a in optimal)


def test_action_costs_and_regrets():
    o = Oracle(ENV)
    state = ENV.initial_state(SINGLE)
    analysis = o.analyze(state)
    assert analysis.value == 1
    for a in analysis.actions:
        if a.optimal:
            assert a.regret == 0
            assert a.cost == 1
        else:
            assert a.regret is not None and a.regret > 0
            assert a.cost == 1 + a.successor_value


def test_cache_returns_identical_results():
    o = Oracle(ENV)
    state = ENV.initial_state(TWO)
    v1 = o.value(state)
    cache_size = len(o._result_cache)
    v2 = o.value(state)
    assert v1 == v2
    assert len(o._result_cache) == cache_size  # second call was a cache hit


def test_oracle_never_exact_when_exhausted():
    o = Oracle(ENV, max_nodes=1)
    # A level needing >1 expansion: budget hit -> not exact, but cached.
    from pathlib import Path
    import json
    raw = json.loads((Path(__file__).resolve().parents[2] / "fixtures" / "levels.json").read_text())[1]
    level = level_from_dict(raw)
    state = ENV.initial_state(level)
    v = o.value(state)
    assert v.exact is False
    assert v.solvable is None
    assert len(o._result_cache) == 1
    v2 = o.value(state)
    assert v2 == v


def test_analysis_does_not_search_successors_after_root_exhaustion():
    o = Oracle(ENV, max_nodes=1)
    state = ENV.initial_state(TWO)

    analysis = o.analyze(state)

    assert analysis.exact is False
    assert analysis.all_successors_exact is False
    assert len(analysis.actions) == len(ENV.legal_actions(state))
    assert all(not action.successor_exact for action in analysis.actions)
    # Only the root A* result is cached. Searching every successor cannot make
    # an exact policy target possible after the root itself is unknown.
    assert len(o._result_cache) == 1
