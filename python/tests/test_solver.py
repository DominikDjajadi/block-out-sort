"""A* / BFS solver tests, including optimality agreement and replay."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from blocksort import (
    Block,
    Cell,
    Environment,
    Exit,
    Level,
    SolveResult,
    level_from_dict,
    solve_astar,
    solve_bfs,
    verify_solution,
)
from blocksort.solver import (
    DEPTH_LIMIT,
    NODE_LIMIT,
    SOLVED,
    UNSOLVABLE,
)
from blocksort.state import State

ENV = Environment()
REPO_ROOT = Path(__file__).resolve().parents[2]


def block(color, cells, unlock_at=0):
    return Block(color, frozenset(Cell(r, c) for r, c in cells), unlock_at)


def make_level(**kw):
    return level_from_dict(kw)


# ---- basic cases ----

def test_empty_board_is_solved_in_zero_moves():
    level = make_level(name="e", cols=3, rows=3, blocks=[],
                       exits=[{"edge": "top", "start": 0, "length": 1, "color": "red"}])
    state = ENV.initial_state(level)
    r = solve_astar(ENV, state)
    assert r.solvable is True and r.move_count == 0 and r.actions == ()
    assert r.termination_reason == SOLVED


def test_single_directly_exitable_block():
    level = make_level(name="x", cols=3, rows=3,
                       blocks=[{"color": "red", "cells": [[1, 1]]}],
                       exits=[{"edge": "top", "start": 1, "length": 1, "color": "red"}])
    state = ENV.initial_state(level)
    r = solve_astar(ENV, state)
    assert r.solvable is True and r.move_count == 1
    assert verify_solution(ENV, state, r.actions, 1)


def test_intermediate_stop_then_exit():
    # Red must move right to align under the gate, then exit up.
    level = make_level(
        name="x", cols=4, rows=4,
        blocks=[{"color": "red", "cells": [[2, 0]]}],
        exits=[{"edge": "top", "start": 2, "length": 1, "color": "red"}],
    )
    state = ENV.initial_state(level)
    r = solve_astar(ENV, state)
    assert r.solvable is True and r.move_count == 2
    assert verify_solution(ENV, state, r.actions, r.move_count)


def test_unsolvable_level_is_proven_not_exhausted():
    # Block can never reach its gate color (no matching gate).
    level = Level(name="u", cols=3, rows=3,
                  blocks=(block("red", [[1, 1]]),),
                  exits=(Exit("top", 0, 1, "blue", 0),))
    state = ENV.initial_state(level)
    r = solve_astar(ENV, state)
    assert r.solvable is False
    assert r.exhausted is False
    assert r.termination_reason == UNSOLVABLE
    assert r.move_count is None


def test_node_limit_reports_unknown_not_unsolvable():
    raw = json.loads((REPO_ROOT / "fixtures" / "levels.json").read_text())[1]
    level = level_from_dict(raw)
    state = ENV.initial_state(level)
    r = solve_astar(ENV, state, max_nodes=1)
    assert r.exhausted is True
    assert r.solvable is None  # never reported as unsolvable
    assert r.termination_reason == NODE_LIMIT


def test_depth_limit_reports_unknown():
    raw = json.loads((REPO_ROOT / "fixtures" / "levels.json").read_text())[0]
    level = level_from_dict(raw)
    state = ENV.initial_state(level)
    r = solve_astar(ENV, state, max_depth=1)  # needs 4 moves
    assert r.exhausted is True
    assert r.solvable is None
    assert r.termination_reason == DEPTH_LIMIT


# ---- mechanics ----

def test_frozen_locked_levels_solvable():
    raw = json.loads((REPO_ROOT / "fixtures" / "levels.json").read_text())[2]
    level = level_from_dict(raw)
    state = ENV.initial_state(level)
    r = solve_astar(ENV, state)
    assert r.solvable is True
    assert verify_solution(ENV, state, r.actions, r.move_count)


def test_irregular_shapes_level_solvable():
    raw = json.loads((REPO_ROOT / "fixtures" / "levels.json").read_text())[3]
    level = level_from_dict(raw)
    state = ENV.initial_state(level)
    r = solve_astar(ENV, state)
    assert r.solvable is True
    assert verify_solution(ENV, state, r.actions, r.move_count)


def test_multiple_optimal_solutions_still_optimal():
    # Two independent blocks each exiting their own gate: 2 moves, many orders.
    level = make_level(
        name="x", cols=5, rows=5,
        blocks=[{"color": "red", "cells": [[0, 0]]},
                {"color": "blue", "cells": [[4, 4]]}],
        exits=[{"edge": "top", "start": 0, "length": 1, "color": "red"},
               {"edge": "bottom", "start": 4, "length": 1, "color": "blue"}],
    )
    state = ENV.initial_state(level)
    r = solve_astar(ENV, state)
    b = solve_bfs(ENV, state)
    assert r.move_count == 2 == b.move_count


# ---- A*/BFS agreement ----

@pytest.mark.parametrize("path,index", [
    ("fixtures/levels.json", 0),
    ("fixtures/levels.json", 1),
    ("fixtures/levels.json", 2),
    ("fixtures/levels.json", 3),
])
def test_astar_bfs_agree_on_levels(path, index):
    raw = json.loads((REPO_ROOT / path).read_text())[index]
    level = level_from_dict(raw)
    state = ENV.initial_state(level)
    a = solve_astar(ENV, state)
    b = solve_bfs(ENV, state)
    assert a.solvable == b.solvable
    assert a.move_count == b.move_count
    assert verify_solution(ENV, state, a.actions, a.move_count)
    assert verify_solution(ENV, state, b.actions, b.move_count)


def test_astar_solves_and_verifies_generated_levels():
    # A* must solve and replay-verify every generated level. BFS is a small
    # reference: it is only required to *agree* on levels it can finish within a
    # modest budget (it is intentionally unoptimized and exhausts on hard ones).
    gen_path = REPO_ROOT / "fixtures" / "generated_levels.json"
    levels = json.loads(gen_path.read_text())
    for raw in levels:
        level = level_from_dict(raw)
        state = ENV.initial_state(level)
        a = solve_astar(ENV, state)
        assert a.solvable is True, raw["name"]
        assert verify_solution(ENV, state, a.actions, a.move_count)
        b = solve_bfs(ENV, state, max_nodes=30_000)
        if not b.exhausted:
            assert a.move_count == b.move_count, raw["name"]
