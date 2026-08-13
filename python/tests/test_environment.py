"""Core environment / transition tests."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from blocksort import (
    Action,
    Block,
    Cell,
    Direction,
    Environment,
    IllegalActionError,
    Level,
    level_from_dict,
)
from blocksort.state import State
from blocksort.validation import is_valid

ENV = Environment()
REPO_ROOT = Path(__file__).resolve().parents[2]


def block(color, cells, unlock_at=0):
    return Block(color, frozenset(Cell(r, c) for r, c in cells), unlock_at)


def state(level, blocks):
    return State.start(level, tuple(blocks))


SLIDE_LEVEL = level_from_dict({
    "name": "slide",
    "cols": 5,
    "rows": 5,
    "holes": [[2, 4]],
    "blocks": [],
    "exits": [{"edge": "top", "start": 0, "length": 2, "color": "red"}],
})


def test_slide_until_another_block():
    s = state(SLIDE_LEVEL, [block("red", [[2, 1]]), block("blue", [[2, 3]])])
    red = s.blocks[0]
    result = ENV.compute_slide(s, red, Direction.RIGHT)
    assert result.steps == 1  # stops one short of the blue block at (2,3)
    assert result.reason == "block"


def test_slide_until_hole():
    s = state(SLIDE_LEVEL, [block("red", [[2, 2]])])
    red = s.blocks[0]
    result = ENV.compute_slide(s, red, Direction.RIGHT)
    assert result.steps == 1  # hole at (2,4) stops it after reaching (2,3)
    assert result.reason == "block"


def test_intermediate_stopping_positions_are_all_legal():
    s = state(SLIDE_LEVEL, [block("red", [[2, 0]])])
    red = s.blocks[0]
    right = [
        a for a in ENV.legal_actions(s)
        if a.direction == Direction.RIGHT and not a.exit
    ]
    # Can stop at every cell from 1..3 (hole at (2,4) blocks the 4th).
    assert sorted(a.distance for a in right) == [1, 2, 3]


def test_matching_gate_allows_exit():
    s = state(SLIDE_LEVEL, [block("red", [[2, 1]])])
    red = s.blocks[0]
    result = ENV.compute_slide(s, red, Direction.UP)
    assert result.reason == "edge"
    assert result.can_exit is True


def test_nonmatching_gate_blocks_exit():
    s = state(SLIDE_LEVEL, [block("blue", [[2, 1]])])
    blue = s.blocks[0]
    result = ENV.compute_slide(s, blue, Direction.UP)
    assert result.reason == "edge"
    assert result.can_exit is False  # gate is red, block is blue


def test_unsolved_state_with_no_actions_is_an_explicit_deadlock():
    s = state(SLIDE_LEVEL, [block("red", [[2, 1]], unlock_at=1)])
    assert not ENV.is_terminal(s)
    assert ENV.legal_actions(s) == []
    assert ENV.is_deadlock(s)


def test_gate_too_narrow_for_block():
    level = level_from_dict({
        "name": "narrow",
        "cols": 5,
        "rows": 5,
        "blocks": [],
        "exits": [{"edge": "top", "start": 1, "length": 1, "color": "red"}],
    })
    s = state(level, [block("red", [[1, 1], [1, 2]])])  # spans columns 1 and 2
    red = s.blocks[0]
    assert ENV.compute_slide(s, red, Direction.UP).can_exit is False


def test_arbitrary_irregular_shape_slides():
    # A plus-pentomino slides down until the bottom row stops at the edge.
    plus = block("purple", [[0, 1], [1, 0], [1, 1], [1, 2], [2, 1]])
    level = level_from_dict({
        "name": "shape",
        "cols": 5,
        "rows": 5,
        "blocks": [],
        "exits": [{"edge": "left", "start": 0, "length": 1, "color": "purple"}],
    })
    s = state(level, [plus])
    result = ENV.compute_slide(s, s.blocks[0], Direction.DOWN)
    assert result.reason == "edge"
    assert result.steps == 2  # bottom cell at row 2 -> rows 2..4 used


def test_apply_illegal_slide_distance_raises():
    s = state(SLIDE_LEVEL, [block("red", [[2, 1]]), block("blue", [[2, 3]])])
    bad = Action(0, Direction.RIGHT, 2, exit=False)  # max legal is 1
    with pytest.raises(IllegalActionError):
        ENV.apply_action(s, bad)


def test_apply_illegal_exit_raises():
    s = state(SLIDE_LEVEL, [block("blue", [[2, 1]])])
    bad = Action(0, Direction.UP, 2, exit=True)  # blue cannot exit a red gate
    with pytest.raises(IllegalActionError):
        ENV.apply_action(s, bad)


def test_apply_out_of_range_index_raises():
    s = state(SLIDE_LEVEL, [block("red", [[2, 1]])])
    with pytest.raises(IllegalActionError):
        ENV.apply_action(s, Action(5, Direction.UP, 1, exit=False))


def test_removing_block_updates_derived_cleared_count():
    s = state(SLIDE_LEVEL, [block("red", [[2, 1]]), block("red", [[3, 1]])])
    assert s.cleared == 0
    exit_action = Action(0, Direction.UP, 2, exit=True)
    s2 = ENV.apply_action(s, exit_action)
    assert s2.remaining == 1
    assert s2.cleared == 1
    assert s2.total_blocks == 2


def test_winning_after_final_exit():
    s = state(SLIDE_LEVEL, [block("red", [[2, 1]])])
    assert not ENV.is_terminal(s)
    s2 = ENV.apply_action(s, Action(0, Direction.UP, 2, exit=True))
    assert ENV.is_terminal(s2)
    assert s2.cleared == 1


def test_handcrafted_levels_load_and_validate():
    data = json.loads((REPO_ROOT / "fixtures" / "levels.json").read_text())
    assert len(data) >= 4
    for raw in data:
        level = level_from_dict(raw)
        assert level.total_blocks > 0
        # The shipped levels must be structurally valid.
        assert is_valid(level), f"{level.name} failed validation"
        # And the initial state must be constructible with legal actions.
        s = ENV.initial_state(level)
        assert ENV.legal_actions(s)
