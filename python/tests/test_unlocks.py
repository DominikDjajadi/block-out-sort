"""Frozen blocks, locked gates, and locked regions before/after thresholds."""

from __future__ import annotations

from blocksort import Action, Block, Cell, Direction, Environment, level_from_dict
from blocksort.state import State

ENV = Environment()


def block(color, cells, unlock_at=0):
    return Block(color, frozenset(Cell(r, c) for r, c in cells), unlock_at)


LEVEL = level_from_dict({
    "name": "unlocks",
    "cols": 5,
    "rows": 5,
    "blocks": [],
    "exits": [
        {"edge": "top", "start": 0, "length": 2, "color": "red"},
        {"edge": "bottom", "start": 0, "length": 2, "color": "green", "unlockAt": 1},
    ],
    "lockedRegions": [{"cells": [[2, 2], [2, 3]], "unlockAt": 2}],
})


def test_frozen_block_before_threshold_has_no_actions():
    s = State.start(LEVEL, (
        block("red", [[0, 1]]),
        block("blue", [[2, 0]], unlock_at=1),
    ))
    frozen = s.blocks[1]
    assert ENV.is_block_frozen(s, frozen)
    assert all(s.blocks[a.block_index] is not frozen for a in ENV.legal_actions(s))


def test_frozen_block_thaws_after_threshold():
    s = State.start(LEVEL, (
        block("red", [[0, 1]]),
        block("blue", [[2, 0]], unlock_at=1),
    ))
    s2 = ENV.apply_action(s, Action(0, Direction.UP, 0, exit=True))
    assert s2.cleared == 1
    blue = s2.blocks[0]
    assert not ENV.is_block_frozen(s2, blue)
    assert any(s2.blocks[a.block_index] is blue for a in ENV.legal_actions(s2))


def test_locked_gate_is_inactive_before_threshold():
    s = State.start(LEVEL, (block("green", [[4, 1]]),))
    green = s.blocks[0]
    result = ENV.compute_slide(s, green, Direction.DOWN)
    assert result.reason == "edge"
    assert result.can_exit is False  # green gate locked until cleared >= 1


def test_locked_gate_activates_after_threshold():
    s = State.start(LEVEL, (
        block("red", [[0, 1]]),
        block("green", [[4, 1]]),
    ))
    s2 = ENV.apply_action(s, Action(0, Direction.UP, 0, exit=True))
    green = s2.blocks[0]
    result = ENV.compute_slide(s2, green, Direction.DOWN)
    assert result.can_exit is True


def test_locked_region_acts_as_wall_before_threshold():
    s = State.start(LEVEL, (block("blue", [[2, 1]]),))
    blue = s.blocks[0]
    result = ENV.compute_slide(s, blue, Direction.RIGHT)
    assert result.reason == "block"
    assert result.steps == 0  # (2,2) is a locked region cell


def test_locked_region_clears_after_threshold():
    s = State.start(LEVEL, (
        block("red", [[0, 0]]),
        block("red", [[0, 1]]),
        block("blue", [[2, 1]]),
    ))
    # Clear two reds so cleared >= 2 and the region unlocks.
    s = ENV.apply_action(s, Action(0, Direction.UP, 0, exit=True))
    s = ENV.apply_action(s, Action(0, Direction.UP, 0, exit=True))
    assert s.cleared == 2
    blue = s.blocks[0]
    result = ENV.compute_slide(s, blue, Direction.RIGHT)
    assert result.steps == 3  # now slides freely to the right edge (2,4)
