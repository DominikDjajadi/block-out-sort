"""The corrected full-extrusion exit rule for irregular polyominoes.

These tests pin the behavior the JavaScript engine was patched to match: a block
may only exit if it can fully extrude off the board with every still-on-board
cell remaining collision-free.
"""

from __future__ import annotations

import pytest

from blocksort import Action, Block, Cell, Direction, Environment, IllegalActionError, level_from_dict
from blocksort.state import State

ENV = Environment()


def block(color, cells, unlock_at=0):
    return Block(color, frozenset(Cell(r, c) for r, c in cells), unlock_at)


# Top gate over columns 1-2; the L-piece's foot at (2,2) must extrude through
# column 2 while the leading cells are already off-board.
TOP_GATE = level_from_dict({
    "name": "extrude",
    "cols": 4,
    "rows": 4,
    "blocks": [],
    "exits": [{"edge": "top", "start": 1, "length": 2, "color": "red"}],
})

L_PIECE = [[0, 1], [1, 1], [2, 1], [2, 2]]


def test_trailing_cell_collision_against_block_forbids_exit():
    s = State.start(TOP_GATE, (
        block("red", L_PIECE),
        block("blue", [[0, 2]]),  # sits in the foot's extrusion path
    ))
    red = s.blocks[0]
    result = ENV.compute_slide(s, red, Direction.UP)
    assert result.reason == "edge"
    assert result.can_exit is False
    # And there is no legal up-exit action for red.
    assert not any(
        a.exit and s.blocks[a.block_index] is red and a.direction == Direction.UP
        for a in ENV.legal_actions(s)
    )
    with pytest.raises(IllegalActionError):
        ENV.apply_action(s, Action(0, Direction.UP, result.steps, exit=True))


def test_clean_extrusion_succeeds():
    s = State.start(TOP_GATE, (block("red", L_PIECE),))
    red = s.blocks[0]
    result = ENV.compute_slide(s, red, Direction.UP)
    assert result.can_exit is True
    s2 = ENV.apply_action(s, Action(0, Direction.UP, result.steps, exit=True))
    assert ENV.is_terminal(s2)


def test_trailing_cell_collision_against_hole_forbids_exit():
    level = level_from_dict({
        "name": "extrude-hole",
        "cols": 4,
        "rows": 4,
        "holes": [[0, 2]],
        "blocks": [],
        "exits": [{"edge": "top", "start": 1, "length": 2, "color": "red"}],
    })
    s = State.start(level, (block("red", L_PIECE),))
    red = s.blocks[0]
    assert ENV.compute_slide(s, red, Direction.UP).can_exit is False


def test_trailing_cell_collision_against_locked_region_forbids_exit():
    level = level_from_dict({
        "name": "extrude-region",
        "cols": 4,
        "rows": 4,
        "blocks": [],
        "exits": [{"edge": "top", "start": 1, "length": 2, "color": "red"}],
        "lockedRegions": [{"cells": [[0, 2]], "unlockAt": 3}],
    })
    s = State.start(level, (block("red", L_PIECE),))
    red = s.blocks[0]
    assert ENV.compute_slide(s, red, Direction.UP).can_exit is False


def test_extrusion_collision_only_along_the_exit_path():
    # A block in column 2 *below* the L-foot's path does NOT block the exit,
    # since the foot extrudes upward and never reaches it.
    s = State.start(TOP_GATE, (
        block("red", L_PIECE),
        block("blue", [[3, 2]]),  # below the piece, out of the upward path
    ))
    red = s.blocks[0]
    assert ENV.compute_slide(s, red, Direction.UP).can_exit is True
