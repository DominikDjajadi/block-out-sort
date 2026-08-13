"""Schema, serialization, and validation tests."""

from __future__ import annotations

from blocksort import (
    Block,
    Cell,
    Direction,
    Exit,
    Level,
    LockedRegion,
    canonical_key,
    dumps_level,
    level_from_dict,
    level_to_dict,
    loads_level,
    validate_level,
    validate_level_data,
)
from blocksort.state import State
from blocksort.validation import is_valid


def test_direction_deltas_and_edges():
    assert Direction.UP.delta == (-1, 0)
    assert Direction.DOWN.delta == (1, 0)
    assert Direction.LEFT.delta == (0, -1)
    assert Direction.RIGHT.delta == (0, 1)
    assert Direction.UP.edge == "top"
    assert Direction.DOWN.edge == "bottom"
    assert Direction.LEFT.edge == "left"
    assert Direction.RIGHT.edge == "right"
    assert Direction.from_name("up") is Direction.UP


def test_cell_is_ordered_and_hashable():
    assert sorted([Cell(2, 1), Cell(0, 3), Cell(0, 1)]) == [
        Cell(0, 1),
        Cell(0, 3),
        Cell(2, 1),
    ]
    assert len({Cell(1, 1), Cell(1, 1)}) == 1


def test_block_translate_and_spans():
    block = Block("red", frozenset({Cell(1, 1), Cell(1, 2), Cell(2, 1)}))
    moved = block.translate(Direction.DOWN, 2)
    assert moved.cells == frozenset({Cell(3, 1), Cell(3, 2), Cell(4, 1)})
    assert block.columns() == frozenset({1, 2})
    assert block.rows() == frozenset({1, 2})
    assert moved.sorted_cells() == (Cell(3, 1), Cell(3, 2), Cell(4, 1))


SAMPLE = {
    "name": "Example",
    "cols": 6,
    "rows": 6,
    "holes": [[0, 0]],
    "blocks": [{"color": "red", "cells": [[2, 1], [2, 2]], "unlockAt": 0}],
    "exits": [{"edge": "top", "start": 1, "length": 2, "color": "red", "unlockAt": 0}],
    "lockedRegions": [{"cells": [[3, 3]], "unlockAt": 2}],
}


def test_serialization_round_trip_preserves_camelcase():
    level = level_from_dict(SAMPLE)
    data = level_to_dict(level)
    assert data["cols"] == 6 and data["rows"] == 6
    assert data["holes"] == [[0, 0]]
    assert data["lockedRegions"][0]["unlockAt"] == 2
    # Re-parsing yields an equivalent level.
    assert level_from_dict(data) == level
    assert loads_level(dumps_level(level)) == level


def test_canonical_key_includes_color_shape_unlock_and_coords():
    level = level_from_dict(SAMPLE)
    state = State.start(level)
    key = canonical_key(state)
    assert key == "red#0#2,1;2,2"


def test_canonical_key_is_order_independent_for_interchangeable_blocks():
    level = Level(name="x", cols=4, rows=4, exits=())
    a = Block("red", frozenset({Cell(0, 0), Cell(0, 1)}))
    b = Block("red", frozenset({Cell(3, 0), Cell(3, 1)}))
    s1 = State(level=level, blocks=(a, b), total_blocks=2)
    s2 = State(level=level, blocks=(b, a), total_blocks=2)
    assert canonical_key(s1) == canonical_key(s2)


def test_canonical_key_distinguishes_unlock_threshold():
    level = Level(name="x", cols=4, rows=4, exits=())
    a = Block("red", frozenset({Cell(0, 0)}), unlock_at=0)
    b = Block("red", frozenset({Cell(0, 0)}), unlock_at=2)
    s1 = State(level=level, blocks=(a,), total_blocks=1)
    s2 = State(level=level, blocks=(b,), total_blocks=1)
    assert canonical_key(s1) != canonical_key(s2)


# ----- validation -----

def _valid_level_dict():
    return {
        "name": "ok",
        "cols": 5,
        "rows": 5,
        "blocks": [{"color": "red", "cells": [[0, 0], [0, 1]]}],
        "exits": [{"edge": "top", "start": 0, "length": 2, "color": "red"}],
    }


def test_valid_level_has_no_errors():
    assert is_valid(level_from_dict(_valid_level_dict()))


def _codes(level: Level) -> set[str]:
    return {e.code for e in validate_level(level)}


def test_invalid_dimensions():
    level = Level(name="x", cols=0, rows=-1, blocks=(), exits=())
    assert "dims" in _codes(level)


def test_out_of_bounds_block_cell():
    d = _valid_level_dict()
    d["blocks"][0]["cells"] = [[0, 0], [0, 9]]
    assert "block_bounds" in _codes(level_from_dict(d))


def test_duplicate_cells_detected_at_dict_layer():
    d = _valid_level_dict()
    d["blocks"][0]["cells"] = [[0, 0], [0, 0]]
    codes = {e.code for e in validate_level_data(d)}
    assert "duplicate_cell" in codes


def test_overlapping_blocks():
    d = _valid_level_dict()
    d["blocks"].append({"color": "blue", "cells": [[0, 1]]})
    d["exits"].append({"edge": "left", "start": 0, "length": 1, "color": "blue"})
    assert "overlap" in _codes(level_from_dict(d))


def test_block_on_hole():
    d = _valid_level_dict()
    d["holes"] = [[0, 0]]
    assert "block_on_hole" in _codes(level_from_dict(d))


def test_block_on_initially_locked_region():
    d = _valid_level_dict()
    d["lockedRegions"] = [{"cells": [[0, 0]], "unlockAt": 1}]
    assert "block_on_locked" in _codes(level_from_dict(d))


def test_invalid_gate_edge():
    d = _valid_level_dict()
    d["exits"][0]["edge"] = "sideways"
    assert "exit_edge" in _codes(level_from_dict(d))


def test_gate_extends_past_edge():
    d = _valid_level_dict()
    d["exits"][0] = {"edge": "top", "start": 4, "length": 3, "color": "red"}
    assert "exit_overflow" in _codes(level_from_dict(d))


def test_invalid_gate_length():
    d = _valid_level_dict()
    d["exits"][0]["length"] = 0
    assert "exit_length" in _codes(level_from_dict(d))


def test_unknown_colors():
    d = _valid_level_dict()
    d["blocks"][0]["color"] = "chartreuse"
    assert "color" in _codes(level_from_dict(d))


def test_invalid_unlock_threshold_on_block():
    d = _valid_level_dict()
    d["blocks"][0]["unlockAt"] = 5  # >= total blocks (1)
    assert "block_unlock" in _codes(level_from_dict(d))


def test_disconnected_block():
    d = _valid_level_dict()
    d["blocks"][0]["cells"] = [[0, 0], [0, 2]]
    assert "disconnected" in _codes(level_from_dict(d))


def test_no_blocks_or_no_exits():
    empty = Level(name="x", cols=4, rows=4, blocks=(), exits=())
    codes = _codes(empty)
    assert "no_blocks" in codes
    assert "no_exits" in codes


def test_block_too_wide_for_any_matching_gate():
    d = _valid_level_dict()
    # 3-wide block, but the only red gate is length 2.
    d["blocks"][0]["cells"] = [[0, 0], [0, 1], [0, 2]]
    assert "gate_too_narrow" in _codes(level_from_dict(d))


def test_validation_returns_multiple_errors_at_once():
    d = {
        "name": "broken",
        "cols": 4,
        "rows": 4,
        "blocks": [{"color": "nope", "cells": [[9, 9]]}],
        "exits": [{"edge": "weird", "start": 0, "length": 0, "color": "red"}],
    }
    codes = {e.code for e in validate_level_data(d)}
    assert {"color", "block_bounds", "exit_edge", "exit_length"} <= codes
