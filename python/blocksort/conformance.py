"""Helpers for the language-neutral conformance fixtures.

Actions in fixtures are *normalized*: a block is identified by its color and its
current (sorted) cells rather than by an engine-specific index, so the same
fixture can be evaluated by the Python and JavaScript engines.
"""

from __future__ import annotations

from typing import Any

from .actions import Action
from .environment import Environment
from .schema import Block, Cell, Direction, Level
from .serialization import level_from_dict
from .state import State


def _cells_to_json(cells: frozenset[Cell]) -> list[list[int]]:
    return [[cell.r, cell.c] for cell in sorted(cells)]


def block_from_spec(spec: dict[str, Any]) -> Block:
    return Block(
        color=spec["color"],
        cells=frozenset(Cell(int(r), int(c)) for r, c in spec["cells"]),
        unlock_at=int(spec.get("unlockAt", 0) or 0),
    )


def build_state(env: Environment, level: Level, setup: dict | None) -> State:
    """Initial state for ``level``, or an override placement from ``setup``."""
    if setup and "blocks" in setup:
        blocks = tuple(block_from_spec(b) for b in setup["blocks"])
        return State.start(level, blocks)
    return env.initial_state(level)


def action_to_normalized(state: State, action: Action) -> dict[str, Any]:
    block = state.blocks[action.block_index]
    return {
        "color": block.color,
        "cells": _cells_to_json(block.cells),
        "dir": action.direction.value,
        "distance": action.distance,
        "exit": action.exit,
    }


def _normalized_hashable(norm: dict[str, Any]) -> tuple:
    return (
        norm["color"],
        tuple(tuple(c) for c in norm["cells"]),
        norm["dir"],
        norm["distance"],
        bool(norm["exit"]),
    )


def normalized_actions(env: Environment, state: State) -> list[dict[str, Any]]:
    """All legal actions as normalized dicts, deterministically sorted."""
    norms = [action_to_normalized(state, a) for a in env.legal_actions(state)]
    norms.sort(key=_normalized_hashable)
    return norms


def actions_equal(a: list[dict], b: list[dict]) -> bool:
    return {_normalized_hashable(x) for x in a} == {_normalized_hashable(y) for y in b}


def find_block_index(state: State, color: str, cells: list[list[int]]) -> int:
    target = frozenset(Cell(int(r), int(c)) for r, c in cells)
    for i, block in enumerate(state.blocks):
        if block.color == color and block.cells == target:
            return i
    raise KeyError(f"no block {color} at {cells}")


def normalized_to_action(state: State, norm: dict[str, Any]) -> Action:
    index = find_block_index(state, norm["color"], norm["cells"])
    return Action(
        block_index=index,
        direction=Direction(norm["dir"]),
        distance=int(norm["distance"]),
        exit=bool(norm["exit"]),
    )


def level_from_fixture(fixture: dict[str, Any]) -> Level:
    return level_from_dict(fixture["level"])
