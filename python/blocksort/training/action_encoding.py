"""Fixed, anchor-based action encoding.

An action is identified by ``(anchor_row, anchor_col, direction, move_code)``
where the anchor is the block's lexicographically smallest cell *before* the
move. This never uses runtime block IDs, so it is stable across serialization
and block reordering. See ``docs/neural_encoding.md`` for the full spec.
"""

from __future__ import annotations

from typing import Any, Iterable

import torch

from ..actions import Action
from ..environment import Environment
from ..schema import Cell, Direction
from ..state import State
from .config import DIRECTION_ORDER, EncodingConfig, EncodingError

_DIR_INDEX: dict[str, int] = {name: i for i, name in enumerate(DIRECTION_ORDER)}


def direction_index(direction: Direction | str) -> int:
    name = direction.value if isinstance(direction, Direction) else str(direction)
    if name not in _DIR_INDEX:
        raise EncodingError(f"unknown direction: {name!r}")
    return _DIR_INDEX[name]


def direction_from_index(index: int) -> Direction:
    if not 0 <= index < len(DIRECTION_ORDER):
        raise EncodingError(f"direction index out of range: {index}")
    return Direction(DIRECTION_ORDER[index])


def anchor_of_cells(cells: Iterable[Cell] | Iterable[tuple[int, int]]) -> tuple[int, int]:
    """Canonical anchor = lexicographically smallest ``(r, c)`` cell."""
    best: tuple[int, int] | None = None
    for cell in cells:
        if isinstance(cell, Cell):
            rc = (cell.r, cell.c)
        else:
            rc = (int(cell[0]), int(cell[1]))
        if best is None or rc < best:
            best = rc
    if best is None:
        raise EncodingError("cannot take anchor of an empty block")
    return best


def _move_code_index(distance: int, is_exit: bool, config: EncodingConfig) -> int:
    if is_exit:
        return config.exit_move_code
    if not 1 <= distance <= config.max_slide_distance:
        raise EncodingError(
            f"slide distance {distance} exceeds max_slide_distance "
            f"{config.max_slide_distance}"
        )
    return distance - 1


def index_from_parts(
    anchor_row: int, anchor_col: int, dir_index: int, move_code_index: int,
    config: EncodingConfig,
) -> int:
    if not 0 <= anchor_row < config.max_rows:
        raise EncodingError(f"anchor_row {anchor_row} out of range")
    if not 0 <= anchor_col < config.max_cols:
        raise EncodingError(f"anchor_col {anchor_col} out of range")
    m = config.move_code_count
    return (((anchor_row * config.max_cols + anchor_col) * 4 + dir_index) * m
            + move_code_index)


def parts_from_index(index: int, config: EncodingConfig) -> tuple[int, int, int, int]:
    if not 0 <= index < config.action_space_size:
        raise EncodingError(f"action index {index} out of range")
    m = config.move_code_count
    move_code_index = index % m
    rest = index // m
    dir_index = rest % 4
    rest //= 4
    anchor_col = rest % config.max_cols
    anchor_row = rest // config.max_cols
    return anchor_row, anchor_col, dir_index, move_code_index


def action_index(state: State, action: Action, config: EncodingConfig) -> int:
    """Fixed index of a legal/serializable ``action`` taken in ``state``."""
    block = state.blocks[action.block_index]
    ar, ac = anchor_of_cells(block.cells)
    di = direction_index(action.direction)
    mc = _move_code_index(action.distance, action.exit, config)
    return index_from_parts(ar, ac, di, mc, config)


def normalized_action_index(norm: dict[str, Any], config: EncodingConfig) -> int:
    """Index from a dataset's normalized action dict (color/cells/dir/...)."""
    ar, ac = anchor_of_cells(norm["cells"])
    di = direction_index(norm["dir"])
    mc = _move_code_index(int(norm["distance"]), bool(norm["exit"]), config)
    return index_from_parts(ar, ac, di, mc, config)


def decode_action(
    env: Environment, state: State, index: int, config: EncodingConfig
) -> Action:
    """Resolve a fixed action index to a concrete :class:`Action` in ``state``.

    Locates the block by anchor; for EXIT, the exit distance is recomputed from
    the environment slide so the produced action matches engine expectations.
    """
    anchor_row, anchor_col, dir_index, move_code_index = parts_from_index(index, config)
    direction = direction_from_index(dir_index)

    block_index = None
    for i, block in enumerate(state.blocks):
        if anchor_of_cells(block.cells) == (anchor_row, anchor_col):
            block_index = i
            break
    if block_index is None:
        raise EncodingError(
            f"no block with anchor ({anchor_row}, {anchor_col}) in state"
        )

    if move_code_index == config.exit_move_code:
        result = env.compute_slide(state, state.blocks[block_index], direction)
        return Action(block_index, direction, result.steps, exit=True)
    distance = move_code_index + 1
    return Action(block_index, direction, distance, exit=False)


def build_legal_action_mask(
    env: Environment,
    state: State,
    config: EncodingConfig,
    legal_actions: Iterable[Action] | None = None,
) -> torch.Tensor:
    """A ``[A]`` float mask (1 = legal) over the fixed action space."""
    mask = torch.zeros(config.action_space_size, dtype=torch.float32)
    actions = env.legal_actions(state) if legal_actions is None else legal_actions
    for action in actions:
        mask[action_index(state, action, config)] = 1.0
    return mask


def encode_sparse_policy_target(
    record: dict[str, Any], config: EncodingConfig
) -> torch.Tensor:
    """Dense ``[A]`` policy target from a dataset record's sparse legal targets."""
    target = torch.zeros(config.action_space_size, dtype=torch.float32)
    legal = record["legal_actions"]
    probs = record["policy_target"]
    if len(legal) != len(probs):
        raise EncodingError("legal_actions and policy_target length mismatch")
    for norm, prob in zip(legal, probs):
        idx = normalized_action_index(norm, config)
        if target[idx] != 0.0:
            raise EncodingError(f"policy target index collision at {idx}")
        target[idx] = float(prob)
    return target


def legal_mask_from_record(
    record: dict[str, Any], config: EncodingConfig
) -> torch.Tensor:
    """Legal-action mask built directly from a record's stored legal actions."""
    mask = torch.zeros(config.action_space_size, dtype=torch.float32)
    for norm in record["legal_actions"]:
        mask[normalized_action_index(norm, config)] = 1.0
    return mask
