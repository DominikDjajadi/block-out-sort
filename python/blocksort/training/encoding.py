"""Deterministic, order-invariant tensor encoding of a game state.

Channel-first board tensor ``[C, max_rows, max_cols]`` plus a small global
feature vector and a valid-cell mask. See ``docs/neural_encoding.md`` for the
authoritative channel map. Nothing here depends on runtime block identity.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from ..environment import Environment
from ..schema import Cell, Direction
from ..state import State
from .config import EncodingConfig, EncodingError

_EDGE_TO_DIR_INDEX = {"top": 0, "bottom": 1, "left": 2, "right": 3}


@dataclass(frozen=True)
class EncodedState:
    board: torch.Tensor          # [C, max_rows, max_cols]
    global_features: torch.Tensor  # [num_global_features]
    valid_cell_mask: torch.Tensor  # [max_rows, max_cols]


class _ChannelMap:
    """Resolves channel offsets for a given color count (see the doc).

    Gates are encoded per direction so a corner cell carrying gates of different
    colors and directions stays fully distinguishable. Each direction block has
    ``num_colors`` color planes followed by active/locked/remaining planes.
    """

    def __init__(self, config: EncodingConfig) -> None:
        k = config.num_colors
        self.k = k
        self.stride = config.gate_dir_stride  # k + 3
        self.board_extent = 0
        self.hole = 1
        self.locked_now = 2
        self.locked_remaining = 3
        self.occupancy0 = 4
        self.anchor = 4 + k
        self.connect_up = self.anchor + 1
        self.connect_down = self.anchor + 2
        self.connect_left = self.anchor + 3
        self.connect_right = self.anchor + 4
        self.frozen_now = self.anchor + 5
        self.frozen_remaining = self.anchor + 6
        self.gate0 = self.frozen_remaining + 1
        self.cleared_plane = self.gate0 + 4 * self.stride
        self.remaining_plane = self.cleared_plane + 1
        self.total = self.remaining_plane + 1

    def gate_color(self, dir_idx: int, color_idx: int) -> int:
        return self.gate0 + dir_idx * self.stride + color_idx

    def gate_active(self, dir_idx: int) -> int:
        return self.gate0 + dir_idx * self.stride + self.k

    def gate_locked(self, dir_idx: int) -> int:
        return self.gate0 + dir_idx * self.stride + self.k + 1

    def gate_remaining(self, dir_idx: int) -> int:
        return self.gate0 + dir_idx * self.stride + self.k + 2


def _connect_offset(channels: _ChannelMap, direction: Direction) -> int:
    return {
        Direction.UP: channels.connect_up,
        Direction.DOWN: channels.connect_down,
        Direction.LEFT: channels.connect_left,
        Direction.RIGHT: channels.connect_right,
    }[direction]


def _gate_border_cells(exit_, rows: int, cols: int) -> list[tuple[int, int]]:
    cells: list[tuple[int, int]] = []
    lo = exit_.start
    hi = exit_.start + exit_.length
    if exit_.edge in ("top", "bottom"):
        r = 0 if exit_.edge == "top" else rows - 1
        for c in range(max(0, lo), min(cols, hi)):
            cells.append((r, c))
    else:
        c = 0 if exit_.edge == "left" else cols - 1
        for r in range(max(0, lo), min(rows, hi)):
            cells.append((r, c))
    return cells


def encode_state(
    env: Environment, state: State, config: EncodingConfig
) -> EncodedState:
    """Encode ``state`` into board / global / mask tensors."""
    level = state.level
    rows, cols = level.rows, level.cols
    if rows > config.max_rows or cols > config.max_cols:
        raise EncodingError(
            f"board {rows}x{cols} exceeds encoding limit "
            f"{config.max_rows}x{config.max_cols}"
        )
    if state.total_blocks > config.max_blocks:
        raise EncodingError(
            f"state total block count {state.total_blocks} exceeds encoding "
            f"limit {config.max_blocks}"
        )

    color_index = {name: i for i, name in enumerate(config.colors)}
    ch = _ChannelMap(config)
    H, W = config.max_rows, config.max_cols
    board = torch.zeros(ch.total, H, W, dtype=torch.float32)
    mask = torch.zeros(H, W, dtype=torch.float32)
    norm = float(config.max_blocks)
    cleared = state.cleared
    remaining = state.remaining

    # ----- structure -----
    for r in range(rows):
        for c in range(cols):
            board[ch.board_extent, r, c] = 1.0
            mask[r, c] = 1.0
            board[ch.cleared_plane, r, c] = cleared / norm
            board[ch.remaining_plane, r, c] = remaining / norm
    for hole in level.holes:
        if 0 <= hole.r < rows and 0 <= hole.c < cols:
            board[ch.hole, hole.r, hole.c] = 1.0
    for region in level.locked_regions:
        if cleared < region.unlock_at:
            rem = max(0, region.unlock_at - cleared) / norm
            for cell in region.cells:
                if 0 <= cell.r < rows and 0 <= cell.c < cols:
                    board[ch.locked_now, cell.r, cell.c] = 1.0
                    board[ch.locked_remaining, cell.r, cell.c] = rem

    # ----- blocks (occupancy, anchor, connectivity, frozen) -----
    for block in state.blocks:
        if block.color not in color_index:
            raise EncodingError(f"unknown color: {block.color!r}")
        ci = color_index[block.color]
        frozen = env.is_block_frozen(state, block)
        frozen_rem = max(0, block.unlock_at - cleared) / norm
        anchor = min((cell.r, cell.c) for cell in block.cells)
        board[ch.anchor, anchor[0], anchor[1]] = 1.0
        for cell in block.cells:
            board[ch.occupancy0 + ci, cell.r, cell.c] = 1.0
            if frozen:
                board[ch.frozen_now, cell.r, cell.c] = 1.0
                board[ch.frozen_remaining, cell.r, cell.c] = frozen_rem
            for direction in Direction:
                nr, nc = cell.r + direction.dr, cell.c + direction.dc
                if Cell(nr, nc) in block.cells:
                    board[_connect_offset(ch, direction), cell.r, cell.c] = 1.0

    # ----- gates (per-direction planes; lossless at corners) -----
    for exit_ in level.exits:
        if exit_.color not in color_index:
            raise EncodingError(f"unknown gate color: {exit_.color!r}")
        gi = color_index[exit_.color]
        active = cleared >= exit_.unlock_at
        gate_rem = max(0, exit_.unlock_at - cleared) / norm
        dir_idx = _EDGE_TO_DIR_INDEX[exit_.edge]
        for (r, c) in _gate_border_cells(exit_, rows, cols):
            board[ch.gate_color(dir_idx, gi), r, c] = 1.0
            if active:
                board[ch.gate_active(dir_idx), r, c] = 1.0
            else:
                board[ch.gate_locked(dir_idx), r, c] = 1.0
                board[ch.gate_remaining(dir_idx), r, c] = max(
                    board[ch.gate_remaining(dir_idx), r, c], gate_rem)

    active_gates = sum(1 for e in level.exits if cleared >= e.unlock_at)
    global_features = torch.tensor([
        cleared / norm,
        remaining / norm,
        level.total_blocks / norm,
        rows / float(config.max_rows),
        cols / float(config.max_cols),
        active_gates / norm,
    ], dtype=torch.float32)

    return EncodedState(board=board, global_features=global_features,
                        valid_cell_mask=mask)


def channel_names(config: EncodingConfig) -> list[str]:
    """Human-readable channel names in order (for docs/debugging)."""
    names = ["board_extent", "hole", "locked_region_now", "locked_region_remaining"]
    names += [f"occupancy[{c}]" for c in config.colors]
    names += ["anchor", "connect_up", "connect_down", "connect_left", "connect_right"]
    names += ["frozen_now", "frozen_remaining"]
    for d in ("up", "down", "left", "right"):
        names += [f"gate_{d}_color[{c}]" for c in config.colors]
        names += [f"gate_{d}_active", f"gate_{d}_locked", f"gate_{d}_remaining"]
    names += ["cleared_plane", "remaining_plane"]
    return names
