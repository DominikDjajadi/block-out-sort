"""Fixed, maskable action space for the designer policy.

The designer's action space is intentionally small (the milestone allows starting
with reverse-slide + stop): a single ``STOP`` action plus one slot per
``(anchor_row, anchor_col, direction, distance)`` reverse slide. Anchors are a
block's identifying cell, so the index is stable under block reordering, and the
flat size depends only on the encoding limits (so the model head is fixed).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from ..schema import Cell, Direction
from ..training.config import DIRECTION_ORDER, EncodingConfig
from .construction import ReverseMove

_DIR_INDEX = {Direction.from_name(name): i for i, name in enumerate(DIRECTION_ORDER)}
_INDEX_DIR = {i: Direction.from_name(name) for i, name in enumerate(DIRECTION_ORDER)}

STOP_INDEX = 0


@dataclass(frozen=True)
class DesignerAction:
    """Either ``STOP`` or a reverse slide of the block anchored at ``anchor``."""

    kind: str                       # "stop" | "reverse"
    anchor: Optional[Cell] = None
    direction: Optional[Direction] = None
    distance: int = 0

    @property
    def is_stop(self) -> bool:
        return self.kind == "stop"

    def to_move(self) -> ReverseMove:
        assert self.kind == "reverse"
        return ReverseMove(anchor=self.anchor, direction=self.direction,
                           distance=self.distance)


STOP = DesignerAction(kind="stop")


class DesignerActionSpace:
    """Bidirectional map between designer actions and flat indices."""

    def __init__(self, encoding: EncodingConfig) -> None:
        self.rows = encoding.max_rows
        self.cols = encoding.max_cols
        self.max_distance = encoding.max_slide_distance
        self.size = 1 + self.rows * self.cols * 4 * self.max_distance

    def index_of(self, action: DesignerAction) -> int:
        if action.is_stop:
            return STOP_INDEX
        return self._reverse_index(action.anchor, action.direction, action.distance)

    def _reverse_index(self, anchor: Cell, direction: Direction, distance: int) -> int:
        if not (0 <= anchor.r < self.rows and 0 <= anchor.c < self.cols):
            raise ValueError(f"anchor {anchor} outside encoding bounds")
        if not (1 <= distance <= self.max_distance):
            raise ValueError(f"distance {distance} outside [1, {self.max_distance}]")
        d = _DIR_INDEX[direction]
        cell = anchor.r * self.cols + anchor.c
        return 1 + (cell * 4 + d) * self.max_distance + (distance - 1)

    def move_index(self, move: ReverseMove) -> int:
        return self._reverse_index(move.anchor, move.direction, move.distance)

    def from_index(self, index: int) -> DesignerAction:
        if index == STOP_INDEX:
            return STOP
        idx = index - 1
        distance = idx % self.max_distance + 1
        idx //= self.max_distance
        d = idx % 4
        idx //= 4
        c = idx % self.cols
        r = idx // self.cols
        return DesignerAction(kind="reverse", anchor=Cell(r, c),
                              direction=_INDEX_DIR[d], distance=distance)

    def legal_mask(self, moves: list[ReverseMove], *, allow_stop: bool = True
                   ) -> list[bool]:
        """Boolean mask over the flat action space for the given legal moves."""
        mask = [False] * self.size
        if allow_stop:
            mask[STOP_INDEX] = True
        for move in moves:
            try:
                mask[self.move_index(move)] = True
            except (ValueError, KeyError):
                continue
        return mask
