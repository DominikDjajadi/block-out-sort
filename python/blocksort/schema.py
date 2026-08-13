"""Typed, immutable level-schema representations for Block Out Sort.

Coordinates use ``(r, c)`` with ``r`` increasing top->bottom and ``c``
increasing left->right, matching the JavaScript engine. All schema types are
frozen dataclasses so they are hashable and safe to share across search states.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Iterable

# The eight palette colors recognized by the JavaScript engine.
COLORS: frozenset[str] = frozenset(
    {"red", "blue", "green", "yellow", "purple", "orange", "teal", "pink"}
)

# Valid gate edges.
EDGES: frozenset[str] = frozenset({"top", "bottom", "left", "right"})


@dataclass(frozen=True, order=True)
class Cell:
    """A single grid cell. Ordered/hashable for deterministic sorting."""

    r: int
    c: int

    def as_tuple(self) -> tuple[int, int]:
        return (self.r, self.c)


class Direction(Enum):
    """An axis-aligned slide direction."""

    UP = "up"
    DOWN = "down"
    LEFT = "left"
    RIGHT = "right"

    @property
    def dr(self) -> int:
        return _DIR_DELTA[self][0]

    @property
    def dc(self) -> int:
        return _DIR_DELTA[self][1]

    @property
    def delta(self) -> tuple[int, int]:
        return _DIR_DELTA[self]

    @property
    def edge(self) -> str:
        """The board edge this direction exits through."""
        return _DIR_EDGE[self]

    @classmethod
    def from_name(cls, name: str) -> "Direction":
        return cls(name)


_DIR_DELTA: dict[Direction, tuple[int, int]] = {
    Direction.UP: (-1, 0),
    Direction.DOWN: (1, 0),
    Direction.LEFT: (0, -1),
    Direction.RIGHT: (0, 1),
}

_DIR_EDGE: dict[Direction, str] = {
    Direction.UP: "top",
    Direction.DOWN: "bottom",
    Direction.LEFT: "left",
    Direction.RIGHT: "right",
}

# Direction that pushes a block away from an edge (used by callers/tests).
EDGE_TO_OUTWARD: dict[str, Direction] = {
    "top": Direction.UP,
    "bottom": Direction.DOWN,
    "left": Direction.LEFT,
    "right": Direction.RIGHT,
}


def _freeze_cells(cells: Iterable[Cell]) -> frozenset[Cell]:
    return frozenset(cells)


@dataclass(frozen=True)
class Block:
    """A connected polyomino, used both as a definition and as a placement.

    ``cells`` are absolute board coordinates. Because blocks are rigid and only
    translate, a moved block is simply a new :class:`Block` with shifted cells.
    """

    color: str
    cells: frozenset[Cell]
    unlock_at: int = 0

    def sorted_cells(self) -> tuple[Cell, ...]:
        """Cells in canonical ``(r, c)`` order."""
        return tuple(sorted(self.cells))

    def translate(self, direction: Direction, steps: int) -> "Block":
        """Return a copy translated ``steps`` cells in ``direction``."""
        dr, dc = direction.delta
        moved = frozenset(
            Cell(cell.r + dr * steps, cell.c + dc * steps) for cell in self.cells
        )
        return Block(color=self.color, cells=moved, unlock_at=self.unlock_at)

    def columns(self) -> frozenset[int]:
        return frozenset(cell.c for cell in self.cells)

    def rows(self) -> frozenset[int]:
        return frozenset(cell.r for cell in self.cells)


@dataclass(frozen=True)
class Exit:
    """An edge gate. ``start``/``length`` index columns (top/bottom) or rows
    (left/right). The gate is active once ``cleared >= unlock_at``."""

    edge: str
    start: int
    length: int
    color: str
    unlock_at: int = 0


@dataclass(frozen=True)
class LockedRegion:
    """Cells that act as walls until ``cleared >= unlock_at``."""

    cells: frozenset[Cell]
    unlock_at: int = 0


@dataclass(frozen=True)
class Level:
    """A complete, static level definition."""

    name: str
    cols: int
    rows: int
    holes: frozenset[Cell] = field(default_factory=frozenset)
    blocks: tuple[Block, ...] = ()
    exits: tuple[Exit, ...] = ()
    locked_regions: tuple[LockedRegion, ...] = ()

    @property
    def total_blocks(self) -> int:
        return len(self.blocks)

    def in_bounds(self, r: int, c: int) -> bool:
        return 0 <= r < self.rows and 0 <= c < self.cols
