"""Immutable search state.

Static level data (dimensions, holes, exits, locked regions, block shapes and
unlock thresholds) lives on :class:`~blocksort.schema.Level`. The dynamic state
holds only what can change: the multiset of remaining blocks and their
positions. ``cleared`` is derived from how many blocks remain, and move count is
intentionally excluded from canonical identity.
"""

from __future__ import annotations

from dataclasses import dataclass

from .schema import Block, Level


@dataclass(frozen=True)
class State:
    """A dynamic game state: the remaining blocks over a static level.

    ``total_blocks`` is the number of blocks present at the *start* of this
    scenario; ``cleared`` is derived from it and how many remain. It is captured
    on the state (rather than read from ``level``) so that conformance scenarios
    can start from an arbitrary placement without the level's definition having
    to match.
    """

    level: Level
    blocks: tuple[Block, ...]
    total_blocks: int

    @classmethod
    def start(cls, level: Level, blocks: tuple[Block, ...] | None = None) -> "State":
        """Build a start state. Defaults to the level's own block placements."""
        start_blocks = tuple(level.blocks) if blocks is None else tuple(blocks)
        return cls(level=level, blocks=start_blocks, total_blocks=len(start_blocks))

    def with_blocks(self, blocks: tuple[Block, ...]) -> "State":
        """A successor state sharing this scenario's static info and total."""
        return State(level=self.level, blocks=blocks, total_blocks=self.total_blocks)

    @property
    def cleared(self) -> int:
        """Number of blocks cleared, derived from how many remain."""
        return self.total_blocks - len(self.blocks)

    @property
    def remaining(self) -> int:
        return len(self.blocks)


def canonical_key(state: State) -> str:
    """A deterministic, order-independent identity for ``state``.

    The key includes each block's color, unlock threshold, and sorted occupied
    coordinates, then sorts the per-block parts so that reordering
    interchangeable blocks yields the same string. It excludes object identity
    and move count and is stable across runs. The format matches the JavaScript
    solver's ``stateKey`` exactly, so keys can be compared cross-language.
    """
    parts = []
    for block in state.blocks:
        cells = ";".join(f"{cell.r},{cell.c}" for cell in block.sorted_cells())
        parts.append(f"{block.color}#{block.unlock_at}#{cells}")
    parts.sort()
    return "|".join(parts)
