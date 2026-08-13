"""Explicit move representation."""

from __future__ import annotations

from dataclasses import dataclass

from .schema import Direction


@dataclass(frozen=True)
class Action:
    """A single legal move.

    ``block_index`` refers to the block's position in the *current* state's
    block tuple. It is transient (it identifies which block to move for this
    transition) and is deliberately **not** part of canonical state identity,
    since same-color/same-shape blocks are interchangeable.

    For a slide, ``distance`` is the number of whole cells to advance and
    ``exit`` is ``False``. For an exit, ``exit`` is ``True`` and ``distance`` is
    the slide-to-boundary distance (informational; the block is removed).
    """

    block_index: int
    direction: Direction
    distance: int
    exit: bool = False
