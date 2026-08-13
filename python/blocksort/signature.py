"""Deterministic static-level signatures.

A signature captures everything *fixed* about a level: board dimensions, holes,
gate definitions, locked regions and their thresholds, and the static block
descriptors (color, normalized shape, unlock threshold). It deliberately
excludes block *positions*, which are dynamic.

The oracle/dataset cache identity is ``(static_signature, canonical_state_key)``
so that two levels sharing a block arrangement but differing in gates, holes,
locked regions, or thresholds never collide. Signatures use SHA-256 over a
canonical JSON string, so they are stable across processes (unlike ``hash()``).
"""

from __future__ import annotations

import hashlib
import json

from .schema import Block, Cell, Level


def _normalized_shape(cells: frozenset[Cell]) -> list[list[int]]:
    min_r = min(c.r for c in cells)
    min_c = min(c.c for c in cells)
    shifted = sorted((c.r - min_r, c.c - min_c) for c in cells)
    return [[r, c] for r, c in shifted]


def _block_descriptor(block: Block) -> list:
    # Static identity of a block: color, unlock threshold, position-free shape.
    return [block.color, block.unlock_at, _normalized_shape(block.cells)]


def static_level_payload(level: Level) -> dict:
    """The canonical, JSON-serializable static description of ``level``."""
    return {
        "cols": level.cols,
        "rows": level.rows,
        "holes": sorted([c.r, c.c] for c in level.holes),
        "exits": sorted(
            [e.edge, e.start, e.length, e.color, e.unlock_at] for e in level.exits
        ),
        "lockedRegions": sorted(
            [sorted([c.r, c.c] for c in reg.cells), reg.unlock_at]
            for reg in level.locked_regions
        ),
        "blocks": sorted(_block_descriptor(b) for b in level.blocks),
    }


def static_level_signature(level: Level) -> str:
    """A stable hex SHA-256 signature of the level's static geometry/rules."""
    payload = static_level_payload(level)
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()
