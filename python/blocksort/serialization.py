"""Load/dump the JavaScript-compatible camelCase JSON level format.

Internally the package uses snake_case dataclasses; the interchange format keeps
the original camelCase keys (``unlockAt``, ``lockedRegions``) so files round-trip
with the JS engine.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .schema import Block, Cell, Exit, Level, LockedRegion


def _cells_from_json(raw: list[list[int]]) -> frozenset[Cell]:
    return frozenset(Cell(int(r), int(c)) for r, c in raw)


def _cells_to_json(cells: frozenset[Cell]) -> list[list[int]]:
    return [[cell.r, cell.c] for cell in sorted(cells)]


def level_from_dict(data: dict[str, Any]) -> Level:
    """Build a :class:`Level` from a parsed JSON/dict in interchange format."""
    blocks = tuple(
        Block(
            color=b["color"],
            cells=_cells_from_json(b["cells"]),
            unlock_at=int(b.get("unlockAt", 0) or 0),
        )
        for b in data.get("blocks", [])
    )
    exits = tuple(
        Exit(
            edge=e["edge"],
            start=int(e["start"]),
            length=int(e["length"]),
            color=e["color"],
            unlock_at=int(e.get("unlockAt", 0) or 0),
        )
        for e in data.get("exits", [])
    )
    locked_regions = tuple(
        LockedRegion(
            cells=_cells_from_json(reg["cells"]),
            unlock_at=int(reg.get("unlockAt", 0) or 0),
        )
        for reg in data.get("lockedRegions", [])
    )
    return Level(
        name=data.get("name", ""),
        cols=int(data["cols"]),
        rows=int(data["rows"]),
        holes=_cells_from_json(data.get("holes", [])),
        blocks=blocks,
        exits=exits,
        locked_regions=locked_regions,
    )


def level_to_dict(level: Level) -> dict[str, Any]:
    """Serialize a :class:`Level` back to interchange (camelCase) format.

    Mirrors the JS conventions: zero ``unlockAt`` and empty ``holes`` /
    ``lockedRegions`` are omitted.
    """
    data: dict[str, Any] = {
        "name": level.name,
        "cols": level.cols,
        "rows": level.rows,
    }
    if level.holes:
        data["holes"] = _cells_to_json(level.holes)

    blocks: list[dict[str, Any]] = []
    for block in level.blocks:
        entry: dict[str, Any] = {
            "color": block.color,
            "cells": _cells_to_json(block.cells),
        }
        if block.unlock_at:
            entry["unlockAt"] = block.unlock_at
        blocks.append(entry)
    data["blocks"] = blocks

    exits: list[dict[str, Any]] = []
    for exit_ in level.exits:
        entry = {
            "edge": exit_.edge,
            "start": exit_.start,
            "length": exit_.length,
            "color": exit_.color,
        }
        if exit_.unlock_at:
            entry["unlockAt"] = exit_.unlock_at
        exits.append(entry)
    data["exits"] = exits

    if level.locked_regions:
        data["lockedRegions"] = [
            {"cells": _cells_to_json(reg.cells), "unlockAt": reg.unlock_at}
            for reg in level.locked_regions
        ]
    return data


def load_level(path: str | Path) -> Level:
    """Load a level from a JSON file."""
    with open(path, "r", encoding="utf-8") as fh:
        return level_from_dict(json.load(fh))


def loads_level(text: str) -> Level:
    """Load a level from a JSON string."""
    return level_from_dict(json.loads(text))


def dumps_level(level: Level, *, indent: int | None = 2) -> str:
    """Serialize a level to a JSON string in interchange format."""
    return json.dumps(level_to_dict(level), indent=indent)
