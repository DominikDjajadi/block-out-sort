"""Strict level validation.

:func:`validate_level` returns *all* detected structural problems (where
practical) rather than failing on the first one, so authors get a complete
picture.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

from .schema import COLORS, EDGES, Block, Cell, Level


@dataclass(frozen=True)
class ValidationError:
    """A single structural problem with a stable ``code`` and a message."""

    code: str
    message: str

    def __str__(self) -> str:  # pragma: no cover - convenience
        return f"[{self.code}] {self.message}"


def _is_connected(cells: frozenset[Cell]) -> bool:
    if not cells:
        return False
    start = next(iter(cells))
    seen = {start}
    queue: deque[Cell] = deque([start])
    while queue:
        cell = queue.popleft()
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            nb = Cell(cell.r + dr, cell.c + dc)
            if nb in cells and nb not in seen:
                seen.add(nb)
                queue.append(nb)
    return len(seen) == len(cells)


def _span(values: frozenset[int]) -> int:
    return max(values) - min(values) + 1


def validate_level(level: Level) -> list[ValidationError]:
    """Return a list of structural errors. Empty means the level is valid."""
    errors: list[ValidationError] = []

    def err(code: str, message: str) -> None:
        errors.append(ValidationError(code, message))

    # Dimensions.
    if not isinstance(level.cols, int) or level.cols < 1:
        err("dims", f"cols must be a positive integer, got {level.cols!r}")
    if not isinstance(level.rows, int) or level.rows < 1:
        err("dims", f"rows must be a positive integer, got {level.rows!r}")

    dims_ok = isinstance(level.cols, int) and isinstance(level.rows, int) and (
        level.cols >= 1 and level.rows >= 1
    )

    total = level.total_blocks

    # No blocks / no exits.
    if total == 0:
        err("no_blocks", "level has no blocks")
    if len(level.exits) == 0:
        err("no_exits", "level has no exits")

    # Holes in bounds.
    if dims_ok:
        for hole in level.holes:
            if not level.in_bounds(hole.r, hole.c):
                err("hole_bounds", f"hole {hole.as_tuple()} is out of bounds")

    # Locked regions.
    initially_locked: set[Cell] = set()
    for ri, region in enumerate(level.locked_regions):
        if not isinstance(region.unlock_at, int) or region.unlock_at < 0:
            err("region_unlock", f"lockedRegion #{ri} has invalid unlockAt "
                                  f"{region.unlock_at!r}")
        if not region.cells:
            err("region_empty", f"lockedRegion #{ri} has no cells")
        for cell in region.cells:
            if dims_ok and not level.in_bounds(cell.r, cell.c):
                err("region_bounds",
                    f"lockedRegion #{ri} cell {cell.as_tuple()} is out of bounds")
        if region.unlock_at > 0:
            initially_locked.update(region.cells)

    # Blocks.
    occupied: dict[Cell, int] = {}
    for bi, block in enumerate(level.blocks):
        label = f"block #{bi} ({block.color})"

        if block.color not in COLORS:
            err("color", f"{label} has unknown color {block.color!r}")

        if not isinstance(block.unlock_at, int) or block.unlock_at < 0:
            err("block_unlock", f"{label} has invalid unlockAt {block.unlock_at!r}")
        elif total and block.unlock_at >= total:
            err("block_unlock",
                f"{label} unlockAt {block.unlock_at} >= total blocks {total}; "
                "it can never unfreeze")

        if not block.cells:
            err("block_empty", f"{label} has no cells")
            continue

        # Duplicate cells: a frozenset hides duplicates, so we cannot detect
        # them after parsing. Validation operates on the parsed Level; duplicate
        # raw cells collapse harmlessly. (Raw-duplicate detection lives in the
        # JSON layer if needed.) Out-of-bounds + connectivity below.

        for cell in block.cells:
            if dims_ok and not level.in_bounds(cell.r, cell.c):
                err("block_bounds", f"{label} cell {cell.as_tuple()} is out of bounds")
            if cell in level.holes:
                err("block_on_hole", f"{label} cell {cell.as_tuple()} sits on a hole")
            if cell in initially_locked:
                err("block_on_locked",
                    f"{label} cell {cell.as_tuple()} sits on an initially locked region")
            if cell in occupied:
                err("overlap",
                    f"{label} overlaps block #{occupied[cell]} at {cell.as_tuple()}")
            else:
                occupied[cell] = bi

        if not _is_connected(block.cells):
            err("disconnected", f"{label} is not orthogonally connected")

        # Reachability: can this block ever fit through a matching gate?
        if block.color in COLORS:
            _check_block_fits(level, block, label, err)

    # Exits.
    for ei, exit_ in enumerate(level.exits):
        label = f"exit #{ei} ({exit_.color} {exit_.edge})"
        if exit_.edge not in EDGES:
            err("exit_edge", f"{label} has invalid edge {exit_.edge!r}")
        if exit_.color not in COLORS:
            err("exit_color", f"{label} has unknown color {exit_.color!r}")
        if not isinstance(exit_.start, int) or exit_.start < 0:
            err("exit_start", f"{label} has invalid start {exit_.start!r}")
        if not isinstance(exit_.length, int) or exit_.length < 1:
            err("exit_length", f"{label} has invalid length {exit_.length!r}")
        if not isinstance(exit_.unlock_at, int) or exit_.unlock_at < 0:
            err("exit_unlock", f"{label} has invalid unlockAt {exit_.unlock_at!r}")
        # Extent check (only when edge + numbers are sane).
        if dims_ok and exit_.edge in EDGES and isinstance(exit_.start, int) \
                and isinstance(exit_.length, int) and exit_.start >= 0 \
                and exit_.length >= 1:
            extent = level.cols if exit_.edge in ("top", "bottom") else level.rows
            if exit_.start + exit_.length > extent:
                err("exit_overflow",
                    f"{label} spans [{exit_.start}, {exit_.start + exit_.length}) "
                    f"past edge extent {extent}")

    return errors


def _check_block_fits(level: Level, block: Block, label: str, err) -> None:
    """Flag a block that can never fit through any matching gate.

    For a top/bottom exit the block's column span must fit the gate length; for
    a left/right exit the row span must fit. The block slides freely along the
    edge, so only the perpendicular span matters.
    """
    col_span = _span(block.columns())
    row_span = _span(block.rows())
    matching = [e for e in level.exits if e.color == block.color]
    if not matching:
        err("no_matching_gate", f"{label} has no matching-color gate")
        return
    for exit_ in matching:
        if exit_.edge in ("top", "bottom"):
            if col_span <= exit_.length:
                return
        else:
            if row_span <= exit_.length:
                return
    err("gate_too_narrow",
        f"{label} (col-span {col_span}, row-span {row_span}) is wider than every "
        "matching gate; it can never exit")


def validate_level_data(data: dict) -> list[ValidationError]:
    """Validate a raw interchange dict, then the parsed level.

    This catches issues that the parsed :class:`Level` cannot represent, namely
    duplicate cells within a block (which collapse into a set on parse), before
    delegating to :func:`validate_level`.
    """
    from .serialization import level_from_dict

    errors: list[ValidationError] = []
    for bi, block in enumerate(data.get("blocks", [])):
        seen: set[tuple[int, int]] = set()
        for cell in block.get("cells", []):
            key = (int(cell[0]), int(cell[1]))
            if key in seen:
                errors.append(
                    ValidationError(
                        "duplicate_cell",
                        f"block #{bi} has duplicate cell {list(key)}",
                    )
                )
            seen.add(key)
    try:
        level = level_from_dict(data)
    except (KeyError, TypeError, ValueError) as exc:
        errors.append(ValidationError("parse", f"could not parse level: {exc}"))
        return errors
    errors.extend(validate_level(level))
    return errors


def is_valid(level: Level) -> bool:
    return not validate_level(level)
