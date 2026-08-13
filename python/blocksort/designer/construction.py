"""Validity-preserving reverse-construction primitives (Python port of the JS
``generator.js`` reverse construction).

Two guarantees this module provides:

* :func:`build_base_level` returns a level that is **solvable by construction**:
  every block is placed just inside its color's gate, slid inward, and accepted
  only if it can still slide back *out* to its gate against everything placed
  before it (blocks exit in reverse placement order).
* :func:`reverse_slide_moves` enumerates moves that are the exact inverse of a
  legal forward slide, so applying one to a solvable level yields another
  solvable level (the previous position is reachable by one forward slide).

The designer environment and the random-generator baseline both build on these.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Optional

from ..environment import Environment
from ..schema import Block, Cell, Direction, Exit, Level
from ..state import State

# direction helpers (mirror generator.js)
_OPP = {
    Direction.UP: Direction.DOWN,
    Direction.DOWN: Direction.UP,
    Direction.LEFT: Direction.RIGHT,
    Direction.RIGHT: Direction.LEFT,
}
_INWARD = {"top": Direction.DOWN, "bottom": Direction.UP,
           "left": Direction.RIGHT, "right": Direction.LEFT}
_OUTWARD = {"top": Direction.UP, "bottom": Direction.DOWN,
            "left": Direction.LEFT, "right": Direction.RIGHT}
_EDGES = ("top", "bottom", "left", "right")

# A small shape library (subset of generator.js SHAPES).
_SHAPES = {
    "mono": [(0, 0)],
    "domino": [(0, 0), (0, 1)],
    "triI": [(0, 0), (0, 1), (0, 2)],
    "triL": [(0, 0), (1, 0), (1, 1)],
    "tetO": [(0, 0), (0, 1), (1, 0), (1, 1)],
}
_DEFAULT_SHAPES = ("mono", "domino", "domino", "triI", "triL", "tetO")


def opposite(direction: Direction) -> Direction:
    return _OPP[direction]


def _normalize(cells: list[tuple[int, int]]) -> list[tuple[int, int]]:
    mr = min(r for r, _ in cells)
    mc = min(c for _, c in cells)
    return [(r - mr, c - mc) for r, c in cells]


def _rotate(cells: list[tuple[int, int]]) -> list[tuple[int, int]]:
    return _normalize([(c, -r) for r, c in cells])


def _random_orientation(rng: random.Random, name: str) -> list[tuple[int, int]]:
    cells = list(_SHAPES[name])
    for _ in range(rng.randint(0, 3)):
        cells = _rotate(cells)
    return _normalize(cells)


def anchor_of(block: Block) -> Cell:
    """The block's identifying cell (smallest ``(r, c)``); unique per block."""
    return block.sorted_cells()[0]


def block_min_rc(cells) -> tuple[int, int]:
    rs = [c.r for c in cells]
    cs = [c.c for c in cells]
    return min(rs), min(cs)


@dataclass(frozen=True)
class GeneratorConfig:
    rows: int = 6
    cols: int = 6
    color_count: int = 3
    density: float = 0.5
    # Must not exceed the checkpoint EncodingConfig.max_blocks used by the
    # designer/protagonist. DesignerEnv enforces this compatibility boundary.
    max_blocks: int = 16
    min_gate_len: int = 2
    max_gate_len: int = 3
    deep_bias: float = 0.8
    place_tries: int = 80
    colors: tuple[str, ...] = ("red", "blue", "green", "yellow", "purple",
                               "orange", "teal", "pink")
    shapes: tuple[str, ...] = _DEFAULT_SHAPES


def _assign_gates(cfg: GeneratorConfig, rng: random.Random):
    colors = list(cfg.colors)
    rng.shuffle(colors)
    colors = colors[:cfg.color_count]
    per_edge = {e: [] for e in _EDGES}
    edge_bag = list(_EDGES)
    rng.shuffle(edge_bag)
    for i, col in enumerate(colors):
        per_edge[edge_bag[i % len(_EDGES)]].append(col)

    gates: list[Exit] = []
    color_gate: dict[str, Exit] = {}
    for edge in _EDGES:
        cols = per_edge[edge]
        if not cols:
            continue
        edge_len = cfg.cols if edge in ("top", "bottom") else cfg.rows
        slot = edge_len // len(cols)
        if slot < 1:
            return None
        for i, col in enumerate(cols):
            max_len = min(cfg.max_gate_len, slot)
            length = max(1, rng.randint(min(cfg.min_gate_len, max_len), max_len))
            slot_start = i * slot
            start = slot_start + rng.randint(0, slot - length)
            gate = Exit(edge=edge, start=start, length=length, color=col)
            gates.append(gate)
            color_gate[col] = gate
    return gates, color_gate, colors


def _level_with(cfg: GeneratorConfig, blocks: list[Block], gates: list[Exit]) -> Level:
    used = {b.color for b in blocks}
    exits = tuple(g for g in gates if g.color in used) or tuple(gates)
    return Level(name="design", cols=cfg.cols, rows=cfg.rows,
                 blocks=tuple(blocks), exits=exits)


def _place_colored(env: Environment, cfg: GeneratorConfig, placed: list[Block],
                   gate: Exit, gates: list[Exit], rng: random.Random
                   ) -> Optional[Block]:
    edge = gate.edge
    inward = _INWARD[edge]
    shape = _random_orientation(rng, rng.choice(list(cfg.shapes)))
    h = max(r for r, _ in shape) + 1
    w = max(c for _, c in shape) + 1

    if edge in ("top", "bottom"):
        if w > gate.length or w > cfg.cols:
            return None
        c0 = gate.start + rng.randint(0, gate.length - w)
        r_off = 0 if edge == "top" else cfg.rows - h
        if r_off < 0:
            return None
        cells = [Cell(r + r_off, c + c0) for r, c in shape]
    else:
        if h > gate.length or h > cfg.rows:
            return None
        r0 = gate.start + rng.randint(0, gate.length - h)
        c_off = 0 if edge == "left" else cfg.cols - w
        if c_off < 0:
            return None
        cells = [Cell(r + r0, c + c_off) for r, c in shape]

    taken = {cl for b in placed for cl in b.cells}
    for cl in cells:
        if not (0 <= cl.r < cfg.rows and 0 <= cl.c < cfg.cols):
            return None
        if cl in taken:
            return None

    cand = Block(color=gate.color, cells=frozenset(cells))
    level = _level_with(cfg, placed + [cand], gates)
    state = env.initial_state(level)
    cand_in_state = state.blocks[len(placed)]

    slide = env.compute_slide(state, cand_in_state, inward)
    if slide.steps <= 0:
        steps = 0
    elif rng.random() < cfg.deep_bias:
        steps = slide.steps
    else:
        steps = rng.randint(0, slide.steps)
    if steps > 0:
        cand = cand_in_state.translate(inward, steps)
    else:
        cand = cand_in_state

    # Must still be able to slide back out to the edge against earlier blocks.
    level = _level_with(cfg, placed + [cand], gates)
    state = env.initial_state(level)
    cand_in_state = state.blocks[len(placed)]
    out = env.compute_slide(state, cand_in_state, _OUTWARD[edge])
    if out.reason != "edge":
        return None
    return cand


def build_base_level(cfg: GeneratorConfig, rng: random.Random,
                     *, env: Optional[Environment] = None) -> Optional[Level]:
    """Build one solvable-by-construction base level, or ``None`` on failure."""
    env = env or Environment()
    ga = _assign_gates(cfg, rng)
    if ga is None:
        return None
    gates, color_gate, colors = ga
    placed: list[Block] = []
    playable = cfg.rows * cfg.cols
    target = round(cfg.density * playable)
    filled = 0
    while filled < target and len(placed) < cfg.max_blocks:
        ok = False
        for _ in range(cfg.place_tries):
            gate = color_gate[rng.choice(colors)]
            cand = _place_colored(env, cfg, placed, gate, gates, rng)
            if cand is not None:
                placed.append(cand)
                filled += len(cand.cells)
                ok = True
                break
        if not ok:
            break
    if len(placed) < 3:
        return None
    return _level_with(cfg, placed, gates)


@dataclass(frozen=True)
class ReverseMove:
    """A validity-preserving reverse slide of the block anchored at ``anchor``."""

    anchor: Cell
    direction: Direction   # the (outward/back) direction the block is slid
    distance: int


def reverse_slide_moves(env: Environment, state: State) -> list[ReverseMove]:
    """Enumerate every legal reverse slide from ``state``.

    A block ``b`` admits a reverse slide in ``back = opp(d)`` when ``b`` is at a
    *stop* in ``d`` (forward steps 0, and not an exit) and can move ``1..k`` cells
    in ``back``. Sliding ``b`` forward by the same amount returns the previous
    position, so solvability is preserved.
    """
    moves: list[ReverseMove] = []
    for block in state.blocks:
        if state.cleared < block.unlock_at:
            continue
        anchor = anchor_of(block)
        for d in Direction:
            fwd = env.compute_slide(state, block, d)
            if fwd.steps != 0:
                continue
            if fwd.reason == "edge" and fwd.can_exit:
                continue
            back = _OPP[d]
            bs = env.compute_slide(state, block, back)
            if bs.steps <= 0:
                continue
            for m in range(1, bs.steps + 1):
                moves.append(ReverseMove(anchor=anchor, direction=back, distance=m))
    return moves


def apply_reverse_move(env: Environment, state: State, move: ReverseMove) -> State:
    """Apply a :class:`ReverseMove`, returning the new state.

    Raises ``ValueError`` if no block is anchored at ``move.anchor`` or the move
    is not currently a legal reverse slide.
    """
    idx = None
    for i, block in enumerate(state.blocks):
        if anchor_of(block) == move.anchor:
            idx = i
            break
    if idx is None:
        raise ValueError(f"no block anchored at {move.anchor}")
    if not isinstance(move.direction, Direction):
        raise ValueError(f"invalid reverse-slide direction: {move.direction!r}")
    if (not isinstance(move.distance, int) or isinstance(move.distance, bool)
            or move.distance < 1):
        raise ValueError(f"invalid reverse-slide distance: {move.distance!r}")
    if move not in reverse_slide_moves(env, state):
        raise ValueError(f"move is not a legal reverse mutation: {move!r}")
    block = state.blocks[idx]
    moved = block.translate(move.direction, move.distance)
    new_blocks = list(state.blocks)
    new_blocks[idx] = moved
    return state.with_blocks(tuple(new_blocks))
