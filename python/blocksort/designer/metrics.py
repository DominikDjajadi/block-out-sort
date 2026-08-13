"""Structural difficulty metrics for a generated level.

These are *structural* / search-based difficulty signals, **not** human difficulty
predictions. Solution-derived metrics require an exact A* solution; when A* does
not complete those fields are ``None``.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import asdict, dataclass
from typing import Optional

from ..environment import Environment
from ..schema import Direction, Level
from ..solver import solve_astar


@dataclass(frozen=True)
class StructuralMetrics:
    num_blocks: int
    immediately_exitable: int
    few_exitable: int                       # num_blocks - immediately_exitable
    optimal_moves: Optional[int]
    extra_moves: Optional[int]              # optimal_moves - num_blocks (forced)
    first_exit_depth: Optional[int]         # moves before the first block exits
    distinct_setup_blocks: Optional[int]    # distinct blocks moved by non-exit slides
    rehandled_blocks: Optional[int]         # blocks touched by >1 solution move
    opening_requires_setup: Optional[float] # 1.0 if first optimal move is a slide

    def to_dict(self) -> dict:
        return asdict(self)


def _immediately_exitable(env: Environment, level: Level) -> int:
    state = env.initial_state(level)
    count = 0
    for block in state.blocks:
        for d in Direction:
            s = env.compute_slide(state, block, d)
            if s.reason == "edge" and s.can_exit:
                count += 1
                break
    return count


def _replay_solution(env: Environment, level: Level, actions):
    """Track per-block move counts and the first exit depth along the solution."""
    state = env.initial_state(level)
    ids: dict = {}
    next_id = 0
    for b in state.blocks:
        ids[b.cells] = next_id
        next_id += 1

    move_counts: dict[int, int] = defaultdict(int)
    setup_blocks: set[int] = set()
    first_exit_depth: Optional[int] = None

    for i, action in enumerate(actions):
        block = state.blocks[action.block_index]
        bid = ids.get(block.cells, next_id)
        if block.cells not in ids:
            next_id += 1
        move_counts[bid] += 1
        if action.exit:
            if first_exit_depth is None:
                first_exit_depth = i
            ids.pop(block.cells, None)
        else:
            setup_blocks.add(bid)
        state = env.apply_action(state, action)
        if not action.exit:
            moved = state.blocks[action.block_index]
            ids.pop(block.cells, None)
            ids[moved.cells] = bid

    rehandled = sum(1 for c in move_counts.values() if c > 1)
    opening_requires_setup = 0.0
    if actions:
        opening_requires_setup = 0.0 if actions[0].exit else 1.0
    return (first_exit_depth if first_exit_depth is not None else len(actions),
            len(setup_blocks), rehandled, opening_requires_setup)


def structural_metrics(env: Environment, level: Level, *,
                       astar_max_nodes: int = 200_000,
                       astar_result=None) -> StructuralMetrics:
    """Compute structural metrics; reuses ``astar_result`` if provided."""
    num_blocks = level.total_blocks
    imm = _immediately_exitable(env, level)

    if astar_result is None:
        astar_result = solve_astar(env, env.initial_state(level),
                                   max_nodes=astar_max_nodes)

    optimal_moves = extra = first_exit = setup = rehandled = opening = None
    if astar_result.solvable is True and astar_result.actions is not None:
        optimal_moves = astar_result.move_count
        extra = optimal_moves - num_blocks
        first_exit, setup, rehandled, opening = _replay_solution(
            env, level, astar_result.actions)

    return StructuralMetrics(
        num_blocks=num_blocks, immediately_exitable=imm,
        few_exitable=num_blocks - imm, optimal_moves=optimal_moves,
        extra_moves=extra, first_exit_depth=first_exit,
        distinct_setup_blocks=setup, rehandled_blocks=rehandled,
        opening_requires_setup=opening)
