"""Exact solvers built on the environment.

The environment is the single source of truth for legal actions, collisions,
intermediate stops, exits, freezing, locked gates/regions, and terminal
detection. The solvers never re-implement movement rules.

A* uses unit action cost (one drag = one move) and the heuristic

    h(state) = number of remaining blocks

which is **admissible** (every remaining block needs at least one exit move to
clear) and in fact **consistent**: an exit reduces both ``h`` and the path by 1,
and a slide leaves ``h`` unchanged while costing 1, so ``h(s) - h(s') <= cost``
always. With a consistent heuristic the first expansion of a state is optimal,
so A* may return as soon as the goal is popped.
"""

from __future__ import annotations

import heapq
import itertools
import time
from collections import deque
from dataclasses import dataclass
from typing import Any, Optional

from .actions import Action
from .conformance import action_to_normalized
from .environment import Environment
from .schema import Direction
from .solution import deserialize_solution, verify_solution
from .state import State

# Termination reasons.
SOLVED = "solved"
UNSOLVABLE = "unsolvable"
NODE_LIMIT = "node_limit"
TIME_LIMIT = "time_limit"
DEPTH_LIMIT = "depth_limit"
INVALID = "invalid"


@dataclass(frozen=True)
class SolveResult:
    """Outcome of a search.

    ``solvable`` is ``True`` (optimal solution found), ``False`` (proven
    unsolvable by exhausting the reachable space), or ``None`` (unknown: a search
    budget was hit — never reported as unsolvable). ``optimal`` is ``True`` only
    when an optimal solution was returned.
    """

    solvable: Optional[bool]
    optimal: bool
    exhausted: bool
    move_count: Optional[int]
    actions: Optional[tuple[Action, ...]]
    states_explored: int
    states_generated: int
    duplicate_states: int
    max_frontier_size: int
    elapsed_seconds: float
    termination_reason: str

    # Serialized (index-independent) form of the solution, when present.
    serialized_actions: Optional[tuple[dict[str, Any], ...]] = None


def _reconstruct(came_from: dict[str, Any], goal_key: str) -> list[dict[str, Any]]:
    """Walk parent links to produce the serialized action path (start -> goal)."""
    path: list[dict[str, Any]] = []
    key: Optional[str] = goal_key
    while key is not None:
        entry = came_from.get(key)
        if entry is None:
            break  # reached the start (value None)
        parent_key, norm_action = entry
        path.append(norm_action)
        key = parent_key
    path.reverse()
    return path


def _finalize_solution(
    env: Environment, state: State, serialized: list[dict[str, Any]]
) -> tuple[tuple[Action, ...], tuple[dict[str, Any], ...]]:
    """Resolve a serialized path to concrete actions (also validates legality)."""
    actions = deserialize_solution(env, state, serialized)
    return actions, tuple(serialized)


def solve_astar(
    env: Environment,
    state: State,
    *,
    max_nodes: int = 250_000,
    max_depth: Optional[int] = None,
    time_limit_seconds: Optional[float] = None,
) -> SolveResult:
    """Exact, optimal A* search for a minimum-move solution.

    Returns a :class:`SolveResult`. The returned path is always replay-verified
    before being reported; a verification failure raises (it would indicate an
    internal inconsistency, not a normal outcome).
    """
    t0 = time.perf_counter()

    def elapsed() -> float:
        return time.perf_counter() - t0

    if state.remaining == 0:
        return SolveResult(
            solvable=True, optimal=True, exhausted=False, move_count=0,
            actions=(), serialized_actions=(), states_explored=0,
            states_generated=1, duplicate_states=0, max_frontier_size=0,
            elapsed_seconds=elapsed(), termination_reason=SOLVED,
        )

    start_key = env.canonical_key(state)
    g_score: dict[str, int] = {start_key: 0}
    came_from: dict[str, Any] = {start_key: None}
    counter = itertools.count()

    def h(s: State) -> int:
        return s.remaining

    h0 = h(state)
    heap: list[tuple[int, int, int, int, State]] = [(h0, h0, next(counter), 0, state)]

    explored = 0
    generated = 1
    duplicates = 0
    max_frontier = 1
    hit_depth = False

    while heap:
        max_frontier = max(max_frontier, len(heap))
        f, hs, _, g, s = heapq.heappop(heap)
        key = env.canonical_key(s)
        if g > g_score.get(key, float("inf")):
            duplicates += 1
            continue  # stale heap entry

        if env.is_terminal(s):
            serialized = _reconstruct(came_from, key)
            actions, ser = _finalize_solution(env, state, serialized)
            if not verify_solution(env, state, actions, expected_move_count=g):
                raise RuntimeError("A* returned a path that failed verification")
            return SolveResult(
                solvable=True, optimal=True, exhausted=False, move_count=g,
                actions=actions, serialized_actions=ser, states_explored=explored,
                states_generated=generated, duplicate_states=duplicates,
                max_frontier_size=max_frontier, elapsed_seconds=elapsed(),
                termination_reason=SOLVED,
            )

        if time_limit_seconds is not None and elapsed() > time_limit_seconds:
            return _exhausted(TIME_LIMIT, explored, generated, duplicates,
                              max_frontier, elapsed())

        explored += 1
        if explored > max_nodes:
            return _exhausted(NODE_LIMIT, explored - 1, generated, duplicates,
                              max_frontier, elapsed())

        if max_depth is not None and g >= max_depth:
            hit_depth = True
            continue  # do not expand beyond the depth limit

        for action in env.legal_actions(s):
            ns = env.apply_action(s, action)
            generated += 1
            nkey = env.canonical_key(ns)
            ng = g + 1
            if ng < g_score.get(nkey, float("inf")):
                g_score[nkey] = ng
                came_from[nkey] = (key, action_to_normalized(s, action))
                hn = h(ns)
                heapq.heappush(heap, (ng + hn, hn, next(counter), ng, ns))
            else:
                duplicates += 1

    # Frontier emptied.
    if hit_depth:
        return _exhausted(DEPTH_LIMIT, explored, generated, duplicates,
                          max_frontier, elapsed())
    return SolveResult(
        solvable=False, optimal=False, exhausted=False, move_count=None,
        actions=None, serialized_actions=None, states_explored=explored,
        states_generated=generated, duplicate_states=duplicates,
        max_frontier_size=max_frontier, elapsed_seconds=elapsed(),
        termination_reason=UNSOLVABLE,
    )


def _exhausted(reason, explored, generated, duplicates, max_frontier, elapsed):
    return SolveResult(
        solvable=None, optimal=False, exhausted=True, move_count=None,
        actions=None, serialized_actions=None, states_explored=explored,
        states_generated=generated, duplicate_states=duplicates,
        max_frontier_size=max_frontier, elapsed_seconds=elapsed,
        termination_reason=reason,
    )


def solve_bfs(
    env: Environment,
    state: State,
    *,
    max_nodes: int = 250_000,
    max_depth: Optional[int] = None,
) -> SolveResult:
    """Breadth-first reference solver.

    With unit action cost, BFS also returns a minimum-move solution. It is kept
    small and is used to cross-check A* optimality on small cases.
    """
    t0 = time.perf_counter()

    if state.remaining == 0:
        return SolveResult(
            solvable=True, optimal=True, exhausted=False, move_count=0,
            actions=(), serialized_actions=(), states_explored=0,
            states_generated=1, duplicate_states=0, max_frontier_size=0,
            elapsed_seconds=time.perf_counter() - t0, termination_reason=SOLVED,
        )

    start_key = env.canonical_key(state)
    depth: dict[str, int] = {start_key: 0}
    came_from: dict[str, Any] = {start_key: None}
    queue: deque[tuple[State, int]] = deque([(state, 0)])
    explored = 0
    generated = 1
    duplicates = 0
    max_frontier = 1
    hit_depth = False

    while queue:
        max_frontier = max(max_frontier, len(queue))
        s, d = queue.popleft()
        explored += 1
        if explored > max_nodes:
            return _exhausted(NODE_LIMIT, explored - 1, generated, duplicates,
                              max_frontier, time.perf_counter() - t0)
        if max_depth is not None and d >= max_depth:
            hit_depth = True
            continue
        for action in env.legal_actions(s):
            ns = env.apply_action(s, action)
            generated += 1
            nkey = env.canonical_key(ns)
            if nkey in depth:
                duplicates += 1
                continue
            depth[nkey] = d + 1
            came_from[nkey] = (env.canonical_key(s), action_to_normalized(s, action))
            if env.is_terminal(ns):
                serialized = _reconstruct(came_from, nkey)
                actions, ser = _finalize_solution(env, state, serialized)
                if not verify_solution(env, state, actions, expected_move_count=d + 1):
                    raise RuntimeError("BFS returned a path that failed verification")
                return SolveResult(
                    solvable=True, optimal=True, exhausted=False, move_count=d + 1,
                    actions=actions, serialized_actions=ser, states_explored=explored,
                    states_generated=generated, duplicate_states=duplicates,
                    max_frontier_size=max_frontier,
                    elapsed_seconds=time.perf_counter() - t0,
                    termination_reason=SOLVED,
                )
            queue.append((ns, d + 1))

    if hit_depth:
        return _exhausted(DEPTH_LIMIT, explored, generated, duplicates,
                          max_frontier, time.perf_counter() - t0)
    return SolveResult(
        solvable=False, optimal=False, exhausted=False, move_count=None,
        actions=None, serialized_actions=None, states_explored=explored,
        states_generated=generated, duplicate_states=duplicates,
        max_frontier_size=max_frontier, elapsed_seconds=time.perf_counter() - t0,
        termination_reason=UNSOLVABLE,
    )


def solve_exit_only(env: Environment, state: State, *, max_nodes: int = 200_000) -> bool:
    """Whether the board can be cleared using only exit moves (no shuffles).

    Mirrors the JavaScript ``solveExitOnly`` for parity diagnostics: branch only
    on "which block exits next". Complete for levels solvable by exiting in some
    order; may report ``False`` for puzzles that genuinely require a shuffle.
    """
    dead: set[str] = set()
    visited = 0

    def dfs(s: State) -> bool:
        nonlocal visited
        if env.is_terminal(s):
            return True
        visited += 1
        if visited > max_nodes:
            return False
        key = env.canonical_key(s)
        if key in dead:
            return False
        for i, block in enumerate(s.blocks):
            if env.is_block_frozen(s, block):
                continue
            for direction in Direction:
                result = env.compute_slide(s, block, direction)
                if result.reason == "edge" and result.can_exit:
                    ns = env.apply_action(s, Action(i, direction, result.steps, exit=True))
                    if dfs(ns):
                        return True
                    break  # one exit direction per block is enough
        dead.add(key)
        return False

    return dfs(state)
