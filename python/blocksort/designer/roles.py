"""Solver roles used to score a generated level.

* **Protagonist** -- the current policy-value model plus a *bounded* graph-search
  budget. This is the solver the designer tries to defeat.
* **Oracle** -- exact A* when it completes within its node budget; otherwise a
  *substantially larger* graph-search budget (a verified, possibly non-optimal
  fallback). The oracle decides whether a level is "really" solvable.

All costs are in moves (lower is better). ``exact`` distinguishes an A*-optimal
cost from an approximate search cost (search values are never claimed exact).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from ..environment import Environment
from ..schema import Level
from ..search.config import SearchConfig
from ..search.graph_search import BlocksortAdapter, GraphSearch
from ..solver import solve_astar


@dataclass(frozen=True)
class SolveOutcome:
    solved: bool
    cost: Optional[float]
    exact: bool
    method: str
    nodes: int = 0


class Protagonist:
    """Bounded neural graph-search solver."""

    def __init__(self, env: Environment, model, encoding_config, value_norm,
                 device, *, simulations: int = 100, c_puct: float = 1.5,
                 dirichlet_alpha: float = 0.0, dirichlet_weight: float = 0.0,
                 temperature: float = 0.0) -> None:
        self.env = env
        self.adapter = BlocksortAdapter(env, model, encoding_config, value_norm,
                                        device)
        self.simulations = simulations
        self.c_puct = c_puct
        self.value_const = getattr(value_norm, "constant", 20.0)
        # Optional root exploration noise. With noise enabled, repeated solves
        # under different seeds genuinely differ, so a per-level solve *rate*
        # (used for frontier estimation) can take values strictly between 0 and
        # 1. With the defaults (noise off) search is fully deterministic.
        self.dirichlet_alpha = dirichlet_alpha
        self.dirichlet_weight = dirichlet_weight
        self.temperature = temperature

    def solve(
        self,
        level: Level,
        *,
        seed: int = 0,
        simulations: int | None = None,
    ) -> SolveOutcome:
        state = self.env.initial_state(level)
        if self.env.is_terminal(state):
            return SolveOutcome(solved=True, cost=0.0, exact=True, method="trivial")
        budget = self.simulations if simulations is None else int(simulations)
        if budget <= 0:
            raise ValueError("protagonist simulations must be positive")
        cfg = SearchConfig(simulations=budget, c_puct=self.c_puct,
                           temperature=self.temperature,
                           dirichlet_alpha=self.dirichlet_alpha,
                           dirichlet_weight=self.dirichlet_weight,
                           value_normalization_constant=self.value_const, seed=seed)
        result = GraphSearch(self.adapter, cfg).run(state)
        if result.solved:
            return SolveOutcome(solved=True, cost=float(result.solution_length),
                                exact=False, method="search",
                                nodes=result.stats.unique_states)
        return SolveOutcome(solved=False, cost=None, exact=False, method="search",
                            nodes=result.stats.unique_states)


class Oracle:
    """Exact A* with a large graph-search fallback."""

    def __init__(self, env: Environment, model, encoding_config, value_norm,
                 device, *, astar_max_nodes: int = 200_000,
                 astar_time_limit_seconds: Optional[float] = None,
                 search_simulations: int = 1000, c_puct: float = 1.5,
                 fallback_on_astar_exhaustion: bool = True) -> None:
        self.env = env
        self.astar_max_nodes = astar_max_nodes
        self.astar_time_limit_seconds = astar_time_limit_seconds
        self.adapter = BlocksortAdapter(env, model, encoding_config, value_norm,
                                        device)
        self.search_simulations = search_simulations
        self.c_puct = c_puct
        self.value_const = getattr(value_norm, "constant", 20.0)
        self.fallback_on_astar_exhaustion = fallback_on_astar_exhaustion

    def solve(self, level: Level, *, seed: int = 0) -> SolveOutcome:
        return self.solve_detailed(level, seed=seed)[0]

    def solve_detailed(self, level: Level, *, seed: int = 0):
        """Return ``(SolveOutcome, astar_result | None)``.

        The A* result (when it completed) is returned so structural metrics can
        reuse the optimal solution without re-running A*.
        """
        state = self.env.initial_state(level)
        if self.env.is_terminal(state):
            return SolveOutcome(solved=True, cost=0.0, exact=True,
                                method="trivial"), None

        result = solve_astar(
            self.env, state, max_nodes=self.astar_max_nodes,
            time_limit_seconds=self.astar_time_limit_seconds)
        if result.solvable is True:
            return SolveOutcome(solved=True, cost=float(result.move_count),
                                exact=True, method="astar",
                                nodes=result.states_explored), result
        if result.solvable is False:
            return SolveOutcome(solved=False, cost=None, exact=True,
                                method="astar", nodes=result.states_explored), result

        if not self.fallback_on_astar_exhaustion:
            return SolveOutcome(
                solved=False, cost=None, exact=False,
                method="astar_exhausted", nodes=result.states_explored), result

        # A* exhausted -> larger graph-search fallback (verified, not optimal).
        cfg = SearchConfig(simulations=self.search_simulations, c_puct=self.c_puct,
                           temperature=0.0,
                           value_normalization_constant=self.value_const, seed=seed)
        sr = GraphSearch(self.adapter, cfg).run(state)
        if sr.solved:
            return SolveOutcome(solved=True, cost=float(sr.solution_length),
                                exact=False, method="search_fallback",
                                nodes=sr.stats.unique_states), result
        return SolveOutcome(solved=False, cost=None, exact=False,
                            method="search_fallback",
                            nodes=sr.stats.unique_states), result
