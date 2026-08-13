"""Exact state-value oracle built on the A* solver.

Definitions:

    V*(s)          = minimum number of moves to clear state s
    Q_cost(s, a)   = 1 + V*(T(s, a))
    regret(s, a)   = Q_cost(s, a) - V*(s)        (>= 0; 0 iff a is optimal)

Exact values are cached by ``(static_level_signature, canonical_state_key)`` so
that levels which differ only in gates/holes/regions/thresholds never share a
cached value. Exhausted-budget results are cached too (``exact=False``,
``solvable=None``) so repeated queries within the same fixed-limit oracle do
not rerun A*.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Optional

from .actions import Action
from .environment import Environment
from .signature import static_level_signature
from .solution import serialize_action
from .solver import SolveResult, solve_astar
from .state import State


@dataclass(frozen=True)
class ValueResult:
    """Exact value of a state. ``value`` is the move count when solvable+exact."""

    value: Optional[int]
    exact: bool
    solvable: Optional[bool]


@dataclass(frozen=True)
class ActionAnalysis:
    action: Action
    serialized: dict[str, Any]
    successor_value: Optional[int]   # V*(T(s,a)); None if unsolvable or unknown
    successor_exact: bool
    cost: Optional[int]              # 1 + successor_value; None if not finite/exact
    regret: Optional[int]            # cost - V*(s); None if not finite/exact
    optimal: bool


@dataclass(frozen=True)
class StateAnalysis:
    state_key: str
    static_signature: str
    terminal: bool
    solvable: bool
    exact: bool
    value: Optional[int]
    legal_actions: tuple[Action, ...]
    actions: tuple[ActionAnalysis, ...]
    all_successors_exact: bool


class Oracle:
    """Caching exact-value oracle. Holds no global state; one per analysis run."""

    def __init__(
        self,
        env: Environment,
        *,
        max_nodes: int = 250_000,
        time_limit_seconds: Optional[float] = None,
        search_observer: Optional[Callable[[dict[str, Any]], None]] = None,
    ) -> None:
        self.env = env
        self.max_nodes = max_nodes
        self.time_limit_seconds = time_limit_seconds
        self.search_observer = search_observer
        self._search_count = 0
        # (static_signature, canonical_state_key) -> cached ValueResult
        self._result_cache: dict[tuple[str, str], ValueResult] = {}
        # Keep the complete solver result as well as its scalar projection.
        # In particular, a solved root contains a verified optimal action path
        # that remains useful if proving every successor later exhausts.
        self._solve_result_cache: dict[tuple[str, str], SolveResult] = {}
        self._sig_cache: dict[Any, str] = {}

    def signature(self, state: State) -> str:
        level = state.level
        sig = self._sig_cache.get(level)
        if sig is None:
            sig = static_level_signature(level)
            self._sig_cache[level] = sig
        return sig

    def value(self, state: State, *, query_role: str = "value") -> ValueResult:
        """Exact ``V*(state)``. Returns ``exact=False`` if the budget was hit."""
        if self.env.is_terminal(state):
            return ValueResult(value=0, exact=True, solvable=True)

        key = (self.signature(state), self.env.canonical_key(state))
        cached = self._result_cache.get(key)
        if cached is not None:
            return cached

        self._search_count += 1
        result = solve_astar(
            self.env, state,
            max_nodes=self.max_nodes,
            time_limit_seconds=self.time_limit_seconds,
        )
        self._solve_result_cache[key] = result
        if self.search_observer is not None:
            self.search_observer({
                "query_index": self._search_count,
                "query_role": query_role,
                "static_level_signature": key[0],
                "state_key": key[1],
                "remaining_blocks": state.remaining,
                "max_nodes": self.max_nodes,
                "time_limit_seconds": self.time_limit_seconds,
                "termination_reason": result.termination_reason,
                "solvable": result.solvable,
                "exact": result.optimal or result.solvable is False,
                "states_explored": result.states_explored,
                "states_generated": result.states_generated,
                "duplicate_states": result.duplicate_states,
                "max_frontier_size": result.max_frontier_size,
                "elapsed_seconds": result.elapsed_seconds,
            })
        if result.solvable is True:
            vr = ValueResult(value=result.move_count, exact=True, solvable=True)
        elif result.solvable is False:
            vr = ValueResult(value=None, exact=True, solvable=False)
        else:
            vr = ValueResult(value=None, exact=False, solvable=None)
        self._result_cache[key] = vr
        return vr

    def cached_solve_result(self, state: State) -> Optional[SolveResult]:
        """Return the solver proof already obtained for ``state``, if any.

        This never starts a search.  Callers can therefore recover a solved
        root's verified path after :meth:`analyze` fails to complete all
        successor values without accidentally spending a second A* budget.
        """
        key = (self.signature(state), self.env.canonical_key(state))
        return self._solve_result_cache.get(key)

    def analyze(self, state: State) -> StateAnalysis:
        """Full exact analysis: value plus per-action cost/regret/optimality."""
        sig = self.signature(state)
        key = self.env.canonical_key(state)
        terminal = self.env.is_terminal(state)
        v = self.value(state, query_role="root")

        if terminal:
            return StateAnalysis(
                state_key=key, static_signature=sig, terminal=True,
                solvable=True, exact=True, value=0, legal_actions=(),
                actions=(), all_successors_exact=True,
            )

        legal = tuple(self.env.legal_actions(state))
        analyses: list[ActionAnalysis] = []
        all_exact = True
        # Exact policy targets require an exact root value and every successor
        # value. If the root already exhausted its budget, no successor search
        # can restore exactness. Likewise, once one successor is unknown, the
        # remaining searches cannot restore a complete exact policy target.
        root_allows_exact_policy = v.exact and v.solvable is True
        successor_search_still_useful = root_allows_exact_policy
        for action in legal:
            successor = self.env.apply_action(state, action)
            if successor_search_still_useful:
                sv = self.value(successor, query_role="successor")
            else:
                sv = ValueResult(value=None, exact=False, solvable=None)
            if not sv.exact:
                all_exact = False
                cost = regret = None
                optimal = False
                successor_search_still_useful = False
            elif sv.solvable is False:
                cost = regret = None  # leads to an unsolvable state (infinite cost)
                optimal = False
            else:
                cost = 1 + sv.value
                regret = (cost - v.value) if (v.exact and v.value is not None) else None
                optimal = regret == 0
            analyses.append(ActionAnalysis(
                action=action,
                serialized=serialize_action(state, action),
                successor_value=sv.value if sv.exact and sv.solvable else None,
                successor_exact=sv.exact,
                cost=cost, regret=regret, optimal=optimal,
            ))

        return StateAnalysis(
            state_key=key, static_signature=sig, terminal=False,
            solvable=(v.solvable is True), exact=v.exact, value=v.value,
            legal_actions=legal, actions=tuple(analyses),
            all_successors_exact=root_allows_exact_policy and all_exact,
        )

    # ----- convenience accessors -----

    def optimal_remaining_moves(self, state: State) -> Optional[int]:
        v = self.value(state)
        return v.value if (v.exact and v.solvable) else None

    def optimal_actions(self, state: State) -> list[Action]:
        analysis = self.analyze(state)
        return [a.action for a in analysis.actions if a.optimal]

    def action_costs(self, state: State) -> list[tuple[Action, Optional[int]]]:
        analysis = self.analyze(state)
        return [(a.action, a.cost) for a in analysis.actions]

    def action_regrets(self, state: State) -> list[tuple[Action, Optional[int]]]:
        analysis = self.analyze(state)
        return [(a.action, a.regret) for a in analysis.actions]
