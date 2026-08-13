"""Exact-oracle evaluation of search vs the raw model, across budgets.

A* is used only here (for ground truth), never inside a simulation. For each
evaluated state we compare the raw-policy action, the search-selected action, and
the exact optimal actions, and we measure value error and solve quality.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Any, Optional

from ..environment import Environment
from ..oracle import Oracle, ValueResult
from ..solution import serialize_action
from ..state import State
from .config import SearchConfig
from .graph_search import BlocksortAdapter, GraphSearch


@dataclass
class _Agg:
    n: int = 0
    raw_optimal: int = 0
    search_optimal: int = 0
    raw_regret_sum: float = 0.0
    raw_regret_n: int = 0
    search_regret_sum: float = 0.0
    search_regret_n: int = 0
    solved: int = 0
    solvable_n: int = 0
    length_gap_sum: float = 0.0
    length_gap_n: int = 0
    nodes_expanded: int = 0
    transposition_hits: int = 0
    cycle_rejections: int = 0
    deadlocks: int = 0
    model_evaluations: int = 0
    model_evaluation_batches: int = 0
    model_evaluation_cache_hits: int = 0
    elapsed: float = 0.0
    value_abs_sum: float = 0.0
    value_n: int = 0

    def summary(self) -> dict[str, Any]:
        raw_accuracy = _safe_div(self.raw_optimal, self.raw_regret_n)
        search_accuracy = _safe_div(self.search_optimal, self.search_regret_n)
        raw_mean_regret = _safe_div(self.raw_regret_sum, self.raw_regret_n)
        search_mean_regret = _safe_div(
            self.search_regret_sum, self.search_regret_n)
        return {
            "states": self.n,
            "total_evaluated_count": self.n,
            "raw_policy_known_regret_count": self.raw_regret_n,
            "raw_policy_unknown_regret_count": self.n - self.raw_regret_n,
            "raw_policy_optimal_accuracy_known": raw_accuracy,
            "raw_policy_oracle_regret_coverage":
                _safe_div(self.raw_regret_n, self.n),
            "raw_policy_confirmed_optimal_count": self.raw_optimal,
            "raw_policy_confirmed_optimal_rate":
                _safe_div(self.raw_optimal, self.n),
            "raw_policy_mean_regret_known": raw_mean_regret,
            "search_known_regret_count": self.search_regret_n,
            "search_unknown_regret_count": self.n - self.search_regret_n,
            "search_optimal_accuracy_known": search_accuracy,
            "search_oracle_regret_coverage":
                _safe_div(self.search_regret_n, self.n),
            "search_confirmed_optimal_count": self.search_optimal,
            "search_confirmed_optimal_rate":
                _safe_div(self.search_optimal, self.n),
            "search_mean_regret_known": search_mean_regret,
            # Compatibility aliases: accuracy now means accuracy among cases
            # where selected-action regret is known.
            "raw_policy_optimal_acc": raw_accuracy,
            "search_optimal_acc": search_accuracy,
            "raw_policy_mean_regret": raw_mean_regret,
            "search_mean_regret": search_mean_regret,
            "solve_rate": _safe_div(self.solved, self.solvable_n),
            "mean_solution_length_gap": _safe_div(self.length_gap_sum, self.length_gap_n),
            "mean_value_mae_raw_moves": _safe_div(self.value_abs_sum, self.value_n),
            "mean_nodes_expanded": _safe_div(self.nodes_expanded, self.n),
            "total_transposition_hits": self.transposition_hits,
            "total_cycle_rejections": self.cycle_rejections,
            "deadlock_states": self.deadlocks,
            "total_model_evaluations": self.model_evaluations,
            "total_model_evaluation_batches": self.model_evaluation_batches,
            "total_model_evaluation_cache_hits":
                self.model_evaluation_cache_hits,
            "mean_model_batch_size": _safe_div(
                self.model_evaluations, self.model_evaluation_batches),
            "total_elapsed_seconds": round(self.elapsed, 4),
        }


def _safe_div(a: float, b: float) -> Optional[float]:
    return (a / b) if b else None


def _action_regret(
    env: Environment,
    oracle: Oracle,
    state: State,
    action,
    *,
    state_value: ValueResult | None = None,
) -> Optional[int]:
    """``regret(s, a) = (1 + V*(child)) - V*(s)``; ``None`` if not exactly known."""
    vs = state_value if state_value is not None else oracle.value(state)
    if not (vs.exact and vs.solvable):
        return None
    child = env.apply_action(state, action)
    vc = oracle.value(child)
    if not vc.exact:
        return None
    if vc.solvable is False:
        return None
    return (1 + vc.value) - vs.value


def evaluate_states(
    env: Environment,
    model,
    encoding_config,
    value_norm,
    states: list[State],
    *,
    budgets: list[int],
    device: str = "cpu",
    c_puct: float = 1.5,
    inference_batch_size: int = 8,
    virtual_loss: float = 1.0,
    seed: int = 0,
    oracle_max_nodes: int = 250_000,
) -> dict[str, Any]:
    """Run the budget sweep and return per-budget aggregate metrics."""
    oracle = Oracle(env, max_nodes=oracle_max_nodes)
    adapter = BlocksortAdapter(env, model, encoding_config, value_norm, device)
    aggs: dict[int, _Agg] = {b: _Agg() for b in budgets}

    for state in states:
        if env.is_terminal(state):
            continue
        analysis = oracle.analyze(state)
        vs_res = ValueResult(
            value=analysis.value,
            exact=analysis.exact,
            solvable=analysis.solvable if not analysis.terminal else True,
        )
        vs = analysis.value if (analysis.exact and analysis.solvable) else None

        # Raw model: priors + value (one evaluation, shared across budgets).
        legal = env.legal_actions(state)
        value_cost = None
        raw_regret = None
        if legal:
            priors, value_cost = adapter.evaluate(state)
            raw_idx = max(range(len(legal)), key=lambda i: (priors[i], -i))
            raw_action = legal[raw_idx]
            raw_regret = _action_regret(env, oracle, state, raw_action,
                                        state_value=vs_res)

        for b in budgets:
            agg = aggs[b]
            agg.n += 1
            cfg = SearchConfig(simulations=b, c_puct=c_puct, temperature=0.0,
                               inference_batch_size=inference_batch_size,
                               virtual_loss=virtual_loss,
                               value_normalization_constant=getattr(value_norm,
                                                                    "constant", 20.0),
                               seed=seed)
            t0 = time.perf_counter()
            res = GraphSearch(adapter, cfg).run(state)
            agg.elapsed += time.perf_counter() - t0

            agg.nodes_expanded += res.stats.nodes_expanded
            agg.transposition_hits += res.stats.transposition_hits
            agg.cycle_rejections += res.stats.cycle_rejections
            agg.model_evaluations += res.stats.model_evaluations
            agg.model_evaluation_batches += \
                res.stats.model_evaluation_batches
            agg.model_evaluation_cache_hits += \
                res.stats.model_evaluation_cache_hits
            if res.termination_reason == "deadlock":
                agg.deadlocks += 1

            # Raw-policy metrics (same regardless of budget).
            if raw_regret is not None:
                agg.raw_regret_sum += raw_regret
                agg.raw_regret_n += 1
                if raw_regret == 0:
                    agg.raw_optimal += 1

            # Search action metrics.
            if res.chosen_action is not None:
                sr = _action_regret(env, oracle, state, res.chosen_action,
                                    state_value=vs_res)
                if sr is not None:
                    agg.search_regret_sum += sr
                    agg.search_regret_n += 1
                    if sr == 0:
                        agg.search_optimal += 1

            # Value MAE (raw moves) using the model estimate.
            if vs is not None and value_cost is not None:
                agg.value_abs_sum += abs(value_cost - vs)
                agg.value_n += 1

            # Solve quality.
            if vs is not None:
                agg.solvable_n += 1
                if res.solved:
                    agg.solved += 1
                    if res.solution_length is not None:
                        agg.length_gap_sum += res.solution_length - vs
                        agg.length_gap_n += 1

    return {str(b): aggs[b].summary() for b in budgets}
