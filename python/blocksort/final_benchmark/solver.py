"""Solver comparison on identical frozen states.

Compares, on the same states: each model's raw greedy policy, graph search using
each model at fixed budgets, and exact A* (where affordable). Reports
optimal-action accuracy, exact action regret, solve rate, solution-length gap
(both on the common set solved by every method and per individual method), value
MAE in moves, runtime, nodes expanded, and transposition hits.
"""

from __future__ import annotations

import time
from typing import Any, Optional

import torch

from ..environment import Environment
from ..oracle import Oracle, ValueResult
from ..state import State
from ..solver import solve_astar
from ..search.config import SearchConfig
from ..search.evaluation import _action_regret
from ..search.graph_search import BlocksortAdapter, GraphSearch
from .common import Protagonist, resolve_device


def _mean(xs: list[float]) -> Optional[float]:
    return (sum(xs) / len(xs)) if xs else None


def compare_solvers(
    states: list[State],
    *,
    models: dict[str, Protagonist],
    budgets: list[int],
    astar_max_nodes: int,
    device: torch.device,
    c_puct: float = 1.5,
    seed: int = 0,
    comparison_budget: int | None = None,
) -> dict[str, Any]:
    if not models:
        raise ValueError("solver comparison requires at least one model")
    if not budgets:
        raise ValueError("solver comparison requires at least one budget")
    if any(isinstance(budget, bool) or not isinstance(budget, int)
           or budget <= 0 for budget in budgets):
        raise ValueError("solver budgets must be positive integers")
    budgets = sorted(set(budgets))
    selected_budget = (
        max(budgets) if comparison_budget is None else comparison_budget)
    if selected_budget not in budgets:
        raise ValueError(
            f"comparison budget {selected_budget} is absent from configured "
            f"budgets {budgets}")
    if any("@" in name for name in models):
        raise ValueError("model identifiers must not contain '@'")
    env = Environment()
    oracle = Oracle(env, max_nodes=astar_max_nodes)
    adapters = {name: BlocksortAdapter(env, p.model, p.enc, p.value_norm, device)
                for name, p in models.items()}
    const = {name: getattr(p.value_norm, "constant", 20.0)
             for name, p in models.items()}

    # method id -> per-state {solved, length} for V*-known states (gap analysis).
    per_state_solved: dict[str, dict[int, tuple[bool, Optional[int]]]] = {}
    # accumulators
    raw_acc = {name: {"opt": 0, "regret_sum": 0.0, "regret_n": 0,
                      "vmae_sum": 0.0, "vmae_n": 0} for name in models}
    search_acc: dict[str, dict[str, Any]] = {}
    for name in models:
        for b in budgets:
            search_acc[f"{name}@{b}"] = {
                "opt": 0, "regret_sum": 0.0, "regret_n": 0,
                "solved": 0, "solvable_n": 0, "runtime": [], "nodes": [],
                "transp": [], "unique": [], "model_evals": [],
                "model_batches": [], "cache_hits": []}
    astar_acc = {"solved": 0, "solvable_n": 0, "runtime": [], "nodes": []}

    per_state_rows: list[dict[str, Any]] = []

    for si, state in enumerate(states):
        if env.is_terminal(state):
            continue
        vs_res = oracle.value(state)
        vs = vs_res.value if (vs_res.exact and vs_res.solvable) else None
        row: dict[str, Any] = {"i": si,
                               "board": f"{state.level.rows}x{state.level.cols}",
                               "vstar": vs, "methods": {}}

        # Exact A* reference (timed).
        t0 = time.perf_counter()
        ares = solve_astar(env, state, max_nodes=astar_max_nodes)
        dt = (time.perf_counter() - t0) * 1000.0
        if vs is not None:
            astar_acc["solvable_n"] += 1
            if ares.solvable is True:
                astar_acc["solved"] += 1
            astar_acc["runtime"].append(dt)
            astar_acc["nodes"].append(ares.states_explored)
            per_state_solved.setdefault("astar", {})[si] = (
                ares.solvable is True, ares.move_count)
        row["methods"]["astar"] = {"solved": ares.solvable is True,
                                   "length": ares.move_count,
                                   "runtime_ms": round(dt, 3),
                                   "nodes": ares.states_explored}

        for name, p in models.items():
            adapter = adapters[name]
            legal = env.legal_actions(state)
            value_cost = None
            raw_regret = None
            if legal:
                priors, value_cost = adapter.evaluate(state)
                raw_idx = max(range(len(legal)), key=lambda i: (priors[i], -i))
                raw_regret = _action_regret(
                    env, oracle, state, legal[raw_idx], state_value=vs_res)
            if raw_regret is not None:
                raw_acc[name]["regret_sum"] += raw_regret
                raw_acc[name]["regret_n"] += 1
                if raw_regret == 0:
                    raw_acc[name]["opt"] += 1
            if vs is not None and value_cost is not None:
                raw_acc[name]["vmae_sum"] += abs(value_cost - vs)
                raw_acc[name]["vmae_n"] += 1

            for b in budgets:
                cfg = SearchConfig(simulations=b, c_puct=c_puct, temperature=0.0,
                                   value_normalization_constant=const[name],
                                   seed=seed)
                gs = GraphSearch(adapter, cfg)
                res = gs.run(state)
                acc = search_acc[f"{name}@{b}"]
                if res.chosen_action is not None:
                    sr = _action_regret(env, oracle, state, res.chosen_action,
                                        state_value=vs_res)
                    if sr is not None:
                        acc["regret_sum"] += sr
                        acc["regret_n"] += 1
                        if sr == 0:
                            acc["opt"] += 1
                acc["runtime"].append(res.stats.elapsed_seconds * 1000.0)
                acc["nodes"].append(res.stats.nodes_expanded)
                acc["transp"].append(res.stats.transposition_hits)
                acc["unique"].append(res.stats.unique_states)
                acc["model_evals"].append(res.stats.model_evaluations)
                acc["model_batches"].append(
                    res.stats.model_evaluation_batches)
                acc["cache_hits"].append(
                    res.stats.model_evaluation_cache_hits)
                if vs is not None:
                    acc["solvable_n"] += 1
                    if res.solved:
                        acc["solved"] += 1
                    per_state_solved.setdefault(f"{name}@{b}", {})[si] = (
                        res.solved, res.solution_length)
                row["methods"][f"{name}@{b}"] = {
                    "solved": res.solved, "length": res.solution_length,
                    "runtime_ms": round(res.stats.elapsed_seconds * 1000.0, 3),
                    "nodes": res.stats.nodes_expanded,
                    "transposition_hits": res.stats.transposition_hits,
                    "model_evaluations": res.stats.model_evaluations,
                    "model_evaluation_batches":
                        res.stats.model_evaluation_batches,
                    "model_evaluation_cache_hits":
                        res.stats.model_evaluation_cache_hits,
                    "termination_reason": res.termination_reason}
        per_state_rows.append(row)

    # Common set: states with V* known and solved by *every* method.
    methods = list(per_state_solved.keys())
    common: Optional[set[int]] = None
    for m in methods:
        solved_set = {i for i, (sv, ln) in per_state_solved[m].items()
                      if sv and ln is not None}
        common = solved_set if common is None else (common & solved_set)
    common = common or set()
    per_state_rows_idx_vstar = {r["i"]: r["vstar"] for r in per_state_rows}

    def _gap_each(m):
        rows = per_state_solved.get(m, {})
        gaps = [ln - per_state_rows_idx_vstar[i]
                for i, (sv, ln) in rows.items()
                if sv and ln is not None and per_state_rows_idx_vstar.get(i) is not None]
        return _mean(gaps)

    def _gap_common(m):
        rows = per_state_solved.get(m, {})
        gaps = [rows[i][1] - per_state_rows_idx_vstar[i] for i in common
                if i in rows and rows[i][0] and rows[i][1] is not None]
        return _mean(gaps)

    report: dict[str, Any] = {
        "schema_version": 2,
        "comparison_metric": "confirmed_optimal_rate",
        "comparison_budget": selected_budget,
        "budgets": budgets,
        "states": len(per_state_rows),
        "common_solved_count": len(common),
        "raw_policy": {}, "search": {}, "astar": {}}
    for name in models:
        a = raw_acc[name]
        known = a["regret_n"]
        total = len(per_state_rows)
        accuracy = (a["opt"] / known) if known else None
        mean_regret = (a["regret_sum"] / known) if known else None
        report["raw_policy"][name] = {
            "optimal_accuracy_known": accuracy,
            "oracle_regret_coverage": (known / total) if total else None,
            "confirmed_optimal_rate":
                (a["opt"] / total) if total else None,
            "mean_regret_known": mean_regret,
            "known_regret_count": known,
            "unknown_regret_count": total - known,
            "total_evaluated_count": total,
            "optimal_acc": accuracy,
            "mean_regret": mean_regret,
            "value_mae_moves": (a["vmae_sum"] / a["vmae_n"]) if a["vmae_n"] else None,
            "n": known}
    for name in models:
        for b in budgets:
            mid = f"{name}@{b}"
            a = search_acc[mid]
            known = a["regret_n"]
            total = len(per_state_rows)
            accuracy = (a["opt"] / known) if known else None
            mean_regret = (a["regret_sum"] / known) if known else None
            report["search"][mid] = {
                "optimal_accuracy_known": accuracy,
                "oracle_regret_coverage": (known / total) if total else None,
                "confirmed_optimal_rate":
                    (a["opt"] / total) if total else None,
                "mean_regret_known": mean_regret,
                "known_regret_count": known,
                "unknown_regret_count": total - known,
                "total_evaluated_count": total,
                "optimal_acc": accuracy,
                "mean_regret": mean_regret,
                "solve_rate": (a["solved"] / a["solvable_n"]) if a["solvable_n"] else None,
                "solution_length_gap_each": _gap_each(mid),
                "solution_length_gap_common": _gap_common(mid),
                "runtime_ms_mean": _mean(a["runtime"]),
                "nodes_expanded_mean": _mean(a["nodes"]),
                "transposition_hits_mean": _mean(a["transp"]),
                "unique_states_mean": _mean(a["unique"]),
                "model_evaluations_mean": _mean(a["model_evals"]),
                "model_evaluation_batches_mean": _mean(a["model_batches"]),
                "model_evaluation_cache_hits_mean": _mean(a["cache_hits"]),
                "mean_model_batch_size": (
                    sum(a["model_evals"]) / sum(a["model_batches"])
                    if sum(a["model_batches"]) else None),
                "n": known}
    report["astar"] = {
        "solve_rate": (astar_acc["solved"] / astar_acc["solvable_n"])
                      if astar_acc["solvable_n"] else None,
        "solution_length_gap_each": 0.0,
        "solution_length_gap_common": 0.0,
        "runtime_ms_mean": _mean(astar_acc["runtime"]),
        "nodes_expanded_mean": _mean(astar_acc["nodes"]),
        "n": astar_acc["solvable_n"]}
    report["note_graph_vs_tree"] = (
        "Search uses a transposition-sharing graph (no separate tree variant "
        "exists; transposition_hits_mean quantifies sharing). A tree-style "
        "re-expansion search was out of the inexpensive scope for this milestone.")
    return report, per_state_rows
