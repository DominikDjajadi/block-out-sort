"""Frozen-split evaluation comparing a candidate vs the previous checkpoint.

All metrics are computed on identical states. Promotion decisions must use the
validation report only; the frozen test report is for monitoring. Solution-length
gap is reported two ways: on the common set of states solved at *every* compared
budget, and on each budget's own solved set.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Optional

import torch

from ..environment import Environment
from ..oracle import Oracle, ValueResult
from ..signature import static_level_signature
from ..solution import serialize_action
from ..state import State
from ..search.config import SearchConfig
from ..search.evaluation import _action_regret
from ..search.graph_search import BlocksortAdapter, GraphSearch
from .promotion import promotion_score


def _board_key(state: State) -> str:
    return f"{state.level.rows}x{state.level.cols}"


def evaluate_checkpoint(
    env: Environment,
    model,
    encoding_config,
    value_norm,
    states: list[State],
    *,
    budgets: list[int],
    oracle: Oracle,
    device,
    c_puct: float = 1.5,
    seed: int = 0,
    precomputed: dict[tuple[str, str], dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Evaluate one model on ``states`` across budgets against the exact oracle."""
    adapter = BlocksortAdapter(env, model, encoding_config, value_norm, device)

    def _precomputed_key(state: State) -> tuple[str, str]:
        return (static_level_signature(state.level), env.canonical_key(state))

    def _action_assessment(
        state: State,
        action,
        vs_res: ValueResult,
    ) -> tuple[Optional[bool], Optional[int]]:
        """Return (optimal classification, exact regret magnitude).

        Listed optimal actions are confirmed optimal.  Absence proves
        non-optimality only when ``classification_complete`` is true; false or
        missing legacy metadata leaves the classification unknown.  Benchmark
        labels contain no stored exact-regret evidence, so numeric regret
        remains unavailable for every precomputed action.
        """
        if precomputed is not None:
            key = _precomputed_key(state)
            entry = precomputed.get(key)
            if entry and vs_res.exact and vs_res.solvable:
                optimal = entry.get("optimal_actions") or []
                if not isinstance(optimal, list):
                    raise ValueError("precomputed optimal_actions must be a list")
                classification_complete = entry.get("classification_complete")
                if (classification_complete is not None
                        and not isinstance(classification_complete, bool)):
                    raise ValueError(
                        "precomputed classification_complete must be boolean")
                ser = serialize_action(state, action)
                if any(opt == ser for opt in optimal):
                    return True, None
                if classification_complete is True:
                    return False, None
                return None, None
        regret = _action_regret(
            env, oracle, state, action, state_value=vs_res)
        return ((regret == 0) if regret is not None else None, regret)

    evaluated_n = 0
    raw_optimal = raw_classification_n = 0
    raw_regret_sum = raw_regret_n = 0
    value_abs = value_n = 0.0
    by_board: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
    by_diff: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))

    # Per-budget accumulators and per-state solve info (for the two gap methods).
    budget_acc = {b: {"opt": 0, "classification_n": 0,
                      "regret_sum": 0.0, "regret_n": 0,
                      "solved": 0, "solved_total": 0,
                      "solvable_n": 0, "deadlocks": 0,
                      "model_evaluations": 0,
                      "model_evaluation_batches": 0,
                      "model_evaluation_cache_hits": 0,
                      "search_elapsed_seconds": 0.0}
                  for b in budgets}
    solved_info: dict[int, dict[int, tuple[bool, Optional[int], Optional[int]]]] = {
        b: {} for b in budgets}
    paired_level_solve_outcomes: list[dict[str, Any]] = []

    for si, state in enumerate(states):
        if env.is_terminal(state):
            continue
        evaluated_n += 1
        paired_row: dict[str, Any] = {
            "static_level_signature": static_level_signature(state.level),
            "budgets": {},
        }
        board_key = _board_key(state)
        by_board[board_key]["total"] += 1
        key = _precomputed_key(state)
        if precomputed and key in precomputed:
            vr = precomputed[key]["value_result"]
            vs_res = ValueResult(
                value=vr.get("value"),
                exact=bool(vr.get("exact")),
                solvable=vr.get("solvable"),
            )
        else:
            vs_res = oracle.value(state)
        vs = vs_res.value if (vs_res.exact and vs_res.solvable) else None

        legal = env.legal_actions(state)
        value_cost = None
        raw_is_optimal = raw_regret = None
        if legal:
            priors, value_cost = adapter.evaluate(state)
            raw_idx = max(range(len(legal)), key=lambda i: (priors[i], -i))
            raw_is_optimal, raw_regret = _action_assessment(
                state, legal[raw_idx], vs_res)

        if raw_is_optimal is not None:
            raw_classification_n += 1
            if raw_is_optimal:
                raw_optimal += 1
        if raw_regret is not None:
            raw_regret_sum += raw_regret
            raw_regret_n += 1
        if vs is not None and value_cost is not None:
            value_abs += abs(value_cost - vs)
            value_n += 1
            bk, dk = board_key, str(vs)
            by_board[bk]["value_mae_sum"] += abs(value_cost - vs)
            by_board[bk]["value_n"] += 1
            by_diff[dk]["total"] += 1
            by_diff[dk]["value_mae_sum"] += abs(value_cost - vs)
            by_diff[dk]["value_n"] += 1
            if raw_is_optimal is not None:
                by_board[bk]["classification_n"] += 1
                by_diff[dk]["classification_n"] += 1
                by_board[bk]["raw_opt"] += raw_is_optimal
                by_diff[dk]["raw_opt"] += raw_is_optimal
            if raw_regret is not None:
                by_board[bk]["regret_n"] += 1
                by_diff[dk]["regret_n"] += 1
                by_board[bk]["regret_sum"] += raw_regret
                by_diff[dk]["regret_sum"] += raw_regret

        for b in budgets:
            cfg = SearchConfig(simulations=b, c_puct=c_puct, temperature=0.0,
                               value_normalization_constant=getattr(value_norm,
                                                                    "constant", 20.0),
                               seed=seed)
            res = GraphSearch(adapter, cfg).run(state)
            acc = budget_acc[b]
            acc["model_evaluations"] += res.stats.model_evaluations
            acc["model_evaluation_batches"] += \
                res.stats.model_evaluation_batches
            acc["model_evaluation_cache_hits"] += \
                res.stats.model_evaluation_cache_hits
            acc["search_elapsed_seconds"] += res.stats.elapsed_seconds
            if res.termination_reason == "deadlock":
                acc["deadlocks"] += 1
            if res.solved:
                # This coverage-safe count deliberately does not depend on the
                # exact oracle. A completed environment trajectory is direct
                # evidence of a solve even when oracle classification times out.
                acc["solved_total"] += 1
            paired_row["budgets"][str(b)] = {"solved": bool(res.solved)}
            if res.chosen_action is not None:
                is_optimal, sr = _action_assessment(
                    state, res.chosen_action, vs_res)
                if is_optimal is not None:
                    acc["classification_n"] += 1
                    if is_optimal:
                        acc["opt"] += 1
                if sr is not None:
                    acc["regret_sum"] += sr
                    acc["regret_n"] += 1
            if vs is not None:
                acc["solvable_n"] += 1
                if res.solved:
                    acc["solved"] += 1
                solved_info[b][si] = (res.solved, res.solution_length, vs)
        paired_level_solve_outcomes.append(paired_row)

    def _safe(a, b):
        return (a / b) if b else None

    budgets_report = {}
    for b in budgets:
        acc = budget_acc[b]
        gap_sum = gap_n = 0.0
        for solved, length, vstar in solved_info[b].values():
            if solved and length is not None and vstar is not None:
                gap_sum += length - vstar
                gap_n += 1
        budgets_report[str(b)] = {
            "total_evaluated_count": evaluated_n,
            "search_optimal_classification_count": acc["classification_n"],
            "search_unknown_classification_count":
                evaluated_n - acc["classification_n"],
            "search_optimal_classification_coverage":
                _safe(acc["classification_n"], evaluated_n),
            "search_exact_regret_count": acc["regret_n"],
            "search_unknown_exact_regret_count":
                evaluated_n - acc["regret_n"],
            "search_exact_regret_coverage":
                _safe(acc["regret_n"], evaluated_n),
            "search_known_regret_count": acc["regret_n"],
            "search_unknown_regret_count": evaluated_n - acc["regret_n"],
            "search_optimal_accuracy_known":
                _safe(acc["opt"], acc["classification_n"]),
            "search_oracle_regret_coverage":
                _safe(acc["regret_n"], evaluated_n),
            "search_confirmed_optimal_count": acc["opt"],
            "search_confirmed_optimal_rate":
                _safe(acc["opt"], evaluated_n),
            "search_mean_regret_known":
                _safe(acc["regret_sum"], acc["regret_n"]),
            # Accuracy uses classification-known actions; numeric regret uses
            # only exact magnitudes.
            "search_optimal_acc":
                _safe(acc["opt"], acc["classification_n"]),
            "search_mean_regret": _safe(acc["regret_sum"], acc["regret_n"]),
            "search_solved_count": acc["solved_total"],
            "search_solve_rate_total":
                _safe(acc["solved_total"], evaluated_n),
            "solve_rate": _safe(acc["solved"], acc["solvable_n"]),
            "solution_length_gap_each": _safe(gap_sum, gap_n),
            "deadlock_states": acc["deadlocks"],
            "model_evaluations": acc["model_evaluations"],
            "model_evaluation_batches": acc["model_evaluation_batches"],
            "model_evaluation_cache_hits":
                acc["model_evaluation_cache_hits"],
            "mean_model_batch_size": _safe(
                acc["model_evaluations"],
                acc["model_evaluation_batches"]),
            "search_elapsed_seconds": acc["search_elapsed_seconds"],
        }

    # Common set solved at every budget (and V* known).
    common = None
    for b in budgets:
        s = {si for si, (solved, length, vstar) in solved_info[b].items()
             if solved and length is not None and vstar is not None}
        common = s if common is None else (common & s)
    common = common or set()
    for b in budgets:
        gap_sum = sum(solved_info[b][si][1] - solved_info[b][si][2] for si in common)
        budgets_report[str(b)]["solution_length_gap_common"] = (
            gap_sum / len(common) if common else None)
    budgets_report["_common_solved_count"] = len(common)

    return {
        "states": evaluated_n,
        "total_evaluated_count": evaluated_n,
        "raw_policy_optimal_classification_count": raw_classification_n,
        "raw_policy_unknown_classification_count":
            evaluated_n - raw_classification_n,
        "raw_policy_optimal_classification_coverage":
            _safe(raw_classification_n, evaluated_n),
        "raw_policy_exact_regret_count": raw_regret_n,
        "raw_policy_unknown_exact_regret_count": evaluated_n - raw_regret_n,
        "raw_policy_exact_regret_coverage":
            _safe(raw_regret_n, evaluated_n),
        "raw_policy_known_regret_count": raw_regret_n,
        "raw_policy_unknown_regret_count": evaluated_n - raw_regret_n,
        "raw_policy_optimal_accuracy_known":
            _safe(raw_optimal, raw_classification_n),
        "raw_policy_oracle_regret_coverage": _safe(raw_regret_n, evaluated_n),
        "raw_policy_confirmed_optimal_count": raw_optimal,
        "raw_policy_confirmed_optimal_rate": _safe(raw_optimal, evaluated_n),
        "raw_policy_mean_regret_known": _safe(raw_regret_sum, raw_regret_n),
        # Accuracy uses classification-known actions; numeric regret uses only
        # exact magnitudes.
        "raw_policy_optimal_acc": _safe(raw_optimal, raw_classification_n),
        "raw_policy_mean_regret": _safe(raw_regret_sum, raw_regret_n),
        "value_mae_moves": _safe(value_abs, value_n),
        # These compact paired outcomes let a promotion gate resample whole
        # levels and enforce per-budget guards without rerunning search.
        "paired_level_solve_outcomes": paired_level_solve_outcomes,
        "budgets": budgets_report,
        "grouped_by_board": _finalize_groups(by_board),
        "grouped_by_difficulty": _finalize_groups(by_diff),
    }


def _finalize_groups(groups: dict[str, dict[str, float]]) -> dict[str, Any]:
    out = {}
    for key, g in sorted(groups.items()):
        value_n = g.get("value_n", 0)
        total = g.get("total", 0)
        classification_n = g.get("classification_n", 0)
        regret_n = g.get("regret_n", 0)
        optimal = g.get("raw_opt", 0.0)
        out[key] = {
            "n": int(value_n),
            "total_evaluated_count": int(total),
            "raw_policy_optimal_classification_count":
                int(classification_n),
            "raw_policy_unknown_classification_count":
                int(total - classification_n),
            "raw_policy_optimal_classification_coverage":
                (classification_n / total) if total else None,
            "raw_policy_exact_regret_count": int(regret_n),
            "raw_policy_unknown_exact_regret_count": int(total - regret_n),
            "raw_policy_exact_regret_coverage":
                (regret_n / total) if total else None,
            "raw_policy_known_regret_count": int(regret_n),
            "raw_policy_unknown_regret_count": int(total - regret_n),
            "raw_policy_optimal_accuracy_known":
                (optimal / classification_n) if classification_n else None,
            "raw_policy_oracle_regret_coverage":
                (regret_n / total) if total else None,
            "raw_policy_confirmed_optimal_count": int(optimal),
            "raw_policy_confirmed_optimal_rate":
                (optimal / total) if total else None,
            "raw_policy_optimal_acc":
                (optimal / classification_n) if classification_n else None,
            "raw_policy_mean_regret_known":
                (g.get("regret_sum", 0.0) / regret_n) if regret_n else None,
            "raw_policy_mean_regret":
                (g.get("regret_sum", 0.0) / regret_n) if regret_n else None,
            "value_mae_moves":
                (g.get("value_mae_sum", 0.0) / value_n) if value_n else None,
        }
    return out
