"""Adaptive difficulty curriculum driven by frontier yield and solve rates."""

from __future__ import annotations

from typing import Any

from .config import CurriculumConfig, CurriculumState


def _clampf(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, round(x, 4)))


def _clampi(x: int, lo: int, hi: int) -> int:
    return max(lo, min(hi, x))


def adapt_curriculum(
    state: CurriculumState,
    mean_solve_rate: float,
    cfg: CurriculumConfig,
    *,
    frontier_acceptance_rate: float | None = None,
    below_frontier_rate: float | None = None,
    above_frontier_rate: float | None = None,
) -> tuple[CurriculumState, dict[str, Any]]:
    """Return ``(new_state, adjustment_record)``.

    Increasing difficulty raises (in order) structural threshold, mutation
    budget, density, then board size. Reducing difficulty unwinds the same
    knobs in reverse. ``adjustment_record`` always captures the trigger even
    when no knob actually moves (e.g. already at a bound or within band).
    """
    changes: dict[str, tuple[Any, Any]] = {}
    new = state
    yield_direction = None
    yield_reason = None
    if frontier_acceptance_rate is not None \
            and frontier_acceptance_rate < cfg.min_frontier_acceptance_rate:
        below = float(below_frontier_rate or 0.0)
        above = float(above_frontier_rate or 0.0)
        if below > above + cfg.frontier_imbalance_margin:
            yield_direction = "reduce"
            yield_reason = (
                f"frontier acceptance {frontier_acceptance_rate:.3f} < "
                f"target {cfg.min_frontier_acceptance_rate:.3f}, with "
                f"too-hard rate {below:.3f} > too-easy rate {above:.3f} "
                f"by more than {cfg.frontier_imbalance_margin:.3f}")
        elif above > below + cfg.frontier_imbalance_margin:
            yield_direction = "increase"
            yield_reason = (
                f"frontier acceptance {frontier_acceptance_rate:.3f} < "
                f"target {cfg.min_frontier_acceptance_rate:.3f}, with "
                f"too-easy rate {above:.3f} > too-hard rate {below:.3f} "
                f"by more than {cfg.frontier_imbalance_margin:.3f}")
        else:
            yield_direction = "hold"
            yield_reason = (
                f"frontier acceptance {frontier_acceptance_rate:.3f} < "
                f"target {cfg.min_frontier_acceptance_rate:.3f}, but "
                f"too-hard rate {below:.3f} and too-easy rate {above:.3f} "
                "are balanced; scalar difficulty is held")

    if yield_direction == "increase" or (
            yield_direction is None
            and mean_solve_rate > cfg.frontier_max_solve_rate):
        direction = "increase"
        reason = yield_reason or (
            f"mean solve rate {mean_solve_rate:.3f} > frontier max "
            f"{cfg.frontier_max_solve_rate:.3f}; levels too easy")
        # Lower the protagonist budget first (makes the protagonist relatively
        # weaker -> harder), then raise structural difficulty.
        ps = _clampi(new.protagonist_simulations - cfg.protagonist_sim_step,
                     cfg.min_protagonist_simulations,
                     cfg.max_protagonist_simulations)
        if ps != new.protagonist_simulations:
            changes["protagonist_simulations"] = (new.protagonist_simulations, ps)
            new = new.replace(protagonist_simulations=ps)
        st = _clampi(new.structural_threshold + cfg.structural_step,
                     cfg.min_structural_threshold, cfg.max_structural_threshold)
        if st != new.structural_threshold:
            changes["structural_threshold"] = (new.structural_threshold, st)
            new = new.replace(structural_threshold=st)
        mb = _clampi(new.mutation_budget + cfg.mutation_step,
                     cfg.min_mutation_budget, cfg.max_mutation_budget)
        if mb != new.mutation_budget:
            changes["mutation_budget"] = (new.mutation_budget, mb)
            new = new.replace(mutation_budget=mb)
        de = _clampf(new.density + cfg.density_step, cfg.min_density, cfg.max_density)
        if de != new.density:
            changes["density"] = (new.density, de)
            new = new.replace(density=de)
        if (cfg.grow_board_when_density_maxed and new.density >= cfg.max_density
                and new.mutation_budget >= cfg.max_mutation_budget):
            rows = _clampi(new.rows + cfg.board_step, cfg.min_board, cfg.max_board)
            cols = _clampi(new.cols + cfg.board_step, cfg.min_board, cfg.max_board)
            if rows != new.rows or cols != new.cols:
                changes["rows"] = (new.rows, rows)
                changes["cols"] = (new.cols, cols)
                new = new.replace(rows=rows, cols=cols)

    elif yield_direction == "reduce" or (
            yield_direction is None
            and mean_solve_rate < cfg.frontier_min_solve_rate):
        direction = "reduce"
        reason = yield_reason or (
            f"mean solve rate {mean_solve_rate:.3f} < frontier min "
            f"{cfg.frontier_min_solve_rate:.3f}; levels too hard")
        # Raise the protagonist budget first (directly increases solve rate),
        # then reduce structural difficulty, then colors, then board size.
        ps = _clampi(new.protagonist_simulations + cfg.protagonist_sim_step,
                     cfg.min_protagonist_simulations,
                     cfg.max_protagonist_simulations)
        if ps != new.protagonist_simulations:
            changes["protagonist_simulations"] = (new.protagonist_simulations, ps)
            new = new.replace(protagonist_simulations=ps)
        de = _clampf(new.density - cfg.density_step, cfg.min_density, cfg.max_density)
        if de != new.density:
            changes["density"] = (new.density, de)
            new = new.replace(density=de)
        mb = _clampi(new.mutation_budget - cfg.mutation_step,
                     cfg.min_mutation_budget, cfg.max_mutation_budget)
        if mb != new.mutation_budget:
            changes["mutation_budget"] = (new.mutation_budget, mb)
            new = new.replace(mutation_budget=mb)
        st = _clampi(new.structural_threshold - cfg.structural_step,
                     cfg.min_structural_threshold, cfg.max_structural_threshold)
        if st != new.structural_threshold:
            changes["structural_threshold"] = (new.structural_threshold, st)
            new = new.replace(structural_threshold=st)
        # Reduce colors when the mutation budget has bottomed out.
        if new.mutation_budget <= cfg.min_mutation_budget:
            cc = _clampi(new.color_count - cfg.color_step,
                         cfg.min_color_count, cfg.max_color_count)
            if cc != new.color_count:
                changes["color_count"] = (new.color_count, cc)
                new = new.replace(color_count=cc)
        # Shrink the board only when density + mutation budget are floored.
        if (cfg.shrink_board_when_floored
                and new.density <= cfg.min_density
                and new.mutation_budget <= cfg.min_mutation_budget):
            rows = _clampi(new.rows - cfg.board_step, cfg.min_board, cfg.max_board)
            cols = _clampi(new.cols - cfg.board_step, cfg.min_board, cfg.max_board)
            if rows != new.rows or cols != new.cols:
                changes["rows"] = (new.rows, rows)
                changes["cols"] = (new.cols, cols)
                new = new.replace(rows=rows, cols=cols)

    else:
        direction = "hold"
        reason = yield_reason or (
            f"mean solve rate {mean_solve_rate:.3f} within frontier band "
            f"[{cfg.frontier_min_solve_rate:.3f}, "
            f"{cfg.frontier_max_solve_rate:.3f}]")

    record = {
        "direction": direction,
        "reason": reason,
        "mean_solve_rate": round(float(mean_solve_rate), 4),
        "frontier_acceptance_rate": (
            round(float(frontier_acceptance_rate), 4)
            if frontier_acceptance_rate is not None else None),
        "below_frontier_rate": (
            round(float(below_frontier_rate), 4)
            if below_frontier_rate is not None else None),
        "above_frontier_rate": (
            round(float(above_frontier_rate), 4)
            if above_frontier_rate is not None else None),
        "changed": bool(changes),
        "changes": {k: {"from": v[0], "to": v[1]} for k, v in changes.items()},
        "before": state.to_dict(),
        "after": new.to_dict(),
    }
    return new, record
