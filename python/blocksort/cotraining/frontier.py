"""Per-level protagonist solve-rate estimation + frontier selection.

A level sits on the protagonist's "learning frontier" when the bounded
protagonist solves it some, but not all, of the time. We estimate a per-level
solve rate by running the bounded protagonist search across several seeds and
averaging the binary solved outcome. Levels are accepted into protagonist
training only when they are valid, oracle-solvable, non-duplicate, and land in
the configured frontier band.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
import math
from typing import Optional

from ..schema import Level
from ..designer.roles import Protagonist
from ..search.seeding import derive_trial_seed, level_search_identity
from .config import CurriculumConfig


@dataclass(frozen=True)
class SolveRateEstimate:
    solve_rate: float
    trials: int
    solved: int
    mean_cost: Optional[float]
    trial_seeds: tuple[int, ...] = ()
    trial_budgets: tuple[int, ...] = ()


def geometric_budget_sweep(
    *,
    center: int,
    trials: int,
    minimum_ratio: float,
    maximum_ratio: float,
    minimum_simulations: int,
    maximum_simulations: int,
) -> tuple[int, ...]:
    """Return a stable log-spaced search-budget sweep around ``center``."""
    if isinstance(center, bool) or not isinstance(center, int) or center <= 0:
        raise ValueError("frontier budget center must be a positive integer")
    if isinstance(trials, bool) or not isinstance(trials, int) or trials <= 0:
        raise ValueError("frontier budget trials must be a positive integer")
    for name, value in (
            ("minimum_ratio", minimum_ratio),
            ("maximum_ratio", maximum_ratio)):
        if (isinstance(value, bool) or not isinstance(value, (int, float))
                or not math.isfinite(float(value)) or value <= 0):
            raise ValueError(f"frontier budget {name} must be finite and positive")
    if minimum_ratio > maximum_ratio:
        raise ValueError(
            "frontier budget minimum_ratio cannot exceed maximum_ratio")
    for name, value in (
            ("minimum_simulations", minimum_simulations),
            ("maximum_simulations", maximum_simulations)):
        if (isinstance(value, bool) or not isinstance(value, int) or value <= 0):
            raise ValueError(
                f"frontier budget {name} must be a positive integer")
    if minimum_simulations > maximum_simulations:
        raise ValueError(
            "frontier budget minimum_simulations cannot exceed "
            "maximum_simulations")

    if trials == 1:
        return (max(minimum_simulations, min(maximum_simulations, center)),)

    low = max(minimum_simulations, min(
        maximum_simulations, int(round(center * minimum_ratio))))
    high = max(minimum_simulations, min(
        maximum_simulations, int(round(center * maximum_ratio))))
    if low == high:
        return (low,) * trials
    log_low = math.log(float(low))
    step = (math.log(float(high)) - log_low) / (trials - 1)
    return tuple(
        int(round(math.exp(log_low + index * step)))
        for index in range(trials)
    )


def estimate_solve_rate(
    protagonist: Protagonist,
    level: Level,
    *,
    trials: int,
    base_seed: int,
    level_identity: str | None = None,
    evaluation_context: str = "cotraining.frontier",
    simulation_budgets: Iterable[int] | None = None,
) -> SolveRateEstimate:
    """Estimate solve probability over explicit, stable per-level trial seeds."""
    if level_identity is None and hasattr(protagonist, "env"):
        level_identity = level_search_identity(protagonist.env, level)
    n = max(1, trials)
    budgets = tuple(simulation_budgets or ())
    if budgets:
        if len(budgets) != n:
            raise ValueError(
                "frontier simulation budget count must equal solve-rate trials")
        if any(isinstance(budget, bool) or not isinstance(budget, int)
               or budget <= 0 for budget in budgets):
            raise ValueError(
                "frontier simulation budgets must be positive integers")
    trial_seeds = tuple(
        derive_trial_seed(
            base_seed,
            trial_index=i,
            level_identity=level_identity,
            evaluation_context=evaluation_context,
        )
        for i in range(n)
    )
    solved = 0
    costs: list[float] = []
    for index, trial_seed in enumerate(trial_seeds):
        if budgets:
            out = protagonist.solve(
                level, seed=trial_seed, simulations=budgets[index])
        else:
            out = protagonist.solve(level, seed=trial_seed)
        if out.solved:
            solved += 1
            if out.cost is not None:
                costs.append(float(out.cost))
    return SolveRateEstimate(
        solve_rate=solved / n,
        trials=n,
        solved=solved,
        mean_cost=(sum(costs) / len(costs)) if costs else None,
        trial_seeds=trial_seeds,
        trial_budgets=budgets,
    )


def in_frontier(solve_rate: float, cfg: CurriculumConfig) -> bool:
    return (cfg.frontier_min_solve_rate <= solve_rate
            <= cfg.frontier_max_solve_rate)


def frontier_distance(solve_rate: float, cfg: CurriculumConfig) -> float:
    """Return distance to the closed frontier band (zero means strict member)."""
    if solve_rate < cfg.frontier_min_solve_rate:
        return cfg.frontier_min_solve_rate - solve_rate
    if solve_rate > cfg.frontier_max_solve_rate:
        return solve_rate - cfg.frontier_max_solve_rate
    return 0.0


def select_frontier_backfill(
    candidates: Iterable[tuple[str, float]],
    *,
    limit: int,
    cfg: CurriculumConfig,
) -> tuple[str, ...]:
    """Select stable nearest-to-frontier candidate identities.

    Callers supply only otherwise usable candidates: valid, non-duplicate, and
    oracle-solvable. Stable identity tie-breaking keeps resumed and repeated
    runs deterministic.
    """
    if limit <= 0:
        return ()
    midpoint = (
        cfg.frontier_min_solve_rate + cfg.frontier_max_solve_rate) / 2.0
    ranked = sorted(
        candidates,
        key=lambda item: (
            frontier_distance(float(item[1]), cfg),
            abs(float(item[1]) - midpoint),
            str(item[0]),
        ),
    )
    return tuple(str(identity) for identity, _rate in ranked[:limit])
