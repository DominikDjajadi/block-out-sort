"""Policy and value target construction from exact oracle analysis."""

from __future__ import annotations

import math
from typing import Optional, Sequence

from ..oracle import ActionAnalysis
from ..training.validation import validate_positive_finite

# Default normalization constant for value targets (documented per record).
DEFAULT_VALUE_NORM_CONSTANT = 20.0
DEFAULT_TEMPERATURE = 1.0


def uniform_optimal_policy(actions: Sequence[ActionAnalysis]) -> list[float]:
    """Uniform over optimal (regret-0) actions; 0 elsewhere.

    ``P(a|s) = 1/|optimal|`` if ``a`` is optimal, else 0.
    """
    optimal = [a for a in actions if a.optimal]
    if not optimal:
        raise ValueError("no optimal action found for a solvable non-terminal state")
    p = 1.0 / len(optimal)
    return [p if a.optimal else 0.0 for a in actions]


def soft_regret_policy(
    actions: Sequence[ActionAnalysis], temperature: float = DEFAULT_TEMPERATURE
) -> list[float]:
    """Softmax over negative regret: ``P(a|s) ∝ exp(-regret(a)/temperature)``.

    Actions whose regret is unknown/infinite (e.g. leading to an unsolvable
    state) receive weight 0. Probabilities are normalized over legal actions.
    """
    temperature = validate_positive_finite(
        "soft-regret policy temperature", temperature)
    weights: list[float] = []
    for a in actions:
        if a.regret is None:
            weights.append(0.0)
        else:
            weights.append(math.exp(-a.regret / temperature))
    total = sum(weights)
    if total <= 0:
        # Should not happen for a solvable non-terminal state (an optimal action
        # has regret 0 -> weight 1). Fall back to uniform-optimal defensively.
        return uniform_optimal_policy(actions)
    return [w / total for w in weights]


def value_target(
    optimal_remaining_moves: int,
    *,
    constant: float = DEFAULT_VALUE_NORM_CONSTANT,
    scheme: str = "neg_over_constant",
) -> dict:
    """Build the value-target block, preserving the raw move count.

    ``normalized_value = -optimal_remaining_moves / constant`` for the default
    ``neg_over_constant`` scheme.
    """
    if scheme != "neg_over_constant":
        raise ValueError(f"unknown value normalization scheme: {scheme}")
    constant = validate_positive_finite(
        "value target normalization constant", constant)
    moves = float(optimal_remaining_moves)
    if not math.isfinite(moves):
        raise ValueError(
            f"optimal_remaining_moves must be finite; got {optimal_remaining_moves!r}")
    return {
        "raw_optimal_moves": optimal_remaining_moves,
        "normalized_value": -moves / constant,
        "normalization": {"scheme": scheme, "constant": constant},
    }
