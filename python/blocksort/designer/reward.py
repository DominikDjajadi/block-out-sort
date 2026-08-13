"""Configurable adversarial reward.

reward = w_regret * adversarial_regret
       + w_structural * structural_difficulty
       + w_novelty * novelty
       - w_invalid * invalidity_penalty
       - w_unsolved * unsolved_by_oracle_penalty
       - w_trivial * triviality_penalty

Hard rules (enforced here):

* If the finalized level is **invalid**, no positive term is granted.
* Positive adversarial / structural / novelty reward is granted **only when the
  oracle solves (or verifies) the level**. A level both solvers fail is never
  rewarded -- it incurs the unsolved-by-oracle penalty.

Every component is preserved in :attr:`RewardBreakdown.components`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from .config import RewardConfig
from .metrics import StructuralMetrics
from .roles import SolveOutcome


@dataclass(frozen=True)
class RewardBreakdown:
    total: float
    components: dict

    def __getitem__(self, key):
        return self.components[key]


def _structural_difficulty(cfg: RewardConfig, m: StructuralMetrics) -> float:
    if m.extra_moves is None:
        return 0.0
    n = max(1, m.num_blocks)
    extra = max(0, m.extra_moves)
    first_exit = (m.first_exit_depth or 0) / n
    rehandled = (m.rehandled_blocks or 0) / n
    few = m.few_exitable / n
    return (cfg.s_extra_moves * extra
            + cfg.s_first_exit_depth * first_exit
            + cfg.s_rehandled * rehandled
            + cfg.s_few_exitable * few)


def _cost_gap_regret(cfg: RewardConfig, protagonist: SolveOutcome,
                     oracle: SolveOutcome) -> float:
    if protagonist.cost is None or oracle.cost is None:
        return 0.0
    base = max(1.0, oracle.cost)
    gap = (protagonist.cost - oracle.cost) / base
    gap = max(0.0, min(cfg.cost_gap_cap, gap))
    return cfg.cost_gap_scale * gap


def compute_reward(
    cfg: RewardConfig,
    *,
    valid: bool,
    oracle: SolveOutcome,
    protagonist: SolveOutcome,
    structural: StructuralMetrics,
    novelty: float,
) -> RewardBreakdown:
    comp = {
        "adversarial_regret": 0.0,
        "structural_difficulty": 0.0,
        "novelty": 0.0,
        "invalidity_penalty": 0.0,
        "unsolved_by_oracle_penalty": 0.0,
        "triviality_penalty": 0.0,
        "oracle_solved": bool(oracle.solved),
        "protagonist_solved": bool(protagonist.solved),
        "oracle_exact": bool(oracle.exact),
        "oracle_cost": oracle.cost,
        "protagonist_cost": protagonist.cost,
    }

    if not valid:
        comp["invalidity_penalty"] = 1.0
        total = -cfg.w_invalid * 1.0
        return RewardBreakdown(total=total, components=comp)

    if not oracle.solved:
        # Never reward a level the oracle cannot solve/verify.
        comp["unsolved_by_oracle_penalty"] = 1.0
        total = -cfg.w_unsolved * 1.0
        return RewardBreakdown(total=total, components=comp)

    # Oracle solved -> adversarial reward is allowed.
    if not protagonist.solved:
        regret = 1.0
    else:
        regret = _cost_gap_regret(cfg, protagonist, oracle)
    comp["adversarial_regret"] = regret

    comp["structural_difficulty"] = _structural_difficulty(cfg, structural)
    comp["novelty"] = float(novelty)

    if structural.extra_moves is not None and \
            structural.extra_moves <= cfg.trivial_extra_threshold:
        comp["triviality_penalty"] = 1.0

    total = (cfg.w_regret * comp["adversarial_regret"]
             + cfg.w_structural * comp["structural_difficulty"]
             + cfg.w_novelty * comp["novelty"]
             - cfg.w_trivial * comp["triviality_penalty"])
    return RewardBreakdown(total=total, components=comp)
