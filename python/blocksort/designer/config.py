"""Configuration objects for the adversarial level designer."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

from .construction import GeneratorConfig

__all__ = ["GeneratorConfig", "RewardConfig", "DesignerConfig"]


@dataclass(frozen=True)
class RewardConfig:
    """Weights for the configurable adversarial reward.

    reward = w_regret * adversarial_regret
           + w_structural * structural_difficulty
           + w_novelty * novelty
           - w_invalid * invalidity_penalty
           - w_unsolved * unsolved_by_oracle_penalty
           - w_trivial * triviality_penalty

    The adversarial / structural / novelty terms are only granted when the oracle
    confirms the level is solvable (never reward a level both solvers fail).
    """

    w_regret: float = 1.0
    w_structural: float = 0.25
    w_novelty: float = 0.2
    w_invalid: float = 2.0
    w_unsolved: float = 1.0
    w_trivial: float = 0.5

    # When both solvers solve, a finer cost-gap regret term (normalized, capped).
    cost_gap_scale: float = 0.5
    cost_gap_cap: float = 1.0

    # Structural metric weights (only applied when A* optimal is known).
    s_extra_moves: float = 0.15
    s_first_exit_depth: float = 0.1
    s_rehandled: float = 0.1
    s_few_exitable: float = 0.1

    # A level is "trivial" if it has <= this many forced extra moves.
    trivial_extra_threshold: int = 0

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class DesignerConfig:
    """Top-level configuration for designer training / generation."""

    generator: GeneratorConfig = field(default_factory=GeneratorConfig)
    reward: RewardConfig = field(default_factory=RewardConfig)

    mutation_budget: int = 12
    protagonist_simulations: int = 100
    oracle_simulations: int = 1000
    astar_max_nodes: int = 200_000
    search_c_puct: float = 1.5

    seed: int = 42
    device: str = "auto"

    def to_dict(self) -> dict:
        return {
            "generator": asdict(self.generator),
            "reward": self.reward.to_dict(),
            "mutation_budget": self.mutation_budget,
            "protagonist_simulations": self.protagonist_simulations,
            "oracle_simulations": self.oracle_simulations,
            "astar_max_nodes": self.astar_max_nodes,
            "search_c_puct": self.search_c_puct,
            "seed": self.seed,
            "device": self.device,
        }
