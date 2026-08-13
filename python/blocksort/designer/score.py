"""Score a finalized level: run the oracle + protagonist, compute structural
metrics, and assemble the adversarial reward. Shared by training and evaluation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ..environment import Environment
from ..schema import Level
from ..search.seeding import derive_trial_seed, level_search_identity
from .config import RewardConfig
from .env import FinalizeResult
from .metrics import StructuralMetrics, structural_metrics
from .reward import RewardBreakdown, compute_reward
from .roles import Oracle, Protagonist, SolveOutcome


@dataclass(frozen=True)
class ScoredLevel:
    reward: RewardBreakdown
    structural: StructuralMetrics
    oracle: SolveOutcome
    protagonist: SolveOutcome
    valid: bool

    def solver_metrics(self) -> dict[str, Any]:
        return {
            "oracle_method": self.oracle.method,
            "oracle_cost": self.oracle.cost,
            "oracle_exact": self.oracle.exact,
            "oracle_nodes": self.oracle.nodes,
            "protagonist_method": self.protagonist.method,
            "protagonist_cost": self.protagonist.cost,
            "protagonist_nodes": self.protagonist.nodes,
        }

    def oracle_result(self) -> dict[str, Any]:
        return {
            "oracle_solved": self.oracle.solved,
            "oracle_exact": self.oracle.exact,
            "oracle_cost": self.oracle.cost,
            "protagonist_solved": self.protagonist.solved,
            "protagonist_cost": self.protagonist.cost,
        }


def score_level(
    env: Environment,
    finalize: FinalizeResult,
    *,
    protagonist: Protagonist,
    oracle: Oracle,
    reward_cfg: RewardConfig,
    novelty: float,
    seed: int = 0,
    astar_max_nodes: int = 200_000,
    construction_solvable: bool = False,
) -> ScoredLevel:
    level: Level = finalize.level
    valid = finalize.valid

    if not valid:
        # Skip solvers entirely for an invalid level.
        empty = StructuralMetrics(
            num_blocks=level.total_blocks, immediately_exitable=0,
            few_exitable=level.total_blocks, optimal_moves=None, extra_moves=None,
            first_exit_depth=None, distinct_setup_blocks=None,
            rehandled_blocks=None, opening_requires_setup=None)
        no_solve = SolveOutcome(solved=False, cost=None, exact=False, method="skip")
        reward = compute_reward(reward_cfg, valid=False, oracle=no_solve,
                                protagonist=no_solve, structural=empty,
                                novelty=novelty)
        return ScoredLevel(reward=reward, structural=empty, oracle=no_solve,
                           protagonist=no_solve, valid=False)

    identity = level_search_identity(env, level)
    oracle_seed = derive_trial_seed(
        seed, trial_index=0, level_identity=identity,
        evaluation_context="designer.score.oracle")
    protagonist_seed = derive_trial_seed(
        seed, trial_index=0, level_identity=identity,
        evaluation_context="designer.score.protagonist")
    oracle_out, astar_result = oracle.solve_detailed(level, seed=oracle_seed)
    if construction_solvable and not oracle_out.solved and not oracle_out.exact:
        # DesignerEnv begins with a solvable-by-construction base and permits
        # only inverse legal slides. Search exhaustion therefore means
        # "optimal cost unknown", not "unsolved".
        oracle_out = SolveOutcome(
            solved=True, cost=None, exact=False, method="construction_proof",
            nodes=oracle_out.nodes)
    protagonist_out = protagonist.solve(level, seed=protagonist_seed)
    metrics = structural_metrics(env, level, astar_max_nodes=astar_max_nodes,
                                 astar_result=astar_result)
    reward = compute_reward(reward_cfg, valid=True, oracle=oracle_out,
                            protagonist=protagonist_out, structural=metrics,
                            novelty=novelty)
    return ScoredLevel(reward=reward, structural=metrics, oracle=oracle_out,
                       protagonist=protagonist_out, valid=True)
