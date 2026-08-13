"""Constrained adversarial neural level generator for Block Out Sort.

A *designer* policy builds levels through **validity-preserving** actions derived
from the existing reverse-construction generator (reverse slides + stop), never by
placing arbitrary cells. Levels are scored by an adversarial reward that favours
levels an *oracle* (exact A* or a large search budget) can solve but the bounded
neural *protagonist* struggles with -- with positive adversarial reward granted
only when the oracle confirms solvability.

Scope (per milestone): no arbitrary raw-cell generation, no simultaneous
unrestricted co-training, no distributed RL / multiprocessing / web serving, and
no human-difficulty prediction.
"""

from __future__ import annotations

from .actions import DesignerAction, DesignerActionSpace, STOP
from .config import DesignerConfig, GeneratorConfig, RewardConfig
from .env import DesignerEnv, DesignerState, FinalizeResult

__all__ = [
    "DesignerAction",
    "DesignerActionSpace",
    "STOP",
    "DesignerConfig",
    "GeneratorConfig",
    "RewardConfig",
    "DesignerEnv",
    "DesignerState",
    "FinalizeResult",
]
