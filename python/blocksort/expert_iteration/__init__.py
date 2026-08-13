"""Hybrid expert iteration for Block Out Sort.

Each iteration generates states, labels them with the strongest available teacher
(exact A* when it completes within budget, otherwise neural-guided graph search),
mixes them with historical replay data, retrains a model initialized from the
previous checkpoint, and promotes the candidate only if it improves a frozen
validation score. Exact and approximate labels are kept distinguishable and exact
targets are never replaced by approximate search targets.

Scope: no adversarial level generation, RL, web serving, multiprocessing, or
distributed training (see milestone scope).
"""

from __future__ import annotations

from .config import ExpertIterationConfig
from .records import SOURCE_EXACT, SOURCE_EXACT_PATH, SOURCE_SEARCH
from .replay import ReplayBuffer

__all__ = [
    "ExpertIterationConfig",
    "ReplayBuffer",
    "SOURCE_EXACT",
    "SOURCE_EXACT_PATH",
    "SOURCE_SEARCH",
]
