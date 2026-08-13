"""Alternating protagonist-designer co-training.

A curriculum loop that alternates between (1) improving the protagonist on
difficult designer-generated levels via the expert-iteration pipeline, (2)
freezing the promoted protagonist, and (3) training the designer against that
stronger protagonist. Generation difficulty adapts to keep levels near the
protagonist's learning frontier (a configurable solve-rate band), and the
training set can be deterministically backfilled with the nearest otherwise
usable levels when strict frontier yield is sparse. The protagonist and
designer are never updated within the same batch.

Scope: no web serving, browser, distributed training, multiprocessing/inference
servers, human-difficulty prediction, or arbitrary raw-cell generation.
"""

from __future__ import annotations

from .config import CoTrainingConfig, CurriculumConfig, CurriculumState
from .curriculum import adapt_curriculum
from .frontier import (
    estimate_solve_rate, frontier_distance, in_frontier,
    select_frontier_backfill)

__all__ = [
    "CoTrainingConfig",
    "CurriculumConfig",
    "CurriculumState",
    "adapt_curriculum",
    "estimate_solve_rate",
    "frontier_distance",
    "in_frontier",
    "select_frontier_backfill",
    "CoTraining",
    "run_cotraining",
]


def __getattr__(name: str):
    if name in {"CoTraining", "run_cotraining"}:
        from .loop import CoTraining, run_cotraining
        return {"CoTraining": CoTraining, "run_cotraining": run_cotraining}[name]
    raise AttributeError(name)
