"""Designer state encoding.

Reuses the protagonist's board encoding (:func:`blocksort.training.encoding.
encode_state` on the working level's initial state) and appends designer-specific
global features describing the construction stage, mutation budget, and causal
structural signals available at every construction step.
"""

from __future__ import annotations

from dataclasses import dataclass
import torch

from ..environment import Environment
from ..schema import Direction
from ..training.config import EncodingConfig
from ..training.encoding import encode_state
from .env import DesignerState

# Number of designer-specific global features appended to the base globals.
DESIGNER_EXTRA_GLOBALS = 8


@dataclass(frozen=True)
class EncodedDesignerState:
    board: torch.Tensor            # [C, H, W]
    global_features: torch.Tensor  # [num_global_features + DESIGNER_EXTRA_GLOBALS]


def num_global_features(encoding: EncodingConfig) -> int:
    return encoding.num_global_features + DESIGNER_EXTRA_GLOBALS


def _immediately_exitable(env: Environment, state) -> int:
    count = 0
    for block in state.blocks:
        for d in Direction:
            s = env.compute_slide(state, block, d)
            if s.reason == "edge" and s.can_exit:
                count += 1
                break
    return count


def encode_designer_state(
    env: Environment,
    state: DesignerState,
    encoding: EncodingConfig,
) -> EncodedDesignerState:
    solve_state = state.to_solve_state(env)
    encoded = encode_state(env, solve_state, encoding)

    max_blocks = max(1, encoding.max_blocks)
    n = solve_state.total_blocks
    imm = _immediately_exitable(env, solve_state)
    setup_fraction = (n - imm) / max(1, n)
    # Replaying every reverse mutation and then exiting every block supplies a
    # constructive (not necessarily optimal) solution upper bound.
    construction_solution_upper_bound = n + state.budget_used
    extras = torch.tensor([
        state.stage_fraction,
        state.budget_used / max(1, state.max_budget),
        state.budget_remaining / max(1, state.max_budget),
        n / max_blocks,
        imm / max_blocks,
        (n - imm) / max_blocks,
        setup_fraction,
        construction_solution_upper_bound / max(1, 2 * max_blocks),
    ], dtype=torch.float32)

    global_features = torch.cat([encoded.global_features, extras], dim=0)
    return EncodedDesignerState(board=encoded.board, global_features=global_features)
