"""Dependency-light, solvable-by-construction level generation helpers."""

from __future__ import annotations

import random
from typing import Optional

from .designer.config import GeneratorConfig
from .designer.construction import (
    apply_reverse_move,
    build_base_level,
    reverse_slide_moves,
)
from .environment import Environment
from .schema import Level
from .validation import validate_level


def random_level(
    env: Environment,
    gen_cfg: GeneratorConfig,
    rng: random.Random,
    *,
    reverse_depth: int,
    tries: int = 50,
) -> Optional[Level]:
    """Generate a schema-valid level with a reverse-construction solution."""
    for _ in range(tries):
        base = build_base_level(gen_cfg, rng, env=env)
        if base is None:
            continue
        if validate_level(base):
            continue
        state = env.initial_state(base)
        for _ in range(max(0, reverse_depth)):
            moves = reverse_slide_moves(env, state)
            if not moves:
                break
            state = apply_reverse_move(env, state, rng.choice(moves))
        return Level(
            name=base.name,
            cols=base.cols,
            rows=base.rows,
            holes=base.holes,
            blocks=state.blocks,
            exits=base.exits,
            locked_regions=base.locked_regions,
        )
    return None
