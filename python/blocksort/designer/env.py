"""Sequential designer environment.

The designer starts from a valid, solvable base level and applies
validity-preserving reverse slides until it submits (``STOP``) or exhausts its
mutation budget. Every reachable level is schema-valid and solvable by
construction; ``finalize`` additionally re-validates and (optionally) verifies
solvability before the level is accepted.

API::

    state      = env.reset(seed)
    actions    = env.legal_actions(state)
    next_state = env.step(state, action)
    result     = env.finalize(state)
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field, replace
from typing import Optional

from ..environment import Environment
from ..schema import Exit, Level
from ..solver import solve_astar
from ..state import State
from ..validation import validate_level
from .actions import STOP, DesignerAction, DesignerActionSpace
from .config import GeneratorConfig
from .construction import apply_reverse_move, build_base_level, reverse_slide_moves


@dataclass(frozen=True)
class DesignerState:
    level: Level
    history: tuple[DesignerAction, ...]
    budget_used: int
    max_budget: int
    stopped: bool
    seed: int

    @property
    def num_mutations(self) -> int:
        return self.budget_used

    @property
    def budget_remaining(self) -> int:
        return max(0, self.max_budget - self.budget_used)

    @property
    def stage_fraction(self) -> float:
        if self.max_budget <= 0:
            return 1.0
        return min(1.0, self.budget_used / self.max_budget)

    def to_solve_state(self, env: Environment) -> State:
        """The puzzle a solver faces: the working level's initial state."""
        return env.initial_state(self.level)


@dataclass(frozen=True)
class FinalizeResult:
    level: Level
    valid: bool
    errors: tuple[str, ...]
    solvable: Optional[bool]        # True / False / None (verification skipped/capped)
    move_count: Optional[int]
    num_blocks: int
    num_mutations: int


class DesignerEnv:
    def __init__(self, generator: Optional[GeneratorConfig] = None, *,
                 mutation_budget: int = 12,
                 encoding=None) -> None:
        from ..training.config import EncodingConfig

        self.env = Environment()
        self.generator = generator or GeneratorConfig()
        self.mutation_budget = mutation_budget
        self.encoding = encoding or EncodingConfig()
        if self.generator.max_blocks > self.encoding.max_blocks:
            raise ValueError(
                "generator max_blocks exceeds checkpoint encoding limit: "
                f"{self.generator.max_blocks} > {self.encoding.max_blocks}")
        self.action_space = DesignerActionSpace(self.encoding)

    # ------------------------------------------------------------------

    def reset(self, seed: int) -> DesignerState:
        """Deterministically build a valid solvable base level for ``seed``.

        If a particular sub-seed fails to produce a base, the next sub-seed is
        tried (deterministically), so ``reset(seed)`` always returns and is
        reproducible.
        """
        for offset in range(10_000):
            rng = random.Random((seed * 2_654_435_761 + offset) & 0xFFFFFFFF)
            level = build_base_level(self.generator, rng, env=self.env)
            if level is None:
                continue
            if validate_level(level):
                continue
            return DesignerState(level=level, history=(), budget_used=0,
                                 max_budget=self.mutation_budget, stopped=False,
                                 seed=seed)
        raise RuntimeError(f"could not build a base level for seed {seed}")

    # ------------------------------------------------------------------

    def legal_moves(self, state: DesignerState):
        if state.stopped or state.budget_remaining <= 0:
            return []
        return reverse_slide_moves(self.env, state.to_solve_state(self.env))

    def legal_actions(self, state: DesignerState) -> list[DesignerAction]:
        actions: list[DesignerAction] = [STOP]
        for move in self.legal_moves(state):
            actions.append(DesignerAction(kind="reverse", anchor=move.anchor,
                                          direction=move.direction,
                                          distance=move.distance))
        return actions

    def legal_mask(self, state: DesignerState) -> list[bool]:
        return self.action_space.legal_mask(self.legal_moves(state), allow_stop=True)

    # ------------------------------------------------------------------

    def step(self, state: DesignerState, action: DesignerAction) -> DesignerState:
        if state.stopped:
            raise ValueError("cannot step a stopped designer state")
        if action.is_stop:
            return replace(state, stopped=True, history=state.history + (action,))
        if state.budget_remaining <= 0:
            raise ValueError("mutation budget exhausted; only STOP is legal")
        if action.kind != "reverse":
            raise ValueError(f"unknown designer action kind: {action.kind!r}")
        new_solve = apply_reverse_move(self.env, state.to_solve_state(self.env),
                                       action.to_move())
        # Reconstruct a Level from the mutated blocks (gates unchanged).
        new_level = Level(name=state.level.name, cols=state.level.cols,
                          rows=state.level.rows, holes=state.level.holes,
                          blocks=new_solve.blocks, exits=state.level.exits,
                          locked_regions=state.level.locked_regions)
        return replace(state, level=new_level, history=state.history + (action,),
                       budget_used=state.budget_used + 1)

    # ------------------------------------------------------------------

    def finalize(self, state: DesignerState, *, verify: bool = True,
                 max_nodes: int = 200_000) -> FinalizeResult:
        level = state.level
        errors = [str(e) for e in validate_level(level)]
        valid = not errors

        solvable: Optional[bool] = None
        move_count: Optional[int] = None
        if verify and valid:
            result = solve_astar(self.env, self.env.initial_state(level),
                                 max_nodes=max_nodes)
            solvable = result.solvable      # True / False / None (capped)
            move_count = result.move_count
        return FinalizeResult(level=level, valid=valid, errors=tuple(errors),
                              solvable=solvable, move_count=move_count,
                              num_blocks=level.total_blocks,
                              num_mutations=state.budget_used)
