"""The Block Out Sort game environment.

This module contains the authoritative Python rules. It mirrors the JavaScript
``Game`` engine, with one deliberate correction: exits require the block to be
able to *fully extrude* off the board collision-free (see :func:`_can_extrude`),
fixing the irregular-shape trailing-cell bug documented in ``ANALYSIS.md``.

The environment is stateless and holds no hidden global state; every method
takes a :class:`~blocksort.state.State` and returns new values.
"""

from __future__ import annotations

from dataclasses import dataclass

from .actions import Action
from .schema import Block, Cell, Direction, Level
from .state import State, canonical_key

SLIDE_REASON_EDGE = "edge"
SLIDE_REASON_BLOCK = "block"


class IllegalActionError(ValueError):
    """Raised when :meth:`Environment.apply_action` receives an illegal move."""


@dataclass(frozen=True)
class SlideResult:
    """Result of :meth:`Environment.compute_slide`.

    - ``steps``: max whole cells the block can advance staying fully on board.
    - ``reason``: ``"edge"`` if stopped by the boundary, ``"block"`` if stopped
      by another block / hole / locked cell.
    - ``can_exit``: ``True`` when the block can fully and legally exit through a
      matching active gate in this direction.
    """

    steps: int
    reason: str
    can_exit: bool


def _region_locked(level: Level, cleared: int, r: int, c: int) -> bool:
    for region in level.locked_regions:
        if cleared >= region.unlock_at:
            continue
        if Cell(r, c) in region.cells:
            return True
    return False


def _exit_active(cleared: int, exit_) -> bool:
    return cleared >= exit_.unlock_at


def _blocked_cell(level: Level, cleared: int, occ: frozenset[Cell], r: int, c: int) -> bool:
    """Whether an in-bounds cell cannot be entered by a moving block."""
    if Cell(r, c) in level.holes:
        return True
    if _region_locked(level, cleared, r, c):
        return True
    if Cell(r, c) in occ:
        return True
    return False


class Environment:
    """Stateless transition function for Block Out Sort."""

    # ----- construction -----

    def initial_state(self, level: Level) -> State:
        """The starting state for ``level`` (all blocks at their definitions)."""
        return State.start(level)

    # ----- queries -----

    def is_block_frozen(self, state: State, block: Block) -> bool:
        return state.cleared < block.unlock_at

    def occupancy(self, state: State, moving: Block) -> frozenset[Cell]:
        """All occupied cells except those of ``moving`` (compared by identity)."""
        cells: set[Cell] = set()
        for block in state.blocks:
            if block is moving:
                continue
            cells.update(block.cells)
        return frozenset(cells)

    def compute_slide(
        self, state: State, block: Block, direction: Direction
    ) -> SlideResult:
        """How far ``block`` can slide in ``direction``, and whether it can exit."""
        level = state.level
        cleared = state.cleared
        occ = self.occupancy(state, block)
        dr, dc = direction.delta
        limit = level.rows + level.cols + 2

        steps = 0
        reason = SLIDE_REASON_EDGE
        while steps < limit:
            off_board = False
            blocked = False
            for cell in block.cells:
                nr = cell.r + dr * (steps + 1)
                nc = cell.c + dc * (steps + 1)
                if not level.in_bounds(nr, nc):
                    off_board = True
                elif _blocked_cell(level, cleared, occ, nr, nc):
                    blocked = True
            if blocked:
                reason = SLIDE_REASON_BLOCK
                break
            if off_board:
                reason = SLIDE_REASON_EDGE
                break
            steps += 1

        can_exit = False
        if reason == SLIDE_REASON_EDGE:
            can_exit = self._can_extrude(state, block, direction, occ)
        return SlideResult(steps=steps, reason=reason, can_exit=can_exit)

    def _gate_covers_lane(
        self, state: State, block: Block, direction: Direction
    ) -> bool:
        """Whether an active matching gate spans every lane the block crosses."""
        level = state.level
        cleared = state.cleared
        vertical = direction.dr != 0
        lane = {cell.c if vertical else cell.r for cell in block.cells}
        for exit_ in level.exits:
            if exit_.edge != direction.edge:
                continue
            if exit_.color != block.color:
                continue
            if not _exit_active(cleared, exit_):
                continue
            if all(exit_.start <= x < exit_.start + exit_.length for x in lane):
                return True
        return False

    def _can_extrude(
        self,
        state: State,
        block: Block,
        direction: Direction,
        occ: frozenset[Cell],
    ) -> bool:
        """Whether ``block`` can fully and legally leave the board.

        Corrected exit rule: a matching active gate must span the block's lane,
        and as the block is translated until *every* cell is off-board, every
        cell that is still inside the board must remain collision-free.
        """
        if not self._gate_covers_lane(state, block, direction):
            return False

        level = state.level
        cleared = state.cleared
        dr, dc = direction.delta
        limit = level.rows + level.cols + 2

        s = 1
        while s <= limit:
            all_off = True
            for cell in block.cells:
                nr = cell.r + dr * s
                nc = cell.c + dc * s
                if level.in_bounds(nr, nc):
                    all_off = False
                    if _blocked_cell(level, cleared, occ, nr, nc):
                        return False
            if all_off:
                return True
            s += 1
        return False

    # ----- actions -----

    def legal_actions(self, state: State) -> list[Action]:
        """Every legal move: each intermediate slide stop and each legal exit."""
        actions: list[Action] = []
        for index, block in enumerate(state.blocks):
            if self.is_block_frozen(state, block):
                continue
            for direction in Direction:
                result = self.compute_slide(state, block, direction)
                for step in range(1, result.steps + 1):
                    actions.append(Action(index, direction, step, exit=False))
                if result.reason == SLIDE_REASON_EDGE and result.can_exit:
                    actions.append(
                        Action(index, direction, result.steps, exit=True)
                    )
        return actions

    def apply_action(self, state: State, action: Action) -> State:
        """Apply ``action`` to ``state``, returning a new state.

        Raises :class:`IllegalActionError` if the action is not legal.
        """
        if not 0 <= action.block_index < len(state.blocks):
            raise IllegalActionError(f"block_index {action.block_index} out of range")
        block = state.blocks[action.block_index]
        if self.is_block_frozen(state, block):
            raise IllegalActionError("cannot move a frozen block")

        result = self.compute_slide(state, block, action.direction)

        if action.exit:
            legal = (
                result.reason == SLIDE_REASON_EDGE
                and result.can_exit
                and action.distance == result.steps
            )
            if not legal:
                raise IllegalActionError("illegal exit action")
            new_blocks = tuple(
                b for i, b in enumerate(state.blocks) if i != action.block_index
            )
            return state.with_blocks(new_blocks)

        if not 1 <= action.distance <= result.steps:
            raise IllegalActionError(
                f"illegal slide distance {action.distance} (max {result.steps})"
            )
        moved = block.translate(action.direction, action.distance)
        new_blocks = list(state.blocks)
        new_blocks[action.block_index] = moved
        return state.with_blocks(tuple(new_blocks))

    # ----- terminal / identity -----

    def is_terminal(self, state: State) -> bool:
        return len(state.blocks) == 0

    def is_deadlock(self, state: State) -> bool:
        """Whether an unsolved state has no legal continuation."""
        return not self.is_terminal(state) and not self.legal_actions(state)

    def canonical_key(self, state: State) -> str:
        return canonical_key(state)
