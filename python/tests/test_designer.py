"""Tests for the constrained adversarial level designer.

Covers designer action validity, solvability preservation, legal-action masking,
deterministic reset, reward-component calculation (incl. the oracle-must-solve
rule and positive regret), duplicate detection, replay persistence, model
forward/backward, one PPO update, a deterministic end-to-end training smoke run,
and generated-solution replay.
"""

from __future__ import annotations

import random
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from blocksort import level_from_dict
from blocksort.environment import Environment
from blocksort.schema import Cell, Direction
from blocksort.solver import SolveResult, TIME_LIMIT, solve_astar
from blocksort.solution import verify_solution
from blocksort.validation import validate_level

from blocksort.training.config import EncodingConfig, ModelConfig, ValueNormConfig
from blocksort.training.model import PolicyValueNet
from blocksort.training.checkpoint import save_checkpoint
from blocksort.training.experiment_identity import UnsupportedResumeError
from blocksort.model_identity import model_state_sha256

from blocksort.designer.config import GeneratorConfig, RewardConfig
from blocksort.designer.env import DesignerEnv, DesignerState, FinalizeResult
from blocksort.designer.actions import (STOP, STOP_INDEX, DesignerAction,
                                        DesignerActionSpace)
from blocksort.designer.construction import (ReverseMove, apply_reverse_move,
                                             reverse_slide_moves)
from blocksort.designer.model import DesignerModelConfig, DesignerNet
from blocksort.designer.checkpoint import (
    DESIGNER_CHECKPOINT_VERSION,
    designer_from_checkpoint,
    load_designer,
    save_designer,
)
from blocksort.designer.metrics import StructuralMetrics, structural_metrics
from blocksort.designer.reward import RewardBreakdown, compute_reward
from blocksort.designer.roles import SolveOutcome
from blocksort.designer.replay import (LevelReplayBuffer, build_level_record,
                                       level_fingerprint)
from blocksort.designer.ppo import (
    Episode,
    PPOConfig,
    Step,
    _sample_seeded_probabilities,
    ppo_update,
    rollout_episode,
)
from blocksort.designer import train as train_mod
from blocksort.designer import pretrain as pretrain_mod
from blocksort.designer import score as score_mod
from blocksort.designer import roles as roles_mod


ENC = EncodingConfig()


def test_generator_block_limit_must_fit_checkpoint_encoding():
    assert GeneratorConfig().max_blocks == ENC.max_blocks
    with pytest.raises(ValueError, match="generator max_blocks.*encoding"):
        DesignerEnv(
            GeneratorConfig(max_blocks=ENC.max_blocks + 1),
            encoding=ENC,
        )


def test_cpu_probability_sampling_preserves_seeded_cdf_semantics():
    probabilities = [0.1, 0.0, 0.3, 0.6]
    legal_mask = [True, False, True, True]
    expected_rng = random.Random(91)
    actual_rng = random.Random(91)

    def legacy_sample(rng):
        threshold = rng.random()
        cumulative = 0.0
        selected = 3
        for index, probability in enumerate(probabilities):
            if probability <= 0:
                continue
            cumulative += probability
            if threshold <= cumulative:
                selected = index
                break
        return selected

    assert [
        _sample_seeded_probabilities(
            probabilities, legal_mask, actual_rng)
        for _ in range(50)
    ] == [legacy_sample(expected_rng) for _ in range(50)]


def test_cpu_probability_sampling_falls_back_to_last_legal_action():
    assert _sample_seeded_probabilities(
        [0.0, 0.0, 0.1],
        [True, False, True],
        random.Random(0),
    ) == 2


@pytest.fixture(scope="module")
def env():
    return DesignerEnv(GeneratorConfig(rows=6, cols=6, color_count=3),
                       mutation_budget=10, encoding=ENC)


# ----------------------------------------------------------------------
# Env: validity, solvability, masking, determinism
# ----------------------------------------------------------------------

def test_reset_deterministic(env):
    a = env.reset(123)
    b = env.reset(123)
    assert a.level == b.level
    assert not validate_level(a.level)


def test_actions_are_valid_and_preserve_validity(env):
    state = env.reset(5)
    rng = random.Random(0)
    for _ in range(8):
        actions = env.legal_actions(state)
        assert STOP in actions
        reverse = [a for a in actions if a.kind == "reverse"]
        if not reverse:
            break
        action = rng.choice(reverse)
        state = env.step(state, action)
        assert not validate_level(state.level)   # every reachable level is valid


def test_solvability_preserved(env):
    base = env.reset(9)
    e = Environment()
    assert solve_astar(e, e.initial_state(base.level), max_nodes=200_000).solvable
    state = base
    rng = random.Random(3)
    for _ in range(10):
        reverse = [a for a in env.legal_actions(state) if a.kind == "reverse"]
        if not reverse:
            break
        state = env.step(state, rng.choice(reverse))
    result = solve_astar(e, e.initial_state(state.level), max_nodes=200_000)
    assert result.solvable is True


def test_legal_mask_matches_actions_and_roundtrips(env):
    state = env.reset(11)
    space = env.action_space
    mask = env.legal_mask(state)
    assert mask[STOP_INDEX] is True
    legal = env.legal_actions(state)
    for action in legal:
        idx = space.index_of(action)
        assert mask[idx] is True
        back = space.from_index(idx)
        if action.kind == "reverse":
            assert (back.anchor == action.anchor
                    and back.direction == action.direction
                    and back.distance == action.distance)
    # mask True count equals the number of distinct legal action indices.
    assert sum(mask) == len({space.index_of(a) for a in legal})


def _ordinary_but_reverse_illegal_case():
    forward_env = Environment()
    level = level_from_dict({
        "name": "reverse-boundary",
        "cols": 3,
        "rows": 3,
        "blocks": [{"color": "red", "cells": [[1, 1]]}],
        "exits": [{"edge": "top", "start": 1, "length": 1, "color": "red"}],
    })
    state = forward_env.initial_state(level)
    move = ReverseMove(anchor=Cell(1, 1), direction=Direction.RIGHT, distance=1)
    return forward_env, level, state, move


def test_apply_reverse_move_rejects_ordinary_slide_outside_reverse_set():
    forward_env, _level, state, move = _ordinary_but_reverse_illegal_case()
    original_key = forward_env.canonical_key(state)

    assert forward_env.compute_slide(
        state, state.blocks[0], move.direction).steps >= move.distance
    assert move not in reverse_slide_moves(forward_env, state)
    with pytest.raises(ValueError, match="legal reverse mutation"):
        apply_reverse_move(forward_env, state, move)
    assert forward_env.canonical_key(state) == original_key


def test_designer_step_rejects_externally_constructed_reverse_illegal_action():
    _forward_env, level, _solve_state, move = \
        _ordinary_but_reverse_illegal_case()
    env = DesignerEnv(
        GeneratorConfig(rows=3, cols=3, color_count=1),
        mutation_budget=1,
        encoding=ENC,
    )
    state = DesignerState(
        level=level,
        history=(),
        budget_used=0,
        max_budget=1,
        stopped=False,
        seed=0,
    )
    action = DesignerAction(
        kind="reverse",
        anchor=move.anchor,
        direction=move.direction,
        distance=move.distance,
    )

    with pytest.raises(ValueError, match="legal reverse mutation"):
        env.step(state, action)
    assert state.level == level
    assert state.history == ()
    assert state.budget_used == 0


def test_all_enumerated_reverse_moves_apply_and_malformed_moves_fail(env):
    designer_state = env.reset(23)
    solve_state = designer_state.to_solve_state(env.env)
    legal_moves = reverse_slide_moves(env.env, solve_state)
    assert legal_moves
    for move in legal_moves:
        result = apply_reverse_move(env.env, solve_state, move)
        assert result != solve_state

    anchor = legal_moves[0].anchor
    with pytest.raises(ValueError, match="no block anchored"):
        apply_reverse_move(
            env.env, solve_state,
            ReverseMove(Cell(-1, -1), Direction.RIGHT, 1))
    with pytest.raises(ValueError, match="distance"):
        apply_reverse_move(
            env.env, solve_state,
            ReverseMove(anchor, legal_moves[0].direction, 0))
    with pytest.raises(ValueError, match="direction"):
        apply_reverse_move(
            env.env, solve_state,
            ReverseMove(anchor, None, 1))


def test_budget_exhaustion_forces_stop(env):
    state = env.reset(15)
    steps = 0
    rng = random.Random(1)
    while state.budget_remaining > 0 and steps < 50:
        reverse = [a for a in env.legal_actions(state) if a.kind == "reverse"]
        if not reverse:
            break
        state = env.step(state, rng.choice(reverse))
        steps += 1
    if state.budget_remaining == 0:
        assert env.legal_actions(state) == [STOP]


# ----------------------------------------------------------------------
# Reward components
# ----------------------------------------------------------------------

def _metrics(extra=3, num_blocks=6):
    return StructuralMetrics(
        num_blocks=num_blocks, immediately_exitable=1,
        few_exitable=num_blocks - 1, optimal_moves=num_blocks + extra,
        extra_moves=extra, first_exit_depth=2, distinct_setup_blocks=2,
        rehandled_blocks=1, opening_requires_setup=1.0)


def test_no_positive_reward_when_oracle_fails():
    cfg = RewardConfig()
    oracle = SolveOutcome(solved=False, cost=None, exact=True, method="astar")
    prot = SolveOutcome(solved=False, cost=None, exact=False, method="search")
    rb = compute_reward(cfg, valid=True, oracle=oracle, protagonist=prot,
                        structural=_metrics(), novelty=1.0)
    assert rb["adversarial_regret"] == 0.0
    assert rb["unsolved_by_oracle_penalty"] == 1.0
    assert rb.total <= 0.0


def test_positive_regret_when_oracle_solves_protagonist_fails():
    cfg = RewardConfig()
    oracle = SolveOutcome(solved=True, cost=10.0, exact=True, method="astar")
    prot = SolveOutcome(solved=False, cost=None, exact=False, method="search")
    rb = compute_reward(cfg, valid=True, oracle=oracle, protagonist=prot,
                        structural=_metrics(), novelty=0.0)
    assert rb["adversarial_regret"] == 1.0
    assert rb.total > 0.0


def test_invalid_level_only_penalized():
    cfg = RewardConfig()
    solved = SolveOutcome(solved=True, cost=5.0, exact=True, method="astar")
    rb = compute_reward(cfg, valid=False, oracle=solved, protagonist=solved,
                        structural=_metrics(), novelty=1.0)
    assert rb["invalidity_penalty"] == 1.0
    assert rb["adversarial_regret"] == 0.0
    assert rb.total == pytest.approx(-cfg.w_invalid)


def test_triviality_penalty_applied_when_no_extra_moves():
    cfg = RewardConfig()
    oracle = SolveOutcome(solved=True, cost=6.0, exact=True, method="astar")
    prot = SolveOutcome(solved=True, cost=6.0, exact=False, method="search")
    rb = compute_reward(cfg, valid=True, oracle=oracle, protagonist=prot,
                        structural=_metrics(extra=0, num_blocks=6), novelty=0.0)
    assert rb["triviality_penalty"] == 1.0


def test_both_solve_uses_cost_gap_regret():
    cfg = RewardConfig()
    oracle = SolveOutcome(solved=True, cost=10.0, exact=True, method="astar")
    prot = SolveOutcome(solved=True, cost=20.0, exact=False, method="search")
    rb = compute_reward(cfg, valid=True, oracle=oracle, protagonist=prot,
                        structural=_metrics(), novelty=0.0)
    assert 0.0 < rb["adversarial_regret"] <= cfg.cost_gap_scale * cfg.cost_gap_cap


def test_designer_trial_seed_depends_on_level_and_solver_role(monkeypatch):
    env = Environment()
    base = {
        "name": "designer-seed",
        "cols": 4,
        "rows": 4,
        "blocks": [{"color": "red", "cells": [[1, 1]]}],
        "exits": [{"edge": "left", "start": 1, "length": 1, "color": "red"}],
    }
    moved = {
        **base,
        "blocks": [{"color": "red", "cells": [[1, 2]]}],
    }
    calls = []

    class SpyOracle:
        def solve_detailed(self, level, *, seed=0):
            calls.append(("oracle", level.blocks[0].cells, seed))
            return SolveOutcome(True, 1.0, True, "astar"), None

    class SpyProtagonist:
        def solve(self, level, *, seed=0):
            calls.append(("protagonist", level.blocks[0].cells, seed))
            return SolveOutcome(False, None, False, "search")

    monkeypatch.setattr(score_mod, "structural_metrics",
                        lambda *args, **kwargs: _metrics(extra=1, num_blocks=1))

    def evaluate(level):
        finalize = FinalizeResult(
            level=level, valid=True, errors=(), solvable=True,
            move_count=1, num_blocks=1, num_mutations=0)
        score_mod.score_level(
            env, finalize, protagonist=SpyProtagonist(), oracle=SpyOracle(),
            reward_cfg=RewardConfig(), novelty=1.0, seed=23)
        return calls[-2:]

    a_first = evaluate(level_from_dict(base))
    a_again = evaluate(level_from_dict(base))
    b = evaluate(level_from_dict(moved))
    assert [entry[2] for entry in a_first] == [entry[2] for entry in a_again]
    assert a_first[0][2] != a_first[1][2]
    assert [entry[2] for entry in a_first] != [entry[2] for entry in b]


def test_construction_proof_converts_only_inconclusive_oracle_outcome(
        monkeypatch):
    env = Environment()
    level = level_from_dict({
        "name": "construction-proof",
        "cols": 4,
        "rows": 4,
        "blocks": [{"color": "red", "cells": [[1, 1]]}],
        "exits": [{
            "edge": "left", "start": 1, "length": 1, "color": "red",
        }],
    })
    finalize = FinalizeResult(
        level=level, valid=True, errors=(), solvable=True, move_count=None,
        num_blocks=1, num_mutations=1)

    class NoSolveProtagonist:
        def solve(self, _level, *, seed=0):
            return SolveOutcome(False, None, False, "search")

    class InconclusiveOracle:
        def solve_detailed(self, _level, *, seed=0):
            return SolveOutcome(
                False, None, False, "astar_exhausted", nodes=17), object()

    class ContradictingExactOracle:
        def solve_detailed(self, _level, *, seed=0):
            return SolveOutcome(False, None, True, "astar", nodes=19), object()

    unknown_metrics = StructuralMetrics(
        num_blocks=1, immediately_exitable=1, few_exitable=0,
        optimal_moves=None, extra_moves=None, first_exit_depth=None,
        distinct_setup_blocks=None, rehandled_blocks=None,
        opening_requires_setup=None)
    monkeypatch.setattr(
        score_mod, "structural_metrics",
        lambda *args, **kwargs: unknown_metrics)
    proven = score_mod.score_level(
        env, finalize, protagonist=NoSolveProtagonist(),
        oracle=InconclusiveOracle(), reward_cfg=RewardConfig(), novelty=0.0,
        construction_solvable=True)
    contradiction = score_mod.score_level(
        env, finalize, protagonist=NoSolveProtagonist(),
        oracle=ContradictingExactOracle(), reward_cfg=RewardConfig(),
        novelty=0.0, construction_solvable=True)

    assert proven.oracle == SolveOutcome(
        True, None, False, "construction_proof", nodes=17)
    assert proven.reward["unsolved_by_oracle_penalty"] == 0.0
    assert contradiction.oracle.solved is False
    assert contradiction.oracle.exact is True
    assert contradiction.reward["unsolved_by_oracle_penalty"] == 1.0


def test_oracle_time_cap_can_skip_neural_fallback(monkeypatch):
    env = Environment()
    level = level_from_dict({
        "name": "bounded-oracle",
        "cols": 4,
        "rows": 4,
        "blocks": [{"color": "red", "cells": [[1, 1]]}],
        "exits": [{
            "edge": "left", "start": 1, "length": 1, "color": "red",
        }],
    })
    observed = {}
    exhausted = SolveResult(
        solvable=None, optimal=False, exhausted=True, move_count=None,
        actions=None, states_explored=23, states_generated=31,
        duplicate_states=4, max_frontier_size=8, elapsed_seconds=0.25,
        termination_reason=TIME_LIMIT, serialized_actions=None)

    def fake_astar(_env, _state, *, max_nodes, time_limit_seconds):
        observed.update({
            "max_nodes": max_nodes,
            "time_limit_seconds": time_limit_seconds,
        })
        return exhausted

    class UnexpectedGraphSearch:
        def __init__(self, *args, **kwargs):
            raise AssertionError("bounded construction scoring must not fall back")

    monkeypatch.setattr(roles_mod, "solve_astar", fake_astar)
    monkeypatch.setattr(roles_mod, "GraphSearch", UnexpectedGraphSearch)
    oracle = roles_mod.Oracle(
        env, object(), ENC, ValueNormConfig(), torch.device("cpu"),
        astar_max_nodes=1234, astar_time_limit_seconds=0.25,
        fallback_on_astar_exhaustion=False)

    outcome, result = oracle.solve_detailed(level)

    assert observed == {"max_nodes": 1234, "time_limit_seconds": 0.25}
    assert result is exhausted
    assert outcome == SolveOutcome(
        False, None, False, "astar_exhausted", nodes=23)


@pytest.mark.parametrize(
    ("rate", "expected"),
    [
        (0.0, 0.0),
        (0.1, 0.5),
        (0.2, 1.0),
        (0.4, 1.0),
        (0.7, 1.0),
        (0.85, 0.5),
        (1.0, 0.0),
    ],
)
def test_frontier_alignment_score_has_plateau_and_taper(rate, expected):
    assert train_mod.frontier_alignment_score(
        rate, minimum=0.2, maximum=0.7) == pytest.approx(expected)


def test_generated_level_frontier_reward_uses_repeated_trials(monkeypatch):
    env = Environment()
    level = level_from_dict({
        "name": "frontier-reward",
        "cols": 4,
        "rows": 4,
        "blocks": [{"color": "red", "cells": [[1, 1]]}],
        "exits": [{
            "edge": "left", "start": 1, "length": 1, "color": "red",
        }],
    })
    finalize = FinalizeResult(
        level=level, valid=True, errors=(), solvable=True, move_count=None,
        num_blocks=1, num_mutations=1)
    trial_outcomes = iter((
        SolveOutcome(False, None, False, "search"),
        SolveOutcome(True, 1.0, False, "search"),
        SolveOutcome(False, None, False, "search"),
        SolveOutcome(True, 1.0, False, "search"),
        SolveOutcome(False, None, False, "search"),
    ))
    observed_seeds = []
    observed_budgets = []

    class TrialProtagonist:
        env = env

        def solve(self, _level, *, seed=0, simulations=None):
            observed_seeds.append(seed)
            observed_budgets.append(simulations)
            return next(trial_outcomes)

    oracle_outcome = SolveOutcome(True, None, False, "construction_proof")
    metrics = StructuralMetrics(
        num_blocks=1, immediately_exitable=0, few_exitable=1,
        optimal_moves=None, extra_moves=None, first_exit_depth=None,
        distinct_setup_blocks=None, rehandled_blocks=None,
        opening_requires_setup=None)

    def fake_score(_env, _finalize, *, protagonist, reward_cfg, **kwargs):
        protagonist_outcome = protagonist.solve(level)
        reward = compute_reward(
            reward_cfg, valid=True, oracle=oracle_outcome,
            protagonist=protagonist_outcome, structural=metrics, novelty=0.0)
        return score_mod.ScoredLevel(
            reward=reward, structural=metrics, oracle=oracle_outcome,
            protagonist=protagonist_outcome, valid=True)

    monkeypatch.setattr(train_mod, "score_level", fake_score)
    scored = train_mod._score_generated_level(
        env, finalize, protagonist=TrialProtagonist(), oracle=object(),
        reward_cfg=RewardConfig(), novelty=0.0, seed=31,
        astar_max_nodes=1, frontier_solve_rate_trials=5,
        frontier_min_solve_rate=0.2, frontier_max_solve_rate=0.7,
        frontier_alignment_weight=1.25,
        evaluation_context="test.designer.frontier",
        frontier_simulation_budgets=(25, 50, 100, 200, 400))

    assert len(observed_seeds) == len(set(observed_seeds)) == 5
    assert observed_budgets == [25, 50, 100, 200, 400]
    assert scored.reward["frontier_solve_rate"] == pytest.approx(0.4)
    assert scored.reward["frontier_in_band"] is True
    assert scored.reward["frontier_alignment"] == pytest.approx(1.0)
    assert scored.reward["frontier_alignment_bonus"] == pytest.approx(1.25)
    assert scored.reward["frontier_extremity_penalty"] == pytest.approx(0.0)
    assert scored.reward["frontier_alignment_adjustment"] == pytest.approx(1.25)
    assert scored.reward["frontier_simulation_budgets"] == \
        [25, 50, 100, 200, 400]
    assert scored.reward.total == pytest.approx(2.25)
    aggregate = train_mod._aggregate([
        SimpleNamespace(scored=scored, reward=scored.reward.total)])
    assert aggregate["mean_frontier_solve_rate"] == pytest.approx(0.4)
    assert aggregate["frontier_in_band_rate"] == pytest.approx(1.0)
    assert aggregate["mean_frontier_alignment"] == pytest.approx(1.0)
    assert aggregate["frontier_evaluated_count"] == 1


def test_frontier_reward_penalizes_always_failed_extreme(monkeypatch):
    env = Environment()
    level = level_from_dict({
        "name": "frontier-extreme",
        "cols": 4,
        "rows": 4,
        "blocks": [{"color": "red", "cells": [[1, 1]]}],
        "exits": [{
            "edge": "left", "start": 1, "length": 1, "color": "red",
        }],
    })
    finalize = FinalizeResult(
        level=level, valid=True, errors=(), solvable=True, move_count=None,
        num_blocks=1, num_mutations=1)
    protagonist_outcome = SolveOutcome(False, None, False, "search")
    oracle_outcome = SolveOutcome(True, None, False, "construction_proof")
    metrics = StructuralMetrics(
        num_blocks=1, immediately_exitable=0, few_exitable=1,
        optimal_moves=None, extra_moves=None, first_exit_depth=None,
        distinct_setup_blocks=None, rehandled_blocks=None,
        opening_requires_setup=None)
    base_reward = compute_reward(
        RewardConfig(), valid=True, oracle=oracle_outcome,
        protagonist=protagonist_outcome, structural=metrics, novelty=0.0)

    class NeverSolves:
        env = env

        def solve(self, _level, *, seed=0):
            return protagonist_outcome

    def fake_score(*args, **kwargs):
        return score_mod.ScoredLevel(
            reward=base_reward, structural=metrics, oracle=oracle_outcome,
            protagonist=protagonist_outcome, valid=True)

    monkeypatch.setattr(train_mod, "score_level", fake_score)
    scored = train_mod._score_generated_level(
        env, finalize, protagonist=NeverSolves(), oracle=object(),
        reward_cfg=RewardConfig(), novelty=0.0, seed=37,
        astar_max_nodes=1, frontier_solve_rate_trials=5,
        frontier_min_solve_rate=0.2, frontier_max_solve_rate=0.7,
        frontier_alignment_weight=1.0,
        evaluation_context="test.designer.extreme")

    assert scored.reward["frontier_solve_rate"] == pytest.approx(0.0)
    assert scored.reward["frontier_alignment_bonus"] == pytest.approx(0.0)
    assert scored.reward["frontier_extremity_penalty"] == pytest.approx(1.0)
    assert scored.reward["frontier_alignment_adjustment"] == pytest.approx(-1.0)
    assert scored.reward.total == pytest.approx(base_reward.total - 1.0)


def test_frontier_bonus_never_rewards_oracle_unverified_level(monkeypatch):
    env = Environment()
    level = level_from_dict({
        "name": "frontier-unverified",
        "cols": 4,
        "rows": 4,
        "blocks": [{"color": "red", "cells": [[1, 1]]}],
        "exits": [{
            "edge": "left", "start": 1, "length": 1, "color": "red",
        }],
    })
    finalize = FinalizeResult(
        level=level, valid=True, errors=(), solvable=None, move_count=None,
        num_blocks=1, num_mutations=1)
    protagonist_outcome = SolveOutcome(False, None, False, "search")
    oracle_outcome = SolveOutcome(False, None, True, "astar")
    metrics = StructuralMetrics(
        num_blocks=1, immediately_exitable=0, few_exitable=1,
        optimal_moves=None, extra_moves=None, first_exit_depth=None,
        distinct_setup_blocks=None, rehandled_blocks=None,
        opening_requires_setup=None)

    class TrialProtagonist:
        env = env

        def solve(self, _level, *, seed=0):
            return protagonist_outcome

    def fake_score(_env, _finalize, *, reward_cfg, **kwargs):
        reward = compute_reward(
            reward_cfg, valid=True, oracle=oracle_outcome,
            protagonist=protagonist_outcome, structural=metrics, novelty=0.0)
        return score_mod.ScoredLevel(
            reward=reward, structural=metrics, oracle=oracle_outcome,
            protagonist=protagonist_outcome, valid=True)

    monkeypatch.setattr(train_mod, "score_level", fake_score)
    scored = train_mod._score_generated_level(
        env, finalize, protagonist=TrialProtagonist(), oracle=object(),
        reward_cfg=RewardConfig(), novelty=0.0, seed=31,
        astar_max_nodes=1, frontier_solve_rate_trials=5,
        frontier_min_solve_rate=0.0, frontier_max_solve_rate=0.7,
        frontier_alignment_weight=2.0,
        evaluation_context="test.designer.unverified")

    assert scored.reward["frontier_alignment"] == pytest.approx(1.0)
    assert scored.reward["frontier_reward_eligible"] is False
    assert scored.reward["frontier_in_band"] is False
    assert scored.reward["frontier_alignment_bonus"] == pytest.approx(0.0)
    assert scored.reward["frontier_extremity_penalty"] == pytest.approx(0.0)
    assert scored.reward["frontier_alignment_adjustment"] == pytest.approx(0.0)
    assert scored.reward.total == pytest.approx(-RewardConfig().w_unsolved)


def test_designer_selection_prioritizes_frontier_over_total_reward():
    extreme_high_reward = {
        "frontier_evaluated_count": 8,
        "frontier_in_band_rate": 0.0,
        "mean_frontier_alignment": 0.0,
        "mean_reward": 10.0,
    }
    frontier_lower_reward = {
        "frontier_evaluated_count": 8,
        "frontier_in_band_rate": 0.125,
        "mean_frontier_alignment": 0.125,
        "mean_reward": 1.0,
    }
    tapered_no_strict_hit = {
        "frontier_evaluated_count": 8,
        "frontier_in_band_rate": 0.0,
        "mean_frontier_alignment": 0.5,
        "mean_reward": 0.5,
    }

    assert train_mod.designer_selection_key(
        frontier_lower_reward) > train_mod.designer_selection_key(
            extreme_high_reward)
    assert train_mod.designer_selection_key(
        tapered_no_strict_hit) > train_mod.designer_selection_key(
            extreme_high_reward)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"frontier_alignment_weight": 1.0,
          "frontier_solve_rate_trials": 1}, "at least two"),
        ({"frontier_alignment_weight": -0.1}, "alignment_weight"),
        ({"frontier_min_solve_rate": 0.8,
          "frontier_max_solve_rate": 0.2}, "cannot exceed"),
        ({"frontier_alignment_weight": 1.0,
          "frontier_solve_rate_trials": 2,
          "frontier_dirichlet_weight": 0.0}, "dirichlet_weight"),
        ({"frontier_budget_min_ratio": 0.0}, "min_ratio"),
        ({"frontier_budget_min_ratio": 2.0,
          "frontier_budget_max_ratio": 1.0}, "cannot exceed"),
        ({"frontier_min_simulations": 401,
          "frontier_max_simulations": 400}, "cannot exceed"),
    ],
)
def test_designer_frontier_config_rejects_invalid_values(kwargs, message):
    with pytest.raises(ValueError, match=message):
        train_mod.TrainConfig(
            protagonist_checkpoint="unused.pt", output_dir="unused",
            **kwargs)


# ----------------------------------------------------------------------
# Replay buffer
# ----------------------------------------------------------------------

def _make_record(env, level, iteration=0, reward=1.0,
                 generator_model_state_sha256="0" * 64,
                 designer_checkpoint="d"):
    return build_level_record(
        env.env, level, trajectory=[0],
        designer_checkpoint=designer_checkpoint,
        generator_model_state_sha256=generator_model_state_sha256,
        protagonist_checkpoint="p",
        oracle_result={"oracle_solved": True, "protagonist_solved": False},
        reward_components={"novelty": 1.0}, structural_metrics={},
        solver_metrics={}, generation_iteration=iteration, reward_total=reward)


def test_duplicate_detection(env, tmp_path):
    level = env.reset(21).level
    buf = LevelReplayBuffer(tmp_path / "lr", max_levels=100)
    s1 = buf.add([_make_record(env, level)])
    s2 = buf.add([_make_record(env, level)])
    assert s1["added"] == 1
    assert s2["duplicates"] == 1 and s2["added"] == 0
    assert len(buf) == 1


def test_replay_persist_and_reload(env, tmp_path):
    level = env.reset(22).level
    identity = model_state_sha256(_designer_model())
    buf = LevelReplayBuffer(tmp_path / "lr", max_levels=100)
    buf.add([_make_record(
        env, level, generator_model_state_sha256=identity)])
    buf.persist()
    reloaded = LevelReplayBuffer(tmp_path / "lr").load()
    assert len(reloaded) == 1
    record = reloaded.records()[0]
    assert record["fingerprint"] == level_fingerprint(env.env, level)
    assert record["generator_model_state_sha256"] == identity
    assert record["provenance_status"] == "model_state_verified"


def test_legacy_replay_provenance_is_explicitly_unverified(env, tmp_path):
    record = _make_record(env, env.reset(24).level)
    record.pop("generator_model_state_sha256")
    record.pop("provenance_status")
    buf = LevelReplayBuffer(tmp_path / "legacy")
    buf.add([record])
    buf.persist()

    loaded = LevelReplayBuffer(tmp_path / "legacy").load().records()[0]
    assert loaded["generator_model_state_sha256"] is None
    assert loaded["provenance_status"] == "legacy_unverified"


def test_model_state_identity_is_content_based_and_rng_neutral(env):
    known_state = {
        "weight": torch.tensor([[1.0, 2.0]], dtype=torch.float32),
        "counter": torch.tensor(3, dtype=torch.int64),
    }
    assert model_state_sha256(known_state) == (
        "405895603f8f8d1cb7a2ceedf3dcb595"
        "ab76e49d36d8b545fa44dd4d7a3cd394")

    first = _designer_model()
    identical = _designer_model()
    identical.load_state_dict(first.state_dict())
    state = {
        name: value.detach().clone()
        for name, value in first.state_dict().items()}
    reversed_state = dict(reversed(list(state.items())))

    python_rng_state = random.getstate()
    torch_rng_state = torch.random.get_rng_state().clone()
    identity = model_state_sha256(first)

    assert identity == model_state_sha256(identical)
    assert identity == model_state_sha256(reversed_state)
    assert random.getstate() == python_rng_state
    assert torch.equal(torch.random.get_rng_state(), torch_rng_state)

    tensor_name = next(
        name for name, value in state.items()
        if torch.is_floating_point(value) and value.numel() > 1)
    changed_value = dict(state)
    changed_value[tensor_name] = state[tensor_name].clone()
    changed_value[tensor_name].view(-1)[0] += 1
    changed_shape = dict(state)
    changed_shape[tensor_name] = state[tensor_name].reshape(-1)
    changed_dtype = dict(state)
    changed_dtype[tensor_name] = state[tensor_name].to(torch.float64)

    assert model_state_sha256(changed_value) != identity
    assert model_state_sha256(changed_shape) != identity
    assert model_state_sha256(changed_dtype) != identity

    level = env.reset(25).level
    at_path_a = _make_record(
        env, level, generator_model_state_sha256=identity,
        designer_checkpoint="path/a/last.pt")
    at_path_b = _make_record(
        env, level, generator_model_state_sha256=model_state_sha256(identical),
        designer_checkpoint="elsewhere/best.pt")
    assert at_path_a["designer_checkpoint"] != at_path_b["designer_checkpoint"]
    assert (at_path_a["generator_model_state_sha256"]
            == at_path_b["generator_model_state_sha256"])


def test_replay_provenance_survives_last_checkpoint_overwrite(
        tmp_path, monkeypatch):
    protagonist_path = tmp_path / "prot.pt"
    _tiny_protagonist(protagonist_path)
    output_dir = tmp_path / "designer"
    original_model_class = train_mod.DesignerNet
    marker = {}
    rollout_identity = {}

    def zero_initialized_model(*args, **kwargs):
        model = original_model_class(*args, **kwargs)
        with torch.no_grad():
            for parameter in model.parameters():
                parameter.zero_()
        marker["name"], _ = next(model.named_parameters())
        return model

    generated_level = DesignerEnv(
        GeneratorConfig(rows=5, cols=5, color_count=2),
        mutation_budget=1, encoding=ENC).reset(17).level

    def fake_rollout(env, model, *args, **kwargs):
        parameter = dict(model.named_parameters())[marker["name"]]
        if "sha256" not in rollout_identity:
            rollout_identity["marker"] = float(
                parameter.detach().flatten()[0])
            rollout_identity["sha256"] = model_state_sha256(model)
        return SimpleNamespace(
            finalize=SimpleNamespace(valid=True, level=generated_level),
            trajectory=[0], steps=[])

    oracle = SolveOutcome(
        solved=True, cost=1.0, exact=True, method="test")
    protagonist = SolveOutcome(
        solved=False, cost=None, exact=False, method="test")
    metrics = StructuralMetrics(
        num_blocks=1, immediately_exitable=0, few_exitable=1,
        optimal_moves=1, extra_moves=0, first_exit_depth=1,
        distinct_setup_blocks=0, rehandled_blocks=0,
        opening_requires_setup=False)
    reward = compute_reward(
        RewardConfig(), valid=True, oracle=oracle, protagonist=protagonist,
        structural=metrics, novelty=1.0)
    scored = SimpleNamespace(
        valid=True, oracle=oracle, protagonist=protagonist,
        structural=metrics, reward=reward,
        oracle_result=lambda: {
            "oracle_solved": True, "protagonist_solved": False},
        solver_metrics=lambda: {})

    def update_marker(model, *args, **kwargs):
        with torch.no_grad():
            dict(model.named_parameters())[marker["name"]].fill_(1.0)
        return {
            "policy_loss": 0.0, "value_loss": 0.0,
            "entropy": 0.0, "steps": 0}

    monkeypatch.setattr(train_mod, "DesignerNet", zero_initialized_model)
    monkeypatch.setattr(train_mod, "rollout_episode", fake_rollout)
    monkeypatch.setattr(train_mod, "score_level", lambda *a, **k: scored)
    monkeypatch.setattr(train_mod, "ppo_update", update_marker)

    train_mod.train_designer(train_mod.TrainConfig(
        protagonist_checkpoint=str(protagonist_path),
        output_dir=str(output_dir),
        episodes=1,
        episodes_per_iter=1,
        mutation_budget=1,
        protagonist_simulations=1,
        oracle_simulations=1,
        astar_max_nodes=1,
        seed=23,
        device="cpu",
        generator=GeneratorConfig(rows=5, cols=5, color_count=2),
        model=DesignerModelConfig(channels=4, residual_blocks=1, hidden_size=8),
        ppo=PPOConfig(epochs=1, minibatch_size=1),
    ))

    record = LevelReplayBuffer(output_dir / "replay").load().records()[0]
    last = load_designer(output_dir / "last.pt")
    post_update_marker = float(
        last["model_state"][marker["name"]].flatten()[0])

    assert rollout_identity["marker"] == pytest.approx(0.0)
    assert post_update_marker == pytest.approx(1.0)
    assert record["designer_checkpoint"] == str(output_dir / "last.pt")
    assert record.get("generator_model_state_sha256") == \
        rollout_identity["sha256"]

    warm_model = original_model_class(
        ENC, DesignerModelConfig(channels=4, residual_blocks=1, hidden_size=8))
    with torch.no_grad():
        for parameter in warm_model.parameters():
            parameter.fill_(0.5)
    warm_identity = model_state_sha256(warm_model)
    warm_path = tmp_path / "warm.pt"
    save_designer(
        warm_path, model=warm_model, encoding_config=ENC,
        model_config=DesignerModelConfig(
            channels=4, residual_blocks=1, hidden_size=8),
        seed=99)
    rollout_identity.clear()
    warm_output = tmp_path / "warm-designer"
    train_mod.train_designer(train_mod.TrainConfig(
        protagonist_checkpoint=str(protagonist_path),
        output_dir=str(warm_output),
        init_designer=str(warm_path),
        episodes=1,
        episodes_per_iter=1,
        mutation_budget=1,
        protagonist_simulations=1,
        oracle_simulations=1,
        astar_max_nodes=1,
        seed=24,
        device="cpu",
        generator=GeneratorConfig(rows=5, cols=5, color_count=2),
        model=DesignerModelConfig(channels=4, residual_blocks=1, hidden_size=8),
        ppo=PPOConfig(epochs=1, minibatch_size=1),
    ))
    warm_record = LevelReplayBuffer(
        warm_output / "replay").load().records()[0]
    assert rollout_identity["marker"] == pytest.approx(0.5)
    assert rollout_identity["sha256"] == warm_identity
    assert warm_record["generator_model_state_sha256"] == warm_identity


def test_eviction_keeps_difficult(env, tmp_path):
    buf = LevelReplayBuffer(tmp_path / "lr", max_levels=2)
    lvls = [env.reset(30 + i).level for i in range(3)]
    buf.add([_make_record(env, lvls[0], reward=0.1)])
    buf.add([_make_record(env, lvls[1], reward=0.2)])
    buf.add([_make_record(env, lvls[2], reward=5.0)])   # hardest evicts easiest
    assert len(buf) == 2
    kept = {r["reward_total"] for r in buf.records()}
    assert 0.1 not in kept and 5.0 in kept


# ----------------------------------------------------------------------
# Model + PPO
# ----------------------------------------------------------------------

def _designer_model():
    return DesignerNet(ENC, DesignerModelConfig(channels=4, residual_blocks=1,
                                                hidden_size=8))


def test_model_forward_and_backward(env):
    model = _designer_model()
    space = DesignerActionSpace(ENC)
    state = env.reset(40)
    from blocksort.designer.encoding import encode_designer_state
    enc = encode_designer_state(env.env, state, ENC)
    logits, value = model(enc.board.unsqueeze(0), enc.global_features.unsqueeze(0))
    assert logits.shape == (1, space.size)
    assert value.shape == (1,)
    loss = logits.sum() + value.sum()
    loss.backward()
    assert any(p.grad is not None for p in model.parameters())


def test_designer_globals_use_causal_structural_signals(env):
    from blocksort.designer.encoding import encode_designer_state

    state = env.reset(40)
    encoded = encode_designer_state(env.env, state, ENC)
    extras = encoded.global_features[-8:]
    solve_state = state.to_solve_state(env.env)

    assert extras[-2].item() >= 0.0
    assert extras[-2].item() <= 1.0
    assert extras[-1].item() == pytest.approx(
        solve_state.total_blocks / (2 * ENC.max_blocks))


def test_legacy_designer_dead_feature_weights_are_zeroed_on_load(tmp_path):
    model = _designer_model()
    state = model.state_dict()
    state["stop_head.0.weight"][:, -2:] = 7.0
    state["value_head.0.weight"][:, -2:] = 9.0
    legacy_path = tmp_path / "legacy.pt"
    torch.save({
        "designer_checkpoint_version": 1,
        "model_state": state,
        "encoding_config": ENC.to_dict(),
        "model_config": model.model_config.to_dict(),
        "seed": 0,
        "metadata": {},
    }, legacy_path)

    checkpoint = load_designer(legacy_path)
    migrated, _encoding, _model_config = designer_from_checkpoint(checkpoint)

    assert DESIGNER_CHECKPOINT_VERSION == 2
    assert torch.count_nonzero(
        migrated.state_dict()["stop_head.0.weight"][:, -2:]) == 0
    assert torch.count_nonzero(
        migrated.state_dict()["value_head.0.weight"][:, -2:]) == 0


def test_one_ppo_update(env):
    model = _designer_model()
    space = DesignerActionSpace(ENC)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    rng = random.Random(0)
    episodes = []
    for i in range(3):
        ep = rollout_episode(env, model, space, ENC, seed=100 + i,
                             device=torch.device("cpu"), rng=rng,
                             verify_finalize=False)
        ep.reward = 1.0 if i % 2 == 0 else -1.0
        episodes.append(ep)
    stats = ppo_update(model, optimizer, episodes, PPOConfig(epochs=2,
                       minibatch_size=8), torch.device("cpu"), seed=0)
    assert stats["steps"] > 0
    assert stats["policy_loss"] == stats["policy_loss"]  # not NaN


@pytest.mark.parametrize("gamma", [0.0, 0.5, 0.999])
def test_ppo_gamma_rejects_silently_unsupported_discounting(gamma):
    with pytest.raises(ValueError, match="not currently configurable.*1.0"):
        PPOConfig(gamma=gamma)


@pytest.mark.parametrize("gamma", [float("nan"), float("inf"), -0.1, 1.1])
def test_ppo_gamma_rejects_nonfinite_or_out_of_range(gamma):
    with pytest.raises(ValueError, match=r"finite and in \[0, 1\]"):
        PPOConfig(gamma=gamma)


def test_ppo_gamma_one_documents_undiscounted_contract():
    assert PPOConfig(gamma=1.0).gamma == 1.0


class _ControlledValueModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.logits = torch.nn.Parameter(torch.tensor([0.0, 0.0]))
        self.value = torch.nn.Parameter(torch.tensor(0.25))

    def forward(self, boards, globs):
        batch_size = boards.shape[0]
        return (self.logits.unsqueeze(0).expand(batch_size, -1),
                self.value.expand(batch_size))


class _SetValueOptimizer:
    """Keep policy fixed and change only the current value after each step."""

    def __init__(self, model):
        self.model = model

    def zero_grad(self, set_to_none=True):
        self.model.zero_grad(set_to_none=set_to_none)

    def step(self):
        with torch.no_grad():
            self.model.value.fill_(1.25)


def _controlled_ppo_stats(rollout_value):
    model = _ControlledValueModel()
    step = Step(
        board=torch.tensor([0.0]),
        globals=torch.tensor([0.0]),
        mask=torch.tensor([True, True]),
        action_index=0,
        # Current chosen log-probability is -log(2), so old log-probability 0
        # fixes the PPO ratio at 0.5 throughout both epochs.
        log_prob=0.0,
        value=rollout_value,
    )
    episode = Episode(
        steps=[step],
        final_state=None,
        finalize=None,
        trajectory=[0],
        reward=1.5,
    )
    return ppo_update(
        model,
        _SetValueOptimizer(model),
        [episode],
        PPOConfig(
            epochs=2,
            minibatch_size=1,
            value_coef=1.0,
            entropy_coef=0.0,
            max_grad_norm=0.0,
        ),
        torch.device("cpu"),
    )


def test_ppo_policy_uses_fixed_rollout_advantage_across_epochs():
    stats = _controlled_ppo_stats(rollout_value=0.5)

    # Fixed advantage is 1.5 - 0.5 = 1.0. The fixed ratio is 0.5, so both
    # epochs must use policy loss -(0.5 * 1.0), even though the current value
    # changes from 0.25 to 1.25 after the first optimization step.
    assert stats["policy_loss"] == pytest.approx(-0.5)
    # Value loss must still use those two current predictions against return 1.5.
    assert stats["value_loss"] == pytest.approx((1.25 ** 2 + 0.25 ** 2) / 2)


def test_ppo_stored_rollout_value_changes_policy_advantage():
    baseline = _controlled_ppo_stats(rollout_value=0.5)
    lower_old_value = _controlled_ppo_stats(rollout_value=0.0)

    assert baseline["policy_loss"] == pytest.approx(-0.5)
    assert lower_old_value["policy_loss"] == pytest.approx(-0.75)
    # Rollout value changes policy advantage only, not the current-value target.
    assert lower_old_value["value_loss"] == pytest.approx(baseline["value_loss"])


# ----------------------------------------------------------------------
# Generated solution replay
# ----------------------------------------------------------------------

def test_generated_solution_replays(env):
    state = env.reset(55)
    rng = random.Random(2)
    for _ in range(6):
        reverse = [a for a in env.legal_actions(state) if a.kind == "reverse"]
        if not reverse:
            break
        state = env.step(state, rng.choice(reverse))
    e = Environment()
    start = e.initial_state(state.level)
    result = solve_astar(e, start, max_nodes=200_000)
    assert result.solvable is True
    assert verify_solution(e, start, result.actions, result.move_count)


# ----------------------------------------------------------------------
# End-to-end training smoke
# ----------------------------------------------------------------------

def _tiny_protagonist(path):
    model_cfg = ModelConfig(channels=4, residual_blocks=1, value_hidden_size=8)
    model = PolicyValueNet(ENC, model_cfg)
    save_checkpoint(path, model=model, optimizer=None, scheduler=None, epoch=0,
                    best_val_metric=None, encoding_config=ENC,
                    model_config=model_cfg, value_norm=ValueNormConfig(), seed=0,
                    dataset_version=1, split_identity=None, metrics={})


def test_designer_training_rejects_generator_above_protagonist_encoding(
        tmp_path):
    checkpoint = tmp_path / "protagonist.pt"
    _tiny_protagonist(checkpoint)
    output = tmp_path / "incompatible-generator"
    cfg = train_mod.TrainConfig(
        protagonist_checkpoint=str(checkpoint),
        output_dir=str(output),
        episodes=1,
        episodes_per_iter=1,
        generator=GeneratorConfig(max_blocks=ENC.max_blocks + 1),
    )

    with pytest.raises(
            ValueError, match="protagonist checkpoint encoding limit"):
        train_mod.train_designer(cfg)
    assert not output.exists()


def test_training_selects_frontier_checkpoint_over_higher_reward(
        tmp_path, monkeypatch):
    checkpoint = tmp_path / "prot.pt"
    _tiny_protagonist(checkpoint)
    no_solve = SolveOutcome(False, None, False, "test")
    metrics = StructuralMetrics(
        num_blocks=0, immediately_exitable=0, few_exitable=0,
        optimal_moves=None, extra_moves=None, first_exit_depth=None,
        distinct_setup_blocks=None, rehandled_blocks=None,
        opening_requires_setup=None)
    invalid_reward = compute_reward(
        RewardConfig(), valid=False, oracle=no_solve, protagonist=no_solve,
        structural=metrics, novelty=0.0)
    invalid_scored = score_mod.ScoredLevel(
        reward=invalid_reward, structural=metrics, oracle=no_solve,
        protagonist=no_solve, valid=False)
    validation_call = 0

    def fake_rollout(*args, **kwargs):
        return SimpleNamespace(
            finalize=SimpleNamespace(valid=False), trajectory=[], steps=[])

    def fake_evaluate(*args, **kwargs):
        nonlocal validation_call
        in_band = validation_call == 1
        validation_call += 1
        reward = 1.0 if in_band else 10.0
        components = {
            "adversarial_regret": 0.0,
            "frontier_solve_rate": 0.4 if in_band else 0.0,
            "frontier_alignment": 1.0 if in_band else 0.0,
            "frontier_in_band": in_band,
        }
        scored = score_mod.ScoredLevel(
            reward=RewardBreakdown(total=reward, components=components),
            structural=metrics,
            oracle=SolveOutcome(True, 1.0, True, "test"),
            protagonist=no_solve,
            valid=True)
        return [
            SimpleNamespace(scored=scored, reward=reward)
            for _ in range(8)
        ]

    monkeypatch.setattr(train_mod, "rollout_episode", fake_rollout)
    monkeypatch.setattr(
        train_mod, "score_level",
        lambda *args, **kwargs: invalid_scored)
    monkeypatch.setattr(
        train_mod, "ppo_update",
        lambda *args, **kwargs: {
            "policy_loss": 0.0, "value_loss": 0.0,
            "entropy": 0.0, "steps": 0,
        })
    monkeypatch.setattr(train_mod, "_evaluate_designer", fake_evaluate)

    cfg = train_mod.TrainConfig(
        protagonist_checkpoint=str(checkpoint),
        output_dir=str(tmp_path / "designer-selection"),
        episodes=2, episodes_per_iter=1, validation_episodes=8,
        mutation_budget=1, protagonist_simulations=1,
        oracle_simulations=1, astar_max_nodes=1,
        seed=17, device="cpu",
        generator=GeneratorConfig(rows=5, cols=5, color_count=2),
        model=DesignerModelConfig(
            channels=4, residual_blocks=1, hidden_size=8),
        ppo=PPOConfig(epochs=1, minibatch_size=1))

    summary = train_mod.train_designer(cfg)
    best = load_designer(summary["best_checkpoint"])

    assert best["metadata"]["iteration"] == 1
    assert summary["best_validation_mean_reward"] == pytest.approx(1.0)
    assert summary["best_validation_metrics"]["frontier_in_band_rate"] == \
        pytest.approx(1.0)


def test_end_to_end_training_smoke(tmp_path):
    ckpt = tmp_path / "prot.pt"
    _tiny_protagonist(ckpt)
    cfg = train_mod.TrainConfig(
        protagonist_checkpoint=str(ckpt), output_dir=str(tmp_path / "designer"),
        episodes=4, episodes_per_iter=2, mutation_budget=4,
        protagonist_simulations=4, oracle_simulations=8, astar_max_nodes=20_000,
        validation_episodes=2, frontier_solve_rate_trials=2,
        frontier_alignment_weight=1.0,
        seed=7, device="cpu", max_replay=50,
        generator=GeneratorConfig(rows=5, cols=5, color_count=2),
        model=DesignerModelConfig(channels=4, residual_blocks=1, hidden_size=8),
        ppo=PPOConfig(epochs=2, minibatch_size=8))
    summary = train_mod.train_designer(cfg)
    out = Path(cfg.output_dir)
    assert (out / "best.pt").exists()
    assert (out / "last.pt").exists()
    assert (out / "summary.json").exists()
    assert summary["iterations"] == 2
    assert summary["validation_episodes"] == 4
    assert len(summary["history"]) == 2
    for row in summary["history"]:
        assert 0.0 <= row["valid_rate"] <= 1.0
        assert row["frontier_evaluated_count"] == 2
        assert row["validation"]["frontier_evaluated_count"] == 2
        assert 0.0 <= row["frontier_in_band_rate"] <= 1.0
        assert 0.0 <= row["validation"]["frontier_in_band_rate"] <= 1.0


def test_designer_episode_count_exact_for_partial_final_batch(
    tmp_path, monkeypatch,
):
    checkpoint = tmp_path / "prot.pt"
    _tiny_protagonist(checkpoint)
    calls = []
    no_solve = SolveOutcome(
        solved=False, cost=None, exact=False, method="test")
    metrics = StructuralMetrics(
        num_blocks=0, immediately_exitable=0, few_exitable=0,
        optimal_moves=None, extra_moves=None, first_exit_depth=None,
        distinct_setup_blocks=None, rehandled_blocks=None,
        opening_requires_setup=None)
    reward = compute_reward(
        RewardConfig(), valid=False, oracle=no_solve, protagonist=no_solve,
        structural=metrics, novelty=0.0)
    scored = SimpleNamespace(
        valid=False, oracle=no_solve, protagonist=no_solve,
        structural=metrics, reward=reward)

    def fake_rollout(*args, **kwargs):
        calls.append(kwargs["seed"])
        return SimpleNamespace(
            finalize=SimpleNamespace(valid=False), trajectory=[], steps=[])

    monkeypatch.setattr(train_mod, "rollout_episode", fake_rollout)
    monkeypatch.setattr(train_mod, "score_level", lambda *a, **k: scored)
    monkeypatch.setattr(
        train_mod, "ppo_update",
        lambda *a, **k: {
            "policy_loss": 0.0, "value_loss": 0.0,
            "entropy": 0.0, "steps": 0})

    output = tmp_path / "designer-partial"
    summary = train_mod.train_designer(train_mod.TrainConfig(
        protagonist_checkpoint=str(checkpoint), output_dir=str(output),
        episodes=10, episodes_per_iter=8, validation_episodes=0,
        protagonist_simulations=1, oracle_simulations=1, astar_max_nodes=1,
        device="cpu",
        generator=GeneratorConfig(rows=4, cols=4, color_count=1),
        model=DesignerModelConfig(channels=4, residual_blocks=1, hidden_size=8),
        ppo=PPOConfig(epochs=1, minibatch_size=8),
    ))
    assert len(calls) == 10
    assert [row["episodes"] for row in summary["history"]] == [8, 2]
    assert summary["requested_training_episodes"] == 10
    assert summary["completed_training_episodes"] == 10
    assert summary["episodes"] == 10
    assert summary["validation_episodes"] == 0


@pytest.mark.parametrize(("episodes", "batch"), [(0, 8), (-1, 8), (1, 0)])
def test_designer_invalid_episode_settings_fail_before_output(
    tmp_path, episodes, batch,
):
    output = tmp_path / "not-created"
    with pytest.raises(ValueError, match="designer episodes"):
        train_mod.TrainConfig(
            protagonist_checkpoint="unused.pt", output_dir=str(output),
            episodes=episodes, episodes_per_iter=batch)
    assert not output.exists()


def test_fresh_designer_initialization_is_seeded_before_first_ppo(
        tmp_path, monkeypatch):
    ckpt = tmp_path / "prot.pt"
    _tiny_protagonist(ckpt)
    captured_states = []

    no_solve = SolveOutcome(
        solved=False, cost=None, exact=False, method="test")
    empty_metrics = StructuralMetrics(
        num_blocks=0, immediately_exitable=0, few_exitable=0,
        optimal_moves=None, extra_moves=None, first_exit_depth=None,
        distinct_setup_blocks=None, rehandled_blocks=None,
        opening_requires_setup=None)
    invalid_reward = compute_reward(
        RewardConfig(), valid=False, oracle=no_solve, protagonist=no_solve,
        structural=empty_metrics, novelty=0.0)
    invalid_score = SimpleNamespace(
        valid=False, oracle=no_solve, protagonist=no_solve,
        structural=empty_metrics, reward=invalid_reward)

    def fake_rollout(*args, **kwargs):
        return SimpleNamespace(
            finalize=SimpleNamespace(valid=False), trajectory=[], steps=[])

    def fake_score(*args, **kwargs):
        return invalid_score

    def capture_before_update(model, optimizer, episodes, cfg, device, **kwargs):
        captured_states.append({
            name: value.detach().cpu().clone()
            for name, value in model.state_dict().items()
        })
        return {
            "policy_loss": 0.0,
            "value_loss": 0.0,
            "entropy": 0.0,
            "steps": 0,
        }

    monkeypatch.setattr(train_mod, "rollout_episode", fake_rollout)
    monkeypatch.setattr(train_mod, "score_level", fake_score)
    monkeypatch.setattr(train_mod, "ppo_update", capture_before_update)

    model_cfg = DesignerModelConfig(
        channels=4, residual_blocks=1, hidden_size=8)

    def run(name, seed, init_designer=None):
        cfg = train_mod.TrainConfig(
            protagonist_checkpoint=str(ckpt),
            output_dir=str(tmp_path / name),
            init_designer=init_designer,
            episodes=1,
            episodes_per_iter=1,
            mutation_budget=1,
            protagonist_simulations=1,
            oracle_simulations=1,
            astar_max_nodes=1,
            seed=seed,
            device="cpu",
            validation_episodes=1,
            generator=GeneratorConfig(rows=5, cols=5, color_count=2),
            model=model_cfg,
            ppo=PPOConfig(epochs=1, minibatch_size=1),
        )
        train_mod.train_designer(cfg)
        return captured_states[-1]

    same_seed_a = run("same-a", 123)
    random.seed(91_001)
    for _ in range(100):
        random.random()
    torch.manual_seed(91_001)
    torch.rand(4096)
    same_seed_b = run("same-b", 123)

    for name, value in same_seed_a.items():
        assert torch.equal(value, same_seed_b[name]), (
            f"first differing tensor: {name}")

    different_seed = run("different", 124)
    assert any(
        not torch.equal(value, different_seed[name])
        for name, value in same_seed_a.items()
    )

    warm_model = DesignerNet(ENC, model_cfg)
    with torch.no_grad():
        for index, parameter in enumerate(warm_model.parameters()):
            parameter.fill_(index + 0.25)
    warm_path = tmp_path / "warm.pt"
    save_designer(
        warm_path, model=warm_model, encoding_config=ENC,
        model_config=model_cfg, seed=777)
    warm_state = {
        name: value.detach().cpu().clone()
        for name, value in warm_model.state_dict().items()
    }

    random.seed(81_002)
    torch.manual_seed(81_002)
    torch.rand(4096)
    loaded_warm_state = run("warm", 999, str(warm_path))
    for name, value in warm_state.items():
        assert torch.equal(value, loaded_warm_state[name]), (
            f"warm-start tensor was not loaded exactly: {name}")


def test_designer_reuse_rejected_after_post_update_checkpoint(
        tmp_path, monkeypatch):
    ckpt = tmp_path / "prot.pt"
    _tiny_protagonist(ckpt)
    original_model_class = train_mod.DesignerNet
    marker = {}
    initial_values = [0.0, 1.0]

    def zero_initialized_model(*args, **kwargs):
        model = original_model_class(*args, **kwargs)
        with torch.no_grad():
            for parameter in model.parameters():
                parameter.zero_()
        marker["name"], _ = next(model.named_parameters())
        with torch.no_grad():
            dict(model.named_parameters())[marker["name"]].fill_(
                initial_values.pop(0))
        return model

    original_rollout = train_mod.rollout_episode
    pending_scores = []

    def rollout_with_parameter_score(env, model, action_space, encoding, **kwargs):
        parameter_value = float(
            dict(model.named_parameters())[marker["name"]].detach().flatten()[0])
        pending_scores.append(10.0 - 5.0 * parameter_value)
        return original_rollout(env, model, action_space, encoding, **kwargs)

    def score_for_rollout(*args, **kwargs):
        return SimpleNamespace(
            valid=False,
            oracle=SimpleNamespace(solved=False),
            protagonist=SimpleNamespace(solved=False),
            structural=SimpleNamespace(extra_moves=None),
            reward=SimpleNamespace(total=pending_scores.pop(0)),
        )

    def increment_model(model, optimizer, episodes, cfg, device, **kwargs):
        with torch.no_grad():
            dict(model.named_parameters())[marker["name"]].add_(1.0)
        return {
            "policy_loss": 0.0,
            "value_loss": 0.0,
            "entropy": 0.0,
            "steps": sum(len(ep.steps) for ep in episodes),
        }

    monkeypatch.setattr(train_mod, "DesignerNet", zero_initialized_model)
    monkeypatch.setattr(train_mod, "rollout_episode", rollout_with_parameter_score)
    monkeypatch.setattr(train_mod, "score_level", score_for_rollout)
    monkeypatch.setattr(train_mod, "ppo_update", increment_model)

    cfg = train_mod.TrainConfig(
        protagonist_checkpoint=str(ckpt),
        output_dir=str(tmp_path / "designer"),
        episodes=1,
        episodes_per_iter=1,
        mutation_budget=1,
        protagonist_simulations=1,
        oracle_simulations=1,
        astar_max_nodes=1,
        seed=11,
        device="cpu",
        generator=GeneratorConfig(rows=5, cols=5, color_count=2),
        model=DesignerModelConfig(channels=4, residual_blocks=1, hidden_size=8),
        ppo=PPOConfig(epochs=1, minibatch_size=1),
    )

    summary = train_mod.train_designer(cfg)
    best = load_designer(summary["best_checkpoint"])
    last = load_designer(summary["last_checkpoint"])
    best_value = float(best["model_state"][marker["name"]].flatten()[0])
    last_value = float(last["model_state"][marker["name"]].flatten()[0])

    assert best_value == pytest.approx(1.0)
    assert last_value == pytest.approx(1.0)
    assert summary["best_validation_mean_reward"] == pytest.approx(5.0)
    assert best["metadata"]["selection_metric"] == {
        "name": "frontier_in_band_alignment_reward_lexicographic_v1",
        "priority": [
            "frontier_in_band_rate",
            "mean_frontier_alignment",
            "validation_mean_reward",
        ],
        "value": {
            "frontier_in_band_rate": pytest.approx(-1.0),
            "mean_frontier_alignment": pytest.approx(-1.0),
            "validation_mean_reward": pytest.approx(5.0),
        },
    }
    assert summary["best_selection_metric"] == \
        best["metadata"]["selection_metric"]
    assert best["metadata"]["checkpoint_model_state"] == "post_update"
    assert last["metadata"]["checkpoint_model_state"] == "post_update"

    artifacts = {
        path: (path.read_bytes(), path.stat().st_mtime_ns)
        for path in (
            Path(cfg.output_dir) / "best.pt",
            Path(cfg.output_dir) / "last.pt",
            Path(cfg.output_dir) / "summary.json")}
    with pytest.raises(UnsupportedResumeError, match="does not currently support"):
        train_mod.train_designer(cfg)
    assert artifacts == {
        path: (path.read_bytes(), path.stat().st_mtime_ns)
        for path in artifacts}


def test_designer_pretraining_rejects_output_reuse_without_mutation(tmp_path):
    out = tmp_path / "pretrain"
    kwargs = {
        "output_dir": str(out),
        "trajectories": 1,
        "epochs": 1,
        "batch_size": 2,
        "generator": GeneratorConfig(rows=5, cols=5, color_count=2),
        "model_config": DesignerModelConfig(
            channels=4, residual_blocks=1, hidden_size=8),
        "device": "cpu",
        "seed": 5,
    }
    pretrain_mod.pretrain_designer(**kwargs)
    artifacts = {
        path: (path.read_bytes(), path.stat().st_mtime_ns)
        for path in (
            out / "best.pt", out / "pretrain_summary.json",
            out / "experiment_spec.json")}
    with pytest.raises(UnsupportedResumeError, match="does not currently support"):
        pretrain_mod.pretrain_designer(**kwargs)
    assert artifacts == {
        path: (path.read_bytes(), path.stat().st_mtime_ns)
        for path in artifacts}


def test_post_update_validation_is_deterministic_and_side_effect_free(monkeypatch):
    model = _ControlledValueModel()
    parameter_state = {
        name: value.detach().clone() for name, value in model.state_dict().items()}
    global_random_state = random.getstate()
    torch_random_state = torch.random.get_rng_state().clone()
    rollout_calls = []
    score_calls = []

    def fake_rollout(env, current_model, action_space, encoding, *,
                     seed, device, rng, verify_finalize):
        rollout_calls.append(
            (seed, rng.random(), torch.is_grad_enabled(), current_model.training))
        return SimpleNamespace(finalize=object())

    def fake_score(*args, novelty, seed, **kwargs):
        score_calls.append((novelty, seed, torch.is_grad_enabled()))
        return SimpleNamespace(reward=SimpleNamespace(total=3.0))

    monkeypatch.setattr(train_mod, "rollout_episode", fake_rollout)
    monkeypatch.setattr(train_mod, "score_level", fake_score)
    kwargs = {
        "protagonist": object(),
        "oracle": object(),
        "reward_cfg": RewardConfig(),
        "validation_episodes": 2,
        "seed": 17,
        "device": torch.device("cpu"),
        "astar_max_nodes": 1,
    }
    fake_env = SimpleNamespace(env=object())

    model.train()
    first = train_mod._evaluate_designer(fake_env, model, object(), ENC, **kwargs)
    first_rollout_calls = list(rollout_calls)
    first_score_calls = list(score_calls)
    assert model.training is True

    rollout_calls.clear()
    score_calls.clear()
    model.eval()
    second = train_mod._evaluate_designer(fake_env, model, object(), ENC, **kwargs)

    assert model.training is False
    assert rollout_calls == first_rollout_calls
    assert score_calls == first_score_calls
    assert len(first) == len(second) == 2
    assert all(call[2:] == (False, False) for call in rollout_calls)
    assert all(novelty == 0.0 and grad_enabled is False
               for novelty, _seed, grad_enabled in score_calls)
    assert random.getstate() == global_random_state
    assert torch.equal(torch.random.get_rng_state(), torch_random_state)
    for name, value in model.state_dict().items():
        assert torch.equal(value, parameter_state[name])
