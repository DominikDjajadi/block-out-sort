"""Tests for alternating protagonist-designer co-training.

Covers frontier filtering, automatic curriculum increase/decrease, frozen
protagonist during designer updates and frozen designer during protagonist
updates, exact-label priority, replay across rounds, benchmark/validation leakage
prevention, candidate promotion/rejection, resume from a completed round, a
deterministic tiny co-training run, and a controlled no-catastrophic-forgetting
fixture.
"""

from __future__ import annotations

import copy
import hashlib
import itertools
import json
import random
import shutil
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from blocksort import Environment, level_from_dict, static_level_signature
from blocksort.training.config import EncodingConfig, ModelConfig, ValueNormConfig
from blocksort.training.model import PolicyValueNet
from blocksort.training.checkpoint import save_checkpoint
from blocksort.training.transaction import (
    atomic_copy, atomic_write_json, relative_to_run,
    resolve_committed_protagonist, sha256_file)
from blocksort.training.dataset import load_records
from blocksort.training.experiment_identity import (
    ExperimentIdentityError, ExperimentSpecIntegrityError)

from blocksort.designer.model import DesignerModelConfig, DesignerNet
from blocksort.designer.checkpoint import save_designer
from blocksort.designer.roles import SolveOutcome
from blocksort.designer import train as designer_train
from blocksort.designer.config import GeneratorConfig as DesGen
from blocksort.designer.ppo import PPOConfig

from blocksort.expert_iteration.records import SOURCE_EXACT, SOURCE_SEARCH
from blocksort.expert_iteration.replay import ReplayBuffer
from blocksort.expert_iteration.evaluate import evaluate_checkpoint, promotion_score
from blocksort.expert_iteration.promotion import promotion_metric_requires_budget
from blocksort.oracle import ValueResult
from blocksort.level_generation import random_level

from blocksort.cotraining.config import (CoTrainingConfig, CurriculumConfig,
                                         CurriculumState)
from blocksort.cotraining.eval_split import create_eval_split_manifest
from blocksort.cotraining.run import build_parser, config_from_args
from blocksort.cotraining.curriculum import adapt_curriculum
from blocksort.cotraining.frontier import (
    estimate_solve_rate, frontier_distance, geometric_budget_sweep, in_frontier,
    select_frontier_backfill)
from blocksort.cotraining.benchmark import _precomputed_lookup, forgetting_report
from blocksort.cotraining import benchmark as benchmark_mod
from blocksort.cotraining import loop as loop_mod
from blocksort.cotraining.loop import CoTraining, drop_frozen_candidates, run_cotraining


ENC = EncodingConfig()
REPO = Path(__file__).resolve().parents[2]
PV_SMOKE = REPO / "data" / "training" / "pv_smoke.jsonl"


# ----------------------------------------------------------------------
# Precomputed benchmark identity
# ----------------------------------------------------------------------

def _benchmark_entry(signature, canonical_key, value):
    return {
        "static_signature": signature,
        "canonical_key": canonical_key,
        "value_result": {"value": value, "exact": True, "solvable": True},
        "optimal_actions": [],
        "termination": "exact",
    }


def test_precomputed_lookup_separates_static_level_identity():
    env = Environment()
    base = {
        "name": "left-exit",
        "cols": 5,
        "rows": 5,
        "blocks": [{"color": "red", "cells": [[2, 2]]}],
        "exits": [{"edge": "left", "start": 2, "length": 1, "color": "red"}],
    }
    changed = copy.deepcopy(base)
    changed["name"] = "right-exit"
    changed["exits"][0]["edge"] = "right"
    level_a = level_from_dict(base)
    level_b = level_from_dict(changed)
    state_a = env.initial_state(level_a)
    state_b = env.initial_state(level_b)
    canonical_key = env.canonical_key(state_a)
    signature_a = static_level_signature(level_a)
    signature_b = static_level_signature(level_b)

    assert env.canonical_key(state_b) == canonical_key
    assert signature_a != signature_b

    entry_a = _benchmark_entry(signature_a, canonical_key, 3)
    entry_b = _benchmark_entry(signature_b, canonical_key, 7)
    lookup = _precomputed_lookup({"collision": [entry_a, entry_b]}, "collision")

    assert len(lookup) == 2
    assert lookup[(signature_a, canonical_key)]["value_result"]["value"] == 3
    assert lookup[(signature_b, canonical_key)]["value_result"]["value"] == 7
    assert ("missing-signature", canonical_key) not in lookup


def test_precomputed_lookup_rejects_unsafe_legacy_and_conflicting_duplicates():
    key = "same-dynamic-key"
    entry = _benchmark_entry("same-static-signature", key, 3)

    duplicate = copy.deepcopy(entry)
    assert len(_precomputed_lookup({"g": [entry, duplicate]}, "g")) == 1

    conflicting = _benchmark_entry("same-static-signature", key, 7)
    with pytest.raises(ValueError, match="conflicting duplicate"):
        _precomputed_lookup({"g": [entry, conflicting]}, "g")

    legacy = copy.deepcopy(entry)
    del legacy["static_signature"]
    with pytest.raises(ValueError, match="static_signature"):
        _precomputed_lookup({"g": [legacy]}, "g")

    malformed_complete = copy.deepcopy(entry)
    malformed_complete["classification_complete"] = "yes"
    with pytest.raises(ValueError, match="classification_complete"):
        _precomputed_lookup({"g": [malformed_complete]}, "g")

    malformed_actions = copy.deepcopy(entry)
    malformed_actions["optimal_actions"] = {}
    with pytest.raises(ValueError, match="optimal_actions"):
        _precomputed_lookup({"g": [malformed_actions]}, "g")

    duplicate_action = copy.deepcopy(entry)
    duplicate_action["optimal_actions"] = [
        {"action": "same"}, {"action": "same"}]
    with pytest.raises(ValueError, match="duplicate optimal action"):
        _precomputed_lookup({"g": [duplicate_action]}, "g")


def test_evaluate_groups_passes_collision_safe_precomputed_lookup(monkeypatch):
    env = Environment()
    left = level_from_dict({
        "name": "left",
        "cols": 5,
        "rows": 5,
        "blocks": [{"color": "red", "cells": [[2, 2]]}],
        "exits": [{"edge": "left", "start": 2, "length": 1, "color": "red"}],
    })
    right_dict = copy.deepcopy({
        "name": "right",
        "cols": 5,
        "rows": 5,
        "blocks": [{"color": "red", "cells": [[2, 2]]}],
        "exits": [{"edge": "right", "start": 2, "length": 1, "color": "red"}],
    })
    right = level_from_dict(right_dict)
    dynamic_key = env.canonical_key(env.initial_state(left))
    labels = {"g": [
        _benchmark_entry(static_level_signature(left), dynamic_key, 3),
        _benchmark_entry(static_level_signature(right), dynamic_key, 7),
    ]}
    captured = {}

    def fake_evaluate(*args, precomputed, **kwargs):
        captured.update(precomputed)
        return {"states": len(args[4])}

    monkeypatch.setattr(benchmark_mod, "evaluate_checkpoint", fake_evaluate)
    result = benchmark_mod.evaluate_groups(
        env, object(), ENC, ValueNormConfig(), {"g": [left, right]},
        exact_oracle=object(), budgets=[], device=torch.device("cpu"),
        c_puct=1.5, seed=0, precomputed_labels=labels)

    assert result["g"]["states"] == 2
    assert len(captured) == 2
    assert captured[(static_level_signature(left), dynamic_key)] \
        ["value_result"]["value"] == 3
    assert captured[(static_level_signature(right), dynamic_key)] \
        ["value_result"]["value"] == 7


def test_evaluate_groups_rejects_nonlegal_precomputed_optimal_action():
    env = Environment()
    level = _cache_test_level()
    state = env.initial_state(level)
    labels = {"g": [{
        "static_signature": static_level_signature(level),
        "canonical_key": env.canonical_key(state),
        "value_result": {"value": 1, "exact": True, "solvable": True},
        "optimal_actions": [{"not": "a legal serialized action"}],
        "classification_complete": True,
        "termination": "exact",
    }]}

    with pytest.raises(ValueError, match="not legal"):
        benchmark_mod.evaluate_groups(
            env, object(), ENC, ValueNormConfig(), {"g": [level]},
            exact_oracle=object(), budgets=[], device=torch.device("cpu"),
            c_puct=1.5, seed=0, precomputed_labels=labels)


def test_evaluate_checkpoint_does_not_fallback_to_dynamic_only_label():
    env = Environment()
    left = level_from_dict({
        "name": "left",
        "cols": 5,
        "rows": 5,
        "blocks": [{"color": "red", "cells": [[2, 2]]}],
        "exits": [{"edge": "left", "start": 2, "length": 1, "color": "red"}],
    })
    right = level_from_dict({
        "name": "right",
        "cols": 5,
        "rows": 5,
        "blocks": [{"color": "red", "cells": [[2, 2]]}],
        "exits": [{"edge": "right", "start": 2, "length": 1, "color": "red"}],
    })
    state_left = env.initial_state(left)
    state_right = env.initial_state(right)
    dynamic_key = env.canonical_key(state_left)
    precomputed = _precomputed_lookup({"g": [
        _benchmark_entry(static_level_signature(left), dynamic_key, 3),
    ]}, "g")

    class RecordingOracle:
        def __init__(self):
            self.calls = 0

        def value(self, state):
            self.calls += 1
            return ValueResult(value=9, exact=True, solvable=True)

    oracle = RecordingOracle()
    model = PolicyValueNet(
        ENC, ModelConfig(channels=4, residual_blocks=1, value_hidden_size=8))
    result = evaluate_checkpoint(
        env, model, ENC, ValueNormConfig(), [state_right], budgets=[1],
        oracle=oracle, device=torch.device("cpu"), precomputed=precomputed)

    assert oracle.calls > 0
    assert "9" in result["grouped_by_difficulty"]
    assert "3" not in result["grouped_by_difficulty"]
    budget = result["budgets"]["1"]
    assert budget["search_solved_count"] in (0, 1)
    assert budget["search_solve_rate_total"] == \
        budget["search_solved_count"] / budget["total_evaluated_count"]
    paired = result["paired_level_solve_outcomes"]
    assert paired == [{
        "static_level_signature": static_level_signature(right),
        "budgets": {"1": {
            "solved": bool(budget["search_solved_count"]),
        }},
    }]


# ----------------------------------------------------------------------
# Persisted benchmark-evaluation cache identity
# ----------------------------------------------------------------------

class _CacheTestModel:
    def __init__(self, marker):
        self.marker = marker

    def state_dict(self):
        return {"marker": torch.tensor([self.marker], dtype=torch.float32)}


def _cache_test_level(name="cache-test", exit_edge="left"):
    return level_from_dict({
        "name": name,
        "cols": 5,
        "rows": 5,
        "blocks": [{"color": "red", "cells": [[2, 2]]}],
        "exits": [{"edge": exit_edge, "start": 2, "length": 1,
                   "color": "red"}],
    })


def _cache_test_labels(env, level, value):
    state = env.initial_state(level)
    return {"g": [_benchmark_entry(
        static_level_signature(level), env.canonical_key(state), value)]}


def _cached_group_eval(tmp_path, monkeypatch, model, groups, *,
                       labels=None, calls=None, checkpoint_sha256=None,
                       budgets=None):
    calls = calls if calls is not None else []
    if checkpoint_sha256 is None:
        checkpoint_sha256 = f"checkpoint-for-marker-{model.marker}"

    def fake_evaluate(*args, precomputed, **kwargs):
        calls.append(model.marker)
        label_value = (next(iter(precomputed.values()))["value_result"]["value"]
                       if precomputed else None)
        return {"states": len(args[4]), "marker": model.marker,
                "label_value": label_value,
                "level_signature": static_level_signature(args[4][0].level)}

    monkeypatch.setattr(benchmark_mod, "evaluate_checkpoint", fake_evaluate)
    result = benchmark_mod.evaluate_groups(
        Environment(), model, ENC, ValueNormConfig(), groups,
        exact_oracle=object(), budgets=budgets or [8], device=torch.device("cpu"),
        c_puct=1.5, seed=17, precomputed_labels=labels,
        progress_dir=tmp_path, tag="candidate",
        checkpoint_sha256=checkpoint_sha256)
    return result, calls


def test_benchmark_cache_does_not_reuse_different_candidate(tmp_path, monkeypatch):
    level = _cache_test_level()
    groups = {"g": [level]}
    result_a, calls = _cached_group_eval(
        tmp_path, monkeypatch, _CacheTestModel(1), groups)
    result_b, calls = _cached_group_eval(
        tmp_path, monkeypatch, _CacheTestModel(2), groups, calls=calls)

    assert result_a["g"]["marker"] == 1
    assert result_b["g"]["marker"] == 2
    assert calls == [1, 2]
    cache_files = list((tmp_path / "candidate").rglob("*.json"))
    assert len(cache_files) == 2
    payloads = [json.loads(path.read_text(encoding="utf-8"))
                for path in cache_files]
    assert {p["metadata"]["metric_semantics_version"] for p in payloads} == {6}
    assert len({p["metadata"]["model_state_sha256"] for p in payloads}) == 2


def test_benchmark_cache_invalidates_replaced_checkpoint_bytes(
        tmp_path, monkeypatch):
    model = _CacheTestModel(2)
    groups = {"g": [_cache_test_level()]}
    checkpoint = tmp_path / "candidate.pt"
    checkpoint.write_bytes(b"checkpoint-content-a")
    identity_a = benchmark_mod.checkpoint_content_hash(checkpoint)
    _result, calls = _cached_group_eval(
        tmp_path, monkeypatch, model, groups,
        checkpoint_sha256=identity_a)
    checkpoint.write_bytes(b"checkpoint-content-b")
    identity_b = benchmark_mod.checkpoint_content_hash(checkpoint)
    _result, calls = _cached_group_eval(
        tmp_path, monkeypatch, model, groups, calls=calls,
        checkpoint_sha256=identity_b)

    assert identity_a != identity_b
    assert calls == [2, 2]


def test_benchmark_cache_invalidates_changed_group_contents(tmp_path, monkeypatch):
    model = _CacheTestModel(3)
    left = _cache_test_level(exit_edge="left")
    right = _cache_test_level(exit_edge="right")
    result_a, calls = _cached_group_eval(
        tmp_path, monkeypatch, model, {"g": [left]})
    result_b, calls = _cached_group_eval(
        tmp_path, monkeypatch, model, {"g": [right]}, calls=calls)

    assert result_a["g"]["states"] == result_b["g"]["states"] == 1
    assert result_a["g"]["level_signature"] == static_level_signature(left)
    assert result_b["g"]["level_signature"] == static_level_signature(right)
    assert calls == [3, 3]


def test_benchmark_cache_invalidates_changed_labels_and_validates_first(
        tmp_path, monkeypatch):
    env = Environment()
    level = _cache_test_level()
    groups = {"g": [level]}
    model = _CacheTestModel(4)
    labels_a = _cache_test_labels(env, level, 1)
    labels_b = _cache_test_labels(env, level, 9)
    result_a, calls = _cached_group_eval(
        tmp_path, monkeypatch, model, groups, labels=labels_a)
    result_b, calls = _cached_group_eval(
        tmp_path, monkeypatch, model, groups, labels=labels_b, calls=calls)

    assert result_a["g"]["label_value"] == 1
    assert result_b["g"]["label_value"] == 9
    assert calls == [4, 4]

    unsafe = copy.deepcopy(labels_b)
    del unsafe["g"][0]["static_signature"]
    with pytest.raises(ValueError, match="static_signature"):
        _cached_group_eval(
            tmp_path, monkeypatch, model, groups, labels=unsafe, calls=calls)


def test_benchmark_cache_rejects_old_metric_semantics_version(
        tmp_path, monkeypatch):
    model = _CacheTestModel(5)
    groups = {"g": [_cache_test_level()]}
    _result, calls = _cached_group_eval(tmp_path, monkeypatch, model, groups)
    cache_files = list((tmp_path / "candidate").rglob("*.json"))
    assert len(cache_files) == 1
    cache = json.loads(cache_files[0].read_text(encoding="utf-8"))
    cache["metadata"]["metric_semantics_version"] = 3
    cache_files[0].write_text(json.dumps(cache), encoding="utf-8")

    _result, calls = _cached_group_eval(
        tmp_path, monkeypatch, model, groups, calls=calls)
    assert calls == [5, 5]


def test_benchmark_cache_reuses_exact_identity(tmp_path, monkeypatch):
    model = _CacheTestModel(6)
    level = _cache_test_level()
    groups = {"g": [level]}
    labels = _cache_test_labels(Environment(), level, 2)
    result_a, calls = _cached_group_eval(
        tmp_path, monkeypatch, model, groups, labels=labels)
    result_b, calls = _cached_group_eval(
        tmp_path, monkeypatch, model, groups, labels=copy.deepcopy(labels),
        calls=calls)

    assert result_b == result_a
    assert calls == [6]


def test_benchmark_cache_identity_includes_effective_budgets(
        tmp_path, monkeypatch):
    model = _CacheTestModel(7)
    groups = {"g": [_cache_test_level()]}
    effective = CoTrainingConfig(
        eval_budgets=(1, 100, 400),
        promotion_budget=32,
    ).eval_budgets
    _result, calls = _cached_group_eval(
        tmp_path, monkeypatch, model, groups, budgets=[1, 100, 400])
    _result, calls = _cached_group_eval(
        tmp_path, monkeypatch, model, groups, calls=calls,
        budgets=list(effective))

    assert calls == [7, 7]
    payloads = [
        json.loads(path.read_text(encoding="utf-8"))
        for path in (tmp_path / "candidate").glob("*.json")
    ]
    assert {tuple(p["metadata"]["evaluation_config"]["budgets"])
            for p in payloads} == {
                (1, 100, 400),
                (1, 32, 100, 400),
            }


# ----------------------------------------------------------------------
# Curriculum
# ----------------------------------------------------------------------

def test_curriculum_increases_when_too_easy():
    cfg = CurriculumConfig()
    state = CurriculumState(density=0.4, mutation_budget=6, structural_threshold=1)
    new, rec = adapt_curriculum(state, mean_solve_rate=0.95, cfg=cfg)
    assert rec["direction"] == "increase"
    assert rec["changed"] is True
    assert new.mutation_budget > state.mutation_budget
    assert new.density >= state.density
    assert "too easy" in rec["reason"]


def test_curriculum_reduces_when_too_hard():
    cfg = CurriculumConfig()
    state = CurriculumState(density=0.5, mutation_budget=10, structural_threshold=2)
    new, rec = adapt_curriculum(state, mean_solve_rate=0.05, cfg=cfg)
    assert rec["direction"] == "reduce"
    assert new.mutation_budget < state.mutation_budget
    assert new.density < state.density
    assert "too hard" in rec["reason"]


def test_curriculum_holds_within_band():
    cfg = CurriculumConfig(frontier_min_solve_rate=0.2, frontier_max_solve_rate=0.7)
    state = CurriculumState()
    new, rec = adapt_curriculum(state, mean_solve_rate=0.45, cfg=cfg)
    assert rec["direction"] == "hold"
    assert rec["changed"] is False
    assert new == state


def test_curriculum_reduces_when_frontier_yield_is_low_and_too_hard_dominates():
    cfg = CurriculumConfig(
        frontier_min_solve_rate=0.2, frontier_max_solve_rate=0.7,
        min_frontier_acceptance_rate=0.1, frontier_imbalance_margin=0.05)
    state = CurriculumState(density=0.45, mutation_budget=8)
    new, rec = adapt_curriculum(
        state, mean_solve_rate=0.25, cfg=cfg,
        frontier_acceptance_rate=0.03,
        below_frontier_rate=0.71,
        above_frontier_rate=0.24)

    assert rec["direction"] == "reduce"
    assert "frontier acceptance" in rec["reason"]
    assert new.density < state.density
    assert new.mutation_budget < state.mutation_budget


def test_curriculum_holds_low_yield_when_rejection_sides_are_balanced():
    cfg = CurriculumConfig(
        min_frontier_acceptance_rate=0.1, frontier_imbalance_margin=0.05)
    state = CurriculumState()
    new, rec = adapt_curriculum(
        state, mean_solve_rate=0.45, cfg=cfg,
        frontier_acceptance_rate=0.03,
        below_frontier_rate=0.46,
        above_frontier_rate=0.44)

    assert rec["direction"] == "hold"
    assert new == state
    assert "balanced" in rec["reason"]


def test_curriculum_records_before_after():
    cfg = CurriculumConfig()
    state = CurriculumState(mutation_budget=6)
    _new, rec = adapt_curriculum(state, mean_solve_rate=0.9, cfg=cfg)
    assert rec["before"]["mutation_budget"] == 6
    assert rec["after"]["mutation_budget"] == rec["changes"]["mutation_budget"]["to"]


def test_curriculum_holds_when_generation_has_no_unique_signal():
    state = CurriculumState(rows=5, cols=5, color_count=2, density=0.4,
                            mutation_budget=4, protagonist_simulations=8)
    generation = {
        "unique_valid_count": 0,
        "mean_solve_rate": None,
        "frontier_acceptance_rate": 0.0,
        "below_frontier_count": 0,
        "above_frontier_count": 0,
    }

    new, rec = loop_mod._adapt_curriculum_from_generation(
        state, generation, CurriculumConfig())

    assert new == state
    assert rec["direction"] == "hold"
    assert rec["changed"] is False
    assert rec["mean_solve_rate"] is None
    assert "no unique nonduplicate valid candidates" in rec["reason"]


# ----------------------------------------------------------------------
# Frontier filtering
# ----------------------------------------------------------------------

class _FakeProtagonist:
    """Deterministic seed-sensitive solver used to inspect trial streams."""

    def __init__(self):
        self.seeds = []

    def solve(self, level, *, seed=0):
        self.seeds.append(seed)
        solved = (seed % 3 == 1)
        return SolveOutcome(solved=solved, cost=5.0 if solved else None,
                            exact=False, method="search")


def test_solve_rate_trial_seeds_are_distinct_and_reproducible():
    first = _FakeProtagonist()
    est = estimate_solve_rate(first, level=object(),
                              trials=4, base_seed=0)
    assert est.solve_rate == pytest.approx(0.5)
    assert est.trial_seeds == tuple(first.seeds)
    assert len(set(est.trial_seeds)) == 4

    second = _FakeProtagonist()
    again = estimate_solve_rate(second, level=object(),
                                trials=4, base_seed=0)
    assert again == est
    assert second.seeds == first.seeds

    cfg = CurriculumConfig(frontier_min_solve_rate=0.2, frontier_max_solve_rate=0.7)
    assert in_frontier(0.5, cfg) is True
    assert in_frontier(0.05, cfg) is False
    assert in_frontier(0.95, cfg) is False


def test_budget_sweep_is_geometric_clamped_and_drives_solve_rate():
    budgets = geometric_budget_sweep(
        center=100, trials=5, minimum_ratio=0.25, maximum_ratio=4.0,
        minimum_simulations=20, maximum_simulations=400)
    assert budgets == (25, 50, 100, 200, 400)
    assert geometric_budget_sweep(
        center=220, trials=5, minimum_ratio=0.25, maximum_ratio=4.0,
        minimum_simulations=20, maximum_simulations=400
    ) == (55, 90, 148, 244, 400)

    class BudgetSensitiveProtagonist:
        def __init__(self):
            self.calls = []

        def solve(self, level, *, seed=0, simulations=None):
            self.calls.append((seed, simulations))
            solved = simulations >= 200
            return SolveOutcome(
                solved=solved, cost=5.0 if solved else None,
                exact=False, method="search")

    protagonist = BudgetSensitiveProtagonist()
    estimate = estimate_solve_rate(
        protagonist, object(), trials=5, base_seed=13,
        simulation_budgets=budgets)

    assert estimate.solve_rate == pytest.approx(0.4)
    assert estimate.trial_budgets == budgets
    assert [budget for _seed, budget in protagonist.calls] == list(budgets)


def test_solve_rate_trial_seeds_are_level_order_independent():
    def run(order):
        return {
            identity: estimate_solve_rate(
                _FakeProtagonist(), object(), trials=5, base_seed=17,
                level_identity=identity,
            ).trial_seeds
            for identity in order
        }

    forward = run(["level-a", "level-b"])
    reverse = run(["level-b", "level-a"])
    assert forward == reverse
    assert forward["level-a"] != forward["level-b"]
    assert all(len(set(seeds)) == 5 for seeds in forward.values())


def test_frontier_solve_rate_automatically_includes_full_level_identity():
    env = Environment()
    base = {
        "name": "seed-level",
        "cols": 4,
        "rows": 4,
        "blocks": [{"color": "red", "cells": [[1, 1]]}],
        "exits": [{"edge": "left", "start": 1, "length": 1, "color": "red"}],
    }
    moved = copy.deepcopy(base)
    moved["blocks"][0]["cells"] = [[1, 2]]

    class EnvFake(_FakeProtagonist):
        def __init__(self):
            super().__init__()
            self.env = env

    level_a = level_from_dict(base)
    level_b = level_from_dict(moved)
    seeds_a = estimate_solve_rate(
        EnvFake(), level_a, trials=4, base_seed=9).trial_seeds
    seeds_b = estimate_solve_rate(
        EnvFake(), level_b, trials=4, base_seed=9).trial_seeds
    repeat_a = estimate_solve_rate(
        EnvFake(), level_a, trials=4, base_seed=9).trial_seeds
    assert seeds_a == repeat_a
    assert seeds_a != seeds_b


def test_ranked_frontier_backfill_selects_nearest_levels_stably():
    cfg = CurriculumConfig(
        frontier_min_solve_rate=0.2, frontier_max_solve_rate=0.7)
    candidates = [
        ("far-easy", 1.0),
        ("near-hard", 0.1),
        ("near-easy", 0.75),
        ("far-hard", 0.0),
    ]

    assert frontier_distance(0.5, cfg) == pytest.approx(0.0)
    assert frontier_distance(0.1, cfg) == pytest.approx(0.1)
    assert frontier_distance(0.75, cfg) == pytest.approx(0.05)
    assert select_frontier_backfill(
        candidates, limit=2, cfg=cfg) == ("near-easy", "near-hard")
    assert select_frontier_backfill(
        [("b", 0.1), ("a", 0.1)], limit=2, cfg=cfg) == ("a", "b")


@pytest.mark.parametrize(("total_limit", "expected"), [
    (0, []),
    (1, ["a0"]),
    (2, ["a0", "a1"]),
    (20, ["a0", "a1", "b0", "b1", "c0"]),
])
def test_benchmark_total_limit_is_global_and_group_ordered(
    total_limit, expected,
):
    groups = {
        "a": ["a0", "a1"],
        "b": ["b0", "b1"],
        "c": ["c0"],
    }
    sampled = benchmark_mod.sample_benchmark_groups(
        groups, per_group_limit=None, total_limit=total_limit, seed=3)
    flattened = [item for values in sampled.values() for item in values]
    assert flattened == expected
    assert sum(map(len, sampled.values())) <= total_limit
    assert list(sampled) == list(groups)


def test_benchmark_limits_validate_and_per_group_counts_match():
    groups = {"a": [1, 2, 3], "b": [4, 5, 6]}
    sampled = benchmark_mod.sample_benchmark_groups(
        groups, per_group_limit=1, total_limit=None, seed=9)
    assert {name: len(values) for name, values in sampled.items()} == {
        "a": 1, "b": 1}
    with pytest.raises(ValueError, match="total_limit"):
        benchmark_mod.sample_benchmark_groups(
            groups, per_group_limit=None, total_limit=-1, seed=9)


# ----------------------------------------------------------------------
# Exact-label priority (replay)
# ----------------------------------------------------------------------

def _rec(source, sig="s", key="k", iteration=1):
    return {"static_level_signature": sig, "state_key": key,
            "target_source": source, "generation_iteration": iteration}


def test_exact_label_priority_in_replay(tmp_path):
    buf = ReplayBuffer(tmp_path / "rp", max_examples=100)
    # search first, then exact -> upgraded to exact.
    buf.add([_rec(SOURCE_SEARCH)], iteration=1)
    buf.add([_rec(SOURCE_EXACT)], iteration=1)
    rec = list(buf._records.values())[0]
    assert rec["target_source"] == SOURCE_EXACT
    # exact present, search arrives -> exact kept.
    buf2 = ReplayBuffer(tmp_path / "rp2", max_examples=100)
    buf2.add([_rec(SOURCE_EXACT)], iteration=1)
    stats = buf2.add([_rec(SOURCE_SEARCH)], iteration=1)
    assert stats["kept_existing"] == 1
    assert list(buf2._records.values())[0]["target_source"] == SOURCE_EXACT


# ----------------------------------------------------------------------
# No leakage
# ----------------------------------------------------------------------

class _S:
    def __init__(self, level):
        self.level = level


def test_drop_frozen_candidates_blocks_leakage():
    from blocksort.signature import static_level_signature
    recs = load_records(PV_SMOKE)
    lv_a = None
    lv_b = None
    from blocksort.serialization import level_from_dict
    for r in recs:
        lv = level_from_dict(r["level"])
        sig = static_level_signature(lv)
        if lv_a is None:
            lv_a, sig_a = lv, sig
        elif sig != sig_a:
            lv_b = lv
            break
    cands = [(_S(lv_a), {}), (_S(lv_b), {})]
    kept = drop_frozen_candidates(cands, {sig_a})
    kept_sigs = {static_level_signature(s.level) for s, _ in kept}
    assert sig_a not in kept_sigs
    assert len(kept) == 1


# ----------------------------------------------------------------------
# Promotion / rejection
# ----------------------------------------------------------------------

def _eval_report(acc, budget=8):
    return {"states": 5, "budgets": {str(budget): {"search_optimal_acc": acc}}}


def test_promotion_and_rejection_decision():
    prev = promotion_score(_eval_report(0.4), metric="search_optimal_acc", budget=8)
    better = promotion_score(_eval_report(0.6), metric="search_optimal_acc", budget=8)
    worse = promotion_score(_eval_report(0.3), metric="search_optimal_acc", budget=8)
    margin = 0.0
    assert (better > prev + margin) is True       # promote
    assert (worse > prev + margin) is False        # reject


def test_missing_promotion_budget_is_added_and_real_score_is_used():
    cfg = CoTrainingConfig(
        eval_budgets=(1, 100, 400),
        promotion_budget=32,
        promotion_metric="search_optimal_acc",
    )
    report = {
        "budgets": {
            str(budget): {
                "search_optimal_acc": 0.75 if budget == 32 else 0.25,
            }
            for budget in cfg.eval_budgets
        },
    }

    assert cfg.eval_budgets == (1, 32, 100, 400)
    assert promotion_score(
        report, metric=cfg.promotion_metric, budget=cfg.promotion_budget
    ) == pytest.approx(0.75)


def test_existing_promotion_budget_is_not_duplicated():
    cfg = CoTrainingConfig(eval_budgets=[1, 32, 100, 32], promotion_budget=32)
    assert cfg.eval_budgets == (1, 32, 100)
    assert cfg.eval_budgets.count(32) == 1


def test_missing_promotion_budget_fails_instead_of_returning_false_zero():
    report = {"budgets": {"1": {}, "100": {}, "400": {}}}
    with pytest.raises(
        ValueError,
        match=r"requested budget 32.*contains budgets \[1, 100, 400\]",
    ):
        promotion_score(report, metric="search_optimal_acc", budget=32)


def test_missing_promotion_metric_field_fails_clearly():
    with pytest.raises(ValueError, match="missing metric field 'search_optimal_acc'"):
        promotion_score(
            {"budgets": {"32": {"solve_rate": 0.0}}},
            metric="search_optimal_acc",
            budget=32,
        )


def test_real_zero_promotion_score_remains_valid():
    assert promotion_score(
        {"budgets": {"32": {"search_optimal_acc": 0.0}}},
        metric="search_optimal_acc",
        budget=32,
    ) == 0.0


@pytest.mark.parametrize(
    ("incumbent", "candidate", "expected"),
    [(0.75, 1.0, True), (0.75, 0.75, False)],
)
def test_budget_32_promotion_keeps_improvement_and_tie_semantics(
        incumbent, candidate, expected):
    prev = promotion_score(
        _eval_report(incumbent, budget=32),
        metric="search_optimal_acc",
        budget=32,
    )
    cand = promotion_score(
        _eval_report(candidate, budget=32),
        metric="search_optimal_acc",
        budget=32,
    )
    assert (cand > prev + 0.0) is expected


def test_raw_policy_promotion_does_not_require_budget():
    assert promotion_metric_requires_budget("search_optimal_acc") is True
    assert promotion_metric_requires_budget("neg_search_regret") is True
    assert promotion_metric_requires_budget("raw_policy_optimal_acc") is False
    cfg = CoTrainingConfig(
        eval_budgets=(100, 1, 100),
        promotion_metric="raw_policy_optimal_acc",
        promotion_budget=32,
    )
    assert cfg.eval_budgets == (1, 100)
    assert promotion_score(
        {"raw_policy_optimal_acc": 0.5},
        metric=cfg.promotion_metric,
        budget=cfg.promotion_budget,
    ) == 0.5


def test_cotraining_cli_config_includes_missing_promotion_budget():
    args = build_parser().parse_args([
        "--protagonist-checkpoint", "protagonist.pt",
        "--designer-checkpoint", "designer.pt",
        "--base-dataset", "base.jsonl",
        "--eval-budgets", "1", "100", "400",
        "--promotion-metric", "search_confirmed_optimal_rate",
        "--promotion-budget", "32",
    ])
    cfg = config_from_args(args)

    assert cfg.eval_budgets == (1, 32, 100, 400)
    assert cfg.eval_limit is None
    assert cfg.to_dict()["eval_budgets"] == [1, 32, 100, 400]


def test_cotraining_cli_config_supports_budget_sweep_promotion():
    args = build_parser().parse_args([
        "--protagonist-checkpoint", "protagonist.pt",
        "--designer-checkpoint", "designer.pt",
        "--base-dataset", "base.jsonl",
        "--eval-budgets", "16",
        "--promotion-metric", "weighted_budget_sweep_confirmed_optimal_rate",
        "--promotion-budgets", "1", "2", "4", "8",
        "--promotion-budget-weights", "0.4", "0.3", "0.2", "0.1",
    ])
    cfg = config_from_args(args)

    assert cfg.eval_budgets == (1, 2, 4, 8, 16)
    assert cfg.promotion_budgets == (1, 2, 4, 8)
    assert cfg.promotion_budget_weights == pytest.approx((0.4, 0.3, 0.2, 0.1))


def test_cotraining_cli_config_supports_paired_promotion_gate():
    args = build_parser().parse_args([
        "--protagonist-checkpoint", "protagonist.pt",
        "--designer-checkpoint", "designer.pt",
        "--base-dataset", "base.jsonl",
        "--promotion-metric", "weighted_budget_sweep_solve_rate",
        "--promotion-budgets", "20", "34", "57", "95", "160",
        "--promotion-budget-weights", "0.2", "0.2", "0.2", "0.2", "0.2",
        "--promotion-margin", "0.01",
        "--promotion-paired-gate",
        "--promotion-max-per-budget-regression", "0.02",
        "--promotion-bootstrap-confidence", "0.95",
        "--promotion-bootstrap-replicates", "10000",
        "--promotion-bootstrap-seed", "8216",
    ])
    cfg = config_from_args(args)

    assert cfg.promotion_paired_gate_enabled is True
    assert cfg.promotion_max_per_budget_regression == pytest.approx(0.02)
    assert cfg.promotion_bootstrap_confidence == pytest.approx(0.95)
    assert cfg.promotion_bootstrap_replicates == 10_000
    assert cfg.promotion_bootstrap_seed == 8216


def test_paired_promotion_gate_requires_weighted_full_solve_metric():
    with pytest.raises(ValueError, match="promotion_paired_gate_enabled"):
        CoTrainingConfig(
            promotion_metric="search_confirmed_optimal_rate",
            promotion_paired_gate_enabled=True,
        )


def test_default_curriculum_stays_in_pretraining_envelope():
    state = CurriculumState()
    assert state.to_dict() == {
        "rows": 5,
        "cols": 5,
        "color_count": 3,
        "density": 0.35,
        "mutation_budget": 4,
        "protagonist_simulations": 100,
        "structural_threshold": 1,
    }
    args = build_parser().parse_args([
        "--protagonist-checkpoint", "protagonist.pt",
        "--designer-checkpoint", "designer.pt",
        "--base-dataset", "base.jsonl",
    ])
    cfg = config_from_args(args)
    assert cfg.initial_curriculum == state
    assert cfg.epochs == 1
    assert cfg.learning_rate == pytest.approx(3e-5)
    assert cfg.trainable_part == "policy_head"
    assert cfg.policy_target_profile == "incumbent_optimal"
    assert cfg.search_value_loss_weight == 0.0
    assert cfg.exact_path_policy_confidence == pytest.approx(0.5)
    assert cfg.eval_budgets == (1, 2, 4, 8, 16)
    assert cfg.promotion_metric == "weighted_budget_sweep_solve_rate"
    assert cfg.promotion_budgets == (4, 8, 16)
    assert cfg.promotion_budget_weights == pytest.approx((0.2, 0.3, 0.5))
    assert cfg.promotion_margin == pytest.approx(0.01)
    assert Path(cfg.eval_levels_dataset).name == \
        "hard_pool_v1_20260723.jsonl"
    assert Path(cfg.eval_split_manifest).name == \
        "hard_pool_v1_20260723_split.json"
    assert cfg.eval_validation_count == 60

    env = Environment()
    generator = loop_mod._gen_cfg(state, max_blocks=ENC.max_blocks)
    levels = [
        random_level(
            env,
            generator,
            random.Random(seed),
            reverse_depth=state.mutation_budget,
        )
        for seed in range(256)
    ]
    assert all(level is not None for level in levels)
    block_counts = [level.total_blocks for level in levels if level is not None]
    assert min(block_counts) >= 3
    assert max(block_counts) <= 7
    assert sum(block_counts) / len(block_counts) == pytest.approx(4.171875)


def test_cotraining_cli_config_supports_full_solve_promotion_and_yield_policy():
    args = build_parser().parse_args([
        "--protagonist-checkpoint", "protagonist.pt",
        "--designer-checkpoint", "designer.pt",
        "--base-dataset", "base.jsonl",
        "--eval-budgets", "1", "2",
        "--promotion-metric", "weighted_budget_sweep_solve_rate",
        "--promotion-budgets", "4", "8", "16",
        "--promotion-budget-weights", "0.2", "0.3", "0.5",
        "--promotion-margin", "0.01",
        "--solve-rate-trials", "5",
        "--frontier-budget-min-ratio", "0.25",
        "--frontier-budget-max-ratio", "4.0",
        "--min-frontier-acceptance-rate", "0.10",
        "--frontier-imbalance-margin", "0.05",
        "--frontier-backfill-target", "12",
        "--min-fresh-levels-to-train", "8",
        "--replay-current-fraction", "0.4",
        "--replay-recent-fraction", "0.2",
        "--replay-historical-fraction", "0.4",
        "--replay-recent-window", "3",
        "--designer-validation-episodes", "8",
        "--designer-frontier-alignment-weight", "1.25",
        "--exploratory-astar-time-limit-seconds", "2.5",
        "--training-astar-time-limit-seconds", "1.5",
        "--generation-checkpoint-interval", "7",
        "--generation-progress-interval", "9",
        "--epochs", "3",
        "--learning-rate", "0.0003",
        "--search-value-loss-weight", "0.25",
        "--policy-anchor-weight", "0.5",
        "--policy-target-profile", "incumbent_optimal",
    ])
    cfg = config_from_args(args)

    assert cfg.eval_budgets == (1, 2, 4, 8, 16)
    assert cfg.promotion_budgets == (4, 8, 16)
    assert cfg.promotion_budget_weights == pytest.approx((0.2, 0.3, 0.5))
    assert cfg.promotion_margin == pytest.approx(0.01)
    assert cfg.solve_rate_trials == 5
    assert cfg.frontier_budget_min_ratio == pytest.approx(0.25)
    assert cfg.frontier_budget_max_ratio == pytest.approx(4.0)
    assert cfg.curriculum.min_frontier_acceptance_rate == pytest.approx(0.1)
    assert cfg.curriculum.frontier_imbalance_margin == pytest.approx(0.05)
    assert cfg.frontier_backfill_target == 12
    assert cfg.min_fresh_levels_to_train == 8
    assert cfg.replay_current_fraction == pytest.approx(0.4)
    assert cfg.replay_recent_fraction == pytest.approx(0.2)
    assert cfg.replay_historical_fraction == pytest.approx(0.4)
    assert cfg.replay_recent_window == 3
    assert cfg.designer_validation_episodes == 8
    assert cfg.designer_frontier_alignment_weight == pytest.approx(1.25)
    assert cfg.exploratory_astar_time_limit_seconds == pytest.approx(2.5)
    assert cfg.training_astar_time_limit_seconds == pytest.approx(1.5)
    assert cfg.generation_checkpoint_interval == 7
    assert cfg.generation_progress_interval == 9
    assert cfg.epochs == 3
    assert cfg.learning_rate == pytest.approx(3e-4)
    assert cfg.search_value_loss_weight == pytest.approx(0.25)
    assert cfg.policy_anchor_weight == pytest.approx(0.5)
    assert cfg.policy_target_profile == "incumbent_optimal"
    assert cfg.trainable_part == "policy_head"


@pytest.mark.parametrize("weight", [-0.1, float("nan"), float("inf")])
def test_cotraining_rejects_invalid_search_value_weight(weight):
    with pytest.raises(ValueError, match="search_value_loss_weight"):
        CoTrainingConfig(search_value_loss_weight=weight)


@pytest.mark.parametrize("weight", [-0.1, float("nan"), float("inf")])
def test_cotraining_rejects_invalid_policy_anchor_weight(weight):
    with pytest.raises(ValueError, match="policy_anchor_weight"):
        CoTrainingConfig(policy_anchor_weight=weight)


def test_cotraining_rejects_invalid_policy_target_profile():
    with pytest.raises(ValueError, match="policy_target_profile"):
        CoTrainingConfig(policy_target_profile="unknown")


def test_cotraining_rejects_guided_policy_targets_for_value_only_update():
    with pytest.raises(ValueError, match="incumbent-guided policy targets"):
        CoTrainingConfig(trainable_part="value_head")

    cfg = CoTrainingConfig(
        trainable_part="value_head",
        policy_target_profile="recorded",
    )
    assert cfg.policy_target_profile == "recorded"


@pytest.mark.parametrize(
    ("current", "recent", "historical"),
    [
        (-0.1, 0.7, 0.4),
        (float("nan"), 0.25, 0.4),
        (0.35, float("inf"), 0.4),
        (0.35, 0.25, 0.35),
    ],
)
def test_cotraining_rejects_invalid_replay_age_fractions(
        current, recent, historical):
    with pytest.raises(ValueError, match="replay"):
        CoTrainingConfig(
            replay_current_fraction=current,
            replay_recent_fraction=recent,
            replay_historical_fraction=historical,
        )


def test_cotraining_requires_positive_current_replay_fraction_and_window():
    with pytest.raises(ValueError, match="replay_current_fraction"):
        CoTrainingConfig(
            replay_current_fraction=0.0,
            replay_recent_fraction=0.25,
            replay_historical_fraction=0.75,
        )
    with pytest.raises(ValueError, match="replay_recent_window"):
        CoTrainingConfig(replay_recent_window=0)


def test_cotraining_rejects_invalid_or_zero_source_weights():
    with pytest.raises(ValueError, match="weight_search"):
        CoTrainingConfig(weight_search=float("nan"))
    with pytest.raises(ValueError, match="at least one source weight"):
        CoTrainingConfig(
            weight_exact_historical=0.0,
            weight_exact_new=0.0,
            weight_search=0.0,
        )


@pytest.mark.parametrize("confidence", [
    -0.1, 1.1, float("nan"), float("inf"), True,
])
def test_cotraining_rejects_invalid_exact_path_confidence(confidence):
    with pytest.raises(ValueError, match="exact_path_policy_confidence"):
        CoTrainingConfig(exact_path_policy_confidence=confidence)


def test_cli_accepts_policy_head_only_training():
    args = build_parser().parse_args([
        "--protagonist-checkpoint", "p.pt",
        "--designer-checkpoint", "d.pt",
        "--base-dataset", "base.jsonl",
        "--trainable-part", "policy_head",
        "--initial-base-split", "base-split.json",
        "--initial-protagonist-replay", "protagonist-replay.jsonl",
        "--initial-designer-replay", "designer-replay.jsonl",
        "--initial-learner-checkpoint", "learner.pt",
        "--shadow-learner",
        "--initialize-only",
    ])

    cfg = config_from_args(args)
    assert cfg.trainable_part == "policy_head"
    assert cfg.initial_base_split == "base-split.json"
    assert cfg.initial_protagonist_replay == "protagonist-replay.jsonl"
    assert cfg.initial_designer_replay == "designer-replay.jsonl"
    assert cfg.initial_learner_checkpoint == "learner.pt"
    assert cfg.initialize_only is True


def test_budget_sweep_config_rejects_weight_mismatch():
    with pytest.raises(ValueError, match="one value per"):
        CoTrainingConfig(
            promotion_metric="weighted_budget_sweep_confirmed_optimal_rate",
            promotion_budgets=(1, 2, 4),
            promotion_budget_weights=(0.5, 0.5),
        )


def test_unweighted_budget_sweep_uses_equal_weights():
    cfg = CoTrainingConfig(
        promotion_metric="budget_sweep_confirmed_optimal_rate",
        promotion_budgets=(1, 2, 4),
    )

    assert cfg.promotion_budget_weights == pytest.approx((1 / 3, 1 / 3, 1 / 3))


def test_unweighted_budget_sweep_rejects_explicit_weights():
    with pytest.raises(ValueError, match="weighted budget-sweep"):
        CoTrainingConfig(
            promotion_metric="budget_sweep_confirmed_optimal_rate",
            promotion_budgets=(1, 2, 4),
            promotion_budget_weights=(0.5, 0.3, 0.2),
        )


@pytest.mark.parametrize("margin", [-0.1, float("nan"), float("inf")])
def test_cotraining_rejects_invalid_promotion_margin(margin):
    with pytest.raises(ValueError, match="promotion_margin"):
        CoTrainingConfig(promotion_margin=margin)


@pytest.mark.parametrize(
    "field,value",
    [
        ("exploratory_astar_time_limit_seconds", 0),
        ("exploratory_astar_time_limit_seconds", float("inf")),
        ("training_astar_time_limit_seconds", -1),
        ("generation_checkpoint_interval", 0),
        ("generation_progress_interval", 1.5),
        ("solve_rate_trials", 0),
        ("designer_validation_episodes", 0),
        ("designer_frontier_alignment_weight", -0.1),
        ("frontier_budget_min_ratio", 0),
        ("frontier_budget_max_ratio", float("inf")),
    ],
)
def test_cotraining_rejects_invalid_speed_control(field, value):
    with pytest.raises(ValueError, match=field):
        CoTrainingConfig(**{field: value})


def test_generation_partial_round_trip_and_integrity(tmp_path):
    path = tmp_path / "generation.partial.json"
    identity = {
        "experiment_fingerprint": "f" * 64,
        "round": 1,
        "curriculum": {"rows": 4},
        "active_protagonist_sha256": "a" * 64,
        "designer_checkpoint_sha256": "d" * 64,
        "generator_model_state_sha256": "g" * 64,
        "levels_per_round": 2,
        "solve_rate_trials": 3,
        "frontier_dirichlet_alpha": 0.5,
        "frontier_dirichlet_weight": 0.4,
        "frontier_budget_min_ratio": 0.25,
        "frontier_budget_max_ratio": 4.0,
        "frontier_simulation_budgets": [25, 100, 400],
    }
    level = {
        "name": "partial",
        "cols": 4,
        "rows": 4,
        "blocks": [{"color": "red", "cells": [[1, 1]]}],
        "exits": [{
            "edge": "left", "start": 1, "length": 1, "color": "red",
        }],
    }
    attempts = [
        {
            "index": 0,
            "status": "valid",
            "fingerprint": "level-0",
            "level": level,
            "trajectory": [1, 2],
            "solve_rate": 2 / 3,
            "solve_rate_trials": 3,
            "solve_rate_solved": 2,
            "solve_rate_budgets": [25, 100, 400],
            "duplicate": False,
            "num_mutations": 2,
        },
        {
            "index": 1,
            "status": "invalid",
            "errors": ["synthetic invalid"],
        },
    ]

    loop_mod._write_generation_partial(
        path, identity=identity, attempts=attempts)

    assert loop_mod._load_generation_partial(
        path, identity=identity) == attempts
    with pytest.raises(RuntimeError, match="different run semantics"):
        loop_mod._load_generation_partial(
            path, identity={**identity, "solve_rate_trials": 5})

    corrupt = json.loads(path.read_text(encoding="utf-8"))
    corrupt["attempts"][0]["solve_rate_solved"] = 1
    atomic_write_json(path, corrupt)
    with pytest.raises(RuntimeError, match="solved count"):
        loop_mod._load_generation_partial(path, identity=identity)


def test_cotraining_default_promotion_metric_is_coverage_safe():
    assert CoTrainingConfig().promotion_metric == \
        "search_confirmed_optimal_rate"


def test_cotraining_cli_accepts_policy_adapter_with_conditioned_targets():
    from blocksort.cotraining.run import build_parser, config_from_args

    args = build_parser().parse_args([
        "--protagonist-checkpoint", "protagonist.pt",
        "--designer-checkpoint", "designer.pt",
        "--base-dataset", "base.jsonl",
        "--output-dir", "run",
        "--trainable-part", "policy_adapter",
        "--policy-target-profile", "incumbent_optimal",
        "--structural-threshold", "2",
    ])

    config = config_from_args(args)

    assert config.trainable_part == "policy_adapter"
    assert config.policy_target_profile == "incumbent_optimal"
    assert config.initial_curriculum.structural_threshold == 2
    args = build_parser().parse_args([
        "--protagonist-checkpoint", "protagonist.pt",
        "--designer-checkpoint", "designer.pt",
        "--base-dataset", "base.jsonl",
    ])
    cfg = config_from_args(args)
    assert cfg.promotion_metric == "weighted_budget_sweep_solve_rate"
    assert cfg.promotion_budgets == (4, 8, 16)


def test_cotraining_cli_can_disable_bundled_hard_eval():
    args = build_parser().parse_args([
        "--protagonist-checkpoint", "protagonist.pt",
        "--designer-checkpoint", "designer.pt",
        "--base-dataset", "base.jsonl",
        "--no-bundled-hard-eval",
    ])

    cfg = config_from_args(args)

    assert cfg.eval_levels_dataset is None
    assert cfg.eval_split_manifest is None
    assert cfg.eval_validation_count is None


# ----------------------------------------------------------------------
# Forgetting metric (controlled fixture)
# ----------------------------------------------------------------------

def _group_report(acc, budget=8):
    return {g: {"states": 4, "budgets": {str(budget): {"search_optimal_acc": acc}}}
            for g in ("handcrafted", "random", "ood")}


def test_no_forgetting_when_identical():
    base = _group_report(0.5)
    cand = _group_report(0.5)
    fr = forgetting_report(base, cand, metric_budget=8)
    assert all(v["delta"] == pytest.approx(0.0) for v in fr.values())


def test_forgetting_detected_when_worse():
    base = _group_report(0.6)
    cand = _group_report(0.3)
    fr = forgetting_report(base, cand, metric_budget=8)
    assert all(v["delta"] < 0 for v in fr.values())


def test_forgetting_reports_classification_and_exact_regret_coverage():
    base = {"g": {"states": 4, "budgets": {"8": {
        "search_optimal_acc": 0.75,
        "search_optimal_classification_coverage": 0.5,
        "search_exact_regret_coverage": 0.25,
    }}}}
    cand = {"g": {"states": 4, "budgets": {"8": {
        "search_optimal_acc": 0.5,
        "search_optimal_classification_coverage": 0.75,
        "search_exact_regret_coverage": 0.0,
    }}}}

    report = forgetting_report(base, cand, metric_budget=8)["g"]
    assert report["delta"] == pytest.approx(-0.25)
    assert report["baseline_classification_coverage"] == pytest.approx(0.5)
    assert report["candidate_classification_coverage"] == pytest.approx(0.75)
    assert report["baseline_exact_regret_coverage"] == pytest.approx(0.25)
    assert report["candidate_exact_regret_coverage"] == pytest.approx(0.0)


# ----------------------------------------------------------------------
# Freezing invariants
# ----------------------------------------------------------------------

def _sha(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def test_uncommitted_promotion_cannot_change_cotraining_resume_checkpoint(
        tmp_path):
    """Committed identity wins even when the convenience mirror is overwritten."""
    root = tmp_path / "run"
    root.mkdir()
    incumbent = root / "incumbent.pt"
    candidate = root / "round_001" / "candidate.pt"
    candidate.parent.mkdir()
    incumbent.write_bytes(b"committed-incumbent")
    candidate.write_bytes(b"uncommitted-candidate")
    best = root / "best.pt"
    best.write_bytes(incumbent.read_bytes())
    state = {
        "completed_rounds": [],
        "best_protagonist": str(best),
        "active_protagonist_checkpoint": "incumbent.pt",
        "active_protagonist_sha256": _sha(incumbent),
        "active_protagonist_source_round": 0,
        "history": [],
    }
    (root / "run_state.json").write_text(json.dumps(state), encoding="utf-8")

    incumbent_sha256 = _sha(best)
    candidate_sha256 = _sha(candidate)
    shutil.copyfile(candidate, best)  # promotion, followed by an injected crash

    resumed = json.loads((root / "run_state.json").read_text(encoding="utf-8"))
    resume_sha256 = _sha(resolve_committed_protagonist(root, resumed))
    assert resumed["completed_rounds"] == []
    assert _sha(best) == candidate_sha256
    assert resume_sha256 == incumbent_sha256


def test_cotraining_crash_after_state_commit_recovers_exactly_once(
        tmp_path, monkeypatch):
    base = _tiny_base_dataset(tmp_path / "base.jsonl")
    protagonist = _tiny_protagonist(tmp_path / "prot.pt")
    designer = _tiny_designer(tmp_path / "designer.pt")
    cfg0 = _tiny_config(
        tmp_path / "run", base, rounds=0,
        protagonist_checkpoint=protagonist, designer_checkpoint=designer)
    initial = run_cotraining(cfg0)["run_state"]
    incumbent_hash = initial["active_protagonist_sha256"]
    candidate_source = _tiny_protagonist(tmp_path / "candidate-source.pt")
    promote = {"value": True}

    def fake_round(self, rnd, base_records, split, enc, model_cfg, value_norm,
                   replay, level_replay, run_state):
        round_dir = self.root / f"round_{rnd:03d}"
        round_dir.mkdir(parents=True, exist_ok=True)
        candidate = round_dir / "candidate.pt"
        atomic_copy(candidate_source, candidate)
        self._crash_point("after_candidate_evaluation")
        self._crash_point("before_promotion_decision")
        if promote["value"]:
            run_state["active_protagonist_checkpoint"] = relative_to_run(
                candidate, self.root)
            run_state["active_protagonist_sha256"] = sha256_file(candidate)
            run_state["active_protagonist_source_round"] = rnd
        summary = {
            "round": rnd, "promoted": promote["value"],
            "promotion_score_prev": 0.1, "promotion_score_candidate": 0.9,
        }
        report = {
            "commit_status": "prepared", "summary": summary,
            "protagonist": {
                "candidate_checkpoint": relative_to_run(candidate, self.root),
            },
        }
        self._crash_point("after_promotion_decision")
        atomic_write_json(round_dir / "report.prepared.json", report)
        return report

    monkeypatch.setattr(CoTraining, "_run_round", fake_round)
    crash_at = {"stage": None}

    def crash(self, stage):
        if stage == crash_at["stage"]:
            raise RuntimeError("injected crash")

    monkeypatch.setattr(CoTraining, "_crash_point", crash)
    cfg1 = replace(cfg0, rounds=1)
    for stage in (
            "after_candidate_evaluation", "before_promotion_decision",
            "after_promotion_decision", "after_artifacts_prepared"):
        crash_at["stage"] = stage
        with pytest.raises(RuntimeError, match="injected crash"):
            run_cotraining(cfg1)
        uncommitted = json.loads(
            (Path(cfg1.output_dir) / "run_state.json").read_text(encoding="utf-8"))
        assert uncommitted["completed_rounds"] == []
        assert uncommitted["active_protagonist_sha256"] == incumbent_hash
    crash_at["stage"] = "after_state_commit"
    with pytest.raises(RuntimeError, match="injected crash"):
        run_cotraining(cfg1)
    state = json.loads(
        (Path(cfg1.output_dir) / "run_state.json").read_text(encoding="utf-8"))
    assert state["completed_rounds"] == [1]
    assert state["active_protagonist_sha256"] != incumbent_hash
    (Path(cfg1.output_dir) / "best.pt").write_bytes(b"stale")

    def must_not_run(*args, **kwargs):
        raise AssertionError("committed round was rerun")

    monkeypatch.setattr(CoTraining, "_run_round", must_not_run)
    monkeypatch.setattr(CoTraining, "_crash_point", lambda self, stage: None)
    first = run_cotraining(cfg1)["run_state"]
    second = run_cotraining(cfg1)["run_state"]
    assert first["completed_rounds"] == second["completed_rounds"] == [1]
    assert sha256_file(Path(cfg1.output_dir) / "best.pt") == \
        first["active_protagonist_sha256"]
    assert first["active_replay_sha256"] == second["active_replay_sha256"]
    assert first["active_level_replay_sha256"] == \
        second["active_level_replay_sha256"]
    assert len(second["commits"]) == 1

    # A crash after the mirror refresh is also post-commit and must not rerun.
    monkeypatch.setattr(CoTraining, "_run_round", fake_round)
    monkeypatch.setattr(CoTraining, "_crash_point", crash)
    crash_at["stage"] = "after_best_refresh"
    cfg2 = replace(cfg1, rounds=2)
    with pytest.raises(RuntimeError, match="injected crash"):
        run_cotraining(cfg2)
    after_mirror = json.loads(
        (Path(cfg2.output_dir) / "run_state.json").read_text(encoding="utf-8"))
    assert after_mirror["completed_rounds"] == [1, 2]
    assert sha256_file(Path(cfg2.output_dir) / "best.pt") == \
        after_mirror["active_protagonist_sha256"]
    monkeypatch.setattr(CoTraining, "_run_round", must_not_run)
    monkeypatch.setattr(CoTraining, "_crash_point", lambda self, stage: None)
    recovered = run_cotraining(cfg2)["run_state"]
    assert recovered["completed_rounds"] == [1, 2]

    # The same deterministic fixture without crashes reaches the same commit.
    clean_cfg = replace(
        cfg0, output_dir=str(tmp_path / "clean-run"), rounds=2)
    monkeypatch.setattr(CoTraining, "_run_round", fake_round)
    uninterrupted = run_cotraining(clean_cfg)["run_state"]
    assert uninterrupted["completed_rounds"] == recovered["completed_rounds"]
    assert uninterrupted["active_protagonist_sha256"] == \
        recovered["active_protagonist_sha256"]
    assert uninterrupted["history"] == recovered["history"]
    assert uninterrupted["active_replay_sha256"] == recovered["active_replay_sha256"]
    assert uninterrupted["active_level_replay_sha256"] == \
        recovered["active_level_replay_sha256"]

    promote["value"] = False
    rejected_cfg = replace(
        cfg0, output_dir=str(tmp_path / "rejected-run"), rounds=1)
    crash_at["stage"] = "after_artifacts_prepared"
    monkeypatch.setattr(CoTraining, "_crash_point", crash)
    with pytest.raises(RuntimeError, match="injected crash"):
        run_cotraining(rejected_cfg)
    rejected = json.loads(
        (Path(rejected_cfg.output_dir) / "run_state.json").read_text(
            encoding="utf-8"))
    assert rejected["completed_rounds"] == []
    rejected_incumbent = rejected["active_protagonist_sha256"]
    monkeypatch.setattr(CoTraining, "_crash_point", lambda self, stage: None)
    rejected = run_cotraining(rejected_cfg)["run_state"]
    assert rejected["history"][0]["promoted"] is False
    assert rejected["active_protagonist_sha256"] == rejected_incumbent


def test_cotraining_stops_durably_after_first_promotion(tmp_path, monkeypatch):
    base = _tiny_base_dataset(tmp_path / "base.jsonl")
    protagonist = _tiny_protagonist(tmp_path / "prot.pt")
    designer = _tiny_designer(tmp_path / "designer.pt")
    cfg0 = replace(_tiny_config(
        tmp_path / "run", base, rounds=0,
        protagonist_checkpoint=protagonist, designer_checkpoint=designer),
        stop_after_promotion=True)
    run_cotraining(cfg0)
    candidate_source = _tiny_protagonist(tmp_path / "candidate-source.pt")
    calls = []

    def fake_round(self, rnd, base_records, split, enc, model_cfg, value_norm,
                   replay, level_replay, run_state):
        calls.append(rnd)
        round_dir = self.root / f"round_{rnd:03d}"
        round_dir.mkdir(parents=True, exist_ok=True)
        candidate = round_dir / "candidate.pt"
        atomic_copy(candidate_source, candidate)
        promoted = rnd == 2
        if promoted:
            run_state["active_protagonist_checkpoint"] = relative_to_run(
                candidate, self.root)
            run_state["active_protagonist_sha256"] = sha256_file(candidate)
            run_state["active_protagonist_source_round"] = rnd
        report = {
            "commit_status": "prepared",
            "summary": {"round": rnd, "promoted": promoted},
            "protagonist": {
                "candidate_checkpoint": relative_to_run(candidate, self.root),
                "candidate_checkpoint_sha256": sha256_file(candidate),
            },
        }
        atomic_write_json(round_dir / "report.prepared.json", report)
        return report

    monkeypatch.setattr(CoTraining, "_run_round", fake_round)
    cfg = replace(cfg0, rounds=3)
    first = run_cotraining(cfg)
    assert calls == [1, 2]
    assert first["rounds"] == 2
    assert first["stopped_after_promotion"] is True
    assert first["run_state"]["completed_rounds"] == [1, 2]
    assert first["run_state"]["terminal_stop"]["round"] == 2

    calls.clear()
    resumed = run_cotraining(cfg)
    assert calls == []
    assert resumed["run_state"]["completed_rounds"] == [1, 2]
    assert resumed["stopped_after_promotion"] is True


def test_cotraining_legacy_state_migrates_to_immutable_checkpoint(tmp_path):
    base = _tiny_base_dataset(tmp_path / "base.jsonl")
    protagonist = _tiny_protagonist(tmp_path / "prot.pt")
    designer = _tiny_designer(tmp_path / "designer.pt")
    cfg = _tiny_config(
        tmp_path / "run", base, rounds=0,
        protagonist_checkpoint=protagonist, designer_checkpoint=designer)
    root = Path(cfg.output_dir)
    root.mkdir(parents=True)
    best = root / "best.pt"
    atomic_copy(protagonist, best)
    state = {
        "completed_rounds": [1],
        "best_protagonist": str(best),
        "history": [{"round": 1, "promoted": False}],
    }
    runner = CoTraining(cfg)
    runner._prepare_checkpoint_state(state)
    committed = resolve_committed_protagonist(root, state)
    assert committed != best
    assert state["active_protagonist_source_round"] == 1
    assert sha256_file(committed) == sha256_file(best)
    assert state["designer_checkpoint"] == str(designer)
    assert state["designer_checkpoint_sha256"] == sha256_file(designer)


def test_cotraining_resume_rejects_changed_promotion_metric(tmp_path):
    base = _tiny_base_dataset(tmp_path / "base.jsonl")
    protagonist = _tiny_protagonist(tmp_path / "prot.pt")
    designer = _tiny_designer(tmp_path / "designer.pt")
    legacy_cfg = _tiny_config(
        tmp_path / "run", base, rounds=0,
        protagonist_checkpoint=protagonist, designer_checkpoint=designer)
    legacy_cfg = replace(legacy_cfg, promotion_metric="search_optimal_acc")
    run_cotraining(legacy_cfg)
    runner = CoTraining(replace(
        legacy_cfg, promotion_metric="search_confirmed_optimal_rate"))
    with pytest.raises(
            ExperimentIdentityError,
            match="semantic_config.promotion_metric"):
        runner.run()
    persisted = json.loads(
        (Path(legacy_cfg.output_dir) / "config.json").read_text(encoding="utf-8"))
    assert runner.cfg.promotion_metric == "search_confirmed_optimal_rate"
    assert persisted["promotion_metric"] == "search_optimal_acc"


def _tiny_protagonist(path):
    model_cfg = ModelConfig(channels=4, residual_blocks=1, value_hidden_size=8)
    model = PolicyValueNet(ENC, model_cfg)
    save_checkpoint(path, model=model, optimizer=None, scheduler=None, epoch=0,
                    best_val_metric=None, encoding_config=ENC,
                    model_config=model_cfg, value_norm=ValueNormConfig(), seed=0,
                    dataset_version=1, split_identity=None, metrics={})
    return Path(path)


def _tiny_designer(path):
    mc = DesignerModelConfig(channels=4, residual_blocks=1, hidden_size=8)
    model = DesignerNet(ENC, mc)
    save_designer(path, model=model, encoding_config=ENC, model_config=mc, seed=0)
    return Path(path)


def test_designer_training_freezes_protagonist(tmp_path):
    prot = tmp_path / "prot.pt"
    _tiny_protagonist(prot)
    before = _sha(prot)
    cfg = designer_train.TrainConfig(
        protagonist_checkpoint=str(prot), output_dir=str(tmp_path / "d"),
        episodes=2, episodes_per_iter=2, mutation_budget=4,
        protagonist_simulations=4, oracle_simulations=8, astar_max_nodes=20_000,
        seed=1, device="cpu", max_replay=20,
        generator=DesGen(rows=5, cols=5, color_count=2),
        model=DesignerModelConfig(channels=4, residual_blocks=1, hidden_size=8),
        ppo=PPOConfig(epochs=1, minibatch_size=8))
    designer_train.train_designer(cfg)
    assert _sha(prot) == before          # protagonist never written by designer


def test_protagonist_finetune_freezes_designer(tmp_path):
    """train_expert touches only the protagonist; a designer model is untouched."""
    from blocksort.expert_iteration.train import source_weights_for, train_expert
    designer = DesignerNet(ENC, DesignerModelConfig(channels=4, residual_blocks=1,
                                                    hidden_size=8))
    designer_before = copy.deepcopy(designer.state_dict())

    records = load_records(PV_SMOKE)[:64]
    weights = source_weights_for(records, 1, weight_exact_historical=1.0,
                                 weight_exact_new=1.0, weight_search=1.0)
    prot = PolicyValueNet(ENC, ModelConfig(channels=4, residual_blocks=1,
                                           value_hidden_size=8))
    prot_before = copy.deepcopy(prot.state_dict())
    train_expert(prot, records, weights, encoding_config=ENC,
                 value_norm=ValueNormConfig(), epochs=1, batch_size=32,
                 learning_rate=1e-3, weight_decay=0.0, grad_clip=1.0,
                 device=torch.device("cpu"), seed=0)
    designer_after = designer.state_dict()
    assert all(torch.equal(designer_before[k], designer_after[k])
               for k in designer_before)                       # designer frozen
    assert any(not torch.equal(prot_before[k], prot.state_dict()[k])
               for k in prot_before)                            # protagonist moved


# ----------------------------------------------------------------------
# Tiny end-to-end co-training (run, resume, replay, leakage, determinism)
# ----------------------------------------------------------------------


def test_fallback_evaluation_selects_distinct_initial_states_by_level():
    records = load_records(PV_SMOKE)
    level_keys = list(dict.fromkeys(
        record["static_level_signature"] for record in records))
    manifest = {
        "train_levels": level_keys[20:],
        "validation_levels": level_keys[:10],
        "test_levels": level_keys[10:20],
    }
    env = Environment()

    selected = loop_mod._level_balanced_eval_records(
        env, records, manifest, "validation", 8)
    reordered = loop_mod._level_balanced_eval_records(
        env, list(reversed(records)), manifest, "validation", 8)
    test_selected = loop_mod._level_balanced_eval_records(
        env, records, manifest, "test", None)

    assert len(selected) == 8
    assert len({record["static_level_signature"] for record in selected}) == 8
    assert [record["state_key"] for record in selected] == [
        record["state_key"] for record in reordered]
    assert all(
        record["state_key"]
        == env.canonical_key(env.initial_state(level_from_dict(record["level"])))
        for record in selected
    )
    assert {
        record["static_level_signature"] for record in selected
    }.isdisjoint({
        record["static_level_signature"] for record in test_selected
    })


def _tiny_base_dataset(path: Path, *, sigs: int = 16, per_sig: int = 5) -> Path:
    records = load_records(PV_SMOKE)
    by_sig: dict[str, int] = {}
    out = []
    for r in records:
        sig = r.get("static_level_signature") or r["level_id"]
        if by_sig.get(sig, 0) >= per_sig:
            continue
        if sig not in by_sig and len(by_sig) >= sigs:
            continue
        by_sig[sig] = by_sig.get(sig, 0) + 1
        out.append(r)
    import json
    with open(path, "w", encoding="utf-8") as fh:
        for r in out:
            fh.write(json.dumps(r) + "\n")
    return path


def _tiny_config(output_dir, base_dataset, *, rounds=2, seed=42,
                 protagonist_checkpoint=None, designer_checkpoint=None):
    return CoTrainingConfig(
        protagonist_checkpoint=str(
            protagonist_checkpoint or Path(output_dir).parent / "prot.pt"),
        designer_checkpoint=str(
            designer_checkpoint or Path(output_dir).parent / "designer.pt"),
        base_dataset=str(base_dataset), output_dir=str(output_dir),
        rounds=rounds, levels_per_round=5, seed=seed, device="cpu",
        frontier_backfill_target=0, min_fresh_levels_to_train=0,
        solve_rate_trials=2, astar_max_nodes=20_000, oracle_simulations=8,
        eval_astar_max_nodes=2_000, eval_astar_time_limit_seconds=None,
        eval_budgets=(1, 8), eval_limit=2, benchmark_total_limit=8,
        skip_forgetting_benchmark=True,
        promotion_budget=8,
        states_per_level=2, train_sample_size=64, epochs=1, batch_size=32,
        learning_rate=1e-3, max_protagonist_replay=2_000,
        designer_episodes=4, designer_episodes_per_iter=2, designer_ppo_epochs=1,
        designer_validation_episodes=1,
        designer_frontier_alignment_weight=0.0,
        max_designer_replay=50, benchmark_count=4, ood_rows=6, ood_cols=6,
        curriculum=CurriculumConfig(frontier_min_solve_rate=0.0,
                                    frontier_max_solve_rate=1.0),
        initial_curriculum=CurriculumState(rows=5, cols=5, color_count=2,
                                           density=0.4, mutation_budget=4,
                                           protagonist_simulations=4))


def test_shadow_learner_identity_is_initialized_and_resumes(tmp_path):
    base = _tiny_base_dataset(tmp_path / "base.jsonl")
    protagonist = _tiny_protagonist(tmp_path / "prot.pt")
    designer = _tiny_designer(tmp_path / "designer.pt")
    cfg = replace(
        _tiny_config(
            tmp_path / "run", base, rounds=0,
            protagonist_checkpoint=protagonist,
            designer_checkpoint=designer),
        initialize_only=True,
        shadow_learner_enabled=True,
    )
    initialized = run_cotraining(cfg)["run_state"]
    assert initialized["active_learner_checkpoint"] == \
        initialized["active_protagonist_checkpoint"]
    assert initialized["active_learner_sha256"] == \
        initialized["active_protagonist_sha256"]
    assert initialized["active_learner_anchor_sha256"] == \
        initialized["active_protagonist_sha256"]

    resumed = run_cotraining(cfg)["run_state"]
    assert resumed["active_learner_sha256"] == \
        initialized["active_learner_sha256"]
    assert resumed["active_learner_anchor_source_round"] == 0


def test_shadow_learner_imports_external_initial_checkpoint(tmp_path):
    base = _tiny_base_dataset(tmp_path / "base.jsonl")
    torch.manual_seed(101)
    champion = _tiny_protagonist(tmp_path / "champion.pt")
    torch.manual_seed(102)
    learner = _tiny_protagonist(tmp_path / "learner.pt")
    designer = _tiny_designer(tmp_path / "designer.pt")
    assert sha256_file(champion) != sha256_file(learner)
    cfg = replace(
        _tiny_config(
            tmp_path / "run", base, rounds=0,
            protagonist_checkpoint=champion,
            designer_checkpoint=designer),
        initialize_only=True,
        shadow_learner_enabled=True,
        initial_learner_checkpoint=str(learner),
    )

    state = run_cotraining(cfg)["run_state"]
    root = Path(cfg.output_dir)
    imported = root / state["active_learner_checkpoint"]
    active_champion = root / state["active_protagonist_checkpoint"]

    assert state["shadow_learner_initialization"] == "external_checkpoint"
    assert state["initial_learner_input_sha256"] == sha256_file(learner)
    assert imported == root / "learner" / "initial.pt"
    assert sha256_file(imported) == sha256_file(learner)
    assert sha256_file(active_champion) == sha256_file(champion)
    assert imported != active_champion
    assert state["active_learner_anchor_checkpoint"] == \
        state["active_learner_checkpoint"]
    assert state["active_learner_source_round"] == 0
    assert state["active_protagonist_source_round"] == 0

    resumed = run_cotraining(cfg)["run_state"]
    assert resumed["active_learner_sha256"] == sha256_file(learner)


def test_initial_learner_requires_shadow_learner():
    with pytest.raises(ValueError, match="requires shadow_learner_enabled"):
        CoTrainingConfig(initial_learner_checkpoint="learner.pt")


def test_shadow_learner_accumulates_and_evaluates_only_at_milestone(
        tmp_path, monkeypatch):
    base = _tiny_base_dataset(tmp_path / "base.jsonl")
    protagonist = _tiny_protagonist(tmp_path / "prot.pt")
    designer = _tiny_designer(tmp_path / "designer.pt")
    cfg = replace(
        _tiny_config(
            tmp_path / "run", base, rounds=2,
            protagonist_checkpoint=protagonist,
            designer_checkpoint=designer),
        levels_per_round=0,
        train_designer_each_round=False,
        curriculum_enabled=False,
        train_sample_size=8,
        shadow_learner_enabled=True,
        learner_milestone_interval=2,
    )
    evaluations = []

    def fake_eval(env, model, enc, vn, states, *, budgets, oracle, device,
                  c_puct=1.5, seed=0):
        evaluations.append(len(states))
        total = len(states)
        return {
            "states": total,
            "total_evaluated_count": total,
            "budgets": {str(budget): {
                "total_evaluated_count": total,
                "search_optimal_classification_count": total,
                "search_optimal_classification_coverage": 1.0,
                "search_confirmed_optimal_count": total,
                "search_confirmed_optimal_rate": 1.0,
                "search_optimal_acc": 1.0,
            } for budget in budgets},
        }

    monkeypatch.setattr(loop_mod, "evaluate_checkpoint", fake_eval)
    result = run_cotraining(cfg)
    state = result["run_state"]
    round1 = json.loads((
        Path(cfg.output_dir) / "round_001" / "report.json").read_text())[
            "protagonist"]
    round2 = json.loads((
        Path(cfg.output_dir) / "round_002" / "report.json").read_text())[
            "protagonist"]

    # The fallback fixture has one validation and one test level, so the only
    # four calls are champion/candidate on those two groups at round 2.
    assert evaluations == [1, 1, 1, 1]
    assert round1["promotion_evaluated"] is False
    assert round1["promotion_decision"] == "not_evaluated"
    assert round1["shadow_learner"]["continuation"]["decision"] == \
        "continue_pending_milestone"
    assert round2["learner_parent_checkpoint_sha256"] == \
        round1["candidate_checkpoint_sha256"]
    assert round2["promotion_evaluated"] is True
    assert round2["promoted"] is False
    assert state["active_protagonist_source_round"] == 0
    assert state["active_learner_source_round"] == 2
    assert state["active_learner_anchor_source_round"] == 2
    assert state["learner_milestones"][0]["continuation"]["decision"] == \
        "accept_as_anchor"

    resumed = run_cotraining(cfg)["run_state"]
    assert resumed["active_learner_sha256"] == state["active_learner_sha256"]


def test_generation_uses_unique_population_and_rejects_contradiction(
        tmp_path, monkeypatch):
    run_root = tmp_path / "run"
    run_root.mkdir()
    protagonist_path = _tiny_protagonist(run_root / "protagonist.pt")
    designer_path = _tiny_designer(run_root / "designer.pt")
    cfg = replace(
        _tiny_config(
            run_root, tmp_path / "unused.jsonl", rounds=1,
            protagonist_checkpoint=protagonist_path,
            designer_checkpoint=designer_path),
        levels_per_round=5,
        frontier_backfill_target=2,
        generation_checkpoint_interval=2,
        generation_progress_interval=10,
        curriculum=CurriculumConfig(
            frontier_min_solve_rate=0.2,
            frontier_max_solve_rate=0.7),
    )
    runner = CoTraining(cfg)
    runner.experiment_fingerprint = "f" * 64
    checkpoint = torch.load(
        protagonist_path, map_location="cpu", weights_only=False)
    previous_model = PolicyValueNet(
        ENC, ModelConfig(channels=4, residual_blocks=1, value_hidden_size=8))
    previous_model.load_state_dict(checkpoint["model_state"])
    run_state = {
        "active_protagonist_checkpoint": "protagonist.pt",
        "active_protagonist_sha256": sha256_file(protagonist_path),
        "designer_checkpoint": str(designer_path),
        "designer_checkpoint_sha256": sha256_file(designer_path),
        "accepted_fingerprints": [],
    }
    positions = [(1, 1), (1, 2), (2, 1), (2, 2), (1, 1)]
    episodes = []
    for index, (row, col) in enumerate(positions):
        level = level_from_dict({
            "name": f"staged-{index}",
            "cols": 5,
            "rows": 5,
            "blocks": [{"color": "red", "cells": [[row, col]]}],
            "exits": [{
                "edge": "left", "start": row, "length": 1, "color": "red",
            }],
        })
        episodes.append(SimpleNamespace(
            finalize=loop_mod.FinalizeResult(
                level=level, valid=True, errors=(), solvable=True,
                move_count=None, num_blocks=1, num_mutations=index),
            trajectory=[index],
        ))
    rates = iter((0.5, 0.0, 1.0, 0.0, 0.9))
    exact_scored = []

    monkeypatch.setattr(
        loop_mod, "rollout_episode",
        lambda *args, **kwargs: episodes.pop(0))
    monkeypatch.setattr(
        loop_mod, "estimate_solve_rate",
        lambda *args, **kwargs: SimpleNamespace(
            solve_rate=(rate := next(rates)),
            trials=3, solved=round(rate * 3),
            trial_budgets=tuple(kwargs["simulation_budgets"])))

    def fake_score(_env, finalize, **kwargs):
        exact_scored.append(finalize.level)
        contradiction = len(exact_scored) == 2
        return SimpleNamespace(
            reward=SimpleNamespace(total=1.0, components={"novelty": 1.0}),
            structural=SimpleNamespace(to_dict=lambda: {}),
            oracle_result=lambda: {
                "oracle_solved": not contradiction,
                "oracle_exact": contradiction,
                "oracle_cost": None,
                "protagonist_solved": False,
                "protagonist_cost": None,
            },
            solver_metrics=lambda: {
                "oracle_method": (
                    "exact_astar" if contradiction else "construction_proof"),
                "protagonist_method": "search",
            },
        )

    monkeypatch.setattr(loop_mod, "score_level", fake_score)
    result = runner._generate_and_accept(
        1, cfg.initial_curriculum, ENC, ValueNormConfig(), previous_model,
        run_state)

    assert len(exact_scored) == 2
    assert result["generated_count"] == 5
    assert result["valid_count"] == 5
    assert result["unique_valid_count"] == 4
    assert result["solve_rate_sample_count"] == 4
    assert result["duplicate_count"] == 1
    assert result["duplicate_rate"] == pytest.approx(0.2)
    assert result["duplicate_rate_among_valid"] == pytest.approx(0.2)
    assert result["mean_solve_rate"] == pytest.approx(0.375)
    assert result["construction_proven_solvable_count"] == 5
    assert result["oracle_solvable_count"] == 4
    assert result["oracle_exact_contradiction_count"] == 1
    assert result["strict_frontier_accepted_count"] == 1
    assert result["frontier_backfill_selected_count"] == 1
    assert result["frontier_backfilled_count"] == 0
    assert result["frontier_acceptance_rate"] == pytest.approx(0.25)
    assert result["training_acceptance_rate"] == pytest.approx(0.25)
    assert result["exact_scored_selected_count"] == 2
    assert result["designer_reward_evaluated_count"] == 2
    assert result["rejections"]["oracle_unsolved"] == 1
    assert len(result["accepted_levels"]) == 1
    assert len(result["accepted_records"]) == 1
    assert result["frontier_diagnostics"]["oracle_rejections"] == [{
        "fingerprint": loop_mod.level_fingerprint(
            runner.env, exact_scored[1]),
        "selection_mode": "ranked_backfill",
        "solve_rate": 0.0,
        "oracle_exact": True,
        "oracle_method": "exact_astar",
        "oracle_cost": None,
    }]
    assert result["frontier_diagnostics"]["schema_version"] == 4
    assert result["frontier_diagnostics"]["frontier_candidate_count"] == 4
    assert result["frontier_diagnostics"]["solve_rate_quantiles"] == {
        "p10": 0.0,
        "p50": 0.5,
        "p90": 1.0,
    }
    rejected_fingerprint = result["frontier_diagnostics"][
        "oracle_rejections"][0]["fingerprint"]
    assert rejected_fingerprint not in run_state["accepted_fingerprints"]
    assert sum(result["rejections"].values()) == 5
    assert all(
        record["solvability_evidence"] == "reverse_construction"
        for record in result["accepted_records"])
    partial = json.loads(
        (run_root / "round_001" / "generation.partial.json").read_text())
    assert len(partial["attempts"]) == 5


def test_cotraining_uses_fixed_rate_with_unequal_coverage(
        tmp_path, monkeypatch):
    base = _tiny_base_dataset(tmp_path / "base.jsonl")
    protagonist = _tiny_protagonist(tmp_path / "prot.pt")
    designer = _tiny_designer(tmp_path / "designer.pt")
    cfg = replace(
        _tiny_config(
            tmp_path / "run", base, rounds=1,
            protagonist_checkpoint=protagonist,
            designer_checkpoint=designer),
        levels_per_round=0, train_designer_each_round=False,
        curriculum_enabled=False, epochs=1, train_sample_size=8)
    scores = [0.8, 0.7, 0.8, 0.7]
    known_counts = [10, 8, 10, 8]

    def fake_eval(env, model, enc, vn, states, *, budgets, oracle, device,
                  c_puct=1.5, seed=0):
        score = scores.pop(0)
        known = known_counts.pop(0)
        confirmed = int(round(score * 10))
        return {
            "states": 10, "total_evaluated_count": 10,
            "budgets": {str(budget): {
                "total_evaluated_count": 10,
                "search_optimal_classification_count": known,
                "search_optimal_classification_coverage": known / 10,
                "search_confirmed_optimal_count": confirmed,
                "search_confirmed_optimal_rate": score,
                "search_optimal_acc": confirmed / known,
            } for budget in budgets},
        }

    monkeypatch.setattr(loop_mod, "evaluate_checkpoint", fake_eval)
    result = run_cotraining(cfg)
    report = json.loads(
        (Path(cfg.output_dir) / "round_001" / "report.json").read_text())
    protagonist_report = report["protagonist"]
    assert result["run_state"]["history"][0]["promoted"] is False
    assert protagonist_report["validation"]["candidate"]["budgets"]["8"][
        "search_optimal_acc"] == pytest.approx(7 / 8)
    assert protagonist_report["promotion_score_prev"] == pytest.approx(0.8)
    assert protagonist_report["promotion_score_candidate"] == pytest.approx(0.7)
    assert protagonist_report[
        "promotion_prev_classification_known_count"] == 10
    assert protagonist_report[
        "promotion_candidate_classification_known_count"] == 8


def test_cotraining_budget_sweep_promotion_uses_all_configured_budgets(
        tmp_path, monkeypatch):
    base = _tiny_base_dataset(tmp_path / "base.jsonl")
    protagonist = _tiny_protagonist(tmp_path / "prot.pt")
    designer = _tiny_designer(tmp_path / "designer.pt")
    cfg = replace(
        _tiny_config(
            tmp_path / "run", base, rounds=1,
            protagonist_checkpoint=protagonist,
            designer_checkpoint=designer),
        levels_per_round=0, train_designer_each_round=False,
        curriculum_enabled=False, epochs=1, train_sample_size=8,
        promotion_metric="weighted_budget_sweep_confirmed_optimal_rate",
        promotion_budgets=(1, 2), promotion_budget_weights=(0.75, 0.25),
        eval_budgets=(1, 2))
    score_sets = [
        {1: 0.2, 2: 0.8}, {1: 0.4, 2: 0.6},
        {1: 0.2, 2: 0.8}, {1: 0.4, 2: 0.6},
    ]

    def fake_eval(env, model, enc, vn, states, *, budgets, oracle, device,
                  c_puct=1.5, seed=0):
        scores = score_sets.pop(0)
        return {
            "states": 10, "total_evaluated_count": 10,
            "budgets": {str(budget): {
                "total_evaluated_count": 10,
                "search_optimal_classification_count": 10,
                "search_optimal_classification_coverage": 1.0,
                "search_confirmed_optimal_count": int(scores[budget] * 10),
                "search_confirmed_optimal_rate": scores[budget],
                "search_optimal_acc": scores[budget],
            } for budget in budgets},
        }

    monkeypatch.setattr(loop_mod, "evaluate_checkpoint", fake_eval)
    monkeypatch.setattr(loop_mod, "train_expert", lambda *args, **kwargs: {
        "examples": len(args[1])})
    result = run_cotraining(cfg)
    report = json.loads(
        (Path(cfg.output_dir) / "round_001" / "report.json").read_text())
    protagonist_report = report["protagonist"]

    assert result["run_state"]["history"][0]["promoted"] is True
    assert protagonist_report["promotion_score_prev"] == pytest.approx(0.35)
    assert protagonist_report["promotion_score_candidate"] == pytest.approx(0.45)
    assert protagonist_report["promotion_comparison_count"] == 20
    assert protagonist_report["promotion_prev_confirmed_optimal_count"] == 10
    assert protagonist_report["promotion_candidate_confirmed_optimal_count"] == 10
    assert set(protagonist_report["promotion_per_budget"]) == {"1", "2"}


def test_cotraining_full_solve_promotion_uses_completed_levels(
        tmp_path, monkeypatch):
    base = _tiny_base_dataset(tmp_path / "base.jsonl")
    protagonist = _tiny_protagonist(tmp_path / "prot.pt")
    designer = _tiny_designer(tmp_path / "designer.pt")
    cfg = replace(
        _tiny_config(
            tmp_path / "run", base, rounds=1,
            protagonist_checkpoint=protagonist,
            designer_checkpoint=designer),
        levels_per_round=0, train_designer_each_round=False,
        curriculum_enabled=False, epochs=1, train_sample_size=8,
        promotion_metric="weighted_budget_sweep_solve_rate",
        promotion_budgets=(4, 8, 16),
        promotion_budget_weights=(0.2, 0.3, 0.5),
        promotion_margin=0.01,
        eval_budgets=(4, 8, 16))
    solve_sets = [
        {4: 1, 8: 2, 16: 3}, {4: 1, 8: 3, 16: 5},
        {4: 1, 8: 2, 16: 3}, {4: 1, 8: 3, 16: 5},
    ]

    def fake_eval(env, model, enc, vn, states, *, budgets, oracle, device,
                  c_puct=1.5, seed=0):
        solved = solve_sets.pop(0)
        total = 10
        return {
            "states": total,
            "total_evaluated_count": total,
            "budgets": {str(budget): {
                "total_evaluated_count": total,
                "search_solved_count": solved[budget],
                "search_solve_rate_total": solved[budget] / total,
            } for budget in budgets},
        }

    monkeypatch.setattr(loop_mod, "evaluate_checkpoint", fake_eval)
    monkeypatch.setattr(loop_mod, "train_expert", lambda *args, **kwargs: {
        "examples": len(args[1])})
    result = run_cotraining(cfg)
    report = json.loads(
        (Path(cfg.output_dir) / "round_001" / "report.json").read_text())
    protagonist_report = report["protagonist"]

    assert result["run_state"]["history"][0]["promoted"] is True
    assert protagonist_report["promotion_evidence_kind"] == "solved"
    assert protagonist_report["promotion_score_prev"] == pytest.approx(0.23)
    assert protagonist_report["promotion_score_candidate"] == \
        pytest.approx(0.36)
    assert protagonist_report["promotion_prev_solved_count"] == 6
    assert protagonist_report["promotion_candidate_solved_count"] == 9
    assert "promotion_prev_confirmed_optimal_count" not in protagonist_report


def test_cotraining_skips_training_and_promotion_without_fresh_levels(
        tmp_path, monkeypatch):
    base = _tiny_base_dataset(tmp_path / "base.jsonl")
    protagonist = _tiny_protagonist(tmp_path / "prot.pt")
    designer = _tiny_designer(tmp_path / "designer.pt")
    cfg = replace(
        _tiny_config(
            tmp_path / "run", base, rounds=1,
            protagonist_checkpoint=protagonist,
            designer_checkpoint=designer),
        levels_per_round=0, min_fresh_levels_to_train=1,
        train_designer_each_round=False, curriculum_enabled=False)
    scores = iter((0.1, 0.9, 0.1, 0.9))

    def fake_eval(env, model, enc, vn, states, *, budgets, oracle, device,
                  c_puct=1.5, seed=0):
        score = next(scores)
        total = 10
        confirmed = round(score * total)
        return {
            "states": total,
            "total_evaluated_count": total,
            "budgets": {str(budget): {
                "total_evaluated_count": total,
                "search_optimal_classification_count": total,
                "search_optimal_classification_coverage": 1.0,
                "search_confirmed_optimal_count": confirmed,
                "search_confirmed_optimal_rate": score,
                "search_optimal_acc": score,
            } for budget in budgets},
        }

    monkeypatch.setattr(loop_mod, "evaluate_checkpoint", fake_eval)

    def unexpected_train(*args, **kwargs):
        raise AssertionError("replay-only fine-tuning must be skipped")

    monkeypatch.setattr(loop_mod, "train_expert", unexpected_train)
    result = run_cotraining(cfg)
    report = json.loads(
        (Path(cfg.output_dir) / "round_001" / "report.json").read_text())
    protagonist_report = report["protagonist"]

    assert result["run_state"]["active_protagonist_source_round"] == 0
    assert protagonist_report["training_performed"] is False
    assert protagonist_report["train"]["skipped"] is True
    assert protagonist_report["train"]["fresh_levels"] == 0
    assert protagonist_report["promotion_decision"] == "skipped"
    assert protagonist_report["promoted"] is False


def test_external_final_test_is_sealed_during_cotraining_round(
        tmp_path, monkeypatch):
    base = _tiny_base_dataset(tmp_path / "base.jsonl")
    protagonist = _tiny_protagonist(tmp_path / "prot.pt")
    designer = _tiny_designer(tmp_path / "designer.pt")
    pool = tmp_path / "hard_pool.jsonl"
    split_path = tmp_path / "hard_pool_split.json"
    pool_records = []
    for index in range(4):
        level_data = {
            "name": f"sealed-eval-{index}",
            "cols": 4 + index,
            "rows": 4 + index,
            "blocks": [{"color": "red", "cells": [[1, 1]]}],
            "exits": [{
                "edge": "left", "start": 1, "length": 1, "color": "red",
            }],
        }
        signature = static_level_signature(level_from_dict(level_data))
        pool_records.append({
            "level_id": signature,
            "static_level_signature": signature,
            "level": level_data,
        })
    pool.write_text(
        "".join(json.dumps(record) + "\n" for record in pool_records),
        encoding="utf-8")
    manifest = create_eval_split_manifest(
        pool, split_path, validation_count=2, split_seed=1729)
    validation_signatures = {
        item["signature"] for item in manifest["promotion_validation"]}
    final_test_signatures = {
        item["signature"] for item in manifest["final_test"]}

    cfg = replace(
        _tiny_config(
            tmp_path / "run", base, rounds=1,
            protagonist_checkpoint=protagonist,
            designer_checkpoint=designer),
        levels_per_round=0, train_designer_each_round=False,
        curriculum_enabled=False, epochs=1, train_sample_size=8,
        eval_levels_dataset=str(pool),
        eval_split_manifest=str(split_path),
        eval_validation_count=2, eval_split_seed=1729)
    evaluated_signatures = []

    def fake_eval(env, model, enc, vn, states, *, budgets, oracle, device,
                  c_puct=1.5, seed=0):
        signatures = {
            static_level_signature(state.level) for state in states}
        evaluated_signatures.append(signatures)
        total = len(states)
        return {
            "states": total,
            "total_evaluated_count": total,
            "budgets": {str(budget): {
                "total_evaluated_count": total,
                "search_optimal_classification_count": total,
                "search_optimal_classification_coverage": 1.0,
                "search_confirmed_optimal_count": total,
                "search_confirmed_optimal_rate": 1.0,
                "search_optimal_acc": 1.0,
            } for budget in budgets},
        }

    monkeypatch.setattr(loop_mod, "evaluate_checkpoint", fake_eval)
    monkeypatch.setattr(loop_mod, "train_expert", lambda *args, **kwargs: {
        "examples": len(args[1])})
    run_cotraining(cfg)
    report = json.loads(
        (Path(cfg.output_dir) / "round_001" / "report.json").read_text())
    protagonist_report = report["protagonist"]

    assert evaluated_signatures == [
        validation_signatures, validation_signatures]
    assert all(
        signatures.isdisjoint(final_test_signatures)
        for signatures in evaluated_signatures)
    assert protagonist_report["evaluation_split"]["used_test_count"] == 0
    assert protagonist_report["evaluation_split"][
        "used_validation_level_count"] == 2
    assert protagonist_report["evaluation_split"][
        "used_test_level_count"] == 0
    assert protagonist_report["evaluation_split"]["final_test_status"] == \
        "sealed"
    assert protagonist_report["frozen_test"] == {
        "status": "sealed",
        "reason": (
            "external final-test levels are evaluated only by the explicit "
            "one-shot final-test command"),
        "previous": None,
        "candidate": None,
    }


@pytest.fixture(scope="module")
def tiny_run(tmp_path_factory):
    root = tmp_path_factory.mktemp("cotrain")
    base = _tiny_base_dataset(root / "base.jsonl")
    out = root / "run"
    _tiny_protagonist(root / "prot.pt")
    _tiny_designer(root / "designer.pt")
    cfg = _tiny_config(out, base, rounds=2)
    designer_hash = _sha(root / "designer.pt")
    result = run_cotraining(cfg)
    return {"root": root, "out": out, "cfg": cfg, "result": result,
            "designer_input_hash": designer_hash}


def test_run_completes_all_rounds(tiny_run):
    rs = tiny_run["result"]["run_state"]
    assert rs["completed_rounds"] == [1, 2]
    assert (tiny_run["out"] / "best.pt").exists()
    assert len(rs["history"]) == 2
    for h in rs["history"]:
        assert h["generated"] == 5
        assert h["generated_count"] == 5
        assert h["valid_count"] == h["valid"]
        assert h["oracle_solvable_count"] == h["oracle_solvable"]
        assert h["duplicate_count"] == h["duplicates"]
        assert h["accepted_count"] == h["accepted"]
        assert sum(h["rejections"].values()) == h["generated_count"]
        assert set(h["rejection_percentages"]) == set(h["rejections"])
        assert 0.0 <= h["frontier_acceptance_rate"] <= 1.0
        assert h["promotion_total_count"] > 0
        assert h["promotion_prev_classification_known_count"] <= \
            h["promotion_total_count"]
        assert h["promotion_candidate_classification_known_count"] <= \
            h["promotion_total_count"]
        assert (h["label_exact"] + h["label_exact_path"]
                + h["label_search"]) <= (
            h["accepted_count"] * (1 + 2 * tiny_run["cfg"].states_per_level))
    for round_number in (1, 2):
        report = json.loads((
            tiny_run["out"] / f"round_{round_number:03d}" / "report.json"
        ).read_text())
        evaluation = report["protagonist"]["evaluation_split"]
        generation = report["generation"]["frontier_diagnostics"]
        assert generation["generator_max_blocks"] == ENC.max_blocks
        assert generation["protagonist_encoding_max_blocks"] == ENC.max_blocks
        assert generation["designer_encoding_max_blocks"] == ENC.max_blocks
        assert evaluation["selection_policy"] == \
            "one_initial_state_per_level_v1"
        assert evaluation["used_validation_count"] == \
            evaluation["used_validation_level_count"]
        assert evaluation["used_test_count"] == \
            evaluation["used_test_level_count"]
        train = report["protagonist"]["train"]
        composition = train["replay_age_composition"]
        assert composition["policy"] == \
            "fresh_recent_historical_quota_v1"
        assert sum(composition["realized_counts"].values()) == \
            train["examples"]
        assert sum(train["gradient_weight_mass_by_age"].values()) == \
            pytest.approx(train["gradient_weight_mass"]["policy_total"])
        assert sum(train["gradient_weight_fraction_by_age"].values()) == \
            pytest.approx(1.0)
        assert train["loss_weighting_policy"] == \
            "caller_provided_nonuniform_loss_weights_v1"
        round_dir = tiny_run["out"] / f"round_{round_number:03d}"
        sample_manifest = json.loads((
            round_dir / "training_sample_manifest.json").read_text(
                encoding="utf-8"))
        sample_records = (
            round_dir / "training_sample.jsonl").read_text(
                encoding="utf-8").splitlines()
        source_sample_records = (
            round_dir / "training_sample_source.jsonl").read_text(
                encoding="utf-8").splitlines()
        policy_target_summary = json.loads((
            round_dir / "training_policy_target_summary.json").read_text(
                encoding="utf-8"))
        policy_weights = json.loads((
            round_dir / "training_policy_weights.json").read_text(
                encoding="utf-8"))
        value_weights = json.loads((
            round_dir / "training_value_weights.json").read_text(
                encoding="utf-8"))
        effective_value_weights = json.loads((
            round_dir / "training_effective_value_weights.json").read_text(
                encoding="utf-8"))
        assert sample_manifest["record_count"] == train["examples"]
        assert sample_manifest["schema_version"] == 2
        assert sample_manifest["semantics"] == \
            "cotraining_protagonist_training_sample_v2"
        assert len(sample_records) == train["examples"]
        assert len(source_sample_records) == train["examples"]
        assert len(policy_weights) == train["examples"]
        assert len(value_weights) == train["examples"]
        assert len(effective_value_weights) == train["examples"]
        assert sample_manifest["sample"]["sha256"] == _sha(
            round_dir / "training_sample.jsonl")
        assert sample_manifest["source_sample"]["sha256"] == _sha(
            round_dir / "training_sample_source.jsonl")
        assert sample_manifest["policy_targets"]["profile"] == \
            tiny_run["cfg"].policy_target_profile
        assert sample_manifest["policy_targets"]["summary"]["sha256"] == \
            _sha(round_dir / "training_policy_target_summary.json")
        assert policy_target_summary["profile"] == \
            tiny_run["cfg"].policy_target_profile
        assert policy_target_summary["sample_sha256"] == \
            sample_manifest["sample"]["sha256"]
        assert policy_target_summary["source_sample_sha256"] == \
            sample_manifest["source_sample"]["sha256"]
        assert policy_target_summary["max_suboptimal_probability_mass"] == 0.0
        assert train["policy_target_profile"] == \
            tiny_run["cfg"].policy_target_profile
        assert train["policy_target_sha256"] == \
            policy_target_summary["policy_target_sha256"]
        assert sample_manifest["policy_weights"]["sha256"] == _sha(
            round_dir / "training_policy_weights.json")
        assert sample_manifest["value_weights"]["sha256"] == _sha(
            round_dir / "training_value_weights.json")
        assert sample_manifest["sampling"]["seed"] == \
            tiny_run["cfg"].seed * 13 + round_number
        assert train["training_sample"]["manifest_sha256"] == _sha(
            round_dir / "training_sample_manifest.json")
        assert not (
            tiny_run["out"] / f"round_{round_number:03d}"
            / "generation.partial.json").exists()


def test_validation_cache_reuses_committed_incumbent(tiny_run):
    report = json.loads(
        (tiny_run["out"] / "round_002" / "report.json").read_text())
    events = {
        event["role"]: event
        for event in report["protagonist"]["validation_cache"]
    }

    assert events["incumbent_validation"]["hit"] is True


def test_input_designer_checkpoint_frozen(tiny_run):
    # The original designer checkpoint file is never mutated in place.
    assert _sha(tiny_run["root"] / "designer.pt") == tiny_run["designer_input_hash"]


def test_active_designer_checkpoint_has_committed_hash(tiny_run):
    state = tiny_run["result"]["run_state"]
    checkpoint = Path(state["designer_checkpoint"])

    assert checkpoint.is_file()
    assert state["designer_checkpoint_sha256"] == _sha(checkpoint)


def test_active_designer_checkpoint_corruption_is_rejected(tmp_path):
    checkpoint = _tiny_designer(tmp_path / "designer.pt")
    expected_hash = _sha(checkpoint)
    model_config = DesignerModelConfig(
        channels=4, residual_blocks=1, hidden_size=8)
    replacement = DesignerNet(ENC, model_config)
    with torch.no_grad():
        for parameter in replacement.parameters():
            parameter.zero_()
    save_designer(
        checkpoint, model=replacement, encoding_config=ENC,
        model_config=model_config, seed=1)
    runner = CoTraining(CoTrainingConfig(
        designer_checkpoint=str(checkpoint),
        output_dir=str(tmp_path / "run"), device="cpu"))

    with pytest.raises(
            ExperimentSpecIntegrityError,
            match="designer checkpoint integrity"):
        runner._validate_existing_run({
            "completed_rounds": [],
            "designer_checkpoint": str(checkpoint),
            "designer_checkpoint_sha256": expected_hash,
        })


def test_current_state_without_active_designer_is_rejected(tmp_path):
    checkpoint = _tiny_designer(tmp_path / "designer.pt")
    runner = CoTraining(CoTrainingConfig(
        designer_checkpoint=str(checkpoint),
        output_dir=str(tmp_path / "run"), device="cpu"))

    with pytest.raises(
            ExperimentSpecIntegrityError,
            match="no active designer checkpoint"):
        runner._validate_existing_run({
            "schema_version": 2,
            "completed_rounds": [],
        })


def test_designer_training_commits_new_checkpoint_hash(tmp_path, monkeypatch):
    root = tmp_path / "run"
    root.mkdir()
    protagonist = _tiny_protagonist(root / "protagonist.pt")
    designer = _tiny_designer(tmp_path / "designer.pt")
    replacement = _tiny_designer(tmp_path / "replacement.pt")
    runner = CoTraining(CoTrainingConfig(
        protagonist_checkpoint=str(protagonist),
        designer_checkpoint=str(designer), output_dir=str(root),
        device="cpu"))
    state = {
        "active_protagonist_checkpoint": "protagonist.pt",
        "active_protagonist_sha256": _sha(protagonist),
        "designer_checkpoint": str(designer),
        "designer_checkpoint_sha256": _sha(designer),
    }
    captured = {}

    def fake_train_designer(cfg):
        captured["config"] = cfg
        return {
            "best_checkpoint": str(replacement),
            "iterations": 1,
            "best_mean_reward": 1.0,
            "best_validation_metrics": {
                "frontier_in_band_rate": 0.5,
            },
            "best_selection_metric": {
                "name":
                    "frontier_in_band_alignment_reward_lexicographic_v1",
            },
        }

    monkeypatch.setattr(loop_mod, "train_designer", fake_train_designer)

    summary = runner._train_designer(
        1, root / "round_001", CurriculumState(), state)

    assert state["designer_checkpoint"] == str(replacement)
    assert state["designer_checkpoint_sha256"] == _sha(replacement)
    assert summary["designer_checkpoint_sha256"] == _sha(replacement)
    assert captured["config"].validation_episodes == 8
    assert captured["config"].frontier_solve_rate_trials == 5
    assert captured["config"].frontier_alignment_weight == pytest.approx(1.0)
    assert captured["config"].frontier_min_solve_rate == pytest.approx(0.2)
    assert captured["config"].frontier_max_solve_rate == pytest.approx(0.7)
    assert summary["best_validation_metrics"]["frontier_in_band_rate"] == \
        pytest.approx(0.5)
    assert summary["best_selection_metric"]["name"] == \
        "frontier_in_band_alignment_reward_lexicographic_v1"


def test_replay_persisted_across_rounds(tiny_run):
    import json
    manifest = json.loads((tiny_run["out"] / "replay" / "manifest.json")
                          .read_text(encoding="utf-8"))
    # iteration 0 seed plus at least one training round shard.
    assert 0 in manifest["iterations"]
    assert any(it >= 1 for it in manifest["iterations"])


def test_training_artifacts_are_immutable_on_retry(tmp_path):
    text_path = tmp_path / "sample.jsonl"
    json_path = tmp_path / "weights.json"

    loop_mod._write_or_verify_text(
        text_path, "{\"state\":1}\n", label="sample")
    loop_mod._write_or_verify_text(
        text_path, "{\"state\":1}\n", label="sample")
    loop_mod._write_or_verify_json(
        json_path, [1.0, 0.5], label="weights")
    loop_mod._write_or_verify_json(
        json_path, [1.0, 0.5], label="weights")

    with pytest.raises(RuntimeError, match="deterministic reconstruction"):
        loop_mod._write_or_verify_text(
            text_path, "{\"state\":2}\n", label="sample")
    with pytest.raises(RuntimeError, match="deterministic reconstruction"):
        loop_mod._write_or_verify_json(
            json_path, [1.0, 0.25], label="weights")


def test_no_leakage_into_replay(tiny_run):
    import json
    out = tiny_run["out"]
    split = json.loads((out / "splits.json").read_text(encoding="utf-8"))
    frozen = set(split["validation_levels"]) | set(split["test_levels"])
    shards = (out / "replay" / "shards")
    seen_frozen = False
    for shard in shards.glob("*.jsonl"):
        for line in shard.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            rec = json.loads(line)
            if rec.get("static_level_signature") in frozen:
                seen_frozen = True
    assert seen_frozen is False


def test_resume_skips_completed_rounds(tiny_run):
    # Re-running the same config resumes: completed rounds stay the same.
    result2 = run_cotraining(tiny_run["cfg"])
    assert result2["run_state"]["completed_rounds"] == [1, 2]
    assert result2["rounds"] == 2


def test_deterministic_tiny_run(tmp_path):
    base = _tiny_base_dataset(tmp_path / "base.jsonl")
    protagonist_path = _tiny_protagonist(tmp_path / "prot.pt")
    designer_path = _tiny_designer(tmp_path / "designer.pt")
    expected_input_hashes = (
        _sha(protagonist_path), _sha(designer_path), _sha(base))

    def metric_tuple(history):
        return (
            history["generated"],
            history["valid"],
            history["oracle_solvable"],
            history["duplicates"],
            history["accepted"],
            round(history["frontier_acceptance_rate"], 6),
            round(history["mean_solve_rate"], 6),
            round(history["mean_designer_reward"], 6),
            history["label_exact"],
            history["label_search"],
            round(history["promotion_score_prev"], 6),
            round(history["promotion_score_candidate"], 6),
            history["promoted"],
        )

    metrics = []
    starting_hashes = []
    output_dirs = []
    for i in range(2):
        out = tmp_path / f"run{i}"
        assert not out.exists()
        output_dirs.append(out.resolve())
        starting_hashes.append(
            (_sha(protagonist_path), _sha(designer_path), _sha(base)))
        cfg = _tiny_config(
            out, base, rounds=1, seed=7,
            protagonist_checkpoint=protagonist_path,
            designer_checkpoint=designer_path)
        res = run_cotraining(cfg)
        h = res["run_state"]["history"][0]
        metrics.append(metric_tuple(h))
        assert starting_hashes[-1] == expected_input_hashes
        assert (_sha(protagonist_path), _sha(designer_path), _sha(base)) == \
            expected_input_hashes
        assert Path(res["run_state"]["best_protagonist"]).resolve().is_relative_to(
            out.resolve())
        assert Path(res["run_state"]["designer_checkpoint"]).resolve().is_relative_to(
            out.resolve())

    assert output_dirs[0] != output_dirs[1]
    assert starting_hashes == [expected_input_hashes, expected_input_hashes]
    assert metrics[0] == metrics[1]

    # A behaviorally different input checkpoint is not hidden by run seeding.
    stop_cfg = DesignerModelConfig(
        channels=4, residual_blocks=1, hidden_size=8)
    stop_designer = DesignerNet(ENC, stop_cfg)
    with torch.no_grad():
        for parameter in stop_designer.parameters():
            parameter.zero_()
        stop_designer.stop_head[-1].bias.fill_(100.0)
        stop_designer.policy_conv[-1].bias.fill_(-100.0)
    different_designer = tmp_path / "different-designer.pt"
    save_designer(
        different_designer, model=stop_designer, encoding_config=ENC,
        model_config=stop_cfg, seed=0)
    assert _sha(different_designer) != expected_input_hashes[1]
    different_out = tmp_path / "different-run"
    different_cfg = _tiny_config(
        different_out, base, rounds=1, seed=7,
        protagonist_checkpoint=protagonist_path,
        designer_checkpoint=different_designer)
    different_result = run_cotraining(different_cfg)
    different_metrics = metric_tuple(
        different_result["run_state"]["history"][0])
    assert different_metrics != metrics[0]
