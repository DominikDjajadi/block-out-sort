"""Tests for hybrid expert iteration.

Covers: exact-over-search priority, search fallback after A* exhaustion,
target-source metadata, cross-iteration deduplication, deterministic weighted
replay sampling, the permanently frozen split, candidate promotion/rejection,
resume behavior, no test/val leakage into training, and a tiny end-to-end run.
"""

from __future__ import annotations

from collections import Counter
import copy
import hashlib
import json
import shutil
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from blocksort.environment import Environment
from blocksort.oracle import Oracle, ValueResult
from blocksort.serialization import level_from_dict
from blocksort.dataset.schema import (
    LABEL_EXACT_PATH_POLICY,
    LABEL_SEARCH_VISIT_POLICY,
    deserialize_state,
)
from blocksort.signature import static_level_signature
from blocksort.solution import serialize_action
from blocksort.state import canonical_key

from blocksort.training.config import EncodingConfig, ModelConfig, ValueNormConfig
from blocksort.training.transaction import (
    CheckpointIntegrityError, atomic_copy, atomic_write_json,
    refresh_best_checkpoint, relative_to_run, resolve_committed_protagonist,
    sha256_file)
from blocksort.training.dataset import load_records
from blocksort.training.experiment_identity import ExperimentIdentityError
from blocksort.training.model import PolicyValueNet
from blocksort.training.checkpoint import save_checkpoint

from blocksort.expert_iteration.config import ExpertIterationConfig
from blocksort.expert_iteration.records import (
    SOURCE_EXACT,
    SOURCE_EXACT_PATH,
    SOURCE_SEARCH,
    build_exact_example,
    build_search_example,
    dedup_key,
    tag_historical,
)
from blocksort.expert_iteration.replay import (
    REPLAY_AGE_CURRENT,
    REPLAY_AGE_HISTORICAL,
    REPLAY_AGE_RECENT,
    ReplayBuffer,
    replay_age_bucket,
)
from blocksort.expert_iteration import evaluate as evaluate_mod
from blocksort.expert_iteration import iterate as iterate_mod
from blocksort.expert_iteration.iterate import ExpertIteration, run_expert_iteration
from blocksort.expert_iteration.labeling import (
    LABEL_MODE_HYBRID,
    LABEL_MODE_SEARCH_ONLY,
    label_states,
)
from blocksort.expert_iteration.evaluate import evaluate_checkpoint, promotion_score
from blocksort.expert_iteration.promotion import validate_promotion_evidence
from blocksort.expert_iteration.run import build_parser, config_from_args
from blocksort.search.evaluation import _Agg
from blocksort.cotraining import benchmark as benchmark_mod
from blocksort.cotraining.benchmark import _precomputed_lookup

REPO_ROOT = Path(__file__).resolve().parents[2]
BASE_DATASET = REPO_ROOT / "data" / "training" / "pv_examples.jsonl"


# ----------------------------------------------------------------------
# Fixtures / helpers
# ----------------------------------------------------------------------

@pytest.fixture(scope="module")
def base_records():
    return load_records(BASE_DATASET)


@pytest.fixture(scope="module")
def enc():
    return EncodingConfig()


def _tiny_checkpoint(path: Path, enc: EncodingConfig) -> Path:
    import torch  # noqa

    model_cfg = ModelConfig(channels=4, residual_blocks=1, value_hidden_size=8)
    value_norm = ValueNormConfig()
    model = PolicyValueNet(enc, model_cfg)
    save_checkpoint(
        path, model=model, optimizer=None, scheduler=None, epoch=0,
        best_val_metric=None, encoding_config=enc, model_config=model_cfg,
        value_norm=value_norm, seed=0, dataset_version=1, split_identity=None,
        metrics={})
    return path


def _adapter(enc, ckpt_path):
    import torch
    from blocksort.training.checkpoint import (configs_from_checkpoint,
                                               load_checkpoint, model_from_checkpoint)
    from blocksort.search.graph_search import BlocksortAdapter

    ckpt = load_checkpoint(ckpt_path, map_location="cpu")
    e, _mc, vn = configs_from_checkpoint(ckpt)
    model = model_from_checkpoint(ckpt, map_location=torch.device("cpu"))
    return BlocksortAdapter(Environment(), model, e, vn, torch.device("cpu")), vn


def _hard_initial_state(base_records):
    """A non-terminal initial state needing several moves (good for fallback)."""
    env = Environment()
    best = None
    for r in base_records:
        if r.get("optimal_remaining_moves", 0) and r["optimal_remaining_moves"] >= 3:
            level = level_from_dict(r["level"])
            st = env.initial_state(level)
            if not env.is_terminal(st):
                return env, st, r["level_id"]
    raise RuntimeError("no suitable state found")


def _exact_example(base_records, iteration=1):
    """Build one genuine exact example from the base dataset."""
    env = Environment()
    oracle = Oracle(env, max_nodes=300_000)
    for r in base_records:
        level = level_from_dict(r["level"])
        state = deserialize_state(level, r["state"])
        if env.is_terminal(state):
            continue
        analysis = oracle.analyze(state)
        rec = build_exact_example(
            analysis, state, level_id=r["level_id"], iteration=iteration,
            astar_max_nodes=300_000, teacher_checkpoint="ckpt", provenance={"x": 1})
        if rec is not None:
            return rec
    raise RuntimeError("no exact example")


def _state_identity(env, state):
    return static_level_signature(state.level), env.canonical_key(state)


class _SelectedActionAdapter:
    def __init__(self, env, selected):
        self.env = env
        self.selected = selected

    def evaluate(self, state):
        legal = self.env.legal_actions(state)
        wanted = self.selected[_state_identity(self.env, state)]
        priors = [1.0 if serialize_action(state, action) == wanted else 0.0
                  for action in legal]
        return priors, 0.0


class _MappedValueOracle:
    def __init__(self, env, values):
        self.env = env
        self.values = values

    def value(self, state):
        return self.values.get(
            _state_identity(self.env, state),
            ValueResult(value=None, exact=False, solvable=None))


_MISSING = object()


def _precomputed_entry(env, state, value, optimal_actions,
                       classification_complete=_MISSING):
    entry = {
        "static_signature": static_level_signature(state.level),
        "canonical_key": env.canonical_key(state),
        "value_result": {"value": value, "exact": True, "solvable": True},
        "optimal_actions": optimal_actions,
        "termination": "exact",
    }
    if classification_complete is not _MISSING:
        entry["classification_complete"] = classification_complete
    return entry


def _evaluate_selected_actions(monkeypatch, env, states, selected, oracle,
                               precomputed):
    adapter = _SelectedActionAdapter(env, selected)
    monkeypatch.setattr(
        evaluate_mod, "BlocksortAdapter",
        lambda *args, **kwargs: adapter)
    return evaluate_checkpoint(
        env, object(), EncodingConfig(), ValueNormConfig(), states,
        budgets=[], oracle=oracle, device=torch.device("cpu"),
        precomputed=precomputed)


# ----------------------------------------------------------------------
# Records / metadata
# ----------------------------------------------------------------------

def test_exact_example_metadata(base_records):
    rec = _exact_example(base_records)
    assert rec["target_source"] == SOURCE_EXACT
    assert rec["value_exact"] is True
    assert rec["teacher_checkpoint"] == "ckpt"
    assert rec["generation_iteration"] == 1
    assert rec["astar"]["termination_reason"] == "exact"
    assert rec["optimal_remaining_moves"] is not None
    assert abs(sum(rec["policy_target"]) - 1.0) < 1e-6


def test_search_example_metadata(enc, tmp_path, base_records):
    from blocksort.search.config import SearchConfig
    from blocksort.search.graph_search import GraphSearch

    ckpt = _tiny_checkpoint(tmp_path / "init.pt", enc)
    adapter, vn = _adapter(enc, ckpt)
    env, state, level_id = _hard_initial_state(base_records)
    result = GraphSearch(adapter, SearchConfig(simulations=16, temperature=1.0,
                                               value_normalization_constant=vn.constant,
                                               seed=0)).run(state)
    rec = build_search_example(
        result, state, level_id=level_id,
        static_signature=static_level_signature(state.level), iteration=2,
        teacher_checkpoint=str(ckpt), simulations=16, policy_temperature=1.0,
        provenance={"sampling": "x"}, astar_max_nodes=1,
        astar_reason="budget_exhausted", value_norm_constant=vn.constant)
    assert rec["target_source"] == SOURCE_SEARCH
    assert rec["label_kind"] == LABEL_SEARCH_VISIT_POLICY
    assert rec["value_exact"] is False
    assert rec["policy_exact"] is False
    assert rec["optimal_actions_complete"] is False
    assert rec["action_values_complete"] is False
    assert rec["optimal_remaining_moves"] is None
    assert rec["search"]["simulations"] == 16
    assert rec["value_target"]["estimate_kind"] == "bounded_search_estimate"
    assert rec["value_target"]["raw_optimal_moves"] >= rec["remaining_blocks"]
    assert rec["search"]["reported_search_value_cost"] \
        == pytest.approx(result.search_value_cost)
    assert abs(sum(rec["policy_target"]) - 1.0) < 1e-6


def test_search_example_clamps_value_to_physical_bounds(base_records):
    env, state, level_id = _hard_initial_state(base_records)
    result = SimpleNamespace(
        legal_action_locators=[{"action": 1}],
        visit_policy=[1.0],
        search_value_cost=1.0,
        visit_counts=[4],
        solved=False,
        solution_verified=False,
        solution_length=None,
    )

    record = build_search_example(
        result,
        state,
        level_id=level_id,
        static_signature=static_level_signature(state.level),
        iteration=1,
        teacher_checkpoint="teacher.pt",
        simulations=4,
        policy_temperature=1.0,
        provenance={"test": True},
        astar_max_nodes=1,
        astar_reason="budget_exhausted",
    )

    assert record["search"]["reported_search_value_cost"] == 1.0
    assert record["search"]["search_value_cost"] == state.remaining
    assert record["value_target"]["lower_bound"] == state.remaining
    assert record["value_target"]["upper_bound"] is None


def test_search_example_clamps_verified_solution_to_upper_bound(base_records):
    env, state, level_id = _hard_initial_state(base_records)
    solution_length = state.remaining + 2
    result = SimpleNamespace(
        legal_action_locators=[{"action": 1}],
        visit_policy=[1.0],
        search_value_cost=999.0,
        visit_counts=[4],
        solved=True,
        solution_verified=True,
        solution_length=solution_length,
    )

    record = build_search_example(
        result,
        state,
        level_id=level_id,
        static_signature=static_level_signature(state.level),
        iteration=1,
        teacher_checkpoint="teacher.pt",
        simulations=4,
        policy_temperature=1.0,
        provenance={"test": True},
        astar_max_nodes=1,
        astar_reason="budget_exhausted",
    )

    assert record["search"]["search_value_cost"] == solution_length
    assert record["value_target"]["upper_bound"] == solution_length


# ----------------------------------------------------------------------
# Labeling: exact first, search fallback on exhaustion
# ----------------------------------------------------------------------

def test_labeling_exact_path(base_records):
    env, state, level_id = _hard_initial_state(base_records)
    oracle = Oracle(env, max_nodes=300_000)
    recs, stats = label_states(
        env, oracle, [(state, {"level_id": level_id})], iteration=1,
        astar_max_nodes=300_000, teacher_checkpoint=None, search_adapter=None,
        search_simulations=8, search_c_puct=1.5, label_policy_temperature=1.0,
        value_norm_constant=20.0)
    assert stats.exact == 1
    assert stats.search == 0
    assert recs[0]["target_source"] == SOURCE_EXACT


def test_labeling_search_fallback_after_exhaustion(enc, tmp_path, base_records):
    ckpt = _tiny_checkpoint(tmp_path / "init.pt", enc)
    adapter, _vn = _adapter(enc, ckpt)
    env, state, level_id = _hard_initial_state(base_records)
    # max_nodes=1 forces A* to exhaust on a multi-move state.
    oracle = Oracle(env, max_nodes=1)
    recs, stats = label_states(
        env, oracle, [(state, {"level_id": level_id})], iteration=1,
        astar_max_nodes=1, teacher_checkpoint=str(ckpt), search_adapter=adapter,
        search_simulations=8, search_c_puct=1.5, label_policy_temperature=1.0,
        value_norm_constant=20.0)
    assert stats.astar_exhausted >= 1
    assert stats.search == 1
    assert recs[0]["target_source"] == SOURCE_SEARCH
    assert recs[0]["value_exact"] is False


def test_labeling_retains_cached_root_path_after_successor_exhaustion(
        base_records):
    env, state, level_id = _hard_initial_state(base_records)
    queries = []
    # This budget solves the four-move root but cannot prove every successor.
    oracle = Oracle(env, max_nodes=4, search_observer=queries.append)
    recs, stats = label_states(
        env, oracle, [(state, {"level_id": level_id})], iteration=1,
        astar_max_nodes=4, teacher_checkpoint=None, search_adapter=None,
        search_simulations=8, search_c_puct=1.5,
        label_policy_temperature=1.0, value_norm_constant=20.0)

    assert stats.exact == 0
    assert stats.exact_path == 1
    assert stats.search == 0
    assert stats.astar_exhausted == 1
    assert recs[0]["target_source"] == SOURCE_EXACT_PATH
    assert recs[0]["value_exact"] is True
    assert recs[0]["action_values_complete"] is False
    assert sum(q["query_role"] == "root" for q in queries) == 1


def test_legacy_hybrid_mode_does_not_enable_path_fallback(base_records):
    env, state, level_id = _hard_initial_state(base_records)
    oracle = Oracle(env, max_nodes=4)
    recs, stats = label_states(
        env, oracle, [(state, {"level_id": level_id})], iteration=1,
        astar_max_nodes=4, teacher_checkpoint=None, search_adapter=None,
        search_simulations=8, search_c_puct=1.5,
        label_policy_temperature=1.0, value_norm_constant=20.0,
        label_mode=LABEL_MODE_HYBRID)

    assert recs == []
    assert stats.exact_path == 0
    assert stats.skipped_other == 1


def test_search_only_mode_does_not_call_astar(
        enc, tmp_path, base_records, monkeypatch):
    ckpt = _tiny_checkpoint(tmp_path / "init.pt", enc)
    adapter, _vn = _adapter(enc, ckpt)
    env, state, level_id = _hard_initial_state(base_records)
    oracle = Oracle(env, max_nodes=300_000)

    def unexpected_analysis(_state):
        raise AssertionError("search_only must not call Oracle.analyze")

    monkeypatch.setattr(oracle, "analyze", unexpected_analysis)
    recs, stats = label_states(
        env, oracle, [(state, {"level_id": level_id})], iteration=1,
        astar_max_nodes=300_000, teacher_checkpoint=str(ckpt),
        search_adapter=adapter, search_simulations=8, search_c_puct=1.5,
        label_policy_temperature=1.0, value_norm_constant=20.0,
        label_mode=LABEL_MODE_SEARCH_ONLY)

    assert stats.astar_skipped == 1
    assert stats.astar_exhausted == 0
    assert stats.search == 1
    assert recs[0]["astar"]["termination_reason"] == "disabled_search_only"


# ----------------------------------------------------------------------
# Replay buffer
# ----------------------------------------------------------------------

def test_exact_takes_priority_over_search(enc, tmp_path, base_records):
    exact = _exact_example(base_records, iteration=1)
    # A search record for the SAME state key.
    search = dict(exact)
    search.update({
        "target_source": SOURCE_SEARCH,
        "label_kind": LABEL_SEARCH_VISIT_POLICY,
        "value_exact": False,
        "policy_exact": False,
        "optimal_actions_complete": False,
        "action_values_complete": False,
    })
    search["search"] = {"simulations": 999}

    # search added first, then exact -> exact wins.
    buf = ReplayBuffer(tmp_path / "a", max_examples=100)
    buf.add([search], iteration=1)
    buf.add([exact], iteration=2)
    assert buf.records()[0]["target_source"] == SOURCE_EXACT

    # exact added first, then search -> exact retained (never downgraded).
    buf2 = ReplayBuffer(tmp_path / "b", max_examples=100)
    buf2.add([exact], iteration=1)
    buf2.add([search], iteration=2)
    assert buf2.records()[0]["target_source"] == SOURCE_EXACT


def test_replay_orders_full_exact_above_path_above_search(base_records, tmp_path):
    full = _exact_example(base_records, iteration=1)
    path = dict(full)
    path.update({
        "target_source": SOURCE_EXACT_PATH,
        "label_kind": LABEL_EXACT_PATH_POLICY,
        "value_exact": True,
        "policy_exact": True,
        "optimal_actions_complete": False,
        "action_values_complete": False,
    })
    path["search"] = None
    search = dict(full)
    search.update({
        "target_source": SOURCE_SEARCH,
        "label_kind": LABEL_SEARCH_VISIT_POLICY,
        "value_exact": False,
        "policy_exact": False,
        "optimal_actions_complete": False,
        "action_values_complete": False,
    })
    search["search"] = {"simulations": 999}

    buf = ReplayBuffer(tmp_path / "priority", max_examples=100)
    assert buf.add([search], iteration=1)["added"] == 1
    assert buf.add([path], iteration=2)["upgraded"] == 1
    assert buf.records()[0]["target_source"] == SOURCE_EXACT_PATH
    assert buf.add([full], iteration=3)["upgraded"] == 1
    assert buf.records()[0]["target_source"] == SOURCE_EXACT
    assert buf.add([path], iteration=4)["kept_existing"] == 1


def test_dedup_across_iterations(enc, tmp_path, base_records):
    exact = _exact_example(base_records, iteration=1)
    buf = ReplayBuffer(tmp_path / "a", max_examples=100)
    s1 = buf.add([exact], iteration=1)
    s2 = buf.add([dict(exact)], iteration=2)
    assert s1["added"] == 1
    assert s2["added"] == 0 and s2["deduped"] == 1
    assert len(buf) == 1


def test_imported_replay_snapshot_is_rebased_as_historical(
        tmp_path, base_records):
    old = _exact_example(base_records, iteration=2)
    source = ReplayBuffer(tmp_path / "source", max_examples=100)
    source.add([old], iteration=2)
    snapshot = tmp_path / "committed.jsonl"
    source.write_snapshot(snapshot)

    imported = ReplayBuffer(tmp_path / "imported", max_examples=100)
    imported.load_snapshot(snapshot, rebase_as_historical=True)

    [record] = imported.records()
    assert record["generation_iteration"] == 0
    assert record["imported_generation_iteration"] == 2
    assert dedup_key(record) == dedup_key(old)


def test_replay_migrates_legacy_search_metadata_and_rejects_conflicts(
        tmp_path):
    legacy = {
        "static_level_signature": "legacy-level",
        "state_key": "legacy-state",
        "target_source": SOURCE_SEARCH,
        "generation_iteration": 1,
    }
    replay = ReplayBuffer(tmp_path / "legacy-search", max_examples=10)
    replay.add([legacy], iteration=1)

    [record] = replay.records()
    assert record["label_kind"] == LABEL_SEARCH_VISIT_POLICY
    assert record["value_exact"] is False
    assert record["policy_exact"] is False
    assert record["optimal_actions_complete"] is False
    assert record["action_values_complete"] is False

    contradictory = {
        **legacy,
        "state_key": "contradictory-state",
        "policy_exact": True,
    }
    with pytest.raises(ValueError, match="inconsistent policy_exact"):
        replay.add([contradictory], iteration=1)


def test_eviction_prefers_exact_and_hard(enc, tmp_path):
    def rec(sig, src, diff, it):
        return {
            "static_level_signature": sig, "state_key": f"k{sig}",
            "target_source": src, "generation_iteration": it,
            "remaining_blocks": diff,
            "optimal_remaining_moves": diff if src == SOURCE_EXACT else None,
            "search": {"simulations": 1, "search_value_cost": diff}
            if src == SOURCE_SEARCH else None,
            "value_target": {"raw_optimal_moves": diff},
        }
    buf = ReplayBuffer(tmp_path / "a", max_examples=2)
    buf.add([rec("s1", SOURCE_SEARCH, 1, 1)], iteration=1)   # easy search
    buf.add([rec("e1", SOURCE_EXACT, 2, 1)], iteration=1)    # exact
    buf.add([rec("e2", SOURCE_EXACT, 9, 1)], iteration=1)    # hard exact (evicts search)
    sources = {r["target_source"] for r in buf.records()}
    assert SOURCE_SEARCH not in sources
    assert len(buf) == 2


def test_replay_sampling_deterministic_and_weighted(enc, tmp_path, base_records):
    env = Environment()
    oracle = Oracle(env, max_nodes=300_000)
    exact_recs, search_recs = [], []
    for r in base_records[:30]:
        level = level_from_dict(r["level"])
        state = deserialize_state(level, r["state"])
        if env.is_terminal(state):
            continue
        a = oracle.analyze(state)
        ex = build_exact_example(a, state, level_id=r["level_id"], iteration=0,
                                 astar_max_nodes=300_000, teacher_checkpoint=None,
                                 provenance={})
        if ex is None:
            continue
        exact_recs.append(ex)
        s = dict(ex)
        s["state_key"] = ex["state_key"] + "_search"
        s.update({
            "target_source": SOURCE_SEARCH,
            "label_kind": LABEL_SEARCH_VISIT_POLICY,
            "value_exact": False,
            "policy_exact": False,
            "optimal_actions_complete": False,
            "action_values_complete": False,
        })
        s["search"] = {"simulations": 8}
        s["generation_iteration"] = 1
        search_recs.append(s)

    buf = ReplayBuffer(tmp_path / "a", max_examples=10_000)
    buf.add(exact_recs, iteration=0)
    buf.add(search_recs, iteration=1)

    kw = dict(current_iteration=1, weight_exact_historical=1.0,
              weight_exact_new=1.0, weight_search=0.0)
    s1 = buf.sample_training_set(200, seed=7, **kw)
    s2 = buf.sample_training_set(200, seed=7, **kw)
    assert [dedup_key(r) for r in s1] == [dedup_key(r) for r in s2]
    # weight_search=0 -> only exact examples drawn.
    assert all(r["target_source"] == SOURCE_EXACT for r in s1)


def test_replay_sampling_without_replacement_maximizes_diversity(
        enc, tmp_path, base_records):
    records = []
    for index in range(12):
        record = dict(_exact_example(base_records, iteration=1))
        record["state_key"] = f"unique-{index}"
        records.append(record)
    buf = ReplayBuffer(tmp_path / "diverse", max_examples=100)
    buf.add(records, iteration=1)
    kwargs = dict(
        current_iteration=1,
        weight_exact_historical=1.0,
        weight_exact_new=1.0,
        weight_search=1.0,
        with_replacement=False,
    )

    sample = buf.sample_training_set(10, seed=19, **kwargs)
    repeated = buf.sample_training_set(10, seed=19, **kwargs)

    assert [dedup_key(record) for record in sample] == [
        dedup_key(record) for record in repeated]
    assert len({dedup_key(record) for record in sample}) == 10


def test_replay_age_quotas_preserve_fresh_gradient_mass_and_diversity(
        tmp_path):
    records = []
    index = 0
    for iteration, count in ((5, 5), (4, 5), (3, 5), (2, 10), (0, 10)):
        for _ in range(count):
            records.append({
                "static_level_signature": f"level-{index}",
                "state_key": f"state-{index}",
                "target_source": SOURCE_EXACT,
                "generation_iteration": iteration,
            })
            index += 1
    replay = ReplayBuffer(tmp_path / "age-quotas", max_examples=100)
    replay.add(records, iteration=5)
    kwargs = dict(
        current_iteration=5,
        current_fraction=0.35,
        recent_fraction=0.25,
        historical_fraction=0.40,
        recent_window=2,
        weight_exact_historical=1.0,
        weight_exact_new=1.5,
        weight_search=0.5,
        seed=23,
        with_replacement=False,
    )

    sample, summary = replay.sample_training_set_with_age_quotas(100, **kwargs)
    repeated, repeated_summary = replay.sample_training_set_with_age_quotas(
        100, **kwargs)

    assert [dedup_key(record) for record in sample] == [
        dedup_key(record) for record in repeated]
    assert summary == repeated_summary
    assert summary["target_counts"] == {
        REPLAY_AGE_CURRENT: 35,
        REPLAY_AGE_RECENT: 25,
        REPLAY_AGE_HISTORICAL: 40,
    }
    assert summary["realized_counts"] == summary["target_counts"]
    assert summary["unique_counts"] == {
        REPLAY_AGE_CURRENT: 5,
        REPLAY_AGE_RECENT: 10,
        REPLAY_AGE_HISTORICAL: 20,
    }
    realized = Counter(
        replay_age_bucket(
            record, current_iteration=5, recent_window=2)
        for record in sample
    )
    assert dict(realized) == summary["realized_counts"]


def test_replay_age_quotas_redistribute_empty_bucket_deterministically(
        tmp_path):
    records = [
        {
            "static_level_signature": f"level-{index}",
            "state_key": f"state-{index}",
            "target_source": SOURCE_EXACT,
            "generation_iteration": 5 if index < 4 else 0,
        }
        for index in range(10)
    ]
    replay = ReplayBuffer(tmp_path / "age-redistribution", max_examples=100)
    replay.add(records, iteration=5)

    _sample, summary = replay.sample_training_set_with_age_quotas(
        100,
        current_iteration=5,
        current_fraction=0.35,
        recent_fraction=0.25,
        historical_fraction=0.40,
        recent_window=2,
        weight_exact_historical=1.0,
        weight_exact_new=1.5,
        weight_search=0.5,
        seed=23,
        with_replacement=False,
    )

    assert summary["available_records"][REPLAY_AGE_RECENT] == 0
    assert summary["target_counts"] == {
        REPLAY_AGE_CURRENT: 47,
        REPLAY_AGE_RECENT: 0,
        REPLAY_AGE_HISTORICAL: 53,
    }
    assert summary["realized_counts"] == summary["target_counts"]


def test_value_replay_sampling_enforces_depth_quotas_and_redistributes(
        tmp_path):
    records = []
    bucket_depths = (2, 5, 8, 12)
    bucket_sizes = (20, 20, 20, 5)
    index = 0
    for depth, count in zip(bucket_depths, bucket_sizes):
        for _ in range(count):
            records.append({
                "static_level_signature": f"level-{index % 13}",
                "state_key": f"state-{index}",
                "target_source": SOURCE_EXACT,
                "generation_iteration": index % 3,
                "value_target": {"raw_optimal_moves": depth},
                "remaining_blocks": depth,
            })
            index += 1
    replay = ReplayBuffer(tmp_path / "value", max_examples=100)
    replay.add(records, iteration=0)
    kwargs = dict(
        current_iteration=2,
        weight_exact_historical=1.0,
        weight_exact_new=1.5,
        weight_search=0.5,
        seed=17,
        depth_fractions=(0.25, 0.25, 0.25, 0.25),
    )

    sample = replay.sample_value_training_set(40, **kwargs)
    repeated = replay.sample_value_training_set(40, **kwargs)
    counts = {
        depth: sum(
            record["value_target"]["raw_optimal_moves"] == depth
            for record in sample)
        for depth in bucket_depths
    }

    assert counts == {2: 10, 5: 10, 8: 15, 12: 5}
    assert [dedup_key(record) for record in sample] == [
        dedup_key(record) for record in repeated]
    assert len({dedup_key(record) for record in sample}) == 40


def test_replay_persist_and_reload(enc, tmp_path, base_records):
    exact = _exact_example(base_records, iteration=1)
    buf = ReplayBuffer(tmp_path / "a", max_examples=100)
    buf.add([exact], iteration=1)
    buf.persist([1])
    reloaded = ReplayBuffer(tmp_path / "a").load()
    assert len(reloaded) == 1
    assert reloaded.records()[0]["state_key"] == exact["state_key"]


def test_expert_dataset_keeps_policy_and_value_weights_separate(
        enc, base_records):
    from blocksort.expert_iteration.train import ExpertDataset

    dataset = ExpertDataset(
        [base_records[0]],
        [1.0],
        value_weights=[3.0],
        encoding_config=enc,
        value_norm=ValueNormConfig(),
    )

    assert float(dataset[0]["weight"]) == 1.0
    assert float(dataset[0]["value_weight"]) == 3.0


def test_pairwise_policy_ranking_loss_abstains_on_ties_and_orders_preferences():
    from blocksort.expert_iteration.train import pairwise_policy_ranking_loss

    logits = torch.zeros((2, 4), requires_grad=True)
    utilities = torch.tensor([
        [1.0, 0.5, 1.0, 0.0],
        [0.5, 0.5, 0.0, 0.0],
    ])
    mask = torch.tensor([
        [True, True, True, False],
        [True, True, False, False],
    ])

    losses, valid, pairs = pairwise_policy_ranking_loss(
        logits, utilities, mask, margin=0.25)

    assert valid.tolist() == [True, False]
    assert pairs.tolist() == [2, 0]
    assert float(losses[0].detach()) == pytest.approx(
        torch.nn.functional.softplus(torch.tensor(0.25)).item())
    assert losses[1] == 0
    losses.sum().backward()
    assert logits.grad[0, 0] < 0
    assert logits.grad[0, 1] > 0
    assert logits.grad[0, 2] < 0
    assert torch.equal(logits.grad[1], torch.zeros(4))


def test_trace_pairwise_hinge_loss_ranks_only_the_evidenced_pair():
    from blocksort.expert_iteration.train import trace_pairwise_hinge_loss

    logits = torch.tensor(
        [[0.0, 0.0, 3.0], [0.5, 0.0, -2.0]], requires_grad=True)
    loss = trace_pairwise_hinge_loss(
        logits,
        torch.tensor([0, 0], dtype=torch.int64),
        torch.tensor([1, 1], dtype=torch.int64),
        margin=0.25,
    )

    assert loss.tolist() == pytest.approx([0.25, 0.0])
    loss.sum().backward()
    assert logits.grad[0].tolist() == pytest.approx([-1.0, 1.0, 0.0])
    assert logits.grad[1].tolist() == pytest.approx([0.0, 0.0, 0.0])


def _trace_preference_record(base_records):
    record = copy.deepcopy(next(
        row for row in base_records
        if sum(float(value) > 0 for value in row["policy_target"]) >= 2))
    record.update({
        "label_kind": "full-exact",
        "value_exact": True,
        "policy_exact": True,
        "optimal_actions_complete": True,
        "action_values_complete": True,
    })
    optimal = [index for index, value in enumerate(record["policy_target"])
               if float(value) > 0]
    preferred, competing = optimal[:2]
    record["trace_preference"] = {
        "both_actions_full_exact_oracle_optimal": True,
        "preferred_action": record["legal_actions"][preferred],
        "competing_action": record["legal_actions"][competing],
        "preferred_action_index": preferred,
        "competing_action_index": competing,
    }
    return record


def test_expert_dataset_encodes_exact_trace_preference(enc, base_records):
    from blocksort.expert_iteration.train import ExpertDataset

    item = ExpertDataset(
        [_trace_preference_record(base_records)], [1.0],
        value_weights=[0.0], encoding_config=enc,
        value_norm=ValueNormConfig())[0]

    assert bool(item["trace_pair_valid"])
    assert item["trace_preferred_index"] \
        != item["trace_competing_index"]


def test_separate_trace_ranking_records_have_no_target_loss_mass(
        enc, base_records):
    from blocksort.expert_iteration.train import train_expert

    model = PolicyValueNet(
        enc, ModelConfig(channels=4, residual_blocks=1, value_hidden_size=8))
    result = train_expert(
        model, base_records[:4], [1.0] * 4,
        encoding_config=enc,
        value_norm=ValueNormConfig(),
        epochs=1,
        batch_size=2,
        learning_rate=1e-3,
        weight_decay=0.0,
        grad_clip=1.0,
        device=torch.device("cpu"),
        seed=31,
        trace_ranking_records=[_trace_preference_record(base_records)],
        trace_ranking_weight=0.1,
        trace_ranking_margin=100.0,
    )

    history = result["history"][0]
    assert history["trace_ranking_loss"] > 0
    assert history["trace_ranking_weight_mass"] == 1.0
    assert result["examples"] == 4
    assert result["gradient_weight_mass"]["policy_total"] == 4.0
    assert result["gradient_weight_mass"]["value_total"] == 4.0
    assert result["trace_ranking"]["target_loss_weight"] == 0.0
    assert result["trace_ranking"]["value_loss_weight"] == 0.0


def test_zero_weight_trace_records_preserve_exact_training_state(
        enc, base_records):
    from blocksort.expert_iteration.train import train_expert

    torch.manual_seed(43)
    control = PolicyValueNet(
        enc, ModelConfig(channels=4, residual_blocks=1, value_hidden_size=8))
    traced = copy.deepcopy(control)
    kwargs = dict(
        encoding_config=enc,
        value_norm=ValueNormConfig(),
        epochs=2,
        batch_size=2,
        learning_rate=1e-3,
        weight_decay=0.0,
        grad_clip=1.0,
        device=torch.device("cpu"),
        seed=37,
    )
    train_expert(control, base_records[:4], [1.0] * 4, **kwargs)
    result = train_expert(
        traced, base_records[:4], [1.0] * 4,
        trace_ranking_records=[_trace_preference_record(base_records)],
        trace_ranking_weight=0.0,
        **kwargs,
    )

    for key, value in control.state_dict().items():
        assert torch.equal(value, traced.state_dict()[key]), key
    assert result["history"][0]["trace_ranking_weight_mass"] == 0.0
    assert result["trace_ranking"]["eligible_examples"] == 1


def test_expert_dataset_encodes_search_utility_only_on_exact_optimal_actions(
        enc, base_records):
    from blocksort.expert_iteration.train import ExpertDataset
    from blocksort.training.action_encoding import normalized_action_index

    record = copy.deepcopy(next(
        row for row in base_records
        if sum(float(value) > 0 for value in row["policy_target"]) >= 2))
    record.update({
        "label_kind": "full-exact",
        "value_exact": True,
        "policy_exact": True,
        "optimal_actions_complete": True,
        "action_values_complete": True,
    })
    utilities = [
        (1.0 if index == 0 else 0.5) if float(probability) > 0 else None
        for index, probability in enumerate(record["policy_target"])
    ]
    record["policy_search_utility"] = utilities

    item = ExpertDataset(
        [record], [1.0], encoding_config=enc,
        value_norm=ValueNormConfig())[0]

    labeled = [index for index, value in enumerate(utilities)
               if value is not None]
    encoded = [normalized_action_index(record["legal_actions"][index], enc)
               for index in labeled]
    assert int(item["policy_rank_mask"].sum()) == len(labeled)
    assert item["policy_rank_utility"][encoded].tolist() == pytest.approx(
        [utilities[index] for index in labeled])

    invalid = copy.deepcopy(record)
    suboptimal = next(index for index, value in enumerate(
        invalid["policy_target"]) if float(value) == 0)
    invalid["policy_search_utility"][suboptimal] = 0.0
    with pytest.raises(ValueError, match="only oracle-optimal"):
        ExpertDataset(
            [invalid], [1.0], encoding_config=enc,
            value_norm=ValueNormConfig())


def test_search_value_supervision_is_disabled_by_default(base_records):
    from blocksort.expert_iteration.train import (
        source_weights_for,
        uniform_loss_weights_for,
        value_supervision_weights_for,
    )

    exact = dict(base_records[0])
    exact["value_exact"] = True
    search = dict(base_records[1])
    search["value_exact"] = False
    search["target_source"] = SOURCE_SEARCH
    historical = dict(exact, generation_iteration=0)
    current = dict(exact, generation_iteration=3)
    exact_path = dict(
        current, label_kind=LABEL_EXACT_PATH_POLICY)

    assert uniform_loss_weights_for([exact, search]) == [1.0, 1.0]
    assert source_weights_for(
        [historical, current, exact_path, search], 3,
        weight_exact_historical=1.0,
        weight_exact_new=1.5,
        weight_search=0.5,
    ) == [1.0, 1.5, 0.75, 0.5]
    assert value_supervision_weights_for(
        [exact, search], [2.0, 0.5]) == [2.0, 0.0]
    assert value_supervision_weights_for(
        [exact, search], [2.0, 0.5],
        search_value_loss_weight=0.2) == pytest.approx([2.0, 0.1])


def test_expert_dataset_allows_policy_only_search_batch(enc, base_records):
    from blocksort.expert_iteration.train import ExpertDataset

    search = copy.deepcopy(base_records[0])
    search["value_exact"] = False
    search["target_source"] = SOURCE_SEARCH
    search["value_target"]["raw_optimal_moves"] = 3.75
    dataset = ExpertDataset(
        [search],
        [1.0],
        value_weights=[0.0],
        encoding_config=enc,
        value_norm=ValueNormConfig(),
    )

    assert float(dataset[0]["weight"]) == 1.0
    assert float(dataset[0]["value_weight"]) == 0.0
    assert float(dataset[0]["value_target"]) == pytest.approx(
        ValueNormConfig().normalize(3.75))


def test_train_expert_migrates_legacy_search_before_value_weighting(
        enc, base_records):
    from blocksort.expert_iteration.train import train_expert

    search = dict(base_records[0])
    search["target_source"] = SOURCE_SEARCH
    search.pop("value_exact", None)
    assert "value_exact" not in search
    model = PolicyValueNet(
        enc, ModelConfig(channels=4, residual_blocks=1, value_hidden_size=8))

    result = train_expert(
        model,
        [search],
        [1.0],
        encoding_config=enc,
        value_norm=ValueNormConfig(),
        epochs=1,
        batch_size=1,
        learning_rate=1e-3,
        weight_decay=0.0,
        grad_clip=1.0,
        device=torch.device("cpu"),
        seed=7,
    )

    assert result["history"][0]["value_loss"] == 0.0
    assert result["value_supervision"]["exact_examples"] == 0
    assert result["value_supervision"]["search_estimate_examples"] == 1
    assert result["value_supervision"]["search_estimate_weight_mass"] == 0.0
    # Training canonicalizes a copy rather than mutating external records.
    assert "value_exact" not in search
    assert result["gradient_weight_mass"]["policy_by_source"] == {
        SOURCE_SEARCH: 1.0,
    }
    assert result["gradient_weight_mass"]["value_by_source"] == {
        SOURCE_SEARCH: 0.0,
    }
    assert result["loss_weighting_policy"] == \
        "uniform_after_weighted_replay_sampling_v1"


def test_train_expert_reports_optional_search_utility_ranking(
        enc, base_records):
    from blocksort.expert_iteration.train import (
        configure_trainable_part,
        train_expert,
    )

    record = copy.deepcopy(next(
        row for row in base_records
        if sum(float(value) > 0 for value in row["policy_target"]) >= 2))
    record.update({
        "label_kind": "full-exact",
        "value_exact": True,
        "policy_exact": True,
        "optimal_actions_complete": True,
        "action_values_complete": True,
    })
    optimal = [index for index, value in enumerate(record["policy_target"])
               if float(value) > 0]
    utilities = [None] * len(record["legal_actions"])
    for rank, index in enumerate(optimal):
        utilities[index] = 1.0 - rank / max(1, len(optimal))
    record["policy_search_utility"] = utilities
    model = PolicyValueNet(
        enc, ModelConfig(channels=4, residual_blocks=1, value_hidden_size=8))
    configure_trainable_part(model, "policy_head")

    result = train_expert(
        model, [record], [1.0],
        encoding_config=enc,
        value_norm=ValueNormConfig(),
        epochs=1,
        batch_size=1,
        learning_rate=1e-3,
        weight_decay=0.0,
        grad_clip=1.0,
        device=torch.device("cpu"),
        seed=19,
        policy_ranking_weight=0.1,
        policy_ranking_margin=0.25,
    )

    assert result["history"][0]["policy_ranking_loss"] > 0
    assert result["history"][0]["policy_ranking_weight_mass"] == 1.0
    assert result["policy_ranking"]["eligible_examples"] == 1
    assert result["policy_ranking"]["preference_pairs"] >= 1
    assert result["policy_ranking"]["weight"] == 0.1


def test_expert_loss_history_is_invariant_to_partial_batch_partitioning(
        enc, base_records):
    from blocksort.expert_iteration.train import train_expert

    class ConstantModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.bias = torch.nn.Parameter(torch.tensor(0.0))

        def forward(self, board, global_features):
            batch_size = board.shape[0]
            logits = self.bias.expand(batch_size, enc.action_space_size)
            value = self.bias.expand(batch_size)
            return logits, value

    records = base_records[:5]
    policy_weights = [1.0, 2.0, 3.0, 5.0, 11.0]
    value_weights = [7.0, 1.0, 4.0, 2.0, 9.0]

    def run(batch_size):
        return train_expert(
            ConstantModel(), records, policy_weights,
            value_weights=value_weights,
            encoding_config=enc,
            value_norm=ValueNormConfig(),
            epochs=1,
            batch_size=batch_size,
            learning_rate=0.0,
            weight_decay=0.0,
            grad_clip=1.0,
            device=torch.device("cpu"),
            seed=17,
            policy_loss_weight=0.7,
            value_loss_weight=1.3,
        )["history"][0]

    full = run(len(records))
    partial = run(2)

    for field in ("loss", "policy_loss", "value_loss"):
        assert partial[field] == pytest.approx(
            full[field], rel=1e-7, abs=1e-7)


def test_expert_training_reports_incumbent_anchor_loss(enc, base_records):
    from blocksort.expert_iteration.train import train_expert

    model_config = ModelConfig(
        channels=4, residual_blocks=1, value_hidden_size=8)
    model = PolicyValueNet(enc, model_config)
    anchor = PolicyValueNet(enc, model_config)
    anchor.load_state_dict(model.state_dict())
    records = base_records[:8]
    result = train_expert(
        model,
        records,
        [1.0] * len(records),
        value_weights=[1.0] * len(records),
        value_anchor_model=anchor,
        value_anchor_weight=0.25,
        encoding_config=enc,
        value_norm=ValueNormConfig(),
        epochs=1,
        batch_size=4,
        learning_rate=1e-3,
        weight_decay=0.0,
        grad_clip=1.0,
        device=torch.device("cpu"),
        seed=7,
    )

    assert result["history"][0]["value_anchor_loss"] > 0
    assert all(
        not parameter.requires_grad
        for parameter in anchor.parameters()
    )


def test_policy_head_only_real_update_preserves_all_other_state(
        enc, base_records):
    from blocksort.expert_iteration.train import (
        configure_trainable_part,
        train_expert,
    )

    model = PolicyValueNet(
        enc, ModelConfig(channels=4, residual_blocks=1, value_hidden_size=8))
    configure_trainable_part(model, "policy_head")
    before = copy.deepcopy(model.state_dict())
    records = base_records[:8]

    train_expert(
        model,
        records,
        [1.0] * len(records),
        encoding_config=enc,
        value_norm=ValueNormConfig(),
        epochs=1,
        batch_size=4,
        learning_rate=1e-3,
        weight_decay=0.0,
        grad_clip=1.0,
        device=torch.device("cpu"),
        seed=13,
    )

    after = model.state_dict()
    changed = {
        name for name in before
        if not torch.equal(before[name], after[name])
    }
    assert changed
    assert all(name.startswith("policy_conv.") for name in changed)


def test_policy_trunk_real_update_freezes_value_head_parameters(
        enc, base_records):
    from blocksort.expert_iteration.train import (
        configure_trainable_part,
        train_expert,
    )

    model = PolicyValueNet(
        enc, ModelConfig(channels=4, residual_blocks=1, value_hidden_size=8))
    anchor = copy.deepcopy(model)
    configure_trainable_part(model, "policy_trunk")
    before = copy.deepcopy(model.state_dict())

    result = train_expert(
        model,
        base_records[:8],
        [1.0] * 8,
        encoding_config=enc,
        value_norm=ValueNormConfig(),
        epochs=1,
        batch_size=4,
        learning_rate=1e-3,
        weight_decay=0.0,
        grad_clip=1.0,
        device=torch.device("cpu"),
        seed=23,
        value_anchor_model=anchor,
        value_anchor_weight=1.0,
    )

    after = model.state_dict()
    changed = {
        name for name in before
        if not torch.equal(before[name], after[name])
    }
    assert any(name.startswith("stem.") for name in changed)
    assert any(name.startswith("trunk.") for name in changed)
    assert any(name.startswith("policy_conv.") for name in changed)
    assert not any(name.startswith("value_conv.") for name in changed)
    assert not any(name.startswith("value_mlp.") for name in changed)
    assert result["history"][0]["value_anchor_loss"] > 0


def test_policy_adapter_real_update_preserves_shared_and_value_state(
        enc, base_records):
    from blocksort.expert_iteration.train import (
        configure_trainable_part,
        train_expert,
    )

    model = PolicyValueNet(
        enc, ModelConfig(
            channels=4, residual_blocks=1, value_hidden_size=8,
            policy_adapter_blocks=1))
    configure_trainable_part(model, "policy_adapter")
    before = copy.deepcopy(model.state_dict())

    train_expert(
        model,
        base_records[:8],
        [1.0] * 8,
        encoding_config=enc,
        value_norm=ValueNormConfig(),
        epochs=1,
        batch_size=4,
        learning_rate=1e-3,
        weight_decay=0.0,
        grad_clip=1.0,
        device=torch.device("cpu"),
        seed=29,
    )

    after = model.state_dict()
    changed = {
        name for name in before
        if not torch.equal(before[name], after[name])
    }
    assert any(name.startswith("policy_adapter.") for name in changed)
    assert any(name.startswith("policy_conv.") for name in changed)
    assert all(
        name.startswith(("policy_adapter.", "policy_conv."))
        for name in changed)
    assert not any(name.startswith("stem.") for name in changed)
    assert not any(name.startswith("trunk.") for name in changed)
    assert not any(name.startswith("value_") for name in changed)


def test_policy_anchor_is_finite_and_scoped_to_historical_records(
        enc, base_records):
    from blocksort.expert_iteration.train import train_expert

    model_config = ModelConfig(
        channels=4, residual_blocks=1, value_hidden_size=8)
    model = PolicyValueNet(enc, model_config)
    anchor = PolicyValueNet(enc, model_config)
    anchor.load_state_dict(model.state_dict())
    records = [dict(record) for record in base_records[:8]]
    for index, record in enumerate(records):
        record["generation_iteration"] = 0 if index < 4 else 1

    result = train_expert(
        model,
        records,
        [1.0] * len(records),
        encoding_config=enc,
        value_norm=ValueNormConfig(),
        epochs=1,
        batch_size=2,
        learning_rate=1e-3,
        weight_decay=0.0,
        grad_clip=1.0,
        device=torch.device("cpu"),
        seed=7,
        policy_anchor_model=anchor,
        policy_anchor_weight=0.25,
        policy_anchor_before_iteration=1,
    )

    history = result["history"][0]
    assert history["policy_anchor_loss"] >= 0
    assert torch.isfinite(torch.tensor(history["policy_anchor_loss"]))
    assert history["policy_anchor_weight_mass"] == pytest.approx(4.0)
    assert result["policy_anchor"] == {
        "weight": 0.25,
        "before_iteration": 1,
        "eligible_examples": 4,
        "scope": "historical_records_only",
        "direction": "kl_incumbent_to_candidate_on_legal_actions",
    }
    assert all(not parameter.requires_grad for parameter in anchor.parameters())


def test_policy_anchor_can_use_separate_records_without_target_loss(
        enc, base_records):
    from blocksort.expert_iteration.train import train_expert

    model_config = ModelConfig(
        channels=4, residual_blocks=1, value_hidden_size=8)
    model = PolicyValueNet(enc, model_config)
    anchor = PolicyValueNet(enc, model_config)
    anchor.load_state_dict(model.state_dict())
    result = train_expert(
        model,
        base_records[:8],
        [1.0] * 8,
        encoding_config=enc,
        value_norm=ValueNormConfig(),
        epochs=1,
        batch_size=2,
        learning_rate=1e-3,
        weight_decay=0.0,
        grad_clip=1.0,
        device=torch.device("cpu"),
        seed=17,
        policy_anchor_model=anchor,
        policy_anchor_weight=0.25,
        policy_anchor_records=base_records[8:12],
    )

    history = result["history"][0]
    assert history["policy_anchor_loss"] >= 0
    assert history["policy_anchor_weight_mass"] == pytest.approx(4.0)
    assert result["gradient_weight_mass"]["policy_total"] == pytest.approx(8.0)
    assert result["policy_anchor"] == {
        "weight": 0.25,
        "before_iteration": None,
        "eligible_examples": 4,
        "scope": "separate_anchor_records",
        "direction": "kl_incumbent_to_candidate_on_legal_actions",
    }


def test_zero_weight_separate_anchor_preserves_control_update(
        enc, base_records):
    from blocksort.expert_iteration.train import train_expert

    model_config = ModelConfig(
        channels=4, residual_blocks=1, value_hidden_size=8)
    control = PolicyValueNet(enc, model_config)
    experimental = copy.deepcopy(control)
    anchor = copy.deepcopy(control)
    kwargs = dict(
        encoding_config=enc,
        value_norm=ValueNormConfig(),
        epochs=1,
        batch_size=2,
        learning_rate=1e-3,
        weight_decay=0.0,
        grad_clip=1.0,
        device=torch.device("cpu"),
        seed=23,
    )
    train_expert(control, base_records[:8], [1.0] * 8, **kwargs)
    train_expert(
        experimental,
        base_records[:8],
        [1.0] * 8,
        policy_anchor_model=anchor,
        policy_anchor_weight=0.0,
        policy_anchor_records=base_records[8:12],
        **kwargs,
    )

    assert all(
        torch.equal(control.state_dict()[name], experimental.state_dict()[name])
        for name in control.state_dict())


# ----------------------------------------------------------------------
# Promotion decision
# ----------------------------------------------------------------------

def test_real_precomputed_counterexample_drops_unknown_regret_magnitude(
        monkeypatch, base_records):
    env = Environment()
    record = next(
        row for row in base_records
        if any(regret is not None and regret > 1
               for regret in row.get("action_regrets", [])))
    state = deserialize_state(level_from_dict(record["level"]), record["state"])
    action_data, true_regret = next(
        (action, regret)
        for action, regret in zip(
            record["legal_actions"], record["action_regrets"])
        if regret is not None and regret > 1)
    selected = {_state_identity(env, state): action_data}
    live = _evaluate_selected_actions(
        monkeypatch, env, [state], selected,
        Oracle(env, max_nodes=250_000), precomputed=None)
    entry = _precomputed_entry(
        env, state, record["optimal_remaining_moves"],
        record["optimal_actions"])
    cached = _evaluate_selected_actions(
        monkeypatch, env, [state], selected,
        Oracle(env, max_nodes=250_000),
        precomputed={_state_identity(env, state): entry})

    assert true_regret == 2
    assert live["raw_policy_mean_regret"] == true_regret
    assert cached["raw_policy_optimal_acc"] is None
    assert cached["raw_policy_mean_regret"] is None


def test_precomputed_nonoptimal_action_does_not_invent_regret_one(
        monkeypatch, base_records):
    env = Environment()
    record = base_records[0]
    state = deserialize_state(level_from_dict(record["level"]), record["state"])
    legal = env.legal_actions(state)
    selected_action = legal[0]
    selected = {_state_identity(env, state):
                serialize_action(state, selected_action)}
    child = env.apply_action(state, selected_action)
    oracle = _MappedValueOracle(env, {
        _state_identity(env, state):
            ValueResult(value=3, exact=True, solvable=True),
        _state_identity(env, child):
            ValueResult(value=5, exact=True, solvable=True),
    })

    live = _evaluate_selected_actions(
        monkeypatch, env, [state], selected, oracle, precomputed=None)
    entry = _precomputed_entry(
        env, state, 3, [serialize_action(state, legal[1])],
        classification_complete=False)
    cached = _evaluate_selected_actions(
        monkeypatch, env, [state], selected, oracle,
        precomputed={_state_identity(env, state): entry})

    assert live["raw_policy_mean_regret"] == 3
    assert cached["raw_policy_optimal_acc"] is None
    assert cached["raw_policy_mean_regret"] is None
    assert cached["raw_policy_optimal_classification_count"] == 0
    assert cached["raw_policy_exact_regret_count"] == 0


def test_legacy_precomputed_labels_preserve_classification_only(
        monkeypatch, base_records):
    env = Environment()
    record = base_records[0]
    state = deserialize_state(level_from_dict(record["level"]), record["state"])
    legal = env.legal_actions(state)
    optimal = serialize_action(state, legal[0])
    selected = {_state_identity(env, state): optimal}
    entry = _precomputed_entry(env, state, 3, [optimal])
    report = _evaluate_selected_actions(
        monkeypatch, env, [state], selected, _MappedValueOracle(env, {}),
        precomputed={_state_identity(env, state): entry})

    assert "action_regrets" not in entry
    assert report["raw_policy_optimal_acc"] == 1.0
    assert report["raw_policy_optimal_classification_count"] == 1
    assert report["raw_policy_exact_regret_count"] == 0
    assert report["raw_policy_mean_regret"] is None

    nonoptimal = serialize_action(state, legal[1])
    unknown = _evaluate_selected_actions(
        monkeypatch, env, [state], {_state_identity(env, state): nonoptimal},
        _MappedValueOracle(env, {}),
        precomputed={_state_identity(env, state): entry})
    assert unknown["raw_policy_optimal_acc"] is None
    assert unknown["raw_policy_optimal_classification_count"] == 0


def test_incomplete_and_complete_precomputed_classification(monkeypatch,
                                                            base_records):
    env = Environment()
    record = base_records[0]
    state = deserialize_state(level_from_dict(record["level"]), record["state"])
    legal = env.legal_actions(state)
    optimal = serialize_action(state, legal[0])
    other = serialize_action(state, legal[1])
    identity = _state_identity(env, state)

    incomplete = _precomputed_entry(
        env, state, 3, [optimal], classification_complete=False)
    listed = _evaluate_selected_actions(
        monkeypatch, env, [state], {identity: optimal},
        _MappedValueOracle(env, {}), precomputed={identity: incomplete})
    unlisted = _evaluate_selected_actions(
        monkeypatch, env, [state], {identity: other},
        _MappedValueOracle(env, {}), precomputed={identity: incomplete})
    assert listed["raw_policy_optimal_acc"] == 1.0
    assert listed["raw_policy_optimal_classification_count"] == 1
    assert unlisted["raw_policy_optimal_acc"] is None
    assert unlisted["raw_policy_optimal_classification_count"] == 0

    complete = _precomputed_entry(
        env, state, 3, [optimal], classification_complete=True)
    listed = _evaluate_selected_actions(
        monkeypatch, env, [state], {identity: optimal},
        _MappedValueOracle(env, {}), precomputed={identity: complete})
    unlisted = _evaluate_selected_actions(
        monkeypatch, env, [state], {identity: other},
        _MappedValueOracle(env, {}), precomputed={identity: complete})
    assert listed["raw_policy_optimal_acc"] == 1.0
    assert unlisted["raw_policy_optimal_acc"] == 0.0
    assert unlisted["raw_policy_optimal_classification_count"] == 1


def test_classification_completeness_aggregation(monkeypatch, base_records):
    env = Environment()
    states = []
    seen = set()
    for record in base_records:
        state = deserialize_state(
            level_from_dict(record["level"]), record["state"])
        identity = _state_identity(env, state)
        if env.is_terminal(state) or identity in seen:
            continue
        seen.add(identity)
        states.append(state)
        if len(states) == 4:
            break

    selected = {}
    entries = {}
    for state in states:
        identity = _state_identity(env, state)
        action = serialize_action(state, env.legal_actions(state)[0])
        selected[identity] = action
    entries[_state_identity(env, states[0])] = _precomputed_entry(
        env, states[0], 2, [selected[_state_identity(env, states[0])]],
        classification_complete=False)
    entries[_state_identity(env, states[1])] = _precomputed_entry(
        env, states[1], 2, [], classification_complete=True)
    entries[_state_identity(env, states[2])] = _precomputed_entry(
        env, states[2], 2, [], classification_complete=False)
    entries[_state_identity(env, states[3])] = _precomputed_entry(
        env, states[3], 2, [])

    report = _evaluate_selected_actions(
        monkeypatch, env, states, selected, _MappedValueOracle(env, {}),
        precomputed=entries)
    assert report["raw_policy_optimal_classification_count"] == 2
    assert report["raw_policy_unknown_classification_count"] == 2
    assert report["raw_policy_optimal_acc"] == pytest.approx(0.5)
    assert report["raw_policy_optimal_classification_coverage"] \
        == pytest.approx(0.5)
    assert report["raw_policy_confirmed_optimal_count"] == 1
    assert report["raw_policy_confirmed_optimal_rate"] == pytest.approx(0.25)
    assert report["raw_policy_exact_regret_count"] == 0
    assert report["raw_policy_mean_regret"] is None


def test_real_benchmark_label_serializes_incomplete_classification(
        tmp_path, monkeypatch, base_records):
    env = Environment()
    record = base_records[0]
    level = level_from_dict(record["level"])
    state = env.initial_state(level)
    oracle = Oracle(env, max_nodes=5)
    analysis = oracle.analyze(state)
    unresolved = [a for a in analysis.actions if not a.successor_exact]
    assert analysis.exact and analysis.solvable
    assert analysis.all_successors_exact is False
    assert len(analysis.actions) == 16
    # Once the first successor exhausts, remaining successor searches are
    # intentionally skipped because a complete classification is already
    # impossible and the evaluation must treat the selected action as unknown.
    assert 0 < len(unresolved) < len(analysis.actions)

    labels = benchmark_mod.ensure_benchmark_labels(
        tmp_path, env, {"g": [level]}, Oracle(env, max_nodes=5))
    entry = labels["g"][0]
    assert entry["classification_complete"] is False
    selected_action = unresolved[0].serialized
    report = _evaluate_selected_actions(
        monkeypatch, env, [state], {_state_identity(env, state): selected_action},
        _MappedValueOracle(env, {}),
        precomputed=_precomputed_lookup(labels, "g"))
    assert report["raw_policy_optimal_classification_count"] == 0
    assert report["raw_policy_optimal_acc"] is None


def test_mixed_classification_and_exact_regret_availability(
        monkeypatch, base_records):
    env = Environment()
    states = []
    seen = set()
    for record in base_records:
        state = deserialize_state(
            level_from_dict(record["level"]), record["state"])
        identity = _state_identity(env, state)
        if env.is_terminal(state) or identity in seen:
            continue
        seen.add(identity)
        states.append(state)
        if len(states) == 4:
            break
    assert len(states) == 4

    selected = {}
    chosen = []
    for state in states:
        action = env.legal_actions(state)[0]
        chosen.append(action)
        selected[_state_identity(env, state)] = serialize_action(state, action)

    precomputed = {
        _state_identity(env, states[1]): _precomputed_entry(
            env, states[1], 2, [], classification_complete=True),
    }
    optimal_child = env.apply_action(states[0], chosen[0])
    regret_child = env.apply_action(states[2], chosen[2])
    oracle = _MappedValueOracle(env, {
        _state_identity(env, states[0]):
            ValueResult(value=2, exact=True, solvable=True),
        _state_identity(env, optimal_child):
            ValueResult(value=1, exact=True, solvable=True),
        _state_identity(env, states[2]):
            ValueResult(value=3, exact=True, solvable=True),
        _state_identity(env, regret_child):
            ValueResult(value=5, exact=True, solvable=True),
    })
    report = _evaluate_selected_actions(
        monkeypatch, env, states, selected, oracle, precomputed=precomputed)

    assert report["raw_policy_optimal_classification_count"] == 3
    assert report["raw_policy_exact_regret_count"] == 2
    assert report["raw_policy_unknown_classification_count"] == 1
    assert report["raw_policy_optimal_acc"] == pytest.approx(1 / 3)
    assert report["raw_policy_mean_regret"] == pytest.approx(1.5)
    assert report["raw_policy_oracle_regret_coverage"] == pytest.approx(0.5)
    grouped = report["grouped_by_board"]["6x6"]
    assert grouped["raw_policy_optimal_classification_count"] == 3
    assert grouped["raw_policy_exact_regret_count"] == 2
    assert grouped["raw_policy_optimal_acc"] == pytest.approx(1 / 3)
    assert grouped["raw_policy_mean_regret"] == pytest.approx(1.5)


def test_unknown_regret_is_excluded_from_optimal_accuracy_denominator():
    agg = _Agg(
        n=4,
        raw_optimal=1,
        raw_regret_sum=2.0,
        raw_regret_n=2,
        search_optimal=1,
        search_regret_sum=2.0,
        search_regret_n=2,
    )
    report = agg.summary()

    assert report["raw_policy_optimal_acc"] == pytest.approx(0.5)
    assert report["search_optimal_acc"] == pytest.approx(0.5)
    assert report["raw_policy_known_regret_count"] == 2
    assert report["raw_policy_unknown_regret_count"] == 2
    assert report["raw_policy_optimal_accuracy_known"] == pytest.approx(0.5)
    assert report["raw_policy_oracle_regret_coverage"] == pytest.approx(0.5)
    assert report["raw_policy_confirmed_optimal_rate"] == pytest.approx(0.25)
    assert report["raw_policy_confirmed_optimal_count"] == 1
    assert report["raw_policy_mean_regret_known"] == pytest.approx(1.0)

    assert report["search_known_regret_count"] == 2
    assert report["search_unknown_regret_count"] == 2
    assert report["search_optimal_accuracy_known"] == pytest.approx(0.5)
    assert report["search_oracle_regret_coverage"] == pytest.approx(0.5)
    assert report["search_confirmed_optimal_rate"] == pytest.approx(0.25)
    assert report["search_confirmed_optimal_count"] == 1
    assert report["search_mean_regret_known"] == pytest.approx(1.0)


def test_promotion_accuracy_is_independent_of_oracle_coverage():
    incumbent = _Agg(n=4, search_optimal=1, search_regret_n=2).summary()
    candidate = _Agg(n=4, search_optimal=2, search_regret_n=4).summary()
    incumbent_report = {"budgets": {"8": incumbent}}
    candidate_report = {"budgets": {"8": candidate}}

    assert incumbent["search_oracle_regret_coverage"] == pytest.approx(0.5)
    assert candidate["search_oracle_regret_coverage"] == pytest.approx(1.0)
    assert promotion_score(incumbent_report, metric="search_optimal_acc", budget=8) \
        == pytest.approx(0.5)
    assert promotion_score(candidate_report, metric="search_optimal_acc", budget=8) \
        == pytest.approx(0.5)


def _coverage_report(*, total, confirmed, known, score=None, budget=8):
    fixed = confirmed / total if total else None
    conditional = confirmed / known if known else None
    return {
        "states": total,
        "total_evaluated_count": total,
        "budgets": {str(budget): {
            "total_evaluated_count": total,
            "search_optimal_classification_count": known,
            "search_optimal_classification_coverage":
                (known / total if total else None),
            "search_confirmed_optimal_count": confirmed,
            "search_optimal_acc": conditional,
            "search_confirmed_optimal_rate":
                fixed if score is None else score,
        }},
    }


def test_fixed_denominator_promotion_with_unequal_coverage():
    incumbent = _coverage_report(total=24, confirmed=18, known=24)
    candidate = _coverage_report(total=24, confirmed=20, known=23)
    evidence = validate_promotion_evidence(
        incumbent, candidate, metric="search_confirmed_optimal_rate", budget=8)
    assert evidence.incumbent_score == pytest.approx(18 / 24)
    assert evidence.candidate_score == pytest.approx(20 / 24)
    assert evidence.incumbent_known_count == 24
    assert evidence.candidate_known_count == 23


def test_fixed_denominator_rejects_misleading_conditional_accuracy():
    incumbent = _coverage_report(total=24, confirmed=18, known=24)
    candidate = _coverage_report(total=24, confirmed=15, known=16)
    assert candidate["budgets"]["8"]["search_optimal_acc"] == pytest.approx(0.9375)
    evidence = validate_promotion_evidence(
        incumbent, candidate, metric="search_confirmed_optimal_rate", budget=8)
    assert evidence.candidate_score == pytest.approx(0.625)
    assert evidence.candidate_score < evidence.incumbent_score


def test_full_coverage_fixed_rate_equals_conditional_accuracy():
    report = _coverage_report(total=24, confirmed=18, known=24)
    budget = report["budgets"]["8"]
    assert budget["search_confirmed_optimal_rate"] == \
        budget["search_optimal_acc"] == pytest.approx(0.75)
    evidence = validate_promotion_evidence(
        report, report, metric="search_optimal_acc", budget=8)
    assert evidence.incumbent_score == pytest.approx(0.75)


def test_coverage_safe_promotion_accepts_real_zero():
    report = _coverage_report(total=24, confirmed=0, known=16)
    evidence = validate_promotion_evidence(
        report, report, metric="search_confirmed_optimal_rate", budget=8)
    assert evidence.incumbent_score == evidence.candidate_score == 0.0


def test_promotion_evidence_rejects_empty_or_mismatched_evaluation():
    empty = _coverage_report(total=0, confirmed=0, known=0)
    full = _coverage_report(total=24, confirmed=18, known=24)
    other = _coverage_report(total=23, confirmed=18, known=23)
    with pytest.raises(ValueError, match="empty evaluation set"):
        validate_promotion_evidence(
            empty, empty, metric="search_confirmed_optimal_rate", budget=8)
    with pytest.raises(ValueError, match="mismatched total evaluated counts"):
        validate_promotion_evidence(
            full, other, metric="search_confirmed_optimal_rate", budget=8)


def test_promotion_evidence_rejects_missing_count_and_invalid_scores():
    valid = _coverage_report(total=24, confirmed=18, known=24)
    missing = json.loads(json.dumps(valid))
    del missing["budgets"]["8"]["search_confirmed_optimal_count"]
    with pytest.raises(ValueError, match="missing count field"):
        validate_promotion_evidence(
            valid, missing, metric="search_confirmed_optimal_rate", budget=8)
    for invalid in (None, float("nan"), float("inf"), "0.5"):
        bad = _coverage_report(total=24, confirmed=18, known=24)
        bad["budgets"]["8"]["search_confirmed_optimal_rate"] = invalid
        with pytest.raises(ValueError, match="unavailable|non-finite|non-numeric"):
            validate_promotion_evidence(
                valid, bad, metric="search_confirmed_optimal_rate", budget=8)


def test_conditional_promotion_requires_full_coverage():
    full = _coverage_report(total=24, confirmed=18, known=24)
    partial = _coverage_report(total=24, confirmed=15, known=16)
    with pytest.raises(ValueError, match="conditional on coverage"):
        validate_promotion_evidence(
            full, partial, metric="search_optimal_acc", budget=8)
    evidence = validate_promotion_evidence(
        full, full, metric="search_optimal_acc", budget=8)
    assert evidence.incumbent_score == evidence.candidate_score == 0.75


@pytest.mark.parametrize(
    ("candidate_confirmed", "margin", "expected"),
    [(19, 0.0, True), (18, 0.0, False), (19, 1 / 24, False)],
)
def test_fixed_rate_promotion_preserves_margin_and_tie_semantics(
        candidate_confirmed, margin, expected):
    incumbent = _coverage_report(total=24, confirmed=18, known=24)
    candidate = _coverage_report(
        total=24, confirmed=candidate_confirmed, known=24)
    evidence = validate_promotion_evidence(
        incumbent, candidate, metric="search_confirmed_optimal_rate", budget=8)
    assert (
        evidence.candidate_score > evidence.incumbent_score + margin
    ) is expected


def test_zero_oracle_coverage_reports_unavailable_accuracy():
    report = _Agg(n=4).summary()
    assert report["raw_policy_optimal_accuracy_known"] is None
    assert report["search_optimal_accuracy_known"] is None
    assert report["raw_policy_oracle_regret_coverage"] == 0.0
    assert report["search_oracle_regret_coverage"] == 0.0
    with pytest.raises(ValueError, match="unavailable metric 'search_optimal_acc'"):
        promotion_score(
            {"budgets": {"8": report}},
            metric="search_optimal_acc",
            budget=8,
        )


def _report_with_score(score, budget=4):
    return {"budgets": {str(budget): {"search_optimal_acc": score}},
            "raw_policy_optimal_acc": score}


def test_promotion_score_extraction():
    rep = _report_with_score(0.75, budget=4)
    assert promotion_score(rep, metric="search_optimal_acc", budget=4) == 0.75
    assert promotion_score(rep, metric="raw_policy_optimal_acc", budget=4) == 0.75


def test_expert_iteration_config_includes_missing_promotion_budget():
    cfg = ExpertIterationConfig(
        eval_budgets=(1, 100, 400),
        promotion_budget=32,
        promotion_metric="search_optimal_acc",
    )
    report = {
        "budgets": {
            str(budget): {
                "search_optimal_acc": 1.0 if budget == 32 else 0.0,
            }
            for budget in cfg.eval_budgets
        },
    }

    assert cfg.eval_budgets == (1, 32, 100, 400)
    assert promotion_score(
        report, metric=cfg.promotion_metric, budget=cfg.promotion_budget
    ) == 1.0
    assert cfg.to_dict()["eval_budgets"] == [1, 32, 100, 400]


@pytest.mark.parametrize("epochs", [0, -1])
def test_expert_epochs_zero_or_negative_fail_before_run_initialization(
    tmp_path, epochs,
):
    output = tmp_path / "never-created"
    with pytest.raises(ValueError, match=r"epochs.*greater than or equal to 1"):
        ExpertIterationConfig(output_dir=str(output), epochs=epochs)
    assert not output.exists()


def test_expert_epochs_one_remains_valid():
    assert ExpertIterationConfig(epochs=1).epochs == 1


def test_expert_iteration_cli_config_includes_missing_promotion_budget():
    args = build_parser().parse_args([
        "--initial-checkpoint", "initial.pt",
        "--base-dataset", "base.jsonl",
        "--eval-budgets", "1", "100", "400",
        "--promotion-budget", "32",
        "--search-value-loss-weight", "0.25",
        "--policy-anchor-weight", "0.5",
    ])
    cfg = config_from_args(args)

    assert cfg.eval_budgets == (1, 32, 100, 400)
    assert cfg.search_value_loss_weight == pytest.approx(0.25)
    assert cfg.policy_anchor_weight == pytest.approx(0.5)
    assert cfg.exact_path_policy_confidence == pytest.approx(0.5)


@pytest.mark.parametrize("weight", [-0.1, float("nan"), float("inf")])
def test_expert_iteration_rejects_invalid_search_value_weight(weight):
    with pytest.raises(ValueError, match="search_value_loss_weight"):
        ExpertIterationConfig(search_value_loss_weight=weight)


@pytest.mark.parametrize("weight", [-0.1, float("nan"), float("inf")])
def test_expert_iteration_rejects_invalid_policy_anchor_weight(weight):
    with pytest.raises(ValueError, match="policy_anchor_weight"):
        ExpertIterationConfig(policy_anchor_weight=weight)


def test_expert_iteration_rejects_invalid_or_zero_source_weights():
    with pytest.raises(ValueError, match="weight_exact_new"):
        ExpertIterationConfig(weight_exact_new=-0.1)
    with pytest.raises(ValueError, match="at least one source weight"):
        ExpertIterationConfig(
            weight_exact_historical=0.0,
            weight_exact_new=0.0,
            weight_search=0.0,
        )


@pytest.mark.parametrize("confidence", [
    -0.1, 1.1, float("nan"), float("inf"), True,
])
def test_expert_iteration_rejects_invalid_exact_path_confidence(confidence):
    with pytest.raises(ValueError, match="exact_path_policy_confidence"):
        ExpertIterationConfig(exact_path_policy_confidence=confidence)


def test_expert_iteration_default_promotion_metric_is_coverage_safe():
    assert ExpertIterationConfig().promotion_metric == \
        "search_confirmed_optimal_rate"


def test_expert_iteration_rejects_unsupported_budget_sweep_promotion():
    with pytest.raises(ValueError, match="co-training only"):
        ExpertIterationConfig(
            promotion_metric="weighted_budget_sweep_confirmed_optimal_rate")


@pytest.mark.parametrize("margin", [-0.1, float("nan"), float("inf")])
def test_expert_iteration_rejects_invalid_promotion_margin(margin):
    with pytest.raises(ValueError, match="promotion_margin"):
        ExpertIterationConfig(promotion_margin=margin)
    args = build_parser().parse_args([
        "--initial-checkpoint", "initial.pt",
        "--base-dataset", "base.jsonl",
    ])
    assert config_from_args(args).promotion_metric == \
        "search_confirmed_optimal_rate"


# ----------------------------------------------------------------------
# Frozen split + leakage
# ----------------------------------------------------------------------

def _tiny_config(tmp_path, base_records, enc, **overrides):
    ckpt = _tiny_checkpoint(tmp_path / "init.pt", enc)
    cfg = dict(
        initial_checkpoint=str(ckpt), base_dataset=str(BASE_DATASET),
        output_dir=str(tmp_path / "run"), iterations=1, levels_per_iteration=2,
        states_per_level=2, astar_max_nodes=200_000, search_simulations=6,
        val_ratio=0.25, test_ratio=0.25, epochs=1, batch_size=32,
        train_sample_size=64, eval_budgets=(1, 3), eval_limit=4,
        promotion_budget=3, device="cpu", seed=1)
    cfg.update(overrides)
    return ExpertIterationConfig(**cfg)


def test_uncommitted_promotion_cannot_change_expert_resume_checkpoint(tmp_path):
    """Committed identity wins even when the convenience mirror is overwritten."""
    root = tmp_path / "run"
    root.mkdir()
    incumbent = root / "incumbent.pt"
    candidate = root / "iter_001" / "candidate.pt"
    candidate.parent.mkdir()
    incumbent.write_bytes(b"committed-incumbent")
    candidate.write_bytes(b"uncommitted-candidate")
    best = root / "best.pt"
    best.write_bytes(incumbent.read_bytes())
    state = {
        "completed_iterations": [],
        "best_checkpoint": str(best),
        "active_protagonist_checkpoint": "incumbent.pt",
        "active_protagonist_sha256": sha256_file(incumbent),
        "active_protagonist_source_iteration": 0,
        "history": [],
    }
    (root / "run_state.json").write_text(json.dumps(state), encoding="utf-8")

    incumbent_sha256 = hashlib.sha256(best.read_bytes()).hexdigest()
    candidate_sha256 = hashlib.sha256(candidate.read_bytes()).hexdigest()
    shutil.copyfile(candidate, best)  # promotion, followed by an injected crash

    resumed = json.loads((root / "run_state.json").read_text(encoding="utf-8"))
    resume_path = resolve_committed_protagonist(root, resumed)
    resume_sha256 = hashlib.sha256(resume_path.read_bytes()).hexdigest()
    assert resumed["completed_iterations"] == []
    assert hashlib.sha256(best.read_bytes()).hexdigest() == candidate_sha256
    assert resume_sha256 == incumbent_sha256


def test_committed_checkpoint_integrity_and_stale_mirror_repair(tmp_path):
    root = tmp_path / "run"
    root.mkdir()
    committed = root / "committed.pt"
    best = root / "best.pt"
    committed.write_bytes(b"authoritative")
    best.write_bytes(b"stale")
    state = {
        "active_protagonist_checkpoint": "committed.pt",
        "active_protagonist_sha256": sha256_file(committed),
    }
    resolved = resolve_committed_protagonist(root, state)
    assert refresh_best_checkpoint(
        resolved, best, state["active_protagonist_sha256"])
    assert best.read_bytes() == b"authoritative"
    committed.write_bytes(b"corrupt")
    with pytest.raises(CheckpointIntegrityError, match="expected sha256=.*observed"):
        resolve_committed_protagonist(root, state)


def test_expert_legacy_state_migrates_to_immutable_checkpoint(
        tmp_path, base_records, enc):
    cfg = _tiny_config(tmp_path, base_records, enc, iterations=0)
    root = Path(cfg.output_dir)
    root.mkdir(parents=True)
    best = root / "best.pt"
    atomic_copy(cfg.initial_checkpoint, best)
    state = {
        "completed_iterations": [1],
        "best_checkpoint": str(best),
        "history": [{"iteration": 1, "promoted": False}],
    }
    runner = ExpertIteration(cfg)
    runner._prepare_checkpoint_state(state)
    committed = resolve_committed_protagonist(root, state)
    assert committed != best
    assert state["active_protagonist_source_iteration"] == 1
    assert sha256_file(committed) == sha256_file(best)


def test_expert_resume_rejects_changed_promotion_metric(
        tmp_path, base_records, enc):
    legacy_cfg = _tiny_config(
        tmp_path, base_records, enc, iterations=0,
        promotion_metric="search_optimal_acc")
    run_expert_iteration(legacy_cfg)
    runner = ExpertIteration(replace(
        legacy_cfg, promotion_metric="search_confirmed_optimal_rate"))
    with pytest.raises(
            ExperimentIdentityError,
            match="semantic_config.promotion_metric"):
        runner.run()
    persisted = json.loads(
        (Path(legacy_cfg.output_dir) / "config.json").read_text(encoding="utf-8"))
    assert runner.cfg.promotion_metric == "search_confirmed_optimal_rate"
    assert persisted["promotion_metric"] == "search_optimal_acc"


def test_expert_crash_after_state_commit_recovers_exactly_once(
        tmp_path, base_records, enc, monkeypatch):
    cfg0 = _tiny_config(tmp_path, base_records, enc, iterations=0)
    initial = run_expert_iteration(cfg0)["run_state"]
    incumbent_hash = initial["active_protagonist_sha256"]
    candidate_source = _tiny_checkpoint(tmp_path / "candidate-source.pt", enc)
    promote = {"value": True}

    def fake_iteration(self, iteration, base_records, split, level_pool, replay,
                       run_state):
        iter_dir = self.root / f"iter_{iteration:03d}"
        iter_dir.mkdir(parents=True, exist_ok=True)
        candidate = iter_dir / "candidate.pt"
        atomic_copy(candidate_source, candidate)
        self._crash_point("after_candidate_evaluation")
        self._crash_point("before_promotion_decision")
        report = {
            "iteration": iteration, "commit_status": "prepared",
            "promoted": promote["value"], "promotion_score_prev": 0.1,
            "promotion_score_candidate": 0.9,
            "candidate_checkpoint": relative_to_run(candidate, self.root),
            "candidate_checkpoint_sha256": sha256_file(candidate),
        }
        self._crash_point("after_promotion_decision")
        atomic_write_json(iter_dir / "report.prepared.json", report)
        return report

    monkeypatch.setattr(ExpertIteration, "_run_iteration", fake_iteration)
    crash_at = {"stage": None}

    def crash(self, stage):
        if stage == crash_at["stage"]:
            raise RuntimeError("injected crash")

    monkeypatch.setattr(ExpertIteration, "_crash_point", crash)
    cfg1 = replace(cfg0, iterations=1)
    for stage in (
            "after_candidate_evaluation", "before_promotion_decision",
            "after_promotion_decision", "after_artifacts_prepared"):
        crash_at["stage"] = stage
        with pytest.raises(RuntimeError, match="injected crash"):
            run_expert_iteration(cfg1)
        uncommitted = json.loads(
            (Path(cfg1.output_dir) / "run_state.json").read_text(encoding="utf-8"))
        assert uncommitted["completed_iterations"] == []
        assert uncommitted["active_protagonist_sha256"] == incumbent_hash
    crash_at["stage"] = "after_state_commit"
    with pytest.raises(RuntimeError, match="injected crash"):
        run_expert_iteration(cfg1)
    state = json.loads(
        (Path(cfg1.output_dir) / "run_state.json").read_text(encoding="utf-8"))
    assert state["completed_iterations"] == [1]
    assert state["active_protagonist_sha256"] != incumbent_hash
    (Path(cfg1.output_dir) / "best.pt").write_bytes(b"stale")

    def must_not_run(*args, **kwargs):
        raise AssertionError("committed iteration was rerun")

    monkeypatch.setattr(ExpertIteration, "_run_iteration", must_not_run)
    monkeypatch.setattr(ExpertIteration, "_crash_point", lambda self, stage: None)
    first = run_expert_iteration(cfg1)["run_state"]
    second = run_expert_iteration(cfg1)["run_state"]
    assert first["completed_iterations"] == second["completed_iterations"] == [1]
    assert sha256_file(Path(cfg1.output_dir) / "best.pt") == \
        first["active_protagonist_sha256"]
    assert first["active_replay_sha256"] == second["active_replay_sha256"]
    assert len(second["commits"]) == 1

    # A crash after the mirror refresh is also post-commit and must not rerun.
    monkeypatch.setattr(ExpertIteration, "_run_iteration", fake_iteration)
    monkeypatch.setattr(ExpertIteration, "_crash_point", crash)
    crash_at["stage"] = "after_best_refresh"
    cfg2 = replace(cfg1, iterations=2)
    with pytest.raises(RuntimeError, match="injected crash"):
        run_expert_iteration(cfg2)
    after_mirror = json.loads(
        (Path(cfg2.output_dir) / "run_state.json").read_text(encoding="utf-8"))
    assert after_mirror["completed_iterations"] == [1, 2]
    assert sha256_file(Path(cfg2.output_dir) / "best.pt") == \
        after_mirror["active_protagonist_sha256"]
    monkeypatch.setattr(ExpertIteration, "_run_iteration", must_not_run)
    monkeypatch.setattr(ExpertIteration, "_crash_point", lambda self, stage: None)
    recovered = run_expert_iteration(cfg2)["run_state"]
    assert recovered["completed_iterations"] == [1, 2]

    # The same deterministic fixture without crashes reaches the same commit.
    clean_cfg = replace(
        cfg0, output_dir=str(tmp_path / "clean-run"), iterations=2)
    monkeypatch.setattr(ExpertIteration, "_run_iteration", fake_iteration)
    uninterrupted = run_expert_iteration(clean_cfg)["run_state"]
    assert uninterrupted["completed_iterations"] == recovered["completed_iterations"]
    assert uninterrupted["active_protagonist_sha256"] == \
        recovered["active_protagonist_sha256"]
    assert uninterrupted["history"] == recovered["history"]
    assert uninterrupted["active_replay_sha256"] == recovered["active_replay_sha256"]

    promote["value"] = False
    rejected_cfg = replace(
        cfg0, output_dir=str(tmp_path / "rejected-run"), iterations=1)
    crash_at["stage"] = "after_artifacts_prepared"
    monkeypatch.setattr(ExpertIteration, "_crash_point", crash)
    with pytest.raises(RuntimeError, match="injected crash"):
        run_expert_iteration(rejected_cfg)
    rejected = json.loads(
        (Path(rejected_cfg.output_dir) / "run_state.json").read_text(
            encoding="utf-8"))
    assert rejected["completed_iterations"] == []
    rejected_incumbent = rejected["active_protagonist_sha256"]
    monkeypatch.setattr(ExpertIteration, "_crash_point", lambda self, stage: None)
    rejected = run_expert_iteration(rejected_cfg)["run_state"]
    assert rejected["history"][0]["promoted"] is False
    assert rejected["active_protagonist_sha256"] == rejected_incumbent


def test_frozen_split_created_and_reused(tmp_path, base_records, enc):
    cfg = _tiny_config(tmp_path, base_records, enc)
    ei = ExpertIteration(cfg)
    ei.root.mkdir(parents=True, exist_ok=True)
    m1 = ei._frozen_split(base_records)
    m2 = ei._frozen_split(base_records)   # second call loads the persisted manifest
    assert m1 == m2
    assert m1["validation_levels"] and m1["test_levels"]
    assert (ei.root / "splits.json").exists()


def test_no_val_or_test_states_enter_replay(tmp_path, base_records, enc, monkeypatch):
    # Skip the heavy training/eval; we only check replay membership after labeling.
    cfg = _tiny_config(tmp_path, base_records, enc)
    result = run_expert_iteration(cfg)
    split = json.loads((Path(cfg.output_dir) / "splits.json").read_text())
    frozen = set(split["validation_levels"]) | set(split["test_levels"])
    reloaded = ReplayBuffer(Path(cfg.output_dir) / "replay").load()
    sigs = {r["static_level_signature"] for r in reloaded.records()}
    assert not (sigs & frozen)


# ----------------------------------------------------------------------
# End-to-end + promotion/rejection via controlled eval
# ----------------------------------------------------------------------

def _patch_eval(monkeypatch, scores, known_counts=None):
    """Patch evaluate_checkpoint to pop scores in call order (val,val,test,test,...)."""
    seq = list(scores)
    known_seq = list(known_counts) if known_counts is not None else None

    def fake(env, model, enc, vn, states, *, budgets, oracle, device, c_puct=1.5,
             seed=0):
        s = seq.pop(0) if seq else 0.0
        known = known_seq.pop(0) if known_seq else 10
        confirmed = int(round(s * 10))
        return {
            "states": 10, "total_evaluated_count": 10,
            "raw_policy_optimal_acc": s,
            "raw_policy_mean_regret": 0.0, "value_mae_moves": 0.0,
            "budgets": {str(b): {
                "total_evaluated_count": 10,
                "search_optimal_classification_count": known,
                "search_optimal_classification_coverage": known / 10,
                "search_confirmed_optimal_count": confirmed,
                "search_confirmed_optimal_rate": s,
                "search_optimal_acc": confirmed / known if known else None,
                "search_mean_regret": 0.0,
                "solve_rate": 1.0, "solution_length_gap_each": 0.0,
                "solution_length_gap_common": 0.0} for b in budgets},
            "grouped_by_board": {}, "grouped_by_difficulty": {}}
    monkeypatch.setattr(iterate_mod, "evaluate_checkpoint", fake)


def test_candidate_promoted_when_better(tmp_path, base_records, enc, monkeypatch):
    cfg = _tiny_config(tmp_path, base_records, enc)
    # call order: prev_val, cand_val, prev_test, cand_test, prev_new, cand_new
    _patch_eval(monkeypatch, [0.1, 0.9, 0.1, 0.9, 0.1, 0.9])
    result = run_expert_iteration(cfg)
    assert result["run_state"]["history"][0]["promoted"] is True


def test_candidate_rejected_when_worse(tmp_path, base_records, enc, monkeypatch):
    cfg = _tiny_config(tmp_path, base_records, enc)
    _patch_eval(monkeypatch, [0.9, 0.1, 0.9, 0.1, 0.9, 0.1])
    result = run_expert_iteration(cfg)
    assert result["run_state"]["history"][0]["promoted"] is False
    # best checkpoint stays the initial one (best.pt was copied from init).
    assert result["run_state"]["best_checkpoint"].endswith("best.pt")


def test_expert_iteration_uses_fixed_rate_with_unequal_coverage(
        tmp_path, base_records, enc, monkeypatch):
    cfg = _tiny_config(tmp_path, base_records, enc)
    # Candidate conditional accuracy is 7/8 > incumbent 8/10, but its fixed
    # rate is 7/10 < 8/10.
    _patch_eval(
        monkeypatch,
        [0.8, 0.7, 0.8, 0.7, 0.8, 0.7],
        known_counts=[10, 8, 10, 8, 10, 8])
    result = run_expert_iteration(cfg)
    report = json.loads(
        (Path(cfg.output_dir) / "iter_001" / "report.json").read_text())
    assert result["run_state"]["history"][0]["promoted"] is False
    assert report["validation"]["candidate"]["budgets"]["3"][
        "search_optimal_acc"] == pytest.approx(7 / 8)
    assert report["promotion_score_prev"] == pytest.approx(0.8)
    assert report["promotion_score_candidate"] == pytest.approx(0.7)
    assert report["promotion_prev_classification_known_count"] == 10
    assert report["promotion_candidate_classification_known_count"] == 8


def test_end_to_end_iteration(tmp_path, base_records, enc, monkeypatch):
    cfg = _tiny_config(tmp_path, base_records, enc)
    _patch_eval(monkeypatch, [0.1, 0.9, 0.1, 0.9, 0.1, 0.9])
    result = run_expert_iteration(cfg)
    out = Path(cfg.output_dir)
    assert (out / "iter_001" / "report.json").exists()
    assert (out / "best.pt").exists()
    report = json.loads((out / "iter_001" / "report.json").read_text())
    assert "validation" in report and "frozen_test" in report
    assert report["replay_size"] > 0
    assert report["promotion_total_count"] == 10
    assert report["promotion_prev_classification_known_count"] == 10
    assert report["promotion_candidate_classification_known_count"] == 10
    assert report["promotion_prev_confirmed_optimal_count"] == 1
    assert report["promotion_candidate_confirmed_optimal_count"] == 9
    assert result["run_state"]["completed_iterations"] == [1]


def test_resume_skips_completed_iterations(tmp_path, base_records, enc, monkeypatch):
    cfg1 = _tiny_config(tmp_path, base_records, enc, iterations=1)
    _patch_eval(monkeypatch, [0.1, 0.9, 0.1, 0.9, 0.1, 0.9])
    run_expert_iteration(cfg1)

    # Preserve the original initial-checkpoint bytes; rebuilding a torch
    # archive at the same path is intentionally a different input identity.
    cfg2 = replace(cfg1, iterations=2)
    _patch_eval(monkeypatch, [0.1, 0.9, 0.1, 0.9, 0.1, 0.9])
    result = run_expert_iteration(cfg2)
    assert result["run_state"]["completed_iterations"] == [1, 2]
    assert len(result["run_state"]["history"]) == 2


def test_deterministic_generation_and_labeling(tmp_path, base_records, enc, monkeypatch):
    _patch_eval(monkeypatch, [0.5] * 12)
    cfg_a = _tiny_config(tmp_path / "A", base_records, enc)
    ra = run_expert_iteration(cfg_a)
    _patch_eval(monkeypatch, [0.5] * 12)
    cfg_b = _tiny_config(tmp_path / "B", base_records, enc)
    rb = run_expert_iteration(cfg_b)
    rep_a = json.loads((Path(cfg_a.output_dir) / "iter_001" / "report.json").read_text())
    rep_b = json.loads((Path(cfg_b.output_dir) / "iter_001" / "report.json").read_text())
    assert rep_a["states_generated"] == rep_b["states_generated"]
    assert rep_a["label_stats"] == rep_b["label_stats"]
