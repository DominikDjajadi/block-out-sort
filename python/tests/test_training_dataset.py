"""Tests for level-grouped splitting and the PyTorch dataset loader."""

from __future__ import annotations

import copy
import json
import math
from pathlib import Path

import pytest
import torch

from blocksort import Environment
from blocksort.dataset.schema import LABEL_SEARCH_VISIT_POLICY
from blocksort.training.config import EncodingConfig, ValueNormConfig
from blocksort.training.dataset import (
    PolicyValueDataset,
    collate_batch,
    load_records,
)
from blocksort.training.splits import (
    SplitRatios,
    collect_level_keys,
    filter_records_for_split,
    load_manifest,
    make_split,
    save_manifest,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
DATASET = REPO_ROOT / "data" / "training" / "pv_examples.jsonl"
CFG = EncodingConfig()
VN = ValueNormConfig()


@pytest.fixture(scope="module")
def records():
    return load_records(DATASET)


# --------------------------------------------------------------------------
# Splits
# --------------------------------------------------------------------------

def _keys(n):
    return [f"lvl{i:03d}" for i in range(n)]


def test_splits_disjoint_and_cover():
    m = make_split(_keys(20), ratios=SplitRatios(0.8, 0.1, 0.1), seed=1)
    tr, va, te = set(m["train_levels"]), set(m["validation_levels"]), set(m["test_levels"])
    assert not (tr & va) and not (tr & te) and not (va & te)
    assert tr | va | te == set(_keys(20))
    assert len(va) == 2 and len(te) == 2 and len(tr) == 16


@pytest.mark.parametrize("ratios", [
    SplitRatios(-0.1, 0.6, 0.5),
    SplitRatios(1.1, 0.0, -0.1),
    SplitRatios(float("nan"), 0.5, 0.5),
    SplitRatios(float("inf"), 0.0, 0.0),
    SplitRatios(0.7, 0.1, 0.1),
    SplitRatios(0.7, 0.2, 0.2),
])
def test_split_ratio_validation_rejects_invalid_components_or_sum(ratios):
    with pytest.raises(ValueError, match="split ratio"):
        make_split(_keys(10), ratios=ratios)


@pytest.mark.parametrize("ratios", [
    SplitRatios(1.0, 0.0, 0.0),
    SplitRatios(0.0, 1.0, 0.0),
    SplitRatios(0.0, 0.0, 1.0),
])
def test_split_ratio_valid_boundaries_cover_without_overlap(ratios):
    keys = _keys(3)
    manifest = make_split(keys, ratios=ratios, seed=4)
    partitions = [
        manifest["train_levels"],
        manifest["validation_levels"],
        manifest["test_levels"],
    ]
    assert sum(len(part) for part in partitions) == len(keys)
    assert set().union(*map(set, partitions)) == set(keys)
    assert all(not (set(a) & set(b))
               for i, a in enumerate(partitions)
               for b in partitions[i + 1:])


def test_splits_deterministic_for_seed():
    a = make_split(_keys(30), seed=7)
    b = make_split(_keys(30), seed=7)
    assert a == b


def test_splits_differ_for_different_seeds():
    a = make_split(_keys(30), seed=1)
    b = make_split(_keys(30), seed=2)
    assert a["train_levels"] != b["train_levels"]
    # still a valid partition
    assert set(a["train_levels"]) | set(a["validation_levels"]) | set(a["test_levels"]) \
        == set(b["train_levels"]) | set(b["validation_levels"]) | set(b["test_levels"])


def test_manifest_round_trip(tmp_path):
    m = make_split(_keys(15), seed=3)
    path = tmp_path / "splits.json"
    save_manifest(m, path)
    assert load_manifest(path) == m


def test_manifest_rejects_missing_ratio_metadata(tmp_path):
    manifest = make_split(_keys(3))
    del manifest["ratios"]["validation"]
    path = tmp_path / "bad-split.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="missing ratios.*validation"):
        load_manifest(path)


def test_small_dataset_edge_case():
    m = make_split(_keys(1), seed=0)
    assert m["train_levels"] == ["lvl000"]
    assert m["validation_levels"] == [] and m["test_levels"] == []


def test_real_records_grouped_no_leakage(records):
    m = make_split(collect_level_keys(records), seed=42)
    tr = filter_records_for_split(records, m, "train")
    va = filter_records_for_split(records, m, "validation")
    tr_keys = {r.get("static_level_signature") for r in tr}
    va_keys = {r.get("static_level_signature") for r in va}
    assert not (tr_keys & va_keys)


# --------------------------------------------------------------------------
# Loader
# --------------------------------------------------------------------------

def test_loader_builds_items(records):
    ds = PolicyValueDataset(records[:20], encoding_config=CFG, value_norm=VN)
    assert len(ds) == 20
    item = ds[0]
    assert item["board"].shape == (CFG.num_board_channels, CFG.max_rows, CFG.max_cols)
    assert item["policy_target"].shape == (CFG.action_space_size,)
    assert item["legal_action_mask"].shape == (CFG.action_space_size,)
    assert isinstance(bool(item["legal_exit_available"]), bool)
    assert isinstance(bool(item["optimal_exit_available"]), bool)


def test_loader_policy_sums_to_one_and_legal_only(records):
    ds = PolicyValueDataset(records[:30], encoding_config=CFG, value_norm=VN)
    for item in ds.items:
        assert float(item["policy_target"].sum()) == pytest.approx(1.0, abs=1e-4)
        illegal_mass = float((item["policy_target"] * (1 - item["legal_action_mask"])).sum())
        assert illegal_mass == 0.0


def test_loader_value_normalization(records):
    ds = PolicyValueDataset(records[:10], encoding_config=CFG, value_norm=VN)
    for item in ds.items:
        raw = item["raw_optimal_moves"]
        assert float(item["value_target"]) == pytest.approx(VN.normalize(raw))


def test_exact_loader_rejects_fractional_approximate_search_record(records):
    search = copy.deepcopy(records[0])
    search.update({
        "label_kind": LABEL_SEARCH_VISIT_POLICY,
        "target_source": "graph_search",
        "value_exact": False,
        "policy_exact": False,
        "optimal_actions_complete": False,
        "action_values_complete": False,
    })
    search["value_target"]["raw_optimal_moves"] = 3.75

    with pytest.raises(ValueError, match="approximate search records"):
        PolicyValueDataset([search], encoding_config=CFG, value_norm=VN)


def test_exact_loader_never_truncates_unmarked_fractional_value(records):
    malformed = copy.deepcopy(records[0])
    malformed["value_target"]["raw_optimal_moves"] = 3.75

    with pytest.raises(ValueError, match="finite integer"):
        PolicyValueDataset([malformed], encoding_config=CFG, value_norm=VN)


@pytest.mark.parametrize("constant", [0.0, -1.0, float("nan"), float("inf")])
def test_value_normalization_rejects_invalid_divisor(constant):
    with pytest.raises(ValueError, match="normalization constant"):
        ValueNormConfig(constant=constant)


def test_value_normalization_small_positive_is_finite():
    norm = ValueNormConfig(constant=1e-9)
    assert math.isfinite(norm.normalize(3))
    assert math.isfinite(norm.denormalize(norm.normalize(3)))


def test_collate_batches(records):
    ds = PolicyValueDataset(records[:8], encoding_config=CFG, value_norm=VN)
    batch = collate_batch([ds[i] for i in range(8)])
    assert batch["board"].shape[0] == 8
    assert batch["policy_target"].shape == (8, CFG.action_space_size)
    assert len(batch["level_id"]) == 8
    assert batch["legal_exit_available"].shape == (8,)
    assert batch["optimal_exit_available"].shape == (8,)


def test_loader_rejects_invalid_policy_target(records):
    bad = copy.deepcopy(records[0])
    # Corrupt the target so it no longer sums to 1.
    bad["policy_target"] = [p + 0.5 for p in bad["policy_target"]]
    with pytest.raises(ValueError):
        PolicyValueDataset([bad], encoding_config=CFG, value_norm=VN)


def test_loader_rejects_oversized_board(records):
    bad = copy.deepcopy(records[0])
    bad["level"]["cols"] = 50
    with pytest.raises(ValueError):
        PolicyValueDataset([bad], encoding_config=CFG, value_norm=VN)
