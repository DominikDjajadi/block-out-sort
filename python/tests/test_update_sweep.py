from __future__ import annotations

import argparse
import json

import pytest
import torch

from blocksort.cotraining import update_sweep
from blocksort.expert_iteration.train import _set_frozen_modules_eval


def test_parse_candidate_defaults_and_anchor():
    gentle = update_sweep._parse_candidate("gentle=0.0001")
    anchored = update_sweep._parse_candidate("anchor=0.0001,2,0.5")
    policy = update_sweep._parse_candidate(
        "policy=0.00003,1,1,policy_head")
    balanced = update_sweep._parse_candidate(
        "balanced=0.00003,1,1,value_head,hard_tail,0.25")
    sampled = update_sweep._parse_candidate(
        "sampled=0.00003,1,1,value_head,hard_tail,0.5,depth_stratified")
    policy_anchored = update_sweep._parse_candidate(
        "policy_anchored=0.00003,1,1,policy_head,standard,0,shared,0.5")
    conditioned = update_sweep._parse_candidate(
        "conditioned=0.0001,2,1,policy_head,standard,0,shared,0,"
        "incumbent_optimal")
    ranked = update_sweep._parse_candidate(
        "ranked=0.0001,2,1,policy_head,standard,0,shared,0,"
        "incumbent_optimal,0.1")
    trunk = update_sweep._parse_candidate(
        "trunk=0.0001,2,1,policy_trunk,standard,1,shared,0,"
        "incumbent_optimal,0.01")
    adapter = update_sweep._parse_candidate(
        "adapter=0.0001,2,1,policy_adapter,standard,0,shared,0,"
        "incumbent_optimal,0.01")

    assert gentle == update_sweep.CandidateSpec("gentle", 1e-4, 1, 1.0)
    assert anchored == update_sweep.CandidateSpec("anchor", 1e-4, 2, 0.5)
    assert policy == update_sweep.CandidateSpec(
        "policy", 3e-5, 1, 1.0, "policy_head")
    assert balanced == update_sweep.CandidateSpec(
        "balanced", 3e-5, 1, 1.0, "value_head", "hard_tail", 0.25)
    assert sampled == update_sweep.CandidateSpec(
        "sampled", 3e-5, 1, 1.0, "value_head", "hard_tail", 0.5,
        "depth_stratified")
    assert policy_anchored == update_sweep.CandidateSpec(
        "policy_anchored", 3e-5, 1, 1.0, "policy_head", "standard", 0.0,
        "shared", 0.5)
    assert conditioned == update_sweep.CandidateSpec(
        "conditioned", 1e-4, 2, 1.0, "policy_head", "standard", 0.0,
        "shared", 0.0, "incumbent_optimal")
    assert ranked == update_sweep.CandidateSpec(
        "ranked", 1e-4, 2, 1.0, "policy_head", "standard", 0.0,
        "shared", 0.0, "incumbent_optimal", 0.1)
    assert trunk == update_sweep.CandidateSpec(
        "trunk", 1e-4, 2, 1.0, "policy_trunk", "standard", 1.0,
        "shared", 0.0, "incumbent_optimal", 0.01)
    assert adapter == update_sweep.CandidateSpec(
        "adapter", 1e-4, 2, 1.0, "policy_adapter", "standard", 0.0,
        "shared", 0.0, "incumbent_optimal", 0.01)


@pytest.mark.parametrize(
    "value",
    [
        "missing",
        "x=0",
        "x=0.1,0",
        "x=0.1,1,0",
        "x=0.1,1,1,unknown",
        "x=0.1,1,1,value_head,unknown",
        "x=0.1,1,1,value_head,hard_tail,-1",
        "x=0.1,1,1,policy_head,hard_tail,0",
        "x=0.1,1,1,value_head,hard_tail,0,unknown",
        "x=0.1,1,1,policy_head,standard,0,depth_stratified",
        "x=0.1,1,1,policy_head,standard,0,shared,-1",
        "x=0.1,1,1,value_head,standard,0,shared,0.5",
        "x=0.1,1,1,policy_head,standard,0,shared,0,unknown",
        "x=0.1,1,1,value_head,standard,0,shared,0,incumbent_optimal",
        "x=0.1,1,1,policy_head,standard,0,shared,0,incumbent_optimal,-1",
        "x=0.1,1,1,value_head,standard,0,shared,0,recorded,0.1",
        "bad name=0.1",
    ],
)
def test_parse_candidate_rejects_invalid_specs(value):
    with pytest.raises(argparse.ArgumentTypeError):
        update_sweep._parse_candidate(value)


def test_interpolate_toward_incumbent_keeps_half_update():
    model = torch.nn.Linear(2, 1, bias=False)
    with torch.no_grad():
        model.weight.copy_(torch.tensor([[3.0, 5.0]]))
    incumbent = {"weight": torch.tensor([[1.0, 1.0]])}

    update_sweep._interpolate_toward_incumbent(model, incumbent, 0.5)

    assert torch.equal(model.weight, torch.tensor([[2.0, 3.0]]))


class _TwoHeadModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.stem = torch.nn.Sequential(
            torch.nn.Linear(2, 2),
            torch.nn.BatchNorm1d(2),
        )
        self.trunk = torch.nn.Linear(2, 2)
        self.policy_adapter = torch.nn.Sequential(torch.nn.Linear(2, 2))
        self.policy_conv = torch.nn.Sequential(
            torch.nn.Linear(2, 2),
            torch.nn.BatchNorm1d(2),
        )
        self.value_conv = torch.nn.Linear(2, 1)
        self.value_mlp = torch.nn.Linear(1, 1)


def test_policy_head_mode_freezes_other_parameters_and_state():
    model = _TwoHeadModel()

    info = update_sweep._configure_trainable_part(model, "policy_head")
    model.train()
    _set_frozen_modules_eval(model)

    assert info["trainable_part"] == "policy_head"
    assert info["trainable_parameters"] < info["total_parameters"]
    assert all(
        parameter.requires_grad
        for parameter in model.policy_conv.parameters()
    )
    assert all(
        not parameter.requires_grad
        for module in (model.stem, model.trunk, model.value_conv, model.value_mlp)
        for parameter in module.parameters()
    )
    assert model.policy_conv[1].training
    assert not model.stem[1].training


def test_value_head_mode_selects_both_value_modules():
    model = _TwoHeadModel()

    update_sweep._configure_trainable_part(model, "value_head")

    assert all(
        parameter.requires_grad
        for module in (model.value_conv, model.value_mlp)
        for parameter in module.parameters()
    )
    assert all(
        not parameter.requires_grad
        for module in (model.stem, model.trunk, model.policy_conv)
        for parameter in module.parameters()
    )


def test_policy_trunk_mode_updates_shared_features_but_freezes_value_head():
    model = _TwoHeadModel()

    info = update_sweep._configure_trainable_part(model, "policy_trunk")
    model.train()
    _set_frozen_modules_eval(model)

    assert info["trainable_part"] == "policy_trunk"
    assert all(
        parameter.requires_grad
        for module in (model.stem, model.trunk, model.policy_conv)
        for parameter in module.parameters()
    )
    assert all(
        not parameter.requires_grad
        for module in (model.value_conv, model.value_mlp)
        for parameter in module.parameters()
    )
    assert model.stem[1].training


def test_policy_adapter_mode_selects_adapter_and_policy_head_only():
    model = _TwoHeadModel()

    info = update_sweep._configure_trainable_part(model, "policy_adapter")

    assert info["trainable_part"] == "policy_adapter"
    assert all(
        parameter.requires_grad
        for module in (model.policy_adapter, model.policy_conv)
        for parameter in module.parameters()
    )
    assert all(
        not parameter.requires_grad
        for module in (model.stem, model.trunk, model.value_conv, model.value_mlp)
        for parameter in module.parameters()
    )


def test_candidate_model_expands_legacy_checkpoint_with_identity_adapter():
    from blocksort.training.config import EncodingConfig, ModelConfig
    from blocksort.training.model import PolicyValueNet

    encoding = EncodingConfig(max_rows=2, max_cols=2, max_slide_distance=2)
    config = ModelConfig(
        channels=4, residual_blocks=1, value_hidden_size=8)
    incumbent = PolicyValueNet(encoding, config).eval()
    checkpoint = {"model_state": incumbent.state_dict()}

    candidate, candidate_config = update_sweep._candidate_model(
        checkpoint,
        encoding=encoding,
        incumbent_model_config=config,
        trainable_part="policy_adapter",
        device=torch.device("cpu"),
    )
    candidate.eval()
    board = torch.randn(
        2, encoding.num_board_channels, encoding.max_rows, encoding.max_cols)
    glob = torch.randn(2, encoding.num_global_features)
    with torch.no_grad():
        expected = incumbent(board, glob)
        observed = candidate(board, glob)

    assert candidate_config.policy_adapter_blocks == 1
    assert torch.equal(observed[0], expected[0])
    assert torch.equal(observed[1], expected[1])


def test_hard_tail_value_weights_are_capped_and_source_aware():
    records = [
        {"value_target": {"raw_optimal_moves": 3}},
        {"value_target": {"raw_optimal_moves": 7}},
        {"value_target": {"raw_optimal_moves": 9.5}},
        {"value_target": {"raw_optimal_moves": 10}},
        {"value_target": {"raw_optimal_moves": 20}},
    ]

    weights, summary = update_sweep._value_weights_for(
        records, [1.0, 1.0, 0.5, 0.5, 1.5], "hard_tail")

    assert weights == [1.0, 2.0, 1.0, 1.5, 4.5]
    assert summary["profile"] == "hard_tail"
    assert summary["total_weight_mass"] == 10.0
    assert summary["by_depth"]["10_plus"] == {
        "examples": 2,
        "weight_mass": 6.0,
        "weight_share": 0.6,
    }


def test_sample_summary_reports_diversity():
    records = [
        {
            "static_level_signature": "level-a",
            "state_key": "state-a",
            "generation_iteration": 0,
            "target_source": "exact_oracle",
        },
        {
            "static_level_signature": "level-b",
            "state_key": "state-b",
            "generation_iteration": 2,
            "target_source": "graph_search",
        },
    ]

    summary = update_sweep._sample_summary(records)

    assert summary == {
        "examples": 2,
        "unique_examples": 2,
        "unique_levels": 2,
        "by_iteration": {"0": 1, "2": 1},
        "by_source": {"exact_oracle": 1, "graph_search": 1},
    }


def test_incumbent_conditioned_targets_preserve_exact_optimal_support():
    records = [
        {
            "state_key": "exact",
            "label_kind": "full-exact",
            "policy_exact": True,
            "optimal_actions_complete": True,
            "legal_actions": [{"a": 0}, {"a": 1}, {"a": 2}],
            "policy_target": [0.5, 0.5, 0.0],
            "policy": {"type": "uniform-optimal"},
        },
        {
            "state_key": "search",
            "label_kind": "search-visit-policy",
            "policy_exact": False,
            "optimal_actions_complete": False,
            "legal_actions": [{"a": 0}, {"a": 1}, {"a": 2}],
            "policy_target": [0.2, 0.3, 0.5],
            "policy": {"type": "visit-policy"},
        },
    ]
    original = json.loads(json.dumps(records))
    incumbent = [[0.8, 0.1, 0.1], [0.7, 0.2, 0.1]]

    conditioned, summary = update_sweep._condition_policy_targets(
        records,
        incumbent,
        profile="incumbent_optimal",
        incumbent_checkpoint_sha256="incumbent-hash",
    )
    sharp, sharp_summary = update_sweep._condition_policy_targets(
        records,
        incumbent,
        profile="incumbent_optimal_sharp",
        incumbent_checkpoint_sha256="incumbent-hash",
    )
    blended, blended_summary = update_sweep._condition_policy_targets(
        records,
        incumbent,
        profile="incumbent_optimal_blend50",
        incumbent_checkpoint_sha256="incumbent-hash",
    )

    assert records == original
    assert conditioned[0]["policy_target"] == pytest.approx([
        8.0 / 9.0, 1.0 / 9.0, 0.0])
    assert sharp[0]["policy_target"] == pytest.approx([
        64.0 / 65.0, 1.0 / 65.0, 0.0])
    assert blended[0]["policy_target"] == pytest.approx([
        25.0 / 36.0, 11.0 / 36.0, 0.0])
    assert conditioned[1] == records[1]
    assert sharp[1] == records[1]
    assert blended[1] == records[1]
    assert conditioned[0]["policy"]["incumbent_checkpoint_sha256"] == \
        "incumbent-hash"
    assert summary["eligible_complete_exact_records"] == 1
    assert summary["changed_records"] == 1
    assert summary["max_suboptimal_probability_mass"] == 0.0
    assert summary["mean_eligible_entropy_after"] < \
        summary["mean_eligible_entropy_before"]
    assert sharp_summary["mean_eligible_entropy_after"] < \
        summary["mean_eligible_entropy_after"]
    assert summary["mean_eligible_entropy_after"] < \
        blended_summary["mean_eligible_entropy_after"]
    assert blended_summary["mean_eligible_entropy_after"] < \
        blended_summary["mean_eligible_entropy_before"]
    assert blended_summary["uniform_optimal_mix"] == pytest.approx(0.5)
    assert blended[0]["policy"]["uniform_optimal_mix"] == pytest.approx(0.5)
    assert sharp_summary["policy_target_sha256"] != \
        summary["policy_target_sha256"]


def test_update_sweep_identity_survives_json_round_trip(tmp_path):
    incumbent = tmp_path / "inc.pt"
    replay = tmp_path / "replay.jsonl"
    incumbent.write_text("checkpoint", encoding="utf-8")
    replay.write_text("{}\n", encoding="utf-8")
    cfg = update_sweep.UpdateSweepConfig(
        incumbent_checkpoint=str(incumbent),
        replay_snapshot=str(replay),
        output_dir=str(tmp_path / "out"),
    )

    identity = update_sweep._identity(
        cfg,
        sample_sha256="sample",
        value_sample_sha256={"shared": "sample"},
        policy_weight_sha256={"shared": "policy"},
        value_weight_sha256={"shared": "value"},
        policy_target_sha256={"recorded": "targets"},
    )

    assert json.loads(json.dumps(identity)) == identity


def test_exact_persisted_sample_and_weights_are_authoritative(
        tmp_path, monkeypatch):
    incumbent = tmp_path / "inc.pt"
    replay = tmp_path / "replay.jsonl"
    sample = tmp_path / "failed_round_sample.jsonl"
    policy_weights = tmp_path / "policy.json"
    value_weights = tmp_path / "value.json"
    incumbent.write_text("checkpoint", encoding="utf-8")
    replay.write_text("{}\n", encoding="utf-8")
    sample.write_text("{}\n{}\n", encoding="utf-8")
    policy_weights.write_text("[1.5, 0.25]", encoding="utf-8")
    value_weights.write_text("[1.5, 0.0]", encoding="utf-8")
    records = [
        {
            "static_level_signature": "level-a",
            "state_key": "state-a",
            "generation_iteration": 1,
            "target_source": "exact_oracle",
        },
        {
            "static_level_signature": "level-b",
            "state_key": "state-b",
            "generation_iteration": 0,
            "target_source": "graph_search",
        },
    ]
    monkeypatch.setattr(update_sweep, "load_records", lambda _path: records)
    cfg = update_sweep.UpdateSweepConfig(
        incumbent_checkpoint=str(incumbent),
        replay_snapshot=str(replay),
        output_dir=str(tmp_path / "out"),
        sample_jsonl=str(sample),
        policy_weights_json=str(policy_weights),
        value_weights_json=str(value_weights),
    )

    loaded, _digest, summary, loaded_policy, loaded_value = (
        update_sweep._load_or_create_sample(cfg, tmp_path / "out"))

    assert loaded == records
    assert loaded_policy == [1.5, 0.25]
    assert loaded_value == [1.5, 0.0]
    assert summary["sampling"]["mode"] == "persisted_exact_sample"
    assert summary["sampling"]["provided_sample_sha256"] == \
        update_sweep.sha256_file(sample)


def test_fallback_sample_matches_cotraining_quota_and_source_weighting(
        tmp_path, monkeypatch):
    incumbent = tmp_path / "inc.pt"
    replay = tmp_path / "replay.jsonl"
    incumbent.write_text("checkpoint", encoding="utf-8")
    replay.write_text("{}\n", encoding="utf-8")
    records = [
        {
            "static_level_signature": "level-a",
            "state_key": "state-a",
            "generation_iteration": 3,
            "target_source": "exact_oracle",
            "label_kind": "full-exact",
        },
        {
            "static_level_signature": "level-b",
            "state_key": "state-b",
            "generation_iteration": 3,
            "target_source": "exact_oracle",
            "label_kind": "exact-path-policy",
        },
        {
            "static_level_signature": "level-c",
            "state_key": "state-c",
            "generation_iteration": 0,
            "target_source": "graph_search",
            "label_kind": "search-policy",
        },
    ]
    observed = {}

    class FakeReplay:
        def __init__(self, *_args, **_kwargs):
            pass

        def load_snapshot(self, path):
            observed["snapshot"] = path
            return self

        def sample_training_set_with_age_quotas(self, size, **kwargs):
            observed["size"] = size
            observed["kwargs"] = kwargs
            return records, {"policy": "fresh_recent_historical_quota_v1"}

    monkeypatch.setattr(update_sweep, "ReplayBuffer", FakeReplay)
    cfg = update_sweep.UpdateSweepConfig(
        incumbent_checkpoint=str(incumbent),
        replay_snapshot=str(replay),
        output_dir=str(tmp_path / "out"),
        sample_size=3,
        current_iteration=3,
        sample_seed=40,
        exact_path_policy_confidence=0.5,
    )

    _records, _digest, summary, policy, value = (
        update_sweep._load_or_create_sample(cfg, tmp_path / "out"))

    assert observed["size"] == 3
    assert observed["kwargs"] == {
        "current_iteration": 3,
        "current_fraction": 0.35,
        "recent_fraction": 0.25,
        "historical_fraction": 0.40,
        "recent_window": 2,
        "weight_exact_historical": 1.0,
        "weight_exact_new": 1.5,
        "weight_search": 0.5,
        "seed": 40,
        "with_replacement": False,
    }
    assert policy == [1.5, 0.75, 0.5]
    assert value == policy
    assert summary["sampling"]["mode"] == \
        "fresh_recent_historical_quota_v1"


def test_explicit_sample_requires_both_weight_files(tmp_path):
    incumbent = tmp_path / "inc.pt"
    replay = tmp_path / "replay.jsonl"
    sample = tmp_path / "sample.jsonl"
    incumbent.write_text("checkpoint", encoding="utf-8")
    replay.write_text("{}\n", encoding="utf-8")
    sample.write_text("{}\n", encoding="utf-8")
    cfg = update_sweep.UpdateSweepConfig(
        incumbent_checkpoint=str(incumbent),
        replay_snapshot=str(replay),
        output_dir=str(tmp_path / "out"),
        sample_jsonl=str(sample),
    )

    with pytest.raises(ValueError, match="must be supplied together"):
        cfg.validate()
