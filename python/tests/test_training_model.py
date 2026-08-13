"""Model, loss, metric, checkpoint, and training-behavior tests."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest
import torch

from blocksort.training.checkpoint import (
    CHECKPOINT_VERSION,
    load_checkpoint,
    model_from_checkpoint,
    save_checkpoint,
)
from blocksort.training.config import (
    EncodingConfig,
    ModelConfig,
    ValueNormConfig,
    model_profile,
)
from blocksort.training.dataset import PolicyValueDataset, collate_batch, load_records
from blocksort.training.losses import (
    compute_losses,
    masked_log_softmax,
    policy_loss_fn,
    value_loss_fn,
)
from blocksort.training.metrics import MetricAccumulator
from blocksort.training.model import PolicyValueNet, count_parameters
from blocksort.training.train import (
    build_parser,
    evaluate_loader,
    model_config_from_args,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
DATASET = REPO_ROOT / "data" / "training" / "pv_examples.jsonl"

SMALL = EncodingConfig(max_rows=2, max_cols=2, max_slide_distance=2)  # A = 48
VN = ValueNormConfig()


def _model(cfg=SMALL, channels=8, blocks=1):
    return PolicyValueNet(cfg, ModelConfig(channels=channels, residual_blocks=blocks,
                                           value_hidden_size=16))


# --------------------------------------------------------------------------
# Model
# --------------------------------------------------------------------------

def test_forward_shapes():
    cfg = SMALL
    model = _model().eval()
    b = 4
    board = torch.randn(b, cfg.num_board_channels, cfg.max_rows, cfg.max_cols)
    glob = torch.randn(b, cfg.num_global_features)
    logits, value = model(board, glob)
    assert logits.shape == (b, cfg.action_space_size)
    assert value.shape == (b,)


def test_policy_adapter_is_exact_identity_and_value_is_decoupled():
    torch.manual_seed(7)
    base_config = ModelConfig(
        channels=8, residual_blocks=1, value_hidden_size=16)
    adapter_config = ModelConfig(
        channels=8, residual_blocks=1, value_hidden_size=16,
        policy_adapter_blocks=1)
    base = PolicyValueNet(SMALL, base_config).eval()
    adapter = PolicyValueNet(SMALL, adapter_config).eval()
    incompatible = adapter.load_state_dict(base.state_dict(), strict=False)

    assert not incompatible.unexpected_keys
    assert incompatible.missing_keys
    assert all(
        name.startswith("policy_adapter.")
        for name in incompatible.missing_keys)
    board = torch.randn(
        3, SMALL.num_board_channels, SMALL.max_rows, SMALL.max_cols)
    glob = torch.randn(3, SMALL.num_global_features)
    with torch.no_grad():
        base_policy, base_value = base(board, glob)
        adapter_policy, adapter_value = adapter(board, glob)

    assert torch.equal(adapter_policy, base_policy)
    assert torch.equal(adapter_value, base_value)


def test_model_config_migrates_legacy_policy_adapter_default():
    legacy = {
        "channels": 8,
        "residual_blocks": 1,
        "value_hidden_size": 16,
        "normalization": "group_norm",
    }

    config = ModelConfig.from_dict(legacy)

    assert config.policy_adapter_blocks == 0
    assert config.to_dict()["policy_adapter_blocks"] == 0


def test_small_groupnorm_profile_is_compact_and_has_no_running_stats():
    config = model_profile("small_groupnorm")
    model = PolicyValueNet(EncodingConfig(), config)

    assert config == ModelConfig(
        channels=32,
        residual_blocks=2,
        value_hidden_size=64,
        normalization="group_norm",
    )
    assert 60_000 < count_parameters(model) < 100_000
    assert any(isinstance(module, torch.nn.GroupNorm)
               for module in model.modules())
    assert not any(isinstance(module, torch.nn.BatchNorm2d)
                   for module in model.modules())


def test_training_cli_defaults_to_small_profile_and_supports_legacy_profile():
    base = ["--dataset", "data.jsonl", "--output-dir", "run"]
    compact = model_config_from_args(build_parser().parse_args(base))
    legacy = model_config_from_args(build_parser().parse_args([
        *base, "--model-profile", "legacy_batchnorm",
    ]))
    overridden = model_config_from_args(build_parser().parse_args([
        *base, "--channels", "48", "--normalization", "batch_norm",
    ]))

    assert compact == model_profile("small_groupnorm")
    assert legacy == model_profile("legacy_batchnorm")
    assert overridden.channels == 48
    assert overridden.normalization == "batch_norm"


def test_policy_flatten_matches_index_formula():
    # A single hot channel/spatial location must land at the formula index.
    cfg = SMALL
    model = _model()
    m = cfg.move_code_count
    p = torch.zeros(1, cfg.policy_channels, cfg.max_rows, cfg.max_cols)
    r, c, d, mc = 1, 0, 2, 1
    p[0, d * m + mc, r, c] = 1.0
    flat = model._flatten_policy(p)
    expected = ((r * cfg.max_cols + c) * 4 + d) * m + mc
    assert int(flat.argmax().item()) == expected


def test_backward_pass_cpu():
    model = _model()
    board = torch.randn(2, SMALL.num_board_channels, SMALL.max_rows, SMALL.max_cols)
    glob = torch.randn(2, SMALL.num_global_features)
    logits, value = model(board, glob)
    loss = logits.sum() + value.sum()
    loss.backward()
    assert any(p.grad is not None for p in model.parameters())


@pytest.mark.skipif(not torch.cuda.is_available(), reason="no CUDA")
def test_cuda_smoke():
    model = _model().cuda().eval()
    board = torch.randn(2, SMALL.num_board_channels, SMALL.max_rows, SMALL.max_cols).cuda()
    glob = torch.randn(2, SMALL.num_global_features).cuda()
    logits, value = model(board, glob)
    assert logits.is_cuda and value.is_cuda


# --------------------------------------------------------------------------
# Losses / masking
# --------------------------------------------------------------------------

def _craft_batch():
    A = SMALL.action_space_size
    legal = torch.zeros(1, A)
    legal[0, [0, 1, 2]] = 1.0
    target = torch.zeros(1, A)
    target[0, 0] = 0.5
    target[0, 1] = 0.5            # two optimal actions
    regret = torch.zeros(1, A)
    regret[0, 2] = 1.0            # idx 2 suboptimal
    value_target = torch.tensor([VN.normalize(4)])
    return {
        "legal_action_mask": legal, "policy_target": target, "action_regret": regret,
        "value_target": value_target, "raw_optimal_moves": torch.tensor([4.0]),
        "rows": torch.tensor([2]), "cols": torch.tensor([2]),
        "remaining_blocks": torch.tensor([2]), "legal_action_count": torch.tensor([3]),
        "provenance_type": ["test"],
    }


def test_masked_log_softmax_all_masked_is_finite():
    A = SMALL.action_space_size
    logits = torch.randn(1, A)
    lp = masked_log_softmax(logits, torch.zeros(1, A))
    assert torch.isfinite(lp).all()


def test_policy_loss_multiple_optimal_and_legal_only():
    batch = _craft_batch()
    logits = torch.zeros(1, SMALL.action_space_size)
    logits[0, [0, 1]] = 20.0  # concentrate on the two optimal actions
    loss = policy_loss_fn(logits, batch["policy_target"], batch["legal_action_mask"])
    # Best achievable CE for a 2-way uniform target equals its entropy (ln 2).
    import math
    assert torch.isfinite(loss) and abs(loss.item() - math.log(2)) < 0.05


def test_terminal_sample_no_nan():
    A = SMALL.action_space_size
    legal = torch.zeros(2, A); legal[0, 0] = 1.0  # sample 1 legal, sample 2 terminal
    target = torch.zeros(2, A); target[0, 0] = 1.0
    logits = torch.randn(2, A)
    loss = policy_loss_fn(logits, target, legal)
    assert torch.isfinite(loss)


def test_value_losses():
    pred = torch.tensor([-0.2, -0.5])
    tgt = torch.tensor([-0.25, -0.4])
    assert value_loss_fn(pred, tgt, loss_type="huber").item() >= 0
    assert value_loss_fn(pred, tgt, loss_type="mse").item() >= 0


# --------------------------------------------------------------------------
# Metrics
# --------------------------------------------------------------------------

def test_metrics_top1_optimal_and_mass():
    batch = _craft_batch()
    logits = torch.zeros(1, SMALL.action_space_size)
    logits[0, 0] = 5.0  # pick optimal idx 0
    value = batch["value_target"].clone()
    acc = MetricAccumulator(VN)
    acc.update(logits, value, batch)
    m = acc.compute()
    assert m["policy_top1_optimal_acc"] == 1.0
    assert m["policy_optimal_mass"] > 0.9
    assert m["policy_selected_regret_mean"] == 0.0
    assert m["value_raw_mae"] == pytest.approx(0.0, abs=1e-5)


def test_metrics_separate_setup_required_from_immediately_exitable():
    setup = _craft_batch()
    setup["optimal_exit_available"] = torch.tensor([False])
    immediate = _craft_batch()
    immediate["optimal_exit_available"] = torch.tensor([True])
    logits = torch.zeros(1, SMALL.action_space_size)
    logits[0, 0] = 5.0
    acc = MetricAccumulator(VN)

    acc.update(logits, setup["value_target"], setup)
    acc.update(logits, immediate["value_target"], immediate)

    grouped = acc.grouped("policy_strategy")
    assert grouped["setup_required"]["count"] == 1
    assert grouped["immediately_exitable"]["count"] == 1
    assert grouped["setup_required"]["policy_top1_optimal_acc"] == 1.0


def test_metrics_value_denormalization():
    batch = _craft_batch()
    # Predict a normalized value that denormalizes to 6 moves while truth is 4.
    value = torch.tensor([VN.normalize(6)])
    acc = MetricAccumulator(VN)
    acc.update(torch.zeros(1, SMALL.action_space_size), value, batch)
    m = acc.compute()
    assert m["value_raw_mae"] == pytest.approx(2.0, abs=1e-4)


def test_partial_path_policy_is_not_counted_as_complete_optimal_set():
    batch = _craft_batch()
    batch["optimal_actions_complete"] = torch.tensor([False])
    batch["action_regret_known_mask"] = torch.zeros_like(batch["action_regret"])
    batch["action_regret_known_mask"][0, 0] = 1.0
    logits = torch.zeros(1, SMALL.action_space_size)
    logits[0, 0] = 5.0
    acc = MetricAccumulator(VN)
    acc.update(logits, batch["value_target"], batch)
    metrics = acc.compute()
    assert metrics["policy_complete_label_count"] == 0
    assert metrics["policy_top1_optimal_acc"] != metrics["policy_top1_optimal_acc"]
    assert metrics["policy_verified_action_top1_acc"] == 1.0


def test_policy_only_evaluation_loss_excludes_value_error():
    batch = _craft_batch()

    class FixedModel(torch.nn.Module):
        def forward(self, board, global_features):
            logits = torch.zeros(
                board.shape[0], SMALL.action_space_size,
                dtype=board.dtype, device=board.device)
            return logits, torch.full(
                (board.shape[0],), VN.normalize(20),
                dtype=board.dtype, device=board.device)

    batch["board"] = torch.zeros(
        1, SMALL.num_board_channels, SMALL.max_rows, SMALL.max_cols)
    batch["global_features"] = torch.zeros(1, SMALL.num_global_features)
    batch["action_regret_known_mask"] = (
        (batch["legal_action_mask"] > 0).float())
    batch["optimal_actions_complete"] = torch.tensor([True])
    metrics, loss = evaluate_loader(
        FixedModel(), [batch], VN, torch.device("cpu"),
        policy_weight=1.0, value_weight=0.0, value_loss_type="huber")
    assert loss == pytest.approx(metrics["policy_cross_entropy"])


def test_evaluation_loss_is_invariant_to_partial_batch_partitioning():
    action_count = SMALL.action_space_size
    count = 5
    legal = torch.zeros(count, action_count)
    target = torch.zeros(count, action_count)
    regret = torch.ones(count, action_count)
    for index, legal_count in enumerate((1, 2, 3, 4, 5)):
        legal[index, :legal_count] = 1.0
        target[index, 0] = 1.0
        regret[index, 0] = 0.0
    raw_moves = torch.tensor([1.0, 2.0, 4.0, 7.0, 11.0])
    batch = {
        "board": torch.zeros(
            count, SMALL.num_board_channels, SMALL.max_rows, SMALL.max_cols),
        "global_features": torch.zeros(count, SMALL.num_global_features),
        "legal_action_mask": legal,
        "policy_target": target,
        "action_regret": regret,
        "action_regret_known_mask": legal.clone(),
        "optimal_actions_complete": torch.ones(count, dtype=torch.bool),
        "value_target": torch.tensor([VN.normalize(v) for v in raw_moves]),
        "raw_optimal_moves": raw_moves,
        "remaining_blocks": torch.ones(count, dtype=torch.int64),
        "legal_action_count": legal.sum(dim=-1).to(torch.int64),
        "rows": torch.full((count,), 2, dtype=torch.int64),
        "cols": torch.full((count,), 2, dtype=torch.int64),
        "provenance_type": ["test"] * count,
    }

    class FixedModel(torch.nn.Module):
        def forward(self, board, global_features):
            return (
                torch.zeros(
                    board.shape[0], action_count,
                    dtype=board.dtype, device=board.device),
                torch.zeros(
                    board.shape[0], dtype=board.dtype, device=board.device),
            )

    def take(start, stop):
        return {
            key: value[start:stop]
            for key, value in batch.items()
        }

    model = FixedModel()
    _full_metrics, full_loss = evaluate_loader(
        model, [batch], VN, torch.device("cpu"),
        policy_weight=0.7, value_weight=1.3, value_loss_type="huber")
    _split_metrics, split_loss = evaluate_loader(
        model, [take(0, 2), take(2, 4), take(4, 5)], VN,
        torch.device("cpu"), policy_weight=0.7, value_weight=1.3,
        value_loss_type="huber")

    assert split_loss == pytest.approx(full_loss, rel=1e-7, abs=1e-7)


# --------------------------------------------------------------------------
# Checkpoint
# --------------------------------------------------------------------------

def test_checkpoint_round_trip_identical_predictions(tmp_path):
    model = _model().eval()
    board = torch.randn(3, SMALL.num_board_channels, SMALL.max_rows, SMALL.max_cols)
    glob = torch.randn(3, SMALL.num_global_features)
    with torch.no_grad():
        l0, v0 = model(board, glob)
    path = tmp_path / "ckpt.pt"
    save_checkpoint(path, model=model, optimizer=None, scheduler=None, epoch=1,
                    best_val_metric=0.0, encoding_config=SMALL,
                    model_config=model.model_config, value_norm=VN, seed=0,
                    dataset_version=1, split_identity=None)
    ckpt = load_checkpoint(path)
    restored = model_from_checkpoint(ckpt).eval()
    with torch.no_grad():
        l1, v1 = restored(board, glob)
    assert torch.allclose(l0, l1, atol=1e-6) and torch.allclose(v0, v1, atol=1e-6)


def test_checkpoint_contains_required_metadata(tmp_path):
    model = _model()
    path = tmp_path / "ckpt.pt"
    save_checkpoint(path, model=model, optimizer=None, scheduler=None, epoch=1,
                    best_val_metric=0.0, encoding_config=SMALL,
                    model_config=model.model_config, value_norm=VN, seed=0,
                    dataset_version=1, split_identity={"seed": 1})
    ckpt = load_checkpoint(path)
    for key in ("encoding_config", "model_config", "value_norm", "color_order",
                "direction_order", "seed", "dataset_version", "torch_version"):
        assert key in ckpt


def test_checkpoint_config_mismatch_detected(tmp_path):
    model = _model()
    path = tmp_path / "ckpt.pt"
    save_checkpoint(path, model=model, optimizer=None, scheduler=None, epoch=1,
                    best_val_metric=0.0, encoding_config=SMALL,
                    model_config=model.model_config, value_norm=VN, seed=0,
                    dataset_version=1, split_identity=None)
    raw = torch.load(path, weights_only=False)
    raw["color_order"] = ["wrong"]
    torch.save(raw, path)
    with pytest.raises(ValueError):
        load_checkpoint(path)


# --------------------------------------------------------------------------
# Training behavior
# --------------------------------------------------------------------------

@pytest.fixture(scope="module")
def small_batch():
    records = load_records(DATASET)[:8]
    cfg = EncodingConfig()
    ds = PolicyValueDataset(records, encoding_config=cfg, value_norm=VN)
    return collate_batch([ds[i] for i in range(len(ds))]), cfg


def test_one_step_changes_parameters(small_batch):
    batch, cfg = small_batch
    torch.manual_seed(0)
    model = PolicyValueNet(cfg, ModelConfig(channels=16, residual_blocks=1,
                                            value_hidden_size=32))
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    before = [p.detach().clone() for p in model.parameters()]
    logits, value = model(batch["board"], batch["global_features"])
    compute_losses(logits, value, batch)["total"].backward()
    opt.step()
    after = list(model.parameters())
    assert any(not torch.allclose(a, b) for a, b in zip(after, before))


def test_tiny_overfit(small_batch):
    batch, cfg = small_batch
    torch.manual_seed(0)
    model = PolicyValueNet(cfg, ModelConfig(channels=32, residual_blocks=2,
                                            value_hidden_size=64))
    opt = torch.optim.AdamW(model.parameters(), lr=1e-2, weight_decay=0.0)
    model.train()
    for _ in range(250):
        opt.zero_grad(set_to_none=True)
        logits, value = model(batch["board"], batch["global_features"])
        compute_losses(logits, value, batch)["total"].backward()
        opt.step()
    model.eval()
    acc = MetricAccumulator(VN)
    with torch.no_grad():
        logits, value = model(batch["board"], batch["global_features"])
        acc.update(logits, value, batch)
    m = acc.compute()
    # Should nearly memorize a handful of examples.
    assert m["policy_top1_optimal_acc"] >= 0.85, m
    assert m["value_raw_mae"] < 1.5, m


def test_training_deterministic_same_seed(small_batch):
    batch, cfg = small_batch

    def final_loss():
        torch.manual_seed(123)
        model = PolicyValueNet(cfg, ModelConfig(channels=16, residual_blocks=1,
                                                value_hidden_size=32))
        opt = torch.optim.AdamW(model.parameters(), lr=5e-3)
        model.train()
        for _ in range(20):
            opt.zero_grad(set_to_none=True)
            logits, value = model(batch["board"], batch["global_features"])
            loss = compute_losses(logits, value, batch)["total"]
            loss.backward(); opt.step()
        return float(loss.item())

    assert final_loss() == pytest.approx(final_loss(), rel=1e-5)
