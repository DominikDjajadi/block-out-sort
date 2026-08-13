from __future__ import annotations

import json

import pytest
import torch

from blocksort.cotraining import checkpoint_interpolation


def test_interpolate_state_dict_scales_floating_update():
    incumbent = {
        "weight": torch.tensor([1.0, 3.0]),
        "counter": torch.tensor(2, dtype=torch.long),
    }
    candidate = {
        "weight": torch.tensor([5.0, 7.0]),
        "counter": torch.tensor(3, dtype=torch.long),
    }

    result = checkpoint_interpolation._interpolate_state_dict(
        incumbent, candidate, 0.25)

    assert torch.equal(result["weight"], torch.tensor([2.0, 4.0]))
    assert torch.equal(result["counter"], candidate["counter"])


@pytest.mark.parametrize("fraction", [0.0, -0.1, 1.1, float("nan")])
def test_config_rejects_invalid_fraction(tmp_path, fraction):
    incumbent = tmp_path / "incumbent.pt"
    candidate = tmp_path / "candidate.pt"
    incumbent.touch()
    candidate.touch()
    cfg = checkpoint_interpolation.CheckpointInterpolationConfig(
        incumbent_checkpoint=str(incumbent),
        candidate_checkpoint=str(candidate),
        output_dir=str(tmp_path / "out"),
        fractions=(fraction,),
    )

    with pytest.raises(ValueError, match="fractions"):
        cfg.validate()


def test_compatible_checkpoint_validation_rejects_config_mismatch():
    incumbent = {
        "checkpoint_version": 1,
        "encoding_config": {
            "max_rows": 6,
            "max_cols": 6,
            "max_slide_distance": 5,
            "max_blocks": 16,
        },
        "model_config": {
            "channels": 128,
            "residual_blocks": 6,
            "value_hidden_size": 256,
        },
        "value_norm": {},
        "color_order": [],
        "direction_order": [],
        "model_state": {"weight": torch.tensor([1.0])},
    }
    candidate = {
        **incumbent,
        "encoding_config": {
            **incumbent["encoding_config"],
            "max_rows": 7,
        },
    }

    with pytest.raises(ValueError, match="encoding_config"):
        checkpoint_interpolation._validate_compatible_checkpoints(
            incumbent, candidate)


def test_compatible_checkpoint_validation_accepts_explicit_legacy_default():
    incumbent = {
        "checkpoint_version": 1,
        "encoding_config": {
            "max_rows": 6,
            "max_cols": 6,
            "max_slide_distance": 5,
            "max_blocks": 16,
        },
        "model_config": {
            "channels": 128,
            "residual_blocks": 6,
            "value_hidden_size": 256,
        },
        "value_norm": {},
        "color_order": [],
        "direction_order": [],
        "model_state": {"weight": torch.tensor([1.0])},
    }
    candidate = {
        **incumbent,
        "model_config": {
            **incumbent["model_config"],
            "normalization": "batch_norm",
        },
    }

    checkpoint_interpolation._validate_compatible_checkpoints(
        incumbent, candidate)


def test_identity_uses_json_normalized_config(tmp_path):
    incumbent = tmp_path / "incumbent.pt"
    candidate = tmp_path / "candidate.pt"
    incumbent.write_bytes(b"incumbent")
    candidate.write_bytes(b"candidate")
    cfg = checkpoint_interpolation.CheckpointInterpolationConfig(
        incumbent_checkpoint=str(incumbent),
        candidate_checkpoint=str(candidate),
        output_dir=str(tmp_path / "out"),
        fractions=(0.25, 0.5),
        required_changed_prefixes=("policy_conv.",),
    )

    identity = checkpoint_interpolation._identity(cfg)

    assert identity == json.loads(json.dumps(identity))
