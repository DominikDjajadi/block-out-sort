"""Checkpoint save/load that bundles all configuration needed for inference.

A checkpoint never stores a bare ``state_dict``: it also carries the model,
encoding, and value-normalization configs, the fixed color/direction orderings,
the dataset/split identity, and provenance, so a model can be rebuilt and used
without re-entering any architectural parameters.
"""

from __future__ import annotations

import subprocess
import os
import tempfile
from pathlib import Path
from typing import Any

import torch

from .config import (
    COLOR_ORDER,
    DIRECTION_ORDER,
    EncodingConfig,
    ModelConfig,
    ValueNormConfig,
)
from .model import PolicyValueNet

CHECKPOINT_VERSION = 1


def _git_commit() -> str | None:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, timeout=5,
        )
        if out.returncode == 0:
            return out.stdout.strip()
    except Exception:  # noqa: BLE001
        pass
    return None


def save_checkpoint(
    path: str | Path,
    *,
    model: PolicyValueNet,
    optimizer: torch.optim.Optimizer | None,
    scheduler: Any | None,
    epoch: int,
    best_val_metric: float | None,
    encoding_config: EncodingConfig,
    model_config: ModelConfig,
    value_norm: ValueNormConfig,
    seed: int,
    dataset_version: int,
    split_identity: dict[str, Any] | None,
    metrics: dict[str, Any] | None = None,
    best_epoch: int | None = None,
    patience_left: int | None = None,
    experiment_fingerprint: str | None = None,
    rng_state: dict[str, Any] | None = None,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "checkpoint_version": CHECKPOINT_VERSION,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict() if optimizer else None,
        "scheduler_state": scheduler.state_dict() if scheduler else None,
        "epoch": epoch,
        "best_val_metric": best_val_metric,
        "best_epoch": best_epoch,
        "patience_left": patience_left,
        "encoding_config": encoding_config.to_dict(),
        "model_config": model_config.to_dict(),
        "value_norm": value_norm.to_dict(),
        "color_order": list(COLOR_ORDER),
        "direction_order": list(DIRECTION_ORDER),
        "seed": seed,
        "dataset_version": dataset_version,
        "split_identity": split_identity,
        "experiment_fingerprint": experiment_fingerprint,
        "rng_state": rng_state,
        "metrics": metrics or {},
        "torch_version": torch.__version__,
        "git_commit": _git_commit(),
    }
    fd, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    os.close(fd)
    temporary = Path(temporary_name)
    try:
        torch.save(payload, temporary)
        with temporary.open("rb+") as handle:
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def load_checkpoint(path: str | Path, map_location: Any = "cpu") -> dict[str, Any]:
    ckpt = torch.load(path, map_location=map_location, weights_only=False)
    if ckpt.get("checkpoint_version") != CHECKPOINT_VERSION:
        raise ValueError(
            f"unsupported checkpoint version {ckpt.get('checkpoint_version')}"
        )
    # Guard against silently pairing a model with a mismatched encoding.
    if ckpt.get("color_order") != list(COLOR_ORDER):
        raise ValueError("checkpoint color_order does not match current COLOR_ORDER")
    if ckpt.get("direction_order") != list(DIRECTION_ORDER):
        raise ValueError("checkpoint direction_order does not match DIRECTION_ORDER")
    return ckpt


def configs_from_checkpoint(
    ckpt: dict[str, Any],
) -> tuple[EncodingConfig, ModelConfig, ValueNormConfig]:
    return (
        EncodingConfig.from_dict(ckpt["encoding_config"]),
        ModelConfig.from_dict(ckpt["model_config"]),
        ValueNormConfig.from_dict(ckpt["value_norm"]),
    )


def model_from_checkpoint(
    ckpt: dict[str, Any], map_location: Any = "cpu"
) -> PolicyValueNet:
    enc, model_cfg, _ = configs_from_checkpoint(ckpt)
    model = PolicyValueNet(enc, model_cfg)
    model.load_state_dict(ckpt["model_state"])
    model.to(map_location)
    model.eval()
    return model
