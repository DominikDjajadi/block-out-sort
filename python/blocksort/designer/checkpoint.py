"""Designer checkpoint save/load (bundles encoding + model configs + metadata)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from ..training.config import EncodingConfig
from .model import DesignerModelConfig, DesignerNet

DESIGNER_CHECKPOINT_VERSION = 2
DESIGNER_ENCODING_SEMANTICS = "causal_structural_globals_v2"


def save_designer(path: str | Path, *, model: DesignerNet,
                  encoding_config: EncodingConfig,
                  model_config: DesignerModelConfig, seed: int,
                  metadata: dict[str, Any] | None = None) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "designer_checkpoint_version": DESIGNER_CHECKPOINT_VERSION,
        "encoding_semantics": DESIGNER_ENCODING_SEMANTICS,
        "model_state": model.state_dict(),
        "encoding_config": encoding_config.to_dict(),
        "model_config": model_config.to_dict(),
        "seed": seed,
        "metadata": metadata or {},
        "torch_version": torch.__version__,
    }, path)


def load_designer(path: str | Path, map_location: Any = "cpu") -> dict[str, Any]:
    ckpt = torch.load(path, map_location=map_location, weights_only=False)
    if ckpt.get("designer_checkpoint_version") not in (1, 2):
        raise ValueError("unsupported designer checkpoint version")
    if (ckpt.get("designer_checkpoint_version") == 2
            and ckpt.get("encoding_semantics") != DESIGNER_ENCODING_SEMANTICS):
        raise ValueError("unsupported designer encoding semantics")
    return ckpt


def designer_state_dict_from_checkpoint(
    ckpt: dict[str, Any],
) -> dict[str, torch.Tensor]:
    """Return model weights migrated to the current global-feature semantics."""
    state = dict(ckpt["model_state"])
    if ckpt.get("designer_checkpoint_version") == 1:
        # The last two v1 global inputs were always zero in normal rollouts, so
        # their input weights are untrained initialization noise. Zeroing them
        # makes legacy behavior stable while allowing v2 fine-tuning to learn
        # the replacement causal structural signals.
        for name in ("stop_head.0.weight", "value_head.0.weight"):
            weight = state[name].clone()
            weight[:, -2:] = 0
            state[name] = weight
    return state


def designer_from_checkpoint(ckpt: dict[str, Any], map_location: Any = "cpu"
                             ) -> tuple[DesignerNet, EncodingConfig,
                                        DesignerModelConfig]:
    enc = EncodingConfig.from_dict(ckpt["encoding_config"])
    model_cfg = DesignerModelConfig.from_dict(ckpt["model_config"])
    model = DesignerNet(enc, model_cfg)
    model.load_state_dict(designer_state_dict_from_checkpoint(ckpt))
    model.to(map_location)
    model.eval()
    return model, enc, model_cfg
