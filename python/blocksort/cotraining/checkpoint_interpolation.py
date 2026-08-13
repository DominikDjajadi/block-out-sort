"""Create checkpoints partway between an incumbent and a trained candidate.

This is a post-training diagnostic.  It performs no optimization and evaluates
nothing; it only linearly scales the candidate's floating-point tensor update.
Non-floating buffers (for example BatchNorm's batch counter) are copied from
the candidate because they cannot be meaningfully interpolated.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch

from ..training.checkpoint import (
    configs_from_checkpoint,
    load_checkpoint,
    model_from_checkpoint,
    save_checkpoint,
)
from ..training.transaction import atomic_write_json, sha256_file


SCHEMA_VERSION = 1
SEMANTICS = "checkpoint_update_interpolation_v1"


@dataclass(frozen=True)
class CheckpointInterpolationConfig:
    incumbent_checkpoint: str
    candidate_checkpoint: str
    output_dir: str
    fractions: tuple[float, ...] = (0.25, 0.5, 0.75)
    required_changed_prefixes: tuple[str, ...] = ()

    def validate(self) -> None:
        for label, value in (
                ("incumbent checkpoint", self.incumbent_checkpoint),
                ("candidate checkpoint", self.candidate_checkpoint)):
            if not Path(value).is_file():
                raise ValueError(f"{label} does not exist: {value}")
        if not self.fractions:
            raise ValueError("at least one interpolation fraction is required")
        if len(set(self.fractions)) != len(self.fractions):
            raise ValueError("interpolation fractions must not contain duplicates")
        if any(
                isinstance(value, bool)
                or not math.isfinite(value)
                or not 0 < value <= 1
                for value in self.fractions):
            raise ValueError("interpolation fractions must be in (0, 1]")
        if any(not prefix for prefix in self.required_changed_prefixes):
            raise ValueError("required changed prefixes must not be empty")


def _canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"),
        ensure_ascii=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _fraction_name(fraction: float) -> str:
    return "fraction_" + format(fraction, ".12g").replace(".", "p")


def _validate_compatible_checkpoints(
    incumbent: dict[str, Any],
    candidate: dict[str, Any],
) -> None:
    for field in (
            "checkpoint_version", "color_order", "direction_order"):
        if incumbent.get(field) != candidate.get(field):
            raise ValueError(
                f"checkpoint field differs and cannot be interpolated: {field}")
    incumbent_configs = configs_from_checkpoint(incumbent)
    candidate_configs = configs_from_checkpoint(candidate)
    for field, incumbent_config, candidate_config in zip(
            ("encoding_config", "model_config", "value_norm"),
            incumbent_configs, candidate_configs):
        if incumbent_config != candidate_config:
            raise ValueError(
                f"checkpoint field differs and cannot be interpolated: {field}")
    incumbent_state = incumbent.get("model_state")
    candidate_state = candidate.get("model_state")
    if not isinstance(incumbent_state, dict) \
            or not isinstance(candidate_state, dict):
        raise ValueError("both checkpoints must contain model_state dictionaries")
    if incumbent_state.keys() != candidate_state.keys():
        raise ValueError("checkpoint model_state keys differ")
    for name in incumbent_state:
        left = incumbent_state[name]
        right = candidate_state[name]
        if not torch.is_tensor(left) or not torch.is_tensor(right):
            raise ValueError(f"model_state entry is not a tensor: {name}")
        if left.shape != right.shape:
            raise ValueError(f"model_state tensor shape differs: {name}")
        if left.dtype != right.dtype:
            raise ValueError(f"model_state tensor dtype differs: {name}")


def _changed_tensor_names(
    incumbent_state: dict[str, torch.Tensor],
    candidate_state: dict[str, torch.Tensor],
) -> list[str]:
    return [
        name for name in incumbent_state
        if not torch.equal(incumbent_state[name], candidate_state[name])
    ]


def _interpolate_state_dict(
    incumbent_state: dict[str, torch.Tensor],
    candidate_state: dict[str, torch.Tensor],
    fraction: float,
) -> dict[str, torch.Tensor]:
    result: dict[str, torch.Tensor] = {}
    for name, incumbent_tensor in incumbent_state.items():
        candidate_tensor = candidate_state[name]
        if torch.is_floating_point(incumbent_tensor) \
                or torch.is_complex(incumbent_tensor):
            result[name] = torch.lerp(
                incumbent_tensor, candidate_tensor, fraction)
        else:
            result[name] = candidate_tensor.clone()
    return result


def _identity(cfg: CheckpointInterpolationConfig) -> dict[str, Any]:
    # Normalize tuples to their persisted JSON representation so an interrupted
    # run can compare its in-memory identity with experiment.json exactly.
    config = json.loads(json.dumps(asdict(cfg)))
    config.pop("output_dir")
    result = {
        "schema_version": SCHEMA_VERSION,
        "semantics": SEMANTICS,
        "config": config,
        "inputs": {
            "incumbent_checkpoint_sha256":
                sha256_file(cfg.incumbent_checkpoint),
            "candidate_checkpoint_sha256":
                sha256_file(cfg.candidate_checkpoint),
        },
    }
    result["fingerprint"] = _canonical_sha256(result)
    return result


def run_checkpoint_interpolation(
    cfg: CheckpointInterpolationConfig,
) -> dict[str, Any]:
    cfg.validate()
    incumbent = load_checkpoint(
        cfg.incumbent_checkpoint, map_location="cpu")
    candidate = load_checkpoint(
        cfg.candidate_checkpoint, map_location="cpu")
    _validate_compatible_checkpoints(incumbent, candidate)

    incumbent_state = incumbent["model_state"]
    candidate_state = candidate["model_state"]
    changed = _changed_tensor_names(incumbent_state, candidate_state)
    if not changed:
        raise ValueError("candidate checkpoint contains no model update")
    unexpected = [
        name for name in changed
        if cfg.required_changed_prefixes
        and not any(
            name.startswith(prefix)
            for prefix in cfg.required_changed_prefixes)
    ]
    if unexpected:
        raise ValueError(
            "candidate changed tensors outside required prefixes: "
            + ", ".join(unexpected))

    root = Path(cfg.output_dir)
    root.mkdir(parents=True, exist_ok=True)
    identity = _identity(cfg)
    identity_path = root / "experiment.json"
    if identity_path.exists():
        persisted = json.loads(identity_path.read_text(encoding="utf-8"))
        if persisted != identity:
            raise RuntimeError(
                "interpolation output belongs to different inputs or settings")
    else:
        atomic_write_json(identity_path, identity)

    encoding, model_config, value_norm = configs_from_checkpoint(incumbent)
    outputs = []
    for fraction in cfg.fractions:
        name = _fraction_name(fraction)
        checkpoint_path = root / f"{name}.pt"
        model = model_from_checkpoint(incumbent, map_location="cpu")
        interpolated_state = _interpolate_state_dict(
            incumbent_state, candidate_state, fraction)
        model.load_state_dict(interpolated_state)
        provenance = {
            "kind": SEMANTICS,
            "fraction": fraction,
            "incumbent_checkpoint_sha256":
                identity["inputs"]["incumbent_checkpoint_sha256"],
            "candidate_checkpoint_sha256":
                identity["inputs"]["candidate_checkpoint_sha256"],
            "changed_tensors": changed,
            "required_changed_prefixes":
                list(cfg.required_changed_prefixes),
        }
        save_checkpoint(
            checkpoint_path,
            model=model,
            optimizer=None,
            scheduler=None,
            epoch=int(candidate.get("epoch", 0)),
            best_val_metric=candidate.get("best_val_metric"),
            encoding_config=encoding,
            model_config=model_config,
            value_norm=value_norm,
            seed=int(candidate.get("seed", 0)),
            dataset_version=int(candidate.get("dataset_version", 1)),
            split_identity=provenance,
            metrics={"checkpoint_interpolation": provenance},
            experiment_fingerprint=identity["fingerprint"],
        )
        outputs.append({
            "name": name,
            "fraction": fraction,
            "checkpoint": str(checkpoint_path),
            "checkpoint_sha256": sha256_file(checkpoint_path),
        })

    summary = {
        **identity,
        "changed_tensor_count": len(changed),
        "changed_tensors": changed,
        "outputs": outputs,
    }
    atomic_write_json(root / "summary.json", summary)
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Create post-training checkpoints containing fractions of one "
            "candidate update."))
    parser.add_argument("--incumbent-checkpoint", required=True)
    parser.add_argument("--candidate-checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--fraction", type=float, action="append", default=None,
        help="update fraction in (0, 1]; repeat for multiple checkpoints")
    parser.add_argument(
        "--require-changed-prefix", action="append", default=None,
        help=(
            "reject the candidate if any changed tensor is outside one of "
            "these prefixes"))
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    cfg = CheckpointInterpolationConfig(
        incumbent_checkpoint=args.incumbent_checkpoint,
        candidate_checkpoint=args.candidate_checkpoint,
        output_dir=args.output_dir,
        fractions=tuple(args.fraction or (0.25, 0.5, 0.75)),
        required_changed_prefixes=tuple(
            args.require_changed_prefix or ()),
    )
    result = run_checkpoint_interpolation(cfg)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
