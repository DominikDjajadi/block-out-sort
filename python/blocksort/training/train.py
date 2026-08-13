"""Supervised training CLI for the policy-value network.

    python -m blocksort.training.train \\
        --dataset data/training/pv_examples.jsonl \\
        --output-dir runs/pv_baseline --seed 42 --epochs 30 --batch-size 128 \\
        --channels 128 --residual-blocks 6 --device auto
"""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import torch
from torch.utils.data import DataLoader

from .checkpoint import save_checkpoint
from .config import (
    MODEL_PROFILES,
    EncodingConfig,
    ModelConfig,
    ValueNormConfig,
    model_profile,
)
from .dataset import PolicyValueDataset, collate_batch, load_records
from .experiment_identity import (
    EVALUATION_SEMANTICS_VERSION, EXPERIMENT_SPEC_FILE,
    TRANSACTION_SCHEMA_VERSION,
    ExperimentSpecIntegrityError, LegacyRunMigrationError,
    UnsupportedResumeError, build_experiment_spec, file_identity,
    hash_canonical_value, runtime_device_provenance,
    validate_continuation_horizon, validate_field_classification,
    validate_or_initialize_experiment)
from .losses import compute_losses
from .metrics import MetricAccumulator, policy_baselines, value_baselines
from .model import PolicyValueNet, count_parameters
from .transaction import (
    atomic_copy, atomic_write_json, relative_to_run, resolve_run_path,
    sha256_file)
from .splits import (
    SplitRatios,
    collect_level_keys,
    filter_records_for_split,
    load_manifest,
    make_split,
    save_manifest,
)


_SUPERVISED_INPUT_FIELDS = {"dataset", "split_manifest"}
_SUPERVISED_OPERATIONAL_FIELDS = {"output_dir", "checkpoint_interval"}
_SUPERVISED_HORIZON_FIELDS = {"epochs"}
_SUPERVISED_DERIVED_FIELDS = {"device"}
_SUPERVISED_UNSUPPORTED_RESUME_FIELDS = {"resume"}
_SUPERVISED_SEMANTIC_FIELDS = {
    "seed", "batch_size", "learning_rate", "weight_decay",
    "policy_loss_weight", "value_loss_weight", "value_loss", "channels",
    "residual_blocks", "value_hidden_size", "model_profile", "normalization",
    "num_workers", "grad_clip",
    "early_stopping_patience", "lr_patience", "train_ratio", "val_ratio",
    "test_ratio", "max_rows", "max_cols", "max_slide_distance", "max_blocks",
    "value_normalization_constant"}


def _supervised_experiment_spec(
    args: argparse.Namespace,
    manifest: dict[str, Any],
    *,
    resolved_device: str | torch.device,
) -> dict[str, Any]:
    values = vars(args)
    validate_field_classification(set(values), {
        "semantic": _SUPERVISED_SEMANTIC_FIELDS,
        "operational": _SUPERVISED_OPERATIONAL_FIELDS,
        "input": _SUPERVISED_INPUT_FIELDS,
        "continuation_horizon": _SUPERVISED_HORIZON_FIELDS,
        "derived": _SUPERVISED_DERIVED_FIELDS,
        "unsupported_on_resume": _SUPERVISED_UNSUPPORTED_RESUME_FIELDS,
    })
    semantic = {
        name: values[name] for name in sorted(_SUPERVISED_SEMANTIC_FIELDS)}
    split_path = (
        Path(args.split_manifest) if args.split_manifest
        else Path(args.output_dir) / "splits.json")
    inputs = {
        "dataset": file_identity(
            args.dataset, kind="policy_value_dataset", count_lines=True,
            format_version=1),
        "split_manifest": {
            "kind": "grouped_split_manifest",
            "path_hint": split_path.as_posix(),
            "sha256": hash_canonical_value(manifest),
            "format_version": manifest.get("version"),
            "level_count": sum(
                len(manifest[name]) for name in (
                    "train_levels", "validation_levels", "test_levels")),
        },
    }
    return build_experiment_spec(
        pipeline="supervised_protagonist", semantic_config=semantic,
        inputs=inputs,
        software_semantics={
            "dataset_schema_version": 1,
            "evaluation_semantics_version": EVALUATION_SEMANTICS_VERSION,
            "split_algorithm_version": manifest.get("version"),
            "transaction_schema_version": TRANSACTION_SCHEMA_VERSION,
            "experiment_identity_version": 1,
            "encoding_block_limit_policy":
                "state_must_not_exceed_checkpoint_v1",
            "loss_aggregation_policy":
                "global_supervision_mass_weighted_v1",
            "runtime": runtime_device_provenance(
                requested_device=args.device,
                resolved_device=resolved_device),
        })


SUPERVISED_RUN_STATE_VERSION = 1


def _supervised_crash_point(stage: str) -> None:
    """Test seam for failures between state commit and mirror refresh."""


def _capture_rng_state(train_loader: DataLoader) -> dict[str, Any]:
    return {
        "python": random.getstate(),
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": (
            torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []),
        "train_loader_generator": (
            train_loader.generator.get_state()
            if train_loader.generator is not None else None),
    }


def _restore_rng_state(
    state: dict[str, Any] | None, train_loader: DataLoader
) -> None:
    if not state:
        raise ExperimentSpecIntegrityError(
            "committed supervised checkpoint is missing continuation RNG state")
    random.setstate(state["python"])
    # ``load_checkpoint(..., map_location="cuda")`` also maps serialized RNG
    # tensors to CUDA.  PyTorch's RNG restoration APIs require CPU byte tensors
    # even when restoring CUDA generator state.
    torch.set_rng_state(state["torch_cpu"].cpu())
    if torch.cuda.is_available() and state.get("torch_cuda"):
        torch.cuda.set_rng_state_all([
            rng_state.cpu() for rng_state in state["torch_cuda"]])
    loader_state = state.get("train_loader_generator")
    if loader_state is not None and train_loader.generator is not None:
        train_loader.generator.set_state(loader_state.cpu())


def _load_supervised_state(root: Path) -> dict[str, Any] | None:
    path = root / "run_state.json"
    if not path.exists():
        return None
    try:
        state = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ExperimentSpecIntegrityError(
            f"cannot read supervised run state: {path}") from exc
    if state.get("schema_version") != SUPERVISED_RUN_STATE_VERSION:
        raise ExperimentSpecIntegrityError(
            "unsupported supervised run-state schema version")
    return state


def _validate_supervised_artifact(
    root: Path, relative: str, expected_sha256: str, label: str
) -> Path:
    path = resolve_run_path(root, relative)
    if not path.is_file():
        raise ExperimentSpecIntegrityError(
            f"committed supervised {label} is missing: {path}")
    observed = sha256_file(path)
    if observed != expected_sha256:
        raise ExperimentSpecIntegrityError(
            f"committed supervised {label} integrity failure: "
            f"expected={expected_sha256}, observed={observed}")
    return path


def _repair_supervised_mirrors(root: Path, state: dict[str, Any]) -> None:
    active = _validate_supervised_artifact(
        root, state["active_checkpoint"], state["active_checkpoint_sha256"],
        "active checkpoint")
    if (not (root / "last.pt").is_file()
            or sha256_file(root / "last.pt")
            != state["active_checkpoint_sha256"]):
        atomic_copy(active, root / "last.pt")
    best_relative = state.get("best_checkpoint")
    if best_relative:
        best = _validate_supervised_artifact(
            root, best_relative, state["best_checkpoint_sha256"],
            "best checkpoint")
        if (not (root / "best.pt").is_file()
                or sha256_file(root / "best.pt")
                != state["best_checkpoint_sha256"]):
            atomic_copy(best, root / "best.pt")
    history = _validate_supervised_artifact(
        root, state["history_path"], state["history_sha256"],
        "history")
    history_document = json.loads(history.read_text(encoding="utf-8"))
    atomic_write_json(root / "history.json", history_document)


def resolve_device(device: str) -> torch.device:
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_dataloaders(
    records: list[dict[str, Any]],
    manifest: dict[str, Any],
    encoding_config: EncodingConfig,
    value_norm: ValueNormConfig,
    *,
    batch_size: int,
    num_workers: int = 0,
    seed: int = 42,
) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for split in ("train", "validation", "test"):
        split_records = filter_records_for_split(records, manifest, split)
        if not split_records:
            out[split] = None
            continue
        ds = PolicyValueDataset(
            split_records, encoding_config=encoding_config, value_norm=value_norm
        )
        shuffle = split == "train"
        generator = torch.Generator().manual_seed(seed) if shuffle else None
        out[split] = DataLoader(
            ds, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers,
            collate_fn=collate_batch, generator=generator,
        )
    return out


def _move_batch(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    moved = dict(batch)
    for key in ("board", "global_features", "legal_action_mask", "policy_target",
                "value_target", "action_regret", "raw_optimal_moves",
                "action_regret_known_mask", "optimal_actions_complete",
                "remaining_blocks", "legal_action_count", "rows", "cols"):
        moved[key] = batch[key].to(device)
    return moved


def _loss_masses(batch: dict[str, Any]) -> tuple[float, float]:
    """Return the policy-valid and scalar-value supervision counts."""
    policy_mass = float(
        (batch["policy_target"].sum(dim=-1) > 0).sum().item())
    value_mass = float(batch["value_target"].numel())
    return policy_mass, value_mass


def _aggregate_policy_value_loss(
    *,
    policy_sum: float,
    policy_mass: float,
    value_sum: float,
    value_mass: float,
    policy_weight: float,
    value_weight: float,
) -> float:
    policy_mean = policy_sum / policy_mass if policy_mass else 0.0
    value_mean = value_sum / value_mass if value_mass else 0.0
    return policy_weight * policy_mean + value_weight * value_mean


@torch.no_grad()
def evaluate_loader(
    model: PolicyValueNet, loader: DataLoader, value_norm: ValueNormConfig,
    device: torch.device, *, policy_weight: float, value_weight: float,
    value_loss_type: str,
) -> tuple[dict[str, float], float]:
    model.eval()
    acc = MetricAccumulator(value_norm)
    policy_sum = policy_mass = 0.0
    value_sum = value_mass = 0.0
    for batch in loader:
        batch = _move_batch(batch, device)
        logits, value = model(batch["board"], batch["global_features"])
        losses = compute_losses(logits, value, batch, policy_weight=policy_weight,
                                value_weight=value_weight, value_loss_type=value_loss_type)
        batch_policy_mass, batch_value_mass = _loss_masses(batch)
        policy_sum += float(losses["policy"].item()) * batch_policy_mass
        policy_mass += batch_policy_mass
        value_sum += float(losses["value"].item()) * batch_value_mass
        value_mass += batch_value_mass
        acc.update(logits, value, batch)
    metrics = acc.compute()
    metrics["loss"] = _aggregate_policy_value_loss(
        policy_sum=policy_sum,
        policy_mass=policy_mass,
        value_sum=value_sum,
        value_mass=value_mass,
        policy_weight=policy_weight,
        value_weight=value_weight,
    )
    return metrics, metrics["loss"]


def model_config_from_args(args: argparse.Namespace) -> ModelConfig:
    """Resolve a named architecture profile plus any explicit overrides."""
    profile = model_profile(args.model_profile)
    return ModelConfig(
        channels=(profile.channels if args.channels is None else args.channels),
        residual_blocks=(
            profile.residual_blocks
            if args.residual_blocks is None else args.residual_blocks),
        value_hidden_size=(
            profile.value_hidden_size
            if args.value_hidden_size is None else args.value_hidden_size),
        normalization=(
            profile.normalization
            if args.normalization is None else args.normalization),
    )


def run_training(args: argparse.Namespace) -> dict[str, Any]:
    output_dir = Path(args.output_dir)
    state = _load_supervised_state(output_dir)
    initialized_markers = (
        "last.pt", "best.pt", "history.json", "checkpoints", "history")
    if state is not None and not getattr(args, "resume", None):
        raise UnsupportedResumeError(
            "Supervised training output directory is already initialized. "
            "Pass --resume pointing to last.pt or the committed active "
            "checkpoint, or use a new output directory.")
    if state is None and getattr(args, "resume", None):
        raise UnsupportedResumeError(
            "Cannot branch an external checkpoint into an uninitialized "
            "supervised output directory. Use a new output directory without "
            "--resume.")
    if (state is None and output_dir.exists()
            and any((output_dir / marker).exists()
                    for marker in initialized_markers)):
        raise LegacyRunMigrationError(
            "Legacy supervised run has no authoritative run_state.json and "
            "cannot be resumed safely. Use a new output directory.")

    encoding_config = EncodingConfig(
        max_rows=args.max_rows, max_cols=args.max_cols,
        max_slide_distance=args.max_slide_distance, max_blocks=args.max_blocks,
    )
    model_config = model_config_from_args(args)
    value_norm = ValueNormConfig(constant=args.value_normalization_constant)
    device = resolve_device(args.device)

    records = load_records(args.dataset)

    # Resolve the split without writing so identity validation precedes all
    # output-directory mutation.
    manifest_path = Path(args.split_manifest) if args.split_manifest \
        else output_dir / "splits.json"
    if manifest_path.exists():
        manifest = load_manifest(manifest_path)
    else:
        ratios = SplitRatios(args.train_ratio, args.val_ratio, args.test_ratio)
        manifest = make_split(collect_level_keys(records), ratios=ratios, seed=args.seed)

    requested_spec = _supervised_experiment_spec(
        args, manifest, resolved_device=device)
    experiment_fingerprint, _ = validate_or_initialize_experiment(
        output_dir, requested_spec, run_state=state)

    completed_epochs = int(state["completed_epochs"]) if state else 0
    validate_continuation_horizon(
        name="epochs", requested=args.epochs, committed=completed_epochs,
        run_dir=output_dir)

    active_path = None
    committed_history: list[dict[str, Any]] = []
    if state is not None:
        active_path = _validate_supervised_artifact(
            output_dir, state["active_checkpoint"],
            state["active_checkpoint_sha256"], "active checkpoint")
        history_path = _validate_supervised_artifact(
            output_dir, state["history_path"], state["history_sha256"],
            "history")
        history_document = json.loads(
            history_path.read_text(encoding="utf-8"))
        committed_history = list(history_document.get("history", []))
        if len(committed_history) != completed_epochs:
            raise ExperimentSpecIntegrityError(
                "committed supervised history length does not match "
                "completed_epochs")
        supplied = Path(args.resume)
        supplied_is_last_mirror = (
            supplied.resolve() == (output_dir / "last.pt").resolve())
        if not supplied_is_last_mirror:
            if not supplied.is_file():
                raise FileNotFoundError(
                    f"supervised resume checkpoint is missing: {supplied}")
            supplied_hash = sha256_file(supplied)
            if supplied_hash != state["active_checkpoint_sha256"]:
                raise ExperimentSpecIntegrityError(
                    f"Cannot resume from checkpoint {supplied}.\n\n"
                    f"This run has committed epoch {completed_epochs} with "
                    f"checkpoint:\n  {state['active_checkpoint']}\n  "
                    f"sha256={state['active_checkpoint_sha256']}\n\n"
                    "Resuming an earlier checkpoint would branch or rewind "
                    "the run. Use a new output directory to start a branch.")

    output_dir.mkdir(parents=True, exist_ok=True)
    seed_everything(args.seed)
    if not manifest_path.exists():
        save_manifest(manifest, manifest_path)
    config_document = {
        key: (str(value) if isinstance(value, Path) else value)
        for key, value in vars(args).items()
        if key != "resume"}
    config_document["target_epochs"] = args.epochs
    config_document["resolved_model_config"] = model_config.to_dict()
    config_document["runtime"] = requested_spec[
        "software_semantics"]["runtime"]
    atomic_write_json(output_dir / "config.json", config_document)

    loaders = build_dataloaders(
        records, manifest, encoding_config, value_norm,
        batch_size=args.batch_size, num_workers=args.num_workers, seed=args.seed,
    )
    train_loader = loaders["train"]
    val_loader = loaders["validation"]
    if train_loader is None:
        raise ValueError("no training records after splitting")

    train_items = train_loader.dataset.items
    baselines = {
        **policy_baselines(train_items, encoding_config),
        **value_baselines(train_items, value_norm)}
    model = PolicyValueNet(encoding_config, model_config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate,
                                  weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=max(1, args.lr_patience))

    start_epoch = completed_epochs + 1
    last_completed_epoch = completed_epochs
    best_metric = float("inf")
    best_epoch = -1
    patience_left = args.early_stopping_patience
    if state is not None:
        from .checkpoint import load_checkpoint
        ckpt = load_checkpoint(active_path, map_location=device)
        model.load_state_dict(ckpt["model_state"])
        if ckpt.get("optimizer_state"):
            optimizer.load_state_dict(ckpt["optimizer_state"])
        if ckpt.get("scheduler_state"):
            scheduler.load_state_dict(ckpt["scheduler_state"])
        last_completed_epoch = int(ckpt.get("epoch", 0))
        if last_completed_epoch != completed_epochs:
            raise ExperimentSpecIntegrityError(
                "active checkpoint epoch does not match committed run state")
        start_epoch = last_completed_epoch + 1

        prior_best = ckpt.get("best_val_metric")
        prior_best_epoch = ckpt.get("best_epoch")
        prior_patience = ckpt.get("patience_left")

        if prior_best is not None:
            best_metric = float(prior_best)
        if prior_best_epoch is not None:
            best_epoch = int(prior_best_epoch)
        if prior_patience is not None:
            patience_left = int(prior_patience)
        _restore_rng_state(ckpt.get("rng_state"), train_loader)

    split_identity = {"seed": manifest["seed"], "group_key": manifest["group_key"],
                      "n_train": len(manifest["train_levels"]),
                      "n_val": len(manifest["validation_levels"]),
                      "n_test": len(manifest["test_levels"])}

    history = committed_history

    def summary_document() -> dict[str, Any]:
        return {
            "best_epoch": best_epoch,
            "best_metric": (
                None if best_metric == float("inf") else best_metric),
            "experiment_fingerprint": experiment_fingerprint,
            "history": history,
            "baselines": baselines,
            "last_completed_epoch": last_completed_epoch,
            "parameters": count_parameters(model),
            "device": str(device),
            "target_epochs": args.epochs,
        }

    if state is None:
        initial_checkpoint = output_dir / "checkpoints" / "epoch_000.pt"
        save_checkpoint(
            initial_checkpoint, model=model, optimizer=optimizer,
            scheduler=scheduler, epoch=0, best_val_metric=None,
            encoding_config=encoding_config, model_config=model_config,
            value_norm=value_norm, seed=args.seed, dataset_version=1,
            split_identity=split_identity, best_epoch=None,
            patience_left=patience_left,
            experiment_fingerprint=experiment_fingerprint,
            rng_state=_capture_rng_state(train_loader))
        history_path = output_dir / "history" / "committed_epoch_000.json"
        atomic_write_json(history_path, summary_document())
        state = {
            "schema_version": SUPERVISED_RUN_STATE_VERSION,
            "experiment_fingerprint": experiment_fingerprint,
            "completed_epochs": 0,
            "target_epochs": args.epochs,
            "active_checkpoint": relative_to_run(
                initial_checkpoint, output_dir),
            "active_checkpoint_sha256": sha256_file(initial_checkpoint),
            "best_checkpoint": None,
            "best_checkpoint_sha256": None,
            "history_path": relative_to_run(history_path, output_dir),
            "history_sha256": sha256_file(history_path),
            "early_stopped": False,
        }
        atomic_write_json(output_dir / "run_state.json", state)
        _supervised_crash_point("after_state_commit")
        _repair_supervised_mirrors(output_dir, state)
    else:
        _repair_supervised_mirrors(output_dir, state)

    if state.get("early_stopped"):
        return summary_document()

    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        train_policy_sum = train_policy_mass = 0.0
        train_value_sum = train_value_mass = 0.0
        for batch in train_loader:
            batch = _move_batch(batch, device)
            optimizer.zero_grad(set_to_none=True)
            logits, value = model(batch["board"], batch["global_features"])
            losses = compute_losses(
                logits, value, batch, policy_weight=args.policy_loss_weight,
                value_weight=args.value_loss_weight, value_loss_type=args.value_loss)
            losses["total"].backward()
            if args.grad_clip and args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            batch_policy_mass, batch_value_mass = _loss_masses(batch)
            train_policy_sum += (
                float(losses["policy"].item()) * batch_policy_mass)
            train_policy_mass += batch_policy_mass
            train_value_sum += (
                float(losses["value"].item()) * batch_value_mass)
            train_value_mass += batch_value_mass
        train_loss = _aggregate_policy_value_loss(
            policy_sum=train_policy_sum,
            policy_mass=train_policy_mass,
            value_sum=train_value_sum,
            value_mass=train_value_mass,
            policy_weight=args.policy_loss_weight,
            value_weight=args.value_loss_weight,
        )

        # Selection / scheduling uses validation only (fallback to train).
        if val_loader is not None:
            val_metrics, val_loss = evaluate_loader(
                model, val_loader, value_norm, device,
                policy_weight=args.policy_loss_weight,
                value_weight=args.value_loss_weight, value_loss_type=args.value_loss)
            selection_loss = val_loss
        else:
            val_metrics, selection_loss = {}, train_loss
        scheduler.step(selection_loss)

        epoch_row = {"epoch": epoch, "train_loss": train_loss,
                     "val_loss": val_metrics.get("loss"),
                     "val_policy_top1": val_metrics.get("policy_top1_optimal_acc"),
                     "val_value_raw_mae": val_metrics.get("value_raw_mae"),
                     "lr": optimizer.param_groups[0]["lr"]}
        history.append(epoch_row)
        last_completed_epoch = epoch
        print(json.dumps(epoch_row))

        improved = selection_loss < best_metric - 1e-6
        if improved:
            best_metric = selection_loss
            best_epoch = epoch
            patience_left = args.early_stopping_patience
        else:
            patience_left -= 1

        early_stopped = (
            args.early_stopping_patience > 0 and patience_left <= 0)
        committed_checkpoint = (
            output_dir / "checkpoints" / f"epoch_{epoch:03d}.pt")
        save_checkpoint(
            committed_checkpoint, model=model, optimizer=optimizer,
            scheduler=scheduler, epoch=epoch, best_val_metric=best_metric,
            encoding_config=encoding_config, model_config=model_config,
            value_norm=value_norm, seed=args.seed, dataset_version=1,
            split_identity=split_identity, metrics=val_metrics,
            best_epoch=best_epoch, patience_left=patience_left,
            experiment_fingerprint=experiment_fingerprint,
            rng_state=_capture_rng_state(train_loader))
        history_path = (
            output_dir / "history" / f"committed_epoch_{epoch:03d}.json")
        atomic_write_json(history_path, summary_document())
        pending_state = dict(state)
        pending_state.update({
            "completed_epochs": epoch,
            "target_epochs": args.epochs,
            "active_checkpoint": relative_to_run(
                committed_checkpoint, output_dir),
            "active_checkpoint_sha256": sha256_file(committed_checkpoint),
            "history_path": relative_to_run(history_path, output_dir),
            "history_sha256": sha256_file(history_path),
            "early_stopped": early_stopped,
        })
        if improved:
            pending_state["best_checkpoint"] = relative_to_run(
                committed_checkpoint, output_dir)
            pending_state["best_checkpoint_sha256"] = \
                pending_state["active_checkpoint_sha256"]
        atomic_write_json(output_dir / "run_state.json", pending_state)
        state = pending_state
        _supervised_crash_point("after_state_commit")
        _repair_supervised_mirrors(output_dir, state)
        if args.checkpoint_interval and epoch % args.checkpoint_interval == 0:
            atomic_copy(committed_checkpoint, output_dir / f"epoch_{epoch}.pt")

        if early_stopped:
            print(json.dumps({"early_stop": True, "epoch": epoch}))
            break

    return summary_document()


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Train the policy-value network.")
    p.add_argument("--dataset", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--split-manifest", default=None)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--epochs", type=int, default=30,
        help="total target epoch count; resume trains only missing epochs")
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--learning-rate", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--policy-loss-weight", type=float, default=1.0)
    p.add_argument("--value-loss-weight", type=float, default=1.0)
    p.add_argument("--value-loss", default="huber", choices=["huber", "mse"])
    p.add_argument(
        "--model-profile",
        choices=sorted(MODEL_PROFILES),
        default="small_groupnorm",
        help="architecture preset for new checkpoints (individual model flags "
             "override the selected profile)",
    )
    p.add_argument("--channels", type=int, default=None)
    p.add_argument("--residual-blocks", type=int, default=None)
    p.add_argument("--value-hidden-size", type=int, default=None)
    p.add_argument(
        "--normalization",
        choices=("batch_norm", "group_norm"),
        default=None,
    )
    p.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--checkpoint-interval", type=int, default=0)
    p.add_argument("--early-stopping-patience", type=int, default=0)
    p.add_argument("--lr-patience", type=int, default=3)
    p.add_argument("--train-ratio", type=float, default=0.8)
    p.add_argument("--val-ratio", type=float, default=0.1)
    p.add_argument("--test-ratio", type=float, default=0.1)
    p.add_argument("--max-rows", type=int, default=8)
    p.add_argument("--max-cols", type=int, default=8)
    p.add_argument("--max-slide-distance", type=int, default=8)
    p.add_argument("--max-blocks", type=int, default=16)
    p.add_argument("--value-normalization-constant", type=float, default=20.0)
    p.add_argument("--resume", default=None, help="checkpoint path to resume from")
    return p


def main(argv: Optional[list[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    run_training(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
