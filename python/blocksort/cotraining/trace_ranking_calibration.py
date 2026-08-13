"""Calibrate trace-pair loss weights without evaluating solver outcomes."""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

from ..expert_iteration.train import (
    ExpertDataset,
    collate,
    configure_trainable_part,
    source_weights_for,
    trace_pairwise_hinge_loss,
    value_supervision_weights_for,
)
from ..training.checkpoint import (
    configs_from_checkpoint,
    load_checkpoint,
    model_from_checkpoint,
)
from ..training.dataset import load_records
from ..training.experiment_identity import hash_canonical_value
from ..training.losses import masked_log_softmax
from ..training.transaction import atomic_write_json, sha256_file
from .policy_distillation_sweep import (
    _canonical_jsonl,
    _condition_records,
    _load_json,
    _resolve_device,
    _sha_bytes,
    reconstruct_source_sample,
)


SCHEMA_VERSION = 1
SEMANTICS = "trace_pair_ranking_gradient_calibration_v1"
FIXED_MARGIN = 0.05
DIAGNOSTIC_MARGINS = (0.0, 0.05, 0.1, 0.25)
TARGET_GRADIENT_RATIOS = {"control": 0.0, "light": 0.05, "moderate": 0.2}


@dataclass(frozen=True)
class TraceRankingCalibrationConfig:
    checkpoint: str
    trace_dataset: str
    source_run: str
    replay_snapshot: str
    output_dir: str
    source_round: int = 1
    device: str = "cuda"

    def validate(self) -> None:
        for label, raw in (
                ("checkpoint", self.checkpoint),
                ("trace dataset", self.trace_dataset),
                ("replay snapshot", self.replay_snapshot)):
            if not Path(raw).is_file():
                raise ValueError(f"{label} does not exist: {raw}")
        if not Path(self.source_run).is_dir():
            raise ValueError("source run does not exist")
        if self.source_round <= 0:
            raise ValueError("source round must be positive")


def _gradient_vector(parameters: list[torch.nn.Parameter]) -> torch.Tensor:
    pieces = []
    for parameter in parameters:
        if parameter.grad is None:
            pieces.append(torch.zeros_like(parameter).reshape(-1))
        else:
            pieces.append(parameter.grad.detach().reshape(-1))
    return torch.cat(pieces) if pieces else torch.zeros(0)


def _parameter_group_norms(
    model,
    trainable_names: set[str],
) -> dict[str, float]:
    totals: dict[str, float] = {}
    for name, parameter in model.named_parameters():
        if name not in trainable_names or parameter.grad is None:
            continue
        group = name.split(".", 1)[0]
        totals[group] = totals.get(group, 0.0) + float(
            parameter.grad.detach().pow(2).sum().item())
    return {
        group: math.sqrt(total)
        for group, total in sorted(totals.items())
    }


def _policy_gradient(
    model,
    dataset: ExpertDataset,
    *,
    batch_size: int,
    device: torch.device,
) -> tuple[float, torch.Tensor, dict[str, float]]:
    parameters = [
        parameter for parameter in model.parameters()
        if parameter.requires_grad]
    names = {
        name for name, parameter in model.named_parameters()
        if parameter.requires_grad}
    model.zero_grad(set_to_none=True)
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False, collate_fn=collate)
    total_mass = sum(float(item["weight"]) for item in dataset.items)
    loss_total = 0.0
    for batch in loader:
        board = batch["board"].to(device)
        glob = batch["global_features"].to(device)
        mask = batch["legal_action_mask"].to(device)
        target = batch["policy_target"].to(device)
        weight = batch["weight"].to(device)
        logits, _ = model(board, glob)
        per_policy = -(target * masked_log_softmax(logits, mask)).sum(dim=-1)
        contribution = (per_policy * weight).sum() / total_mass
        contribution.backward()
        loss_total += float(contribution.detach().item())
    vector = _gradient_vector(parameters)
    return loss_total, vector, _parameter_group_norms(model, names)


def _trace_gradient(
    model,
    dataset: ExpertDataset,
    *,
    margin: float,
    device: torch.device,
) -> tuple[dict[str, Any], torch.Tensor, dict[str, float]]:
    parameters = [
        parameter for parameter in model.parameters()
        if parameter.requires_grad]
    names = {
        name for name, parameter in model.named_parameters()
        if parameter.requires_grad}
    model.zero_grad(set_to_none=True)
    batch = collate(dataset.items)
    board = batch["board"].to(device)
    glob = batch["global_features"].to(device)
    preferred = batch["trace_preferred_index"].to(device)
    competing = batch["trace_competing_index"].to(device)
    logits, _ = model(board, glob)
    preferred_logits = logits.gather(1, preferred[:, None]).squeeze(1)
    competing_logits = logits.gather(1, competing[:, None]).squeeze(1)
    gaps = preferred_logits - competing_logits
    per_example = trace_pairwise_hinge_loss(
        logits, preferred, competing, margin=margin)
    loss = per_example.mean()
    loss.backward()
    vector = _gradient_vector(parameters)
    report = {
        "margin": margin,
        "mean_loss": float(loss.detach().item()),
        "active_example_count": int((gaps < margin).sum().item()),
        "example_count": len(dataset),
        "preferred_minus_competing_logit_gaps": [
            float(value) for value in gaps.detach().cpu().tolist()],
        "per_example_hinge_losses": [
            float(value) for value in per_example.detach().cpu().tolist()],
    }
    return report, vector, _parameter_group_norms(model, names)


def _rounded_weight(value: float) -> float:
    return float(f"{value:.6g}")


def run_trace_ranking_calibration(
    cfg: TraceRankingCalibrationConfig,
) -> dict[str, Any]:
    cfg.validate()
    source = Path(cfg.source_run)
    source_config = _load_json(source / "config.json")
    source_round = source / f"round_{cfg.source_round:03d}"
    manifest_path = source_round / "training_sample_manifest.json"
    manifest = _load_json(manifest_path)
    checkpoint = load_checkpoint(cfg.checkpoint, map_location="cpu")
    encoding, _model_config, value_norm = configs_from_checkpoint(checkpoint)
    device = _resolve_device(cfg.device)
    model = model_from_checkpoint(checkpoint, map_location=device)
    trainable = configure_trainable_part(
        model, str(source_config["trainable_part"]))
    model.eval()

    replay_records = load_records(cfg.replay_snapshot)
    raw_source = reconstruct_source_sample(replay_records, manifest)
    conditioned = _condition_records(
        raw_source,
        champion_model=model,
        encoding=encoding,
        value_norm=value_norm,
        device=device,
        batch_size=int(source_config["batch_size"]),
        champion_sha256=sha256_file(cfg.checkpoint),
    )
    if _sha_bytes(_canonical_jsonl(conditioned)) != manifest["sample"]["sha256"]:
        raise RuntimeError("conditioned calibration sample hash mismatch")
    weights = source_weights_for(
        conditioned,
        cfg.source_round,
        weight_exact_historical=float(
            source_config["weight_exact_historical"]),
        weight_exact_new=float(source_config["weight_exact_new"]),
        weight_search=float(source_config["weight_search"]),
        exact_path_policy_confidence=float(
            source_config["exact_path_policy_confidence"]),
    )
    effective_value_weights = value_supervision_weights_for(
        conditioned,
        weights,
        search_value_loss_weight=float(
            source_config["search_value_loss_weight"]),
    )
    persisted_weights = _load_json(
        source_round / "training_policy_weights.json")
    persisted_effective = _load_json(
        source_round / "training_effective_value_weights.json")
    if weights != persisted_weights or effective_value_weights \
            != persisted_effective:
        raise RuntimeError("calibration weights do not reproduce source round")

    main_dataset = ExpertDataset(
        conditioned,
        weights,
        value_weights=effective_value_weights,
        encoding_config=encoding,
        value_norm=value_norm,
    )
    trace_records = load_records(cfg.trace_dataset)
    trace_dataset = ExpertDataset(
        trace_records,
        [1.0] * len(trace_records),
        value_weights=[0.0] * len(trace_records),
        encoding_config=encoding,
        value_norm=value_norm,
    )
    if not trace_records or not all(
            bool(item["trace_pair_valid"]) for item in trace_dataset.items):
        raise ValueError("calibration requires valid trace-pair records")

    policy_loss, policy_gradient, policy_groups = _policy_gradient(
        model,
        main_dataset,
        batch_size=int(source_config["batch_size"]),
        device=device,
    )
    policy_norm = float(torch.linalg.vector_norm(policy_gradient).item())
    diagnostics = {}
    trace_vectors = {}
    trace_groups = {}
    for margin in DIAGNOSTIC_MARGINS:
        report, vector, groups = _trace_gradient(
            model, trace_dataset, margin=margin, device=device)
        diagnostics[str(margin)] = report
        trace_vectors[margin] = vector
        trace_groups[margin] = groups
    selected_vector = trace_vectors[FIXED_MARGIN]
    trace_norm = float(torch.linalg.vector_norm(selected_vector).item())
    if policy_norm <= 0 or trace_norm <= 0:
        raise RuntimeError("calibration produced a zero gradient norm")
    cosine = float(torch.dot(policy_gradient, selected_vector).item() /
                   (policy_norm * trace_norm))
    arms = {}
    for arm, target_ratio in TARGET_GRADIENT_RATIOS.items():
        raw_weight = 0.0 if target_ratio == 0 else (
            target_ratio * policy_norm / trace_norm)
        weight = _rounded_weight(raw_weight)
        arms[arm] = {
            "trace_ranking_weight": weight,
            "target_trace_to_policy_gradient_norm_ratio": target_ratio,
            "realized_initial_ratio_after_rounding": (
                weight * trace_norm / policy_norm),
        }

    direction_counts: dict[str, int] = {}
    for record in trace_records:
        role = record["trace_preference"]["successful_role"]
        direction = f"{role}_only"
        direction_counts[direction] = direction_counts.get(direction, 0) + 1
    result = {
        "schema_version": SCHEMA_VERSION,
        "semantics": SEMANTICS,
        "status": "completed_without_solver_evaluation",
        "config": asdict(cfg),
        "inputs": {
            "checkpoint_sha256": sha256_file(cfg.checkpoint),
            "trace_dataset_sha256": sha256_file(cfg.trace_dataset),
            "source_config_sha256": sha256_file(source / "config.json"),
            "source_manifest_sha256": sha256_file(manifest_path),
            "source_sample_sha256": manifest["sample"]["sha256"],
            "replay_snapshot_sha256": sha256_file(cfg.replay_snapshot),
        },
        "training_contract": {
            "trainable_part": source_config["trainable_part"],
            "main_record_count": len(conditioned),
            "main_policy_weight_mass": sum(weights),
            "trace_record_count": len(trace_records),
            "trace_direction_counts": direction_counts,
            "trace_target_loss_weight": 0.0,
            "trace_value_loss_weight": 0.0,
            "trace_batching": "one_shuffled_evenly_spread_pass_per_epoch_v1",
        },
        "policy_gradient": {
            "mean_policy_loss": policy_loss,
            "l2_norm": policy_norm,
            "parameter_group_l2_norms": policy_groups,
        },
        "trace_gradient_diagnostics": diagnostics,
        "selected_trace_gradient": {
            "margin": FIXED_MARGIN,
            "l2_norm": trace_norm,
            "parameter_group_l2_norms": trace_groups[FIXED_MARGIN],
            "cosine_with_policy_gradient": cosine,
        },
        "coefficient_rule": {
            "formula": (
                "weight = target_ratio * policy_gradient_l2 / "
                "trace_gradient_l2"),
            "rounding": "six_significant_digits",
            "target_ratios": TARGET_GRADIENT_RATIOS,
            "outcome_metrics_consulted": False,
        },
        "arms": arms,
        "trainable": trainable,
        "evaluation_policy": {
            "solver_search_run": False,
            "retention_pool_loaded": False,
            "promotion_pool_loaded": False,
            "sealed_final_test_loaded_or_evaluated": False,
        },
    }
    output = Path(cfg.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    atomic_write_json(output / "calibration.json", result)
    preregistration = {
        "schema_version": SCHEMA_VERSION,
        "semantics": "matched_trace_pair_ranking_sweep_preregistration_v1",
        "status": "frozen_before_matched_training",
        "research_question": (
            "Does weak first-divergence pairwise supervision reduce bounded-"
            "search threshold delays without suppressing frontier learning?"),
        "calibration": {
            "path": str(output / "calibration.json"),
            "sha256": sha256_file(output / "calibration.json"),
            "outcome_metrics_consulted": False,
        },
        "fixed_inputs": result["inputs"],
        "fixed_training": {
            "trace_margin": FIXED_MARGIN,
            "trace_dataset_one_pass_per_epoch": True,
            "trace_records_contribute_policy_ce": False,
            "trace_records_contribute_value_loss": False,
            "all_source_samples_and_training_seeds_matched_across_arms": True,
            "initial_checkpoint_matched_across_arms": True,
            "all_non_trace_training_settings_unchanged": True,
        },
        "arms": arms,
        "evaluation_constraints": {
            "existing_retention_guard_unchanged": True,
            "existing_frontier_promotion_contract_unchanged": True,
            "learner_only_improvements_must_be_reported_symmetrically": True,
            "sealed_final_test_must_not_be_loaded_or_evaluated": True,
        },
        "interpretation": {
            "success": (
                "an arm preserves frontier progress while materially reducing "
                "budget-threshold retention failures relative to control"),
            "failure": (
                "no anchored arm improves the stability/plasticity tradeoff; "
                "do not tune coefficients on the retention outcomes"),
        },
    }
    preregistration["fingerprint"] = hash_canonical_value(preregistration)
    atomic_write_json(output / "preregistration.json", preregistration)
    return result


def _parse_args(argv: list[str] | None = None) -> TraceRankingCalibrationConfig:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--trace-dataset", required=True)
    parser.add_argument("--source-run", required=True)
    parser.add_argument("--replay-snapshot", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--source-round", type=int, default=1)
    parser.add_argument("--device", default="cuda")
    return TraceRankingCalibrationConfig(**vars(parser.parse_args(argv)))


def main(argv: list[str] | None = None) -> int:
    result = run_trace_ranking_calibration(_parse_args(argv))
    print(json.dumps({
        "policy_gradient": result["policy_gradient"],
        "selected_trace_gradient": result["selected_trace_gradient"],
        "arms": result["arms"],
        "evaluation_policy": result["evaluation_policy"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
