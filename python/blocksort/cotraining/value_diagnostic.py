"""Audit value-target fit and held-out calibration across checkpoints.

This diagnostic performs batched network inference only. It does not run MCTS,
generate labels, alter checkpoints, or inspect the sealed final test.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import math
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import fmean
from typing import Any

import torch

from ..dataset.schema import deserialize_state
from ..environment import Environment
from ..serialization import level_from_dict
from ..training.checkpoint import (
    configs_from_checkpoint,
    load_checkpoint,
    model_from_checkpoint,
)
from ..training.dataset import load_records
from ..training.encoding import encode_state
from ..training.splits import filter_records_for_split, load_manifest
from ..training.transaction import atomic_write_json, atomic_write_text, sha256_file


SCHEMA_VERSION = 1
SEMANTICS = "frozen_value_calibration_diagnostic_v1"
_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]*$")


@dataclass(frozen=True)
class NamedCheckpoint:
    name: str
    path: str

    def validate(self) -> None:
        if not _NAME.fullmatch(self.name):
            raise ValueError(
                "checkpoint names must contain only letters, numbers, "
                "underscores, and hyphens")
        if not Path(self.path).is_file():
            raise ValueError(f"checkpoint does not exist: {self.path}")


@dataclass(frozen=True)
class ValueDiagnosticConfig:
    replay_sample: str
    base_dataset: str
    split_manifest: str
    checkpoints: tuple[NamedCheckpoint, ...]
    reference_name: str
    output_dir: str
    heldout_split: str = "validation"
    batch_size: int = 512
    device: str = "cuda"
    current_iteration: int = 2
    weight_exact_historical: float = 1.0
    weight_exact_new: float = 1.5
    weight_search: float = 0.5

    def validate(self) -> None:
        for label, path in (
                ("replay sample", self.replay_sample),
                ("base dataset", self.base_dataset),
                ("split manifest", self.split_manifest)):
            if not Path(path).is_file():
                raise ValueError(f"{label} does not exist: {path}")
        if self.heldout_split != "validation":
            raise ValueError(
                "value diagnostics are restricted to the validation split")
        if not self.checkpoints:
            raise ValueError("at least one checkpoint is required")
        names: set[str] = set()
        for checkpoint in self.checkpoints:
            checkpoint.validate()
            if checkpoint.name in names:
                raise ValueError(
                    f"duplicate checkpoint name: {checkpoint.name}")
            names.add(checkpoint.name)
        if self.reference_name not in names:
            raise ValueError(
                "reference_name must identify one of the checkpoints")
        if (isinstance(self.batch_size, bool)
                or not isinstance(self.batch_size, int)
                or self.batch_size <= 0):
            raise ValueError("batch_size must be a positive integer")
        if (isinstance(self.current_iteration, bool)
                or not isinstance(self.current_iteration, int)
                or self.current_iteration < 0):
            raise ValueError(
                "current_iteration must be a non-negative integer")
        for label, value in (
                ("weight_exact_historical", self.weight_exact_historical),
                ("weight_exact_new", self.weight_exact_new),
                ("weight_search", self.weight_search)):
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{label} must be finite and non-negative")


def _parse_checkpoint(value: str) -> NamedCheckpoint:
    name, separator, path = value.partition("=")
    if not separator or not name or not path:
        raise argparse.ArgumentTypeError(
            "checkpoints must use NAME=PATH syntax")
    result = NamedCheckpoint(name=name, path=path)
    try:
        result.validate()
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc
    return result


def _resolve_device(name: str) -> torch.device:
    if name == "auto":
        name = "cuda" if torch.cuda.is_available() else "cpu"
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA was requested but this Python environment has no "
            "CUDA-enabled PyTorch device")
    return torch.device(name)


def _is_exact(record: dict[str, Any]) -> bool:
    return bool(record.get(
        "value_exact",
        record.get("optimal_remaining_moves") is not None,
    ))


def _record_metadata(record: dict[str, Any]) -> dict[str, Any]:
    return {
        "target": float(record["value_target"]["raw_optimal_moves"]),
        "exact": _is_exact(record),
        "source": str(record.get("target_source", "exact_oracle")),
        "iteration": int(record.get("generation_iteration", 0)),
        "remaining_blocks": int(record["remaining_blocks"]),
        "static_level_signature": str(record["static_level_signature"]),
        "state_key": str(record["state_key"]),
    }


def _encode_inputs(
    records: list[dict[str, Any]],
    encoding_config,
) -> tuple[list[torch.Tensor], list[torch.Tensor], list[dict[str, Any]]]:
    env = Environment()
    boards: list[torch.Tensor] = []
    global_features: list[torch.Tensor] = []
    metadata: list[dict[str, Any]] = []
    for index, record in enumerate(records):
        try:
            level = level_from_dict(record["level"])
            state = deserialize_state(level, record["state"])
            encoded = encode_state(env, state, encoding_config)
            meta = _record_metadata(record)
        except (KeyError, TypeError, ValueError, IndexError) as exc:
            raise ValueError(
                f"failed to encode diagnostic record {index}") from exc
        if (not torch.isfinite(encoded.board).all()
                or not torch.isfinite(encoded.global_features).all()):
            raise ValueError(
                f"diagnostic record {index} encoded to nonfinite inputs")
        boards.append(encoded.board)
        global_features.append(encoded.global_features)
        metadata.append(meta)
    return boards, global_features, metadata


@torch.no_grad()
def _predict_raw_values(
    model,
    boards: list[torch.Tensor],
    global_features: list[torch.Tensor],
    *,
    value_norm,
    batch_size: int,
    device: torch.device,
) -> list[float]:
    predictions: list[float] = []
    model.eval()
    for start in range(0, len(boards), batch_size):
        stop = min(start + batch_size, len(boards))
        board_batch = torch.stack(boards[start:stop]).to(device)
        global_batch = torch.stack(global_features[start:stop]).to(device)
        _, normalized = model(board_batch, global_batch)
        predictions.extend(
            value_norm.denormalize(float(value))
            for value in normalized.detach().cpu().reshape(-1)
        )
    return predictions


def _pearson(targets: list[float], predictions: list[float]) -> float | None:
    if len(targets) < 2:
        return None
    target_mean = fmean(targets)
    prediction_mean = fmean(predictions)
    target_ss = sum((value - target_mean) ** 2 for value in targets)
    prediction_ss = sum(
        (value - prediction_mean) ** 2 for value in predictions)
    if target_ss == 0 or prediction_ss == 0:
        return None
    covariance = sum(
        (target - target_mean) * (prediction - prediction_mean)
        for target, prediction in zip(targets, predictions)
    )
    return covariance / math.sqrt(target_ss * prediction_ss)


def _metric_summary(
    metadata: list[dict[str, Any]],
    predictions: list[float],
    indices: list[int] | None = None,
) -> dict[str, Any]:
    selected = indices if indices is not None else list(range(len(metadata)))
    if not selected:
        return {"count": 0}
    targets = [metadata[index]["target"] for index in selected]
    predicted = [predictions[index] for index in selected]
    errors = [
        prediction - target
        for target, prediction in zip(targets, predicted)
    ]
    return {
        "count": len(selected),
        "target_mean_raw_moves": fmean(targets),
        "prediction_mean_raw_moves": fmean(predicted),
        "bias_raw_moves": fmean(errors),
        "mae_raw_moves": fmean(abs(error) for error in errors),
        "rmse_raw_moves": math.sqrt(fmean(error ** 2 for error in errors)),
        "within_one_move_rate":
            sum(abs(error) <= 1.0 for error in errors) / len(errors),
        "pearson": _pearson(targets, predicted),
    }


def _grouped_metrics(
    metadata: list[dict[str, Any]],
    predictions: list[float],
    key,
    *,
    include=None,
) -> dict[str, dict[str, Any]]:
    groups: dict[str, list[int]] = {}
    for index, item in enumerate(metadata):
        if include is not None and not include(item):
            continue
        groups.setdefault(str(key(item)), []).append(index)
    return {
        name: _metric_summary(metadata, predictions, indices)
        for name, indices in sorted(groups.items())
    }


def _worst_exact_errors(
    metadata: list[dict[str, Any]],
    predictions: list[float],
    *,
    limit: int = 20,
) -> list[dict[str, Any]]:
    rows = []
    for item, prediction in zip(metadata, predictions):
        if not item["exact"]:
            continue
        error = prediction - item["target"]
        rows.append({
            **item,
            "prediction": prediction,
            "error": error,
            "absolute_error": abs(error),
        })
    rows.sort(
        key=lambda row: (
            -row["absolute_error"],
            row["static_level_signature"],
            row["state_key"],
        ))
    return rows[:limit]


def _model_report(
    metadata: list[dict[str, Any]],
    predictions: list[float],
) -> dict[str, Any]:
    exact_indices = [
        index for index, item in enumerate(metadata) if item["exact"]]
    search_indices = [
        index for index, item in enumerate(metadata) if not item["exact"]]
    return {
        "all_stored_labels":
            _metric_summary(metadata, predictions),
        "exact_labels":
            _metric_summary(metadata, predictions, exact_indices),
        "search_labels":
            _metric_summary(metadata, predictions, search_indices),
        "by_source_and_iteration": _grouped_metrics(
            metadata,
            predictions,
            lambda item: f"{item['source']}:iteration_{item['iteration']}",
        ),
        "exact_by_generation_iteration": _grouped_metrics(
            metadata,
            predictions,
            lambda item: item["iteration"],
            include=lambda item: item["exact"],
        ),
        "exact_by_optimal_moves": _grouped_metrics(
            metadata,
            predictions,
            lambda item: int(item["target"]),
            include=lambda item: item["exact"],
        ),
        "exact_by_remaining_blocks": _grouped_metrics(
            metadata,
            predictions,
            lambda item: item["remaining_blocks"],
            include=lambda item: item["exact"],
        ),
        "worst_exact_errors":
            _worst_exact_errors(metadata, predictions),
    }


def _target_distribution(
    metadata: list[dict[str, Any]],
    indices: list[int],
) -> dict[str, Any]:
    if not indices:
        return {"count": 0}
    values = [metadata[index]["target"] for index in indices]
    mean = fmean(values)
    return {
        "count": len(values),
        "mean_raw_moves": mean,
        "stddev_raw_moves":
            math.sqrt(fmean((value - mean) ** 2 for value in values)),
        "min_raw_moves": min(values),
        "max_raw_moves": max(values),
    }


def _label_report(
    records: list[dict[str, Any]],
    metadata: list[dict[str, Any]],
    cfg: ValueDiagnosticConfig,
    *,
    include_training_weights: bool,
) -> dict[str, Any]:
    groups: dict[str, list[int]] = {}
    for index, item in enumerate(metadata):
        key = f"{item['source']}:iteration_{item['iteration']}"
        groups.setdefault(key, []).append(index)

    weight_mass: dict[str, float] = {}
    for item in metadata:
        if item["source"] == "graph_search":
            weight = cfg.weight_search
        elif (cfg.current_iteration > 0
              and item["iteration"] >= cfg.current_iteration):
            weight = cfg.weight_exact_new
        else:
            weight = cfg.weight_exact_historical
        key = f"{item['source']}:iteration_{item['iteration']}"
        weight_mass[key] = weight_mass.get(key, 0.0) + weight

    solved_search_deltas = []
    for record, item in zip(records, metadata):
        search = record.get("search") or {}
        solution_length = search.get("solution_length")
        if (not item["exact"] and search.get("solved")
                and solution_length is not None):
            solved_search_deltas.append(
                item["target"] - float(solution_length))

    grouped = {}
    for key, indices in sorted(groups.items()):
        grouped[key] = _target_distribution(metadata, indices)
        if include_training_weights:
            grouped[key]["effective_training_weight_mass"] = weight_mass[key]

    return {
        "by_source_and_iteration": grouped,
        "exact_count": sum(item["exact"] for item in metadata),
        "search_count": sum(not item["exact"] for item in metadata),
        "solved_search_target_vs_solution": {
            "count": len(solved_search_deltas),
            "mean_target_minus_solution_length":
                fmean(solved_search_deltas)
                if solved_search_deltas else None,
            "mae_target_vs_solution_length":
                fmean(abs(value) for value in solved_search_deltas)
                if solved_search_deltas else None,
        },
    }


def _comparison(
    reference: dict[str, Any],
    candidate: dict[str, Any],
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for slice_name in ("all_stored_labels", "exact_labels", "search_labels"):
        ref = reference[slice_name]
        cand = candidate[slice_name]
        if not ref.get("count") or not cand.get("count"):
            result[slice_name] = {"count": cand.get("count", 0)}
            continue
        result[slice_name] = {
            "count": cand["count"],
            "mae_delta_raw_moves":
                cand["mae_raw_moves"] - ref["mae_raw_moves"],
            "rmse_delta_raw_moves":
                cand["rmse_raw_moves"] - ref["rmse_raw_moves"],
            "bias_delta_raw_moves":
                cand["bias_raw_moves"] - ref["bias_raw_moves"],
            "within_one_move_rate_delta":
                cand["within_one_move_rate"]
                - ref["within_one_move_rate"],
        }
    return result


def _write_csv(path: Path, datasets: dict[str, Any]) -> None:
    handle = io.StringIO()
    fields = [
        "dataset", "model", "slice", "count", "target_mean_raw_moves",
        "prediction_mean_raw_moves", "bias_raw_moves", "mae_raw_moves",
        "rmse_raw_moves", "within_one_move_rate", "pearson",
    ]
    writer = csv.DictWriter(handle, fieldnames=fields)
    writer.writeheader()
    for dataset_name, dataset in datasets.items():
        for model_name, model in dataset["models"].items():
            for slice_name in (
                    "all_stored_labels", "exact_labels", "search_labels"):
                metrics = model[slice_name]
                writer.writerow({
                    "dataset": dataset_name,
                    "model": model_name,
                    "slice": slice_name,
                    **{
                        field: metrics.get(field)
                        for field in fields
                        if field not in ("dataset", "model", "slice")
                    },
                })
    atomic_write_text(path, handle.getvalue())


def run_value_diagnostic(cfg: ValueDiagnosticConfig) -> dict[str, Any]:
    cfg.validate()
    device = _resolve_device(cfg.device)
    root = Path(cfg.output_dir)
    root.mkdir(parents=True, exist_ok=True)

    loaded = {
        named.name: load_checkpoint(named.path, map_location="cpu")
        for named in cfg.checkpoints
    }
    reference_checkpoint = loaded[cfg.reference_name]
    encoding, model_config, value_norm = configs_from_checkpoint(
        reference_checkpoint)
    expected_configs = (
        encoding.to_dict(), model_config.to_dict(), value_norm.to_dict())
    for named in cfg.checkpoints:
        candidate_configs = configs_from_checkpoint(loaded[named.name])
        observed = tuple(config.to_dict() for config in candidate_configs)
        if observed != expected_configs:
            raise ValueError(
                f"checkpoint configuration differs from reference: "
                f"{named.name}")

    replay_records = load_records(cfg.replay_sample)
    base_records = load_records(cfg.base_dataset)
    manifest = load_manifest(cfg.split_manifest)
    heldout_records = filter_records_for_split(
        base_records, manifest, cfg.heldout_split)
    if not replay_records:
        raise ValueError("replay sample contains no records")
    if not heldout_records:
        raise ValueError("held-out validation split contains no records")

    replay_signatures = {
        str(record["static_level_signature"]) for record in replay_records}
    heldout_signatures = {
        str(record["static_level_signature"]) for record in heldout_records}
    overlap = replay_signatures.intersection(heldout_signatures)
    if overlap:
        raise RuntimeError(
            "replay sample overlaps the held-out validation levels: "
            f"{len(overlap)} signature(s)")

    datasets: dict[str, Any] = {}
    for dataset_name, records in (
            ("fixed_replay_sample", replay_records),
            ("heldout_exact_validation", heldout_records)):
        boards, global_features, metadata = _encode_inputs(records, encoding)
        model_reports = {}
        for named in cfg.checkpoints:
            print(
                f"evaluating {dataset_name} with {named.name}",
                flush=True,
            )
            model = model_from_checkpoint(
                loaded[named.name], map_location=device)
            predictions = _predict_raw_values(
                model,
                boards,
                global_features,
                value_norm=value_norm,
                batch_size=cfg.batch_size,
                device=device,
            )
            model_reports[named.name] = _model_report(metadata, predictions)
            del model
        reference_report = model_reports[cfg.reference_name]
        datasets[dataset_name] = {
            "records": len(records),
            "unique_levels": len({
                item["static_level_signature"] for item in metadata}),
            "labels": _label_report(
                records,
                metadata,
                cfg,
                include_training_weights=(
                    dataset_name == "fixed_replay_sample"),
            ),
            "models": model_reports,
            "comparisons_to_reference": {
                name: _comparison(reference_report, report)
                for name, report in model_reports.items()
                if name != cfg.reference_name
            },
        }

    result = {
        "schema_version": SCHEMA_VERSION,
        "semantics": SEMANTICS,
        "config": json.loads(json.dumps(asdict(cfg))),
        "inputs": {
            "replay_sample_sha256": sha256_file(cfg.replay_sample),
            "base_dataset_sha256": sha256_file(cfg.base_dataset),
            "split_manifest_sha256": sha256_file(cfg.split_manifest),
            "checkpoint_sha256": {
                named.name: sha256_file(named.path)
                for named in cfg.checkpoints
            },
        },
        "device": str(device),
        "validation_leakage_check": {
            "replay_unique_levels": len(replay_signatures),
            "heldout_unique_levels": len(heldout_signatures),
            "overlapping_level_signatures": 0,
            "status": "passed",
        },
        "datasets": datasets,
        "final_test_status": "sealed_not_evaluated",
    }
    atomic_write_json(root / "summary.json", result)
    _write_csv(root / "summary.csv", datasets)
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Compare value calibration on fixed replay and held-out exact "
            "validation records."))
    parser.add_argument("--replay-sample", required=True)
    parser.add_argument("--base-dataset", required=True)
    parser.add_argument("--split-manifest", required=True)
    parser.add_argument(
        "--checkpoint", action="append", type=_parse_checkpoint,
        required=True, metavar="NAME=PATH")
    parser.add_argument("--reference-name", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--heldout-split", default="validation")
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--current-iteration", type=int, default=2)
    parser.add_argument("--weight-exact-historical", type=float, default=1.0)
    parser.add_argument("--weight-exact-new", type=float, default=1.5)
    parser.add_argument("--weight-search", type=float, default=0.5)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    cfg = ValueDiagnosticConfig(
        replay_sample=args.replay_sample,
        base_dataset=args.base_dataset,
        split_manifest=args.split_manifest,
        checkpoints=tuple(args.checkpoint),
        reference_name=args.reference_name,
        output_dir=args.output_dir,
        heldout_split=args.heldout_split,
        batch_size=args.batch_size,
        device=args.device,
        current_iteration=args.current_iteration,
        weight_exact_historical=args.weight_exact_historical,
        weight_exact_new=args.weight_exact_new,
        weight_search=args.weight_search,
    )
    result = run_value_diagnostic(cfg)
    for dataset_name, dataset in result["datasets"].items():
        print(f"=== {dataset_name} ===")
        for model_name, report in dataset["models"].items():
            metrics = report["exact_labels"]
            print(
                f"{model_name}: exact count={metrics['count']}, "
                f"MAE={metrics.get('mae_raw_moves', float('nan')):.3f}, "
                f"bias={metrics.get('bias_raw_moves', float('nan')):+.3f}, "
                f"within-one={metrics.get('within_one_move_rate', 0):.3f}")
    print("validation leakage check: passed")
    print("final test: sealed and not evaluated")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
