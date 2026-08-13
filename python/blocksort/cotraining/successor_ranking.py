"""Audit whether value heads rank legal successor states correctly.

Scalar value calibration can improve while MCTS behavior regresses.  This
diagnostic uses fully exact validation records and asks the more direct
question: if actions were ranked only by the value predicted for their
successors, would optimal actions appear ahead of worse alternatives?

The command performs batched network inference only.  It does not run MCTS,
generate oracle labels, alter checkpoints, or inspect the sealed final test.
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

from ..conformance import normalized_to_action
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
SEMANTICS = "heldout_successor_value_ranking_v1"
_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]*$")
_TIE_TOLERANCE = 1e-9


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
class SuccessorRankingConfig:
    dataset: str
    split_manifest: str
    checkpoints: tuple[NamedCheckpoint, ...]
    reference_name: str
    output_dir: str
    heldout_split: str = "validation"
    batch_size: int = 512
    device: str = "cuda"
    max_cost: float = 1_000.0

    def validate(self) -> None:
        for label, path in (
                ("dataset", self.dataset),
                ("split manifest", self.split_manifest)):
            if not Path(path).is_file():
                raise ValueError(f"{label} does not exist: {path}")
        if self.heldout_split != "validation":
            raise ValueError(
                "successor ranking is restricted to the validation split")
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
        if not math.isfinite(self.max_cost) or self.max_cost <= 0:
            raise ValueError("max_cost must be finite and positive")


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


def _depth_bucket(value: float) -> str:
    if value <= 3:
        return "1_to_3"
    if value <= 6:
        return "4_to_6"
    if value < 10:
        return "7_to_9"
    return "10_plus"


def _is_exact(record: dict[str, Any]) -> bool:
    value_exact = bool(record.get(
        "value_exact",
        record.get("optimal_remaining_moves") is not None,
    ))
    # Successor ranking needs every action cost, not merely an exact root value.
    return value_exact and bool(record.get("action_values_complete", True))


def _prepare_examples(
    records: list[dict[str, Any]],
    encoding_config,
    *,
    max_cost: float,
) -> tuple[
        list[dict[str, Any]],
        list[torch.Tensor],
        list[torch.Tensor],
        list[tuple[int, int]],
]:
    """Reconstruct exact states and flatten successors needing inference."""
    env = Environment()
    examples: list[dict[str, Any]] = []
    boards: list[torch.Tensor] = []
    global_features: list[torch.Tensor] = []
    locations: list[tuple[int, int]] = []

    for record_index, record in enumerate(records):
        if not _is_exact(record):
            continue
        try:
            target = float(record["optimal_remaining_moves"])
            legal_actions = record["legal_actions"]
            action_costs = record["action_costs"]
            if (not isinstance(legal_actions, list)
                    or not isinstance(action_costs, list)
                    or not legal_actions
                    or len(legal_actions) != len(action_costs)):
                raise ValueError(
                    "legal_actions and action_costs must be aligned nonempty lists")
            if not math.isfinite(target) or target < 1:
                raise ValueError(
                    "optimal_remaining_moves must be finite and positive")

            parsed_costs: list[float | None] = []
            for cost in action_costs:
                if cost is None:
                    parsed_costs.append(None)
                    continue
                number = float(cost)
                if not math.isfinite(number) or number < target:
                    raise ValueError(
                        "action costs must be null or finite and no better "
                        "than the optimal state value")
                parsed_costs.append(number)
            if not any(
                    cost is not None
                    and math.isclose(cost, target, abs_tol=_TIE_TOLERANCE)
                    for cost in parsed_costs):
                raise ValueError("record contains no optimal legal action")

            level = level_from_dict(record["level"])
            state = deserialize_state(level, record["state"])
            predicted_costs: list[float | None] = [None] * len(legal_actions)
            example_index = len(examples)
            for action_index, serialized in enumerate(legal_actions):
                action = normalized_to_action(state, serialized)
                successor = env.apply_action(state, action)
                if env.is_terminal(successor):
                    predicted_costs[action_index] = 1.0
                elif env.is_deadlock(successor):
                    predicted_costs[action_index] = 1.0 + max_cost
                else:
                    encoded = encode_state(
                        env, successor, encoding_config)
                    if (not torch.isfinite(encoded.board).all()
                            or not torch.isfinite(
                                encoded.global_features).all()):
                        raise ValueError(
                            "successor encoded to nonfinite inputs")
                    boards.append(encoded.board)
                    global_features.append(encoded.global_features)
                    locations.append((example_index, action_index))

            examples.append({
                "record_index": record_index,
                "level_id": str(record.get("level_id", "")),
                "static_level_signature":
                    str(record["static_level_signature"]),
                "state_key": str(record["state_key"]),
                "optimal_remaining_moves": target,
                "depth_bucket": _depth_bucket(target),
                "remaining_blocks": int(record["remaining_blocks"]),
                "action_costs": parsed_costs,
                "predicted_action_costs": predicted_costs,
            })
        except (KeyError, TypeError, ValueError, IndexError) as exc:
            raise ValueError(
                f"failed to prepare exact diagnostic record "
                f"{record_index}") from exc

    if not examples:
        raise ValueError(
            "held-out validation split contains no fully exact records")
    return examples, boards, global_features, locations


@torch.no_grad()
def _predict_action_costs(
    model,
    examples: list[dict[str, Any]],
    boards: list[torch.Tensor],
    global_features: list[torch.Tensor],
    locations: list[tuple[int, int]],
    *,
    value_norm,
    batch_size: int,
    max_cost: float,
    device: torch.device,
) -> list[list[float]]:
    predictions = [
        list(example["predicted_action_costs"])
        for example in examples
    ]
    model.eval()
    cursor = 0
    for start in range(0, len(boards), batch_size):
        stop = min(start + batch_size, len(boards))
        board_batch = torch.stack(boards[start:stop]).to(device)
        global_batch = torch.stack(global_features[start:stop]).to(device)
        _, normalized = model(board_batch, global_batch)
        for value in normalized.detach().cpu().reshape(-1):
            example_index, action_index = locations[cursor]
            raw_cost = value_norm.denormalize(float(value))
            predictions[example_index][action_index] = (
                1.0 + min(max_cost, max(0.0, raw_cost)))
            cursor += 1
    if cursor != len(locations):
        raise RuntimeError("successor prediction count mismatch")
    if any(
            value is None or not math.isfinite(value)
            for row in predictions for value in row):
        raise RuntimeError("successor prediction table is incomplete")
    return [[float(value) for value in row] for row in predictions]


def _ordering_score(better: float, worse: float) -> float:
    """Return 1 for correct order, 0.5 for a tie, and 0 for reversal."""
    if better < worse - _TIE_TOLERANCE:
        return 1.0
    if math.isclose(better, worse, abs_tol=_TIE_TOLERANCE):
        return 0.5
    return 0.0


def _state_metrics(
    example: dict[str, Any],
    predictions: list[float],
) -> dict[str, Any]:
    costs = example["action_costs"]
    target = example["optimal_remaining_moves"]
    if len(costs) != len(predictions):
        raise ValueError("one successor prediction is required per action")

    optimal = [
        index for index, cost in enumerate(costs)
        if cost is not None
        and math.isclose(cost, target, abs_tol=_TIE_TOLERANCE)
    ]
    suboptimal = [index for index in range(len(costs)) if index not in optimal]
    chosen = min(
        range(len(predictions)),
        key=lambda index: (predictions[index], index),
    )
    chosen_cost = costs[chosen]
    chosen_regret = (
        None if chosen_cost is None else float(chosen_cost - target))

    optimal_pair_score = 0.0
    optimal_pair_count = 0
    for optimal_index in optimal:
        for worse_index in suboptimal:
            optimal_pair_score += _ordering_score(
                predictions[optimal_index], predictions[worse_index])
            optimal_pair_count += 1

    all_pair_score = 0.0
    all_pair_count = 0
    for left in range(len(costs)):
        for right in range(left + 1, len(costs)):
            left_cost = costs[left]
            right_cost = costs[right]
            if left_cost is None and right_cost is None:
                continue
            if (left_cost is not None and right_cost is not None
                    and math.isclose(
                        left_cost, right_cost,
                        abs_tol=_TIE_TOLERANCE)):
                continue
            if right_cost is None or (
                    left_cost is not None and left_cost < right_cost):
                better, worse = left, right
            else:
                better, worse = right, left
            all_pair_score += _ordering_score(
                predictions[better], predictions[worse])
            all_pair_count += 1

    decision_state = bool(optimal and suboptimal)
    best_margin = (
        min(predictions[index] for index in suboptimal)
        - min(predictions[index] for index in optimal)
        if decision_state else None
    )
    return {
        "level_id": example["level_id"],
        "static_level_signature": example["static_level_signature"],
        "state_key": example["state_key"],
        "optimal_remaining_moves": target,
        "depth_bucket": example["depth_bucket"],
        "remaining_blocks": example["remaining_blocks"],
        "legal_actions": len(costs),
        "optimal_actions": len(optimal),
        "decision_state": decision_state,
        "selected_action_index": chosen,
        "selected_action_is_optimal": chosen in optimal,
        "selected_action_oracle_cost": chosen_cost,
        "selected_action_regret": chosen_regret,
        "selected_action_is_infinite": chosen_cost is None,
        "optimal_pair_score": optimal_pair_score,
        "optimal_pair_count": optimal_pair_count,
        "all_pair_score": all_pair_score,
        "all_pair_count": all_pair_count,
        "best_optimal_vs_suboptimal_margin_raw_moves": best_margin,
        "oracle_action_costs": costs,
        "predicted_action_costs": predictions,
    }


def _aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {"states": 0, "decision_states": 0}
    decision_rows = [row for row in rows if row["decision_state"]]
    optimal_pair_count = sum(row["optimal_pair_count"] for row in rows)
    all_pair_count = sum(row["all_pair_count"] for row in rows)
    finite_regrets = [
        row["selected_action_regret"]
        for row in decision_rows
        if row["selected_action_regret"] is not None
    ]
    infinite_choices = sum(
        row["selected_action_is_infinite"] for row in decision_rows)
    margins = [
        row["best_optimal_vs_suboptimal_margin_raw_moves"]
        for row in decision_rows
    ]
    return {
        "states": len(rows),
        "decision_states": len(decision_rows),
        "forced_or_all_optimal_states": len(rows) - len(decision_rows),
        "legal_actions": sum(row["legal_actions"] for row in rows),
        "top1_optimal_rate": (
            sum(row["selected_action_is_optimal"] for row in decision_rows)
            / len(decision_rows)
            if decision_rows else None
        ),
        "optimal_vs_suboptimal_pair_accuracy": (
            sum(row["optimal_pair_score"] for row in rows)
            / optimal_pair_count
            if optimal_pair_count else None
        ),
        "optimal_vs_suboptimal_pairs": optimal_pair_count,
        "all_distinct_cost_pair_accuracy": (
            sum(row["all_pair_score"] for row in rows) / all_pair_count
            if all_pair_count else None
        ),
        "all_distinct_cost_pairs": all_pair_count,
        "mean_selected_oracle_regret": (
            fmean(finite_regrets) if finite_regrets else None
        ),
        "infinite_cost_choice_count": infinite_choices,
        "infinite_cost_choice_rate": (
            infinite_choices / len(decision_rows)
            if decision_rows else None
        ),
        "mean_best_optimal_margin_raw_moves": (
            fmean(margins) if margins else None
        ),
    }


def _grouped(
    rows: list[dict[str, Any]],
    key,
) -> dict[str, dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault(str(key(row)), []).append(row)
    return {
        name: _aggregate(items)
        for name, items in sorted(groups.items())
    }


def _worst_choices(
    rows: list[dict[str, Any]],
    *,
    limit: int = 20,
) -> list[dict[str, Any]]:
    failures = [
        row for row in rows
        if row["decision_state"] and not row["selected_action_is_optimal"]
    ]
    failures.sort(key=lambda row: (
        row["selected_action_regret"] is not None,
        -(row["selected_action_regret"] or 0.0),
        row["static_level_signature"],
        row["state_key"],
    ))
    return failures[:limit]


def _model_report(
    examples: list[dict[str, Any]],
    predictions: list[list[float]],
) -> dict[str, Any]:
    rows = [
        _state_metrics(example, predicted)
        for example, predicted in zip(examples, predictions)
    ]
    return {
        "overall": _aggregate(rows),
        "by_depth_bucket":
            _grouped(rows, lambda row: row["depth_bucket"]),
        "by_optimal_moves":
            _grouped(rows, lambda row: int(row["optimal_remaining_moves"])),
        "by_remaining_blocks":
            _grouped(rows, lambda row: row["remaining_blocks"]),
        "worst_value_choices": _worst_choices(rows),
    }


def _comparison(
    reference: dict[str, Any],
    candidate: dict[str, Any],
) -> dict[str, Any]:
    ref = reference["overall"]
    cand = candidate["overall"]
    result = {"decision_states": cand["decision_states"]}
    for metric in (
            "top1_optimal_rate",
            "optimal_vs_suboptimal_pair_accuracy",
            "all_distinct_cost_pair_accuracy",
            "mean_selected_oracle_regret",
            "infinite_cost_choice_rate",
            "mean_best_optimal_margin_raw_moves"):
        ref_value = ref.get(metric)
        cand_value = cand.get(metric)
        result[f"{metric}_delta"] = (
            cand_value - ref_value
            if ref_value is not None and cand_value is not None
            else None
        )
    return result


def _write_csv(path: Path, reports: dict[str, Any]) -> None:
    handle = io.StringIO()
    fields = [
        "model", "slice", "states", "decision_states", "legal_actions",
        "top1_optimal_rate", "optimal_vs_suboptimal_pair_accuracy",
        "optimal_vs_suboptimal_pairs", "all_distinct_cost_pair_accuracy",
        "all_distinct_cost_pairs", "mean_selected_oracle_regret",
        "infinite_cost_choice_count", "infinite_cost_choice_rate",
        "mean_best_optimal_margin_raw_moves",
    ]
    writer = csv.DictWriter(handle, fieldnames=fields)
    writer.writeheader()
    for model_name, report in reports.items():
        slices = {"overall": report["overall"]}
        slices.update({
            f"depth:{name}": metrics
            for name, metrics in report["by_depth_bucket"].items()
        })
        for slice_name, metrics in slices.items():
            writer.writerow({
                "model": model_name,
                "slice": slice_name,
                **{
                    field: metrics.get(field)
                    for field in fields
                    if field not in ("model", "slice")
                },
            })
    atomic_write_text(path, handle.getvalue())


def run_successor_ranking(
    cfg: SuccessorRankingConfig,
) -> dict[str, Any]:
    cfg.validate()
    device = _resolve_device(cfg.device)
    root = Path(cfg.output_dir)
    root.mkdir(parents=True, exist_ok=True)

    loaded = {
        named.name: load_checkpoint(named.path, map_location="cpu")
        for named in cfg.checkpoints
    }
    reference_checkpoint = loaded[cfg.reference_name]
    encoding, _model_config, value_norm = configs_from_checkpoint(
        reference_checkpoint)
    expected_configs = (encoding.to_dict(), value_norm.to_dict())
    for named in cfg.checkpoints:
        observed_encoding, _observed_model, observed_value_norm = (
            configs_from_checkpoint(loaded[named.name]))
        observed = (
            observed_encoding.to_dict(), observed_value_norm.to_dict())
        if observed != expected_configs:
            raise ValueError(
                "checkpoint encoding or value normalization differs from "
                f"reference: "
                f"{named.name}")

    records = load_records(cfg.dataset)
    manifest = load_manifest(cfg.split_manifest)
    heldout_records = filter_records_for_split(
        records, manifest, cfg.heldout_split)
    if not heldout_records:
        raise ValueError("held-out validation split contains no records")
    examples, boards, global_features, locations = _prepare_examples(
        heldout_records, encoding, max_cost=cfg.max_cost)

    reports = {}
    for named in cfg.checkpoints:
        print(
            f"evaluating successor rankings with {named.name}",
            flush=True,
        )
        model = model_from_checkpoint(
            loaded[named.name], map_location=device)
        predictions = _predict_action_costs(
            model,
            examples,
            boards,
            global_features,
            locations,
            value_norm=value_norm,
            batch_size=cfg.batch_size,
            max_cost=cfg.max_cost,
            device=device,
        )
        reports[named.name] = _model_report(examples, predictions)
        del model

    reference = reports[cfg.reference_name]
    result = {
        "schema_version": SCHEMA_VERSION,
        "semantics": SEMANTICS,
        "config": json.loads(json.dumps(asdict(cfg))),
        "inputs": {
            "dataset_sha256": sha256_file(cfg.dataset),
            "split_manifest_sha256": sha256_file(cfg.split_manifest),
            "checkpoint_sha256": {
                named.name: sha256_file(named.path)
                for named in cfg.checkpoints
            },
        },
        "device": str(device),
        "heldout_split": cfg.heldout_split,
        "records_in_split": len(heldout_records),
        "fully_exact_records": len(examples),
        "successors_evaluated_by_network": len(locations),
        "models": reports,
        "comparisons_to_reference": {
            name: _comparison(reference, report)
            for name, report in reports.items()
            if name != cfg.reference_name
        },
        "final_test_status": "sealed_not_evaluated",
    }
    atomic_write_json(root / "summary.json", result)
    _write_csv(root / "summary.csv", reports)
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Compare one-ply successor value rankings on fully exact, "
            "held-out validation records."))
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--split-manifest", required=True)
    parser.add_argument(
        "--checkpoint", action="append", type=_parse_checkpoint,
        required=True, metavar="NAME=PATH")
    parser.add_argument("--reference-name", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--heldout-split", default="validation")
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-cost", type=float, default=1_000.0)
    return parser


def _format_metric(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.3f}"


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    cfg = SuccessorRankingConfig(
        dataset=args.dataset,
        split_manifest=args.split_manifest,
        checkpoints=tuple(args.checkpoint),
        reference_name=args.reference_name,
        output_dir=args.output_dir,
        heldout_split=args.heldout_split,
        batch_size=args.batch_size,
        device=args.device,
        max_cost=args.max_cost,
    )
    result = run_successor_ranking(cfg)
    for model_name, report in result["models"].items():
        metrics = report["overall"]
        top1 = _format_metric(metrics["top1_optimal_rate"])
        pair_accuracy = _format_metric(
            metrics["optimal_vs_suboptimal_pair_accuracy"])
        regret = _format_metric(
            metrics["mean_selected_oracle_regret"])
        print(
            f"{model_name}: decision states={metrics['decision_states']}, "
            f"top-1 optimal={top1}, "
            f"optimal-pair accuracy={pair_accuracy}, "
            f"mean selected regret={regret}",
        )
    print("final test: sealed and not evaluated")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
