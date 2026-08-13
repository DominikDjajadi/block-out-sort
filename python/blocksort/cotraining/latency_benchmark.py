"""Warm end-to-end latency benchmark for neural-guided graph search.

This is an operational diagnostic, not a model-quality promotion evaluation.
It deliberately accepts only an explicit development level file and never
loads an evaluation split or the sealed final test set.
"""

from __future__ import annotations

import argparse
import json
import math
import platform
import random
import statistics
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..environment import Environment
from ..schema import Level
from ..serialization import level_from_dict
from ..training.transaction import atomic_write_json, sha256_file


_SCHEMA_VERSION = 1
_SEMANTICS = "warm_single_level_graph_search_latency_v1"


@dataclass(frozen=True)
class NamedCheckpoint:
    name: str
    path: str


@dataclass(frozen=True)
class LatencyBenchmarkConfig:
    checkpoints: tuple[NamedCheckpoint, ...]
    levels_path: str
    output_path: str
    budgets: tuple[int, ...] = (4, 8, 16, 32, 64, 128)
    sample_count: int = 100
    repeats: int = 2
    warmup_runs: int = 4
    c_puct: float = 1.5
    inference_batch_size: int = 8
    virtual_loss: float = 1.0
    seed: int = 2103
    device: str = "cuda"
    progress_interval: int = 25

    def validate(self) -> None:
        if not self.checkpoints:
            raise ValueError("at least one checkpoint is required")
        names: set[str] = set()
        for checkpoint in self.checkpoints:
            if not checkpoint.name or checkpoint.name in names:
                raise ValueError("checkpoint names must be non-empty and unique")
            names.add(checkpoint.name)
            if not Path(checkpoint.path).is_file():
                raise ValueError(
                    f"checkpoint does not exist: {checkpoint.path}")
        if not Path(self.levels_path).is_file():
            raise ValueError(f"level file does not exist: {self.levels_path}")
        if not self.budgets or len(set(self.budgets)) != len(self.budgets):
            raise ValueError("budgets must be non-empty and unique")
        if any(isinstance(value, bool) or not isinstance(value, int)
               or value <= 0 for value in self.budgets):
            raise ValueError("budgets must contain positive integers")
        for label, value, minimum in (
                ("sample_count", self.sample_count, 1),
                ("repeats", self.repeats, 1),
                ("warmup_runs", self.warmup_runs, 0),
                ("inference_batch_size", self.inference_batch_size, 1),
                ("progress_interval", self.progress_interval, 1)):
            if (isinstance(value, bool) or not isinstance(value, int)
                    or value < minimum):
                raise ValueError(f"{label} must be an integer >= {minimum}")
        if not math.isfinite(self.c_puct) or self.c_puct < 0:
            raise ValueError("c_puct must be finite and non-negative")
        if not math.isfinite(self.virtual_loss) or self.virtual_loss < 0:
            raise ValueError("virtual_loss must be finite and non-negative")


def _parse_checkpoint(value: str) -> NamedCheckpoint:
    name, separator, path = value.partition("=")
    if not separator or not name or not path:
        raise argparse.ArgumentTypeError(
            "checkpoints must use NAME=PATH syntax")
    return NamedCheckpoint(name=name, path=path)


def _load_levels(path: str | Path) -> list[Level]:
    source = Path(path)
    text = source.read_text(encoding="utf-8")
    try:
        decoded = json.loads(text)
        raw = decoded if isinstance(decoded, list) else [decoded]
    except json.JSONDecodeError:
        raw = [json.loads(line) for line in text.splitlines() if line.strip()]
    levels: list[Level] = []
    for index, item in enumerate(raw):
        if not isinstance(item, dict):
            raise ValueError(f"level item {index} is not an object")
        level_data = item.get("level", item)
        if not isinstance(level_data, dict):
            raise ValueError(f"level item {index} has no level object")
        levels.append(level_from_dict(level_data))
    if not levels:
        raise ValueError("level file contains no levels")
    return levels


def _percentile(values: list[float], fraction: float) -> float:
    if not values:
        raise ValueError("cannot calculate a percentile of no values")
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = fraction * (len(ordered) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _summarize_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    elapsed = [float(row["elapsed_seconds"]) for row in rows]
    solved = sum(bool(row["solved"]) for row in rows)
    simulations = sum(int(row["budget"]) for row in rows)
    total_seconds = sum(elapsed)
    return {
        "requests": len(rows),
        "unique_levels": len({int(row["level_index"]) for row in rows}),
        "repeats": len({int(row["repeat"]) for row in rows}),
        "solved_requests": solved,
        "solve_rate": solved / len(rows),
        "latency_ms": {
            "mean": 1000.0 * statistics.fmean(elapsed),
            "p50": 1000.0 * _percentile(elapsed, 0.50),
            "p90": 1000.0 * _percentile(elapsed, 0.90),
            "p95": 1000.0 * _percentile(elapsed, 0.95),
            "p99": 1000.0 * _percentile(elapsed, 0.99),
            "max": 1000.0 * max(elapsed),
        },
        "throughput": {
            "requests_per_second": len(rows) / total_seconds,
            "nominal_simulations_per_second": simulations / total_seconds,
        },
    }


def _cuda_synchronize(torch, device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def run_latency_benchmark(cfg: LatencyBenchmarkConfig) -> dict[str, Any]:
    cfg.validate()

    import torch

    from ..search.config import SearchConfig
    from ..search.graph_search import BlocksortAdapter, GraphSearch
    from ..search.seeding import derive_trial_seed, level_search_identity
    from ..training.checkpoint import (
        configs_from_checkpoint, load_checkpoint, model_from_checkpoint)

    device = torch.device(cfg.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")

    all_levels = _load_levels(cfg.levels_path)
    if cfg.sample_count > len(all_levels):
        raise ValueError(
            f"sample_count {cfg.sample_count} exceeds {len(all_levels)} levels")
    level_indices = list(range(len(all_levels)))
    random.Random(cfg.seed).shuffle(level_indices)
    level_indices = sorted(level_indices[:cfg.sample_count])
    levels = [all_levels[index] for index in level_indices]
    env = Environment()
    identities = [level_search_identity(env, level) for level in levels]

    checkpoint_results: dict[str, Any] = {}
    for checkpoint_index, named in enumerate(cfg.checkpoints):
        load_started = time.perf_counter()
        checkpoint = load_checkpoint(named.path, map_location="cpu")
        encoding, _model_cfg, value_norm = configs_from_checkpoint(checkpoint)
        model = model_from_checkpoint(checkpoint, map_location=device)
        model.eval()
        adapter = BlocksortAdapter(env, model, encoding, value_norm, device)
        _cuda_synchronize(torch, device)
        load_seconds = time.perf_counter() - load_started

        for warmup_index in range(cfg.warmup_runs):
            level = levels[warmup_index % len(levels)]
            warmup_cfg = SearchConfig(
                simulations=max(cfg.budgets), c_puct=cfg.c_puct,
                temperature=0.0,
                inference_batch_size=cfg.inference_batch_size,
                virtual_loss=cfg.virtual_loss,
                value_normalization_constant=getattr(
                    value_norm, "constant", 20.0),
                seed=derive_trial_seed(
                    cfg.seed, trial_index=warmup_index,
                    level_identity=identities[warmup_index % len(levels)],
                    evaluation_context=(
                        f"latency_warmup_budget={max(cfg.budgets)}")),
            )
            GraphSearch(adapter, warmup_cfg).run(env.initial_state(level))
        _cuda_synchronize(torch, device)

        tasks = [
            (repeat, local_index, budget)
            for repeat in range(cfg.repeats)
            for local_index in range(len(levels))
            for budget in cfg.budgets
        ]
        random.Random(cfg.seed + 10_000 * (checkpoint_index + 1)).shuffle(tasks)
        rows: list[dict[str, Any]] = []
        for task_index, (repeat, local_index, budget) in enumerate(tasks, 1):
            level = levels[local_index]
            search_cfg = SearchConfig(
                simulations=budget, c_puct=cfg.c_puct, temperature=0.0,
                inference_batch_size=cfg.inference_batch_size,
                virtual_loss=cfg.virtual_loss,
                value_normalization_constant=getattr(
                    value_norm, "constant", 20.0),
                seed=derive_trial_seed(
                    cfg.seed,
                    trial_index=(
                        repeat * len(cfg.budgets)
                        + cfg.budgets.index(budget)),
                    level_identity=identities[local_index],
                    evaluation_context=f"latency_budget={budget}"),
            )
            _cuda_synchronize(torch, device)
            started = time.perf_counter()
            result = GraphSearch(adapter, search_cfg).run(
                env.initial_state(level))
            _cuda_synchronize(torch, device)
            elapsed = time.perf_counter() - started
            rows.append({
                "repeat": repeat,
                "level_index": level_indices[local_index],
                "level_identity": identities[local_index],
                "budget": budget,
                "elapsed_seconds": elapsed,
                "solved": result.solved,
                "solution_length": result.solution_length,
                "termination_reason": result.termination_reason,
                "model_evaluations": result.stats.model_evaluations,
                "model_evaluation_batches":
                    result.stats.model_evaluation_batches,
                "nodes_expanded": result.stats.nodes_expanded,
            })
            if task_index % cfg.progress_interval == 0:
                print(
                    f"{named.name}: {task_index}/{len(tasks)} requests",
                    flush=True)

        per_budget = {
            str(budget): _summarize_rows(
                [row for row in rows if row["budget"] == budget])
            for budget in cfg.budgets
        }
        checkpoint_results[named.name] = {
            "checkpoint": named.path,
            "checkpoint_sha256": sha256_file(named.path),
            "model_load_seconds": load_seconds,
            "per_budget": per_budget,
            "rows": rows,
        }
        del adapter, model, checkpoint
        if device.type == "cuda":
            torch.cuda.empty_cache()

    result = {
        "schema_version": _SCHEMA_VERSION,
        "semantics": _SEMANTICS,
        "purpose": (
            "Operational latency/throughput diagnostic only; not promotion "
            "evidence and not a sealed-final evaluation."),
        "config": {
            "levels_path": cfg.levels_path,
            "levels_sha256": sha256_file(cfg.levels_path),
            "budgets": list(cfg.budgets),
            "sample_count": cfg.sample_count,
            "repeats": cfg.repeats,
            "warmup_runs": cfg.warmup_runs,
            "c_puct": cfg.c_puct,
            "inference_batch_size": cfg.inference_batch_size,
            "virtual_loss": cfg.virtual_loss,
            "seed": cfg.seed,
            "device": cfg.device,
            "selected_level_indices": level_indices,
            "selected_level_identities": identities,
        },
        "system": {
            "platform": platform.platform(),
            "python": sys.version,
            "torch": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "cuda_device": (
                torch.cuda.get_device_name(device)
                if device.type == "cuda" else None),
        },
        "checkpoints": checkpoint_results,
    }
    atomic_write_json(cfg.output_path, result)
    return result


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint", action="append", type=_parse_checkpoint, required=True,
        help="named checkpoint as NAME=PATH (repeatable)")
    parser.add_argument("--levels", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--budgets", type=int, nargs="+", default=[4, 8, 16, 32, 64, 128])
    parser.add_argument("--sample-count", type=int, default=100)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--warmup-runs", type=int, default=4)
    parser.add_argument("--c-puct", type=float, default=1.5)
    parser.add_argument("--inference-batch-size", type=int, default=8)
    parser.add_argument("--virtual-loss", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=2103)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--progress-interval", type=int, default=25)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    cfg = LatencyBenchmarkConfig(
        checkpoints=tuple(args.checkpoint), levels_path=args.levels,
        output_path=args.output, budgets=tuple(args.budgets),
        sample_count=args.sample_count, repeats=args.repeats,
        warmup_runs=args.warmup_runs, c_puct=args.c_puct,
        inference_batch_size=args.inference_batch_size,
        virtual_loss=args.virtual_loss, seed=args.seed, device=args.device,
        progress_interval=args.progress_interval)
    result = run_latency_benchmark(cfg)
    for name, checkpoint in result["checkpoints"].items():
        print(name)
        for budget in result["config"]["budgets"]:
            summary = checkpoint["per_budget"][str(budget)]
            latency = summary["latency_ms"]
            print(
                f"  budget {budget}: p50={latency['p50']:.1f} ms, "
                f"p95={latency['p95']:.1f} ms, "
                f"mean={latency['mean']:.1f} ms")
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
