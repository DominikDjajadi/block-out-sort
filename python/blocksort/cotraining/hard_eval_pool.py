"""Generate harder held-out evaluation pools for co-training promotion."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from ..environment import Environment
from ..oracle import Oracle
from ..serialization import level_from_dict, level_to_dict
from ..signature import static_level_signature
from ..training.transaction import atomic_write_json, atomic_write_text, sha256_file
from ..designer.config import GeneratorConfig
from .eval_split import (
    HARD_EVAL_POOL_SCHEMA_VERSION, canonical_eval_pool_sha256)
from .generation import random_level


DEFAULT_BUCKETS = (
    "in_distribution_hard",
    "adversarial_designer_hard",
    "ood_larger_board",
    "ood_more_colors",
    "mutation_stress",
)
SUPPORTED_BUCKETS = frozenset(DEFAULT_BUCKETS)
PARTIAL_CHECKPOINT_SCHEMA_VERSION = 2


@dataclass(frozen=True)
class HardPoolConfig:
    output: str
    total_count: int | None = None
    per_bucket_counts: dict[str, int] | None = None
    buckets: tuple[str, ...] = DEFAULT_BUCKETS
    seed: int = 1729
    rows: int = 6
    cols: int = 6
    color_count: int = 3
    density_min: float = 0.45
    density_max: float = 0.65
    mutation_budget_min: int = 8
    mutation_budget_max: int = 24
    oracle_validation_budget: int = 200_000
    oracle_validation_time_limit_seconds: float | None = None
    designer_checkpoint: str | None = None
    protagonist_checkpoint: str | None = None
    exclude_level_files: tuple[str, ...] = ()
    difficulty_min: float | None = None
    difficulty_max: float | None = None
    protagonist_simulations: int = 4
    protagonist_trials: int = 5
    device: str = "cpu"
    max_attempts_per_level: int = 200
    checkpoint_every_attempts: int = 100
    resume: bool = True


def _canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"),
        ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _level_items_from_file(path: str | Path) -> list[dict[str, Any]]:
    source = Path(path)
    try:
        text = source.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as exc:
        raise ValueError(f"could not read excluded level file: {source}") from exc
    try:
        decoded = json.loads(text)
    except json.JSONDecodeError:
        try:
            decoded = [
                json.loads(line)
                for line in text.splitlines()
                if line.strip()
            ]
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"excluded level file is not valid JSON or JSONL: {source}"
            ) from exc
    items = decoded if isinstance(decoded, list) else [decoded]
    if not items:
        raise ValueError(f"excluded level file is empty: {source}")
    result = []
    for index, item in enumerate(items):
        if not isinstance(item, dict):
            raise ValueError(
                f"excluded level file {source} item {index} is not an object")
        # Hard-pool records and supervised replay records both wrap the source
        # level under ``level``.  Accept either so a confirmation pool can
        # explicitly exclude the data used to train the checkpoint under test.
        if "hard_eval_pool_schema_version" in item or "level" in item:
            item = item.get("level")
            if not isinstance(item, dict):
                raise ValueError(
                    f"excluded level file {source} item {index} has no level")
        result.append(item)
    return result


def _load_excluded_level_sources(
    paths: tuple[str, ...],
) -> tuple[set[str], list[dict[str, Any]]]:
    signatures: set[str] = set()
    sources = []
    for path in paths:
        items = _level_items_from_file(path)
        source_signatures = {
            static_level_signature(level_from_dict(item))
            for item in items
        }
        signatures.update(source_signatures)
        sources.append({
            "path": str(path),
            "sha256": sha256_file(path),
            "level_count": len(items),
            "unique_signature_count": len(source_signatures),
        })
    return signatures, sources


def _parse_counts(text: str | None) -> dict[str, int] | None:
    if not text:
        return None
    result: dict[str, int] = {}
    for item in text.split(","):
        name, sep, raw_count = item.partition("=")
        name = name.strip()
        if not sep:
            raise ValueError("per-bucket counts must use name=count entries")
        if not name:
            raise ValueError("per-bucket counts require a bucket name")
        if name in result:
            raise ValueError(f"duplicate per-bucket count for {name!r}")
        count = int(raw_count)
        if count <= 0:
            raise ValueError("per-bucket counts must be positive")
        result[name] = count
    return result


def _target_counts(cfg: HardPoolConfig) -> dict[str, int]:
    if not cfg.buckets:
        raise ValueError("buckets must contain at least one bucket")
    if len(set(cfg.buckets)) != len(cfg.buckets):
        raise ValueError("buckets must not contain duplicates")
    unknown_buckets = set(cfg.buckets) - SUPPORTED_BUCKETS
    if unknown_buckets:
        raise ValueError(f"unknown hard-eval buckets: {unknown_buckets}")
    if cfg.total_count is not None and cfg.per_bucket_counts is not None:
        raise ValueError(
            "total_count and per_bucket_counts are mutually exclusive")
    if cfg.per_bucket_counts:
        unknown = set(cfg.per_bucket_counts) - set(cfg.buckets)
        if unknown:
            raise ValueError(f"per-bucket counts named unknown buckets: {unknown}")
        for bucket, count in cfg.per_bucket_counts.items():
            if isinstance(count, bool) or not isinstance(count, int) or count <= 0:
                raise ValueError(
                    f"per-bucket count for {bucket!r} must be a positive integer")
        return dict(cfg.per_bucket_counts)
    if (isinstance(cfg.total_count, bool)
            or not isinstance(cfg.total_count, int)
            or cfg.total_count <= 0):
        raise ValueError("provide --total-count or --per-bucket-counts")
    base = cfg.total_count // len(cfg.buckets)
    extra = cfg.total_count % len(cfg.buckets)
    return {
        bucket: count
        for index, bucket in enumerate(cfg.buckets)
        if (count := base + (1 if index < extra else 0)) > 0
    }


def _validate_config(cfg: HardPoolConfig, counts: dict[str, int]) -> None:
    for name, value in (("rows", cfg.rows), ("cols", cfg.cols),
                        ("color_count", cfg.color_count),
                        ("oracle_validation_budget",
                         cfg.oracle_validation_budget),
                        ("protagonist_simulations",
                         cfg.protagonist_simulations),
                        ("protagonist_trials", cfg.protagonist_trials),
                        ("max_attempts_per_level",
                         cfg.max_attempts_per_level),
                        ("checkpoint_every_attempts",
                         cfg.checkpoint_every_attempts)):
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"{name} must be a positive integer")
    if not 0.0 <= cfg.density_min <= cfg.density_max <= 1.0:
        raise ValueError("density range must be ordered within [0, 1]")
    if (cfg.oracle_validation_time_limit_seconds is not None
            and (isinstance(cfg.oracle_validation_time_limit_seconds, bool)
                 or not isinstance(
                     cfg.oracle_validation_time_limit_seconds, (int, float))
                 or not math.isfinite(
                     float(cfg.oracle_validation_time_limit_seconds))
                 or cfg.oracle_validation_time_limit_seconds <= 0)):
        raise ValueError(
            "oracle_validation_time_limit_seconds must be finite and positive")
    if (isinstance(cfg.mutation_budget_min, bool)
            or isinstance(cfg.mutation_budget_max, bool)
            or not isinstance(cfg.mutation_budget_min, int)
            or not isinstance(cfg.mutation_budget_max, int)
            or not 0 <= cfg.mutation_budget_min <= cfg.mutation_budget_max):
        raise ValueError(
            "mutation budget range must contain ordered non-negative integers")
    if (counts.get("adversarial_designer_hard", 0) > 0
            and not cfg.designer_checkpoint):
        raise ValueError(
            "the adversarial_designer_hard bucket requires "
            "--designer-checkpoint; it is never generated by the random baseline")
    has_difficulty = (cfg.difficulty_min is not None
                      or cfg.difficulty_max is not None)
    if has_difficulty and not cfg.protagonist_checkpoint:
        raise ValueError(
            "difficulty-band filtering requires --protagonist-checkpoint")


def _resolve_device(name: str):
    import torch
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def _bucket_parameters(
    cfg: HardPoolConfig, bucket: str, rng: random.Random
) -> tuple[GeneratorConfig, int, dict[str, Any]]:
    rows, cols = cfg.rows, cfg.cols
    color_count = cfg.color_count
    mutation_min, mutation_max = cfg.mutation_budget_min, cfg.mutation_budget_max
    if bucket == "ood_larger_board":
        rows += 1
        cols += 1
    elif bucket == "ood_more_colors":
        color_count += 1
    elif bucket == "mutation_stress":
        mutation_min = max(mutation_min, (mutation_min + mutation_max) // 2)
    density = rng.uniform(cfg.density_min, cfg.density_max)
    mutation_budget = rng.randint(mutation_min, mutation_max)
    gen_cfg = GeneratorConfig(
        rows=rows, cols=cols, color_count=color_count, density=density)
    params = {
        "rows": rows,
        "cols": cols,
        "color_count": color_count,
        "density": density,
        "mutation_budget": mutation_budget,
        "bucket": bucket,
    }
    return gen_cfg, mutation_budget, params


def _load_protagonist(cfg: HardPoolConfig):
    if not cfg.protagonist_checkpoint:
        return None
    if cfg.difficulty_min is None or cfg.difficulty_max is None:
        raise ValueError(
            "protagonist filtering requires --difficulty-band MIN MAX")
    if not 0.0 <= cfg.difficulty_min <= cfg.difficulty_max <= 1.0:
        raise ValueError("difficulty band must be within [0, 1]")
    from ..training.checkpoint import (
        configs_from_checkpoint, load_checkpoint, model_from_checkpoint)
    from ..designer.roles import Protagonist

    device = _resolve_device(cfg.device)
    checkpoint = load_checkpoint(cfg.protagonist_checkpoint, map_location="cpu")
    enc, _model_cfg, value_norm = configs_from_checkpoint(checkpoint)
    model = model_from_checkpoint(checkpoint, map_location=device)
    return Protagonist(
        Environment(), model, enc, value_norm, device,
        simulations=cfg.protagonist_simulations)


def _load_designer_generator(cfg: HardPoolConfig, *, required: bool):
    if not required:
        return None
    from ..designer.checkpoint import designer_from_checkpoint, load_designer

    device = _resolve_device(cfg.device)
    checkpoint = load_designer(cfg.designer_checkpoint, map_location="cpu")
    model, encoding, _model_config = designer_from_checkpoint(
        checkpoint, map_location=device)
    return {
        "model": model,
        "encoding": encoding,
        "device": device,
        "checkpoint_sha256": sha256_file(cfg.designer_checkpoint),
    }


def _designer_level(bundle, gen_cfg, *, mutation_budget, generation_seed,
                    rng):
    from ..designer.actions import DesignerActionSpace
    from ..designer.env import DesignerEnv
    from ..designer.ppo import rollout_episode

    encoding = bundle["encoding"]
    designer_env = DesignerEnv(
        gen_cfg, mutation_budget=mutation_budget, encoding=encoding)
    action_space = DesignerActionSpace(encoding)
    episode = rollout_episode(
        designer_env, bundle["model"], action_space, encoding,
        seed=generation_seed, device=bundle["device"], rng=rng,
        verify_finalize=False)
    return episode.finalize.level if episode.finalize.valid else None


def _partial_checkpoint_path(destination: Path) -> Path:
    return destination.with_name(destination.name + ".partial.json")


def _generation_identity_sha256(
    cfg: HardPoolConfig,
    counts: dict[str, int],
    *,
    protagonist_checkpoint_sha256: str | None,
    designer_checkpoint_sha256: str | None,
    excluded_level_sources: list[dict[str, Any]],
) -> str:
    config = asdict(cfg)
    for operational_field in (
            "output", "max_attempts_per_level",
            "checkpoint_every_attempts", "resume"):
        config.pop(operational_field, None)
    return _canonical_sha256({
        "hard_eval_pool_schema_version": HARD_EVAL_POOL_SCHEMA_VERSION,
        "generation_config": config,
        "resolved_bucket_counts": counts,
        "protagonist_checkpoint_sha256": protagonist_checkpoint_sha256,
        "designer_checkpoint_sha256": designer_checkpoint_sha256,
        "excluded_level_sources": excluded_level_sources,
    })


def _validated_partial_checkpoint(
    path: Path,
    *,
    expected_identity_sha256: str,
    counts: dict[str, int],
) -> dict[str, Any]:
    try:
        checkpoint = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise RuntimeError(
            f"hard-pool partial checkpoint is unreadable: {path}") from exc
    if checkpoint.get("partial_checkpoint_schema_version") != \
            PARTIAL_CHECKPOINT_SCHEMA_VERSION:
        raise RuntimeError(
            "hard-pool partial checkpoint has an unsupported schema version")
    if checkpoint.get("generation_identity_sha256") != \
            expected_identity_sha256:
        raise RuntimeError(
            "hard-pool partial checkpoint belongs to different generation "
            "settings or checkpoint contents")
    if checkpoint.get("target_counts") != counts:
        raise RuntimeError(
            "hard-pool partial checkpoint target counts do not match")

    attempts = checkpoint.get("attempts_by_bucket")
    if not isinstance(attempts, dict) or set(attempts) != set(counts):
        raise RuntimeError(
            "hard-pool partial checkpoint has invalid attempt counters")
    if any(isinstance(value, bool) or not isinstance(value, int) or value < 0
           for value in attempts.values()):
        raise RuntimeError(
            "hard-pool partial checkpoint has invalid attempt counters")
    for name in ("duplicates_rejected", "excluded_rejected", "oracle_rejected",
                 "protagonist_rejected"):
        value = checkpoint.get(name)
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise RuntimeError(
                f"hard-pool partial checkpoint has invalid {name}")

    records = checkpoint.get("records")
    if not isinstance(records, list):
        raise RuntimeError(
            "hard-pool partial checkpoint has invalid records")
    bucket_order = {bucket: index for index, bucket in enumerate(counts)}
    accepted_by_bucket = {bucket: 0 for bucket in counts}
    seen: set[str] = set()
    prior_bucket_index = 0
    for index, record in enumerate(records):
        if not isinstance(record, dict):
            raise RuntimeError(
                f"hard-pool partial checkpoint record {index} is invalid")
        bucket = record.get("generation_bucket")
        if bucket not in counts:
            raise RuntimeError(
                f"hard-pool partial checkpoint record {index} has an "
                "unexpected bucket")
        current_bucket_index = bucket_order[bucket]
        if current_bucket_index < prior_bucket_index:
            raise RuntimeError(
                "hard-pool partial checkpoint records are out of bucket order")
        prior_bucket_index = current_bucket_index
        accepted_by_bucket[bucket] += 1
        if accepted_by_bucket[bucket] > counts[bucket]:
            raise RuntimeError(
                f"hard-pool partial checkpoint exceeds target for {bucket}")
        try:
            level = level_from_dict(record["level"])
        except (KeyError, TypeError, ValueError) as exc:
            raise RuntimeError(
                f"hard-pool partial checkpoint record {index} has an "
                "invalid level") from exc
        signature = static_level_signature(level)
        if record.get("static_level_signature") != signature:
            raise RuntimeError(
                f"hard-pool partial checkpoint record {index} has a "
                "signature mismatch")
        if signature in seen:
            raise RuntimeError(
                "hard-pool partial checkpoint contains a duplicate signature")
        seen.add(signature)
        if record.get("canonical_level_sha256") != _canonical_sha256(
                level_to_dict(level)):
            raise RuntimeError(
                f"hard-pool partial checkpoint record {index} has a "
                "level hash mismatch")
    for bucket, bucket_index in bucket_order.items():
        if bucket_index < prior_bucket_index \
                and accepted_by_bucket[bucket] != counts[bucket]:
            raise RuntimeError(
                "hard-pool partial checkpoint advanced before completing "
                f"bucket {bucket}")
    return checkpoint


def generate_hard_eval_pool(cfg: HardPoolConfig) -> dict[str, Any]:
    destination = Path(cfg.output)
    if destination.exists():
        raise FileExistsError(f"hard eval pool already exists: {destination}")
    partial_path = _partial_checkpoint_path(destination)
    if partial_path.exists() and not cfg.resume:
        raise FileExistsError(
            "hard eval pool has a partial checkpoint; resume it or move/remove "
            f"the checkpoint after inspection: {partial_path}")
    counts = _target_counts(cfg)
    _validate_config(cfg, counts)
    excluded_signatures, excluded_level_sources = \
        _load_excluded_level_sources(cfg.exclude_level_files)
    env = Environment()
    oracle = Oracle(
        env, max_nodes=cfg.oracle_validation_budget,
        time_limit_seconds=cfg.oracle_validation_time_limit_seconds)
    protagonist = _load_protagonist(cfg)
    designer = _load_designer_generator(
        cfg, required=counts.get("adversarial_designer_hard", 0) > 0)
    protagonist_checkpoint_sha256 = (
        sha256_file(cfg.protagonist_checkpoint)
        if cfg.protagonist_checkpoint else None)
    designer_checkpoint_sha256 = (
        designer["checkpoint_sha256"] if designer is not None else None)
    generation_identity_sha256 = _generation_identity_sha256(
        cfg, counts,
        protagonist_checkpoint_sha256=protagonist_checkpoint_sha256,
        designer_checkpoint_sha256=designer_checkpoint_sha256,
        excluded_level_sources=excluded_level_sources)
    rng = random.Random(cfg.seed)
    seen: set[str] = set()
    records: list[dict[str, Any]] = []
    attempts_by_bucket: dict[str, int] = {bucket: 0 for bucket in counts}
    duplicates = excluded_rejected = oracle_rejected = protagonist_rejected = 0
    resumed_from_partial = False
    if partial_path.exists():
        partial = _validated_partial_checkpoint(
            partial_path,
            expected_identity_sha256=generation_identity_sha256,
            counts=counts)
        records = partial["records"]
        attempts_by_bucket = partial["attempts_by_bucket"]
        duplicates = partial["duplicates_rejected"]
        excluded_rejected = partial["excluded_rejected"]
        oracle_rejected = partial["oracle_rejected"]
        protagonist_rejected = partial["protagonist_rejected"]
        seen = {record["static_level_signature"] for record in records}
        for _ in range(sum(attempts_by_bucket.values())):
            rng.randrange(2**63)
        resumed_from_partial = True
        print(
            f"resuming hard-pool generation from {partial_path}: "
            f"{len(records)}/{sum(counts.values())} accepted, "
            f"{sum(attempts_by_bucket.values())} attempts",
            flush=True)

    def save_partial() -> None:
        atomic_write_json(partial_path, {
            "partial_checkpoint_schema_version":
                PARTIAL_CHECKPOINT_SCHEMA_VERSION,
            "generation_identity_sha256": generation_identity_sha256,
            "target_counts": counts,
            "attempts_by_bucket": attempts_by_bucket,
            "duplicates_rejected": duplicates,
            "excluded_rejected": excluded_rejected,
            "oracle_rejected": oracle_rejected,
            "protagonist_rejected": protagonist_rejected,
            "records": records,
        })

    accepted_by_bucket = {
        bucket: sum(record["generation_bucket"] == bucket
                    for record in records)
        for bucket in counts
    }
    total_attempts_at_checkpoint = sum(attempts_by_bucket.values())
    try:
        for bucket, target in counts.items():
            accepted = accepted_by_bucket[bucket]
            max_attempts = target * cfg.max_attempts_per_level
            if accepted < target:
                print(
                    f"hard-pool bucket {bucket}: resuming at "
                    f"{accepted}/{target} accepted, "
                    f"{attempts_by_bucket[bucket]}/{max_attempts} attempts",
                    flush=True)
            while accepted < target \
                    and attempts_by_bucket[bucket] < max_attempts:
                attempts_by_bucket[bucket] += 1
                generation_seed = rng.randrange(2**63)
                local_rng = random.Random(generation_seed)
                gen_cfg, mutation_budget, params = _bucket_parameters(
                    cfg, bucket, local_rng)
                if bucket == "adversarial_designer_hard":
                    level = _designer_level(
                        designer, gen_cfg, mutation_budget=mutation_budget,
                        generation_seed=generation_seed, rng=local_rng)
                    generation_method = "trained_designer"
                else:
                    level = random_level(
                        env, gen_cfg, local_rng, reverse_depth=mutation_budget)
                    generation_method = "random_reverse_construction"
                if level is not None:
                    level_data = level_to_dict(level)
                    signature = static_level_signature(level)
                    if signature in excluded_signatures:
                        excluded_rejected += 1
                    elif signature in seen:
                        duplicates += 1
                    else:
                        protagonist_status = None
                        retained = True
                        if protagonist is not None:
                            from .frontier import estimate_solve_rate

                            estimate = estimate_solve_rate(
                                protagonist, level,
                                trials=cfg.protagonist_trials,
                                base_seed=cfg.seed,
                                evaluation_context=(
                                    "cotraining.hard_eval_pool"))
                            retained = (
                                cfg.difficulty_min
                                <= estimate.solve_rate
                                <= cfg.difficulty_max)
                            protagonist_status = {
                                "enabled": True,
                                "checkpoint": cfg.protagonist_checkpoint,
                                "checkpoint_sha256":
                                    protagonist_checkpoint_sha256,
                                "difficulty_band": [
                                    cfg.difficulty_min,
                                    cfg.difficulty_max],
                                "simulations":
                                    cfg.protagonist_simulations,
                                "trials": estimate.trials,
                                "solved": estimate.solved,
                                "solve_rate": estimate.solve_rate,
                                "retained": retained,
                            }
                            if not retained:
                                protagonist_rejected += 1
                        if retained:
                            state = env.initial_state(level)
                            value = oracle.value(state)
                            oracle_status = {
                                "max_nodes": cfg.oracle_validation_budget,
                                "time_limit_seconds":
                                    cfg.oracle_validation_time_limit_seconds,
                                "exact": value.exact,
                                "solvable": value.solvable,
                                "optimal_remaining_moves": value.value,
                            }
                            if not (value.exact and value.solvable):
                                oracle_rejected += 1
                            else:
                                record = {
                                    "hard_eval_pool_schema_version":
                                        HARD_EVAL_POOL_SCHEMA_VERSION,
                                    "level_id": signature,
                                    "static_level_signature": signature,
                                    "canonical_level_sha256":
                                        _canonical_sha256(level_data),
                                    "generation_bucket": bucket,
                                    "generation_seed": generation_seed,
                                    "generation_parameters": {
                                        **params,
                                        "generation_method":
                                            generation_method,
                                        "global_seed": cfg.seed,
                                        "oracle_validation_budget":
                                            cfg.oracle_validation_budget,
                                        "designer_checkpoint": (
                                            cfg.designer_checkpoint
                                            if generation_method
                                            == "trained_designer" else None),
                                        "designer_checkpoint_sha256": (
                                            designer[
                                                "checkpoint_sha256"]
                                            if generation_method
                                            == "trained_designer" else None),
                                    },
                                    "oracle_validation": oracle_status,
                                    "protagonist_filter":
                                        protagonist_status,
                                    "level": level_data,
                                }
                                seen.add(signature)
                                records.append(record)
                                accepted += 1
                                accepted_by_bucket[bucket] = accepted
                                save_partial()
                                total_attempts_at_checkpoint = sum(
                                    attempts_by_bucket.values())
                                print(
                                    f"hard-pool bucket {bucket}: accepted "
                                    f"{accepted}/{target} after "
                                    f"{attempts_by_bucket[bucket]} attempts "
                                    f"(total {len(records)}/"
                                    f"{sum(counts.values())})",
                                    flush=True)
                current_total_attempts = sum(attempts_by_bucket.values())
                if current_total_attempts - total_attempts_at_checkpoint \
                        >= cfg.checkpoint_every_attempts:
                    save_partial()
                    total_attempts_at_checkpoint = current_total_attempts
                    print(
                        f"hard-pool bucket {bucket}: checkpointed "
                        f"{accepted}/{target} accepted after "
                        f"{attempts_by_bucket[bucket]} attempts",
                        flush=True)
            if accepted < target:
                raise RuntimeError(
                    f"could only generate {accepted}/{target} records for "
                    f"{bucket}; partial progress is saved at {partial_path}; "
                    "increase --max-attempts-per-level or relax filters")
    except BaseException:
        save_partial()
        print(
            f"hard-pool generation stopped; resumable progress saved at "
            f"{partial_path}",
            flush=True)
        raise

    text = "".join(
        json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n"
        for record in records)
    atomic_write_text(destination, text)
    if partial_path.exists():
        partial_path.unlink()
    return {
        "output": str(destination),
        "records": len(records),
        "pool_sha256": canonical_eval_pool_sha256(records),
        "buckets": counts,
        "attempts_by_bucket": attempts_by_bucket,
        "duplicates_rejected": duplicates,
        "excluded_rejected": excluded_rejected,
        "excluded_unique_signature_count": len(excluded_signatures),
        "excluded_level_sources": excluded_level_sources,
        "oracle_rejected": oracle_rejected,
        "protagonist_rejected": protagonist_rejected,
        "resumed_from_partial": resumed_from_partial,
        "generation_config": asdict(cfg),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate a hard held-out evaluation pool JSONL.")
    parser.add_argument("--output", required=True)
    parser.add_argument("--total-count", type=int, default=None)
    parser.add_argument("--per-bucket-counts", default=None,
                        help="comma-separated bucket=count entries")
    parser.add_argument("--buckets", nargs="+", default=list(DEFAULT_BUCKETS))
    parser.add_argument("--seed", type=int, default=1729)
    parser.add_argument("--rows", type=int, default=6)
    parser.add_argument("--cols", type=int, default=6)
    parser.add_argument("--color-count", type=int, default=3)
    parser.add_argument("--density-range", type=float, nargs=2,
                        default=(0.45, 0.65))
    parser.add_argument("--mutation-budget-range", type=int, nargs=2,
                        default=(8, 24))
    parser.add_argument("--oracle-validation-budget", type=int, default=200_000)
    parser.add_argument(
        "--oracle-validation-time-limit-seconds", type=float, default=30.0,
        help="per-candidate exact-oracle time cap (0 = no limit)")
    parser.add_argument(
        "--designer-checkpoint", default=None,
        help="required when generating the adversarial_designer_hard bucket")
    parser.add_argument("--protagonist-checkpoint", default=None)
    parser.add_argument(
        "--exclude-levels", action="append", default=[],
        help="JSON level list or hard-pool JSONL whose levels must be excluded; "
             "repeat for multiple files")
    parser.add_argument("--difficulty-band", type=float, nargs=2, default=None)
    parser.add_argument("--protagonist-simulations", type=int, default=4)
    parser.add_argument("--protagonist-trials", type=int, default=5)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--max-attempts-per-level", type=int, default=200)
    parser.add_argument(
        "--checkpoint-every-attempts", type=int, default=100,
        help="persist progress after this many attempts without an acceptance")
    parser.add_argument(
        "--no-resume", action="store_false", dest="resume",
        help="refuse to use an existing .partial.json checkpoint")
    parser.set_defaults(resume=True)
    return parser


def config_from_args(args: argparse.Namespace) -> HardPoolConfig:
    difficulty = args.difficulty_band or (None, None)
    return HardPoolConfig(
        output=args.output,
        total_count=args.total_count,
        per_bucket_counts=_parse_counts(args.per_bucket_counts),
        buckets=tuple(args.buckets),
        seed=args.seed,
        rows=args.rows,
        cols=args.cols,
        color_count=args.color_count,
        density_min=args.density_range[0],
        density_max=args.density_range[1],
        mutation_budget_min=args.mutation_budget_range[0],
        mutation_budget_max=args.mutation_budget_range[1],
        oracle_validation_budget=args.oracle_validation_budget,
        oracle_validation_time_limit_seconds=(
            None if args.oracle_validation_time_limit_seconds <= 0
            else args.oracle_validation_time_limit_seconds),
        designer_checkpoint=args.designer_checkpoint,
        protagonist_checkpoint=args.protagonist_checkpoint,
        exclude_level_files=tuple(args.exclude_levels),
        difficulty_min=difficulty[0],
        difficulty_max=difficulty[1],
        protagonist_simulations=args.protagonist_simulations,
        protagonist_trials=args.protagonist_trials,
        device=args.device,
        max_attempts_per_level=args.max_attempts_per_level,
        checkpoint_every_attempts=args.checkpoint_every_attempts,
        resume=args.resume,
    )


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    summary = generate_hard_eval_pool(config_from_args(args))
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
