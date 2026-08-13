"""Disk-backed replay buffer (JSONL shards + manifest, no database).

Layout under ``<root>/``::

    manifest.json          # version, seed, counts, per-iteration shard list
    shards/iter_000.jsonl  # records added in iteration 0 (the base dataset)
    shards/iter_001.jsonl  # records added in iteration 1
    ...

The in-memory view is a dict keyed by ``(static_signature, state_key)``. On
collision, label priority is full exact > exact path > neural search. Eviction
(when over ``max_examples``) drops, in order, the least valuable records first:
search, exact path, then full exact; within a source, easier and older records
go first so older *difficult* states are preserved.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Iterable, Optional

from .records import (
    SOURCE_EXACT,
    SOURCE_EXACT_PATH,
    SOURCE_SEARCH,
    dedup_key,
    difficulty,
    ensure_record_label_metadata,
)
from ..training.transaction import atomic_write_json, atomic_write_text

REPLAY_VERSION = 1

REPLAY_AGE_CURRENT = "current_round"
REPLAY_AGE_RECENT = "recent_rounds"
REPLAY_AGE_HISTORICAL = "historical"
REPLAY_AGE_BUCKETS = (
    REPLAY_AGE_CURRENT,
    REPLAY_AGE_RECENT,
    REPLAY_AGE_HISTORICAL,
)


def replay_age_bucket(
    record: dict[str, Any],
    *,
    current_iteration: int,
    recent_window: int,
) -> str:
    """Classify replay provenance relative to the active training round."""
    iteration = int(record.get("generation_iteration", 0))
    if current_iteration > 0 and iteration == current_iteration:
        return REPLAY_AGE_CURRENT
    if (current_iteration > 0 and 0 < iteration < current_iteration
            and iteration >= current_iteration - recent_window):
        return REPLAY_AGE_RECENT
    return REPLAY_AGE_HISTORICAL


def _sims(record: dict[str, Any]) -> int:
    search = record.get("search") or {}
    return int(search.get("simulations", 0))


def _source_quality(record: dict[str, Any]) -> int:
    """Replay priority: neural search < exact path < complete exact policy."""
    return {
        SOURCE_SEARCH: 0,
        SOURCE_EXACT_PATH: 1,
        SOURCE_EXACT: 2,
    }.get(record.get("target_source", SOURCE_EXACT), 0)


class ReplayBuffer:
    def __init__(self, root: str | Path, *, max_examples: int = 50_000,
                 seed: int = 42) -> None:
        self.root = Path(root)
        self.shards_dir = self.root / "shards"
        self.max_examples = max_examples
        self.seed = seed
        self._records: dict[tuple[str, str], dict[str, Any]] = {}
        self._iterations: list[int] = []

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def load(self) -> "ReplayBuffer":
        manifest = self.root / "manifest.json"
        if not manifest.exists():
            return self
        meta = json.loads(manifest.read_text(encoding="utf-8"))
        self.max_examples = meta.get("max_examples", self.max_examples)
        self.seed = meta.get("seed", self.seed)
        self._records.clear()
        for it in meta.get("iterations", []):
            shard = self.shards_dir / f"iter_{it:03d}.jsonl"
            if not shard.exists():
                raise ValueError(
                    f"replay manifest references missing shard: {shard}")
            for line in shard.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if line:
                    rec = ensure_record_label_metadata(json.loads(line))
                    self._records[dedup_key(rec)] = rec
        self._iterations = list(meta.get("iterations", []))
        return self

    def load_snapshot(
        self,
        path: str | Path,
        *,
        rebase_as_historical: bool = False,
    ) -> "ReplayBuffer":
        """Load a full replay view.

        A snapshot imported into a new experiment can be rebased to iteration
        zero. This preserves its original iteration as provenance while
        ensuring old examples are weighted as historical rather than being
        mistaken for fresh data in the new run's first rounds.
        """
        path = Path(path)
        if not path.is_file():
            raise ValueError(f"committed replay snapshot is missing: {path}")
        self._records.clear()
        self._iterations.clear()
        for number, line in enumerate(
                path.read_text(encoding="utf-8").splitlines(), start=1):
            if not line.strip():
                continue
            try:
                record = ensure_record_label_metadata(json.loads(line))
                if rebase_as_historical:
                    record = dict(record)
                    record.setdefault(
                        "imported_generation_iteration",
                        int(record.get("generation_iteration", 0)),
                    )
                    record["generation_iteration"] = 0
                self._records[dedup_key(record)] = record
                iteration = int(record.get("generation_iteration", 0))
                if iteration not in self._iterations:
                    self._iterations.append(iteration)
            except (TypeError, ValueError, KeyError, json.JSONDecodeError) as exc:
                raise ValueError(
                    f"invalid committed replay snapshot {path} at line {number}"
                ) from exc
        return self

    def write_snapshot(self, path: str | Path) -> None:
        rows = sorted(
            self._records.values(),
            key=lambda record: tuple(str(part) for part in dedup_key(record)))
        text = "".join(
            json.dumps(record, sort_keys=True) + "\n" for record in rows)
        atomic_write_text(path, text)

    def _save_manifest(self) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        manifest = {
            "version": REPLAY_VERSION,
            "seed": self.seed,
            "max_examples": self.max_examples,
            "iterations": sorted(set(self._iterations)),
            "total_examples": len(self._records),
            "counts_by_source": self.counts_by_source(),
        }
        atomic_write_json(self.root / "manifest.json", manifest)

    def _write_iteration_shard(self, iteration: int) -> None:
        self.shards_dir.mkdir(parents=True, exist_ok=True)
        shard = self.shards_dir / f"iter_{iteration:03d}.jsonl"
        rows = [r for r in self._records.values()
                if r.get("generation_iteration", 0) == iteration]
        atomic_write_text(
            shard, "".join(json.dumps(r) + "\n" for r in rows))

    def persist(self, iterations: Iterable[int]) -> None:
        """Rewrite all referenced iteration shards + the manifest from memory."""
        wanted = sorted(set(self._iterations) | set(iterations))
        for it in wanted:
            self._write_iteration_shard(it)
        self._iterations = wanted
        self._save_manifest()

    # ------------------------------------------------------------------
    # Mutation
    # ------------------------------------------------------------------

    def add(self, records: Iterable[dict[str, Any]], iteration: int) -> dict[str, int]:
        """Add records with dedup + exact priority. Returns add statistics."""
        stats = {"added": 0, "deduped": 0, "upgraded": 0, "kept_existing": 0}
        for new in records:
            new = ensure_record_label_metadata(new)
            key = dedup_key(new)
            old = self._records.get(key)
            if old is None:
                self._records[key] = new
                stats["added"] += 1
                continue
            stats["deduped"] += 1
            if self._should_replace(old, new):
                # Preserve the original discovery iteration for shard stability
                # unless the new record is a strict exact upgrade.
                self._records[key] = new
                stats["upgraded"] += 1
            else:
                # Merge provenance so we keep a trail of how the state arose.
                old.setdefault("provenance", []).extend(new.get("provenance", []))
                stats["kept_existing"] += 1
        if iteration not in self._iterations:
            self._iterations.append(iteration)
        self._evict_if_needed()
        return stats

    @staticmethod
    def _should_replace(old: dict[str, Any], new: dict[str, Any]) -> bool:
        old_quality = _source_quality(old)
        new_quality = _source_quality(new)
        if new_quality > old_quality:
            return True
        if old_quality > new_quality:
            return False
        if old_quality == 0:
            # Both search: prefer more simulations (stronger teacher).
            return _sims(new) > _sims(old)
        return False                  # same exact quality: keep first (stable)

    def _evict_if_needed(self) -> None:
        if len(self._records) <= self.max_examples:
            return
        # Ascending = least valuable first.  Drop search before exact, easy
        # before hard, old before new.
        ordered = sorted(
            self._records.items(),
            key=lambda kv: (
                _source_quality(kv[1]),
                difficulty(kv[1]),
                kv[1].get("generation_iteration", 0),
            ),
        )
        excess = len(self._records) - self.max_examples
        for key, _ in ordered[:excess]:
            del self._records[key]

    # ------------------------------------------------------------------
    # Views
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self._records)

    def records(self) -> list[dict[str, Any]]:
        return list(self._records.values())

    def counts_by_source(self) -> dict[str, int]:
        out = {SOURCE_EXACT: 0, SOURCE_EXACT_PATH: 0, SOURCE_SEARCH: 0}
        for r in self._records.values():
            out[r.get("target_source", SOURCE_EXACT)] = out.get(
                r.get("target_source", SOURCE_EXACT), 0) + 1
        return out

    # ------------------------------------------------------------------
    # Sampling
    # ------------------------------------------------------------------

    def sample_training_set(
        self,
        size: int,
        *,
        current_iteration: int,
        weight_exact_historical: float,
        weight_exact_new: float,
        weight_search: float,
        seed: int,
        exclude_keys: Optional[set] = None,
        with_replacement: bool = True,
    ) -> list[dict[str, Any]]:
        """Weighted, level/iteration-balanced sample.

        Source weights set the base mass; per-level and per-iteration inverse
        frequencies balance the draw so no single level/iteration dominates.
        Deterministic for a fixed ``seed``. When ``with_replacement`` is false,
        every eligible replay record is used at most once before any record is
        repeated.
        """
        exclude = exclude_keys or set()
        pool = [
            record
            for record in self._records.values()
            if dedup_key(record) not in exclude
        ]
        return self._weighted_sample(
            pool,
            size,
            current_iteration=current_iteration,
            weight_exact_historical=weight_exact_historical,
            weight_exact_new=weight_exact_new,
            weight_search=weight_search,
            seed=seed,
            with_replacement=with_replacement,
        )

    @staticmethod
    def _weighted_sample(
        pool: list[dict[str, Any]],
        size: int,
        *,
        current_iteration: int,
        weight_exact_historical: float,
        weight_exact_new: float,
        weight_search: float,
        seed: int,
        with_replacement: bool,
    ) -> list[dict[str, Any]]:
        """Apply source/level/iteration balancing within an explicit pool."""
        import random

        if not pool or size <= 0:
            return []
        if not with_replacement:
            # Snapshot loading is key-sorted whereas a fresh run retains insert
            # order. Sort here so interruption/resume cannot change the draw.
            pool = list(pool)
            pool.sort(key=lambda record: tuple(
                str(part) for part in dedup_key(record)))

        level_counts: dict[str, int] = {}
        iter_counts: dict[int, int] = {}
        for r in pool:
            level_counts[r["static_level_signature"]] = level_counts.get(
                r["static_level_signature"], 0) + 1
            it = r.get("generation_iteration", 0)
            iter_counts[it] = iter_counts.get(it, 0) + 1

        weights = []
        for r in pool:
            src = r.get("target_source", SOURCE_EXACT)
            it = r.get("generation_iteration", 0)
            if src == SOURCE_SEARCH:
                base = weight_search
            elif it >= current_iteration and current_iteration > 0:
                base = weight_exact_new
            else:
                base = weight_exact_historical
            lvl_balance = 1.0 / level_counts[r["static_level_signature"]]
            iter_balance = 1.0 / iter_counts[it]
            weights.append(max(0.0, base) * lvl_balance * iter_balance)

        total = sum(weights)
        if total <= 0:
            weights = [1.0] * len(pool)
        rng = random.Random(seed)
        if with_replacement:
            return rng.choices(pool, weights=weights, k=size)

        eligible = [
            (index, weight)
            for index, weight in enumerate(weights)
            if weight > 0
        ]
        if not eligible:
            return []
        unique_count = min(size, len(eligible))
        # Efraimidis-Spirakis weighted sampling without replacement. Sorting
        # by the random key also gives a deterministic weighted sample order.
        ranked = sorted(
            eligible,
            key=lambda item: math.log(max(rng.random(), 1e-300)) / item[1],
            reverse=True,
        )
        selected = [pool[index] for index, _weight in ranked[:unique_count]]
        if unique_count < size:
            positive_pool = [pool[index] for index, _weight in eligible]
            positive_weights = [weight for _index, weight in eligible]
            selected.extend(rng.choices(
                positive_pool,
                weights=positive_weights,
                k=size - unique_count,
            ))
        return selected

    def sample_training_set_with_age_quotas(
        self,
        size: int,
        *,
        current_iteration: int,
        current_fraction: float,
        recent_fraction: float,
        historical_fraction: float,
        recent_window: int,
        weight_exact_historical: float,
        weight_exact_new: float,
        weight_search: float,
        seed: int,
        exclude_keys: Optional[set] = None,
        with_replacement: bool = False,
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        """Sample explicit age buckets, redistributing only empty-bucket quota.

        Each non-empty bucket receives its configured share after fractions for
        empty buckets are renormalized away. Sampling without replacement uses
        every distinct record in a bucket before repeating, so a small current
        pool can still contribute its intended sample share. Source-aware loss
        weights determine the realized gradient mass after selection.
        """
        if isinstance(size, bool) or not isinstance(size, int) or size <= 0:
            raise ValueError("replay quota sample size must be a positive integer")
        if (isinstance(recent_window, bool)
                or not isinstance(recent_window, int) or recent_window < 0):
            raise ValueError("replay recent window must be a non-negative integer")

        fractions = {
            REPLAY_AGE_CURRENT: float(current_fraction),
            REPLAY_AGE_RECENT: float(recent_fraction),
            REPLAY_AGE_HISTORICAL: float(historical_fraction),
        }
        if any(not math.isfinite(value) or value < 0 for value in fractions.values()):
            raise ValueError("replay age fractions must be finite and non-negative")
        if not math.isclose(
                sum(fractions.values()), 1.0, rel_tol=0.0, abs_tol=1e-6):
            raise ValueError("replay age fractions must sum to 1.0")

        exclude = exclude_keys or set()
        buckets = {name: [] for name in REPLAY_AGE_BUCKETS}
        for record in self._records.values():
            if dedup_key(record) in exclude:
                continue
            bucket = replay_age_bucket(
                record,
                current_iteration=current_iteration,
                recent_window=recent_window,
            )
            buckets[bucket].append(record)

        active = [
            name for name in REPLAY_AGE_BUCKETS
            if buckets[name] and fractions[name] > 0
        ]
        if not active:
            summary = {
                "policy": "fresh_recent_historical_quota_v1",
                "recent_window": recent_window,
                "configured_fractions": fractions,
                "available_records": {
                    name: len(buckets[name]) for name in REPLAY_AGE_BUCKETS},
                "target_counts": {name: 0 for name in REPLAY_AGE_BUCKETS},
                "realized_counts": {name: 0 for name in REPLAY_AGE_BUCKETS},
                "unique_counts": {name: 0 for name in REPLAY_AGE_BUCKETS},
                "realized_fractions": {name: 0.0 for name in REPLAY_AGE_BUCKETS},
            }
            return [], summary

        active_mass = sum(fractions[name] for name in active)
        raw_counts = {
            name: size * fractions[name] / active_mass for name in active}
        target_counts = {name: 0 for name in REPLAY_AGE_BUCKETS}
        for name in active:
            target_counts[name] = int(math.floor(raw_counts[name]))
        remainder = size - sum(target_counts.values())
        ranked_remainders = sorted(
            active,
            key=lambda name: (
                -(raw_counts[name] - target_counts[name]),
                REPLAY_AGE_BUCKETS.index(name),
            ),
        )
        for index in range(remainder):
            target_counts[ranked_remainders[index % len(ranked_remainders)]] += 1

        sampled: list[dict[str, Any]] = []
        realized_counts = {name: 0 for name in REPLAY_AGE_BUCKETS}
        unique_counts = {name: 0 for name in REPLAY_AGE_BUCKETS}
        for index, name in enumerate(REPLAY_AGE_BUCKETS):
            count = target_counts[name]
            bucket_sample = self._weighted_sample(
                buckets[name],
                count,
                current_iteration=current_iteration,
                weight_exact_historical=weight_exact_historical,
                weight_exact_new=weight_exact_new,
                weight_search=weight_search,
                seed=(seed * 1_000_003 + index) & 0xFFFFFFFF,
                with_replacement=with_replacement,
            )
            sampled.extend(bucket_sample)
            realized_counts[name] = len(bucket_sample)
            unique_counts[name] = len({
                dedup_key(record) for record in bucket_sample})

        total = len(sampled)
        summary = {
            "policy": "fresh_recent_historical_quota_v1",
            "recent_window": recent_window,
            "configured_fractions": fractions,
            "available_records": {
                name: len(buckets[name]) for name in REPLAY_AGE_BUCKETS},
            "target_counts": target_counts,
            "realized_counts": realized_counts,
            "unique_counts": unique_counts,
            "realized_fractions": {
                name: (realized_counts[name] / total if total else 0.0)
                for name in REPLAY_AGE_BUCKETS
            },
        }
        return sampled, summary

    def sample_value_training_set(
        self,
        size: int,
        *,
        current_iteration: int,
        weight_exact_historical: float,
        weight_exact_new: float,
        weight_search: float,
        seed: int,
        depth_fractions: tuple[float, float, float, float] = (
            0.35, 0.30, 0.25, 0.10),
        exclude_keys: Optional[set] = None,
    ) -> list[dict[str, Any]]:
        """Sample without replacement using explicit value-depth quotas.

        The four depth buckets are 1-3, 4-6, 7-9, and 10+ raw moves.
        Sampling remains source-, level-, and iteration-aware *within* each
        bucket. If a bucket cannot fill its quota, the unused capacity is
        offered to the remaining buckets from hardest to easiest.
        """
        import random

        if isinstance(size, bool) or not isinstance(size, int) or size <= 0:
            raise ValueError("value sample size must be a positive integer")
        if (len(depth_fractions) != 4
                or any(not math.isfinite(value) or value < 0
                       for value in depth_fractions)
                or sum(depth_fractions) <= 0):
            raise ValueError(
                "value depth fractions must contain four finite, "
                "non-negative values with positive total mass")

        excluded = exclude_keys or set()
        buckets: list[list[dict[str, Any]]] = [[], [], [], []]
        for record in self._records.values():
            if dedup_key(record) in excluded:
                continue
            raw_moves = float(
                record["value_target"]["raw_optimal_moves"])
            if not math.isfinite(raw_moves) or raw_moves < 0:
                raise ValueError(
                    f"invalid replay value target: {raw_moves}")
            if raw_moves <= 3:
                bucket = 0
            elif raw_moves <= 6:
                bucket = 1
            elif raw_moves < 10:
                bucket = 2
            else:
                bucket = 3
            buckets[bucket].append(record)

        normalized = [
            fraction / sum(depth_fractions)
            for fraction in depth_fractions
        ]
        raw_quotas = [size * fraction for fraction in normalized]
        quotas = [math.floor(value) for value in raw_quotas]
        remainder = size - sum(quotas)
        remainder_order = sorted(
            range(4),
            key=lambda index: (
                raw_quotas[index] - quotas[index], index),
            reverse=True,
        )
        for index in remainder_order[:remainder]:
            quotas[index] += 1

        rng = random.Random(seed)
        ranked_buckets: list[list[dict[str, Any]]] = []
        for bucket in buckets:
            bucket.sort(key=lambda record: tuple(
                str(part) for part in dedup_key(record)))
            level_counts: dict[str, int] = {}
            iteration_counts: dict[int, int] = {}
            for record in bucket:
                signature = record["static_level_signature"]
                iteration = int(record.get("generation_iteration", 0))
                level_counts[signature] = level_counts.get(signature, 0) + 1
                iteration_counts[iteration] = (
                    iteration_counts.get(iteration, 0) + 1)

            eligible = []
            for record in bucket:
                source = record.get("target_source", SOURCE_EXACT)
                iteration = int(record.get("generation_iteration", 0))
                if source == SOURCE_SEARCH:
                    base = weight_search
                elif current_iteration > 0 and iteration >= current_iteration:
                    base = weight_exact_new
                else:
                    base = weight_exact_historical
                weight = (
                    max(0.0, base)
                    / level_counts[record["static_level_signature"]]
                    / iteration_counts[iteration]
                )
                if weight > 0:
                    key = math.log(max(rng.random(), 1e-300)) / weight
                    eligible.append((key, record))
            eligible.sort(key=lambda item: item[0], reverse=True)
            ranked_buckets.append([record for _key, record in eligible])

        selected_by_bucket: list[list[dict[str, Any]]] = []
        for ranked, quota in zip(ranked_buckets, quotas):
            selected_by_bucket.append(ranked[:quota])

        selected_count = sum(len(bucket) for bucket in selected_by_bucket)
        target_count = min(
            size, sum(len(bucket) for bucket in ranked_buckets))
        deficit = target_count - selected_count
        if deficit > 0:
            # Preserve hard-state representation when redistributing quota.
            for index in (3, 2, 1, 0):
                already = len(selected_by_bucket[index])
                available = ranked_buckets[index][already:]
                take = min(deficit, len(available))
                selected_by_bucket[index].extend(available[:take])
                deficit -= take
                if deficit == 0:
                    break

        return [
            record
            for bucket in selected_by_bucket
            for record in bucket
        ]
