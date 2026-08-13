"""Disk-backed level replay buffer (JSONL shards + manifest, no database).

Stores accepted generated levels with their full provenance: the level, designer
trajectory, immutable designer model-state identity, informational checkpoint
paths, oracle result, reward components, structural metrics, solver metrics,
generation iteration, and a structural fingerprint. Exact duplicates (same
fingerprint) are rejected. When over capacity the least valuable levels are
evicted first, preserving historically difficult, frontier (oracle-solved /
protagonist-failed), and novel levels.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Optional

from ..environment import Environment
from ..schema import Level
from ..serialization import level_from_dict, level_to_dict
from ..signature import static_level_signature
from ..state import canonical_key
from ..training.transaction import atomic_write_json, atomic_write_text

REPLAY_VERSION = 1
VERIFIED_MODEL_STATE = "model_state_verified"
LEGACY_UNVERIFIED = "legacy_unverified"


def level_fingerprint(env: Environment, level: Level) -> str:
    sig = static_level_signature(level)
    key = canonical_key(env.initial_state(level))
    return hashlib.sha256(f"{sig}|{key}".encode("utf-8")).hexdigest()


def build_level_record(
    env: Environment,
    level: Level,
    *,
    trajectory: list[int],
    designer_checkpoint: Optional[str],
    generator_model_state_sha256: str,
    protagonist_checkpoint: Optional[str],
    oracle_result: dict[str, Any],
    reward_components: dict[str, Any],
    structural_metrics: dict[str, Any],
    solver_metrics: dict[str, Any],
    generation_iteration: int,
    reward_total: float,
) -> dict[str, Any]:
    if not generator_model_state_sha256:
        raise ValueError("generator_model_state_sha256 is required")
    fp = level_fingerprint(env, level)
    frontier = bool(oracle_result.get("oracle_solved")) and not bool(
        oracle_result.get("protagonist_solved"))
    novelty = float(reward_components.get("novelty", 0.0))
    retention = reward_total + (1.0 if frontier else 0.0) + 0.5 * novelty
    return {
        "version": REPLAY_VERSION,
        "fingerprint": fp,
        "static_level_signature": static_level_signature(level),
        "level": level_to_dict(level),
        "trajectory": list(trajectory),
        # Informational only: this path may be moved, reused, or overwritten.
        "designer_checkpoint": designer_checkpoint,
        # Authoritative identity of the exact parameters used for generation.
        "generator_model_state_sha256": generator_model_state_sha256,
        "provenance_status": VERIFIED_MODEL_STATE,
        "protagonist_checkpoint": protagonist_checkpoint,
        "oracle_result": oracle_result,
        "reward_components": reward_components,
        "structural_metrics": structural_metrics,
        "solver_metrics": solver_metrics,
        "generation_iteration": generation_iteration,
        "reward_total": reward_total,
        "frontier": frontier,
        "retention": retention,
    }


class LevelReplayBuffer:
    def __init__(self, root: str | Path, *, max_levels: int = 5_000,
                 seed: int = 42) -> None:
        self.root = Path(root)
        self.shards_dir = self.root / "shards"
        self.max_levels = max_levels
        self.seed = seed
        self._records: dict[str, dict[str, Any]] = {}
        self._iterations: list[int] = []

    # ---- persistence ----

    def load(self) -> "LevelReplayBuffer":
        manifest = self.root / "manifest.json"
        if not manifest.exists():
            return self
        meta = json.loads(manifest.read_text(encoding="utf-8"))
        self.max_levels = meta.get("max_levels", self.max_levels)
        self.seed = meta.get("seed", self.seed)
        for it in meta.get("iterations", []):
            shard = self.shards_dir / f"iter_{it:03d}.jsonl"
            if not shard.exists():
                raise ValueError(
                    f"level-replay manifest references missing shard: {shard}")
            for line in shard.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if line:
                    rec = json.loads(line)
                    if rec.get("generator_model_state_sha256") is None:
                        rec["generator_model_state_sha256"] = None
                        rec["provenance_status"] = LEGACY_UNVERIFIED
                    else:
                        rec.setdefault(
                            "provenance_status", VERIFIED_MODEL_STATE)
                    self._records[rec["fingerprint"]] = rec
        self._iterations = list(meta.get("iterations", []))
        return self

    def load_snapshot(self, path: str | Path) -> "LevelReplayBuffer":
        path = Path(path)
        if not path.is_file():
            raise ValueError(f"committed level-replay snapshot is missing: {path}")
        self._records.clear()
        self._iterations.clear()
        for number, line in enumerate(
                path.read_text(encoding="utf-8").splitlines(), start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
                self._records[record["fingerprint"]] = record
            except (TypeError, KeyError, json.JSONDecodeError) as exc:
                raise ValueError(
                    f"invalid committed level-replay snapshot {path} "
                    f"at line {number}") from exc
        self._iterations = sorted({
            int(record["generation_iteration"])
            for record in self._records.values()})
        return self

    def write_snapshot(self, path: str | Path) -> None:
        rows = sorted(self._records.values(), key=lambda record: record["fingerprint"])
        atomic_write_text(
            path,
            "".join(json.dumps(record, sort_keys=True) + "\n" for record in rows))

    def persist(self) -> None:
        self.shards_dir.mkdir(parents=True, exist_ok=True)
        iters = sorted({r["generation_iteration"] for r in self._records.values()})
        for it in iters:
            shard = self.shards_dir / f"iter_{it:03d}.jsonl"
            rows = [r for r in self._records.values()
                    if r["generation_iteration"] == it]
            atomic_write_text(
                shard, "".join(json.dumps(r) + "\n" for r in rows))
        self._iterations = iters
        manifest = {
            "version": REPLAY_VERSION, "seed": self.seed,
            "max_levels": self.max_levels, "iterations": iters,
            "total_levels": len(self._records),
            "frontier_levels": sum(1 for r in self._records.values()
                                   if r.get("frontier")),
        }
        self.root.mkdir(parents=True, exist_ok=True)
        atomic_write_json(self.root / "manifest.json", manifest)

    # ---- mutation ----

    def add(self, records: Iterable[dict[str, Any]]) -> dict[str, int]:
        stats = {"added": 0, "duplicates": 0}
        for rec in records:
            fp = rec["fingerprint"]
            if fp in self._records:
                stats["duplicates"] += 1
                continue
            self._records[fp] = rec
            stats["added"] += 1
        self._evict_if_needed()
        return stats

    def _evict_if_needed(self) -> None:
        if len(self._records) <= self.max_levels:
            return
        ordered = sorted(self._records.items(),
                         key=lambda kv: (kv[1].get("retention", 0.0),
                                         kv[1].get("generation_iteration", 0)))
        excess = len(self._records) - self.max_levels
        for fp, _ in ordered[:excess]:
            del self._records[fp]

    # ---- views ----

    def __len__(self) -> int:
        return len(self._records)

    def records(self) -> list[dict[str, Any]]:
        return list(self._records.values())

    def fingerprints(self) -> set[str]:
        return set(self._records.keys())

    def levels(self) -> list[Level]:
        return [level_from_dict(r["level"]) for r in self._records.values()]
