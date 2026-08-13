"""Deterministic, level-grouped train/validation/test splitting.

States from the same level are highly correlated, so the split is by *level*
(static signature, falling back to level_id), never by individual state. A split
manifest is saved so the same levels stay in the same partition across runs.
"""

from __future__ import annotations

import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

SPLIT_MANIFEST_VERSION = 1


@dataclass(frozen=True)
class SplitRatios:
    train: float = 0.8
    validation: float = 0.1
    test: float = 0.1

    def validate(self) -> None:
        for name, value in (
            ("train", self.train),
            ("validation", self.validation),
            ("test", self.test),
        ):
            if not isinstance(value, (int, float)) or isinstance(value, bool):
                raise ValueError(f"split ratio {name} must be numeric; got {value!r}")
            if not math.isfinite(value) or not 0.0 <= value <= 1.0:
                raise ValueError(
                    f"split ratio {name} must be finite and in [0, 1]; got {value!r}")
        total = self.train + self.validation + self.test
        if not math.isclose(total, 1.0, rel_tol=0.0, abs_tol=1e-6):
            raise ValueError(f"split ratios must sum to 1.0 (got {total})")


def group_key(record: dict[str, Any]) -> str:
    """The level-identifying key for grouping (signature preferred)."""
    return record.get("static_level_signature") or record["level_id"]


def collect_level_keys(records: list[dict[str, Any]]) -> list[str]:
    """Unique level keys in deterministic (sorted) order."""
    return sorted({group_key(r) for r in records})


def make_split(
    level_keys: list[str],
    *,
    ratios: SplitRatios = SplitRatios(),
    seed: int = 42,
) -> dict[str, Any]:
    """Partition unique level keys into train/validation/test deterministically."""
    ratios.validate()
    keys = sorted(set(level_keys))
    rng = random.Random(seed)
    shuffled = keys[:]
    rng.shuffle(shuffled)

    n = len(shuffled)
    n_val = int(n * ratios.validation)
    n_test = int(n * ratios.test)
    n_train = n - n_val - n_test

    train = sorted(shuffled[:n_train])
    validation = sorted(shuffled[n_train:n_train + n_val])
    test = sorted(shuffled[n_train + n_val:])

    if not 0 <= n_train <= n_train + n_val <= n:
        raise ValueError(
            "split partition boundaries are inconsistent: "
            f"train_end={n_train}, validation_end={n_train + n_val}, total={n}")
    if set(train) & set(validation) or set(train) & set(test) \
            or set(validation) & set(test):
        raise ValueError("split construction produced overlapping partitions")
    if set(train) | set(validation) | set(test) != set(keys):
        raise ValueError("split construction did not cover every level exactly once")

    return {
        "version": SPLIT_MANIFEST_VERSION,
        "seed": seed,
        "group_key": "static_level_signature",
        "ratios": {"train": ratios.train, "validation": ratios.validation,
                   "test": ratios.test},
        "train_levels": train,
        "validation_levels": validation,
        "test_levels": test,
    }


def save_manifest(manifest: dict[str, Any], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def load_manifest(path: str | Path) -> dict[str, Any]:
    manifest = json.loads(Path(path).read_text(encoding="utf-8"))
    if manifest.get("version") != SPLIT_MANIFEST_VERSION:
        raise ValueError(f"unsupported split manifest version {manifest.get('version')}")
    ratio_data = manifest.get("ratios")
    if not isinstance(ratio_data, dict):
        raise ValueError("split manifest is missing ratio metadata")
    missing = {"train", "validation", "test"} - set(ratio_data)
    if missing:
        raise ValueError(
            f"split manifest is missing ratios: {', '.join(sorted(missing))}")
    SplitRatios(
        train=ratio_data["train"],
        validation=ratio_data["validation"],
        test=ratio_data["test"],
    ).validate()
    _assert_disjoint(manifest)
    return manifest


def _assert_disjoint(manifest: dict[str, Any]) -> None:
    partitions = {}
    for name in ("train_levels", "validation_levels", "test_levels"):
        values = manifest.get(name)
        if not isinstance(values, list):
            raise ValueError(f"split manifest field {name} must be a list")
        if len(values) != len(set(values)):
            raise ValueError(f"split manifest field {name} contains duplicates")
        partitions[name] = set(values)
    train = partitions["train_levels"]
    val = partitions["validation_levels"]
    test = partitions["test_levels"]
    if train & val or train & test or val & test:
        raise ValueError("split manifest partitions overlap")


def split_of_key(manifest: dict[str, Any], key: str) -> str | None:
    if key in set(manifest["train_levels"]):
        return "train"
    if key in set(manifest["validation_levels"]):
        return "validation"
    if key in set(manifest["test_levels"]):
        return "test"
    return None


def filter_records_for_split(
    records: list[dict[str, Any]], manifest: dict[str, Any], split: str
) -> list[dict[str, Any]]:
    wanted = {
        "train": set(manifest["train_levels"]),
        "validation": set(manifest["validation_levels"]),
        "test": set(manifest["test_levels"]),
    }[split]
    return [r for r in records if group_key(r) in wanted]
