"""Small, Windows-safe primitives for committed training state.

Run state is the transaction boundary.  Files referenced by it are immutable
and content-addressed; mutable compatibility files are repaired mirrors only.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any


class CheckpointIntegrityError(RuntimeError):
    """A committed checkpoint is missing or does not match its recorded hash."""


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _temporary_path(destination: Path) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent)
    os.close(fd)
    return Path(name)


def atomic_write_bytes(destination: str | Path, data: bytes) -> None:
    destination = Path(destination)
    temporary = _temporary_path(destination)
    try:
        with temporary.open("wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()


def atomic_write_text(destination: str | Path, text: str) -> None:
    atomic_write_bytes(destination, text.encode("utf-8"))


def atomic_write_json(destination: str | Path, value: Any) -> None:
    atomic_write_text(destination, json.dumps(value, indent=2) + "\n")


def atomic_copy(source: str | Path, destination: str | Path) -> None:
    source, destination = Path(source), Path(destination)
    temporary = _temporary_path(destination)
    try:
        with source.open("rb") as src, temporary.open("wb") as dst:
            shutil.copyfileobj(src, dst)
            dst.flush()
            os.fsync(dst.fileno())
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()


def relative_to_run(path: str | Path, run_dir: str | Path) -> str:
    return Path(path).resolve().relative_to(Path(run_dir).resolve()).as_posix()


def resolve_run_path(run_dir: str | Path, relative: str) -> Path:
    root = Path(run_dir).resolve()
    resolved = (root / relative).resolve()
    if not resolved.is_relative_to(root):
        raise ValueError(f"committed path escapes run directory: {relative!r}")
    return resolved


def resolve_committed_protagonist(
    run_dir: str | Path, run_state: dict[str, Any]
) -> Path:
    relative = run_state.get("active_protagonist_checkpoint")
    expected = run_state.get("active_protagonist_sha256")
    if not relative or not expected:
        raise ValueError("run state has no committed protagonist checkpoint identity")
    path = resolve_run_path(run_dir, relative)
    if not path.is_file():
        raise CheckpointIntegrityError(
            f"committed protagonist checkpoint is missing: {path}")
    observed = sha256_file(path)
    if observed != expected:
        raise CheckpointIntegrityError(
            "committed protagonist checkpoint integrity failure: "
            f"path={path}, expected sha256={expected}, observed sha256={observed}")
    return path


def refresh_best_checkpoint(
    source: str | Path, best_path: str | Path, expected_sha256: str
) -> bool:
    """Repair the convenience mirror.  Return whether it changed."""
    best = Path(best_path)
    if best.is_file() and sha256_file(best) == expected_sha256:
        return False
    atomic_copy(source, best)
    observed = sha256_file(best)
    if observed != expected_sha256:
        raise CheckpointIntegrityError(
            f"best.pt repair failed: expected {expected_sha256}, observed {observed}")
    return True
