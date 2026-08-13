"""Create a clean, reproducible source archive for GameSlop.

This script uses only the Python standard library.  It intentionally exports
an allowlisted source tree instead of copying the repository and trying to
remove dangerous or bulky files afterward.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import tempfile
import zipfile
from pathlib import Path, PurePosixPath


SCHEMA_VERSION = 1
ARCHIVE_TIMESTAMP = (2000, 1, 1, 0, 0, 0)
ROOT_FILES = frozenset({
    ".gitignore",
    "ANALYSIS.md",
    "README.md",
    "index.html",
    "styles.css",
    "python/pyproject.toml",
    "python/requirements-dev.txt",
})
SOURCE_DIRECTORIES = (
    "docs",
    "fixtures",
    "js",
    "python/blocksort",
    "python/tests",
    "tools",
)
SAMPLE_DATA_FILES = frozenset({
    "data/training/pv_examples.jsonl",
    "data/training/pv_smoke.jsonl",
    "data/training/smoke_levels.json",
})
EXCLUDED_DIRECTORY_NAMES = frozenset({
    ".agents",
    ".codex",
    ".git",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    ".venv",
    "__pycache__",
    "build",
    "dist",
    "node_modules",
    "runs",
    "venv",
})
EXCLUDED_SUFFIXES = frozenset({
    ".7z", ".ckpt", ".key", ".log", ".npy", ".npz", ".onnx",
    ".pem", ".pkl", ".pt", ".pth", ".pyc", ".pyo", ".tar",
    ".tmp", ".zip",
})
SENSITIVE_NAMES = frozenset({
    ".npmrc",
    ".pypirc",
    "credentials.json",
    "id_ed25519",
    "id_rsa",
    "secrets.json",
})


def _portable(path: Path) -> str:
    return path.as_posix()


def _is_source_path(relative: Path, *, include_sample_data: bool) -> bool:
    portable = _portable(relative)
    if portable in ROOT_FILES:
        return True
    if include_sample_data and portable in SAMPLE_DATA_FILES:
        return True
    return any(
        portable == directory or portable.startswith(directory + "/")
        for directory in SOURCE_DIRECTORIES
    )


def _is_safe_path(relative: Path) -> bool:
    lower_parts = tuple(part.lower() for part in relative.parts)
    if any(part in EXCLUDED_DIRECTORY_NAMES for part in lower_parts[:-1]):
        return False
    name = lower_parts[-1]
    if name == ".env" or name.startswith(".env."):
        return False
    if name in SENSITIVE_NAMES or name.endswith(".local"):
        return False
    return relative.suffix.lower() not in EXCLUDED_SUFFIXES


def collect_files(
    root: Path,
    *,
    include_sample_data: bool = True,
) -> list[Path]:
    """Return sorted, safe source paths relative to ``root``."""
    root = root.resolve()
    candidates: set[Path] = set()
    for value in ROOT_FILES:
        candidates.add(Path(value))
    for directory in SOURCE_DIRECTORIES:
        base = root / directory
        if base.is_dir():
            candidates.update(
                path.relative_to(root)
                for path in base.rglob("*")
                if path.is_file() and not path.is_symlink()
            )
    if include_sample_data:
        candidates.update(Path(value) for value in SAMPLE_DATA_FILES)

    return sorted(
        (
            relative for relative in candidates
            if (root / relative).is_file()
            and not (root / relative).is_symlink()
            and _is_source_path(
                relative, include_sample_data=include_sample_data)
            and _is_safe_path(relative)
        ),
        key=lambda value: _portable(value).lower(),
    )


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _zip_info(name: str) -> zipfile.ZipInfo:
    info = zipfile.ZipInfo(name, ARCHIVE_TIMESTAMP)
    info.compress_type = zipfile.ZIP_DEFLATED
    info.create_system = 3
    info.external_attr = (0o100644 & 0xFFFF) << 16
    return info


def create_archive(
    root: Path,
    output: Path,
    *,
    archive_root: str = "GameSlop",
    include_sample_data: bool = True,
) -> dict[str, object]:
    """Atomically write and verify a deterministic source ZIP."""
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", archive_root):
        raise ValueError(
            "archive root must contain only letters, numbers, dots, "
            "underscores, and hyphens")
    root = root.resolve()
    output = output.resolve()
    files = collect_files(root, include_sample_data=include_sample_data)
    if not files:
        raise RuntimeError("no exportable source files were found")

    entries: list[dict[str, object]] = []
    payloads: list[tuple[Path, bytes]] = []
    for relative in files:
        data = (root / relative).read_bytes()
        payloads.append((relative, data))
        entries.append({
            "path": _portable(relative),
            "bytes": len(data),
            "sha256": _sha256(data),
        })
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "kind": "gameslop_source_export",
        "archive_root": archive_root,
        "sample_data_included": include_sample_data,
        "file_count": len(entries),
        "files": entries,
    }
    manifest_data = (
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")

    output.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=output.stem + ".", suffix=".tmp", dir=output.parent)
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        with zipfile.ZipFile(
            temporary, "w", compression=zipfile.ZIP_DEFLATED,
            compresslevel=9,
        ) as archive:
            for relative, data in payloads:
                archive.writestr(
                    _zip_info(f"{archive_root}/{_portable(relative)}"), data,
                    compresslevel=9,
                )
            archive.writestr(
                _zip_info(f"{archive_root}/EXPORT-MANIFEST.json"),
                manifest_data,
                compresslevel=9,
            )
        _verify_archive(temporary, manifest)
        os.replace(temporary, output)
    finally:
        temporary.unlink(missing_ok=True)

    return {
        "output": str(output),
        "files": len(entries),
        "source_bytes": sum(int(entry["bytes"]) for entry in entries),
        "zip_bytes": output.stat().st_size,
        "sha256": _sha256(output.read_bytes()),
        "sample_data_included": include_sample_data,
    }


def _verify_archive(path: Path, manifest: dict[str, object]) -> None:
    archive_root = str(manifest["archive_root"])
    entries = manifest["files"]
    assert isinstance(entries, list)
    with zipfile.ZipFile(path, "r") as archive:
        if archive.testzip() is not None:
            raise RuntimeError("ZIP integrity verification failed")
        persisted = json.loads(
            archive.read(f"{archive_root}/EXPORT-MANIFEST.json"))
        if persisted != manifest:
            raise RuntimeError("ZIP manifest verification failed")
        for entry in entries:
            assert isinstance(entry, dict)
            data = archive.read(f"{archive_root}/{entry['path']}")
            if len(data) != entry["bytes"] or _sha256(data) != entry["sha256"]:
                raise RuntimeError(
                    f"ZIP content verification failed: {entry['path']}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Create a clean, reproducible GameSlop source ZIP.")
    parser.add_argument(
        "--output", default=None,
        help="archive path (default: dist/GameSlop-source.zip)")
    parser.add_argument(
        "--archive-root", default="GameSlop",
        help="top-level folder inside the ZIP")
    parser.add_argument(
        "--no-sample-data", action="store_true",
        help="omit the three versioned smoke/sample datasets")
    parser.add_argument(
        "--list", action="store_true",
        help="print selected files without creating an archive")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    root = Path(__file__).resolve().parents[1]
    include_sample_data = not args.no_sample_data
    files = collect_files(root, include_sample_data=include_sample_data)
    if args.list:
        for path in files:
            print(_portable(path))
        print(f"\n{len(files)} files")
        return 0

    output = (
        Path(args.output)
        if args.output is not None
        else root / "dist" / "GameSlop-source.zip"
    )
    if not output.is_absolute():
        output = root / output
    result = create_archive(
        root,
        output,
        archive_root=args.archive_root,
        include_sample_data=include_sample_data,
    )
    print(f"Created {result['output']}")
    print(
        f"{result['files']} files; "
        f"{result['zip_bytes'] / (1024 * 1024):.2f} MiB; "
        f"SHA-256 {result['sha256']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
