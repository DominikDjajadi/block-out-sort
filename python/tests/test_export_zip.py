from __future__ import annotations

import importlib.util
import json
import zipfile
from pathlib import Path


_SCRIPT = Path(__file__).resolve().parents[2] / "tools" / "export_zip.py"
_SPEC = importlib.util.spec_from_file_location("gameslop_export_zip", _SCRIPT)
assert _SPEC is not None and _SPEC.loader is not None
export_zip = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(export_zip)


def _write(root: Path, relative: str, value: str = "content") -> None:
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(value, encoding="utf-8")


def test_collect_files_uses_allowlist_and_excludes_artifacts(tmp_path):
    _write(tmp_path, "README.md")
    _write(tmp_path, "js/game.js")
    _write(tmp_path, "python/blocksort/core.py")
    _write(tmp_path, "python/blocksort/.env", "SECRET=1")
    _write(tmp_path, "python/blocksort/model.pt")
    _write(tmp_path, "python/runs/report.py")
    _write(tmp_path, "notes/private.txt")
    _write(tmp_path, "data/training/pv_smoke.jsonl", "{}\n")

    files = {
        path.as_posix()
        for path in export_zip.collect_files(tmp_path)
    }

    assert files == {
        "README.md",
        "data/training/pv_smoke.jsonl",
        "js/game.js",
        "python/blocksort/core.py",
    }


def test_archive_is_deterministic_and_manifest_is_verified(tmp_path):
    _write(tmp_path, "README.md", "hello\n")
    _write(tmp_path, "js/game.js", "console.log('hello');\n")
    first = tmp_path / "first.zip"
    second = tmp_path / "second.zip"

    result_one = export_zip.create_archive(
        tmp_path, first, include_sample_data=False)
    result_two = export_zip.create_archive(
        tmp_path, second, include_sample_data=False)

    assert result_one["sha256"] == result_two["sha256"]
    assert first.read_bytes() == second.read_bytes()
    with zipfile.ZipFile(first) as archive:
        manifest = json.loads(
            archive.read("GameSlop/EXPORT-MANIFEST.json"))
        assert manifest["kind"] == "gameslop_source_export"
        assert manifest["file_count"] == 2
        assert manifest["sample_data_included"] is False
        assert {item["path"] for item in manifest["files"]} == {
            "README.md",
            "js/game.js",
        }
