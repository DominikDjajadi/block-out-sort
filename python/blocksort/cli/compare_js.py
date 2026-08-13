"""Structured JavaScript vs Python solver parity comparison.

Runs ``tools/export_js_solver_results.js`` (JSON in/out) and the Python A*
solver on the same levels and compares **solvability**, **minimum move count**,
and **exit-only status** (explored-state counts are intentionally not compared,
since tie-breaking and dedup differ). On mismatch, full diagnostics are emitted.

    python -m blocksort.cli.compare_js --levels fixtures/levels.json
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Optional

from ..environment import Environment
from ..serialization import level_from_dict, level_to_dict
from ..solution import verify_solution
from ..solver import solve_astar, solve_exit_only

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_JS_TOOL = REPO_ROOT / "tools" / "export_js_solver_results.js"


def _run_js(level_dicts: list[dict], node_exe: str, js_tool: Path, max_nodes: int) -> list[dict]:
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False, encoding="utf-8") as tmp:
        json.dump(level_dicts, tmp)
        tmp_path = tmp.name
    try:
        proc = subprocess.run(
            [node_exe, str(js_tool), tmp_path, str(max_nodes)],
            capture_output=True, text=True, encoding="utf-8", check=True,
        )
    finally:
        Path(tmp_path).unlink(missing_ok=True)
    return json.loads(proc.stdout)


def compare_level_dicts(
    level_dicts: list[dict],
    *,
    node_exe: str = "node",
    js_tool: Path = DEFAULT_JS_TOOL,
    max_nodes: int = 250_000,
) -> dict[str, Any]:
    """Compare JS and Python solvers on a list of level dicts. Returns a report."""
    js_results = _run_js(level_dicts, node_exe, Path(js_tool), max_nodes)
    env = Environment()
    entries: list[dict[str, Any]] = []
    mismatch_count = 0

    for i, raw in enumerate(level_dicts):
        level = level_from_dict(raw)
        state = env.initial_state(level)
        py = solve_astar(env, state, max_nodes=max_nodes)
        py_exit_only = solve_exit_only(env, state)
        replay_ok = (
            py.actions is not None
            and verify_solution(env, state, py.actions, py.move_count)
        )
        js = js_results[i] if i < len(js_results) else {}

        mismatches: list[str] = []
        # Solvability (skip if either side is unknown / budget-hit).
        if py.solvable is not None and js.get("solvable") is not None:
            if bool(py.solvable) != bool(js["solvable"]):
                mismatches.append(
                    f"solvable: py={py.solvable} js={js['solvable']}")
        # Minimum move count (only when both solved).
        if py.solvable is True and js.get("solvable") is True:
            if py.move_count != js.get("minMoves"):
                mismatches.append(
                    f"minMoves: py={py.move_count} js={js.get('minMoves')}")
        # Exit-only status.
        if py.solvable is not None:
            if bool(py_exit_only) != bool(js.get("exitOnly")):
                mismatches.append(
                    f"exitOnly: py={py_exit_only} js={js.get('exitOnly')}")
        # Returned Python solution must replay (when solved).
        if py.solvable is True and not replay_ok:
            mismatches.append("python solution failed replay verification")

        entry: dict[str, Any] = {
            "index": i,
            "name": raw.get("name"),
            "match": not mismatches,
            "python": {
                "solvable": py.solvable,
                "move_count": py.move_count,
                "exit_only": py_exit_only,
                "termination_reason": py.termination_reason,
                "replay_ok": replay_ok,
            },
            "javascript": js,
        }
        if mismatches:
            mismatch_count += 1
            entry["mismatches"] = mismatches
            entry["level"] = level_to_dict(level)
            entry["python_solution"] = list(py.serialized_actions or ())
        entries.append(entry)

    return {
        "levels": len(level_dicts),
        "mismatches": mismatch_count,
        "results": entries,
    }


def _load_level_dicts(path: str | Path) -> list[dict]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(data, dict) and "levels" in data:
        data = data["levels"]
    if isinstance(data, dict):
        data = [data]
    return data


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Compare JS and Python solvers.")
    p.add_argument("--levels", required=True, help="path to a levels JSON file")
    p.add_argument("--node", default="node", help="node executable")
    p.add_argument("--js-tool", default=str(DEFAULT_JS_TOOL))
    p.add_argument("--max-nodes", type=int, default=250_000)
    return p


def main(argv: Optional[list[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    level_dicts = _load_level_dicts(args.levels)
    report = compare_level_dicts(
        level_dicts, node_exe=args.node, js_tool=Path(args.js_tool),
        max_nodes=args.max_nodes,
    )
    print(json.dumps(report, indent=2))
    return 0 if report["mismatches"] == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
