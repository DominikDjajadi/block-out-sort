"""Solve a single level with exact A* and optionally print the solution.

    python -m blocksort.cli.solve --levels fixtures/levels.json --level-index 0 --show-solution
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from typing import Optional

from ..environment import Environment
from ..solution import describe_solution
from ..solver import solve_astar, solve_bfs
from ..dataset.generate import load_levels


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Solve a level with exact A*.")
    p.add_argument("--levels", required=True, help="path to a levels JSON file")
    p.add_argument("--level-index", type=int, default=0)
    p.add_argument("--max-nodes", type=int, default=250_000)
    p.add_argument("--time-limit", type=float, default=None,
                   help="optional wall-clock limit in seconds")
    p.add_argument("--show-solution", action="store_true")
    p.add_argument("--bfs", action="store_true", help="also run BFS and check agreement")
    return p


def main(argv: Optional[list[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    levels = load_levels(args.levels)
    if not 0 <= args.level_index < len(levels):
        print(f"level-index out of range (0..{len(levels) - 1})", file=sys.stderr)
        return 2

    level_id, level = levels[args.level_index]
    env = Environment()
    state = env.initial_state(level)
    result = solve_astar(env, state, max_nodes=args.max_nodes,
                         time_limit_seconds=args.time_limit)

    summary = {
        "level_id": level_id,
        "name": level.name,
        "solvable": result.solvable,
        "optimal": result.optimal,
        "exhausted": result.exhausted,
        "move_count": result.move_count,
        "termination_reason": result.termination_reason,
        "states_explored": result.states_explored,
        "states_generated": result.states_generated,
        "duplicate_states": result.duplicate_states,
        "max_frontier_size": result.max_frontier_size,
        "elapsed_seconds": round(result.elapsed_seconds, 6),
    }

    if args.bfs:
        bfs = solve_bfs(env, state, max_nodes=args.max_nodes)
        summary["bfs_move_count"] = bfs.move_count
        summary["bfs_agrees"] = (bfs.move_count == result.move_count)

    print(json.dumps(summary, indent=2))

    if args.show_solution and result.actions is not None:
        print("\nSolution:")
        print(describe_solution(result.actions, env, state))
    return 0


if __name__ == "__main__":
    sys.exit(main())
