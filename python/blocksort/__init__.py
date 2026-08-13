"""Block Out Sort — Python game environment and ML stack.

Core engine: a correct, well-tested port of the JavaScript rules (environment,
solver, validation). Extended packages add supervised training, neural-guided
search, expert iteration, adversarial level design, and co-training — see
README for CLI entry points.
"""

from __future__ import annotations

from .actions import Action
from .environment import Environment, IllegalActionError, SlideResult
from .schema import (
    COLORS,
    EDGES,
    Block,
    Cell,
    Direction,
    Exit,
    Level,
    LockedRegion,
)
from .serialization import (
    dumps_level,
    level_from_dict,
    level_to_dict,
    load_level,
    loads_level,
)
from .state import State, canonical_key
from .validation import ValidationError, validate_level, validate_level_data, is_valid
from .signature import static_level_signature, static_level_payload
from .solver import (
    SolveResult,
    solve_astar,
    solve_bfs,
    solve_exit_only,
    SOLVED,
    UNSOLVABLE,
    NODE_LIMIT,
    TIME_LIMIT,
    DEPTH_LIMIT,
    INVALID,
)
from .solution import (
    describe_solution,
    deserialize_action,
    deserialize_solution,
    replay_solution,
    serialize_action,
    serialize_solution,
    verify_solution,
)
from .oracle import Oracle, StateAnalysis, ActionAnalysis, ValueResult

__all__ = [
    "Action",
    "ActionAnalysis",
    "Block",
    "COLORS",
    "Cell",
    "DEPTH_LIMIT",
    "Direction",
    "EDGES",
    "Environment",
    "Exit",
    "IllegalActionError",
    "INVALID",
    "Level",
    "LockedRegion",
    "NODE_LIMIT",
    "Oracle",
    "SOLVED",
    "SlideResult",
    "SolveResult",
    "State",
    "StateAnalysis",
    "TIME_LIMIT",
    "UNSOLVABLE",
    "ValidationError",
    "ValueResult",
    "canonical_key",
    "describe_solution",
    "deserialize_action",
    "deserialize_solution",
    "dumps_level",
    "is_valid",
    "level_from_dict",
    "level_to_dict",
    "load_level",
    "loads_level",
    "replay_solution",
    "serialize_action",
    "serialize_solution",
    "solve_astar",
    "solve_bfs",
    "solve_exit_only",
    "static_level_payload",
    "static_level_signature",
    "validate_level",
    "validate_level_data",
    "verify_solution",
]
