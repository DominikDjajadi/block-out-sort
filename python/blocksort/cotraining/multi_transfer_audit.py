"""Shared-incumbent transfer audit for several protagonist candidates.

Generated holdout groups are optional, allowing a focused validation-only
screen.  Conversely, ``generated_only`` supports a training-side screen before
the promotion-validation set is consulted.  The final-test role stays sealed.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from ..schema import Level
from ..training.transaction import (
    atomic_write_json,
    atomic_write_text,
    sha256_file,
)
from .transfer_audit import (
    GeneratedGroup,
    _evaluate_checkpoint_group,
    _load_level_list,
    _load_validation_levels,
    _paired_group_summary,
    _parse_generated_group,
    _validate_budgets,
    _validate_partial_rows,
)


SCHEMA_VERSION = 3
SEMANTICS = "multi_candidate_paired_transfer_audit_v3"
_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]*$")


@dataclass(frozen=True)
class Candidate:
    name: str
    checkpoint: str

    def validate(self) -> None:
        if not _NAME.fullmatch(self.name) or self.name == "incumbent":
            raise ValueError(
                "candidate names must contain only letters, numbers, "
                "underscores, and hyphens and cannot be 'incumbent'")
        if not Path(self.checkpoint).is_file():
            raise ValueError(
                f"candidate checkpoint does not exist: {self.checkpoint}")


@dataclass(frozen=True)
class MultiTransferAuditConfig:
    incumbent_checkpoint: str
    candidates: tuple[Candidate, ...]
    generated_groups: tuple[GeneratedGroup, ...]
    eval_levels_dataset: str | None
    eval_split_manifest: str | None
    output_dir: str
    generated_budgets: tuple[int, ...] = (55, 90, 148, 244, 400)
    validation_budgets: tuple[int, ...] = (4, 8, 16, 32, 64)
    c_puct: float = 1.5
    seed: int = 2045
    device: str = "cpu"
    progress_interval: int = 5
    validation_first: bool = False
    gate_validation_budgets: tuple[int, ...] = ()
    gate_validation_weights: tuple[float, ...] = ()
    gate_margin: float = 0.0
    generated_only: bool = False

    def validate(self) -> None:
        for label, path in (("incumbent checkpoint", self.incumbent_checkpoint),):
            if not Path(path).is_file():
                raise ValueError(f"{label} does not exist: {path}")
        if not isinstance(self.generated_only, bool):
            raise ValueError("generated_only must be a boolean")
        if not self.generated_only:
            for label, path in (
                    ("evaluation dataset", self.eval_levels_dataset),
                    ("evaluation split manifest", self.eval_split_manifest)):
                if path is None or not Path(path).is_file():
                    raise ValueError(f"{label} does not exist: {path}")
        elif not self.generated_groups:
            raise ValueError(
                "generated_only requires at least one generated group")
        if not self.candidates:
            raise ValueError("at least one candidate is required")
        names: set[str] = set()
        for candidate in self.candidates:
            candidate.validate()
            if candidate.name in names:
                raise ValueError(f"duplicate candidate name: {candidate.name}")
            names.add(candidate.name)
        group_names: set[str] = set()
        for group in self.generated_groups:
            group.validate()
            if group.name in group_names:
                raise ValueError(f"duplicate generated group name: {group.name}")
            group_names.add(group.name)
        _validate_budgets("generated_budgets", self.generated_budgets)
        if not self.generated_only:
            _validate_budgets("validation_budgets", self.validation_budgets)
        if not isinstance(self.validation_first, bool):
            raise ValueError("validation_first must be a boolean")
        if self.generated_only and self.validation_first:
            raise ValueError(
                "validation_first cannot be used with generated_only")
        if self.generated_only and self.gate_validation_budgets:
            raise ValueError(
                "a validation gate cannot be used with generated_only")
        if self.gate_validation_budgets:
            if not self.validation_first:
                raise ValueError(
                    "validation_first is required when a validation gate is used")
            _validate_budgets(
                "gate_validation_budgets", self.gate_validation_budgets)
            missing = set(self.gate_validation_budgets).difference(
                self.validation_budgets)
            if missing:
                raise ValueError(
                    "gate validation budgets must be included in validation "
                    f"budgets: {sorted(missing)}")
            if (len(self.gate_validation_weights)
                    != len(self.gate_validation_budgets)):
                raise ValueError(
                    "one gate validation weight is required per gate budget")
            if (any(not math.isfinite(weight) or weight < 0
                    for weight in self.gate_validation_weights)
                    or sum(self.gate_validation_weights) <= 0):
                raise ValueError(
                    "gate validation weights must be finite, non-negative, "
                    "and have a positive sum")
        elif self.gate_validation_weights:
            raise ValueError(
                "gate validation budgets are required when weights are supplied")
        if not math.isfinite(self.gate_margin) or self.gate_margin < 0:
            raise ValueError("gate_margin must be finite and non-negative")
        if not math.isfinite(self.c_puct) or self.c_puct < 0:
            raise ValueError("c_puct must be finite and non-negative")
        if isinstance(self.seed, bool) or not isinstance(self.seed, int):
            raise ValueError("seed must be an integer")
        if (isinstance(self.progress_interval, bool)
                or not isinstance(self.progress_interval, int)
                or self.progress_interval <= 0):
            raise ValueError("progress_interval must be a positive integer")


def _parse_candidate(value: str) -> Candidate:
    name, separator, checkpoint = value.partition("=")
    if not separator or not name or not checkpoint:
        raise argparse.ArgumentTypeError(
            "candidates must use NAME=CHECKPOINT syntax")
    return Candidate(name=name, checkpoint=checkpoint)


def _canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"),
        ensure_ascii=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _identity(cfg: MultiTransferAuditConfig) -> dict[str, Any]:
    config = json.loads(json.dumps(asdict(cfg)))
    config.pop("output_dir")
    result = {
        "schema_version": SCHEMA_VERSION,
        "semantics": SEMANTICS,
        "config": config,
        "inputs": {
            "incumbent_checkpoint_sha256":
                sha256_file(cfg.incumbent_checkpoint),
            "candidate_checkpoint_sha256": {
                candidate.name: sha256_file(candidate.checkpoint)
                for candidate in cfg.candidates
            },
            "eval_levels_dataset_sha256": (
                None if cfg.eval_levels_dataset is None
                else sha256_file(cfg.eval_levels_dataset)),
            "eval_split_manifest_sha256": (
                None if cfg.eval_split_manifest is None
                else sha256_file(cfg.eval_split_manifest)),
            "generated_groups": {
                group.name: sha256_file(group.path)
                for group in cfg.generated_groups
            },
        },
    }
    result["fingerprint"] = _canonical_sha256(result)
    return result


def _evaluate_or_resume(
    *,
    checkpoint: str,
    role: str,
    group_name: str,
    levels: list[Level],
    budgets: tuple[int, ...],
    source: dict[str, Any],
    identity: dict[str, Any],
    evaluation_dir: Path,
    cfg: MultiTransferAuditConfig,
) -> list[dict[str, Any]]:
    path = evaluation_dir / f"{group_name}.{role}.json"
    partial_path = path.with_name(path.name + ".partial.json")
    expected_header = {
        "schema_version": SCHEMA_VERSION,
        "semantics": SEMANTICS,
        "experiment_fingerprint": identity["fingerprint"],
        "group": group_name,
        "role": role,
        "checkpoint": checkpoint,
        "checkpoint_sha256": sha256_file(checkpoint),
        "source": source,
        "budgets": list(budgets),
        "seed": cfg.seed,
        "c_puct": cfg.c_puct,
    }
    if path.exists():
        result = json.loads(path.read_text(encoding="utf-8"))
        header = {key: result.get(key) for key in expected_header}
        if header != expected_header:
            raise RuntimeError(
                f"incompatible cached multi-candidate evaluation: {path}")
        if partial_path.exists():
            partial_path.unlink()
        print(f"reusing {group_name} {role} evaluation", flush=True)
        return result["levels"]

    initial_rows: list[dict[str, Any]] = []
    if partial_path.exists():
        partial = json.loads(partial_path.read_text(encoding="utf-8"))
        initial_rows = _validate_partial_rows(
            partial_path,
            partial,
            expected_header=expected_header,
            levels=levels,
            budgets=budgets,
        )
        print(
            f"resuming {group_name} {role} at "
            f"{len(initial_rows)}/{len(levels)} levels",
            flush=True,
        )
    else:
        print(f"evaluating {group_name} with {role}", flush=True)

    def checkpoint_rows(rows):
        atomic_write_json(partial_path, {**expected_header, "levels": rows})

    rows = _evaluate_checkpoint_group(
        checkpoint,
        levels,
        group_name=group_name,
        budgets=budgets,
        c_puct=cfg.c_puct,
        seed=cfg.seed,
        device=cfg.device,
        progress_interval=cfg.progress_interval,
        initial_rows=initial_rows,
        checkpoint_rows=checkpoint_rows,
    )
    atomic_write_json(path, {**expected_header, "levels": rows})
    if partial_path.exists():
        partial_path.unlink()
    return rows


def _write_csv(path: Path, candidates: list[dict[str, Any]]) -> None:
    handle = io.StringIO()
    fields = [
        "candidate", "group", "budget", "levels", "incumbent_solved",
        "candidate_solved", "candidate_only", "incumbent_only",
        "both_solved", "neither_solved", "solve_rate_delta",
        "mean_solution_length_delta_on_both_solved",
    ]
    writer = csv.DictWriter(handle, fieldnames=fields)
    writer.writeheader()
    for candidate in candidates:
        for summary in candidate["groups"]:
            for budget in summary["budgets"]:
                item = summary["per_budget"][str(budget)]
                writer.writerow({
                    "candidate": candidate["name"],
                    "group": summary["group"],
                    "budget": budget,
                    **{
                        key: item[key]
                        for key in fields
                        if key not in ("candidate", "group", "budget")
                    },
                })
    atomic_write_text(path, handle.getvalue())


def _validation_gate(
    summary: dict[str, Any],
    cfg: MultiTransferAuditConfig,
) -> dict[str, Any]:
    weight_total = sum(cfg.gate_validation_weights)
    incumbent_score = 0.0
    candidate_score = 0.0
    for budget, weight in zip(
            cfg.gate_validation_budgets,
            cfg.gate_validation_weights):
        item = summary["per_budget"][str(budget)]
        incumbent_score += weight * item["incumbent_solve_rate"]
        candidate_score += weight * item["candidate_solve_rate"]
    incumbent_score /= weight_total
    candidate_score /= weight_total
    required_score = incumbent_score + cfg.gate_margin
    return {
        "group": "promotion_validation",
        "budgets": list(cfg.gate_validation_budgets),
        "weights": list(cfg.gate_validation_weights),
        "margin": cfg.gate_margin,
        "incumbent_score": incumbent_score,
        "candidate_score": candidate_score,
        "required_score": required_score,
        "passed": candidate_score > required_score,
        "generated_holdout_status":
            "eligible" if candidate_score > required_score
            else "skipped_validation_regression",
    }


def run_multi_transfer_audit(
    cfg: MultiTransferAuditConfig,
) -> dict[str, Any]:
    cfg.validate()
    root = Path(cfg.output_dir)
    root.mkdir(parents=True, exist_ok=True)
    identity = _identity(cfg)
    identity_path = root / "experiment.json"
    if identity_path.exists():
        persisted = json.loads(identity_path.read_text(encoding="utf-8"))
        if persisted != identity:
            raise RuntimeError(
                "multi-candidate audit output belongs to different inputs or "
                "settings")
    else:
        atomic_write_json(identity_path, identity)

    generated_groups: list[
        tuple[str, list[Level], tuple[int, ...], dict[str, Any]]
    ] = []
    for group in cfg.generated_groups:
        generated_groups.append((
            group.name,
            _load_level_list(group.path),
            cfg.generated_budgets,
            {
                "kind": "generated_holdout",
                "path": group.path,
                "sha256": sha256_file(group.path),
            },
        ))
    validation_groups = []
    if not cfg.generated_only:
        assert cfg.eval_levels_dataset is not None
        assert cfg.eval_split_manifest is not None
        validation_groups.append((
            "promotion_validation",
            _load_validation_levels(
                cfg.eval_levels_dataset, cfg.eval_split_manifest),
            cfg.validation_budgets,
            {
                "kind": "promotion_validation",
                "dataset": cfg.eval_levels_dataset,
                "dataset_sha256": sha256_file(cfg.eval_levels_dataset),
                "split_manifest": cfg.eval_split_manifest,
                "split_manifest_sha256": sha256_file(cfg.eval_split_manifest),
            },
        ))
    groups = (
        [*validation_groups, *generated_groups]
        if cfg.validation_first
        else [*generated_groups, *validation_groups]
    )

    evaluation_dir = root / "evaluations"
    evaluation_dir.mkdir(parents=True, exist_ok=True)
    candidate_results = [
        {
            "name": candidate.name,
            "checkpoint": candidate.checkpoint,
            "checkpoint_sha256": sha256_file(candidate.checkpoint),
            "groups": [],
        }
        for candidate in cfg.candidates
    ]
    gated_candidate_names = {
        candidate.name for candidate in cfg.candidates
    }
    for group_name, levels, budgets, source in groups:
        selected = list(zip(cfg.candidates, candidate_results))
        if (cfg.gate_validation_budgets
                and group_name != "promotion_validation"):
            selected = [
                pair for pair in selected
                if pair[0].name in gated_candidate_names
            ]
            if not selected:
                print(
                    f"skipping {group_name}: no candidate passed validation",
                    flush=True,
                )
                continue
        incumbent_rows = _evaluate_or_resume(
            checkpoint=cfg.incumbent_checkpoint,
            role="incumbent",
            group_name=group_name,
            levels=levels,
            budgets=budgets,
            source=source,
            identity=identity,
            evaluation_dir=evaluation_dir,
            cfg=cfg,
        )
        for candidate, candidate_result in selected:
            candidate_rows = _evaluate_or_resume(
                checkpoint=candidate.checkpoint,
                role=candidate.name,
                group_name=group_name,
                levels=levels,
                budgets=budgets,
                source=source,
                identity=identity,
                evaluation_dir=evaluation_dir,
                cfg=cfg,
            )
            summary = _paired_group_summary(
                group_name, budgets, incumbent_rows, candidate_rows)
            candidate_result["groups"].append(summary)
            if (group_name == "promotion_validation"
                    and cfg.gate_validation_budgets):
                gate = _validation_gate(summary, cfg)
                candidate_result["validation_gate"] = gate
                if not gate["passed"]:
                    gated_candidate_names.discard(candidate.name)

    result = {
        "schema_version": SCHEMA_VERSION,
        "semantics": SEMANTICS,
        "experiment_fingerprint": identity["fingerprint"],
        "incumbent_checkpoint": cfg.incumbent_checkpoint,
        "incumbent_checkpoint_sha256":
            identity["inputs"]["incumbent_checkpoint_sha256"],
        "candidates": candidate_results,
        "final_test_status": "sealed_not_evaluated",
    }
    atomic_write_json(root / "summary.json", result)
    _write_csv(root / "summary.csv", candidate_results)
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Compare several candidates with one shared incumbent on frozen "
            "generated holdouts and/or promotion validation."))
    parser.add_argument("--incumbent-checkpoint", required=True)
    parser.add_argument(
        "--candidate", action="append", type=_parse_candidate,
        required=True, metavar="NAME=CHECKPOINT")
    parser.add_argument(
        "--generated-group", action="append", type=_parse_generated_group,
        default=None, metavar="NAME=PATH")
    parser.add_argument("--eval-levels-dataset", default=None)
    parser.add_argument("--eval-split-manifest", default=None)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--generated-budgets", type=int, nargs="+",
        default=[55, 90, 148, 244, 400])
    parser.add_argument(
        "--validation-budgets", type=int, nargs="+",
        default=[4, 8, 16, 32, 64])
    parser.add_argument("--c-puct", type=float, default=1.5)
    parser.add_argument("--seed", type=int, default=2045)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--progress-interval", type=int, default=5)
    parser.add_argument(
        "--validation-first", action="store_true",
        help="evaluate promotion validation before generated holdouts")
    parser.add_argument(
        "--generated-only", action="store_true",
        help="screen generated holdouts without reading promotion validation")
    parser.add_argument(
        "--gate-validation-budgets", type=int, nargs="+", default=[])
    parser.add_argument(
        "--gate-validation-weights", type=float, nargs="+", default=[])
    parser.add_argument("--gate-margin", type=float, default=0.0)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    cfg = MultiTransferAuditConfig(
        incumbent_checkpoint=args.incumbent_checkpoint,
        candidates=tuple(args.candidate),
        generated_groups=tuple(args.generated_group or ()),
        eval_levels_dataset=args.eval_levels_dataset,
        eval_split_manifest=args.eval_split_manifest,
        output_dir=args.output_dir,
        generated_budgets=tuple(args.generated_budgets),
        validation_budgets=tuple(args.validation_budgets),
        c_puct=args.c_puct,
        seed=args.seed,
        device=args.device,
        progress_interval=args.progress_interval,
        validation_first=args.validation_first,
        gate_validation_budgets=tuple(args.gate_validation_budgets),
        gate_validation_weights=tuple(args.gate_validation_weights),
        gate_margin=args.gate_margin,
        generated_only=args.generated_only,
    )
    result = run_multi_transfer_audit(cfg)
    for candidate in result["candidates"]:
        print(f"=== {candidate['name']} ===")
        gate = candidate.get("validation_gate")
        if gate is not None:
            print(
                "validation gate: "
                f"{'passed' if gate['passed'] else 'failed'} "
                f"(candidate={gate['candidate_score']:.3f}, "
                f"required={gate['required_score']:.3f})")
        for group in candidate["groups"]:
            print(f"-- {group['group']} --")
            for budget in group["budgets"]:
                item = group["per_budget"][str(budget)]
                print(
                    f"budget {budget}: "
                    f"incumbent={item['incumbent_solved']}/{item['levels']}, "
                    f"candidate={item['candidate_solved']}/{item['levels']}, "
                    f"delta={item['solve_rate_delta']:+.3f}")
    print("final test: sealed and not evaluated")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
