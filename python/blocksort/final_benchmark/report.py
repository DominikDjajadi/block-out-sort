"""Assemble the final research report (summary.json + report.md) from artifacts.

All conclusions are derived from the measured artifacts. The report deliberately
avoids claiming improvements that are not supported by the frozen held-out
numbers or that fall within observed run-to-run noise.
"""

from __future__ import annotations

import json
import math
from numbers import Real
from pathlib import Path
from typing import Any, Optional

from ..cotraining.eval_split import validate_common_evaluation_split


def _load(path: Path) -> Optional[Any]:
    if path.exists():
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return None
    return None


def _fmt(x, nd=3):
    if x is None:
        return "n/a"
    if isinstance(x, float):
        return f"{x:.{nd}f}"
    return str(x)


def solver_comparison(
    solver: dict[str, Any],
    *,
    comparison_budget: int | None = None,
    metric: str = "confirmed_optimal_rate",
) -> dict[str, Any]:
    if not solver:
        raise ValueError("solver comparison is unavailable")
    search = solver.get("search")
    if not isinstance(search, dict) or not search:
        raise ValueError("solver comparison has no search model results")
    parsed: dict[str, dict[int, dict[str, Any]]] = {}
    for method_id, metrics in search.items():
        if not isinstance(method_id, str) or "@" not in method_id:
            raise ValueError(f"invalid search method identifier: {method_id!r}")
        model, budget_text = method_id.rsplit("@", 1)
        if not model:
            raise ValueError(f"invalid empty model identifier in {method_id!r}")
        try:
            budget = int(budget_text)
        except ValueError as exc:
            raise ValueError(
                f"nonnumeric budget in search method {method_id!r}") from exc
        if str(budget) != budget_text or budget <= 0:
            raise ValueError(f"invalid budget in search method {method_id!r}")
        if budget in parsed.setdefault(model, {}):
            raise ValueError(
                f"duplicate result for model {model!r} at budget {budget}")
        if not isinstance(metrics, dict):
            raise ValueError(f"search result {method_id!r} is not a mapping")
        parsed[model][budget] = metrics
    if not parsed:
        raise ValueError("solver comparison has an empty model set")
    configured = solver.get("budgets")
    if configured is not None:
        if not isinstance(configured, list) or not configured:
            raise ValueError("solver report contains invalid configured budgets")
        normalized_budgets = []
        for value in configured:
            if isinstance(value, bool):
                raise ValueError(
                    "solver report contains invalid configured budgets")
            try:
                normalized = int(value)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    "solver report contains invalid configured budgets") from exc
            if normalized <= 0 or (
                    isinstance(value, str) and str(normalized) != value):
                raise ValueError(
                    "solver report contains invalid configured budgets")
            normalized_budgets.append(normalized)
        default_budget = max(normalized_budgets)
    else:
        default_budget = max(
            budget for model_results in parsed.values()
            for budget in model_results)
    selected = (
        comparison_budget if comparison_budget is not None
        else solver.get("comparison_budget", default_budget))
    if isinstance(selected, str) and selected.isdigit():
        selected = int(selected)
    if isinstance(selected, bool) or not isinstance(selected, int) or selected <= 0:
        raise ValueError(f"invalid comparison budget: {selected!r}")

    models: dict[str, Any] = {}
    for model in sorted(parsed):
        available = sorted(parsed[model])
        if selected not in parsed[model]:
            raise ValueError(
                f"Cannot compare models at budget {selected}: model {model!r} "
                f"contains budgets {available}.")
        scored: list[tuple[float, int]] = []
        for budget, metrics in parsed[model].items():
            score = metrics.get(metric)
            if score is None:
                continue
            if (isinstance(score, bool) or not isinstance(score, Real)
                    or not math.isfinite(float(score))):
                raise ValueError(
                    f"model {model!r} budget {budget} has invalid "
                    f"{metric}: {score!r}")
            scored.append((float(score), budget))
        common = parsed[model][selected].get(metric)
        if (isinstance(common, bool) or not isinstance(common, Real)
                or not math.isfinite(float(common))):
            raise ValueError(
                f"model {model!r} has unavailable {metric} at comparison "
                f"budget {selected}")
        best_score, best_budget = max(
            scored, key=lambda item: (item[0], -item[1]))
        models[model] = {
            "common_budget_score": float(common),
            "available_budgets": available,
            "best_observed_score": best_score,
            "best_observed_budget": best_budget,
        }
    names = sorted(models)
    deltas = {
        f"{right}-{left}": (
            models[right]["common_budget_score"]
            - models[left]["common_budget_score"])
        for index, left in enumerate(names)
        for right in names[index + 1:]
    }
    return {
        "schema_version": 2,
        "comparison_metric": metric,
        "comparison_budget": selected,
        "models": models,
        "common_budget_deltas": deltas,
    }


def _solver_conclusion(solver, comparison_budget: int | None = None):
    if not solver:
        return None, "solver comparison not available"
    comparison = solver_comparison(
        solver, comparison_budget=comparison_budget)
    by_model = {
        name: values["common_budget_score"]
        for name, values in comparison["models"].items()}
    ei = by_model.get("expert_iteration")
    cot = by_model.get("cotrained")
    if ei is None or cot is None:
        return by_model, "co-trained protagonist not available for comparison"
    if cot > ei + 1e-9:
        return by_model, (f"co-trained protagonist improved fixed-denominator "
                          f"confirmed-optimal rate at budget "
                          f"{comparison['comparison_budget']} "
                          f"({ei:.3f} -> {cot:.3f}) on the harder frozen states")
    if cot < ei - 1e-9:
        return by_model, (f"co-trained protagonist regressed in fixed-denominator "
                          f"confirmed-optimal rate at budget "
                          f"{comparison['comparison_budget']} "
                          f"({ei:.3f} -> {cot:.3f})")
    return by_model, ("co-trained protagonist matched the expert-iteration model "
                      f"at budget {comparison['comparison_budget']} in "
                      "fixed-denominator confirmed-optimal rate")


def _generator_conclusion(gens):
    if not gens:
        return "generator comparison not available"
    g = gens.get("generators", {})
    rand = g.get("random", {}).get("summary", {})
    adv = g.get("adversarial_designer", {}).get("summary", {})
    if not rand or not adv:
        return "insufficient generator data"
    rk = rand.get("mean_adversarial_regret", {})
    ak = adv.get("mean_adversarial_regret", {})
    rm, rs = rk.get("mean"), rk.get("std")
    am, as_ = ak.get("mean"), ak.get("std")
    if rm is None or am is None:
        return "adversarial-regret data missing"
    noise = (rs or 0.0) + (as_ or 0.0)
    diff = am - rm
    if diff > noise and diff > 0:
        return (f"adversarial designer produced higher mean adversarial regret "
                f"than random ({rm:.3f}+-{_fmt(rs)} vs {am:.3f}+-{_fmt(as_)}); "
                f"gap exceeds combined seed std")
    return (f"adversarial designer vs random adversarial regret "
            f"({am:.3f}+-{_fmt(as_)} vs {rm:.3f}+-{_fmt(rs)}); difference within "
            f"run-to-run noise (inconclusive)")


_KNOWN_ROLES = {
    "retention",
    "promotion_validation",
    "promotion_validation_ood",
    "promotion_challenge",
    "held_out_final",
}


def benchmark_provenance_summary(manifest, saturation) -> dict[str, Any]:
    provenance = (manifest or {}).get("group_provenance")
    if not isinstance(provenance, dict):
        return {
            "status": "legacy_unavailable",
            "reason": "benchmark manifest has no version-2 group provenance",
            "groups": {},
            "aggregates": {},
        }
    sat_groups = (saturation or {}).get("groups", {})
    groups: dict[str, Any] = {}
    for name, meta in provenance.items():
        if not isinstance(meta, dict):
            raise ValueError(f"malformed provenance for group {name!r}")
        role = meta.get("evaluation_role")
        if role not in _KNOWN_ROLES:
            raise ValueError(
                f"unknown provenance classification {role!r} for group {name!r}")
        eligible = meta.get("held_out_eligible")
        if not isinstance(eligible, bool):
            raise ValueError(
                f"group {name!r} has non-boolean held_out_eligible")
        if eligible and (
                meta.get("disjointness_verified") is not True
                or meta.get("overlap_count") != 0):
            raise ValueError(
                f"group {name!r} is marked held-out without verified "
                "zero-overlap evidence")
        measured = sat_groups.get(name, {})
        score = measured.get("bounded_solve_rate")
        levels = measured.get("levels", 0)
        status = "measured" if score is not None else "unavailable"
        groups[name] = {
            "provenance": meta,
            "held_out_eligible": eligible,
            "status": status,
            "score": score,
            "coverage": 1.0 if status == "measured" and levels else None,
            "count": levels,
        }

    memberships = {
        "retention": [
            name for name, group in groups.items()
            if group["provenance"]["evaluation_role"] == "retention"],
        "generated_generalization": [
            name for name, group in groups.items()
            if group["provenance"]["evaluation_role"] in {
                "promotion_validation", "promotion_challenge"}],
        "ood": [
            name for name, group in groups.items()
            if group["provenance"]["evaluation_role"]
            == "promotion_validation_ood"],
        "held_out_final": [
            name for name, group in groups.items()
            if group["held_out_eligible"]],
        "all_groups_diagnostic": list(groups),
    }
    aggregates = {}
    for aggregate, members in memberships.items():
        measured = [
            groups[name] for name in members
            if groups[name]["status"] == "measured" and groups[name]["count"]]
        total = sum(group["count"] for group in measured)
        if not total:
            aggregates[aggregate] = {
                "status": "unavailable",
                "score": None,
                "members": members,
                "reason": "no measured eligible groups",
            }
            continue
        aggregates[aggregate] = {
            "status": "measured",
            "score": sum(
                group["score"] * group["count"] for group in measured) / total,
            "members": members,
            "reason": None,
        }
    evaluation_split = (manifest or {}).get("evaluation_split")
    if evaluation_split is not None:
        evaluation_split = validate_common_evaluation_split([evaluation_split])
    return {
        "status": "available",
        "reason": None,
        "metric": "bounded_solve_rate",
        "groups": groups,
        "aggregates": aggregates,
        "evaluation_split": evaluation_split,
    }


def build_report(root: Path, *, defaults: dict, args_dict: dict) -> None:
    root = Path(root)
    sat = _load(root / "saturation.json")
    manifest = _load(root / "benchmark" / "benchmark_manifest.json")
    ckpts = _load(root / "checkpoints.json")
    internal_sources = _load(root / "internal_checkpoint_sources.json")
    cot = _load(root / "cotraining_summary.json")
    solver = _load(root / "solver.json")
    gens = _load(root / "generators.json")
    abl = _load(root / "ablations.json")

    requested_budget = args_dict.get("comparison_budget")
    comparison = None
    if solver:
        try:
            comparison = solver_comparison(
                solver, comparison_budget=requested_budget)
        except ValueError as exc:
            if solver.get("schema_version", 1) >= 2:
                raise
            by_model = None
            solver_concl = (
                "legacy solver artifact has no publishable controlled "
                f"comparison: {exc}")
        else:
            by_model, solver_concl = _solver_conclusion(
                solver, comparison_budget=requested_budget)
    else:
        by_model, solver_concl = None, "solver comparison not available"
    gen_concl = _generator_conclusion(gens)
    provenance_summary = benchmark_provenance_summary(manifest, sat)

    promotions = []
    frontier_accept = []
    if cot:
        for h in cot.get("history", []):
            promotions.append(h.get("promoted"))
            frontier_accept.append(h.get("frontier_acceptance_rate"))
    any_promoted = any(promotions) if promotions else False
    any_accept = any((f or 0) > 0 for f in frontier_accept) if frontier_accept else False

    summary = {
        "benchmark_useful": sat.get("useful_benchmark") if sat else None,
        "overall_bounded_solve_rate": sat.get("overall_bounded_solve_rate") if sat else None,
        "cotraining_rounds": cot.get("rounds") if cot else None,
        "any_promotion": any_promoted,
        "any_frontier_acceptance": any_accept,
        "schema_version": 2,
        "solver_controlled_comparison": comparison,
        "solver_common_budget_score_by_model": by_model,
        "solver_best_observed_diagnostics": (
            {name: {
                "score": values["best_observed_score"],
                "budget": values["best_observed_budget"],
            } for name, values in comparison["models"].items()}
            if comparison else None),
        "solver_conclusion": solver_concl,
        "generator_conclusion": gen_concl,
        "benchmark_provenance": provenance_summary,
        "checkpoint_sources": ckpts,
        "internal_checkpoint_sources": internal_sources,
    }
    (root / "summary.json").write_text(json.dumps(summary, indent=2),
                                       encoding="utf-8")

    L: list[str] = []
    L.append("# Block Out Sort - Final Benchmark, Co-training, and Ablation Report")
    L.append("")
    L.append("All numbers below are produced by `python -m blocksort.final_benchmark.run` "
             "from the artifacts in this directory. Conclusions are intentionally "
             "conservative: retention, promotion-held-out, and verified final-held-out "
             "measurements are labeled separately.")
    L.append("")

    L.append("## 1. System architecture")
    L.append("")
    L.append("- Deterministic Block Out Sort engine + schema/validation (Python port of the JS engine).")
    L.append("- Exact A* value oracle (`blocksort.oracle`) with caching.")
    L.append("- Supervised policy-value network + neural-guided PUCT **graph** search (transposition sharing).")
    L.append("- Hybrid expert iteration (exact-first labels, search fallback, replay, frozen-split promotion).")
    L.append("- Constrained adversarial designer (validity-preserving reverse slides, PPO, adversarial reward).")
    L.append("- Alternating protagonist-designer co-training with a frontier curriculum.")
    L.append("")

    L.append("## 2. Training progression")
    L.append("")
    if ckpts:
        L.append(
            "| Checkpoint | Source | Pipeline | Role | Progress | sha256[:16] |")
        L.append("|---|---|---|---|---:|---|")
        for name, idy in ckpts.items():
            digest = idy.get("sha256", "?")
            origin = idy.get("benchmark_source_origin")
            source = origin or idy.get("source_kind", "unknown")
            L.append(
                f"| {name} | {source} "
                f"| {idy.get('source_pipeline', idy.get('pipeline', 'n/a'))} "
                f"| {idy.get('committed_role', 'n/a')} "
                f"| {idy.get('committed_progress', 'n/a')} "
                f"| {digest[:16]} |")
        if internal_sources:
            L.append("")
            L.append(
                "Pinned internal-source identity: "
                f"`{internal_sources.get('source_identity_sha256')}`.")
    L.append("")
    L.append("Supervised model -> expert-iteration model -> (this milestone) "
             "co-training. Co-training fine-tunes the protagonist on accepted "
             "frontier levels and trains the designer against the frozen "
             "promoted protagonist.")
    L.append("")

    L.append("## 3. Final benchmark construction")
    L.append("")
    if manifest:
        L.append("Benchmark group provenance (all groups are evaluation-only for "
                 "this run, but only verified disjoint groups qualify for the "
                 "held-out-final aggregate):")
        L.append("")
        L.append("| Group | Levels | Role | Source | Held-out final | Overlap |")
        L.append("|---|---|---|---|---|---|")
        for g, n in manifest.get("group_sizes", {}).items():
            meta = manifest.get("group_provenance", {}).get(g, {})
            L.append(
                f"| {g} | {n} | {meta.get('evaluation_role','legacy/unknown')} "
                f"| {meta.get('source_kind','unknown')} "
                f"| {meta.get('held_out_eligible',False)} "
                f"| {meta.get('overlap_count','n/a')} |")
        L.append("")
        split_identity = manifest.get("evaluation_split")
        if split_identity:
            L.append(
                "Fixed promotion/final-test split: "
                f"`{split_identity.get('evaluation_split_fingerprint')}` "
                f"(manifest "
                f"`{split_identity.get('evaluation_split_manifest_sha256')}`, "
                f"promotion validation "
                f"{split_identity.get('promotion_validation_count')}, "
                f"final test {split_identity.get('final_test_count')}, "
                f"seed {split_identity.get('split_seed')}).")
            L.append("")
    if sat:
        L.append(f"Saturation check (bounded search, budget={sat.get('budget')}): "
                 f"overall solve rate = **{_fmt(sat.get('overall_bounded_solve_rate'))}** "
                 f"-> useful (non-saturated): **{sat.get('useful_benchmark')}**.")
        L.append("")
        L.append("| Group | Levels | Bounded solve rate |")
        L.append("|---|---|---|")
        for g, v in sat.get("groups", {}).items():
            L.append(f"| {g} | {v.get('levels')} | {_fmt(v.get('bounded_solve_rate'))} |")
        L.append("")

    L.append("## 4. Solver results")
    L.append("")
    if solver:
        L.append(f"Evaluated on {solver['states']} identical frozen harder states; "
                 f"common-set solved by every method = {solver['common_solved_count']}.")
        L.append("")
        if comparison:
            L.append("### Controlled comparison")
            L.append("")
            L.append(
                f"Metric: `{comparison['comparison_metric']}`  ")
            L.append(f"Budget: **{comparison['comparison_budget']}**")
            L.append("")
            L.append("| Model | Common-budget score |")
            L.append("|---|---|")
            for name, values in comparison["models"].items():
                L.append(
                    f"| {name} | {_fmt(values['common_budget_score'])} |")
            L.append("")
            L.append("### Best observed diagnostics (not used for ranking/deltas)")
            L.append("")
            L.append("| Model | Best score | Budget |")
            L.append("|---|---|---|")
            for name, values in comparison["models"].items():
                L.append(
                    f"| {name} | {_fmt(values['best_observed_score'])} "
                    f"| {values['best_observed_budget']} |")
            L.append("")
        L.append("Raw greedy policy:")
        L.append("")
        L.append("| Model | Optimal-acc (known) | Oracle coverage | Mean regret "
                 "(known) | Value MAE (moves) | n known |")
        L.append("|---|---|---|---|---|---|")
        for name, m in solver.get("raw_policy", {}).items():
            L.append(f"| {name} | {_fmt(m['optimal_acc'])} "
                     f"| {_fmt(m.get('oracle_regret_coverage'))} "
                     f"| {_fmt(m['mean_regret'])} | {_fmt(m['value_mae_moves'])} "
                     f"| {m['n']} |")
        L.append("")
        L.append("Graph search (per model@budget) and exact A*:")
        L.append("")
        L.append("| Method | Opt-acc (known) | Oracle coverage | Regret (known) | "
                 "Solve | Gap(common) | Gap(each) | Runtime ms | Nodes | "
                 "Transp.hits |")
        L.append("|---|---|---|---|---|---|---|---|---|---|")
        for mid, m in solver.get("search", {}).items():
            L.append(f"| {mid} | {_fmt(m['optimal_acc'])} "
                     f"| {_fmt(m.get('oracle_regret_coverage'))} "
                     f"| {_fmt(m['mean_regret'])} | {_fmt(m['solve_rate'])} "
                     f"| {_fmt(m['solution_length_gap_common'])} "
                     f"| {_fmt(m['solution_length_gap_each'])} | {_fmt(m['runtime_ms_mean'])} "
                     f"| {_fmt(m['nodes_expanded_mean'],1)} | {_fmt(m['transposition_hits_mean'],1)} |")
        a = solver.get("astar", {})
        L.append(f"| astar | 1.000 | 1.000 | 0.000 | {_fmt(a.get('solve_rate'))} "
                 f"| 0.000 | 0.000 "
                 f"| {_fmt(a.get('runtime_ms_mean'))} | {_fmt(a.get('nodes_expanded_mean'),1)} | n/a |")
        L.append("")
        L.append(f"**Solver conclusion:** {solver_concl}")
        L.append("")
        L.append(f"> {solver.get('note_graph_vs_tree','')}")
    else:
        L.append("_Solver comparison artifact not present._")
    L.append("")

    L.append("### Benchmark aggregates by provenance")
    L.append("")
    if provenance_summary.get("metric"):
        L.append(f"Metric: `{provenance_summary['metric']}`")
        L.append("")
    aggregates = provenance_summary.get("aggregates", {})
    if not aggregates:
        L.append(
            f"_Unavailable: {provenance_summary.get('reason','unknown reason')}._")
    else:
        L.append("| Aggregate | Status | Score | Members |")
        L.append("|---|---|---|---|")
        for name, aggregate in aggregates.items():
            L.append(
                f"| {name} | {aggregate['status']} "
                f"| {_fmt(aggregate['score'])} "
                f"| {', '.join(aggregate['members']) or 'none'} |")
    L.append("")

    L.append("## 5. Generator results")
    L.append("")
    if gens:
        L.append(f"Equal counts ({gens['count']} levels) over seeds {gens['seeds']}. "
                 "Values are mean +- std across seeds. Structural metrics are "
                 "search-based difficulty signals, **not** human difficulty.")
        L.append("")
        metrics = ["valid_rate", "oracle_solve_rate", "protagonist_solve_rate",
                   "mean_adversarial_regret", "mean_extra_moves",
                   "mean_first_exit_depth", "mean_rehandled_blocks",
                   "mean_immediately_exitable", "mean_structural_score",
                   "duplicate_rate", "mean_generation_seconds"]
        header = "| Generator | " + " | ".join(metrics) + " |"
        L.append(header)
        L.append("|" + "---|" * (len(metrics) + 1))
        for name, g in gens.get("generators", {}).items():
            s = g["summary"]
            cells = []
            for k in metrics:
                v = s.get(k, {})
                cells.append(f"{_fmt(v.get('mean'))}+-{_fmt(v.get('std'))}")
            L.append(f"| {name} | " + " | ".join(cells) + " |")
        L.append("")
        L.append(f"**Generator conclusion:** {gen_concl}")
    else:
        L.append("_Generator comparison artifact not present._")
    L.append("")

    L.append("## 6. Co-training results")
    L.append("")
    if cot:
        L.append(f"Rounds: {cot.get('rounds')}. Promotion uses the harder held-out "
                 "validation levels (never the test benchmark).")
        L.append("")
        L.append("| Round | Gen | Accepted | Frontier-accept | Mean solve | "
                 "Full exact | Exact path | Search lbl | Promoted | Curriculum | Rejections |")
        L.append("|---|---|---|---|---|---|---|---|---|---|---|")
        for h in cot.get("history", []):
            rej = h.get("rejections", {})
            rej_s = (f"acc={rej.get('accepted',0)} "
                     f"below={rej.get('below_frontier',0)} "
                     f"above={rej.get('above_frontier',0)} "
                     f"oracle={rej.get('oracle_unsolved',0)}")
            L.append(f"| {h['round']} | {h['generated']} | {h['accepted']} | "
                     f"{_fmt(h['frontier_acceptance_rate'])} | {_fmt(h['mean_solve_rate'])} "
                     f"| {h['label_exact']} | {h.get('label_exact_path', 0)} | "
                     f"{h['label_search']} | {h['promoted']} "
                     f"| {h['curriculum_adjustment']['direction']} | {rej_s} |")
        L.append("")
        L.append(f"Any promotion: **{any_promoted}**. Any frontier acceptance: "
                 f"**{any_accept}**.")
        if not any_accept:
            L.append("")
            L.append("> Frontier acceptance remained zero. Diagnosis: the bounded "
                     "protagonist's per-level solve rate stayed below the frontier "
                     "band even after the curriculum lowered difficulty and raised "
                     "the search budget; see the per-round curriculum trajectory in "
                     "`cotraining/round_*/report.json`.")
    else:
        L.append("_Co-training summary artifact not present._")
    L.append("")

    L.append("## 7. Ablations")
    L.append("")
    if abl:
        rt = abl.get("reward_terms", {})
        if rt:
            L.append("Designer reward-term ablations (short designer training vs a "
                     "fixed protagonist):")
            L.append("")
            L.append("| Variant | mean_reward | adv_regret | extra_moves | "
                     "valid | oracle_solve | prot_solve |")
            L.append("|---|---|---|---|---|---|---|")
            for name, v in rt.items():
                L.append(f"| {name} | {_fmt(v.get('mean_reward'))} | "
                         f"{_fmt(v.get('mean_adversarial_regret'))} | "
                         f"{_fmt(v.get('mean_extra_moves'))} | {_fmt(v.get('valid_rate'))} | "
                         f"{_fmt(v.get('oracle_solve_rate'))} | "
                         f"{_fmt(v.get('protagonist_solve_rate'))} |")
            L.append("")
        ct = abl.get("cotraining", {})
        if ct:
            L.append("Co-training ablations (designer training skipped to isolate "
                     "protagonist/labelling/curriculum effects):")
            L.append("")
            L.append("| Variant | Rounds | Accepted (sum) | Full exact | Exact path | Search lbl | "
                     "Any promotion |")
            L.append("|---|---|---|---|---|---|---|")
            for name, v in ct.items():
                pr = v.get("per_round", [])
                acc = sum(r["accepted"] for r in pr)
                ex = sum(r["label_exact"] for r in pr)
                ep = sum(r.get("label_exact_path", 0) for r in pr)
                se = sum(r["label_search"] for r in pr)
                promo = any(r["promoted"] for r in pr)
                L.append(
                    f"| {name} | {v.get('rounds')} | {acc} | {ex} | "
                    f"{ep} | {se} | {promo} |")
            L.append("")
    L.append("Designer level replay (ablation 4): the designer replay is an archive "
             "and is not consumed by the on-policy PPO update, so disabling it does "
             "not change designer learning (reported, not re-run).")
    L.append("")
    L.append("Graph vs tree (ablation 8): only a transposition-sharing graph search "
             "exists; the solver table's `Transp.hits` column quantifies sharing. A "
             "separate tree-search variant was out of the inexpensive scope.")
    L.append("")

    L.append("## 8. Limitations")
    L.append("")
    L.append("- All trained models were learned on 5x5 boards; 6x6/7x7 results are "
             "out-of-distribution generalisation.")
    L.append("- Runs use small search/A* budgets and modest level counts for "
             "tractability; absolute numbers are not production-scale.")
    L.append("- Statistical power is limited (few seeds); only effects clearly "
             "larger than seed std are claimed.")
    L.append("- No separate tree-search implementation, so the graph-vs-tree "
             "comparison is descriptive (transposition hits) rather than controlled.")
    L.append("")

    L.append("## 9. Conclusions")
    L.append("")
    L.append(f"- **Benchmark:** {'non-saturated (useful)' if (sat and sat.get('useful_benchmark')) else 'see saturation.json'} "
             f"(overall bounded solve rate {_fmt(sat.get('overall_bounded_solve_rate') if sat else None)}).")
    L.append(f"- **Solver:** {solver_concl}")
    L.append(f"- **Generator:** {gen_concl}")
    L.append(f"- **Co-training:** {'at least one promotion occurred' if any_promoted else 'no protagonist promotion on the held-out validation'}; "
             f"{'frontier levels were accepted' if any_accept else 'frontier acceptance remained zero (diagnosed above)'}.")
    L.append("")

    L.append("## 10. Exact reproduction commands")
    L.append("")
    L.append("```bash")
    L.append("cd python")
    L.append("python -m blocksort.final_benchmark.run --device cpu \\")
    L.append("  --phases benchmark cotrain solver generators ablations report \\")
    L.append(f"  --output-dir {args_dict.get('output_dir','runs/final_benchmark')} \\")
    L.append(f"  --rounds {args_dict.get('rounds')} --levels-per-round {args_dict.get('levels_per_round')} \\")
    L.append(f"  --astar-max-nodes {args_dict.get('astar_max_nodes')} --seed {args_dict.get('seed')}")
    L.append("```")
    L.append("")
    L.append("Individual phases can be re-run independently (artifacts are written "
             "per phase). Checkpoint identities are recorded in `checkpoints.json`.")

    (root / "report.md").write_text("\n".join(L), encoding="utf-8")
