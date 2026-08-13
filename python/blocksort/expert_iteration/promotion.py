"""Promotion configuration and score extraction contracts."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
import math
from numbers import Real
from typing import Any


_BUDGETED_METRIC_FIELDS = {
    "search_optimal_acc": "search_optimal_acc",
    "search_confirmed_optimal_rate": "search_confirmed_optimal_rate",
    "neg_search_regret": "search_mean_regret",
}
_NON_BUDGETED_METRIC_FIELDS = {
    "raw_policy_optimal_acc": "raw_policy_optimal_acc",
    "raw_policy_confirmed_optimal_rate": "raw_policy_confirmed_optimal_rate",
}
_BUDGET_SWEEP_METRICS = {
    "budget_sweep_confirmed_optimal_rate",
    "weighted_budget_sweep_confirmed_optimal_rate",
    "weighted_budget_sweep_solve_rate",
}
_CONDITIONAL_METRICS = {
    "search_optimal_acc",
    "raw_policy_optimal_acc",
    "neg_search_regret",
}
DEFAULT_BUDGET_SWEEP_BUDGETS = (1, 2, 4, 8)
DEFAULT_BUDGET_SWEEP_WEIGHTS = (0.4, 0.3, 0.2, 0.1)


@dataclass(frozen=True)
class PromotionEvidence:
    incumbent_score: float
    candidate_score: float
    total_count: int
    incumbent_known_count: int
    candidate_known_count: int
    incumbent_coverage: float
    candidate_coverage: float
    incumbent_confirmed_count: int | None = None
    candidate_confirmed_count: int | None = None
    budget_list: tuple[int, ...] | None = None
    budget_weights: tuple[float, ...] | None = None
    per_budget: dict[str, Any] | None = None
    evidence_kind: str = "confirmed_optimal"

    @property
    def comparison_count(self) -> int:
        """Number of state-budget comparisons represented by aggregate counts."""
        multiplier = len(self.budget_list) if self.budget_list is not None else 1
        return self.total_count * multiplier

    def report_fields(self) -> dict[str, Any]:
        fields = {
            "promotion_evidence_kind": self.evidence_kind,
            "promotion_score_prev": self.incumbent_score,
            "promotion_score_candidate": self.candidate_score,
            "promotion_total_count": self.total_count,
            "promotion_comparison_count": self.comparison_count,
        }
        if self.evidence_kind == "solved":
            fields.update({
                "promotion_prev_evaluated_count":
                    self.incumbent_known_count,
                "promotion_candidate_evaluated_count":
                    self.candidate_known_count,
                "promotion_prev_solved_count":
                    self.incumbent_confirmed_count,
                "promotion_candidate_solved_count":
                    self.candidate_confirmed_count,
            })
        else:
            fields.update({
                "promotion_prev_classification_known_count":
                    self.incumbent_known_count,
                "promotion_candidate_classification_known_count":
                    self.candidate_known_count,
                "promotion_prev_classification_coverage":
                    self.incumbent_coverage,
                "promotion_candidate_classification_coverage":
                    self.candidate_coverage,
                "promotion_prev_confirmed_optimal_count":
                    self.incumbent_confirmed_count,
                "promotion_candidate_confirmed_optimal_count":
                    self.candidate_confirmed_count,
            })
        if self.budget_list is not None:
            fields.update({
                "promotion_budget_list": list(self.budget_list),
                "promotion_budget_weights": list(self.budget_weights or ()),
                "promotion_per_budget": self.per_budget or {},
            })
        return fields


def promotion_metric_requires_budget(metric: str) -> bool:
    """Return whether a supported promotion metric is search-budget specific."""
    if metric in _BUDGET_SWEEP_METRICS:
        return False
    if metric in _BUDGETED_METRIC_FIELDS:
        return True
    if metric in _NON_BUDGETED_METRIC_FIELDS:
        return False
    raise ValueError(f"unknown promotion metric: {metric}")


def promotion_metric_requires_budget_sweep(metric: str) -> bool:
    """Return whether a supported promotion metric requires many budgets."""
    if metric in _BUDGET_SWEEP_METRICS:
        return True
    if metric in _BUDGETED_METRIC_FIELDS or metric in _NON_BUDGETED_METRIC_FIELDS:
        return False
    raise ValueError(f"unknown promotion metric: {metric}")


def normalize_budget_sweep_config(
    budgets: Iterable[int] | None,
    weights: Iterable[float] | None,
) -> tuple[tuple[int, ...], tuple[float, ...]]:
    """Validate budget-sweep promotion knobs and return canonical tuples."""
    raw_budgets = tuple(DEFAULT_BUDGET_SWEEP_BUDGETS if budgets is None
                        else budgets)
    normalized_budgets: list[int] = []
    for value in raw_budgets:
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError("promotion_budgets must contain only integers")
        if value <= 0:
            raise ValueError("promotion_budgets must contain only positive budgets")
        if value in normalized_budgets:
            raise ValueError("promotion_budgets must not contain duplicates")
        normalized_budgets.append(value)
    if len(normalized_budgets) < 2:
        raise ValueError(
            "budget-sweep promotion requires at least two explicit budgets")

    if weights is None:
        if tuple(normalized_budgets) == DEFAULT_BUDGET_SWEEP_BUDGETS:
            raw_weights = DEFAULT_BUDGET_SWEEP_WEIGHTS
        else:
            raw_weights = tuple(1.0 for _ in normalized_budgets)
    else:
        raw_weights = tuple(weights)
    if len(raw_weights) != len(normalized_budgets):
        raise ValueError(
            "promotion_budget_weights must have exactly one value per "
            "promotion_budgets entry")
    normalized_weights = []
    for value in raw_weights:
        if isinstance(value, bool) or not isinstance(value, Real):
            raise ValueError("promotion_budget_weights must be numeric")
        weight = float(value)
        if not math.isfinite(weight) or weight < 0.0:
            raise ValueError(
                "promotion_budget_weights must be finite non-negative numbers")
        normalized_weights.append(weight)
    total = sum(normalized_weights)
    if total <= 0.0:
        raise ValueError("promotion_budget_weights must sum to a positive value")
    return tuple(normalized_budgets), tuple(weight / total
                                            for weight in normalized_weights)


def effective_eval_budgets(
    budgets: Iterable[int],
    *,
    promotion_metric: str,
    promotion_budget: int,
    promotion_budgets: Iterable[int] = (),
) -> tuple[int, ...]:
    """Return deterministic, unique budgets sufficient for promotion."""
    raw_budgets = tuple(budgets)
    if not raw_budgets:
        raise ValueError("eval_budgets must contain at least one budget")
    normalized: set[int] = set()
    for budget in raw_budgets:
        if isinstance(budget, bool) or not isinstance(budget, int):
            raise ValueError("eval_budgets must contain only integers")
        if budget <= 0:
            raise ValueError("eval_budgets must contain only positive budgets")
        normalized.add(budget)
    if promotion_metric_requires_budget_sweep(promotion_metric):
        sweep_budgets, _weights = normalize_budget_sweep_config(
            tuple(promotion_budgets) or None, None)
        normalized.update(sweep_budgets)
    elif promotion_metric_requires_budget(promotion_metric):
        if (isinstance(promotion_budget, bool)
                or not isinstance(promotion_budget, int)
                or promotion_budget <= 0):
            raise ValueError("promotion_budget must be a positive integer")
        normalized.add(promotion_budget)
    return tuple(sorted(normalized))


def _available_budget_keys(budgets: Mapping[Any, Any]) -> list[int]:
    available = []
    for key in budgets:
        try:
            available.append(int(key))
        except (TypeError, ValueError):
            continue
    return sorted(available)


def _required_value(container: Mapping[str, Any], field: str, context: str) -> float:
    if field not in container:
        raise ValueError(f"{context} is missing metric field {field!r}.")
    value = container[field]
    if value is None:
        raise ValueError(f"{context} has unavailable metric {field!r} (None).")
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(
            f"{context} has non-numeric metric {field!r}: {value!r}.")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(
            f"{context} has non-finite metric {field!r}: {value!r}.")
    return result


def _metric_container(
    report: Mapping[str, Any], *, metric: str, budget: int
) -> Mapping[str, Any]:
    if not promotion_metric_requires_budget(metric):
        return report
    budgets = report.get("budgets")
    if not isinstance(budgets, Mapping):
        raise ValueError(
            f"Promotion metric {metric!r} requires budget {budget}, "
            "but the evaluation report has no budgets mapping.")
    budget_key = str(budget)
    if budget_key not in budgets:
        raise ValueError(
            f"Promotion metric {metric!r} requested budget {budget}, "
            f"but evaluation report contains budgets "
            f"{_available_budget_keys(budgets)}.")
    container = budgets[budget_key]
    if not isinstance(container, Mapping):
        raise ValueError(
            f"Promotion metric {metric!r} requested budget {budget}, "
            "but that budget report is not a metric mapping.")
    return container


def _required_count(container: Mapping[str, Any], field: str, context: str) -> int:
    if field not in container:
        raise ValueError(f"{context} is missing count field {field!r}.")
    value = container[field]
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{context} has invalid count {field!r}: {value!r}.")
    if value < 0:
        raise ValueError(f"{context} has negative count {field!r}: {value}.")
    return value


def _evidence_fields(metric: str) -> tuple[str, str, str | None]:
    if metric.startswith("search_") and metric != "neg_search_regret":
        return (
            "search_optimal_classification_count",
            "search_optimal_classification_coverage",
            "search_confirmed_optimal_count",
        )
    if metric.startswith("raw_policy_"):
        return (
            "raw_policy_optimal_classification_count",
            "raw_policy_optimal_classification_coverage",
            "raw_policy_confirmed_optimal_count",
        )
    if metric == "neg_search_regret":
        return (
            "search_exact_regret_count",
            "search_exact_regret_coverage",
            None,
        )
    raise ValueError(f"unknown promotion metric: {metric}")


def validate_promotion_evidence(
    incumbent_report: Mapping[str, Any],
    candidate_report: Mapping[str, Any],
    *,
    metric: str,
    budget: int,
) -> PromotionEvidence:
    """Validate comparable evidence and return both larger-is-better scores."""
    incumbent_container = _metric_container(
        incumbent_report, metric=metric, budget=budget)
    candidate_container = _metric_container(
        candidate_report, metric=metric, budget=budget)
    context = f"Promotion evidence for metric {metric!r}"
    incumbent_total = _required_count(
        incumbent_container, "total_evaluated_count", f"{context} (incumbent)")
    candidate_total = _required_count(
        candidate_container, "total_evaluated_count", f"{context} (candidate)")
    if incumbent_total == 0 or candidate_total == 0:
        raise ValueError(f"{context} has an empty evaluation set.")
    if incumbent_total != candidate_total:
        raise ValueError(
            f"{context} has mismatched total evaluated counts: "
            f"incumbent={incumbent_total}, candidate={candidate_total}.")
    incumbent_score = promotion_score(
        incumbent_report, metric=metric, budget=budget)
    candidate_score = promotion_score(
        candidate_report, metric=metric, budget=budget)

    known_field, coverage_field, confirmed_field = _evidence_fields(metric)
    incumbent_known = _required_count(
        incumbent_container, known_field, f"{context} (incumbent)")
    candidate_known = _required_count(
        candidate_container, known_field, f"{context} (candidate)")
    incumbent_coverage = _required_value(
        incumbent_container, coverage_field, f"{context} (incumbent)")
    candidate_coverage = _required_value(
        candidate_container, coverage_field, f"{context} (candidate)")
    for role, known, coverage in (
            ("incumbent", incumbent_known, incumbent_coverage),
            ("candidate", candidate_known, candidate_coverage)):
        if known > incumbent_total:
            raise ValueError(
                f"{context} ({role}) known count {known} exceeds total "
                f"{incumbent_total}.")
        expected = known / incumbent_total
        if not math.isclose(coverage, expected, rel_tol=1e-9, abs_tol=1e-12):
            raise ValueError(
                f"{context} ({role}) coverage {coverage} does not match "
                f"known/total {known}/{incumbent_total}.")

    incumbent_confirmed = candidate_confirmed = None
    if confirmed_field is not None:
        incumbent_confirmed = _required_count(
            incumbent_container, confirmed_field, f"{context} (incumbent)")
        candidate_confirmed = _required_count(
            candidate_container, confirmed_field, f"{context} (candidate)")
        for role, confirmed, score in (
                ("incumbent", incumbent_confirmed, incumbent_score),
                ("candidate", candidate_confirmed, candidate_score)):
            role_known = (incumbent_known if role == "incumbent"
                          else candidate_known)
            if confirmed > role_known:
                raise ValueError(
                    f"{context} ({role}) confirmed count {confirmed} exceeds "
                    f"classification-known count {role_known}.")
            if metric.endswith("confirmed_optimal_rate"):
                expected = confirmed / incumbent_total
                if not math.isclose(score, expected, rel_tol=1e-9, abs_tol=1e-12):
                    raise ValueError(
                        f"{context} ({role}) score {score} does not match "
                        f"confirmed/total {confirmed}/{incumbent_total}.")

    if metric in _CONDITIONAL_METRICS and (
            incumbent_known != incumbent_total
            or candidate_known != candidate_total):
        raise ValueError(
            f"Promotion metric {metric!r} is conditional on coverage and "
            "requires full classification/evidence coverage for both models; "
            f"incumbent={incumbent_known}/{incumbent_total}, "
            f"candidate={candidate_known}/{candidate_total}. Use "
            "'search_confirmed_optimal_rate' for coverage-safe search promotion.")
    if confirmed_field is not None and metric in {
            "search_optimal_acc", "raw_policy_optimal_acc"}:
        for role, confirmed, known, score in (
                ("incumbent", incumbent_confirmed, incumbent_known,
                 incumbent_score),
                ("candidate", candidate_confirmed, candidate_known,
                 candidate_score)):
            expected = confirmed / known
            if not math.isclose(score, expected, rel_tol=1e-9, abs_tol=1e-12):
                raise ValueError(
                    f"{context} ({role}) score {score} does not match "
                    f"confirmed/known {confirmed}/{known}.")

    return PromotionEvidence(
        incumbent_score=incumbent_score,
        candidate_score=candidate_score,
        total_count=incumbent_total,
        incumbent_known_count=incumbent_known,
        candidate_known_count=candidate_known,
        incumbent_coverage=incumbent_coverage,
        candidate_coverage=candidate_coverage,
        incumbent_confirmed_count=incumbent_confirmed,
        candidate_confirmed_count=candidate_confirmed,
    )


def _validate_solve_rate_evidence(
    incumbent_report: Mapping[str, Any],
    candidate_report: Mapping[str, Any],
    *,
    budget: int,
) -> PromotionEvidence:
    """Validate full-level solve counts with total-evaluated denominators."""
    context = f"Full-level solve promotion evidence at budget {budget}"

    def fields(report: Mapping[str, Any], role: str) -> tuple[int, int, float]:
        budgets = report.get("budgets")
        if not isinstance(budgets, Mapping):
            raise ValueError(f"{context} ({role}) has no budgets mapping")
        container = budgets.get(str(budget))
        if not isinstance(container, Mapping):
            raise ValueError(f"{context} ({role}) is missing budget {budget}")
        total = _required_count(
            container, "total_evaluated_count", f"{context} ({role})")
        solved = _required_count(
            container, "search_solved_count", f"{context} ({role})")
        rate = _required_value(
            container, "search_solve_rate_total", f"{context} ({role})")
        if total == 0:
            raise ValueError(f"{context} ({role}) has an empty evaluation set")
        if solved > total:
            raise ValueError(
                f"{context} ({role}) solved count {solved} exceeds total "
                f"{total}")
        expected = solved / total
        if not math.isclose(rate, expected, rel_tol=1e-9, abs_tol=1e-12):
            raise ValueError(
                f"{context} ({role}) rate {rate} does not match "
                f"solved/total {solved}/{total}")
        return total, solved, rate

    incumbent_total, incumbent_solved, incumbent_rate = fields(
        incumbent_report, "incumbent")
    candidate_total, candidate_solved, candidate_rate = fields(
        candidate_report, "candidate")
    if incumbent_total != candidate_total:
        raise ValueError(
            f"{context} has mismatched total evaluated counts: "
            f"incumbent={incumbent_total}, candidate={candidate_total}")
    return PromotionEvidence(
        incumbent_score=incumbent_rate,
        candidate_score=candidate_rate,
        total_count=incumbent_total,
        incumbent_known_count=incumbent_total,
        candidate_known_count=candidate_total,
        incumbent_coverage=1.0,
        candidate_coverage=1.0,
        incumbent_confirmed_count=incumbent_solved,
        candidate_confirmed_count=candidate_solved,
        evidence_kind="solved",
    )


def validate_budget_sweep_promotion_evidence(
    incumbent_report: Mapping[str, Any],
    candidate_report: Mapping[str, Any],
    *,
    budgets: Iterable[int],
    weights: Iterable[float] | None = None,
    metric: str = "weighted_budget_sweep_confirmed_optimal_rate",
) -> PromotionEvidence:
    """Validate comparable evidence across many search budgets."""
    if metric not in _BUDGET_SWEEP_METRICS:
        raise ValueError(f"unknown budget-sweep promotion metric: {metric}")
    budget_list, budget_weights = normalize_budget_sweep_config(
        tuple(budgets), None if weights is None else tuple(weights))
    evidence_kind = (
        "solved" if metric == "weighted_budget_sweep_solve_rate"
        else "confirmed_optimal")
    per_budget: dict[str, Any] = {}
    incumbent_score = candidate_score = 0.0
    total_count: int | None = None
    inc_known = cand_known = inc_confirmed = cand_confirmed = 0
    for budget, weight in zip(budget_list, budget_weights):
        if evidence_kind == "solved":
            evidence = _validate_solve_rate_evidence(
                incumbent_report, candidate_report, budget=budget)
        else:
            evidence = validate_promotion_evidence(
                incumbent_report,
                candidate_report,
                metric="search_confirmed_optimal_rate",
                budget=budget,
            )
        if total_count is None:
            total_count = evidence.total_count
        elif evidence.total_count != total_count:
            raise ValueError(
                "budget-sweep promotion evidence has inconsistent total counts")
        incumbent_score += evidence.incumbent_score * weight
        candidate_score += evidence.candidate_score * weight
        inc_known += evidence.incumbent_known_count
        cand_known += evidence.candidate_known_count
        inc_confirmed += evidence.incumbent_confirmed_count or 0
        cand_confirmed += evidence.candidate_confirmed_count or 0
        outcome = "tie"
        if evidence.candidate_score > evidence.incumbent_score:
            outcome = "win"
        elif evidence.candidate_score < evidence.incumbent_score:
            outcome = "loss"
        budget_fields = {
            "weight": weight,
            "incumbent_score": evidence.incumbent_score,
            "candidate_score": evidence.candidate_score,
            "total_count": evidence.total_count,
            "outcome": outcome,
        }
        if evidence_kind == "solved":
            budget_fields.update({
                "incumbent_solved_count":
                    evidence.incumbent_confirmed_count,
                "candidate_solved_count":
                    evidence.candidate_confirmed_count,
            })
        else:
            budget_fields.update({
                "incumbent_classification_known_count":
                    evidence.incumbent_known_count,
                "candidate_classification_known_count":
                    evidence.candidate_known_count,
                "incumbent_classification_coverage":
                    evidence.incumbent_coverage,
                "candidate_classification_coverage":
                    evidence.candidate_coverage,
                "incumbent_confirmed_optimal_count":
                    evidence.incumbent_confirmed_count,
                "candidate_confirmed_optimal_count":
                    evidence.candidate_confirmed_count,
            })
        per_budget[str(budget)] = budget_fields
    assert total_count is not None
    denominator = total_count * len(budget_list)
    return PromotionEvidence(
        incumbent_score=incumbent_score,
        candidate_score=candidate_score,
        total_count=total_count,
        incumbent_known_count=inc_known,
        candidate_known_count=cand_known,
        incumbent_coverage=inc_known / denominator,
        candidate_coverage=cand_known / denominator,
        incumbent_confirmed_count=inc_confirmed,
        candidate_confirmed_count=cand_confirmed,
        budget_list=budget_list,
        budget_weights=budget_weights,
        per_budget=per_budget,
        evidence_kind=evidence_kind,
    )


def promotion_score(report: Mapping[str, Any], *, metric: str, budget: int) -> float:
    """Extract a real, available larger-is-better score from an evaluation."""
    if promotion_metric_requires_budget_sweep(metric):
        raise ValueError(
            f"Promotion metric {metric!r} requires budget-sweep evidence; "
            "use validate_budget_sweep_promotion_evidence.")
    requires_budget = promotion_metric_requires_budget(metric)
    if requires_budget:
        budget_report = _metric_container(report, metric=metric, budget=budget)
        field = _BUDGETED_METRIC_FIELDS[metric]
        value = _required_value(
            budget_report, field,
            f"Promotion metric {metric!r} at budget {budget}")
        return -value if metric == "neg_search_regret" else value

    field = _NON_BUDGETED_METRIC_FIELDS[metric]
    return _required_value(report, field, f"Promotion metric {metric!r}")
