"""Group-awareness ablation from cached KDD candidate logs.

This is a standard-library companion to ``bandha_strengthen_group_awareness``.
It works directly from the frozen compact W16-style candidate logs used by the
KDD group-complaint robustness checks, so it can be run on machines without
numpy/pandas.  The protocol is the same deployment-style selection rule:

* choose high/low complaint groups on validation under the unpriced W-BCE rule;
* select each family on validation subject to a capture floor;
* score exactly one selected policy per family on held-out test.

The output is value-unit only, which is the KDD positive fairness unit.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import math
import statistics
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path


COMPACT_DECODE: dict[str, dict[str, str]] = {
    "dataset": {
        "0": "australian_mtpl_group_blind",
        "1": "australian_mtpl_group_aware",
        "2": "australian_stress_moderate",
        "3": "australian_stress_strong",
        "4": "australian_stress_extreme",
        "5": "belgian_mtpl_group_blind",
        "6": "belgian_mtpl_group_aware",
        "7": "kuairand_group_blind",
        "8": "kuairand_group_aware",
        "9": "bondora_group_blind",
        "10": "bondora_group_aware",
    },
    "family": {
        "0": "natural_group_blind",
        "1": "natural_group_aware",
        "2": "constructed_stress",
        "3": "web_activity_group_blind",
        "4": "web_activity_group_aware",
    },
    "split": {
        "0": "validation",
        "1": "test",
    },
    "group": {
        "0": "Female",
        "1": "Male",
        "2": "full_active",
        "3": "high_active",
        "4": "middle_active",
        "5": "2_14_day_new",
        "6": "low_active",
        "7": "single_low_active",
        "8": "30day_retention",
        "9": "day_new",
        "10": "UNKNOWN",
    },
}

DEFAULT_DATASETS: tuple[str, ...] = (
    "australian_mtpl_group_aware",
    "australian_mtpl_group_blind",
    "belgian_mtpl_group_aware",
    "belgian_mtpl_group_blind",
    "bondora_group_aware",
)

FIXED_POLICIES: tuple[str, ...] = (
    "global_value",
    "equal_group_value",
    "equal_group_decline",
    "equal_group_protection",
)

DEFAULT_LAMBDAS: tuple[float, ...] = (
    0.0,
    *tuple(i * 0.005 for i in range(1, 61)),
    0.35,
    0.4,
    0.45,
    0.5,
    0.6,
    0.75,
    1.0,
    1.5,
    2.0,
    3.0,
    5.0,
    10.0,
)

SELECTION_FIELDS: tuple[str, ...] = (
    "dataset",
    "source",
    "seed",
    "capture_floor",
    "family",
    "policy",
    "lambda",
    "target",
    "budget_cap",
    "budget_protocol",
    "validation_thinning_alpha",
    "baseline_test_capture",
    "baseline_test_gap",
    "validation_capture_fraction",
    "test_capture_fraction",
    "validation_value_budget",
    "test_value_budget",
    "validation_budget_feasible",
    "validation_gap",
    "test_gap",
    "test_gap_closure_raw",
    "test_gap_closure_clipped",
)

COMPARISON_FIELDS: tuple[str, ...] = (
    "dataset",
    "source",
    "seed",
    "capture_floor",
    "closure_kind",
    "comparison",
    "left_family",
    "right_family",
    "left_closure",
    "right_closure",
    "lift",
    "left_test_capture_fraction",
    "right_test_capture_fraction",
    "left_test_gap",
    "right_test_gap",
)

SUMMARY_FIELDS: tuple[str, ...] = (
    "capture_floor",
    "closure_kind",
    "comparison",
    "materiality",
    "n_cells",
    "cell_positive",
    "cell_negative",
    "cell_tied",
    "median_lift",
    "mean_lift",
    "mean_lift_ci95",
    "source_positive",
    "source_negative",
    "source_tied",
    "source_sign_p_one_sided",
    "source_median_lifts",
)

SOURCE_FIELDS: tuple[str, ...] = (
    "capture_floor",
    "closure_kind",
    "comparison",
    "materiality",
    "source",
    "n_cells",
    "cell_positive",
    "cell_negative",
    "cell_tied",
    "median_lift",
    "mean_lift",
)

ELIGIBILITY_FIELDS: tuple[str, ...] = (
    "dataset",
    "source",
    "seed",
    "family",
    "target",
    "capture_floor",
    "status",
    "reason",
    "configured_beta",
    "budget_protocol",
    "baseline_validation_thinning_alpha",
    "fixed_candidate_count",
    "account_candidate_count",
    "group_candidate_count",
    "selected_fixed_policy",
    "selected_account_lambda",
    "selected_group_lambda",
)

EPS = 1e-12


@dataclass(frozen=True)
class Candidate:
    dataset: str
    family: str
    seed: str
    split: str
    group: str
    value: float
    loss: float
    base_ratio: float
    estimated_value: float
    complaint_any: bool
    fixed_selected: dict[str, bool]
    estimated_loss: float = 0.0


@dataclass(frozen=True)
class Metrics:
    capture: float
    value_budget: float
    signed_gap: float
    high_group: str
    low_group: str


@dataclass(frozen=True)
class PolicyResult:
    policy: str
    lambd: float | None
    validation_metrics: Metrics
    test_metrics: Metrics
    validation_thinning_alpha: float = 1.0


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        action="append",
        required=True,
        type=Path,
    )
    parser.add_argument("--dataset", action="append", default=None)
    parser.add_argument("--target", type=float, default=0.10)
    parser.add_argument("--capture-floors", default="0.90,0.95,0.98,1.00")
    parser.add_argument("--materialities", default="0,0.01")
    parser.add_argument("--lambda-grid", default="")
    parser.add_argument(
        "--selection-output",
        type=Path,
        default=root / "results/csvs/bandha_cached_group_awareness_selection.csv",
    )
    parser.add_argument(
        "--comparison-output",
        type=Path,
        default=root / "results/csvs/bandha_cached_group_awareness_comparison.csv",
    )
    parser.add_argument(
        "--summary-output",
        type=Path,
        default=root / "results/csvs/bandha_cached_group_awareness_summary.csv",
    )
    parser.add_argument(
        "--source-output",
        type=Path,
        default=root / "results/csvs/bandha_cached_group_awareness_source.csv",
    )
    parser.add_argument(
        "--eligibility-output",
        type=Path,
        default=root / "results/csvs/bandha_cached_group_awareness_eligibility.csv",
    )
    return parser.parse_args()


def parse_float_list(text: str) -> list[float]:
    return [float(value.strip()) for value in text.split(",") if value.strip()]


def parse_lambdas(text: str) -> list[float]:
    if text.strip():
        values = parse_float_list(text)
    else:
        values = list(DEFAULT_LAMBDAS)
    return sorted({round(max(0.0, value), 12) for value in values})


def decode(field: str, value: str) -> str:
    return COMPACT_DECODE.get(field, {}).get(str(value), str(value))


def to_float(value: str | None) -> float:
    if value is None or value == "":
        return float("nan")
    return float(value)


def fmt(value: float) -> str:
    if math.isnan(value):
        return ""
    if math.isinf(value):
        return "inf" if value > 0 else "-inf"
    return f"{value:.10g}"


def source_name(dataset: str) -> str:
    if dataset.startswith("australian_"):
        return "Australian MTPL"
    if dataset.startswith("belgian_"):
        return "Belgian MTPL"
    if dataset.startswith("bondora_"):
        return "Bondora"
    if dataset.startswith("cms_"):
        return "CMS provider fraud"
    if dataset.startswith("euhealth_"):
        return "EU health insurance"
    if dataset.startswith("german_credit_"):
        return "German credit"
    if dataset.startswith("uci_credit_"):
        return "UCI credit"
    if dataset.startswith("kuairand_"):
        return "KuaiRand activity"
    return dataset


def first_present(row: dict[str, str], *names: str) -> str:
    for name in names:
        if name in row:
            return row[name]
    raise KeyError(f"missing required column; tried {', '.join(names)}")


def selected_bool(row: dict[str, str], compact_name: str, verbose_name: str) -> bool:
    return to_float(first_present(row, compact_name, verbose_name)) > 0.5


def open_candidate_csv(path: Path):
    if path.suffix == ".gz":
        return gzip.open(path, "rt", newline="")
    return path.open(newline="")


def read_inputs(
    paths: list[Path],
    datasets: set[str],
    target: float,
) -> dict[tuple[str, str, str, str], list[Candidate]]:
    grouped: dict[tuple[str, str, str, str], list[Candidate]] = defaultdict(list)
    for path in paths:
        with open_candidate_csv(path) as handle:
            reader = csv.DictReader(handle)
            fields = set(reader.fieldnames or [])
            compact = {"d", "f", "s", "t", "p", "g", "A", "B", "q", "ah"}.issubset(
                fields
            )
            for row in reader:
                if compact:
                    dataset = decode("dataset", row["d"])
                    family = decode("family", row["f"])
                    seed = row["s"]
                    row_target = to_float(row["t"])
                    split = decode("split", row["p"]).lower()
                    group = decode("group", row["g"])
                    value = max(0.0, to_float(row["A"]))
                    loss = max(0.0, to_float(row["B"]))
                    score = to_float(row["q"])
                    estimated_value = max(0.0, to_float(row["ah"]))
                    estimated_loss = max(0.0, to_float(row["bh"]))
                    selected_harm = to_float(row["h"]) > 0.5
                    selected_net_harm = to_float(row["nh"]) > 0.5
                    fixed_selected = {
                        "global_value": to_float(row["gv"]) > 0.5,
                        "equal_group_value": to_float(row["egv"]) > 0.5,
                        "equal_group_decline": to_float(row["egd"]) > 0.5,
                        "equal_group_protection": to_float(row["egp"]) > 0.5,
                    }
                else:
                    dataset = row["dataset"]
                    family = row.get("family", "unknown")
                    seed = row["seed"]
                    row_target = to_float(row["target"])
                    split = row.get("sample_split", row.get("split", "")).lower()
                    group = row["group"]
                    value = max(0.0, to_float(row["value"]))
                    loss = max(0.0, to_float(row["loss"]))
                    score = to_float(row["score"])
                    estimated_value = max(
                        0.0,
                        to_float(first_present(row, "a_hat", "estimated_value")),
                    )
                    estimated_loss = max(
                        0.0,
                        to_float(first_present(row, "b_hat", "estimated_loss")),
                    )
                    selected_harm = selected_bool(
                        row,
                        "selected_harm",
                        "selected_criterion_harm",
                    )
                    selected_net_harm = selected_bool(
                        row,
                        "selected_net_harm",
                        "selected_criterion_net_harm",
                    )
                    fixed_selected = {
                        "global_value": selected_bool(
                            row,
                            "selected_global_value",
                            "selected_global_value",
                        ),
                        "equal_group_value": selected_bool(
                            row,
                            "selected_equal_group_value",
                            "selected_equal_group_value",
                        ),
                        "equal_group_decline": selected_bool(
                            row,
                            "selected_equal_group_decline",
                            "selected_equal_group_decline",
                        ),
                        "equal_group_protection": selected_bool(
                            row,
                            "selected_equal_group_protection",
                            "selected_equal_group_protection",
                        ),
                    }
                if dataset not in datasets or not math.isclose(row_target, target):
                    continue
                grouped[(dataset, family, seed, split)].append(
                    Candidate(
                        dataset=dataset,
                        family=family,
                        seed=seed,
                        split=split,
                        group=group,
                        value=value,
                        loss=loss,
                        base_ratio=math.exp(max(-50.0, min(50.0, score))),
                        estimated_value=estimated_value,
                        estimated_loss=estimated_loss,
                        complaint_any=(not selected_harm) or (not selected_net_harm),
                        fixed_selected=fixed_selected,
                    )
                )
    return grouped


def safe_divide(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator > 0.0 else float("nan")


def group_base_rates(candidates: list[Candidate]) -> dict[str, float]:
    total_rows = float(len(candidates))
    by_group: dict[str, float] = defaultdict(float)
    for candidate in candidates:
        by_group[candidate.group] += candidate.estimated_value
    return {group: value / total_rows for group, value in by_group.items()}


def policy_scores(
    candidates: list[Candidate],
    policy: str,
    lambd: float,
    high_group: str,
    low_group: str,
    rates: dict[str, float],
) -> list[float]:
    scores: list[float] = []
    for candidate in candidates:
        if candidate.estimated_value <= 0.0:
            scores.append(candidate.base_ratio if candidate.base_ratio > 0.0 else 0.0)
            continue
        adjusted_loss = candidate.base_ratio * candidate.estimated_value
        if candidate.complaint_any:
            if policy == "account_penalty":
                adjusted_loss -= lambd * candidate.estimated_value
            elif policy == "group_penalty":
                if candidate.group == high_group and rates.get(high_group, 0.0) > 0.0:
                    adjusted_loss -= lambd * candidate.estimated_value / rates[high_group]
                elif candidate.group == low_group and rates.get(low_group, 0.0) > 0.0:
                    adjusted_loss += lambd * candidate.estimated_value / rates[low_group]
            elif policy != "baseline":
                raise ValueError(f"unknown policy: {policy}")
        if adjusted_loss <= 0.0:
            scores.append(0.0)
        else:
            scores.append(adjusted_loss / candidate.estimated_value)
    return scores


def select_for_budget(
    candidates: list[Candidate],
    scores: list[float],
    target: float,
) -> tuple[float, list[bool]]:
    total_value = sum(candidate.value for candidate in candidates)
    if not candidates or total_value <= 0.0 or target <= 0.0:
        return math.inf, [False] * len(candidates)
    order = sorted(range(len(scores)), key=lambda index: (-scores[index], index))
    cumulative = 0.0
    threshold = 0.0
    for index in order:
        if scores[index] <= 0.0:
            break
        cumulative += candidates[index].value
        threshold = scores[index]
        if cumulative >= target * total_value:
            break
    selected = [score >= threshold and score > 0.0 for score in scores]
    return threshold, selected


def metrics(
    candidates: list[Candidate],
    selected: list[bool],
    high_group: str | None,
    low_group: str | None,
) -> Metrics:
    total_loss = sum(candidate.loss for candidate in candidates)
    total_value = sum(candidate.value for candidate in candidates)
    total_by_group: dict[str, float] = defaultdict(float)
    complaint_by_group: dict[str, float] = defaultdict(float)
    for candidate, is_selected in zip(candidates, selected, strict=True):
        total_by_group[candidate.group] += candidate.value
        if is_selected and candidate.complaint_any:
            complaint_by_group[candidate.group] += candidate.value
    burdens = {
        group: safe_divide(complaint_by_group[group], total)
        for group, total in total_by_group.items()
    }
    finite = {group: value for group, value in burdens.items() if math.isfinite(value)}
    if finite:
        observed_high = max(finite, key=lambda group: finite[group])
        observed_low = min(finite, key=lambda group: finite[group])
    else:
        observed_high = ""
        observed_low = ""
    if high_group is None or low_group is None:
        high_group = observed_high
        low_group = observed_low
    return Metrics(
        capture=safe_divide(
            sum(
                candidate.loss
                for candidate, is_selected in zip(candidates, selected, strict=True)
                if is_selected
            ),
            total_loss,
        ),
        value_budget=safe_divide(
            sum(
                candidate.value
                for candidate, is_selected in zip(candidates, selected, strict=True)
                if is_selected
            ),
            total_value,
        ),
        signed_gap=burdens.get(high_group, float("nan"))
        - burdens.get(low_group, float("nan")),
        high_group=observed_high,
        low_group=observed_low,
    )


def fixed_policy_result(
    policy: str,
    validation: list[Candidate],
    test: list[Candidate],
    high_group: str,
    low_group: str,
) -> PolicyResult:
    return PolicyResult(
        policy=policy,
        lambd=None,
        validation_metrics=metrics(
            validation,
            [candidate.fixed_selected[policy] for candidate in validation],
            high_group,
            low_group,
        ),
        test_metrics=metrics(
            test,
            [candidate.fixed_selected[policy] for candidate in test],
            high_group,
            low_group,
        ),
    )


def direct_policy_result(
    policy: str,
    lambd: float,
    validation: list[Candidate],
    test: list[Candidate],
    target: float,
    high_group: str,
    low_group: str,
    rates: dict[str, float],
) -> PolicyResult:
    validation_scores = policy_scores(validation, policy, lambd, high_group, low_group, rates)
    threshold, validation_selected = select_for_budget(validation, validation_scores, target)
    test_scores = policy_scores(test, policy, lambd, high_group, low_group, rates)
    test_selected = [score >= threshold and score > 0.0 for score in test_scores]
    return PolicyResult(
        policy=policy,
        lambd=lambd,
        validation_metrics=metrics(validation, validation_selected, high_group, low_group),
        test_metrics=metrics(test, test_selected, high_group, low_group),
    )


def thinning_alpha(metrics_value: Metrics, budget_cap: float) -> float:
    budget = metrics_value.value_budget
    if not math.isfinite(budget):
        return 1.0
    if budget <= budget_cap + EPS:
        return 1.0
    if budget <= EPS:
        return 1.0
    return budget_cap / budget


def scale_metrics(metrics_value: Metrics, alpha: float) -> Metrics:
    return Metrics(
        capture=metrics_value.capture * alpha,
        value_budget=metrics_value.value_budget * alpha,
        signed_gap=metrics_value.signed_gap * alpha,
        high_group=metrics_value.high_group,
        low_group=metrics_value.low_group,
    )


def thinned_policy_result(result: PolicyResult, budget_cap: float) -> PolicyResult:
    alpha = thinning_alpha(result.validation_metrics, budget_cap)
    return PolicyResult(
        policy=result.policy,
        lambd=result.lambd,
        validation_metrics=scale_metrics(result.validation_metrics, alpha),
        test_metrics=scale_metrics(result.test_metrics, alpha),
        validation_thinning_alpha=alpha,
    )


def choose_policy(
    candidates: list[PolicyResult],
    baseline_validation_capture: float,
    capture_floor: float,
    budget_cap: float,
) -> PolicyResult | None:
    eligible = [
        candidate
        for candidate in candidates
        if math.isfinite(candidate.validation_metrics.capture)
        and candidate.validation_metrics.capture
        >= capture_floor * baseline_validation_capture
        and math.isfinite(candidate.validation_metrics.value_budget)
        and candidate.validation_metrics.value_budget <= budget_cap + 1e-10
    ]
    if not eligible:
        return None
    return min(
        eligible,
        key=lambda result: (
            abs(result.validation_metrics.signed_gap)
            if math.isfinite(result.validation_metrics.signed_gap)
            else math.inf,
            -result.validation_metrics.capture
            if math.isfinite(result.validation_metrics.capture)
            else math.inf,
            result.lambd if result.lambd is not None else -1.0,
            result.policy,
        ),
    )


def closure(gap: float, baseline_gap: float, *, clipped: bool) -> float:
    if baseline_gap <= 0.0:
        value = 1.0 if gap <= 0.0 else 0.0
    else:
        value = 1.0 - gap / baseline_gap
    if clipped:
        return max(0.0, min(1.0, value))
    return value


def mean(values: Iterable[float]) -> float:
    finite = [value for value in values if math.isfinite(value)]
    return statistics.fmean(finite) if finite else float("nan")


def ci95(values: Iterable[float]) -> float:
    finite = [value for value in values if math.isfinite(value)]
    if len(finite) < 2:
        return 0.0 if finite else float("nan")
    return 1.96 * statistics.stdev(finite) / math.sqrt(len(finite))


def sign(value: float, materiality: float) -> int:
    if value > materiality:
        return 1
    if value < -materiality:
        return -1
    return 0


def binomial_upper_tail(successes: int, trials: int) -> float:
    if trials <= 0:
        return 1.0
    return sum(math.comb(trials, k) for k in range(successes, trials + 1)) / (2**trials)


def write_rows(path: Path, rows: list[dict[str, str]], fields: tuple[str, ...]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def build_outputs(
    grouped: dict[tuple[str, str, str, str], list[Candidate]],
    capture_floors: list[float],
    materialities: list[float],
    lambdas: list[float],
    target: float,
) -> tuple[
    list[dict[str, str]],
    list[dict[str, str]],
    list[dict[str, str]],
    list[dict[str, str]],
    list[dict[str, str]],
]:
    selection_rows: list[dict[str, str]] = []
    comparison_rows: list[dict[str, str]] = []
    eligibility_rows: list[dict[str, str]] = []
    keys = sorted({(dataset, family, seed) for dataset, family, seed, _ in grouped})
    for dataset, family, seed in keys:
        validation = grouped.get((dataset, family, seed, "validation"), [])
        test = grouped.get((dataset, family, seed, "test"), [])
        if not validation or not test:
            continue
        rates = group_base_rates(validation)
        if len(rates) != 2:
            continue
        baseline_scores = policy_scores(
            validation,
            "baseline",
            0.0,
            *sorted(rates),
            rates,
        )
        _, baseline_selected = select_for_budget(validation, baseline_scores, target)
        provisional = metrics(validation, baseline_selected, None, None)
        high_group = provisional.high_group
        low_group = provisional.low_group
        if not high_group or not low_group:
            continue
        fixed_candidates = [
            thinned_policy_result(
                fixed_policy_result(policy, validation, test, high_group, low_group),
                target,
            )
            for policy in FIXED_POLICIES
        ]
        account_candidates = [
            thinned_policy_result(
                direct_policy_result(
                    "account_penalty",
                    lambd,
                    validation,
                    test,
                    target,
                    high_group,
                    low_group,
                    rates,
                ),
                target,
            )
            for lambd in lambdas
        ]
        group_candidates = [
            thinned_policy_result(
                direct_policy_result(
                    "group_penalty",
                    lambd,
                    validation,
                    test,
                    target,
                    high_group,
                    low_group,
                    rates,
                ),
                target,
            )
            for lambd in lambdas
        ]
        baseline = thinned_policy_result(
            fixed_policy_result(
                "global_value",
                validation,
                test,
                high_group,
                low_group,
            ),
            target,
        )
        baseline_validation_capture = baseline.validation_metrics.capture
        baseline_test_capture = baseline.test_metrics.capture
        baseline_test_gap = abs(baseline.test_metrics.signed_gap)
        for capture_floor in capture_floors:
            eligibility = {
                "dataset": dataset,
                "source": source_name(dataset),
                "seed": seed,
                "family": family,
                "target": fmt(target),
                "capture_floor": fmt(capture_floor),
                "status": "paired_eligible",
                "reason": "ok",
                "configured_beta": fmt(target),
                "budget_protocol": "validation_frozen_randomized_thinning",
                "baseline_validation_thinning_alpha": fmt(
                    baseline.validation_thinning_alpha
                ),
                "fixed_candidate_count": str(len(fixed_candidates)),
                "account_candidate_count": str(len(account_candidates)),
                "group_candidate_count": str(len(group_candidates)),
                "selected_fixed_policy": "",
                "selected_account_lambda": "",
                "selected_group_lambda": "",
            }
            if not (
                baseline_validation_capture > 0.0
                and baseline_test_capture > 0.0
                and baseline_test_gap > 0.0
            ):
                eligibility["status"] = "invalid_baseline"
                eligibility["reason"] = "baseline_capture_or_gap_is_zero"
                eligibility_rows.append(eligibility)
                continue
            selected = {
                "fixed": choose_policy(
                    fixed_candidates,
                    baseline_validation_capture,
                    capture_floor,
                    target,
                ),
                "account": choose_policy(
                    account_candidates,
                    baseline_validation_capture,
                    capture_floor,
                    target,
                ),
                "group": choose_policy(
                    group_candidates,
                    baseline_validation_capture,
                    capture_floor,
                    target,
                ),
            }
            if selected["fixed"] is None:
                eligibility["status"] = "fixed_infeasible"
                eligibility["reason"] = (
                    "no_fixed_policy_meets_capture_floor_and_configured_beta"
                )
                eligibility_rows.append(eligibility)
                continue
            if selected["account"] is None:
                eligibility["status"] = "account_infeasible"
                eligibility["reason"] = (
                    "no_account_policy_meets_capture_floor_and_configured_beta"
                )
                eligibility_rows.append(eligibility)
                continue
            if selected["group"] is None:
                eligibility["status"] = "group_infeasible"
                eligibility["reason"] = (
                    "no_group_policy_meets_capture_floor_and_configured_beta"
                )
                eligibility_rows.append(eligibility)
                continue
            eligibility["selected_fixed_policy"] = selected["fixed"].policy
            eligibility["selected_account_lambda"] = (
                fmt(selected["account"].lambd)
                if selected["account"].lambd is not None
                else ""
            )
            eligibility["selected_group_lambda"] = (
                fmt(selected["group"].lambd)
                if selected["group"].lambd is not None
                else ""
            )
            eligibility_rows.append(eligibility)
            measures: dict[str, dict[str, float]] = {}
            for selected_family, result in selected.items():
                assert result is not None
                test_gap = abs(result.test_metrics.signed_gap)
                test_capture_fraction = result.test_metrics.capture / baseline_test_capture
                measures[selected_family] = {
                    "validation_capture_fraction": result.validation_metrics.capture
                    / baseline_validation_capture,
                    "test_capture_fraction": test_capture_fraction,
                    "validation_gap": abs(result.validation_metrics.signed_gap),
                    "test_gap": test_gap,
                    "raw": closure(test_gap, baseline_test_gap, clipped=False),
                    "clipped": closure(test_gap, baseline_test_gap, clipped=True),
                }
                selection_rows.append(
                    {
                        "dataset": dataset,
                        "source": source_name(dataset),
                        "seed": seed,
                        "capture_floor": fmt(capture_floor),
                        "family": selected_family,
                        "policy": result.policy,
                        "lambda": fmt(result.lambd) if result.lambd is not None else "",
                        "target": fmt(target),
                        "budget_cap": fmt(target),
                        "budget_protocol": "validation_frozen_randomized_thinning",
                        "validation_thinning_alpha": fmt(
                            result.validation_thinning_alpha
                        ),
                        "baseline_test_capture": fmt(baseline_test_capture),
                        "baseline_test_gap": fmt(baseline_test_gap),
                        "validation_capture_fraction": fmt(
                            measures[selected_family]["validation_capture_fraction"]
                        ),
                        "test_capture_fraction": fmt(test_capture_fraction),
                        "validation_value_budget": fmt(
                            result.validation_metrics.value_budget
                        ),
                        "test_value_budget": fmt(result.test_metrics.value_budget),
                        "validation_budget_feasible": str(
                            int(result.validation_metrics.value_budget <= target + 1e-10)
                        ),
                        "validation_gap": fmt(
                            measures[selected_family]["validation_gap"]
                        ),
                        "test_gap": fmt(test_gap),
                        "test_gap_closure_raw": fmt(measures[selected_family]["raw"]),
                        "test_gap_closure_clipped": fmt(
                            measures[selected_family]["clipped"]
                        ),
                    }
                )
            for closure_kind in ("raw", "clipped"):
                for comparison, left, right in (
                    ("group_vs_account", "group", "account"),
                    ("group_vs_fixed", "group", "fixed"),
                    ("account_vs_fixed", "account", "fixed"),
                ):
                    comparison_rows.append(
                        {
                            "dataset": dataset,
                            "source": source_name(dataset),
                            "seed": seed,
                            "capture_floor": fmt(capture_floor),
                            "closure_kind": closure_kind,
                            "comparison": comparison,
                            "left_family": left,
                            "right_family": right,
                            "left_closure": fmt(measures[left][closure_kind]),
                            "right_closure": fmt(measures[right][closure_kind]),
                            "lift": fmt(
                                measures[left][closure_kind]
                                - measures[right][closure_kind]
                            ),
                            "left_test_capture_fraction": fmt(
                                measures[left]["test_capture_fraction"]
                            ),
                            "right_test_capture_fraction": fmt(
                                measures[right]["test_capture_fraction"]
                            ),
                            "left_test_gap": fmt(measures[left]["test_gap"]),
                            "right_test_gap": fmt(measures[right]["test_gap"]),
                        }
                    )

    source_medians_by_materiality: dict[
        tuple[str, str, str, float], dict[str, float]
    ] = defaultdict(dict)
    grouped_for_source: dict[tuple[str, str, str, str], list[float]] = defaultdict(list)
    for row in comparison_rows:
        grouped_for_source[
            (
                row["capture_floor"],
                row["closure_kind"],
                row["comparison"],
                row["source"],
            )
        ].append(to_float(row["lift"]))
    source_rows: list[dict[str, str]] = []
    for key, lifts in sorted(grouped_for_source.items()):
        floor, closure_kind, comparison, source = key
        for materiality in materialities:
            signs = [sign(value, materiality) for value in lifts]
            median_lift = statistics.median(lifts)
            source_medians_by_materiality[
                (floor, closure_kind, comparison, materiality)
            ][source] = median_lift
            source_rows.append(
                {
                    "capture_floor": floor,
                    "closure_kind": closure_kind,
                    "comparison": comparison,
                    "materiality": fmt(materiality),
                    "source": source,
                    "n_cells": str(len(lifts)),
                    "cell_positive": str(sum(value > 0 for value in signs)),
                    "cell_negative": str(sum(value < 0 for value in signs)),
                    "cell_tied": str(sum(value == 0 for value in signs)),
                    "median_lift": fmt(median_lift),
                    "mean_lift": fmt(mean(lifts)),
                }
            )

    summary_rows: list[dict[str, str]] = []
    grouped_for_summary: dict[tuple[str, str, str], list[float]] = defaultdict(list)
    for row in comparison_rows:
        grouped_for_summary[
            (row["capture_floor"], row["closure_kind"], row["comparison"])
        ].append(to_float(row["lift"]))
    for key, lifts in sorted(grouped_for_summary.items()):
        floor, closure_kind, comparison = key
        for materiality in materialities:
            cell_signs = [sign(value, materiality) for value in lifts]
            source_medians = source_medians_by_materiality[
                (floor, closure_kind, comparison, materiality)
            ]
            source_signs = [sign(value, materiality) for value in source_medians.values()]
            source_pos = sum(value > 0 for value in source_signs)
            source_neg = sum(value < 0 for value in source_signs)
            source_trials = source_pos + source_neg
            summary_rows.append(
                {
                    "capture_floor": floor,
                    "closure_kind": closure_kind,
                    "comparison": comparison,
                    "materiality": fmt(materiality),
                    "n_cells": str(len(lifts)),
                    "cell_positive": str(sum(value > 0 for value in cell_signs)),
                    "cell_negative": str(sum(value < 0 for value in cell_signs)),
                    "cell_tied": str(sum(value == 0 for value in cell_signs)),
                    "median_lift": fmt(statistics.median(lifts)),
                    "mean_lift": fmt(mean(lifts)),
                    "mean_lift_ci95": fmt(ci95(lifts)),
                    "source_positive": str(source_pos),
                    "source_negative": str(source_neg),
                    "source_tied": str(sum(value == 0 for value in source_signs)),
                    "source_sign_p_one_sided": fmt(
                        binomial_upper_tail(source_pos, source_trials)
                    ),
                    "source_median_lifts": ";".join(
                        f"{source}:{fmt(source_medians[source])}"
                        for source in sorted(source_medians)
                    ),
                }
            )
    return selection_rows, comparison_rows, summary_rows, source_rows, eligibility_rows


def main() -> None:
    args = parse_args()
    datasets = set(args.dataset or DEFAULT_DATASETS)
    grouped = read_inputs(args.input, datasets, args.target)
    selection_rows, comparison_rows, summary_rows, source_rows, eligibility_rows = build_outputs(
        grouped,
        parse_float_list(args.capture_floors),
        parse_float_list(args.materialities),
        parse_lambdas(args.lambda_grid),
        args.target,
    )
    write_rows(args.selection_output, selection_rows, SELECTION_FIELDS)
    write_rows(args.comparison_output, comparison_rows, COMPARISON_FIELDS)
    write_rows(args.summary_output, summary_rows, SUMMARY_FIELDS)
    write_rows(args.source_output, source_rows, SOURCE_FIELDS)
    write_rows(args.eligibility_output, eligibility_rows, ELIGIBILITY_FIELDS)
    print(f"wrote {args.selection_output} ({len(selection_rows)} rows)")
    print(f"wrote {args.comparison_output} ({len(comparison_rows)} rows)")
    print(f"wrote {args.summary_output} ({len(summary_rows)} rows)")
    print(f"wrote {args.source_output} ({len(source_rows)} rows)")
    print(f"wrote {args.eligibility_output} ({len(eligibility_rows)} rows)")
    print("main cached value-unit ablation, floor=0.95, raw closure, materiality=0.01")
    for row in summary_rows:
        if (
            row["capture_floor"] == "0.95"
            and row["closure_kind"] == "raw"
            and row["materiality"] == "0.01"
        ):
            print(
                row["comparison"],
                "median",
                row["median_lift"],
                "sources",
                f"{row['source_positive']}/{row['source_negative']}/{row['source_tied']}",
                "p",
                row["source_sign_p_one_sided"],
            )


if __name__ == "__main__":
    main()
