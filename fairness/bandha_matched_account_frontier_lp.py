"""Validation-only matched comparator for group-aware complaint pricing.

The existing group-awareness claim compares a per-group price frontier against
a group-blind account-price frontier.  This script gives the group-blind side a
matched finite-bank comparator: for each dataset, seed, currency, and capture
floor, choose a mixture over the full frozen account-frontier operating-point
bank that minimizes the validation signed complaint gap while meeting the
configured validation capture floor and blocked-value budget.  The chosen
mixture is then scored on test rows.  Mixtures are evaluated by mixing signed
gaps first and taking the absolute value after mixing.

No test quantity is used in the selection rule.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path


FLOORS: tuple[float, ...] = (0.90, 0.95, 0.98, 1.00)
EPS = 1e-12

DETAIL_FIELDS: tuple[str, ...] = (
    "source_artifact",
    "dataset",
    "source",
    "seed",
    "currency",
    "target",
    "capture_floor",
    "eligibility_status",
    "eligibility_reason",
    "budget_cap",
    "budget_protocol",
    "baseline_validation_thinning_alpha",
    "baseline_test_gap",
    "group_lambda",
    "group_validation_thinning_alpha",
    "group_validation_signed_gap",
    "group_validation_gap",
    "group_test_signed_gap",
    "group_test_gap",
    "group_validation_capture_fraction",
    "group_test_capture_fraction",
    "group_validation_value_budget",
    "group_test_value_budget",
    "group_validation_over_nominal_target",
    "group_test_closure_raw",
    "account_left_lambda",
    "account_right_lambda",
    "account_left_weight",
    "account_support_size",
    "account_component_weights_json",
    "account_lp_solver_status",
    "account_lp_gap_tolerance",
    "account_lp_secondary_tolerance",
    "account_lp_capture_residual",
    "account_lp_budget_residual",
    "account_lp_validation_signed_gap",
    "account_lp_validation_gap",
    "account_lp_test_signed_gap",
    "account_lp_test_gap",
    "account_lp_validation_capture_fraction",
    "account_lp_test_capture_fraction",
    "account_lp_validation_value_budget",
    "account_lp_test_value_budget",
    "account_lp_validation_over_nominal_target",
    "account_lp_validation_budget_feasible",
    "account_lp_test_closure_raw",
    "lift_vs_account_lp",
)

SUMMARY_FIELDS: tuple[str, ...] = (
    "source_artifact",
    "dataset",
    "source",
    "currency",
    "target",
    "capture_floor",
    "intended_cells",
    "paired_eligible_cells",
    "group_infeasible_cells",
    "account_infeasible_cells",
    "missing_input_cells",
    "n_seeds",
    "mean_group_closure",
    "mean_account_lp_closure",
    "median_lift_vs_account_lp",
    "mean_lift_vs_account_lp",
    "positive_seeds",
    "negative_seeds",
    "tied_seeds",
    "mean_group_test_gap",
    "mean_account_lp_test_gap",
    "mean_group_test_capture_fraction",
    "mean_account_lp_test_capture_fraction",
)

ELIGIBILITY_FIELDS: tuple[str, ...] = (
    "source_artifact",
    "dataset",
    "source",
    "seed",
    "currency",
    "target",
    "capture_floor",
    "status",
    "reason",
    "configured_beta",
    "group_candidate_count",
    "account_candidate_count",
)


@dataclass(frozen=True)
class Mixture:
    components: tuple[tuple[dict[str, str], float], ...]
    solver_status: str = "optimal_vertex_enumeration"
    gap_tolerance: float = 1e-10
    secondary_tolerance: float = 1e-10

    @property
    def left(self) -> dict[str, str]:
        return self.components[0][0]

    @property
    def right(self) -> dict[str, str]:
        if len(self.components) > 1:
            return self.components[1][0]
        return self.components[0][0]

    @property
    def left_weight(self) -> float:
        return self.components[0][1]


@dataclass(frozen=True)
class Point:
    row: dict[str, str]
    signed_gap: float
    capture: float
    budget: float
    lambda_value: float


@dataclass(frozen=True)
class LPResult:
    value: float
    solution: list[float]


@dataclass(frozen=True)
class SolverChoice:
    weights: list[float]
    solver_status: str
    gap_tolerance: float
    secondary_tolerance: float


class LinearProgram:
    """Two-phase simplex for max c*x subject to A*x <= b and x >= 0."""

    def __init__(
        self,
        a_matrix: list[list[float]],
        b_vector: list[float],
        c_vector: list[float],
    ) -> None:
        self.m = len(b_vector)
        self.n = len(c_vector)
        self.basis = [0] * self.m
        self.nonbasis = [0] * (self.n + 1)
        self.tableau = [
            [0.0 for _ in range(self.n + 2)] for _ in range(self.m + 2)
        ]
        for i in range(self.m):
            for j in range(self.n):
                self.tableau[i][j] = a_matrix[i][j]
            self.basis[i] = self.n + i
            self.tableau[i][self.n] = -1.0
            self.tableau[i][self.n + 1] = b_vector[i]
        for j in range(self.n):
            self.nonbasis[j] = j
            self.tableau[self.m][j] = -c_vector[j]
        self.nonbasis[self.n] = -1
        self.tableau[self.m + 1][self.n] = 1.0

    def pivot(self, row: int, col: int) -> None:
        inv = 1.0 / self.tableau[row][col]
        for i in range(self.m + 2):
            if i == row:
                continue
            for j in range(self.n + 2):
                if j == col:
                    continue
                self.tableau[i][j] -= (
                    self.tableau[row][j] * self.tableau[i][col] * inv
                )
        for j in range(self.n + 2):
            if j != col:
                self.tableau[row][j] *= inv
        for i in range(self.m + 2):
            if i != row:
                self.tableau[i][col] *= -inv
        self.tableau[row][col] = inv
        self.basis[row], self.nonbasis[col] = self.nonbasis[col], self.basis[row]

    def simplex(self, phase: int) -> bool:
        objective_row = self.m + 1 if phase == 1 else self.m
        while True:
            entering = -1
            for col in range(self.n + 1):
                if phase == 2 and self.nonbasis[col] == -1:
                    continue
                if entering == -1:
                    entering = col
                    continue
                value = self.tableau[objective_row][col]
                best = self.tableau[objective_row][entering]
                if value < best - 1e-10 or (
                    abs(value - best) <= 1e-10
                    and self.nonbasis[col] < self.nonbasis[entering]
                ):
                    entering = col
            if self.tableau[objective_row][entering] >= -1e-10:
                return True

            leaving = -1
            for row in range(self.m):
                if self.tableau[row][entering] <= 1e-10:
                    continue
                if leaving == -1:
                    leaving = row
                    continue
                ratio = self.tableau[row][self.n + 1] / self.tableau[row][entering]
                best_ratio = (
                    self.tableau[leaving][self.n + 1]
                    / self.tableau[leaving][entering]
                )
                if ratio < best_ratio - 1e-10 or (
                    abs(ratio - best_ratio) <= 1e-10
                    and self.basis[row] < self.basis[leaving]
                ):
                    leaving = row
            if leaving == -1:
                return False
            self.pivot(leaving, entering)

    def solve(self) -> LPResult | None:
        row = min(range(self.m), key=lambda idx: self.tableau[idx][self.n + 1])
        if self.tableau[row][self.n + 1] < -1e-10:
            self.pivot(row, self.n)
            if not self.simplex(1) or self.tableau[self.m + 1][self.n + 1] < -1e-10:
                return None
            if abs(self.tableau[self.m + 1][self.n + 1]) > 1e-8:
                return None
            artificial_row = next(
                (idx for idx, basic in enumerate(self.basis) if basic == -1),
                None,
            )
            if artificial_row is not None:
                entering = next(
                    (
                        col
                        for col in range(self.n + 1)
                        if abs(self.tableau[artificial_row][col]) > 1e-10
                    ),
                    None,
                )
                if entering is not None:
                    self.pivot(artificial_row, entering)
        if not self.simplex(2):
            return None
        solution = [0.0] * self.n
        for row_idx, basic in enumerate(self.basis):
            if 0 <= basic < self.n:
                solution[basic] = self.tableau[row_idx][self.n + 1]
        return LPResult(self.tableau[self.m][self.n + 1], solution)


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--frontier-input",
        action="append",
        type=Path,
        default=[
            root / "results/csvs/wsdm_group_complaint_burden_frontier.csv",
            root / "results/csvs/wsdm_group_complaint_burden_frontier_natural_wbce.csv",
            root / "results/csvs/bandha_group_complaint_promoted_frontier.csv",
            root / "results/csvs/bandha_euhealth_group_complaint_frontier.csv",
        ],
    )
    parser.add_argument("--currency", default="value")
    parser.add_argument("--capture-floors", default="0.90,0.95,0.98,1.00")
    parser.add_argument(
        "--detail-output",
        type=Path,
        default=root / "results/csvs/bandha_matched_account_frontier_lp_detail.csv",
    )
    parser.add_argument(
        "--summary-output",
        type=Path,
        default=root / "results/csvs/bandha_matched_account_frontier_lp_summary.csv",
    )
    parser.add_argument(
        "--eligibility-output",
        type=Path,
        default=root / "results/csvs/bandha_matched_account_frontier_lp_eligibility.csv",
    )
    return parser.parse_args()


def source_name(dataset: str) -> str:
    if dataset.startswith("australian_stress_"):
        return dataset.replace("australian_stress_", "Australian stress ")
    if dataset.startswith("freMPL"):
        return "freMPL"
    if dataset.startswith("norauto"):
        return "Norwegian auto"
    if dataset.startswith("swmotorcycle"):
        return "Swedish motorcycle"
    names = {
        "australian_mtpl_group_aware": "Australian MTPL",
        "australian_mtpl_group_blind": "Australian MTPL group-blind score",
        "belgian_mtpl_group_aware": "Belgian MTPL",
        "belgian_mtpl_group_blind": "Belgian MTPL group-blind score",
        "beMTPL97": "Belgian MTPL",
        "bondora_group_aware": "Bondora",
        "euhealth_gender": "EU health insurance",
    }
    return names.get(dataset, dataset)


def f(value: object | None) -> float:
    if value is None or value == "":
        return float("nan")
    return float(value)


def fmt(value: float) -> str:
    if math.isnan(value):
        return ""
    if math.isinf(value):
        return "inf" if value > 0.0 else "-inf"
    return f"{value:.10g}"


def parse_floors(text: str) -> list[float]:
    return [float(value.strip()) for value in text.split(",") if value.strip()]


def read_rows(path: Path, currency: str) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return [row for row in csv.DictReader(handle) if row["currency"] == currency]


def artifact_name(path: Path) -> str:
    return path.name


def key_for(row: dict[str, str], source_artifact: str) -> tuple[str, str, str, str, str]:
    return (
        source_artifact,
        row["dataset"],
        row["seed"],
        row["currency"],
        row["target"],
    )


def metric(row: dict[str, str], column: str) -> float:
    return f(row[column])


def mix_value(mixture: Mixture, column: str) -> float:
    return sum(weight * metric(row, column) for row, weight in mixture.components)


def closure(gap: float, baseline_gap: float) -> float:
    if baseline_gap <= 0.0:
        return 1.0 if gap <= 0.0 else 0.0
    return 1.0 - gap / baseline_gap


def feasible(row: dict[str, str], floor: float, budget_cap: float) -> bool:
    return (
        metric(row, "validation_capture_fraction_vs_baseline") + EPS >= floor
        and metric(row, "validation_value_budget") <= budget_cap + EPS
    )


def choose_single(
    rows: list[dict[str, str]],
    floor: float,
    budget_cap: float,
) -> dict[str, str] | None:
    candidates = [row for row in rows if feasible(row, floor, budget_cap)]
    if not candidates:
        return None
    return min(
        candidates,
        key=lambda row: (
            abs(metric(row, "validation_signed_complaint_gap")),
            -metric(row, "validation_capture_fraction_vs_baseline"),
            metric(row, "validation_value_budget"),
            metric(row, "lambda"),
        ),
    )


def choose_account_mixture(
    rows: list[dict[str, str]],
    floor: float,
    budget_cap: float,
) -> Mixture | None:
    points = [
        Point(
            row=row,
            signed_gap=metric(row, "validation_signed_complaint_gap"),
            capture=metric(row, "validation_capture_fraction_vs_baseline"),
            budget=metric(row, "validation_value_budget"),
            lambda_value=f(row["lambda"]),
        )
        for row in rows
    ]
    return choose_account_mixture_from_points(points, floor, budget_cap)


def choose_account_mixture_from_points(
    points: list[Point],
    floor: float,
    budget_cap: float,
) -> Mixture | None:
    if not points:
        return None
    choice = solve_account_lp(points, floor, budget_cap)
    if choice is None:
        return None
    components = tuple(
        (point.row, weight)
        for point, weight in zip(points, choice.weights)
        if weight > 1e-9
    )
    if not components:
        return None
    total = sum(weight for _row, weight in components)
    return Mixture(
        tuple((row, weight / total) for row, weight in components),
        solver_status=choice.solver_status,
        gap_tolerance=choice.gap_tolerance,
        secondary_tolerance=choice.secondary_tolerance,
    )


def solve_account_lp(
    points: list[Point],
    floor: float,
    budget_cap: float,
) -> SolverChoice | None:
    n = len(points)
    gap_tol = 1e-10
    secondary_tol = 1e-10
    validation_tol = 1e-8
    ones = [1.0] * n
    gaps = [point.signed_gap for point in points]
    captures = [point.capture for point in points]
    budgets = [point.budget for point in points]

    # First pass: minimize t subject to |gap| <= t, capture >= floor, budget <= beta.
    first_a: list[list[float]] = []
    first_b: list[float] = []
    first_a.append(gaps + [-1.0])
    first_b.append(0.0)
    first_a.append([-gap for gap in gaps] + [-1.0])
    first_b.append(0.0)
    first_a.append([-capture for capture in captures] + [0.0])
    first_b.append(-floor)
    first_a.append(budgets + [0.0])
    first_b.append(budget_cap)
    first_a.append(ones + [0.0])
    first_b.append(1.0)
    first_a.append([-1.0 for _ in points] + [0.0])
    first_b.append(-1.0)
    first = LinearProgram(first_a, first_b, [0.0] * n + [-1.0]).solve()
    if first is None:
        return None
    optimum_gap = max(0.0, first.value * -1.0)
    first_weights = first.solution[:n]
    if not solution_valid(
        first_weights,
        gaps,
        captures,
        budgets,
        floor,
        budget_cap,
        optimum_gap + gap_tol,
        validation_tol,
    ):
        return None

    # Second pass: maximize validation capture at the optimum gap tolerance.
    second_a, second_b = base_weight_constraints(
        gaps,
        captures,
        budgets,
        floor,
        budget_cap,
        optimum_gap + gap_tol,
    )
    second = LinearProgram(second_a, second_b, captures).solve()
    if second is None or not solution_valid(
        second.solution,
        gaps,
        captures,
        budgets,
        floor,
        budget_cap,
        optimum_gap + gap_tol,
        validation_tol,
    ):
        return SolverChoice(
            first_weights,
            "optimal_simplex_stdlib_primary_only",
            gap_tol,
            secondary_tol,
        )
    optimum_capture = second.value

    # Third pass: deterministic tie-break by minimizing validation spend.
    third_a, third_b = base_weight_constraints(
        gaps,
        captures,
        budgets,
        floor,
        budget_cap,
        optimum_gap + gap_tol,
    )
    third_a.append([-capture for capture in captures])
    third_b.append(-(optimum_capture - secondary_tol))
    third = LinearProgram(third_a, third_b, [-budget for budget in budgets]).solve()
    if third is not None and solution_valid(
        third.solution,
        gaps,
        captures,
        budgets,
        floor,
        budget_cap,
        optimum_gap + gap_tol,
        validation_tol,
        min_capture=optimum_capture - secondary_tol,
    ):
        return SolverChoice(
            third.solution,
            "optimal_simplex_stdlib_gap_capture_spend",
            gap_tol,
            secondary_tol,
        )
    return SolverChoice(
        second.solution,
        "optimal_simplex_stdlib_gap_capture_spend_fallback",
        gap_tol,
        secondary_tol,
    )


def solution_valid(
    weights: list[float],
    gaps: list[float],
    captures: list[float],
    budgets: list[float],
    floor: float,
    budget_cap: float,
    gap_cap: float,
    tolerance: float,
    *,
    min_capture: float | None = None,
) -> bool:
    if len(weights) != len(gaps):
        return False
    if any(weight < -tolerance for weight in weights):
        return False
    weight_sum = sum(weights)
    signed_gap = sum(weight * gap for weight, gap in zip(weights, gaps))
    capture = sum(weight * value for weight, value in zip(weights, captures))
    budget = sum(weight * value for weight, value in zip(weights, budgets))
    if abs(weight_sum - 1.0) > tolerance:
        return False
    if abs(signed_gap) > gap_cap + tolerance:
        return False
    if capture + tolerance < floor:
        return False
    if min_capture is not None and capture + tolerance < min_capture:
        return False
    if budget > budget_cap + tolerance:
        return False
    return True


def base_weight_constraints(
    gaps: list[float],
    captures: list[float],
    budgets: list[float],
    floor: float,
    budget_cap: float,
    gap_cap: float,
) -> tuple[list[list[float]], list[float]]:
    ones = [1.0] * len(gaps)
    a: list[list[float]] = [
        gaps,
        [-gap for gap in gaps],
        [-capture for capture in captures],
        budgets,
        ones,
        [-1.0 for _ in gaps],
    ]
    b = [gap_cap, gap_cap, -floor, budget_cap, 1.0, -1.0]
    return a, b


def baseline_row(account_rows: list[dict[str, str]]) -> dict[str, str] | None:
    for row in account_rows:
        if abs(f(row["lambda"])) <= EPS:
            return row
    return min(account_rows, key=lambda row: abs(f(row["lambda"]))) if account_rows else None


def component_weights_json(mixture: Mixture) -> str:
    payload = [
        {
            "lambda": f(row["lambda"]),
            "weight": weight,
            "validation_thinning_alpha": f(row.get("validation_thinning_alpha")),
            "validation_signed_gap": metric(row, "validation_signed_complaint_gap"),
            "validation_capture_fraction": metric(row, "validation_capture_fraction_vs_baseline"),
            "validation_value_budget": metric(row, "validation_value_budget"),
        }
        for row, weight in mixture.components
    ]
    return json.dumps(payload, separators=(",", ":"), sort_keys=True)


def thinning_alpha(row: dict[str, str], budget_cap: float) -> float:
    budget = metric(row, "validation_value_budget")
    if budget <= budget_cap + EPS:
        return 1.0
    if budget <= EPS:
        return 1.0
    return budget_cap / budget


def adjust_rows_to_budget(
    rows: list[dict[str, str]],
    budget_cap: float,
    baseline_validation_capture: float,
    baseline_test_capture: float,
) -> list[dict[str, str]]:
    adjusted: list[dict[str, str]] = []
    for row in rows:
        alpha = thinning_alpha(row, budget_cap)
        out = dict(row)
        out["validation_thinning_alpha"] = alpha
        for column in (
            "validation_capture",
            "test_capture",
            "validation_value_budget",
            "test_value_budget",
            "validation_signed_complaint_gap",
            "test_signed_complaint_gap",
            "validation_complaint_gap",
            "test_complaint_gap",
            "validation_complaint_burden",
            "test_complaint_burden",
        ):
            if column in out:
                out[column] = alpha * metric(row, column)
        validation_capture = metric(out, "validation_capture")
        test_capture = metric(out, "test_capture")
        out["validation_capture_fraction_vs_baseline"] = (
            validation_capture / baseline_validation_capture
            if baseline_validation_capture > EPS
            else float("nan")
        )
        out["test_capture_fraction_vs_baseline"] = (
            test_capture / baseline_test_capture
            if baseline_test_capture > EPS
            else float("nan")
        )
        adjusted.append(out)
    return adjusted


def adjusted_policy_rows(
    account_rows: list[dict[str, str]],
    group_rows: list[dict[str, str]],
    budget_cap: float,
) -> tuple[list[dict[str, str]], list[dict[str, str]], dict[str, str] | None]:
    base = baseline_row(account_rows)
    if base is None:
        return account_rows, group_rows, None
    base_alpha = thinning_alpha(base, budget_cap)
    baseline_validation_capture = base_alpha * metric(base, "validation_capture")
    baseline_test_capture = base_alpha * metric(base, "test_capture")
    adjusted_account = adjust_rows_to_budget(
        account_rows,
        budget_cap,
        baseline_validation_capture,
        baseline_test_capture,
    )
    adjusted_group = adjust_rows_to_budget(
        group_rows,
        budget_cap,
        baseline_validation_capture,
        baseline_test_capture,
    )
    return adjusted_account, adjusted_group, baseline_row(adjusted_account)


def detail_rows(
    frontier_inputs: list[Path],
    currency: str,
    floors: list[float],
) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    by_cell: dict[tuple[str, str, str, str, str], dict[str, list[dict[str, str]]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for path in frontier_inputs:
        source_artifact = artifact_name(path)
        for row in read_rows(path, currency):
            by_cell[key_for(row, source_artifact)][row["policy"]].append(row)

    output: list[dict[str, str]] = []
    eligibility_rows: list[dict[str, str]] = []
    for key, by_policy in sorted(by_cell.items()):
        source_artifact, dataset, seed, row_currency, target = key
        budget_cap = f(target)
        raw_account_rows = sorted(
            by_policy.get("account_penalty", []),
            key=lambda row: f(row["lambda"]),
        )
        raw_group_rows = sorted(
            by_policy.get("group_penalty", []),
            key=lambda row: f(row["lambda"]),
        )
        account_rows, group_rows, base = adjusted_policy_rows(
            raw_account_rows,
            raw_group_rows,
            budget_cap,
        )
        baseline_gap = abs(metric(base, "test_signed_complaint_gap")) if base is not None else float("nan")
        for floor in floors:
            status = "paired_eligible"
            reason = "ok"
            group: dict[str, str] | None = None
            account_mix: Mixture | None = None
            if base is None or not account_rows:
                status = "missing_input"
                reason = "missing_account_baseline"
            elif not group_rows:
                status = "missing_input"
                reason = "missing_group_bank"
            else:
                group = choose_single(group_rows, floor, budget_cap)
                if group is None:
                    status = "group_infeasible"
                    reason = "no_group_policy_meets_capture_floor_and_configured_beta"
                else:
                    account_mix = choose_account_mixture(account_rows, floor, budget_cap)
                    if account_mix is None:
                        status = "account_infeasible"
                        reason = "no_account_mixture_meets_capture_floor_and_configured_beta"

            row = {
                "source_artifact": source_artifact,
                "dataset": dataset,
                "source": source_name(dataset),
                "seed": seed,
                "currency": row_currency,
                "target": target,
                "capture_floor": fmt(floor),
                "eligibility_status": status,
                "eligibility_reason": reason,
                "budget_cap": fmt(budget_cap),
                "budget_protocol": "validation_frozen_randomized_thinning",
                "baseline_validation_thinning_alpha": (
                    fmt(metric(base, "validation_thinning_alpha"))
                    if base is not None
                    else ""
                ),
                "baseline_test_gap": fmt(baseline_gap),
                "group_lambda": "",
                "group_validation_thinning_alpha": "",
                "group_validation_signed_gap": "",
                "group_validation_gap": "",
                "group_test_signed_gap": "",
                "group_test_gap": "",
                "group_validation_capture_fraction": "",
                "group_test_capture_fraction": "",
                "group_validation_value_budget": "",
                "group_test_value_budget": "",
                "group_validation_over_nominal_target": "",
                "group_test_closure_raw": "",
                "account_left_lambda": "",
                "account_right_lambda": "",
                "account_left_weight": "",
                "account_support_size": "",
                "account_component_weights_json": "",
                "account_lp_solver_status": "infeasible" if status != "missing_input" else status,
                "account_lp_gap_tolerance": "",
                "account_lp_secondary_tolerance": "",
                "account_lp_capture_residual": "",
                "account_lp_budget_residual": "",
                "account_lp_validation_signed_gap": "",
                "account_lp_validation_gap": "",
                "account_lp_test_signed_gap": "",
                "account_lp_test_gap": "",
                "account_lp_validation_capture_fraction": "",
                "account_lp_test_capture_fraction": "",
                "account_lp_validation_value_budget": "",
                "account_lp_test_value_budget": "",
                "account_lp_validation_over_nominal_target": "",
                "account_lp_validation_budget_feasible": "0",
                "account_lp_test_closure_raw": "",
                "lift_vs_account_lp": "",
            }

            if group is not None:
                group_validation_signed_gap = metric(group, "validation_signed_complaint_gap")
                group_test_signed_gap = metric(group, "test_signed_complaint_gap")
                group_gap = abs(group_test_signed_gap)
                row.update(
                    {
                        "group_lambda": group["lambda"],
                        "group_validation_thinning_alpha": fmt(
                            metric(group, "validation_thinning_alpha")
                        ),
                        "group_validation_signed_gap": fmt(group_validation_signed_gap),
                        "group_validation_gap": fmt(abs(group_validation_signed_gap)),
                        "group_test_signed_gap": fmt(group_test_signed_gap),
                        "group_test_gap": fmt(group_gap),
                        "group_validation_capture_fraction": fmt(
                            metric(group, "validation_capture_fraction_vs_baseline")
                        ),
                        "group_test_capture_fraction": fmt(
                            metric(group, "test_capture_fraction_vs_baseline")
                        ),
                        "group_validation_value_budget": fmt(
                            metric(group, "validation_value_budget")
                        ),
                        "group_test_value_budget": fmt(metric(group, "test_value_budget")),
                        "group_validation_over_nominal_target": fmt(
                            metric(group, "validation_value_budget") - budget_cap
                        ),
                        "group_test_closure_raw": fmt(closure(group_gap, baseline_gap)),
                    }
                )

            if account_mix is not None:
                account_validation_signed_gap = mix_value(
                    account_mix,
                    "validation_signed_complaint_gap",
                )
                account_test_signed_gap = mix_value(
                    account_mix,
                    "test_signed_complaint_gap",
                )
                account_validation_capture = mix_value(
                    account_mix,
                    "validation_capture_fraction_vs_baseline",
                )
                account_validation_budget = mix_value(account_mix, "validation_value_budget")
                account_gap = abs(account_test_signed_gap)
                account_closure = closure(account_gap, baseline_gap)
                group_closure = f(row["group_test_closure_raw"])
                row.update(
                    {
                        "account_left_lambda": account_mix.left["lambda"],
                        "account_right_lambda": account_mix.right["lambda"],
                        "account_left_weight": fmt(account_mix.left_weight),
                        "account_support_size": str(len(account_mix.components)),
                        "account_component_weights_json": component_weights_json(account_mix),
                        "account_lp_solver_status": account_mix.solver_status,
                        "account_lp_gap_tolerance": fmt(account_mix.gap_tolerance),
                        "account_lp_secondary_tolerance": fmt(account_mix.secondary_tolerance),
                        "account_lp_capture_residual": fmt(account_validation_capture - floor),
                        "account_lp_budget_residual": fmt(budget_cap - account_validation_budget),
                        "account_lp_validation_signed_gap": fmt(account_validation_signed_gap),
                        "account_lp_validation_gap": fmt(abs(account_validation_signed_gap)),
                        "account_lp_test_signed_gap": fmt(account_test_signed_gap),
                        "account_lp_test_gap": fmt(account_gap),
                        "account_lp_validation_capture_fraction": fmt(account_validation_capture),
                        "account_lp_test_capture_fraction": fmt(
                            mix_value(account_mix, "test_capture_fraction_vs_baseline")
                        ),
                        "account_lp_validation_value_budget": fmt(account_validation_budget),
                        "account_lp_test_value_budget": fmt(
                            mix_value(account_mix, "test_value_budget")
                        ),
                        "account_lp_validation_over_nominal_target": fmt(
                            account_validation_budget - budget_cap
                        ),
                        "account_lp_validation_budget_feasible": (
                            "1" if account_validation_budget <= budget_cap + 1e-10 else "0"
                        ),
                        "account_lp_test_closure_raw": fmt(account_closure),
                        "lift_vs_account_lp": fmt(group_closure - account_closure),
                    }
                )

            output.append(row)
            eligibility_rows.append(
                {
                    "source_artifact": source_artifact,
                    "dataset": dataset,
                    "source": source_name(dataset),
                    "seed": seed,
                    "currency": row_currency,
                    "target": target,
                    "capture_floor": fmt(floor),
                    "status": status,
                    "reason": reason,
                    "configured_beta": fmt(budget_cap),
                    "group_candidate_count": str(len(raw_group_rows)),
                    "account_candidate_count": str(len(raw_account_rows)),
                }
            )
    return output, eligibility_rows


def summarize(details: list[dict[str, str]]) -> list[dict[str, str]]:
    by_group: dict[tuple[str, ...], list[dict[str, str]]] = defaultdict(list)
    for row in details:
        key = (
            row["source_artifact"],
            row["dataset"],
            row["source"],
            row["currency"],
            row["target"],
            row["capture_floor"],
        )
        by_group[key].append(row)
    output: list[dict[str, str]] = []
    for key, rows in sorted(by_group.items()):
        source_artifact, dataset, source, currency, target, floor = key
        eligible = [row for row in rows if row["eligibility_status"] == "paired_eligible"]
        lifts = [f(row["lift_vs_account_lp"]) for row in eligible]
        missing = [row for row in rows if row["eligibility_status"] == "missing_input"]
        group_infeasible = [
            row for row in rows if row["eligibility_status"] == "group_infeasible"
        ]
        account_infeasible = [
            row for row in rows if row["eligibility_status"] == "account_infeasible"
        ]
        metric_rows = eligible or []
        output.append(
            {
                "source_artifact": source_artifact,
                "dataset": dataset,
                "source": source,
                "currency": currency,
                "target": target,
                "capture_floor": floor,
                "intended_cells": str(len(rows)),
                "paired_eligible_cells": str(len(eligible)),
                "group_infeasible_cells": str(len(group_infeasible)),
                "account_infeasible_cells": str(len(account_infeasible)),
                "missing_input_cells": str(len(missing)),
                "n_seeds": str(len({row["seed"] for row in eligible})),
                "mean_group_closure": fmt(
                    statistics.fmean(f(row["group_test_closure_raw"]) for row in metric_rows)
                )
                if metric_rows
                else "",
                "mean_account_lp_closure": fmt(
                    statistics.fmean(f(row["account_lp_test_closure_raw"]) for row in metric_rows)
                )
                if metric_rows
                else "",
                "median_lift_vs_account_lp": fmt(statistics.median(lifts)) if lifts else "",
                "mean_lift_vs_account_lp": fmt(statistics.fmean(lifts)) if lifts else "",
                "positive_seeds": str(sum(1 for value in lifts if value > EPS)),
                "negative_seeds": str(sum(1 for value in lifts if value < -EPS)),
                "tied_seeds": str(sum(1 for value in lifts if abs(value) <= EPS)),
                "mean_group_test_gap": fmt(
                    statistics.fmean(f(row["group_test_gap"]) for row in metric_rows)
                )
                if metric_rows
                else "",
                "mean_account_lp_test_gap": fmt(
                    statistics.fmean(f(row["account_lp_test_gap"]) for row in metric_rows)
                )
                if metric_rows
                else "",
                "mean_group_test_capture_fraction": fmt(
                    statistics.fmean(f(row["group_test_capture_fraction"]) for row in metric_rows)
                )
                if metric_rows
                else "",
                "mean_account_lp_test_capture_fraction": fmt(
                    statistics.fmean(
                        f(row["account_lp_test_capture_fraction"]) for row in metric_rows
                    )
                )
                if metric_rows
                else "",
            }
        )
    return output


def write_csv(path: Path, rows: list[dict[str, str]], fields: tuple[str, ...]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    floors = parse_floors(args.capture_floors)
    details, eligibility = detail_rows(args.frontier_input, args.currency, floors)
    summaries = summarize(details)
    write_csv(args.detail_output, details, DETAIL_FIELDS)
    write_csv(args.summary_output, summaries, SUMMARY_FIELDS)
    write_csv(args.eligibility_output, eligibility, ELIGIBILITY_FIELDS)
    print(
        f"Wrote {len(details)} matched-account detail rows and "
        f"{len(summaries)} summary rows; "
        f"{sum(1 for row in details if row['eligibility_status'] == 'paired_eligible')} "
        "paired-eligible cells"
    )


if __name__ == "__main__":
    main()
