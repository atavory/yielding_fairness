"""Validation-normalized, extended complaint-price bank from frozen candidates.

This post-review sensitivity preserves every original physical policy, extends
both families using validation only, freezes selection and LP weights before
reading test rows, and reports every recorded applicable binary-group source.
Historical generators and outputs are left intact.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import itertools
import json
import math
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import bandha_cached_group_awareness as legacy
import numpy as np
from scipy.optimize import linprog
from scipy.stats import ttest_1samp

FLOORS = (0.90, 0.95, 0.98, 1.0)
CURRENCIES = ("value", "count", "loss")
SEVEN = {
    "Australian MTPL",
    "Belgian MTPL",
    "Bondora",
    "EU health insurance",
    "French portfolio",
    "Norwegian auto",
    "Swedish motorcycle",
}
DATASETS = set(legacy.DEFAULT_DATASETS) | {
    "euhealth_gender",
    "cms_gender",
    "cms_race",
    "uci_credit_sex",
    "uci_credit_age",
    "german_credit_sex",
    "german_credit_age",
    "beMTPL97",
    "norauto",
    "swmotorcycle",
    *(f"freMPL{i}" for i in range(1, 10)),
    *(f"freMPL{i}[Gender]" for i in range(1, 10)),
    "norauto[Male]",
    "swmotorcycle[Gender]",
}


def source_name(dataset):
    dataset = dataset.split("[")[0]
    if dataset.startswith("freMPL"):
        return "French portfolio"
    return {
        "beMTPL97": "Belgian MTPL",
        "norauto": "Norwegian auto",
        "swmotorcycle": "Swedish motorcycle",
    }.get(dataset, legacy.source_name(dataset))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_freeze(path):
    if path.exists():
        return json.loads(path.read_text())
    with gzip.open(path.with_suffix(path.suffix + ".gz"), "rt") as handle:
        return json.load(handle)


def write_csv(path, rows):
    if not rows:
        return
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def read_file(path, split):
    """Read only the requested split; validation passes never retain test rows."""
    grouped = defaultdict(list)
    op = gzip.open if path.suffix == ".gz" else open
    with op(path, "rt", newline="") as handle:
        reader = csv.DictReader(handle)
        compact = "d" in reader.fieldnames
        for row in reader:
            if compact:
                dataset = legacy.decode("dataset", row["d"])
                row_split = legacy.decode("split", row["p"])
                seed, target = row["s"], float(row["t"])
                group = legacy.decode("group", row["g"])
                a, b, q, ah, bh = (float(row[n]) for n in ("A", "B", "q", "ah", "bh"))
                objection = not (float(row["h"]) > 0.5 and float(row["nh"]) > 0.5)
                fixed = [float(row[n]) > 0.5 for n in ("gv", "egv", "egd", "egp")]
            else:
                dataset, seed, target = (
                    row["dataset"],
                    row["seed"],
                    float(row["target"]),
                )
                row_split = row.get("sample_split", row.get("split", ""))
                group = row["group"]
                a, b, q = (float(row[n]) for n in ("value", "loss", "score"))
                ah = float(row.get("estimated_value", row.get("a_hat", "0")))
                bh = float(row.get("estimated_loss", row.get("b_hat", "0")))
                objection = not (
                    float(
                        row.get(
                            "selected_criterion_harm", row.get("selected_harm", "0")
                        )
                    )
                    > 0.5
                    and float(
                        row.get(
                            "selected_criterion_net_harm",
                            row.get("selected_net_harm", "0"),
                        )
                    )
                    > 0.5
                )
                fixed = [
                    float(row["selected_" + name]) > 0.5
                    for name in legacy.FIXED_POLICIES
                ]
            if (
                dataset not in DATASETS
                or row_split.lower() != split
                or not math.isclose(target, 0.1)
            ):
                continue
            grouped[(dataset, seed)].append(
                (
                    group,
                    max(a, 0.0),
                    max(b, 0.0),
                    math.exp(np.clip(q, -50.0, 50.0)),
                    max(ah, 0.0),
                    max(bh, 0.0),
                    objection,
                    *fixed,
                )
            )
    return grouped


def arrays(rows, names=None):
    groups = np.array([row[0] for row in rows])
    names = sorted(set(groups)) if names is None else names
    data = np.array([row[1:] for row in rows], dtype=np.float64)
    return {
        "a": data[:, 0],
        "b": data[:, 1],
        "ratio": data[:, 2],
        "ah": data[:, 3],
        "bh": data[:, 4],
        "objection": data[:, 5].astype(bool),
        "fixed": data[:, 6:].astype(bool),
        "groups": groups,
        "names": names,
        "gid": np.array([names.index(g) for g in groups]),
    }


def unit(data, currency, predicted=False):
    if currency == "count":
        return np.ones(len(data["a"]))
    return data[
        ("ah" if currency == "value" else "bh")
        if predicted
        else ("a" if currency == "value" else "b")
    ]


def scores(data, family, price, currency, rates, high, raw=False):
    loss_side = data["ratio"] * data["ah"]
    potential = unit(data, currency, True)
    if family == "account":
        coefficient = price if raw else price / sum(rates)
        loss_side = loss_side - coefficient * potential * data["objection"]
    elif family == "group":
        coefficient = np.where(
            data["gid"] == high, price / rates[high], -price / rates[1 - high]
        )
        loss_side = loss_side - coefficient * potential * data["objection"]
    result = np.maximum(data["ratio"], 0.0).copy()
    np.divide(
        np.maximum(loss_side, 0.0), data["ah"], out=result, where=data["ah"] > 0.0
    )
    return result


def threshold_mask(data, score):
    total = data["a"].sum()
    if total <= 0:
        return math.inf, np.zeros(len(score), dtype=bool)
    order = np.argsort(-score, kind="stable")
    positive = order[score[order] > 0.0]
    if not len(positive):
        return 0.0, np.zeros(len(score), dtype=bool)
    cumulative = np.cumsum(data["a"][positive])
    j = min(int(np.searchsorted(cumulative, 0.1 * total)), len(positive) - 1)
    threshold = float(score[positive[j]])
    return threshold, (score >= threshold) & (score > 0.0)


def measure(data, mask, currency, high, alpha=None):
    amount = unit(data, currency)
    selected_value = float(data["a"][mask].sum())
    spend = selected_value / data["a"].sum()
    if alpha is None:
        alpha = 0.1 / spend if spend > 0.1 + 1e-12 else 1.0
    masses = []
    for g in range(2):
        group = data["gid"] == g
        masses.append(
            {
                "group": data["names"][g],
                "total_mass": float(amount[group].sum()),
                "potential_mass": float(amount[group & data["objection"]].sum()),
                "selected_mass": float(amount[group & mask].sum()) * alpha,
                "complaint_mass": float(amount[group & mask & data["objection"]].sum())
                * alpha,
            }
        )
    burdens = [
        m["complaint_mass"] / m["total_mass"] if m["total_mass"] > 0 else math.nan
        for m in masses
    ]
    return {
        "capture": float(data["b"][mask].sum()) / data["b"].sum() * alpha,
        "budget": float(spend * alpha),
        "gap": burdens[high] - burdens[1 - high],
        "alpha": float(alpha),
        "masses": masses,
    }


def choose(bank, family, baseline, floor):
    candidates = [
        i
        for i, point in enumerate(bank)
        if point["family"] == family
        and point["validation"]["capture"] >= floor * baseline["capture"]
        and point["validation"]["budget"] <= 0.1 + 1e-10
        and math.isfinite(point["validation"]["gap"])
    ]
    return (
        min(
            candidates,
            key=lambda i: (
                abs(bank[i]["validation"]["gap"]),
                -bank[i]["validation"]["capture"],
                bank[i]["lambda"],
                bank[i]["family"],
            ),
        )
        if candidates
        else None
    )


def lp_select(bank, baseline, floor):
    ids = [i for i, point in enumerate(bank) if point["family"] == "account"]
    gaps = np.array([bank[i]["validation"]["gap"] for i in ids])
    captures = np.array(
        [bank[i]["validation"]["capture"] / baseline["capture"] for i in ids]
    )
    budgets = np.array([bank[i]["validation"]["budget"] for i in ids])
    n = len(ids)
    options = {"dual_feasibility_tolerance": 1e-9, "primal_feasibility_tolerance": 1e-9}
    first = linprog(
        np.r_[np.zeros(n), 1.0],
        A_ub=np.array(
            [
                np.r_[gaps, -1.0],
                np.r_[-gaps, -1.0],
                np.r_[-captures, 0.0],
                np.r_[budgets, 0.0],
            ]
        ),
        b_ub=[0.0, 0.0, -floor, 0.1],
        A_eq=[np.r_[np.ones(n), 0.0]],
        b_eq=[1.0],
        bounds=(0.0, None),
        method="highs",
        options=options,
    )
    if not first.success:
        return None
    gap_cap = first.fun + 1e-10
    constraints = np.array([gaps, -gaps, -captures, budgets])
    rhs = [gap_cap, gap_cap, -floor, 0.1]
    second = linprog(
        -captures,
        A_ub=constraints,
        b_ub=rhs,
        A_eq=[np.ones(n)],
        b_eq=[1.0],
        bounds=(0.0, None),
        method="highs",
        options=options,
    )
    if not second.success:
        raise RuntimeError("LP capture tie-break failed")
    third = linprog(
        budgets,
        A_ub=np.vstack([constraints, -captures]),
        b_ub=[*rhs, second.fun + 1e-10],
        A_eq=[np.ones(n)],
        b_eq=[1.0],
        bounds=(0.0, None),
        method="highs",
        options=options,
    )
    result = third if third.success else second
    weights = result.x
    if not (
        abs(weights.sum() - 1.0) < 1e-7
        and weights.min() >= -1e-9
        and weights @ captures >= floor - 1e-7
        and weights @ budgets <= 0.1 + 1e-7
        and abs(weights @ gaps) <= gap_cap + 1e-7
    ):
        raise RuntimeError(
            f"LP original constraints failed: sum={weights.sum()}, "
            f"min={weights.min()}, capture={weights @ captures}, floor={floor}, "
            f"budget={weights @ budgets}, gap={weights @ gaps}, cap={gap_cap}"
        )
    return {
        "components": [[ids[j], float(w)] for j, w in enumerate(weights) if w > 0.0],
        "primary_optimum_gap": float(first.fun),
        "gap_tolerance": 1e-10,
        "validation_gap": float(weights @ gaps),
        "validation_capture_fraction": float(weights @ captures),
        "validation_budget": float(weights @ budgets),
        "solver": "scipy_highs_gap_capture_spend",
    }


def freeze_case(data, currency):
    rates = [
        float(unit(data, currency, True)[data["gid"] == g].sum()) / len(data["a"])
        for g in range(2)
    ]
    if min(rates) <= 0.0 or any(
        unit(data, currency)[data["gid"] == g].sum() <= 0.0 for g in range(2)
    ):
        return {"status": "invalid_group_denominator", "currency": currency}
    baseline_mask = threshold_mask(
        data, scores(data, "baseline", 0.0, currency, rates, 0)
    )[1]
    provisional = measure(data, baseline_mask, currency, 0)
    high = 0 if provisional["gap"] >= 0.0 else 1
    baseline = measure(data, baseline_mask, currency, high)
    original = list(legacy.DEFAULT_LAMBDAS)
    axis = sorted(set(original + [sum(rates) * x for x in original]))
    bank = []

    def add(family, price):
        threshold, mask = threshold_mask(
            data, scores(data, family, price, currency, rates, high)
        )
        bank.append(
            {
                "family": family,
                "lambda": float(price),
                "threshold": threshold,
                "validation": measure(data, mask, currency, high),
            }
        )

    for price in axis:
        for family in ("account", "group"):
            add(family, price)
    identity_failures = 0
    for price in original:
        for family in ("account", "group"):
            normalized = sum(rates) * price if family == "account" else price
            old_mask = threshold_mask(
                data, scores(data, family, price, currency, rates, high, raw=True)
            )[1]
            new_mask = threshold_mask(
                data, scores(data, family, normalized, currency, rates, high)
            )[1]
            identity_failures += int(not np.array_equal(old_mask, new_mask))
    history, previous, stable = [], None, 0
    for _extension in range(25):
        chosen = {
            f"{floor}:{family}": choose(bank, family, baseline, floor)
            for floor in FLOORS
            for family in ("account", "group")
        }
        fingerprint = tuple(chosen.values())
        edge = any(
            i is not None and bank[i]["lambda"] == axis[-1] for i in chosen.values()
        )
        stable = stable + 1 if not edge and fingerprint == previous else 0
        history.append(
            {
                "maximum_lambda": axis[-1],
                "edge_selected": edge,
                "stable_extensions": stable,
                "axis_size": len(axis),
            }
        )
        if not edge and stable >= 2:
            break
        previous = fingerprint
        axis.append(axis[-1] * 2.0)
        for family in ("account", "group"):
            add(family, axis[-1])
    else:
        raise RuntimeError("Validation coverage remained open after 24 extensions")
    fixed = {}
    for j, family in enumerate(legacy.FIXED_POLICIES):
        fixed[family] = measure(data, data["fixed"][:, j], currency, high)
    return {
        "status": "frozen",
        "currency": currency,
        "rates": rates,
        "high": high,
        "groups": data["names"],
        "baseline": baseline,
        "bank": bank,
        "fixed": fixed,
        "axis": axis,
        "coverage": history,
        "identity_failures": identity_failures,
        "selected": chosen,
        "lp": {str(floor): lp_select(bank, baseline, floor) for floor in FLOORS},
    }


def evaluate_case(data, case, dataset, seed):
    currency, high, rates = case["currency"], case["high"], case["rates"]
    baseline_point = next(
        p for p in case["bank"] if p["family"] == "account" and p["lambda"] == 0.0
    )
    base_score = scores(data, "account", 0.0, currency, rates, high)
    base = measure(
        data,
        (base_score >= baseline_point["threshold"]) & (base_score > 0.0),
        currency,
        high,
        case["baseline"]["alpha"],
    )
    tested = {}

    def evaluate(index):
        if index not in tested:
            point = case["bank"][index]
            score = scores(
                data, point["family"], point["lambda"], currency, rates, high
            )
            tested[index] = measure(
                data,
                (score >= point["threshold"]) & (score > 0.0),
                currency,
                high,
                point["validation"]["alpha"],
            )
        return tested[index]

    def closure(gap):
        return (
            1.0 - abs(gap) / abs(base["gap"])
            if base["gap"] != 0.0
            else float(gap == 0.0)
        )

    output = []
    for floor in FLOORS:
        row = {
            "dataset": dataset,
            "source": source_name(dataset),
            "seed": seed,
            "currency": currency,
            "capture_floor": floor,
            "baseline_test_gap": abs(base["gap"]),
            "baseline_test_capture": base["capture"],
        }
        for family in ("account", "group"):
            i = case["selected"][f"{floor}:{family}"]
            if i is None:
                continue
            point, metrics = case["bank"][i], evaluate(i)
            row.update(
                {
                    family + "_lambda": point["lambda"],
                    family + "_test_gap": abs(metrics["gap"]),
                    family + "_closure": closure(metrics["gap"]),
                    family + "_test_capture_fraction": metrics["capture"]
                    / base["capture"]
                    if base["capture"] > 0.0
                    else math.nan,
                    family + "_test_value_budget": metrics["budget"],
                    family + "_validation_capture_fraction": point["validation"][
                        "capture"
                    ]
                    / case["baseline"]["capture"],
                    family + "_validation_value_budget": point["validation"]["budget"],
                }
            )
        mixture = case["lp"][str(floor)]
        if mixture is not None:
            gap = sum(
                weight * evaluate(i)["gap"] for i, weight in mixture["components"]
            )
            capture = sum(
                weight * evaluate(i)["capture"] for i, weight in mixture["components"]
            )
            budget = sum(
                weight * evaluate(i)["budget"] for i, weight in mixture["components"]
            )
            row.update(
                {
                    "lp_test_gap": abs(gap),
                    "lp_closure": closure(gap),
                    "lp_test_capture_fraction": capture / base["capture"]
                    if base["capture"] > 0.0
                    else math.nan,
                    "lp_test_value_budget": budget,
                }
            )
        for family in legacy.FIXED_POLICIES:
            metrics = measure(
                data,
                data["fixed"][:, legacy.FIXED_POLICIES.index(family)],
                currency,
                high,
                case["fixed"][family]["alpha"],
            )
            row[family + "_closure"] = closure(metrics["gap"])
        row["paired_eligible"] = all(
            f + "_closure" in row for f in ("account", "group")
        )
        row["lift"] = row.get("group_closure", math.nan) - row.get(
            "account_closure", math.nan
        )
        row["lift_vs_lp"] = row.get("group_closure", math.nan) - row.get(
            "lp_closure", math.nan
        )
        output.append(row)
    return output, tested, base


def summaries(rows):
    output = []
    for currency, floor, scope, comparator in itertools.product(
        CURRENCIES, FLOORS, ("seven", "all"), ("lift", "lift_vs_lp")
    ):
        grouped = defaultdict(list)
        for row in rows:
            if (
                row["currency"] == currency
                and row["capture_floor"] == floor
                and (scope == "all" or row["source"] in SEVEN)
                and row["paired_eligible"]
                and math.isfinite(row[comparator])
            ):
                grouped[row["source"]].append(row[comparator])
        for aggregation in ("median", "mean"):
            aggregate = np.median if aggregation == "median" else np.mean
            values = np.array([aggregate(grouped[s]) for s in sorted(grouped)])
            if not len(values):
                continue
            flips = (
                np.array(list(itertools.product((-1.0, 1.0), repeat=len(values))))
                @ values
                / len(values)
            )
            output.append(
                {
                    "currency": currency,
                    "capture_floor": floor,
                    "scope": scope,
                    "comparator": comparator,
                    "aggregation": aggregation,
                    "n_cells": sum(map(len, grouped.values())),
                    "n_sources": len(values),
                    "cell_median": float(
                        np.median([v for vs in grouped.values() for v in vs])
                    ),
                    "cell_mean": float(
                        np.mean([v for vs in grouped.values() for v in vs])
                    ),
                    "source_mean": float(values.mean()),
                    "t_two_sided_p": float(ttest_1samp(values, 0.0).pvalue),
                    "sign_flip_one_sided_p": float(
                        np.mean(flips >= values.mean() - 1e-12)
                    ),
                    "source_values": dict(zip(sorted(grouped), values.tolist())),
                }
            )
    return output


def freeze_file(arguments):
    path, root = arguments
    cases = []
    grouped = read_file(path, "validation")
    for (dataset, seed), raw in sorted(grouped.items()):
        data = arrays(raw)
        if len(data["names"]) != 2:
            cases.append(
                {
                    "status": "not_binary",
                    "dataset": dataset,
                    "seed": seed,
                    "input": str(path.relative_to(root)),
                }
            )
            continue
        for currency in CURRENCIES:
            case = freeze_case(data, currency)
            case.update(
                {
                    "dataset": dataset,
                    "seed": seed,
                    "input": str(path.relative_to(root)),
                }
            )
            cases.append(case)
        print("frozen", dataset, seed, flush=True)
    del grouped
    return cases


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--base-dir", type=Path, default=Path(__file__).resolve().parents[1]
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument("--stage", choices=("freeze", "test", "all"), default="all")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume validation-only completed input files",
    )
    args = parser.parse_args()
    root = args.base_dir
    output = args.output or root / "results/bandha_normalized_prices"
    output.mkdir(parents=True, exist_ok=True)
    paths = sorted((root / "frozen_candidate_inputs").glob("*.csv.gz"))
    paths += sorted(
        (root / "results/bandha_public_auto_repro").glob("*_candidates.csv.gz")
    )
    freeze_path = output / "validation_freeze.json"
    if args.stage in ("freeze", "all"):
        frozen = {
            "protocol": "post_review_normalized_preserved_extended_validation_only",
            "inputs": [
                {"path": str(p.relative_to(root)), "sha256": sha(p)} for p in paths
            ],
            "numpy": np.__version__,
            "cases": [],
        }
        completed = set()
        if args.resume and freeze_path.exists():
            previous = json.loads(freeze_path.read_text())
            assert previous["inputs"] == frozen["inputs"]
            frozen["cases"] = previous["cases"]
            completed = {case["input"] for case in previous["cases"]}
        remaining = [
            path for path in paths if str(path.relative_to(root)) not in completed
        ]
        with ProcessPoolExecutor(max_workers=4) as executor:
            for cases in executor.map(
                freeze_file, [(path, root) for path in remaining]
            ):
                frozen["cases"].extend(cases)
        frozen["cases"].sort(
            key=lambda case: (
                case["input"],
                case["dataset"],
                case["seed"],
                case.get("currency", ""),
            )
        )
        freeze_path.write_text(json.dumps(frozen, allow_nan=True))
        print("VALIDATION FREEZE", sha(freeze_path), flush=True)
    if args.stage in ("test", "all"):
        frozen = load_freeze(freeze_path)
        for record in frozen["inputs"]:
            assert sha(root / record["path"]) == record["sha256"]
        rows, mass_rows = [], []
        for path in paths:
            cases = [
                c
                for c in frozen["cases"]
                if c["input"] == str(path.relative_to(root)) and c["status"] == "frozen"
            ]
            if not cases:
                continue
            grouped = read_file(path, "test")
            for case in cases:
                data = arrays(grouped[(case["dataset"], case["seed"])], case["groups"])
                result, tested, _baseline = evaluate_case(
                    data, case, case["dataset"], case["seed"]
                )
                rows.extend(result)
                for i, metrics in tested.items():
                    for split, measure_result in (
                        ("validation", case["bank"][i]["validation"]),
                        ("test", metrics),
                    ):
                        for mass in measure_result["masses"]:
                            mass_rows.append(
                                {
                                    "dataset": case["dataset"],
                                    "seed": case["seed"],
                                    "currency": case["currency"],
                                    "bank_index": i,
                                    "family": case["bank"][i]["family"],
                                    "lambda": case["bank"][i]["lambda"],
                                    "split": split,
                                    **mass,
                                }
                            )
            print("tested", path.name, flush=True)
        fields = sorted(set().union(*(set(row) for row in rows)))
        rows = [{key: row.get(key, "") for key in fields} for row in rows]
        write_csv(output / "heldout_comparison.csv", rows)
        write_csv(output / "selected_component_group_masses.csv", mass_rows)
        summary = summaries(rows)
        (output / "source_sensitivity.json").write_text(json.dumps(summary, indent=2))
        print(
            json.dumps(
                [
                    r
                    for r in summary
                    if r["currency"] == "value" and r["capture_floor"] == 0.95
                ],
                indent=2,
            )
        )


if __name__ == "__main__":
    main()
