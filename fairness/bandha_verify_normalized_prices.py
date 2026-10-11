"""Independently check normalized price coverage and LP optima with stdlib simplex."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from pathlib import Path

from bandha_matched_account_frontier_lp import LinearProgram


def verify_case(case):
    failures = []
    if case["status"] != "frozen":
        return failures, 0
    bank, baseline = case["bank"], case["baseline"]
    if case["identity_failures"]:
        failures.append("original_policy_identity")
    final = case["coverage"][-1]
    if final["edge_selected"] or final["stable_extensions"] < 2:
        failures.append("coverage")
    account = [point for point in bank if point["family"] == "account"]
    gap = [point["validation"]["gap"] for point in account]
    capture = [
        point["validation"]["capture"] / baseline["capture"] for point in account
    ]
    budget = [point["validation"]["budget"] for point in account]
    for point in bank:
        if point["lambda"] == 0 and any(
            abs(point["validation"][key] - baseline[key]) > 1e-10
            for key in ("capture", "budget", "gap", "alpha")
        ):
            failures.append("zero_price")
    checked = 0
    for floor, mixture in case["lp"].items():
        matrix = [
            gap + [-1.0],
            [-value for value in gap] + [-1.0],
            [-value for value in capture] + [0.0],
            budget + [0.0],
            [1.0] * len(account) + [0.0],
            [-1.0] * len(account) + [0.0],
        ]
        result = LinearProgram(
            matrix,
            [0.0, 0.0, -float(floor), 0.1, 1.0, -1.0],
            [0.0] * len(account) + [-1.0],
        ).solve()
        checked += 1
        if mixture is None or result is None:
            if (mixture is None) != (result is None):
                failures.append("LP_eligibility:" + floor)
            continue
        optimum = max(0.0, -result.value)
        if abs(optimum - mixture["primary_optimum_gap"]) > 1e-7:
            failures.append("LP_optimum:" + floor)
        components = mixture["components"]
        total = sum(weight for _, weight in components)
        weighted = {
            key: sum(bank[i]["validation"][key] * weight for i, weight in components)
            for key in ("capture", "budget", "gap")
        }
        if not (
            abs(total - 1.0) <= 1e-7
            and all(weight >= 0.0 for _, weight in components)
            and weighted["capture"] / baseline["capture"] >= float(floor) - 1e-7
            and weighted["budget"] <= 0.1 + 1e-7
            and abs(weighted["gap"]) <= optimum + 1.01e-7
        ):
            failures.append("LP_primal:" + floor)
        for family in ("account", "group"):
            candidates = [
                (i, point)
                for i, point in enumerate(bank)
                if point["family"] == family
                and point["validation"]["capture"] >= float(floor) * baseline["capture"]
                and point["validation"]["budget"] <= 0.1 + 1e-10
            ]
            chosen = (
                min(
                    candidates,
                    key=lambda item: (
                        abs(item[1]["validation"]["gap"]),
                        -item[1]["validation"]["capture"],
                        item[1]["lambda"],
                        item[1]["family"],
                    ),
                )[0]
                if candidates
                else None
            )
            if chosen != case["selected"][f"{float(floor)}:{family}"]:
                failures.append("validation_selection:" + floor + ":" + family)
    return failures, checked


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--freeze", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    contents = (
        gzip.decompress(args.freeze.read_bytes())
        if args.freeze.suffix == ".gz"
        else args.freeze.read_bytes()
    )
    frozen = json.loads(contents)
    report = {
        "freeze_sha256": hashlib.sha256(contents).hexdigest(),
        "independent_solver": "existing_stdlib_two_phase_simplex",
        "cases": 0,
        "lp_cells": 0,
        "failures": [],
    }
    for case in frozen["cases"]:
        failures, checked = verify_case(case)
        report["cases"] += 1
        report["lp_cells"] += checked
        if failures:
            report["failures"].append(
                {
                    "dataset": case["dataset"],
                    "seed": case["seed"],
                    "currency": case.get("currency"),
                    "failures": failures,
                }
            )
        if report["cases"] % 60 == 0:
            print("verified", report["cases"], flush=True)
    args.output.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    raise SystemExit(bool(report["failures"]))


if __name__ == "__main__":
    main()
