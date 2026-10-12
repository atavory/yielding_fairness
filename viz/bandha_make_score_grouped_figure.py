"""Render the single-column score comparison from frozen paired summaries."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


VALIDATION_FROZEN = {"KuaiRand", "Australian MTPL", "Fannie Mae", "Freddie Mac"}


def read_metric(path: Path) -> dict[str, dict[str, str]]:
    with path.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    result = {row["dataset"]: row for row in rows}
    if len(rows) != 10 or len(result) != 10:
        raise ValueError("Expected ten distinct score datasets")
    for row in rows:
        if row["selected_method"] != "W-BCE(raw)" or int(row["n_splits"]) != 10:
            raise ValueError("The plot requires fixed raw W-BCE and ten paired splits")
        values = [float(row[key]) for key in ("delta_mean", "delta_ci95")]
        if not all(math.isfinite(value) for value in values) or values[1] < 0:
            raise ValueError("Invalid paired mean or confidence interval")
    return result


def format_axes(ax, names, y) -> None:
    labels = [
        name + (" [V]" if name in VALIDATION_FROZEN else " [T]") for name in names
    ]
    ax.set_yticks(y, labels, fontsize=6.8)
    ax.invert_yaxis()
    ax.set_xlim(-0.05, 0.34)
    ax.set_xticks([-0.05, 0, 0.1, 0.2, 0.3], ["−.05", "0", ".10", ".20", ".30"])
    ax.axvline(0, color="#333333", linewidth=0.6)
    ax.set_xlabel("Raw W-BCE − BCE (points)", labelpad=3)
    ax.grid(axis="x", color="#dddddd", linewidth=0.4)
    ax.set_axisbelow(True)
    ax.tick_params(axis="both", length=0, pad=3)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.legend(
        loc="lower center",
        bbox_to_anchor=(0.36, 1.01),
        ncol=2,
        frameon=False,
        handlelength=1.2,
        columnspacing=1.0,
    )


def draw(gini, capture, output: Path) -> list[dict]:
    plt.switch_backend("Agg")
    if set(gini) != set(capture):
        raise ValueError("Both metrics must describe the same datasets")
    names = sorted(gini, key=lambda name: float(gini[name]["delta_mean"]), reverse=True)
    plt.rcParams.update({"font.size": 7, "pdf.fonttype": 42, "ps.fonttype": 42})
    fig, ax = plt.subplots(figsize=(3.33, 3.0))
    y = np.arange(len(names))
    plotted = []
    for offset, metric, label, color in (
        (-0.17, gini, "Yield Gini", "#4a4d91"),
        (0.17, capture, "Loss capture", "#b66c46"),
    ):
        means = [float(metric[name]["delta_mean"]) for name in names]
        intervals = [float(metric[name]["delta_ci95"]) for name in names]
        ax.barh(y + offset, means, height=0.28, color=color, label=label)
        ax.errorbar(
            means,
            y + offset,
            xerr=intervals,
            fmt="none",
            ecolor="#333333",
            elinewidth=0.55,
            capsize=1.8,
            capthick=0.55,
        )
        plotted.extend(
            {"dataset": name, "metric": label, "mean": mean, "ci95": interval}
            for name, mean, interval in zip(names, means, intervals)
        )
    format_axes(ax, names, y)
    fig.subplots_adjust(left=0.35, right=0.98, bottom=0.13, top=0.91)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, metadata={"CreationDate": None, "ModDate": None})
    fig.savefig(output.with_suffix(".png"), dpi=240)
    plt.close(fig)
    return plotted


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--base-dir", type=Path, default=Path(__file__).resolve().parents[1]
    )
    args = parser.parse_args()
    paths = [
        args.base_dir / f"results/csvs/kdd_weighting_{metric}_wins.csv"
        for metric in ("gini", "capture")
    ]
    output = args.base_dir / "results/figures/bandha_score_weighting_grouped.pdf"
    plotted = draw(*(read_metric(path) for path in paths), output)
    provenance = {
        "generator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "inputs": {
            str(path.relative_to(args.base_dir)): hashlib.sha256(
                path.read_bytes()
            ).hexdigest()
            for path in paths
        },
        "output_sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
        "comparison": "Fixed raw W-BCE minus BCE; 95% paired Student t intervals, ten splits",
        "dataset_order": "Descending yield-Gini mean difference; both metrics share this order",
        "capture_protocol": {
            name: "V" if name in VALIDATION_FROZEN else "T"
            for name in read_metric(paths[0])
        },
        "plotted_points": plotted,
    }
    report = args.base_dir / "results/bandha_score_grouped_figure_provenance.json"
    report.write_text(json.dumps(provenance, indent=2) + "\n")
    print(f"Generated {output}: twenty unchanged paired metric summaries")


if __name__ == "__main__":
    main()
