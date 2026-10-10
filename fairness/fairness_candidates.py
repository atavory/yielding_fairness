"""Log candidate-level decisions for the W16 criterion-disagreement experiment."""

from __future__ import annotations

import argparse
import csv
import logging
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TextIO

import numpy as np
import pandas as pd


logger: logging.Logger = logging.getLogger(__name__)

POLICIES: tuple[str, ...] = (
    "global_value",
    "equal_group_value",
    "equal_group_decline",
    "equal_group_protection",
)

CRITERIA: tuple[str, ...] = (
    "wbce_ratio",
    "plugin_ratio",
    "harm",
    "net_harm",
    "floor_ratio",
)

STRESS_COHORTS: tuple[tuple[str, float], ...] = (
    ("moderate", 0.50),
    ("strong", 0.25),
    ("extreme", 0.10),
)

BASE_CANDIDATE_FIELDS: tuple[str, ...] = (
    "dataset",
    "family",
    "seed",
    "model_variant",
    "cohort",
    "quantile",
    "target",
    "sample_split",
    "n",
    "candidate_id",
    "group",
    "label",
    "value",
    "loss",
    "score",
    "a_hat",
    "b_hat",
    "seconds",
    "net_lambda",
    "floor_a0",
)

METHOD_FIELDS: tuple[str, ...] = (
    "score_bce",
    "threshold_bce_value",
    "selected_bce_value",
)

POLICY_FIELDS: tuple[str, ...] = tuple(
    field
    for policy in POLICIES
    for field in (
        f"threshold_{policy}",
        f"lambda_{policy}",
        f"selected_{policy}",
    )
)

CRITERION_FIELDS: tuple[str, ...] = tuple(
    field
    for criterion in CRITERIA
    for field in (
        f"threshold_criterion_{criterion}",
        f"selected_criterion_{criterion}",
    )
)

CANDIDATE_FIELDS: tuple[str, ...] = (
    BASE_CANDIDATE_FIELDS + METHOD_FIELDS + POLICY_FIELDS + CRITERION_FIELDS
)

COMPACT_CANDIDATE_FIELDS: tuple[str, ...] = (
    "d",
    "f",
    "s",
    "t",
    "p",
    "g",
    "A",
    "B",
    "q",
    "qb",
    "tb",
    "bv",
    "ah",
    "bh",
    "gv",
    "egv",
    "egd",
    "egp",
    "w",
    "pr",
    "h",
    "nh",
    "fr",
)

COMPACT_FIELD_MAP: tuple[tuple[str, str], ...] = (
    ("dataset", "d"),
    ("family", "f"),
    ("seed", "s"),
    ("target", "t"),
    ("sample_split", "p"),
    ("group", "g"),
    ("value", "A"),
    ("loss", "B"),
    ("score", "q"),
    ("score_bce", "qb"),
    ("threshold_bce_value", "tb"),
    ("selected_bce_value", "bv"),
    ("a_hat", "ah"),
    ("b_hat", "bh"),
    ("selected_global_value", "gv"),
    ("selected_equal_group_value", "egv"),
    ("selected_equal_group_decline", "egd"),
    ("selected_equal_group_protection", "egp"),
    ("selected_criterion_wbce_ratio", "w"),
    ("selected_criterion_plugin_ratio", "pr"),
    ("selected_criterion_harm", "h"),
    ("selected_criterion_net_harm", "nh"),
    ("selected_criterion_floor_ratio", "fr"),
)

COMPACT_VALUE_CODES: dict[str, dict[str, str]] = {
    "dataset": {
        "australian_mtpl_group_blind": "0",
        "australian_mtpl_group_aware": "1",
        "australian_stress_moderate": "2",
        "australian_stress_strong": "3",
        "australian_stress_extreme": "4",
        "belgian_mtpl_group_blind": "5",
        "belgian_mtpl_group_aware": "6",
        "kuairand_group_blind": "7",
        "kuairand_group_aware": "8",
        "bondora_group_blind": "9",
        "bondora_group_aware": "10",
    },
    "family": {
        "natural_group_blind": "0",
        "natural_group_aware": "1",
        "constructed_stress": "2",
        "web_activity_group_blind": "3",
        "web_activity_group_aware": "4",
    },
    "sample_split": {
        "validation": "0",
        "test": "1",
    },
    "group": {
        "Female": "0",
        "Male": "1",
        "female": "0",
        "male": "1",
        "full_active": "2",
        "high_active": "3",
        "middle_active": "4",
        "2_14_day_new": "5",
        "low_active": "6",
        "single_low_active": "7",
        "30day_retention": "8",
        "day_new": "9",
        "UNKNOWN": "10",
    },
}

COMPACT_FLOAT_FIELDS: frozenset[str] = frozenset(
    (
        "target",
        "value",
        "loss",
        "score",
        "score_bce",
        "threshold_bce_value",
        "a_hat",
        "b_hat",
    )
)

AUSTRALIAN_NUMERIC: tuple[str, ...] = ("Exposure", "VehValue")
AUSTRALIAN_CATEGORICAL: tuple[str, ...] = (
    "VehAge",
    "VehBody",
    "Gender",
    "DrivAge",
)
BELGIAN_NUMERIC: tuple[str, ...] = (
    "expo",
    "ageph",
    "bm",
    "power",
    "agec",
    "long",
    "lat",
)
BELGIAN_CATEGORICAL: tuple[str, ...] = (
    "coverage",
    "sex",
    "fuel",
    "use",
    "fleet",
)
BONDORA_NUMERIC: tuple[str, ...] = (
    "ApplicationSignedHour",
    "ApplicationSignedWeekday",
    "Age",
    "AppliedAmount",
    "Amount",
    "Interest",
    "LoanDuration",
    "MonthlyPayment",
    "IncomeFromPrincipalEmployer",
    "IncomeFromPension",
    "IncomeFromFamilyAllowance",
    "IncomeFromSocialWelfare",
    "IncomeFromLeavePay",
    "IncomeFromChildSupport",
    "IncomeOther",
    "IncomeTotal",
    "ExistingLiabilities",
    "LiabilitiesTotal",
    "RefinanceLiabilities",
    "DebtToIncome",
    "FreeCash",
    "MonthlyPaymentDay",
    "NoOfPreviousLoansBeforeLoan",
    "AmountOfPreviousLoansBeforeLoan",
    "PreviousRepaymentsBeforeLoan",
    "PreviousEarlyRepaymentsCountBeforeLoan",
)
BONDORA_CATEGORICAL: tuple[str, ...] = (
    "NewCreditCustomer",
    "VerificationType",
    "LanguageCode",
    "Gender",
    "Country",
    "UseOfLoan",
    "Education",
    "MaritalStatus",
    "EmploymentStatus",
    "EmploymentDurationCurrentEmployer",
    "WorkExperience",
    "OccupationArea",
    "HomeOwnershipType",
)
BONDORA_OUTCOMES: tuple[str, ...] = (
    "Status",
    "PrincipalWriteOffs",
    "InterestAndPenaltyPaymentsMade",
)
KUAIRAND_LOG_COLUMNS: tuple[str, ...] = (
    "user_id",
    "video_id",
    "date",
    "hourmin",
    "is_hate",
    "play_time_ms",
    "duration_ms",
    "tab",
)
KUAIRAND_USER_COLUMNS: tuple[str, ...] = (
    "user_id",
    "user_active_degree",
    "is_lowactive_period",
    "is_live_streamer",
    "is_video_author",
    "follow_user_num",
    "follow_user_num_range",
    "fans_user_num",
    "fans_user_num_range",
    "friend_user_num",
    "friend_user_num_range",
    "register_days",
    "register_days_range",
)
KUAIRAND_VIDEO_COLUMNS: tuple[str, ...] = (
    "video_id",
    "video_type",
    "upload_dt",
    "upload_type",
    "video_duration",
    "server_width",
    "server_height",
    "music_type",
)
KUAIRAND_NUMERIC: tuple[str, ...] = (
    "hour",
    "tab",
    "duration_seconds",
    "video_age_days",
    "is_lowactive_period",
    "is_live_streamer",
    "is_video_author",
    "follow_user_num",
    "fans_user_num",
    "friend_user_num",
    "register_days",
    "video_duration_seconds",
    "server_width",
    "server_height",
)
KUAIRAND_CATEGORICAL: tuple[str, ...] = (
    "day",
    "user_active_degree",
    "follow_user_num_range",
    "fans_user_num_range",
    "friend_user_num_range",
    "register_days_range",
    "video_type",
    "upload_type",
    "music_type",
)


@dataclass(frozen=True)
class LoadedDataset:
    name: str
    frame: pd.DataFrame
    label: np.ndarray
    loss: np.ndarray
    value: np.ndarray
    groups: np.ndarray
    numeric_columns: tuple[str, ...]
    categorical_columns: tuple[str, ...]
    group_column: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument(
        "--dataset",
        choices=("australian_mtpl", "belgian_mtpl", "bondora", "kuairand"),
        default="australian_mtpl",
    )
    parser.add_argument(
        "--model-variant",
        choices=("both", "group_blind", "group_aware"),
        default="both",
    )
    parser.add_argument("--max-rows", type=int, default=0)
    parser.add_argument("--sample-seed", type=int, default=42)
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument("--seeds", type=int, default=10)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--pairs-per-batch", type=int, default=2048)
    parser.add_argument("--targets", default="0.1")
    parser.add_argument("--net-lambda", type=float, default=1.0)
    parser.add_argument("--floor-a0", type=float, default=0.0)
    parser.add_argument("--floor-quantile", type=float, default=0.05)
    parser.add_argument(
        "--dataset-scope",
        choices=("all", "natural", "stress"),
        default="all",
    )
    parser.add_argument(
        "--output-schema",
        choices=("compact", "verbose"),
        default="compact",
        help="Use compact coded schema by default so the frozen CSV fits Overleaf Git.",
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def parse_targets(text: str) -> list[float]:
    return [float(value.strip()) for value in text.split(",") if value.strip()]


def read_csv_input(path: str, dataset: str) -> pd.DataFrame:
    lowered = path.lower()
    if not lowered.endswith((".csv", ".csv.gz")):
        raise RuntimeError(
            f"Buck W16 {dataset} runs require a concrete CSV input."
        )
    return pd.read_csv(path)


def normalize_binary_groups(values: pd.Series) -> np.ndarray:
    normalized = values.fillna("MISSING").astype(str).str.strip().str.lower()
    groups = normalized.map({"female": "Female", "male": "Male"})
    if groups.isna().any():
        unknown = sorted(normalized[groups.isna()].unique())
        raise RuntimeError(f"Unexpected group values: {unknown}")
    return groups.to_numpy()


def load_australian_data(path: str) -> LoadedDataset:
    frame = read_csv_input(path, "Australian MTPL")
    loss = pd.to_numeric(frame["ClaimAmount"], errors="coerce").fillna(0)
    exposure = pd.to_numeric(frame["Exposure"], errors="coerce").fillna(0)
    label = (loss > 0).astype(np.float32).to_numpy()
    value = exposure.to_numpy(dtype=np.float32) * (1.0 - label)
    valid = (exposure > 0).to_numpy()
    kept = frame.loc[valid, list(AUSTRALIAN_NUMERIC + AUSTRALIAN_CATEGORICAL)]
    groups = normalize_binary_groups(kept["Gender"])
    return LoadedDataset(
        name="australian_mtpl",
        frame=kept.reset_index(drop=True),
        label=label[valid],
        loss=loss.loc[valid].to_numpy(dtype=np.float32),
        value=value[valid],
        groups=groups,
        numeric_columns=AUSTRALIAN_NUMERIC,
        categorical_columns=AUSTRALIAN_CATEGORICAL,
        group_column="Gender",
    )


def load_belgian_data(path: str) -> LoadedDataset:
    frame = read_csv_input(path, "Belgian MTPL")
    loss = pd.to_numeric(frame["amount"], errors="coerce").fillna(0)
    exposure = pd.to_numeric(frame["expo"], errors="coerce").fillna(0)
    label = (loss > 0).astype(np.float32).to_numpy()
    value = exposure.to_numpy(dtype=np.float32) * (1.0 - label)
    valid = (exposure > 0).to_numpy()
    kept = frame.loc[valid, list(BELGIAN_NUMERIC + BELGIAN_CATEGORICAL)]
    groups = normalize_binary_groups(kept["sex"])
    return LoadedDataset(
        name="belgian_mtpl",
        frame=kept.reset_index(drop=True),
        label=label[valid],
        loss=loss.loc[valid].to_numpy(dtype=np.float32),
        value=value[valid],
        groups=groups,
        numeric_columns=BELGIAN_NUMERIC,
        categorical_columns=BELGIAN_CATEGORICAL,
        group_column="sex",
    )


def load_bondora_data(path: str) -> LoadedDataset:
    columns = list(BONDORA_NUMERIC + BONDORA_CATEGORICAL + BONDORA_OUTCOMES)
    frame = pd.read_csv(path, usecols=columns, low_memory=False)
    frame = frame.loc[frame["Status"] == "Repaid"].reset_index(drop=True)
    loss = pd.to_numeric(frame["PrincipalWriteOffs"], errors="coerce").fillna(0)
    value = pd.to_numeric(
        frame["InterestAndPenaltyPaymentsMade"], errors="coerce"
    ).fillna(0)
    label = (loss > 0).astype(np.float32).to_numpy()
    value_array = value.to_numpy(dtype=np.float32) * (1.0 - label)
    gender = pd.to_numeric(frame["Gender"], errors="coerce")
    valid = (((loss.to_numpy(dtype=np.float32) + value_array) > 0) & gender.isin([0, 1])).to_numpy()
    kept = frame.loc[valid, list(BONDORA_NUMERIC + BONDORA_CATEGORICAL)]
    groups = gender.loc[valid].astype(int).map({0: "Male", 1: "Female"}).to_numpy()
    return LoadedDataset(
        name="bondora",
        frame=kept.reset_index(drop=True),
        label=label[valid],
        loss=loss.loc[valid].to_numpy(dtype=np.float32),
        value=value_array[valid],
        groups=groups,
        numeric_columns=BONDORA_NUMERIC,
        categorical_columns=BONDORA_CATEGORICAL,
        group_column="Gender",
    )


def load_kuairand_data(args: argparse.Namespace) -> LoadedDataset:
    path = Path(args.input)
    logs = pd.read_csv(
        path / "log_random_4_22_to_5_08_pure.csv",
        usecols=KUAIRAND_LOG_COLUMNS,
    )
    if args.max_rows and len(logs) > args.max_rows:
        logs = logs.sample(args.max_rows, random_state=args.sample_seed).sort_index()
    users = pd.read_csv(path / "user_features_pure.csv", usecols=KUAIRAND_USER_COLUMNS)
    videos = pd.read_csv(
        path / "video_features_basic_pure.csv", usecols=KUAIRAND_VIDEO_COLUMNS
    )
    frame = logs.merge(users, on="user_id", how="left", validate="many_to_one")
    frame = frame.merge(videos, on="video_id", how="left", validate="many_to_one")

    interaction_date = pd.to_datetime(
        frame["date"].astype("Int64").astype(str), format="%Y%m%d", errors="coerce"
    )
    upload_date = pd.to_datetime(frame["upload_dt"], errors="coerce")
    frame["hour"] = pd.to_numeric(frame["hourmin"], errors="coerce") // 100
    frame["day"] = interaction_date.dt.dayofweek.astype("Int64").astype(str)
    frame["duration_seconds"] = (
        pd.to_numeric(frame["duration_ms"], errors="coerce") / 1000.0
    )
    frame["video_duration_seconds"] = (
        pd.to_numeric(frame["video_duration"], errors="coerce") / 1000.0
    )
    frame["video_age_days"] = (interaction_date - upload_date).dt.days

    loss = pd.to_numeric(frame["is_hate"], errors="coerce").fillna(0)
    value = pd.to_numeric(frame["play_time_ms"], errors="coerce").fillna(0) / 1000.0
    label = (loss > 0).astype(np.float32).to_numpy()
    valid = ((loss >= 0) & (value >= 0)).to_numpy()
    kept = frame.loc[valid, list(KUAIRAND_NUMERIC + KUAIRAND_CATEGORICAL)]
    groups = kept["user_active_degree"].fillna("UNKNOWN").astype(str).to_numpy()
    return LoadedDataset(
        name="kuairand",
        frame=kept.reset_index(drop=True),
        label=label[valid],
        loss=loss.loc[valid].to_numpy(dtype=np.float32),
        value=value.loc[valid].to_numpy(dtype=np.float32),
        groups=groups,
        numeric_columns=KUAIRAND_NUMERIC,
        categorical_columns=KUAIRAND_CATEGORICAL,
        group_column="user_active_degree",
    )


def load_dataset(args: argparse.Namespace) -> LoadedDataset:
    if args.dataset == "australian_mtpl":
        return load_australian_data(args.input)
    if args.dataset == "belgian_mtpl":
        return load_belgian_data(args.input)
    if args.dataset == "bondora":
        return load_bondora_data(args.input)
    if args.dataset == "kuairand":
        return load_kuairand_data(args)
    raise ValueError(f"Unknown dataset: {args.dataset}")


def finite_exp(value: float) -> float:
    if not math.isfinite(value):
        return value
    if value > 700:
        return float("inf")
    return math.exp(value)


def format_value(value: float | int | str) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, int):
        return str(value)
    if math.isnan(value):
        return ""
    if math.isinf(value):
        return "inf" if value > 0 else "-inf"
    return f"{value:.10g}"


def format_compact_float(value: str) -> str:
    if value == "":
        return value
    parsed = float(value)
    if math.isnan(parsed):
        return ""
    if math.isinf(parsed):
        return "inf" if parsed > 0 else "-inf"
    return f"{parsed:.7g}"


def compact_value(field: str, value: str) -> str:
    code_map = COMPACT_VALUE_CODES.get(field)
    if code_map is not None:
        return code_map[value]
    if field in COMPACT_FLOAT_FIELDS:
        return format_compact_float(value)
    return value


def compact_row(row: dict[str, str]) -> dict[str, str]:
    return {
        compact: compact_value(verbose, row[verbose])
        for verbose, compact in COMPACT_FIELD_MAP
    }


def write_candidate_row(writer: csv.DictWriter, row: dict[str, str]) -> None:
    if tuple(writer.fieldnames or ()) == COMPACT_CANDIDATE_FIELDS:
        writer.writerow(compact_row(row))
    else:
        writer.writerow(row)


def threshold_map(
    groups: np.ndarray,
    default_threshold: float,
) -> dict[str, float]:
    return {str(group): default_threshold for group in np.unique(groups)}


def threshold_for_mass(scores: np.ndarray, mass: np.ndarray, fraction: float) -> float:
    order = np.argsort(-scores, kind="stable")
    cumulative = np.cumsum(mass[order], dtype=np.float64)
    if cumulative[-1] <= 0:
        return float("inf")
    position = min(
        np.searchsorted(cumulative, fraction * cumulative[-1], side="left"),
        len(order) - 1,
    )
    return float(scores[order[position]])


def equal_decline_thresholds_at_value_budget(
    scores: np.ndarray,
    groups: np.ndarray,
    value: np.ndarray,
    value_fraction: float,
) -> tuple[dict[str, float], float]:
    curves = {}
    for group in np.unique(groups):
        mask = groups == group
        order = np.argsort(-scores[mask], kind="stable")
        ordered_scores = scores[mask][order]
        tie_boundaries = np.r_[
            np.flatnonzero(ordered_scores[:-1] != ordered_scores[1:]),
            len(ordered_scores) - 1,
        ]
        curves[str(group)] = (
            ordered_scores,
            np.cumsum(value[mask][order], dtype=np.float64),
            tie_boundaries,
        )
    low, high = 0.0, 1.0
    thresholds: dict[str, float] = {}
    for _ in range(30):
        decline_fraction = (low + high) / 2.0
        spent_value = 0.0
        thresholds = {}
        for group, (ordered_scores, cumulative_value, tie_boundaries) in curves.items():
            index = min(
                int(np.ceil(decline_fraction * len(ordered_scores))) - 1,
                len(ordered_scores) - 1,
            )
            index = max(index, 0)
            thresholds[group] = float(ordered_scores[index])
            tie_end = tie_boundaries[np.searchsorted(tie_boundaries, index)]
            spent_value += cumulative_value[tie_end]
        spent = spent_value / max(value.sum(), 1e-12)
        if spent < value_fraction:
            low = decline_fraction
        else:
            high = decline_fraction
    return thresholds, (low + high) / 2.0


def equal_protection_thresholds_at_value_budget(
    scores: np.ndarray,
    groups: np.ndarray,
    value: np.ndarray,
    loss: np.ndarray,
    value_fraction: float,
) -> tuple[dict[str, float], float]:
    curves = {}
    for group in np.unique(groups):
        mask = groups == group
        order = np.argsort(-scores[mask], kind="stable")
        ordered_scores = scores[mask][order]
        tie_boundaries = np.r_[
            np.flatnonzero(ordered_scores[:-1] != ordered_scores[1:]),
            len(ordered_scores) - 1,
        ]
        curves[str(group)] = (
            ordered_scores,
            np.cumsum(loss[mask][order], dtype=np.float64),
            np.cumsum(value[mask][order], dtype=np.float64),
            tie_boundaries,
        )
    low, high = 0.0, 1.0
    thresholds: dict[str, float] = {}
    for _ in range(30):
        protection_fraction = (low + high) / 2.0
        spent_value = 0.0
        thresholds = {}
        for group, (
            ordered_scores,
            cumulative_loss,
            cumulative_value,
            tie_boundaries,
        ) in curves.items():
            if cumulative_loss[-1] <= 0:
                index = len(ordered_scores) - 1
            else:
                index = min(
                    np.searchsorted(
                        cumulative_loss,
                        protection_fraction * cumulative_loss[-1],
                        side="left",
                    ),
                    len(ordered_scores) - 1,
                )
            thresholds[group] = float(ordered_scores[index])
            tie_end = tie_boundaries[np.searchsorted(tie_boundaries, index)]
            spent_value += cumulative_value[tie_end]
        spent = spent_value / max(value.sum(), 1e-12)
        if spent < value_fraction:
            low = protection_fraction
        else:
            high = protection_fraction
    return thresholds, (low + high) / 2.0


def policy_thresholds(
    val_scores: np.ndarray,
    val_groups: np.ndarray,
    val_value: np.ndarray,
    val_loss: np.ndarray,
    target: float,
) -> dict[str, dict[str, float]]:
    global_threshold = threshold_for_mass(val_scores, val_value, target)
    value_thresholds = {
        str(group): threshold_for_mass(
            val_scores[val_groups == group],
            val_value[val_groups == group],
            target,
        )
        for group in np.unique(val_groups)
    }
    decline_thresholds, _ = equal_decline_thresholds_at_value_budget(
        val_scores, val_groups, val_value, target
    )
    protection_thresholds, _ = equal_protection_thresholds_at_value_budget(
        val_scores, val_groups, val_value, val_loss, target
    )
    return {
        "global_value": threshold_map(val_groups, global_threshold),
        "equal_group_value": value_thresholds,
        "equal_group_decline": decline_thresholds,
        "equal_group_protection": protection_thresholds,
    }


def select_by_thresholds(
    scores: np.ndarray,
    groups: np.ndarray,
    thresholds: dict[str, float],
) -> np.ndarray:
    return np.asarray(
        [
            score >= thresholds.get(str(group), float("inf"))
            for score, group in zip(scores, groups)
        ],
        dtype=bool,
    )


def policy_selections(
    test_scores: np.ndarray,
    test_groups: np.ndarray,
    thresholds: dict[str, dict[str, float]],
) -> dict[str, np.ndarray]:
    return {
        policy: select_by_thresholds(test_scores, test_groups, policy_threshold)
        for policy, policy_threshold in thresholds.items()
    }


def safe_ratio(numerator: np.ndarray, denominator: np.ndarray) -> np.ndarray:
    result = np.zeros_like(numerator, dtype=np.float64)
    positive = denominator > 1e-12
    result[positive] = numerator[positive] / denominator[positive]
    result[(~positive) & (numerator > 1e-12)] = float("inf")
    return result


def floor_denominator(val_a_hat: np.ndarray, args: argparse.Namespace) -> float:
    if args.floor_a0 > 0:
        return float(args.floor_a0)
    positive = val_a_hat[val_a_hat > 0]
    if len(positive) == 0:
        return 1e-12
    quantile = min(max(float(args.floor_quantile), 0.0), 1.0)
    return float(np.quantile(positive, quantile))


def criterion_score_maps(
    val_scores: np.ndarray,
    test_scores: np.ndarray,
    val_a_hat: np.ndarray,
    test_a_hat: np.ndarray,
    val_b_hat: np.ndarray,
    test_b_hat: np.ndarray,
    args: argparse.Namespace,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], float]:
    floor_a0 = floor_denominator(val_a_hat, args)
    val_criteria = {
        "wbce_ratio": val_scores.astype(np.float64),
        "plugin_ratio": safe_ratio(val_b_hat, val_a_hat),
        "harm": val_b_hat.astype(np.float64),
        "net_harm": val_b_hat.astype(np.float64)
        - float(args.net_lambda) * val_a_hat.astype(np.float64),
        "floor_ratio": val_b_hat.astype(np.float64) / np.maximum(val_a_hat, floor_a0),
    }
    test_criteria = {
        "wbce_ratio": test_scores.astype(np.float64),
        "plugin_ratio": safe_ratio(test_b_hat, test_a_hat),
        "harm": test_b_hat.astype(np.float64),
        "net_harm": test_b_hat.astype(np.float64)
        - float(args.net_lambda) * test_a_hat.astype(np.float64),
        "floor_ratio": test_b_hat.astype(np.float64) / np.maximum(test_a_hat, floor_a0),
    }
    return val_criteria, test_criteria, floor_a0


def criterion_thresholds(
    val_criteria: dict[str, np.ndarray],
    val_value: np.ndarray,
    target: float,
) -> dict[str, float]:
    return {
        criterion: threshold_for_mass(scores, val_value, target)
        for criterion, scores in val_criteria.items()
    }


def criterion_selections(
    test_criteria: dict[str, np.ndarray],
    thresholds: dict[str, float],
) -> dict[str, np.ndarray]:
    return {
        criterion: np.asarray(scores >= thresholds[criterion], dtype=bool)
        for criterion, scores in test_criteria.items()
    }


def stratified_split(
    indices: np.ndarray,
    labels: np.ndarray,
    test_fraction: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    left_parts = []
    right_parts = []
    scoped_labels = labels[indices]
    for label in np.unique(scoped_labels):
        cell = indices[scoped_labels == label].copy()
        rng.shuffle(cell)
        if len(cell) <= 1:
            split = 0
        else:
            split = int(round(test_fraction * len(cell)))
            split = min(max(split, 1), len(cell) - 1)
        right_parts.append(cell[:split])
        left_parts.append(cell[split:])
    left = np.concatenate(left_parts)
    right = np.concatenate(right_parts)
    rng.shuffle(left)
    rng.shuffle(right)
    return left, right


def one_hot(values: pd.Series, levels: tuple[str, ...]) -> np.ndarray:
    result = np.zeros((len(values), len(levels)), dtype=np.float32)
    level_index = {level: index for index, level in enumerate(levels)}
    for row, value in enumerate(values.fillna("MISSING").astype(str).to_numpy()):
        column = level_index.get(value)
        if column is not None:
            result[row, column] = 1.0
    return result


def transform_split(
    frame: pd.DataFrame,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    test_idx: np.ndarray,
    include_group: bool,
    numeric_columns: tuple[str, ...],
    categorical_columns: tuple[str, ...],
    group_column: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    categorical_columns = [
        column for column in categorical_columns if include_group or column != group_column
    ]
    numeric = frame[list(numeric_columns)].apply(pd.to_numeric, errors="coerce")
    medians = numeric.iloc[train_idx].median().fillna(0)
    numeric = numeric.fillna(medians).fillna(0)
    train_numeric = numeric.iloc[train_idx].to_numpy(dtype=np.float64)
    mean = train_numeric.mean(axis=0)
    scale = train_numeric.std(axis=0)
    scale[scale <= 0] = 1.0
    categorical = frame[categorical_columns].fillna("MISSING").astype(str)
    levels = {
        column: tuple(sorted(categorical.iloc[train_idx][column].unique()))
        for column in categorical_columns
    }

    def build(index: np.ndarray) -> np.ndarray:
        parts = [
            ((numeric.iloc[index].to_numpy(dtype=np.float64) - mean) / scale).astype(
                np.float32
            )
        ]
        for column in categorical_columns:
            parts.append(one_hot(categorical.iloc[index][column], levels[column]))
        return np.concatenate(parts, axis=1)

    return build(train_idx), build(val_idx), build(test_idx)


def constructed_cohort(
    frame: pd.DataFrame,
    label: np.ndarray,
    loss: np.ndarray,
    value: np.ndarray,
    quantile: float,
    seed: int,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    gender = frame["Gender"].astype(str).to_numpy()
    exposure = pd.to_numeric(frame["Exposure"], errors="coerce").to_numpy()
    female_cut = np.quantile(exposure[gender == "Female"], 1.0 - quantile)
    male_cut = np.quantile(exposure[gender == "Male"], quantile)
    candidate = ((gender == "Female") & (exposure >= female_cut)) | (
        (gender == "Male") & (exposure <= male_cut)
    )
    rng = np.random.default_rng(seed)
    chosen = []
    good_cells = [
        np.flatnonzero(candidate & (gender == group) & (label == 0))
        for group in ("Female", "Male")
    ]
    good_count = min(map(len, good_cells))
    chosen.extend(rng.choice(cell, good_count, replace=False) for cell in good_cells)

    bad_cells = [
        np.flatnonzero(candidate & (gender == group) & (label == 1))
        for group in ("Female", "Male")
    ]
    left = bad_cells[0][np.argsort(loss[bad_cells[0]])]
    right = bad_cells[1][np.argsort(loss[bad_cells[1]])]
    i = j = 0
    matched_left, matched_right = [], []
    log_loss = np.log(np.maximum(loss, 1e-12))
    while i < len(left) and j < len(right):
        difference = log_loss[left[i]] - log_loss[right[j]]
        if abs(difference) <= np.log(1.15):
            matched_left.append(left[i])
            matched_right.append(right[j])
            i += 1
            j += 1
        elif difference < 0:
            i += 1
        else:
            j += 1
    if len(matched_left) < 20:
        raise RuntimeError("Too few loss-matched bad cases")
    chosen.extend([np.asarray(matched_left), np.asarray(matched_right)])
    index = np.sort(np.concatenate(chosen))
    return (
        frame.iloc[index].reset_index(drop=True),
        label[index],
        loss[index],
        value[index],
        gender[index],
        exposure[index],
    )


def write_candidates(
    writer: csv.DictWriter,
    context: dict[str, str | float | int],
    sample_split: str,
    test_idx: np.ndarray,
    test_scores: np.ndarray,
    test_groups: np.ndarray,
    test_label: np.ndarray,
    test_value: np.ndarray,
    test_loss: np.ndarray,
    test_a_hat: np.ndarray,
    test_b_hat: np.ndarray,
    test_bce_scores: np.ndarray,
    bce_threshold: float,
    bce_selected: np.ndarray,
    thresholds: dict[str, dict[str, float]],
    selections: dict[str, np.ndarray],
    criterion_threshold: dict[str, float],
    criterion_selection: dict[str, np.ndarray],
) -> None:
    for position, candidate_id in enumerate(test_idx):
        group = str(test_groups[position])
        row: dict[str, str] = {
            key: format_value(value) for key, value in context.items()
        }
        row["sample_split"] = sample_split
        row.update(
            {
                "candidate_id": str(int(candidate_id)),
                "group": group,
                "label": format_value(float(test_label[position])),
                "value": format_value(float(test_value[position])),
                "loss": format_value(float(test_loss[position])),
                "score": format_value(float(test_scores[position])),
                "score_bce": format_value(float(test_bce_scores[position])),
                "threshold_bce_value": format_value(float(bce_threshold)),
                "selected_bce_value": "1" if bce_selected[position] else "0",
                "a_hat": format_value(float(test_a_hat[position])),
                "b_hat": format_value(float(test_b_hat[position])),
            }
        )
        for policy in POLICIES:
            threshold = thresholds[policy].get(group, float("inf"))
            row[f"threshold_{policy}"] = format_value(threshold)
            row[f"lambda_{policy}"] = format_value(finite_exp(threshold))
            row[f"selected_{policy}"] = "1" if selections[policy][position] else "0"
        for criterion in CRITERIA:
            row[f"threshold_criterion_{criterion}"] = format_value(
                criterion_threshold[criterion]
            )
            row[f"selected_criterion_{criterion}"] = (
                "1" if criterion_selection[criterion][position] else "0"
            )
        write_candidate_row(writer, row)


def run_model(
    x_train: np.ndarray,
    x_val: np.ndarray,
    x_test: np.ndarray,
    loss_train: np.ndarray,
    value_train: np.ndarray,
    label_train: np.ndarray,
    seed: int,
    args: argparse.Namespace,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    float,
]:
    started = time.perf_counter()
    _ = (label_train, args.batch_size, args.pairs_per_batch)
    x_fit, y_fit, weights = weighted_logistic_rows(x_train, loss_train, value_train)
    coef, intercept = fit_weighted_logistic(
        x_fit,
        y_fit,
        weights,
        steps=max(100, args.epochs * 50),
        seed=seed,
    )
    bce_coef, bce_intercept = fit_weighted_logistic(
        x_train,
        label_train.astype(np.float32),
        np.ones(len(label_train), dtype=np.float64),
        steps=max(100, args.epochs * 50),
        seed=seed + 3,
    )
    x_eval = np.concatenate([x_val, x_test])
    scores = x_eval @ coef + intercept
    bce_scores = x_eval @ bce_coef + bce_intercept

    value_coef, value_intercept, value_location, value_scale = fit_raw_mean_regression(
        x_train,
        value_train,
        steps=max(100, args.epochs * 50),
        seed=seed + 1,
    )
    loss_coef, loss_intercept, loss_location, loss_scale = fit_raw_mean_regression(
        x_train,
        loss_train,
        steps=max(100, args.epochs * 50),
        seed=seed + 2,
    )
    a_hat = np.maximum(
        predict_raw_mean_regression(
            x_eval,
            value_coef,
            value_intercept,
            value_location,
            value_scale,
        ),
        1e-9,
    )
    b_hat = predict_raw_mean_regression(
        x_eval,
        loss_coef,
        loss_intercept,
        loss_location,
        loss_scale,
    )
    val_count = len(x_val)
    return (
        scores[:val_count],
        scores[val_count:],
        bce_scores[:val_count],
        bce_scores[val_count:],
        a_hat[:val_count],
        a_hat[val_count:],
        b_hat[:val_count],
        b_hat[val_count:],
        time.perf_counter() - started,
    )


def sigmoid(values: np.ndarray) -> np.ndarray:
    result = np.empty_like(values, dtype=np.float64)
    positive = values >= 0
    result[positive] = 1.0 / (1.0 + np.exp(-values[positive]))
    exp_values = np.exp(values[~positive])
    result[~positive] = exp_values / (1.0 + exp_values)
    return result


def fit_weighted_logistic(
    x: np.ndarray,
    y: np.ndarray,
    weights: np.ndarray,
    steps: int,
    seed: int,
) -> tuple[np.ndarray, float]:
    rng = np.random.default_rng(seed)
    x64 = x.astype(np.float64)
    y64 = y.astype(np.float64)
    weights64 = weights.astype(np.float64)
    weight_sum = max(float(weights64.sum()), 1e-12)
    coef = rng.normal(0.0, 1e-3, size=x64.shape[1])
    intercept = 0.0
    coef_m = np.zeros_like(coef)
    coef_v = np.zeros_like(coef)
    intercept_m = 0.0
    intercept_v = 0.0
    beta1, beta2 = 0.9, 0.999
    learning_rate = 0.05
    l2 = 1e-4
    for step in range(1, steps + 1):
        error = sigmoid(x64 @ coef + intercept) - y64
        weighted_error = weights64 * error / weight_sum
        grad_coef = x64.T @ weighted_error + l2 * coef
        grad_intercept = float(weighted_error.sum())

        coef_m = beta1 * coef_m + (1.0 - beta1) * grad_coef
        coef_v = beta2 * coef_v + (1.0 - beta2) * grad_coef * grad_coef
        intercept_m = beta1 * intercept_m + (1.0 - beta1) * grad_intercept
        intercept_v = beta2 * intercept_v + (1.0 - beta2) * grad_intercept**2

        coef -= (
            learning_rate
            * (coef_m / (1.0 - beta1**step))
            / (np.sqrt(coef_v / (1.0 - beta2**step)) + 1e-8)
        )
        intercept -= (
            learning_rate
            * (intercept_m / (1.0 - beta1**step))
            / (math.sqrt(intercept_v / (1.0 - beta2**step)) + 1e-8)
        )
    return coef.astype(np.float32), float(intercept)


def fit_raw_mean_regression(
    x: np.ndarray,
    y: np.ndarray,
    steps: int,
    seed: int,
) -> tuple[np.ndarray, float, float, float]:
    rng = np.random.default_rng(seed)
    x64 = x.astype(np.float64)
    target = np.maximum(y.astype(np.float64), 0.0)
    location = float(target.mean()) if len(target) else 0.0
    scale_value = float(target.std(ddof=0)) if len(target) else 1.0
    if scale_value <= 1e-12:
        scale_value = 1.0
    target = (target - location) / scale_value
    coef = rng.normal(0.0, 1e-3, size=x64.shape[1])
    intercept = float(target.mean()) if len(target) else 0.0
    coef_m = np.zeros_like(coef)
    coef_v = np.zeros_like(coef)
    intercept_m = 0.0
    intercept_v = 0.0
    beta1, beta2 = 0.9, 0.999
    learning_rate = 0.03
    l2 = 1e-4
    scale = max(float(len(target)), 1.0)
    for step in range(1, steps + 1):
        residual = x64 @ coef + intercept - target
        grad_coef = (x64.T @ residual) / scale + l2 * coef
        grad_intercept = float(residual.mean()) if len(residual) else 0.0

        coef_m = beta1 * coef_m + (1.0 - beta1) * grad_coef
        coef_v = beta2 * coef_v + (1.0 - beta2) * grad_coef * grad_coef
        intercept_m = beta1 * intercept_m + (1.0 - beta1) * grad_intercept
        intercept_v = beta2 * intercept_v + (1.0 - beta2) * grad_intercept**2

        coef -= (
            learning_rate
            * (coef_m / (1.0 - beta1**step))
            / (np.sqrt(coef_v / (1.0 - beta2**step)) + 1e-8)
        )
        intercept -= (
            learning_rate
            * (intercept_m / (1.0 - beta1**step))
            / (math.sqrt(intercept_v / (1.0 - beta2**step)) + 1e-8)
        )
    return coef.astype(np.float32), float(intercept), location, scale_value


def predict_raw_mean_regression(
    x: np.ndarray,
    coef: np.ndarray,
    intercept: float,
    location: float,
    scale_value: float,
) -> np.ndarray:
    linear = x.astype(np.float64) @ coef.astype(np.float64) + intercept
    return np.maximum(linear * scale_value + location, 0.0).astype(np.float32)


def weighted_logistic_rows(
    x_train: np.ndarray,
    loss_train: np.ndarray,
    value_train: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    positive = loss_train > 0
    negative = value_train > 0
    x_fit = np.concatenate([x_train[positive], x_train[negative]])
    y_fit = np.concatenate(
        [
            np.ones(int(positive.sum()), dtype=np.float32),
            np.zeros(int(negative.sum()), dtype=np.float32),
        ]
    )
    weights = np.concatenate([loss_train[positive], value_train[negative]]).astype(
        np.float64
    )
    weights = weights / max(float(weights.mean()), 1e-12)
    return x_fit, y_fit, weights


def run_natural(
    writer: csv.DictWriter,
    args: argparse.Namespace,
    targets: list[float],
) -> None:
    data = load_dataset(args)
    frame = data.frame
    label = data.label
    loss = data.loss
    value = data.value
    groups = data.groups
    indices = np.arange(len(frame))
    for seed in range(args.seed_start, args.seed_start + args.seeds):
        outer_train, test_idx = stratified_split(indices, label, 0.20, seed)
        train_idx, val_idx = stratified_split(outer_train, label, 0.20, 10_000 + seed)
        variants = (("group_blind", False), ("group_aware", True))
        if args.model_variant != "both":
            variants = tuple(item for item in variants if item[0] == args.model_variant)
        for variant, include_group in variants:
            x_train, x_val, x_test = transform_split(
                frame,
                train_idx,
                val_idx,
                test_idx,
                include_group,
                data.numeric_columns,
                data.categorical_columns,
                data.group_column,
            )
            (
                val_scores,
                test_scores,
                val_bce_scores,
                test_bce_scores,
                val_a_hat,
                test_a_hat,
                val_b_hat,
                test_b_hat,
                elapsed,
            ) = run_model(
                x_train,
                x_val,
                x_test,
                loss[train_idx],
                value[train_idx],
                label[train_idx],
                120_000 + seed + (10_000 if include_group else 0),
                args,
            )
            context = {
                "dataset": f"{data.name}_{variant}",
                "family": (
                    "web_activity_group_aware"
                    if data.name == "kuairand" and include_group
                    else "web_activity_group_blind"
                    if data.name == "kuairand"
                    else "natural_group_aware"
                    if include_group
                    else "natural_group_blind"
                ),
                "seed": seed,
                "model_variant": variant,
                "cohort": "",
                "quantile": "",
                "n": len(frame),
                "seconds": elapsed,
            }
            emit_targets(
                writer,
                context,
                targets,
                args,
                val_scores,
                test_scores,
                val_bce_scores,
                test_bce_scores,
                val_a_hat,
                test_a_hat,
                val_b_hat,
                test_b_hat,
                groups[val_idx],
                groups[test_idx],
                label[test_idx],
                value[val_idx],
                value[test_idx],
                loss[val_idx],
                loss[test_idx],
                label[val_idx],
                test_idx,
                val_idx,
            )


def run_stress(
    writer: csv.DictWriter,
    args: argparse.Namespace,
    targets: list[float],
) -> None:
    if args.dataset != "australian_mtpl":
        raise RuntimeError("Constructed stress cohorts are defined for Australian MTPL.")
    full_data = load_australian_data(args.input)
    full_frame = full_data.frame
    full_label = full_data.label
    full_loss = full_data.loss
    full_value = full_data.value
    for cohort, quantile in STRESS_COHORTS:
        for seed in range(args.seed_start, args.seed_start + args.seeds):
            frame, label, loss, value, groups, _ = constructed_cohort(
                full_frame,
                full_label,
                full_loss,
                full_value,
                quantile,
                40_000 + seed,
            )
            indices = np.arange(len(frame))
            train_val, test_idx = stratified_split(indices, label, 0.20, seed)
            train_idx, val_idx = stratified_split(train_val, label, 0.20, 10_000 + seed)
            x_train, x_val, x_test = transform_split(
                frame,
                train_idx,
                val_idx,
                test_idx,
                include_group=False,
                numeric_columns=full_data.numeric_columns,
                categorical_columns=full_data.categorical_columns,
                group_column=full_data.group_column,
            )
            (
                val_scores,
                test_scores,
                val_bce_scores,
                test_bce_scores,
                val_a_hat,
                test_a_hat,
                val_b_hat,
                test_b_hat,
                elapsed,
            ) = run_model(
                x_train,
                x_val,
                x_test,
                loss[train_idx],
                value[train_idx],
                label[train_idx],
                300_000 + seed,
                args,
            )
            context = {
                "dataset": f"australian_stress_{cohort}",
                "family": "constructed_stress",
                "seed": seed,
                "model_variant": "group_blind",
                "cohort": cohort,
                "quantile": quantile,
                "n": len(frame),
                "seconds": elapsed,
            }
            emit_targets(
                writer,
                context,
                targets,
                args,
                val_scores,
                test_scores,
                val_bce_scores,
                test_bce_scores,
                val_a_hat,
                test_a_hat,
                val_b_hat,
                test_b_hat,
                groups[val_idx],
                groups[test_idx],
                label[test_idx],
                value[val_idx],
                value[test_idx],
                loss[val_idx],
                loss[test_idx],
                label[val_idx],
                test_idx,
                val_idx,
            )


def emit_targets(
    writer: csv.DictWriter,
    context: dict[str, str | float | int],
    targets: list[float],
    args: argparse.Namespace,
    val_scores: np.ndarray,
    test_scores: np.ndarray,
    val_bce_scores: np.ndarray,
    test_bce_scores: np.ndarray,
    val_a_hat: np.ndarray,
    test_a_hat: np.ndarray,
    val_b_hat: np.ndarray,
    test_b_hat: np.ndarray,
    val_groups: np.ndarray,
    test_groups: np.ndarray,
    test_label: np.ndarray,
    val_value: np.ndarray,
    test_value: np.ndarray,
    val_loss: np.ndarray,
    test_loss: np.ndarray,
    val_label: np.ndarray,
    test_idx: np.ndarray,
    val_idx: np.ndarray,
) -> None:
    for target in targets:
        thresholds = policy_thresholds(
            val_scores, val_groups, val_value, val_loss, target
        )
        bce_threshold = threshold_for_mass(val_bce_scores, val_value, target)
        val_bce_selected = np.asarray(val_bce_scores >= bce_threshold, dtype=bool)
        test_bce_selected = np.asarray(test_bce_scores >= bce_threshold, dtype=bool)
        val_selections = policy_selections(val_scores, val_groups, thresholds)
        test_selections = policy_selections(test_scores, test_groups, thresholds)
        val_criteria, test_criteria, floor_a0 = criterion_score_maps(
            val_scores,
            test_scores,
            val_a_hat,
            test_a_hat,
            val_b_hat,
            test_b_hat,
            args,
        )
        criterion_threshold = criterion_thresholds(val_criteria, val_value, target)
        val_criterion_selection = criterion_selections(
            val_criteria, criterion_threshold
        )
        test_criterion_selection = criterion_selections(
            test_criteria, criterion_threshold
        )
        write_candidates(
            writer,
            {
                **context,
                "target": target,
                "net_lambda": args.net_lambda,
                "floor_a0": floor_a0,
            },
            "validation",
            val_idx,
            val_scores,
            val_groups,
            val_label,
            val_value,
            val_loss,
            val_a_hat,
            val_b_hat,
            val_bce_scores,
            bce_threshold,
            val_bce_selected,
            thresholds,
            val_selections,
            criterion_threshold,
            val_criterion_selection,
        )
        write_candidates(
            writer,
            {
                **context,
                "target": target,
                "net_lambda": args.net_lambda,
                "floor_a0": floor_a0,
            },
            "test",
            test_idx,
            test_scores,
            test_groups,
            test_label,
            test_value,
            test_loss,
            test_a_hat,
            test_b_hat,
            test_bce_scores,
            bce_threshold,
            test_bce_selected,
            thresholds,
            test_selections,
            criterion_threshold,
            test_criterion_selection,
        )


def open_output(path: Path) -> TextIO:
    path.parent.mkdir(parents=True, exist_ok=True)
    return path.open("w", newline="")


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args = parse_args()
    targets = parse_targets(args.targets)
    fields = (
        COMPACT_CANDIDATE_FIELDS
        if args.output_schema == "compact"
        else CANDIDATE_FIELDS
    )
    with open_output(args.output) as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        if args.dataset_scope in {"all", "natural"}:
            run_natural(writer, args, targets)
        if args.dataset_scope in {"all", "stress"}:
            if args.dataset == "australian_mtpl":
                run_stress(writer, args, targets)
            elif args.dataset_scope == "stress":
                raise RuntimeError(
                    "Constructed stress cohorts are defined for Australian MTPL."
                )
            else:
                logger.info("skipping unavailable stress cohorts for %s", args.dataset)
    logger.info(f"saved {args.output}")


if __name__ == "__main__":
    main()
