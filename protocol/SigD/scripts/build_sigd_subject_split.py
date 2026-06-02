#!/usr/bin/env python3
"""Build a subject-disjoint split for the SigD common 10s protocol."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import random
import statistics
import sys
from datetime import datetime, timezone
from typing import Any, Iterable

import yaml


DEFAULT_CONFIG = Path("protocol/SigD/config/sigd_protocol_10s_k5m1.yaml")
SPLITS = ("train", "val", "test")
SPLIT_COLUMNS = [
    "subject_id",
    "split",
    "session_count",
    "total_common_available_windows",
    "primary_pair_count",
    "all_pair_count",
    "min_gap_days",
    "median_gap_days",
    "max_gap_days",
]


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def detect_root(root_arg: str | None) -> Path:
    if root_arg:
        return Path(root_arg).expanduser().resolve()
    for candidate in [Path.cwd().resolve(), *Path.cwd().resolve().parents]:
        if (candidate / "preprocessing" / "SigD").exists():
            return candidate
    raise SystemExit("Could not detect SIM_PPG root. Pass --root PATH.")


def load_config(root: Path, config_path: str | None) -> dict[str, Any]:
    path = Path(config_path).expanduser().resolve() if config_path else root / DEFAULT_CONFIG
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def resolve(root: Path, path: str | Path) -> Path:
    value = Path(path)
    return value if value.is_absolute() else root / value


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]], columns: list[str], overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise SystemExit(f"Output exists; pass --overwrite: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column, "") for column in columns})


def write_json(path: Path, payload: dict[str, Any], overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise SystemExit(f"Output exists; pass --overwrite: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def as_float(value: Any) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(parsed):
        return None
    return parsed


def bool_value(value: Any) -> bool:
    return str(value).strip().lower() in {"true", "1", "yes", "y"}


def numeric_summary(values: Iterable[Any]) -> dict[str, Any]:
    clean = sorted(value for value in (as_float(item) for item in values) if value is not None)
    if not clean:
        return {"count": 0, "min": None, "median": None, "max": None, "mean": None}
    return {
        "count": len(clean),
        "min": clean[0],
        "median": statistics.median(clean),
        "max": clean[-1],
        "mean": sum(clean) / len(clean),
    }


def distribution(values: Iterable[Any]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for value in values:
        key = str(value)
        counts[key] = counts.get(key, 0) + 1
    return dict(sorted(counts.items()))


def median(values: list[float]) -> float | str:
    return statistics.median(values) if values else ""


def subject_features(config: dict[str, Any], subject_rows: list[dict[str, str]], pair_rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    """Create one feature row per eligible subject."""

    eligible_subjects = {
        row["subject_id"]: row
        for row in subject_rows
        if bool_value(row.get("retained_after_preprocessing_k5"))
    }
    gaps_by_subject: dict[str, list[float]] = {subject: [] for subject in eligible_subjects}
    primary_pairs: dict[str, int] = {subject: 0 for subject in eligible_subjects}
    all_pairs: dict[str, int] = {subject: 0 for subject in eligible_subjects}
    earliest = {
        subject: row.get("earliest_available_session_timestamp", "")
        for subject, row in eligible_subjects.items()
    }

    for row in pair_rows:
        subject = row.get("subject_id", "")
        if subject not in eligible_subjects:
            continue
        if not bool_value(row.get("supports_common_10s_k5_m1", row.get("supports_10s_k5_m1"))):
            continue
        all_pairs[subject] += 1
        gap = as_float(row.get("gap_days"))
        if gap is not None:
            gaps_by_subject[subject].append(gap)
        if row.get("enrollment_session_timestamp") == earliest[subject]:
            primary_pairs[subject] += 1

    features = []
    for subject, row in sorted(eligible_subjects.items()):
        gaps = gaps_by_subject[subject]
        features.append(
            {
                "subject_id": subject,
                "session_count": int(float(row.get("successful_preprocessed_sessions", 0) or 0)),
                "total_common_available_windows": int(float(row.get("total_common_available_windows", row.get("total_model_input_available_windows", 0)) or 0)),
                "primary_pair_count": primary_pairs[subject],
                "all_pair_count": all_pairs[subject],
                "min_gap_days": min(gaps) if gaps else "",
                "median_gap_days": median(gaps),
                "max_gap_days": max(gaps) if gaps else "",
            }
        )
    return features


def stratified_subject_split(features: list[dict[str, Any]], config: dict[str, Any]) -> list[dict[str, Any]]:
    """Assign subjects to splits with deterministic ranked round-robin balancing."""

    expected = {split: int(config["expected_subject_counts"][split]) for split in SPLITS}
    if sum(expected.values()) != len(features):
        raise SystemExit(
            f"Expected subject counts sum to {sum(expected.values())}, but eligible subjects={len(features)}"
        )

    seed = int(config["split_seed"])
    rng = random.Random(seed)
    rows = [dict(item) for item in features]
    for row in rows:
        row["_jitter"] = rng.random()
    rows.sort(
        key=lambda row: (
            -int(row["primary_pair_count"]),
            -int(row["all_pair_count"]),
            -int(row["session_count"]),
            -int(row["total_common_available_windows"]),
            row["_jitter"],
        )
    )

    assigned: dict[str, list[dict[str, Any]]] = {split: [] for split in SPLITS}

    for row in rows:
        open_splits = [split for split in SPLITS if len(assigned[split]) < expected[split]]
        if not open_splits:
            raise RuntimeError("No split capacity left.")
        best_split = min(
            open_splits,
            key=lambda split: (
                len(assigned[split]) / expected[split],
                {"train": 0, "val": 1, "test": 2}[split],
            ),
        )
        row["split"] = best_split
        assigned[best_split].append(row)

    output = []
    for split in SPLITS:
        for row in sorted(assigned[split], key=lambda item: item["subject_id"]):
            clean = {key: value for key, value in row.items() if not key.startswith("_")}
            output.append(clean)
    return output


def split_summary(rows: list[dict[str, Any]], config: dict[str, Any]) -> dict[str, Any]:
    """Summarize split balance and subject-disjointness."""

    by_split = {split: [row for row in rows if row["split"] == split] for split in SPLITS}
    subject_sets = {split: {row["subject_id"] for row in items} for split, items in by_split.items()}
    overlaps = {
        f"{a}_{b}": sorted(subject_sets[a] & subject_sets[b])
        for index, a in enumerate(SPLITS)
        for b in SPLITS[index + 1 :]
    }
    per_split = {}
    for split, items in by_split.items():
        per_split[split] = {
            "subject_count": len(items),
            "session_count": numeric_summary(row["session_count"] for row in items),
            "common_window_count": numeric_summary(row["total_common_available_windows"] for row in items),
            "primary_pair_count": numeric_summary(row["primary_pair_count"] for row in items),
            "all_pair_count": numeric_summary(row["all_pair_count"] for row in items),
            "min_gap_days": numeric_summary(row["min_gap_days"] for row in items),
            "median_gap_days": numeric_summary(row["median_gap_days"] for row in items),
            "max_gap_days": numeric_summary(row["max_gap_days"] for row in items),
        }
    return {
        "protocol_id": config["protocol_id"],
        "input_protocol_id": config["input_protocol_id"],
        "split_seed": config["split_seed"],
        "generated_datetime_utc": utc_now_iso(),
        "eligible_subject_count": len(rows),
        "split_subject_counts": {split: len(items) for split, items in by_split.items()},
        "expected_subject_counts": config["expected_subject_counts"],
        "subject_overlap_check": {
            "train_val_overlap": len(overlaps["train_val"]),
            "train_test_overlap": len(overlaps["train_test"]),
            "val_test_overlap": len(overlaps["val_test"]),
            "overlap_subjects": overlaps,
            "passed": all(len(items) == 0 for items in overlaps.values()),
        },
        "split_distribution_summary": per_split,
        "morphology_validity_used_for_protocol": config.get("morphology_validity_used_for_protocol", False),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build SigD subject-disjoint split.")
    parser.add_argument("--root", default=None)
    parser.add_argument("--config", default=None)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = detect_root(args.root)
    config = load_config(root, args.config)
    snapshot = resolve(root, config["preprocessing_snapshot_dir"])
    subject_rows = read_csv_rows(snapshot / "postqc_subject_summary_10s.csv")
    pair_rows = read_csv_rows(snapshot / "postqc_interval_pairs_10s.csv")
    features = subject_features(config, subject_rows, pair_rows)
    split_rows = stratified_subject_split(features, config)

    out_dir = root / "protocol" / "SigD" / "metadata"
    seed = config["split_seed"]
    split_path = out_dir / f"subject_split_seed{seed}.csv"
    summary_path = out_dir / f"subject_split_summary_seed{seed}.json"
    write_csv(split_path, split_rows, SPLIT_COLUMNS, args.overwrite)
    write_json(summary_path, split_summary(split_rows, config), args.overwrite)
    print(
        "subject_split "
        + " ".join(f"{split}={sum(1 for row in split_rows if row['split'] == split)}" for split in SPLITS)
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
