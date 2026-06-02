#!/usr/bin/env python3
"""Audit reconstructed SigD-Core raw ranges without preprocessing or SQI."""

from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import statistics
from collections import Counter, defaultdict
from datetime import datetime, timezone
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np

from parse_sigd_annotations import (
    DATASET_NAME,
    DATASET_VERSION,
    detect_root,
    load_config,
    setup_logging,
    sigd_dir,
)


AVAILABLE_STATUSES = {"success", "skipped_existing"}

DATASET_AUDIT_BASE_COLUMNS = [
    "raw_range_id",
    "subject_id",
    "session_timestamp",
    "extraction_status",
    "failure_reason",
    "output_npz_path",
    "npz_exists",
    "fs",
    "requested_duration_seconds",
    "extracted_duration_seconds",
    "extracted_samples",
    "nan_ratio",
    "inf_count",
    "flatline_ratio_raw",
]

SUBJECT_SUMMARY_BASE_COLUMNS = [
    "subject_id",
    "successful_sessions",
    "successful_raw_ranges",
    "total_duration_seconds",
    "has_at_least_2_success_sessions",
    "earliest_session_timestamp",
    "latest_session_timestamp",
    "max_gap_days",
]

SESSION_SUMMARY_BASE_COLUMNS = [
    "subject_id",
    "session_timestamp",
    "successful_raw_ranges",
    "total_duration_seconds",
]

INTERVAL_PAIR_BASE_COLUMNS = [
    "subject_id",
    "enrollment_session_timestamp",
    "probe_session_timestamp",
    "gap_hours",
    "gap_days",
    "enrollment_success_raw_ranges",
    "probe_success_raw_ranges",
    "enrollment_total_duration_seconds",
    "probe_total_duration_seconds",
    "supports_raw_cross_session_pair",
    "supports_any_candidate_cross_session_protocol",
    "supports_future_cross_session_verification",
]


def dataset_audit_columns(window_lengths: list[int]) -> list[str]:
    """Return dynamic raw-range audit columns for configured window lengths."""

    return (
        DATASET_AUDIT_BASE_COLUMNS
        + [f"possible_nonoverlap_windows_{length}s" for length in window_lengths]
        + [f"supports_{length}s_window" for length in window_lengths]
    )


def subject_summary_columns(window_lengths: list[int]) -> list[str]:
    """Return dynamic subject summary columns for configured window lengths."""

    return (
        SUBJECT_SUMMARY_BASE_COLUMNS[:4]
        + [f"possible_windows_{length}s" for length in window_lengths]
        + [SUBJECT_SUMMARY_BASE_COLUMNS[4]]
        + [
            f"eligible_for_future_{length}s_cross_session_protocol"
            for length in window_lengths
        ]
        + SUBJECT_SUMMARY_BASE_COLUMNS[5:]
    )


def session_summary_columns(window_lengths: list[int]) -> list[str]:
    """Return dynamic session summary columns for configured window lengths."""

    return (
        SESSION_SUMMARY_BASE_COLUMNS
        + [f"possible_windows_{length}s" for length in window_lengths]
        + [f"supports_{length}s_window" for length in window_lengths]
    )


def interval_pair_columns(window_lengths: list[int]) -> list[str]:
    """Return dynamic interval pair columns for configured window lengths."""

    return (
        INTERVAL_PAIR_BASE_COLUMNS[:-2]
        + [
            f"supports_future_{length}s_cross_session_protocol"
            for length in window_lengths
        ]
        + INTERVAL_PAIR_BASE_COLUMNS[-2:]
    )


def utc_now_iso() -> str:
    """Return a UTC ISO-8601 timestamp with seconds precision."""

    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    """Read a UTF-8 CSV file if it exists."""

    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]], columns: list[str]) -> None:
    """Write rows with fixed column order."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column, "") for column in columns})


def dump_json(path: Path, payload: dict[str, Any]) -> None:
    """Write a readable UTF-8 JSON file."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def as_float(value: Any) -> float | None:
    """Parse a float value, returning None for blanks and NaN."""

    if value in {"", None}:
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(parsed):
        return None
    return parsed


def numeric_summary(values: list[float]) -> dict[str, Any]:
    """Return compact summary statistics for numeric lists."""

    clean = [float(value) for value in values if value is not None and not math.isnan(value)]
    if not clean:
        return {"count": 0, "min": None, "median": None, "max": None, "mean": None}
    return {
        "count": len(clean),
        "min": min(clean),
        "median": statistics.median(clean),
        "max": max(clean),
        "mean": float(sum(clean) / len(clean)),
    }


def distribution(values: list[Any]) -> dict[str, int]:
    """Return a JSON-friendly distribution."""

    return {str(k): int(v) for k, v in sorted(Counter(values).items())}


def possible_windows(duration_seconds: float | None, window_seconds: int) -> int:
    """Estimate non-overlapping raw-level windows from duration only."""

    if duration_seconds is None or duration_seconds <= 0:
        return 0
    return int(duration_seconds // window_seconds)


def parse_session_datetime(session_timestamp: str) -> datetime | None:
    """Parse the public surrogate session timestamp if possible."""

    try:
        return datetime.strptime(session_timestamp, "%Y-%m-%d-%H-%M")
    except ValueError:
        return None


def load_npz_index(root: Path) -> dict[str, str]:
    """Index reconstructed NPZ files by raw_range_id without using pickle."""

    index: dict[str, str] = {}
    base = sigd_dir(root)
    for npz_path in (base / "data" / "raw_ranges").glob("**/*.npz"):
        try:
            with np.load(npz_path, allow_pickle=False) as data:
                raw_range_id = str(data["raw_range_id"])
            index[raw_range_id] = str(npz_path.relative_to(base))
        except Exception as exc:
            logging.warning("Could not inspect NPZ %s: %s", npz_path, exc)
    return index


def npz_exists_for_row(root: Path, row: dict[str, str], npz_index: dict[str, str]) -> bool:
    """Check whether the row has a corresponding NPZ on disk."""

    rel = row.get("output_npz_path", "")
    if rel and (sigd_dir(root) / rel).exists():
        return True
    return row.get("raw_range_id", "") in npz_index


def build_dataset_audit_rows(
    root: Path,
    extraction_rows: list[dict[str, str]],
    window_lengths: list[int],
) -> list[dict[str, Any]]:
    """Build raw-range audit rows with preliminary future-window counts."""

    npz_index = load_npz_index(root)
    rows: list[dict[str, Any]] = []
    for row in extraction_rows:
        duration = as_float(row.get("extracted_duration_seconds"))
        audit_row: dict[str, Any] = {
            "raw_range_id": row.get("raw_range_id", ""),
            "subject_id": row.get("subject_id", ""),
            "session_timestamp": row.get("session_timestamp", ""),
            "extraction_status": row.get("extraction_status", ""),
            "failure_reason": row.get("failure_reason", ""),
            "output_npz_path": row.get("output_npz_path", ""),
            "npz_exists": npz_exists_for_row(root, row, npz_index),
            "fs": row.get("fs", ""),
            "requested_duration_seconds": row.get("requested_duration_seconds", ""),
            "extracted_duration_seconds": row.get("extracted_duration_seconds", ""),
            "extracted_samples": row.get("extracted_samples", ""),
            "nan_ratio": row.get("nan_ratio", ""),
            "inf_count": row.get("inf_count", ""),
            "flatline_ratio_raw": row.get("flatline_ratio_raw", ""),
        }
        for length in window_lengths:
            count = possible_windows(duration, length)
            audit_row[f"possible_nonoverlap_windows_{length}s"] = count
            audit_row[f"supports_{length}s_window"] = count > 0
        rows.append(audit_row)
    return rows


def available_rows(dataset_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return rows that correspond to available reconstructed raw ranges."""

    return [
        row
        for row in dataset_rows
        if row.get("extraction_status") in AVAILABLE_STATUSES and row.get("npz_exists")
    ]


def build_session_summary(
    available: list[dict[str, Any]], window_lengths: list[int]
) -> list[dict[str, Any]]:
    """Summarize available raw ranges by subject/session."""

    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in available:
        grouped[(row["subject_id"], row["session_timestamp"])].append(row)

    summary_rows: list[dict[str, Any]] = []
    for (subject_id, session_timestamp), rows in sorted(grouped.items()):
        total_duration = sum(as_float(row["extracted_duration_seconds"]) or 0.0 for row in rows)
        item: dict[str, Any] = {
            "subject_id": subject_id,
            "session_timestamp": session_timestamp,
            "successful_raw_ranges": len(rows),
            "total_duration_seconds": total_duration,
        }
        for length in window_lengths:
            windows = sum(int(row.get(f"possible_nonoverlap_windows_{length}s", 0)) for row in rows)
            item[f"possible_windows_{length}s"] = windows
            item[f"supports_{length}s_window"] = windows > 0
        summary_rows.append(item)
    return summary_rows


def sorted_session_timestamps(timestamps: list[str]) -> list[str]:
    """Sort session timestamps chronologically when parseable, else lexically."""

    return sorted(
        timestamps,
        key=lambda value: (
            parse_session_datetime(value) is None,
            parse_session_datetime(value) or value,
        ),
    )


def build_subject_summary(
    session_rows: list[dict[str, Any]], window_lengths: list[int]
) -> list[dict[str, Any]]:
    """Summarize cross-session availability by subject."""

    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in session_rows:
        grouped[row["subject_id"]].append(row)

    subjects: list[dict[str, Any]] = []
    for subject_id, sessions in sorted(grouped.items()):
        timestamps = sorted_session_timestamps([row["session_timestamp"] for row in sessions])
        total_duration = sum(float(row["total_duration_seconds"]) for row in sessions)
        item: dict[str, Any] = {
            "subject_id": subject_id,
            "successful_sessions": len(sessions),
            "successful_raw_ranges": sum(int(row["successful_raw_ranges"]) for row in sessions),
            "total_duration_seconds": total_duration,
            "has_at_least_2_success_sessions": len(sessions) >= 2,
            "earliest_session_timestamp": timestamps[0] if timestamps else "",
            "latest_session_timestamp": timestamps[-1] if timestamps else "",
            "max_gap_days": "",
        }
        parsed = [parse_session_datetime(value) for value in timestamps]
        parsed_clean = [value for value in parsed if value is not None]
        if len(parsed_clean) >= 2:
            item["max_gap_days"] = (
                max(parsed_clean) - min(parsed_clean)
            ).total_seconds() / 86400.0
        for length in window_lengths:
            windows = sum(int(row.get(f"possible_windows_{length}s", 0)) for row in sessions)
            item[f"possible_windows_{length}s"] = windows
            sessions_with_windows = sum(
                1 for row in sessions if int(row.get(f"possible_windows_{length}s", 0)) > 0
            )
            item[f"eligible_for_future_{length}s_cross_session_protocol"] = (
                sessions_with_windows >= 2
            )
        subjects.append(item)
    return subjects


def build_interval_pairs(
    session_rows: list[dict[str, Any]],
    window_lengths: list[int],
    primary_window_length: int = 10,
) -> list[dict[str, Any]]:
    """Build all same-subject successful session pairs for gap auditing."""

    if primary_window_length not in window_lengths:
        raise ValueError(
            f"primary window length {primary_window_length}s is not in candidate window lengths"
        )

    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in session_rows:
        grouped[row["subject_id"]].append(row)

    pairs: list[dict[str, Any]] = []
    for subject_id, sessions in sorted(grouped.items()):
        sorted_sessions = sorted(
            sessions,
            key=lambda row: (
                parse_session_datetime(row["session_timestamp"]) is None,
                parse_session_datetime(row["session_timestamp"]) or row["session_timestamp"],
            ),
        )
        for enrollment, probe in combinations(sorted_sessions, 2):
            enrollment_ts = enrollment["session_timestamp"]
            probe_ts = probe["session_timestamp"]
            enrollment_dt = parse_session_datetime(enrollment_ts)
            probe_dt = parse_session_datetime(probe_ts)
            gap_hours = ""
            gap_days = ""
            if enrollment_dt is not None and probe_dt is not None:
                seconds = (probe_dt - enrollment_dt).total_seconds()
                gap_hours = seconds / 3600.0
                gap_days = seconds / 86400.0
            future_flags = {
                f"supports_future_{length}s_cross_session_protocol": (
                    int(enrollment.get(f"possible_windows_{length}s", 0)) > 0
                    and int(probe.get(f"possible_windows_{length}s", 0)) > 0
                )
                for length in window_lengths
            }
            pairs.append(
                {
                    "subject_id": subject_id,
                    "enrollment_session_timestamp": enrollment_ts,
                    "probe_session_timestamp": probe_ts,
                    "gap_hours": gap_hours,
                    "gap_days": gap_days,
                    "enrollment_success_raw_ranges": enrollment["successful_raw_ranges"],
                    "probe_success_raw_ranges": probe["successful_raw_ranges"],
                    "enrollment_total_duration_seconds": enrollment[
                        "total_duration_seconds"
                    ],
                    "probe_total_duration_seconds": probe["total_duration_seconds"],
                    "supports_raw_cross_session_pair": True,
                    **future_flags,
                    "supports_any_candidate_cross_session_protocol": any(
                        future_flags.values()
                    ),
                    "supports_future_cross_session_verification": future_flags[
                        f"supports_future_{primary_window_length}s_cross_session_protocol"
                    ],
                }
            )
    return pairs


def build_annotation_summary(annotation_rows: list[dict[str, str]]) -> dict[str, Any]:
    """Compute annotation-level audit statistics from the annotation manifest."""

    subjects = {row["subject_id"] for row in annotation_rows}
    sessions = {(row["subject_id"], row["session_timestamp"]) for row in annotation_rows}
    success = [row for row in annotation_rows if row.get("annotation_parse_status") == "success"]
    failures = [row for row in annotation_rows if row.get("annotation_parse_status") != "success"]
    by_subject: dict[str, set[str]] = defaultdict(set)
    for row in annotation_rows:
        by_subject[row["subject_id"]].add(row["session_timestamp"])
    durations = [
        as_float(row.get("requested_duration_seconds")) or 0.0
        for row in success
        if as_float(row.get("requested_duration_seconds")) is not None
    ]
    return {
        "annotation_subjects": len(subjects),
        "annotation_sessions": len(sessions),
        "annotation_raw_ranges": len(annotation_rows),
        "annotation_requested_total_duration_seconds": sum(durations),
        "annotation_parsing_success_count": len(success),
        "annotation_parsing_failure_count": len(failures),
        "annotation_subjects_with_at_least_2_sessions": sum(
            1 for subject_sessions in by_subject.values() if len(subject_sessions) >= 2
        ),
    }


def build_extraction_summary(
    extraction_rows: list[dict[str, str]], available: list[dict[str, Any]]
) -> dict[str, Any]:
    """Compute extraction-level counts and failure distributions."""

    failed = [row for row in extraction_rows if row.get("extraction_status") == "failed"]
    skipped = [
        row for row in extraction_rows if row.get("extraction_status") == "skipped_existing"
    ]
    return {
        "successful_subjects_with_available_npz": len({row["subject_id"] for row in available}),
        "successful_sessions_with_available_npz": len(
            {(row["subject_id"], row["session_timestamp"]) for row in available}
        ),
        "successful_raw_ranges_with_available_npz": len(available),
        "failed_raw_ranges": len(failed),
        "skipped_existing_raw_ranges": len(skipped),
        "failure_reason_distribution": distribution(
            [row.get("failure_reason") or "none" for row in failed]
        ),
        "total_successful_waveform_duration_seconds": sum(
            as_float(row.get("extracted_duration_seconds")) or 0.0 for row in available
        ),
    }


def build_raw_integrity_summary(available: list[dict[str, Any]]) -> dict[str, Any]:
    """Summarize raw integrity fields allowed in reconstruction audit."""

    return {
        "fs_distribution": distribution([row.get("fs", "") for row in available]),
        "nan_ratio": numeric_summary(
            [as_float(row.get("nan_ratio")) for row in available if as_float(row.get("nan_ratio")) is not None]
        ),
        "inf_count": numeric_summary(
            [as_float(row.get("inf_count")) for row in available if as_float(row.get("inf_count")) is not None]
        ),
        "flatline_ratio_raw": numeric_summary(
            [
                as_float(row.get("flatline_ratio_raw"))
                for row in available
                if as_float(row.get("flatline_ratio_raw")) is not None
            ]
        ),
        "raw_range_duration_seconds": numeric_summary(
            [
                as_float(row.get("extracted_duration_seconds"))
                for row in available
                if as_float(row.get("extracted_duration_seconds")) is not None
            ]
        ),
    }


def build_cross_session_summary(
    subject_rows: list[dict[str, Any]], interval_pairs: list[dict[str, Any]]
) -> dict[str, Any]:
    """Summarize cross-session availability."""

    subjects_two_sessions = [
        row for row in subject_rows if row.get("has_at_least_2_success_sessions")
    ]
    earliest_later_pairs = sum(
        max(int(row.get("successful_sessions", 0)) - 1, 0) for row in subject_rows
    )
    return {
        "subjects_with_at_least_2_success_sessions": len(subjects_two_sessions),
        "subjects_with_earliest_successful_session": len(subject_rows),
        "earliest_session_to_later_session_pair_count": earliest_later_pairs,
        "all_same_subject_successful_session_pair_count": len(interval_pairs),
        "session_timestamps_are_public_surrogate_timestamps": True,
    }


def build_window_summary(
    dataset_rows: list[dict[str, Any]],
    session_rows: list[dict[str, Any]],
    subject_rows: list[dict[str, Any]],
    window_lengths: list[int],
) -> dict[str, Any]:
    """Summarize preliminary future-window availability estimates."""

    payload: dict[str, Any] = {
        "interpretation": (
            "Raw-level non-overlapping window counts are preliminary upper-bound "
            "availability estimates; preprocessing may change usable counts."
        )
    }
    for length in window_lengths:
        payload[f"{length}s"] = {
            "raw_ranges_supporting_window": sum(
                1 for row in dataset_rows if row.get(f"supports_{length}s_window") is True
            ),
            "total_possible_nonoverlap_windows": sum(
                int(row.get(f"possible_nonoverlap_windows_{length}s", 0))
                for row in dataset_rows
            ),
            "sessions_supporting_window": sum(
                1 for row in session_rows if row.get(f"supports_{length}s_window") is True
            ),
            "subjects_eligible_for_future_cross_session_protocol": sum(
                1
                for row in subject_rows
                if row.get(f"eligible_for_future_{length}s_cross_session_protocol") is True
            ),
        }
    return payload


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(
        description="Audit SigD-Core reconstructed raw ranges."
    )
    parser.add_argument("--root", type=str, default=None, help="sim_ppg root path")
    parser.add_argument("--config", type=str, default=None, help="config YAML path")
    parser.add_argument("--verbose", action="store_true", help="Enable debug logging")
    return parser.parse_args()


def main() -> int:
    """CLI entry point."""

    args = parse_args()
    root = detect_root(args.root)
    setup_logging(root, "audit_sigd_core.log", args.verbose)
    config = load_config(root)
    if args.config:
        config_path = Path(args.config).expanduser().resolve()
        if config_path.exists():
            import yaml

            with config_path.open("r", encoding="utf-8") as handle:
                config.update(yaml.safe_load(handle) or {})

    window_lengths = [int(value) for value in config["candidate_window_lengths_seconds"]]
    primary_window_length = int(config.get("primary_future_window_length_seconds") or 10)
    annotation_rows = read_csv_rows(sigd_dir(root) / "metadata" / "sigd_annotation_manifest.csv")
    extraction_rows = read_csv_rows(sigd_dir(root) / "metadata" / "sigd_extraction_manifest.csv")

    dataset_rows = build_dataset_audit_rows(root, extraction_rows, window_lengths)
    available = available_rows(dataset_rows)
    session_rows = build_session_summary(available, window_lengths)
    subject_rows = build_subject_summary(session_rows, window_lengths)
    interval_pairs = build_interval_pairs(
        session_rows, window_lengths, primary_window_length
    )

    write_csv(
        sigd_dir(root) / "metadata" / "sigd_core_dataset_audit.csv",
        dataset_rows,
        dataset_audit_columns(window_lengths),
    )
    write_csv(
        sigd_dir(root) / "metadata" / "sigd_core_subject_summary.csv",
        subject_rows,
        subject_summary_columns(window_lengths),
    )
    write_csv(
        sigd_dir(root) / "metadata" / "sigd_core_session_summary.csv",
        session_rows,
        session_summary_columns(window_lengths),
    )
    write_csv(
        sigd_dir(root) / "metadata" / "sigd_core_interval_pairs.csv",
        interval_pairs,
        interval_pair_columns(window_lengths),
    )

    summary = {
        "dataset_name": DATASET_NAME,
        "dataset_version": DATASET_VERSION,
        "generated_datetime_utc": utc_now_iso(),
        "research_scope": (
            "SigD-Core (waveform-only public reconstruction) for cross-session "
            "PPG verification raw waveform acquisition and audit only."
        ),
        "clinical_metadata_used": False,
        "demographic_analysis_supported": False,
        "source_manifest_path": "metadata/source_manifest.json",
        "annotation_summary": build_annotation_summary(annotation_rows),
        "extraction_summary": build_extraction_summary(extraction_rows, available),
        "raw_integrity_summary": build_raw_integrity_summary(available),
        "cross_session_availability_summary": build_cross_session_summary(
            subject_rows, interval_pairs
        ),
        "future_window_availability_summary": build_window_summary(
            dataset_rows, session_rows, subject_rows, window_lengths
        ),
        "limitations": [
            "waveform_only_public_reconstruction",
            "does_not_reproduce_original_demographic_cohort",
            "surrogate_timestamps_only",
            "clinical_sensor_context_not_consumer_wearable_context",
            "window_counts_are_preprocessing_preliminary_estimates",
        ],
        "recommended_next_stage": [
            "separate preprocessing pipeline",
            "filtering/resampling/window generation",
            "SQI/morphology calculation",
            "subject-disjoint split generation",
            "SIM-PPG model implementation",
        ],
    }
    dump_json(sigd_dir(root) / "metadata" / "sigd_core_audit_summary.json", summary)

    logging.info(
        "Audit complete: available_raw_ranges=%d available_subjects=%d interval_pairs=%d",
        len(available),
        len(subject_rows),
        len(interval_pairs),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
