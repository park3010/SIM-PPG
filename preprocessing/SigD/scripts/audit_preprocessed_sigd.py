#!/usr/bin/env python3
"""Audit post-preprocessing 10s SigD window availability."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

SCRIPT_DIR = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SCRIPT_DIR))

from common import detect_root, load_config, preprocessing_dir, read_csv_rows, read_json, resolve_path, setup_logging, write_csv, write_json  # noqa: E402
from window_audit import interval_pair_rows, postqc_summary, session_summary_rows, subject_summary_rows  # noqa: E402


SESSION_COLUMNS = [
    "subject_id",
    "session_timestamp",
    "total_candidate_windows",
    "common_available_windows",
    "model_input_available_windows",
    "excluded_windows",
    "svri_valid_windows",
    "sqi_valid_windows",
    "ipa_valid_windows",
    "supports_common_10s_k1",
    "supports_common_10s_k3",
    "supports_common_10s_k5",
    "supports_10s_k1",
    "supports_10s_k3",
    "supports_10s_k5",
]

SUBJECT_COLUMNS = [
    "subject_id",
    "successful_preprocessed_sessions",
    "total_common_available_windows",
    "total_model_input_available_windows",
    "has_at_least_2_common_sessions_k1",
    "has_at_least_2_common_sessions_k3",
    "has_at_least_2_common_sessions_k5",
    "has_at_least_2_sessions_k1",
    "has_at_least_2_sessions_k3",
    "has_at_least_2_sessions_k5",
    "earliest_available_session_timestamp",
    "latest_available_session_timestamp",
    "max_gap_days",
    "raw_level_10s_eligible",
    "retained_after_preprocessing_k1",
    "retained_after_preprocessing_k3",
    "retained_after_preprocessing_k5",
]

PAIR_COLUMNS = [
    "subject_id",
    "enrollment_session_timestamp",
    "probe_session_timestamp",
    "gap_hours",
    "gap_days",
    "enrollment_common_available_windows",
    "probe_common_available_windows",
    "enrollment_available_windows",
    "probe_available_windows",
    "supports_common_10s_k1_m1",
    "supports_common_10s_k3_m1",
    "supports_common_10s_k5_m1",
    "supports_common_10s_k3_m3",
    "supports_common_10s_k5_m5",
    "supports_10s_k1_m1",
    "supports_10s_k3_m1",
    "supports_10s_k5_m1",
    "supports_10s_k3_m3",
    "supports_10s_k5_m5",
    "raw_level_pair_was_eligible",
    "retained_after_preprocessing",
]


def paths(root: Path, config: dict, window_seconds: int, smoke: bool) -> dict[str, Path]:
    """Return audit input/output paths."""

    if smoke:
        return {
            "manifest": preprocessing_dir(root) / f"metadata/preprocessing_manifest_{window_seconds}s_smoke.csv",
            "session": preprocessing_dir(root) / f"metadata/postqc_session_summary_{window_seconds}s_smoke.csv",
            "subject": preprocessing_dir(root) / f"metadata/postqc_subject_summary_{window_seconds}s_smoke.csv",
            "pairs": preprocessing_dir(root) / f"metadata/postqc_interval_pairs_{window_seconds}s_smoke.csv",
            "summary": preprocessing_dir(root) / f"metadata/postqc_summary_{window_seconds}s_smoke.json",
        }
    return {
        "manifest": resolve_path(root, config["output"]["preprocessing_manifest_path"]),
        "session": resolve_path(root, config["output"]["postqc_session_summary_path"]),
        "subject": resolve_path(root, config["output"]["postqc_subject_summary_path"]),
        "pairs": resolve_path(root, config["output"]["postqc_interval_pairs_path"]),
        "summary": resolve_path(root, config["output"]["postqc_summary_path"]),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit preprocessed SigD windows.")
    parser.add_argument("--root", type=str, default=None)
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--window-seconds", type=int, default=10)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = detect_root(args.root)
    setup_logging(root, "audit_preprocessed_sigd.log", args.verbose)
    config = load_config(root, args.config)
    out = paths(root, config, args.window_seconds, args.smoke)
    manifest_rows = read_csv_rows(out["manifest"])
    raw_subject_summary = read_csv_rows(resolve_path(root, config["input"]["subject_summary"]))
    raw_interval_pairs = read_csv_rows(resolve_path(root, config["input"]["interval_pairs"]))
    raw_audit_summary = read_json(resolve_path(root, config["input"]["snapshot_dir"]) / "sigd_core_audit_summary.json")

    sessions = session_summary_rows(manifest_rows)
    subjects = subject_summary_rows(sessions, raw_subject_summary)
    pairs = interval_pair_rows(sessions, raw_interval_pairs)
    summary = postqc_summary(manifest_rows, sessions, subjects, pairs, raw_audit_summary, config)

    write_csv(out["session"], sessions, SESSION_COLUMNS)
    write_csv(out["subject"], subjects, SUBJECT_COLUMNS)
    write_csv(out["pairs"], pairs, PAIR_COLUMNS)
    write_json(out["summary"], summary)
    print(
        "postqc_windows={common_input_available_windows} subjects_k1={postqc_subjects_k1} "
        "pairs_k1_m1={postqc_all_session_pairs_k1_m1}".format(**summary)
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
