#!/usr/bin/env python3
"""Preprocess SigD-Core raw ranges into common standardized PPG windows."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
import sys
from typing import Any

import numpy as np
from tqdm import tqdm

SCRIPT_DIR = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SCRIPT_DIR))

from common import (  # noqa: E402
    as_float,
    detect_root,
    distribution,
    load_config,
    numeric_summary,
    preprocessing_dir,
    read_csv_rows,
    resolve_path,
    setup_logging,
    sha256_file,
    sha256_jsonable,
    utc_now_iso,
    write_csv,
    write_json,
)
from signal_processing import process_raw_range  # noqa: E402
from snapshot_validation import available_extraction_rows, raw_npz_path, select_extraction_rows, validate_snapshot  # noqa: E402


MANIFEST_COLUMNS = [
    "window_id",
    "parent_raw_range_id",
    "subject_id",
    "session_timestamp",
    "session_index_within_subject",
    "raw_range_index_within_session",
    "raw_npz_path",
    "raw_npz_sha256",
    "preprocessing_version",
    "input_protocol_id",
    "comparison_role",
    "preprocessing_profile",
    "native_or_common_input",
    "normalization_policy",
    "preprocessing_config_sha256",
    "input_snapshot_sha256_reference",
    "window_index_within_raw_range",
    "window_start_sample_in_raw_range",
    "window_end_sample_in_raw_range",
    "window_start_seconds_in_raw_range",
    "window_end_seconds_in_raw_range",
    "source_fs",
    "target_fs",
    "window_seconds",
    "expected_output_samples",
    "actual_output_samples",
    "resampled",
    "trailing_remainder_samples_in_parent_range",
    "raw_range_processing_status",
    "window_processing_status",
    "model_input_available",
    "common_input_available",
    "exclusion_reason",
    "array_index",
    "original_window_nonfinite_count",
    "original_window_nonfinite_ratio",
    "original_window_interpolated",
    "raw_flatline_ratio_window",
    "filtered_flatline_ratio_window",
    "filtered_mean",
    "filtered_std",
    "filtered_min",
    "filtered_max",
    "filtered_has_nonfinite",
    "svri",
    "svri_valid_mask",
    "svri_failure_reason",
    "sqi_skewness",
    "sqi_valid_mask",
    "sqi_failure_reason",
    "ipa",
    "ipa_valid_mask",
    "ipa_failure_reason",
    "aux_morphology_annotation_available",
    "aux_morphology_any_available",
    "aux_morphology_all_available",
    "raw_subject_eligible_for_10s_cross_session_protocol",
    "raw_parent_session_supports_10s_protocol",
]


def output_paths(root: Path, config: dict[str, Any], window_seconds: int, smoke: bool) -> dict[str, Path]:
    """Return mode-specific preprocessing output paths."""

    if smoke:
        return {
            "array": preprocessing_dir(root)
            / f"data/windows_{window_seconds}s_smoke/ppg_filtered_windows_{window_seconds}s_125hz.npy",
            "manifest": preprocessing_dir(root)
            / f"metadata/preprocessing_manifest_{window_seconds}s_smoke.csv",
            "summary": preprocessing_dir(root)
            / f"metadata/preprocessing_summary_{window_seconds}s_smoke.json",
        }
    return {
        "array": resolve_path(root, config["output"]["windows_array_path"]),
        "manifest": resolve_path(root, config["output"]["preprocessing_manifest_path"]),
        "summary": resolve_path(root, config["output"]["summary_path"]),
    }


def load_raw_subject_eligibility(root: Path, config: dict[str, Any]) -> dict[str, bool]:
    """Load raw-level 10s subject eligibility from snapshot subject summary."""

    rows = read_csv_rows(resolve_path(root, config["input"]["subject_summary"]))
    return {
        row["subject_id"]: str(row.get("eligible_for_future_10s_cross_session_protocol", "")).lower()
        == "true"
        for row in rows
    }


def session_supports_10s(rows: list[dict[str, str]]) -> set[tuple[str, str]]:
    """Derive raw parent sessions with at least one 10s raw candidate."""

    supported = set()
    for row in rows:
        duration = as_float(row.get("extracted_duration_seconds")) or 0.0
        if duration >= 10.0:
            supported.add((row["subject_id"], row["session_timestamp"]))
    return supported


def failed_raw_range_row(manifest_row: dict[str, str], reason: str, config: dict[str, Any], config_hash: str, snapshot_ref: str) -> dict[str, Any]:
    """Create a manifest row for a raw range that produced no candidate windows."""

    return {
        "window_id": f"{manifest_row['raw_range_id']}_raw_range_failed",
        "parent_raw_range_id": manifest_row["raw_range_id"],
        "subject_id": manifest_row["subject_id"],
        "session_timestamp": manifest_row["session_timestamp"],
        "session_index_within_subject": manifest_row.get("session_index_within_subject", ""),
        "raw_range_index_within_session": manifest_row.get("annotation_range_index_within_session", ""),
        "preprocessing_version": config["preprocessing_version"],
        "input_protocol_id": config["input_protocol_id"],
        "comparison_role": config["comparison_role"],
        "preprocessing_profile": config["preprocessing_profile"],
        "native_or_common_input": config["native_or_common_input"],
        "normalization_policy": config["normalization_policy"],
        "preprocessing_config_sha256": config_hash,
        "input_snapshot_sha256_reference": snapshot_ref,
        "raw_range_processing_status": "failed",
        "window_processing_status": "failed",
        "model_input_available": False,
        "common_input_available": False,
        "exclusion_reason": reason,
        "aux_morphology_annotation_available": False,
        "aux_morphology_any_available": False,
        "aux_morphology_all_available": False,
    }


def summary_from_rows(config: dict[str, Any], rows: list[dict[str, Any]], raw_summaries: list[dict[str, Any]], array: np.ndarray, window_seconds: int, snapshot_validation_path: Path) -> dict[str, Any]:
    """Build preprocessing summary JSON."""

    available_rows = [row for row in rows if row.get("common_input_available") is True]
    excluded = [row for row in rows if row.get("common_input_available") is not True]
    return {
        "preprocessing_name": config["preprocessing_name"],
        "preprocessing_version": config["preprocessing_version"],
        "input_protocol_id": config["input_protocol_id"],
        "comparison_role": config["comparison_role"],
        "preprocessing_profile": config["preprocessing_profile"],
        "native_or_common_input": config["native_or_common_input"],
        "normalization_policy": config["normalization_policy"],
        "applicable_primary_models": config["applicable_primary_models"],
        "native_input_outputs_generated": config["native_input_outputs_generated"],
        "input_dataset_name": config["input"]["dataset_name"],
        "input_dataset_version": "waveform_only_public_reconstruction_v1",
        "input_snapshot_validation_path": str(snapshot_validation_path),
        "papagei_reference_manifest_path": "preprocessing/SigD/metadata/papagei_reference_manifest.json",
        "generated_datetime_utc": utc_now_iso(),
        "window_seconds": window_seconds,
        "target_fs": config["signal"]["target_fs_hz"],
        "filter_parameters": {
            "backend": config["papagei_alignment"]["filter_backend"],
            "lowcut_hz": config["papagei_alignment"]["filter_lowcut_hz"],
            "highcut_hz": config["papagei_alignment"]["filter_highcut_hz"],
            "order": config["papagei_alignment"]["filter_order"],
        },
        "normalization_deferred_to_model_loader": True,
        "available_input_raw_ranges": len(raw_summaries),
        "processed_raw_ranges": sum(1 for item in raw_summaries if item.get("status") == "success"),
        "failed_raw_ranges_during_preprocessing": sum(1 for item in raw_summaries if item.get("status") != "success"),
        "total_candidate_windows": len(rows),
        "model_input_available_windows": len(available_rows),
        "common_input_available_windows": len(available_rows),
        "excluded_windows": len(excluded),
        "exclusion_reason_distribution": distribution(row.get("exclusion_reason", "") or "none" for row in excluded),
        "interpolation_applied_raw_ranges": sum(1 for item in raw_summaries if item.get("interpolated")),
        "interpolation_affected_windows": sum(1 for row in rows if row.get("original_window_interpolated") is True),
        "resampled_windows": sum(1 for row in rows if row.get("resampled") is True),
        "morphology_validity": {
            "aux_morphology_annotation_available_count": sum(
                1 for row in rows if row.get("aux_morphology_annotation_available") is True
            ),
            "aux_morphology_any_available_count": sum(
                1 for row in rows if row.get("aux_morphology_any_available") is True
            ),
            "aux_morphology_all_available_count": sum(
                1 for row in rows if row.get("aux_morphology_all_available") is True
            ),
            "svri_valid_count": sum(1 for row in rows if row.get("svri_valid_mask") is True),
            "sqi_valid_count": sum(1 for row in rows if row.get("sqi_valid_mask") is True),
            "ipa_valid_count": sum(1 for row in rows if row.get("ipa_valid_mask") is True),
            "ipa_invalid_reason_distribution": distribution(
                row.get("ipa_failure_reason", "") or "none"
                for row in rows
                if row.get("ipa_valid_mask") is not True
            ),
        },
        "quality_statistics": {
            "original_window_nonfinite_ratio": numeric_summary(row.get("original_window_nonfinite_ratio") for row in rows),
            "raw_flatline_ratio_window": numeric_summary(row.get("raw_flatline_ratio_window") for row in rows),
            "filtered_flatline_ratio_window": numeric_summary(row.get("filtered_flatline_ratio_window") for row in rows),
            "sqi_skewness": numeric_summary(row.get("sqi_skewness") for row in rows if row.get("sqi_valid_mask") is True),
        },
        "array_shape": list(array.shape),
        "array_dtype": str(array.dtype),
        "limitations": [
            "common_input_protocol_not_model_native_input",
            "quality_threshold_not_applied",
            "normalization_deferred_to_model_loader",
            "morphology_targets_are_auxiliary_annotations_not_common_eligibility_filters",
            "postqc_protocol_eligibility_must_be_recomputed",
        ],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Preprocess SigD-Core raw PPG ranges into windows.")
    parser.add_argument("--root", type=str, default=None)
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--window-seconds", type=int, default=10)
    parser.add_argument("--limit-subjects", type=int, default=None)
    parser.add_argument("--limit-raw-ranges", type=int, default=None)
    parser.add_argument("--subject-id", type=str, default=None)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--verify-selected-npz-hashes", action="store_true")
    parser.add_argument("--verify-all-npz-hashes", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = detect_root(args.root)
    setup_logging(root, "preprocess_sigd_windows.log", args.verbose)
    config = load_config(root, args.config)
    if args.window_seconds not in [int(x) for x in config["windowing"]["supported_window_seconds"]]:
        raise SystemExit(f"Unsupported window length: {args.window_seconds}")
    if not args.smoke and not args.verify_all_npz_hashes:
        raise SystemExit("Full preprocessing requires --verify-all-npz-hashes.")

    paths = output_paths(root, config, args.window_seconds, args.smoke)
    if paths["array"].exists() and not args.overwrite and not args.smoke:
        raise SystemExit(f"Output exists; pass --overwrite: {paths['array']}")

    extraction_rows = read_csv_rows(resolve_path(root, config["input"]["extraction_manifest"]))
    available_rows = available_extraction_rows(config, extraction_rows)
    selected_rows = available_rows
    if args.subject_id:
        selected_rows = [row for row in selected_rows if row["subject_id"] == args.subject_id]
    if args.limit_subjects is not None:
        allowed = []
        for row in selected_rows:
            if row["subject_id"] not in allowed:
                allowed.append(row["subject_id"])
            if len(allowed) >= args.limit_subjects:
                break
        selected_rows = [row for row in selected_rows if row["subject_id"] in set(allowed)]
    if args.smoke and args.limit_raw_ranges is None:
        selected_rows = select_extraction_rows(selected_rows, None, 2)
    elif args.limit_raw_ranges is not None:
        selected_rows = selected_rows[: args.limit_raw_ranges]

    snapshot_validation_path = preprocessing_dir(root) / "metadata" / "input_snapshot_validation.json"
    validation = validate_snapshot(
        root,
        config,
        limit_raw_ranges=len(selected_rows) if args.smoke else None,
        subject_id=args.subject_id,
        verify_selected_npz_hashes=args.verify_selected_npz_hashes or args.smoke,
        verify_all_npz_hashes=args.verify_all_npz_hashes,
        output_path=snapshot_validation_path,
    )
    if not validation["snapshot_valid"]:
        raise SystemExit("Snapshot validation failed.")

    config_hash = sha256_jsonable(config)
    snapshot_ref = sha256_file(resolve_path(root, config["input"]["snapshot_sha256_file"]))
    raw_subject_eligible = load_raw_subject_eligibility(root, config)
    session_10s = session_supports_10s(available_rows)

    all_rows: list[dict[str, Any]] = []
    all_arrays: list[np.ndarray] = []
    raw_summaries: list[dict[str, Any]] = []
    for row in tqdm(selected_rows, desc="preprocess raw ranges", unit="range"):
        npz_path = raw_npz_path(root, row)
        window_rows, arrays, raw_summary = process_raw_range(
            root,
            row,
            npz_path,
            config,
            args.window_seconds,
            config_hash,
            snapshot_ref,
        )
        raw_summary = {"raw_range_id": row["raw_range_id"], **raw_summary}
        raw_summaries.append(raw_summary)
        if not window_rows and raw_summary.get("status") != "success":
            window_rows = [
                failed_raw_range_row(
                    row,
                    raw_summary.get("failure_reason", "raw_range_processing_failed"),
                    config,
                    config_hash,
                    snapshot_ref,
                )
            ]
        local_array_index = 0
        for window_row in window_rows:
            if window_row.get("common_input_available") is True:
                window_row["array_index"] = len(all_arrays)
                all_arrays.append(arrays[local_array_index])
                local_array_index += 1
            window_row["raw_subject_eligible_for_10s_cross_session_protocol"] = raw_subject_eligible.get(row["subject_id"], False)
            window_row["raw_parent_session_supports_10s_protocol"] = (
                row["subject_id"], row["session_timestamp"]
            ) in session_10s
            all_rows.append(window_row)

    expected_samples = int(round(float(config["signal"]["target_fs_hz"]) * args.window_seconds))
    array = (
        np.stack(all_arrays).astype(np.float32)
        if all_arrays
        else np.empty((0, expected_samples), dtype=np.float32)
    )
    paths["array"].parent.mkdir(parents=True, exist_ok=True)
    np.save(paths["array"], array)
    write_csv(paths["manifest"], all_rows, MANIFEST_COLUMNS)
    summary = summary_from_rows(config, all_rows, raw_summaries, array, args.window_seconds, snapshot_validation_path)
    write_json(paths["summary"], summary)
    logging.info("Wrote %s shape=%s", paths["array"], array.shape)
    logging.info("Wrote %s rows=%d", paths["manifest"], len(all_rows))
    print(
        f"candidate_windows={len(all_rows)} common_input_available={array.shape[0]} "
        f"array_shape={array.shape}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
