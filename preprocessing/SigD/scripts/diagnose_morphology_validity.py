#!/usr/bin/env python3
"""Run a separated morphology-validity diagnostic pilot for SigD windows."""

from __future__ import annotations

import argparse
from collections import defaultdict
import logging
import os
from pathlib import Path
import random
import sys
from typing import Any

import numpy as np
from tqdm import tqdm

SCRIPT_DIR = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SCRIPT_DIR))

from common import (  # noqa: E402
    bool_from_any,
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
from morphology_targets import compute_ipa  # noqa: E402
from signal_processing import process_raw_range  # noqa: E402
from snapshot_validation import available_extraction_rows, raw_npz_path, snapshot_hash_results  # noqa: E402


PILOT_COLUMNS = [
    "window_id",
    "parent_raw_range_id",
    "subject_id",
    "session_timestamp",
    "window_index_within_raw_range",
    "common_input_available",
    "model_input_available",
    "filtered_std",
    "filtered_flatline_ratio_window",
    "sqi_skewness",
    "svri",
    "svri_valid_mask",
    "sqi_valid_mask",
    "ipa",
    "ipa_valid_mask",
    "ipa_failure_reason",
    "aux_morphology_annotation_available",
    "aux_morphology_any_available",
    "aux_morphology_all_available",
    "official_compatible_ipa",
    "official_compatible_ipa_valid_mask",
    "official_compatible_ipa_failure_reason",
    "official_zscore_ipa",
    "official_zscore_ipa_valid_mask",
    "official_zscore_ipa_failure_reason",
    "wrapper_matches_official_compatible",
    "plot_path",
]


def zscore(values: np.ndarray) -> np.ndarray:
    """Return per-window z-score values when possible."""

    x = np.asarray(values, dtype=float)
    std = float(np.std(x))
    if std <= 1.0e-12:
        return x * np.nan
    return (x - float(np.mean(x))) / std


def select_pilot_rows(
    rows: list[dict[str, str]],
    *,
    min_subjects: int,
    min_raw_ranges: int,
    seed: int,
) -> list[dict[str, str]]:
    """Select a deterministic pilot with cross-session subjects preferred."""

    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[row["subject_id"]].append(row)

    subjects = list(grouped)
    cross_session = [
        subject
        for subject, items in grouped.items()
        if len({row["session_timestamp"] for row in items}) >= 2
    ]
    rng = random.Random(seed)
    rng.shuffle(cross_session)
    rng.shuffle(subjects)

    selected_subjects: list[str] = []
    for subject in cross_session + subjects:
        if subject not in selected_subjects:
            selected_subjects.append(subject)
        selected_count = sum(len(grouped[item]) for item in selected_subjects)
        if len(selected_subjects) >= min_subjects and selected_count >= min_raw_ranges:
            break

    selected: list[dict[str, str]] = []
    for subject in selected_subjects:
        subject_rows = grouped[subject][:]
        rng.shuffle(subject_rows)
        selected.append(subject_rows[0])

    remaining = [
        row
        for subject in selected_subjects
        for row in grouped[subject]
        if row not in selected
    ]
    rng.shuffle(remaining)
    selected.extend(remaining[: max(0, min_raw_ranges - len(selected))])

    if len(selected) < min_raw_ranges:
        leftovers = [row for row in rows if row not in selected]
        rng.shuffle(leftovers)
        selected.extend(leftovers[: min_raw_ranges - len(selected)])

    order = {row["raw_range_id"]: index for index, row in enumerate(rows)}
    return sorted(selected[: max(min_raw_ranges, len(selected_subjects))], key=lambda row: order[row["raw_range_id"]])


def verify_selected_hashes(root: Path, rows: list[dict[str, str]]) -> list[str]:
    """Verify selected raw NPZ hashes against the reconstruction manifest."""

    failures = []
    for row in rows:
        path = raw_npz_path(root, row)
        expected = row.get("npz_sha256", "")
        actual = sha256_file(path) if path.exists() else ""
        if not path.exists():
            failures.append(f"missing_npz:{row['raw_range_id']}")
        elif expected and actual != expected:
            failures.append(f"npz_sha256_mismatch:{row['raw_range_id']}")
    return failures


def bool_value(value: Any) -> bool:
    """Parse booleans from row values."""

    return bool_from_any(value)


def rows_by_ratio(rows: list[dict[str, Any]], key: str) -> list[dict[str, Any]]:
    """Return per-subject/session IPA valid ratios."""

    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row[key])].append(row)
    output = []
    for group, items in sorted(grouped.items()):
        total = sum(1 for item in items if bool_value(item.get("common_input_available")))
        valid = sum(1 for item in items if bool_value(item.get("ipa_valid_mask")))
        output.append(
            {
                key: group,
                "common_input_available_windows": total,
                "ipa_valid_windows": valid,
                "ipa_valid_ratio": (valid / total) if total else None,
            }
        )
    return output


def session_ratio_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return per-session IPA valid ratios."""

    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["subject_id"]), str(row["session_timestamp"]))].append(row)
    output = []
    for (subject_id, session_timestamp), items in sorted(grouped.items()):
        total = sum(1 for item in items if bool_value(item.get("common_input_available")))
        valid = sum(1 for item in items if bool_value(item.get("ipa_valid_mask")))
        output.append(
            {
                "subject_id": subject_id,
                "session_timestamp": session_timestamp,
                "common_input_available_windows": total,
                "ipa_valid_windows": valid,
                "ipa_valid_ratio": (valid / total) if total else None,
            }
        )
    return output


def summarize_by_ipa(rows: list[dict[str, Any]], valid: bool, field: str) -> dict[str, Any]:
    """Summarize a numeric field for IPA-valid or IPA-invalid rows."""

    return numeric_summary(
        row.get(field)
        for row in rows
        if bool_value(row.get("common_input_available")) and bool_value(row.get("ipa_valid_mask")) is valid
    )


def save_plot(path: Path, waveform: np.ndarray, row: dict[str, Any], title: str) -> None:
    """Save one diagnostic waveform plot."""

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 3))
    ax.plot(np.asarray(waveform, dtype=float), linewidth=1.0)
    ax.set_title(title)
    ax.set_xlabel("sample")
    ax.set_ylabel("filtered PPG")
    ax.text(
        0.01,
        0.98,
        f"{row['subject_id']} {row['session_timestamp']}\n{row['window_id']}",
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=8,
    )
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def maybe_plot_window(
    plot_dir: Path,
    row: dict[str, Any],
    waveform: np.ndarray,
    valid_saved: list[str],
    invalid_saved: dict[str, int],
) -> str:
    """Save at most 5 valid plots and 5 invalid plots per failure reason."""

    if bool_value(row.get("ipa_valid_mask")):
        if len(valid_saved) >= 5:
            return ""
        filename = f"ipa_valid_{len(valid_saved):02d}_{row['window_id']}.png"
        path = plot_dir / filename
        save_plot(path, waveform, row, "IPA valid")
        valid_saved.append(str(path))
        return str(path)

    reason = str(row.get("ipa_failure_reason") or "unknown")
    count = invalid_saved.get(reason, 0)
    if count >= 5:
        return ""
    filename = f"ipa_invalid_{reason}_{count:02d}_{row['window_id']}.png"
    safe = "".join(ch if ch.isalnum() or ch in {"_", "-", "."} else "_" for ch in filename)
    path = plot_dir / safe
    save_plot(path, waveform, row, f"IPA invalid: {reason}")
    invalid_saved[reason] = count + 1
    return str(path)


def build_summary(
    *,
    config: dict[str, Any],
    selected_rows: list[dict[str, str]],
    pilot_rows: list[dict[str, Any]],
    raw_summaries: list[dict[str, Any]],
    validation_failures: list[str],
    plot_paths: list[str],
    seed: int,
) -> dict[str, Any]:
    """Build diagnostic pilot summary."""

    common_rows = [row for row in pilot_rows if bool_value(row.get("common_input_available"))]
    total = len(common_rows)
    svri_valid = sum(1 for row in common_rows if bool_value(row.get("svri_valid_mask")))
    sqi_valid = sum(1 for row in common_rows if bool_value(row.get("sqi_valid_mask")))
    ipa_valid = sum(1 for row in common_rows if bool_value(row.get("ipa_valid_mask")))
    return {
        "diagnostic_name": "morphology_validity_pilot_10s",
        "generated_datetime_utc": utc_now_iso(),
        "input_protocol_id": config["input_protocol_id"],
        "comparison_role": config["comparison_role"],
        "preprocessing_profile": config["preprocessing_profile"],
        "normalization_policy": config["normalization_policy"],
        "seed": seed,
        "processed_subjects": len({row["subject_id"] for row in selected_rows}),
        "processed_raw_ranges": len(selected_rows),
        "processed_sessions": len({(row["subject_id"], row["session_timestamp"]) for row in selected_rows}),
        "raw_range_processing_status_distribution": distribution(item.get("status") for item in raw_summaries),
        "total_candidate_windows": len(pilot_rows),
        "common_input_available_windows": total,
        "sVRI_valid_count": svri_valid,
        "sVRI_valid_ratio": (svri_valid / total) if total else None,
        "SQI_valid_count": sqi_valid,
        "SQI_valid_ratio": (sqi_valid / total) if total else None,
        "IPA_valid_count": ipa_valid,
        "IPA_valid_ratio": (ipa_valid / total) if total else None,
        "IPA_failure_reason_distribution": distribution(
            row.get("ipa_failure_reason") or "none"
            for row in common_rows
            if not bool_value(row.get("ipa_valid_mask"))
        ),
        "aux_morphology_any_available_count": sum(
            1 for row in common_rows if bool_value(row.get("aux_morphology_any_available"))
        ),
        "aux_morphology_all_available_count": sum(
            1 for row in common_rows if bool_value(row.get("aux_morphology_all_available"))
        ),
        "subject_ipa_valid_ratios": rows_by_ratio(common_rows, "subject_id"),
        "session_ipa_valid_ratios": session_ratio_rows(common_rows),
        "ipa_valid_window_summaries": {
            "filtered_std": summarize_by_ipa(common_rows, True, "filtered_std"),
            "sqi_skewness": summarize_by_ipa(common_rows, True, "sqi_skewness"),
            "filtered_flatline_ratio_window": summarize_by_ipa(common_rows, True, "filtered_flatline_ratio_window"),
        },
        "ipa_invalid_window_summaries": {
            "filtered_std": summarize_by_ipa(common_rows, False, "filtered_std"),
            "sqi_skewness": summarize_by_ipa(common_rows, False, "sqi_skewness"),
            "filtered_flatline_ratio_window": summarize_by_ipa(common_rows, False, "filtered_flatline_ratio_window"),
        },
        "official_compute_ipa_comparison": {
            "official_reference_file": "preprocessing/SigD/official_reference/PaPaGei/morphology.py",
            "official_behavior_summary": "first-two-relative-minima beat, first internal minimum split, systolic/diastolic trapz ratio, 0 on IndexError",
            "wrapper_behavior_summary": "same target definition with explicit valid mask and failure reason instead of 0 sentinel",
            "wrapper_official_compatible_mismatch_count": sum(
                1 for row in common_rows if not bool_value(row.get("wrapper_matches_official_compatible"))
            ),
            "official_zscore_ipa_valid_count": sum(
                1 for row in common_rows if bool_value(row.get("official_zscore_ipa_valid_mask"))
            ),
        },
        "plot_paths": plot_paths,
        "validation_failures": validation_failures,
        "full_preprocessing_executed": False,
        "common_eligibility_uses_morphology_masks": False,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Diagnose SigD morphology target validity.")
    parser.add_argument("--root", type=str, default=None)
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--window-seconds", type=int, default=10)
    parser.add_argument("--min-subjects", type=int, default=20)
    parser.add_argument("--min-raw-ranges", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = detect_root(args.root)
    setup_logging(root, "morphology_validity_pilot.log", args.verbose)
    logging.getLogger("matplotlib").setLevel(logging.WARNING)
    os.environ.setdefault("MPLCONFIGDIR", str(Path("/tmp") / "sim_ppg_matplotlib"))
    config = load_config(root, args.config)
    if args.window_seconds != 10:
        raise SystemExit("Diagnostic pilot currently targets the primary 10s protocol.")

    diagnostics_dir = preprocessing_dir(root) / "metadata" / "diagnostics"
    plots_dir = diagnostics_dir / "plots"
    csv_path = diagnostics_dir / "morphology_validity_pilot_10s.csv"
    summary_path = diagnostics_dir / "morphology_validity_pilot_10s_summary.json"

    hash_results = snapshot_hash_results(root, config)
    if not all(item["sha256_match"] for item in hash_results):
        raise SystemExit("Snapshot metadata hash validation failed.")

    extraction_rows = read_csv_rows(resolve_path(root, config["input"]["extraction_manifest"]))
    available_rows = available_extraction_rows(config, extraction_rows)
    selected_rows = select_pilot_rows(
        available_rows,
        min_subjects=args.min_subjects,
        min_raw_ranges=args.min_raw_ranges,
        seed=args.seed,
    )
    validation_failures = verify_selected_hashes(root, selected_rows)
    if validation_failures:
        raise SystemExit("Selected NPZ hash validation failed: " + "; ".join(validation_failures[:5]))

    config_hash = sha256_jsonable(config)
    snapshot_ref = sha256_file(resolve_path(root, config["input"]["snapshot_sha256_file"]))

    pilot_rows: list[dict[str, Any]] = []
    raw_summaries: list[dict[str, Any]] = []
    plot_paths: list[str] = []
    valid_saved: list[str] = []
    invalid_saved: dict[str, int] = {}

    for manifest_row in tqdm(selected_rows, desc="diagnose raw ranges", unit="range"):
        npz_path = raw_npz_path(root, manifest_row)
        window_rows, arrays, raw_summary = process_raw_range(
            root,
            manifest_row,
            npz_path,
            config,
            args.window_seconds,
            config_hash,
            snapshot_ref,
        )
        raw_summaries.append({"raw_range_id": manifest_row["raw_range_id"], **raw_summary})
        for row in window_rows:
            if not bool_value(row.get("common_input_available")):
                continue
            waveform = arrays[int(row["array_index"])]
            official = compute_ipa(waveform, float(row["target_fs"]))
            official_z = compute_ipa(zscore(waveform), float(row["target_fs"]))
            row["official_compatible_ipa"] = official["ipa"]
            row["official_compatible_ipa_valid_mask"] = official["ipa_valid_mask"]
            row["official_compatible_ipa_failure_reason"] = official["ipa_failure_reason"]
            row["official_zscore_ipa"] = official_z["ipa"]
            row["official_zscore_ipa_valid_mask"] = official_z["ipa_valid_mask"]
            row["official_zscore_ipa_failure_reason"] = official_z["ipa_failure_reason"]
            row_valid = bool_value(row.get("ipa_valid_mask"))
            official_valid = bool_value(official["ipa_valid_mask"])
            if row_valid and official_valid:
                matches = bool(np.isclose(float(row["ipa"]), float(official["ipa"]), rtol=1.0e-6, atol=1.0e-8))
            else:
                matches = row_valid == official_valid and row.get("ipa_failure_reason") == official["ipa_failure_reason"]
            row["wrapper_matches_official_compatible"] = matches
            row["plot_path"] = maybe_plot_window(plots_dir, row, waveform, valid_saved, invalid_saved)
            if row["plot_path"]:
                plot_paths.append(row["plot_path"])
            pilot_rows.append(row)

    write_csv(csv_path, pilot_rows, PILOT_COLUMNS)
    summary = build_summary(
        config=config,
        selected_rows=selected_rows,
        pilot_rows=pilot_rows,
        raw_summaries=raw_summaries,
        validation_failures=validation_failures,
        plot_paths=plot_paths,
        seed=args.seed,
    )
    write_json(summary_path, summary)
    print(
        "pilot_subjects={processed_subjects} raw_ranges={processed_raw_ranges} "
        "windows={total_candidate_windows} common={common_input_available_windows} "
        "ipa_valid={IPA_valid_count} ratio={IPA_valid_ratio}".format(**summary)
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
