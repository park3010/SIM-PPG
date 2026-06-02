#!/usr/bin/env python3
"""Reconstruct SigD-Core raw PLETH ranges from public WFDB records."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import math
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from tqdm import tqdm

from parse_sigd_annotations import (
    DATASET_NAME,
    DATASET_VERSION,
    build_candidate_record_paths,
    detect_root,
    load_config,
    setup_logging,
    sha256_file,
    sigd_dir,
)


EXTRACTION_COLUMNS = [
    "raw_range_id",
    "subject_id",
    "session_timestamp",
    "session_index_within_subject",
    "annotation_range_index_within_session",
    "requested_channel_name",
    "record_candidate_path_1",
    "record_candidate_path_2",
    "resolved_wfdb_record_name",
    "output_npz_path",
    "extraction_status",
    "failure_reason",
    "fs",
    "record_sig_len",
    "sampfrom",
    "sampto",
    "requested_duration_seconds",
    "extracted_samples",
    "extracted_duration_seconds",
    "duration_difference_seconds",
    "has_pleth",
    "nan_count",
    "nan_ratio",
    "inf_count",
    "flatline_ratio_raw",
    "ppg_min",
    "ppg_max",
    "ppg_mean",
    "ppg_std",
    "npz_sha256",
    "source_manifest_version",
    "extraction_datetime_utc",
]

FAILED_RAW_RANGE_COLUMNS = [
    "raw_range_id",
    "subject_id",
    "session_timestamp",
    "failure_reason",
    "resolved_wfdb_record_name",
    "record_candidate_path_1",
    "record_candidate_path_2",
    "extraction_datetime_utc",
]

SUCCESS_STATUS = "success"
FAILED_STATUS = "failed"
SKIPPED_EXISTING_STATUS = "skipped_existing"
DRY_RUN_STATUS = "dry_run"
HEADER_CHECKED_STATUS = "header_checked"

FAIL_ANNOTATION_PARSE = "annotation_parse_failed"
FAIL_RECORD_PATH = "record_path_not_resolved"
FAIL_HEADER_READ = "header_read_failed"
FAIL_PLETH = "pleth_not_available"
FAIL_RANGE = "range_out_of_bounds"
FAIL_EMPTY = "empty_signal"
FAIL_ALL_NAN = "all_nan_signal"
FAIL_WFDB = "wfdb_read_error"
FAIL_SAVE = "save_error"
FAIL_UNKNOWN = "unknown_error"
FAIL_EXISTING_FAILED = "existing_npz_with_failed_previous_status"

DRY_RUN_MANIFEST = "sigd_dry_run_manifest.csv"
HEADER_CHECK_MANIFEST = "sigd_header_check_manifest.csv"
EXTRACTION_MANIFEST = "sigd_extraction_manifest.csv"

RECOVERED_STAT_COLUMNS = [
    "fs",
    "extracted_samples",
    "extracted_duration_seconds",
    "duration_difference_seconds",
    "nan_count",
    "nan_ratio",
    "inf_count",
    "flatline_ratio_raw",
    "ppg_min",
    "ppg_max",
    "ppg_mean",
    "ppg_std",
    "npz_sha256",
]


@dataclass
class ResolutionResult:
    """WFDB record path resolution result."""

    success: bool
    resolved_wfdb_record_name: str = ""
    wfdb_record_name: str = ""
    resolved_pn_dir: str = ""
    header: Any | None = None
    failure_reason: str = ""
    errors: list[str] | None = None


def utc_now_iso() -> str:
    """Return a UTC ISO-8601 timestamp with seconds precision."""

    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def import_wfdb() -> Any:
    """Import wfdb lazily so dry-run can work without remote dependencies."""

    try:
        import wfdb
    except ImportError as exc:  # pragma: no cover - depends on environment
        raise RuntimeError(
            "wfdb is required for --header-check and waveform reconstruction. "
            "Install dataset/SigD/requirements.txt first."
        ) from exc
    return wfdb


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    """Read a UTF-8 CSV file into dictionaries."""

    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]], columns: list[str]) -> None:
    """Write rows to a UTF-8 CSV with fixed column order."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column, "") for column in columns})


def compact_utc_timestamp() -> str:
    """Return a compact UTC timestamp for extraction history filenames."""

    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


def load_source_manifest_version(root: Path) -> str:
    """Return a compact source manifest version string for provenance."""

    path = sigd_dir(root) / "metadata" / "source_manifest.json"
    if not path.exists():
        return "source_manifest_missing"
    with path.open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    commit = manifest.get("official_repo_commit")
    return str(commit or f"{manifest.get('dataset_version', DATASET_VERSION)}")


def select_rows(
    rows: list[dict[str, str]],
    subject_id: str | None,
    limit_subjects: int | None,
    limit_ranges: int | None,
) -> list[dict[str, str]]:
    """Apply CLI row filters while preserving manifest order."""

    selected = [row for row in rows if not subject_id or row["subject_id"] == subject_id]
    if limit_subjects is not None:
        allowed: set[str] = set()
        for row in selected:
            allowed.add(row["subject_id"])
            if len(allowed) >= limit_subjects:
                break
        selected = [row for row in selected if row["subject_id"] in allowed]
    if limit_ranges is not None:
        selected = selected[:limit_ranges]
    return selected


def output_npz_path(root: Path, row: dict[str, str]) -> Path:
    """Return the NPZ path for one successful raw range."""

    return (
        sigd_dir(root)
        / "data"
        / "raw_ranges"
        / row["subject_id"]
        / row["session_timestamp"]
        / f"range_{int(row['annotation_range_index_within_session']):03d}.npz"
    )


def relative_to_sigd(root: Path, path: Path) -> str:
    """Return a SigD-directory-relative path when possible."""

    try:
        return str(path.relative_to(sigd_dir(root)))
    except ValueError:
        return str(path)


def base_result_row(
    root: Path,
    annotation_row: dict[str, str],
    source_manifest_version: str,
) -> dict[str, Any]:
    """Create an extraction manifest row with fixed defaults."""

    out_path = output_npz_path(root, annotation_row)
    return {
        "raw_range_id": annotation_row.get("raw_range_id", ""),
        "subject_id": annotation_row.get("subject_id", ""),
        "session_timestamp": annotation_row.get("session_timestamp", ""),
        "session_index_within_subject": annotation_row.get(
            "session_index_within_subject", ""
        ),
        "annotation_range_index_within_session": annotation_row.get(
            "annotation_range_index_within_session", ""
        ),
        "requested_channel_name": annotation_row.get("requested_channel_name", "PLETH"),
        "record_candidate_path_1": annotation_row.get("record_candidate_path_1", ""),
        "record_candidate_path_2": annotation_row.get("record_candidate_path_2", ""),
        "resolved_wfdb_record_name": "",
        "output_npz_path": relative_to_sigd(root, out_path),
        "extraction_status": "",
        "failure_reason": "",
        "fs": "",
        "record_sig_len": "",
        "sampfrom": "",
        "sampto": "",
        "requested_duration_seconds": annotation_row.get(
            "requested_duration_seconds", ""
        ),
        "extracted_samples": "",
        "extracted_duration_seconds": "",
        "duration_difference_seconds": "",
        "has_pleth": "",
        "nan_count": "",
        "nan_ratio": "",
        "inf_count": "",
        "flatline_ratio_raw": "",
        "ppg_min": "",
        "ppg_max": "",
        "ppg_mean": "",
        "ppg_std": "",
        "npz_sha256": "",
        "source_manifest_version": source_manifest_version,
        "extraction_datetime_utc": utc_now_iso(),
    }


def candidate_paths_from_row(row: dict[str, str]) -> list[str]:
    """Read candidate WFDB paths from a manifest row or rebuild them."""

    paths = [
        row.get("record_candidate_path_1", ""),
        row.get("record_candidate_path_2", ""),
    ]
    if all(paths):
        return paths
    return build_candidate_record_paths(row["subject_id"], row["session_timestamp"])


def wfdb_lookup_attempts(candidate_path: str, pn_dir: str) -> list[tuple[str, str, str]]:
    """Build WFDB lookup attempts for path and pn_dir compatibility."""

    path = Path(candidate_path)
    attempts = [(candidate_path, pn_dir, candidate_path)]
    if path.parent != Path("."):
        attempts.append(
            (
                path.name,
                f"{pn_dir}/{path.parent.as_posix()}",
                candidate_path,
            )
        )
    if candidate_path.startswith("matched/"):
        without_matched = candidate_path.removeprefix("matched/")
        stripped_path = Path(without_matched)
        attempts.append((without_matched, pn_dir, without_matched))
        if stripped_path.parent != Path("."):
            attempts.append(
                (
                    stripped_path.name,
                    f"{pn_dir}/{stripped_path.parent.as_posix()}",
                    without_matched,
                )
            )

    deduped: list[tuple[str, str, str]] = []
    seen: set[tuple[str, str]] = set()
    for record_name, attempt_pn_dir, canonical_name in attempts:
        key = (record_name, attempt_pn_dir)
        if key not in seen:
            deduped.append((record_name, attempt_pn_dir, canonical_name))
            seen.add(key)
    return deduped


def resolve_remote_record_name(
    candidate_paths: list[str], pn_dir: str
) -> ResolutionResult:
    """Resolve the first WFDB record path whose remote header can be read."""

    wfdb = import_wfdb()
    errors: list[str] = []
    for candidate in candidate_paths:
        for record_name, attempt_pn_dir, canonical_name in wfdb_lookup_attempts(
            candidate, pn_dir
        ):
            try:
                header = wfdb.rdheader(record_name, pn_dir=attempt_pn_dir)
                return ResolutionResult(
                    success=True,
                    resolved_wfdb_record_name=canonical_name,
                    wfdb_record_name=record_name,
                    resolved_pn_dir=attempt_pn_dir,
                    header=header,
                    errors=errors,
                )
            except Exception as exc:  # row-level remote failures are expected
                errors.append(
                    f"{attempt_pn_dir}/{record_name}:{type(exc).__name__}:{exc}"
                )
    return ResolutionResult(
        success=False,
        failure_reason=FAIL_RECORD_PATH,
        errors=errors,
    )


def header_signal_names(header: Any) -> list[str]:
    """Extract signal names from a WFDB header-like object."""

    names = getattr(header, "sig_name", None)
    if names is None:
        names = getattr(header, "sig_names", None)
    if names is None:
        return []
    return [str(name) for name in names]


def header_info(header: Any, channel_name: str) -> dict[str, Any]:
    """Return selected header metadata without reading waveform arrays."""

    names = header_signal_names(header)
    n_seg = getattr(header, "n_seg", None)
    if names:
        has_pleth: str | bool = channel_name in names
    elif n_seg and int(n_seg) > 1:
        has_pleth = "unknown_until_range_read"
    else:
        has_pleth = False
    return {
        "fs": getattr(header, "fs", ""),
        "record_sig_len": getattr(header, "sig_len", ""),
        "has_pleth": has_pleth,
        "sig_name": names,
    }


def sample_indices(row: dict[str, str], fs: float) -> tuple[int, int]:
    """Convert annotation seconds to WFDB sample indices using actual fs."""

    start_seconds = float(row["offset_start_seconds"])
    end_seconds = float(row["offset_end_seconds"])
    return int(round(start_seconds * fs)), int(round(end_seconds * fs))


def previous_manifest_rows(root: Path) -> dict[str, dict[str, str]]:
    """Load previous actual extraction rows keyed by raw_range_id."""

    path = sigd_dir(root) / "metadata" / EXTRACTION_MANIFEST
    if not path.exists():
        return {}
    rows = read_csv_rows(path)
    return {row["raw_range_id"]: row for row in rows if row.get("raw_range_id")}


def is_previous_success_valid(root: Path, row: dict[str, str]) -> bool:
    """Check that a previous success row still points to a matching NPZ."""

    rel = row.get("output_npz_path", "")
    digest = row.get("npz_sha256", "")
    if not rel or not digest:
        return False
    path = sigd_dir(root) / rel
    return path.exists() and sha256_file(path) == digest


def apply_previous_stats(
    target: dict[str, Any], previous: dict[str, str], status: str
) -> dict[str, Any]:
    """Copy previous extraction stats while changing the current run status."""

    for column in EXTRACTION_COLUMNS:
        if column in previous and column not in {
            "extraction_status",
            "extraction_datetime_utc",
        }:
            target[column] = previous[column]
    target["extraction_status"] = status
    target["extraction_datetime_utc"] = utc_now_iso()
    target["failure_reason"] = ""
    return target


def previous_success_has_required_stats(previous: dict[str, str]) -> bool:
    """Check whether a previous success row has all audit-critical stats."""

    if previous.get("extraction_status") != SUCCESS_STATUS:
        return False
    return all(previous.get(column) not in {"", None} for column in RECOVERED_STAT_COLUMNS)


def npz_scalar(value: Any) -> Any:
    """Convert a zero-dimensional NPZ array to a plain Python value."""

    array = np.asarray(value)
    if array.shape == ():
        return array.item()
    if array.size == 1:
        return array.reshape(-1)[0].item()
    return array


def recover_existing_npz_stats(
    root: Path,
    annotation_row: dict[str, str],
    result: dict[str, Any],
    config: dict[str, Any],
) -> dict[str, Any]:
    """Recover skipped_existing stats from an existing NPZ using safe loading."""

    out_path = output_npz_path(root, annotation_row)
    try:
        with np.load(out_path, allow_pickle=False) as data:
            raw_range_id = str(npz_scalar(data["raw_range_id"]))
            channel_name = str(npz_scalar(data["channel_name"]))
            if raw_range_id != annotation_row["raw_range_id"]:
                result["extraction_status"] = FAILED_STATUS
                result["failure_reason"] = "existing_npz_raw_range_id_mismatch"
                return result
            if channel_name != config["signal_channel"]:
                result["extraction_status"] = FAILED_STATUS
                result["failure_reason"] = FAIL_PLETH
                return result
            ppg = np.asarray(data["ppg"], dtype=np.float32).reshape(-1)
            fs = float(npz_scalar(data["fs"]))
            requested_duration = float(annotation_row["requested_duration_seconds"])
            result.update(raw_integrity_stats(ppg, fs, requested_duration))
            result.update(
                {
                    "fs": fs,
                    "sampfrom": int(npz_scalar(data["sampfrom"]))
                    if "sampfrom" in data
                    else "",
                    "sampto": int(npz_scalar(data["sampto"])) if "sampto" in data else "",
                    "resolved_wfdb_record_name": str(
                        npz_scalar(data["resolved_wfdb_record_name"])
                    )
                    if "resolved_wfdb_record_name" in data
                    else "",
                    "has_pleth": True,
                    "output_npz_path": relative_to_sigd(root, out_path),
                    "npz_sha256": sha256_file(out_path),
                    "extraction_status": SKIPPED_EXISTING_STATUS,
                    "failure_reason": "",
                }
            )
            return result
    except Exception as exc:
        logging.exception("Could not recover existing NPZ stats for %s", out_path)
        result["extraction_status"] = FAILED_STATUS
        result["failure_reason"] = f"existing_npz_recovery_failed:{type(exc).__name__}:{exc}"
        return result


def validate_requested_channel_metadata(
    fields: dict[str, Any], channel_name: str
) -> tuple[bool, str]:
    """Validate that WFDB returned the requested channel before saving."""

    names = fields.get("sig_name") or fields.get("sig_names")
    if names:
        normalized = [str(name) for name in names]
        return channel_name in normalized, f"returned_channels={normalized}"
    logging.info(
        "WFDB did not return signal names after channel_names=[%s]; "
        "trusting the successful range-limited channel request.",
        channel_name,
    )
    return True, "channel_names_request_succeeded_without_returned_signal_names"


def cleanup_created_npz_on_failure(
    output_path: Path, created_this_run: bool, result: dict[str, Any]
) -> None:
    """Delete an NPZ created by this run when the row ultimately fails."""

    if (
        created_this_run
        and result.get("extraction_status") == FAILED_STATUS
        and output_path.exists()
    ):
        try:
            output_path.unlink()
            logging.warning("Deleted failed NPZ created in this run: %s", output_path)
        except OSError as exc:
            logging.error("Could not delete failed NPZ %s: %s", output_path, exc)


def read_pleth_range(
    resolved_record_name: str,
    pn_dir: str,
    sampfrom: int,
    sampto: int,
    channel_name: str,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Read a range-limited PLETH signal from WFDB without saving full records."""

    wfdb = import_wfdb()
    try:
        try:
            signals, fields = wfdb.rdsamp(
                resolved_record_name,
                sampfrom=sampfrom,
                sampto=sampto,
                channel_names=[channel_name],
                pn_dir=pn_dir,
                warn_empty=False,
            )
        except TypeError:
            signals, fields = wfdb.rdsamp(
                resolved_record_name,
                sampfrom=sampfrom,
                sampto=sampto,
                channel_names=[channel_name],
                pn_dir=pn_dir,
            )
        array = np.asarray(signals)
        return array, dict(fields or {})
    except Exception as first_exc:
        logging.debug("wfdb.rdsamp failed; trying rdrecord: %s", first_exc)
        try:
            try:
                record = wfdb.rdrecord(
                    resolved_record_name,
                    sampfrom=sampfrom,
                    sampto=sampto,
                    channel_names=[channel_name],
                    pn_dir=pn_dir,
                    physical=True,
                    m2s=True,
                    force_channels=True,
                )
            except TypeError:
                record = wfdb.rdrecord(
                    resolved_record_name,
                    sampfrom=sampfrom,
                    sampto=sampto,
                    channel_names=[channel_name],
                    pn_dir=pn_dir,
                    physical=True,
                    m2s=True,
                )
            array = np.asarray(record.p_signal)
            fields = {
                "fs": getattr(record, "fs", None),
                "sig_name": getattr(record, "sig_name", None),
            }
            return array, fields
        except Exception as second_exc:
            raise RuntimeError(
                f"{FAIL_WFDB}: rdsamp={first_exc}; rdrecord={second_exc}"
            ) from second_exc


def ppg_vector(signals: np.ndarray) -> np.ndarray:
    """Convert a WFDB signal matrix into a 1D float32 PPG vector."""

    array = np.asarray(signals)
    if array.size == 0:
        return np.asarray([], dtype=np.float32)
    if array.ndim == 2:
        if array.shape[1] < 1:
            return np.asarray([], dtype=np.float32)
        array = array[:, 0]
    return np.asarray(array, dtype=np.float32).reshape(-1)


def raw_integrity_stats(ppg: np.ndarray, fs: float, requested_duration: float) -> dict[str, Any]:
    """Compute raw integrity statistics allowed at reconstruction time."""

    extracted_samples = int(ppg.size)
    nan_mask = np.isnan(ppg)
    inf_mask = np.isinf(ppg)
    finite = ppg[np.isfinite(ppg)]
    nan_count = int(nan_mask.sum())
    inf_count = int(inf_mask.sum())
    nan_ratio = float(nan_count / extracted_samples) if extracted_samples else math.nan
    extracted_duration = float(extracted_samples / fs) if fs else math.nan
    if finite.size >= 2:
        flatline_ratio = float(np.mean(np.diff(finite) == 0))
    else:
        flatline_ratio = 0.0
    stats: dict[str, Any] = {
        "extracted_samples": extracted_samples,
        "extracted_duration_seconds": extracted_duration,
        "duration_difference_seconds": extracted_duration - requested_duration,
        "nan_count": nan_count,
        "nan_ratio": nan_ratio,
        "inf_count": inf_count,
        "flatline_ratio_raw": flatline_ratio,
        "ppg_min": float(np.min(finite)) if finite.size else "",
        "ppg_max": float(np.max(finite)) if finite.size else "",
        "ppg_mean": float(np.mean(finite)) if finite.size else "",
        "ppg_std": float(np.std(finite)) if finite.size else "",
    }
    return stats


def save_raw_range_npz(
    output_path: Path,
    ppg: np.ndarray,
    fs: float,
    row: dict[str, str],
    sampfrom: int,
    sampto: int,
    resolved_wfdb_record_name: str,
    config: dict[str, Any],
) -> None:
    """Save one SigD-Core raw PLETH range as a compressed NPZ."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        ppg=ppg.astype(np.float32, copy=False),
        fs=np.float64(fs),
        raw_range_id=np.asarray(row["raw_range_id"]),
        subject_id=np.asarray(row["subject_id"]),
        session_timestamp=np.asarray(row["session_timestamp"]),
        annotation_range_index_within_session=np.int64(
            int(row["annotation_range_index_within_session"])
        ),
        offset_start_seconds=np.int64(int(float(row["offset_start_seconds"]))),
        offset_end_seconds=np.int64(int(float(row["offset_end_seconds"]))),
        requested_duration_seconds=np.float64(
            float(row["requested_duration_seconds"])
        ),
        sampfrom=np.int64(sampfrom),
        sampto=np.int64(sampto),
        resolved_wfdb_record_name=np.asarray(resolved_wfdb_record_name),
        channel_name=np.asarray(config["signal_channel"]),
        source_database=np.asarray(config["physionet_database"]),
        source_version=np.asarray(str(config["physionet_version"])),
        dataset_name=np.asarray(DATASET_NAME),
        dataset_version=np.asarray(config["dataset_version"]),
    )


def process_dry_run(
    root: Path, row: dict[str, str], source_manifest_version: str
) -> dict[str, Any]:
    """Create a dry-run manifest row without remote PhysioNet access."""

    result = base_result_row(root, row, source_manifest_version)
    if row.get("annotation_parse_status") != "success":
        result["extraction_status"] = FAILED_STATUS
        result["failure_reason"] = FAIL_ANNOTATION_PARSE
    else:
        result["extraction_status"] = DRY_RUN_STATUS
    return result


def process_header_check(
    root: Path,
    row: dict[str, str],
    config: dict[str, Any],
    source_manifest_version: str,
) -> dict[str, Any]:
    """Resolve remote header metadata without reading or saving waveform arrays."""

    result = base_result_row(root, row, source_manifest_version)
    if row.get("annotation_parse_status") != "success":
        result["extraction_status"] = FAILED_STATUS
        result["failure_reason"] = FAIL_ANNOTATION_PARSE
        return result

    try:
        resolution = resolve_remote_record_name(
            candidate_paths_from_row(row), config["physionet_pn_dir"]
        )
        if not resolution.success or resolution.header is None:
            result["extraction_status"] = FAILED_STATUS
            result["failure_reason"] = resolution.failure_reason or FAIL_HEADER_READ
            return result
        info = header_info(resolution.header, config["signal_channel"])
        result.update(
            {
                "resolved_wfdb_record_name": resolution.resolved_wfdb_record_name,
                "extraction_status": HEADER_CHECKED_STATUS,
                "fs": info["fs"],
                "record_sig_len": info["record_sig_len"],
                "has_pleth": info["has_pleth"],
            }
        )
        return result
    except Exception as exc:
        logging.exception("Header check failed for %s", row.get("raw_range_id"))
        result["extraction_status"] = FAILED_STATUS
        result["failure_reason"] = f"{FAIL_HEADER_READ}:{type(exc).__name__}:{exc}"
        return result


def process_extraction(
    root: Path,
    row: dict[str, str],
    config: dict[str, Any],
    source_manifest_version: str,
    overwrite: bool,
    resume: bool,
    previous_rows: dict[str, dict[str, str]],
) -> dict[str, Any]:
    """Extract and save one range-limited PLETH waveform row."""

    result = base_result_row(root, row, source_manifest_version)
    if row.get("annotation_parse_status") != "success":
        result["extraction_status"] = FAILED_STATUS
        result["failure_reason"] = FAIL_ANNOTATION_PARSE
        return result

    out_path = output_npz_path(root, row)
    existed_before = out_path.exists()
    previous = previous_rows.get(row["raw_range_id"])
    if existed_before and not overwrite:
        if previous and previous.get("extraction_status") == FAILED_STATUS:
            result["extraction_status"] = FAILED_STATUS
            result["failure_reason"] = FAIL_EXISTING_FAILED
            result["npz_sha256"] = sha256_file(out_path)
            logging.warning(
                "Existing NPZ has a previous failed manifest row and will not be reused: %s",
                out_path,
            )
            return result
        if (
            previous
            and is_previous_success_valid(root, previous)
            and previous_success_has_required_stats(previous)
        ):
            return apply_previous_stats(result, previous, SKIPPED_EXISTING_STATUS)
        return recover_existing_npz_stats(root, row, result, config)

    try:
        created_this_run = False
        resolution = resolve_remote_record_name(
            candidate_paths_from_row(row), config["physionet_pn_dir"]
        )
        if not resolution.success or resolution.header is None:
            result["extraction_status"] = FAILED_STATUS
            result["failure_reason"] = resolution.failure_reason or FAIL_RECORD_PATH
            return result

        info = header_info(resolution.header, config["signal_channel"])
        fs = float(info["fs"])
        sampfrom, sampto = sample_indices(row, fs)
        record_sig_len = info["record_sig_len"]
        result.update(
            {
                "resolved_wfdb_record_name": resolution.resolved_wfdb_record_name,
                "fs": fs,
                "record_sig_len": record_sig_len,
                "sampfrom": sampfrom,
                "sampto": sampto,
                "has_pleth": info["has_pleth"],
            }
        )

        if record_sig_len not in {"", None} and sampto > int(record_sig_len):
            result["extraction_status"] = FAILED_STATUS
            result["failure_reason"] = FAIL_RANGE
            return result
        if sampfrom < 0 or sampto <= sampfrom:
            result["extraction_status"] = FAILED_STATUS
            result["failure_reason"] = FAIL_RANGE
            return result

        signals, fields = read_pleth_range(
            resolution.wfdb_record_name or resolution.resolved_wfdb_record_name,
            resolution.resolved_pn_dir or config["physionet_pn_dir"],
            sampfrom,
            sampto,
            config["signal_channel"],
        )
        ppg = ppg_vector(signals)
        if ppg.size == 0:
            result["extraction_status"] = FAILED_STATUS
            result["failure_reason"] = FAIL_EMPTY
            return result
        if np.all(np.isnan(ppg)):
            result["extraction_status"] = FAILED_STATUS
            result["failure_reason"] = FAIL_ALL_NAN
            return result
        if not np.any(np.isfinite(ppg)):
            result["extraction_status"] = FAILED_STATUS
            result["failure_reason"] = FAIL_EMPTY
            return result

        channel_ok, channel_detail = validate_requested_channel_metadata(
            fields, config["signal_channel"]
        )
        if not channel_ok:
            result["has_pleth"] = False
            result["extraction_status"] = FAILED_STATUS
            result["failure_reason"] = f"{FAIL_PLETH}:{channel_detail}"
            return result
        logging.debug("PLETH channel validation: %s", channel_detail)

        requested_duration = float(row["requested_duration_seconds"])
        result.update(raw_integrity_stats(ppg, fs, requested_duration))
        result["has_pleth"] = True

        try:
            save_raw_range_npz(
                out_path,
                ppg,
                fs,
                row,
                sampfrom,
                sampto,
                resolution.resolved_wfdb_record_name,
                config,
            )
            created_this_run = True
        except Exception as exc:
            logging.exception("Could not save %s", out_path)
            result["extraction_status"] = FAILED_STATUS
            result["failure_reason"] = f"{FAIL_SAVE}:{type(exc).__name__}:{exc}"
            cleanup_created_npz_on_failure(
                out_path, (not existed_before) and out_path.exists(), result
            )
            return result

        result["output_npz_path"] = relative_to_sigd(root, out_path)
        result["npz_sha256"] = sha256_file(out_path)
        result["extraction_status"] = SUCCESS_STATUS
        cleanup_created_npz_on_failure(out_path, created_this_run, result)
        return result
    except RuntimeError as exc:
        logging.exception("WFDB read failed for %s", row.get("raw_range_id"))
        result["extraction_status"] = FAILED_STATUS
        result["failure_reason"] = str(exc)
        return result
    except Exception as exc:
        logging.exception("Unknown extraction error for %s", row.get("raw_range_id"))
        result["extraction_status"] = FAILED_STATUS
        result["failure_reason"] = f"{FAIL_UNKNOWN}:{type(exc).__name__}:{exc}"
        return result


def failed_rows_for_export(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return failed extraction rows for failed_raw_ranges.csv."""

    failed = []
    for row in rows:
        if row.get("extraction_status") == FAILED_STATUS:
            failed.append({column: row.get(column, "") for column in FAILED_RAW_RANGE_COLUMNS})
    return failed


def merge_actual_extraction_results(
    existing_rows: list[dict[str, Any]],
    new_rows: list[dict[str, Any]],
    annotation_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Merge actual extraction rows into a current-state manifest snapshot."""

    merged: dict[str, dict[str, Any]] = {}
    first_seen_order: dict[str, int] = {}

    for row in existing_rows:
        raw_range_id = str(row.get("raw_range_id", ""))
        if not raw_range_id:
            continue
        if raw_range_id not in first_seen_order:
            first_seen_order[raw_range_id] = len(first_seen_order)
        merged[raw_range_id] = dict(row)

    for row in new_rows:
        raw_range_id = str(row.get("raw_range_id", ""))
        if not raw_range_id:
            continue
        if raw_range_id not in first_seen_order:
            first_seen_order[raw_range_id] = len(first_seen_order)
        merged[raw_range_id] = dict(row)

    annotation_order = {
        str(row.get("raw_range_id", "")): index
        for index, row in enumerate(annotation_rows)
        if row.get("raw_range_id")
    }

    return sorted(
        merged.values(),
        key=lambda row: (
            0
            if str(row.get("raw_range_id", "")) in annotation_order
            else 1,
            annotation_order.get(
                str(row.get("raw_range_id", "")),
                first_seen_order.get(str(row.get("raw_range_id", "")), 0),
            ),
        ),
    )


def write_actual_extraction_manifest_merged(
    root: Path,
    new_rows: list[dict[str, Any]],
    annotation_rows: list[dict[str, Any]],
) -> tuple[Path, list[dict[str, Any]]]:
    """Write the merged current-state actual extraction manifest."""

    path = output_manifest_path(root, dry_run=False, header_check=False)
    existing_rows = read_csv_rows(path) if path.exists() else []
    merged_rows = merge_actual_extraction_results(
        existing_rows, new_rows, annotation_rows
    )
    write_csv(path, merged_rows, EXTRACTION_COLUMNS)
    return path, merged_rows


def failed_records_csv_path(root: Path) -> Path:
    """Return the failed raw range CSV path."""

    return sigd_dir(root) / "data" / "failed_records" / "failed_raw_ranges.csv"


def write_failed_records_from_current_state(
    root: Path, merged_rows: list[dict[str, Any]]
) -> Path:
    """Write failed raw ranges from the merged actual manifest state."""

    path = failed_records_csv_path(root)
    write_csv(path, failed_rows_for_export(merged_rows), FAILED_RAW_RANGE_COLUMNS)
    return path


def output_manifest_path(root: Path, dry_run: bool, header_check: bool) -> Path:
    """Return the mode-specific manifest path for this reconstruction run."""

    metadata_dir = sigd_dir(root) / "metadata"
    if dry_run:
        return metadata_dir / DRY_RUN_MANIFEST
    if header_check:
        return metadata_dir / HEADER_CHECK_MANIFEST
    return metadata_dir / EXTRACTION_MANIFEST


def extraction_history_label(args: argparse.Namespace) -> str:
    """Classify an actual extraction run as smoke or full for history."""

    if (
        args.limit_subjects is None
        and args.limit_ranges is None
        and args.subject_id is None
    ):
        return "full_extraction"
    return "smoke_extraction"


def write_extraction_history(
    root: Path, rows: list[dict[str, Any]], label: str
) -> Path:
    """Write a timestamped actual extraction history CSV."""

    history_dir = sigd_dir(root) / "metadata" / "extraction_history"
    history_dir.mkdir(parents=True, exist_ok=True)
    path = history_dir / f"{label}_{compact_utc_timestamp()}.csv"
    write_csv(path, rows, EXTRACTION_COLUMNS)
    return path


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(
        description=(
            "Range-limited PLETH extraction without persistent full-record storage "
            "for SigD-Core."
        )
    )
    parser.add_argument("--root", type=str, default=None, help="sim_ppg root path")
    parser.add_argument(
        "--manifest",
        type=str,
        default=None,
        help="Path to sigd_annotation_manifest.csv",
    )
    parser.add_argument("--limit-subjects", type=int, default=None)
    parser.add_argument("--limit-ranges", type=int, default=None)
    parser.add_argument("--subject-id", type=str, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--header-check", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def main() -> int:
    """CLI entry point."""

    args = parse_args()
    root = detect_root(args.root)
    setup_logging(root, "reconstruct_sigd_core.log", args.verbose)
    config = load_config(root)
    source_manifest_version = load_source_manifest_version(root)

    manifest_path = (
        Path(args.manifest).expanduser().resolve()
        if args.manifest
        else sigd_dir(root) / "metadata" / "sigd_annotation_manifest.csv"
    )
    if not manifest_path.exists():
        raise SystemExit(f"Missing annotation manifest: {manifest_path}")

    rows = read_csv_rows(manifest_path)
    selected = select_rows(
        rows, args.subject_id, args.limit_subjects, args.limit_ranges
    )
    previous_rows = previous_manifest_rows(root)
    output_manifest = output_manifest_path(root, args.dry_run, args.header_check)
    failed_csv = failed_records_csv_path(root)

    logging.info(
        "Starting %s for %d raw ranges",
        "dry-run"
        if args.dry_run
        else "header-check"
        if args.header_check
        else "range-limited PLETH extraction without persistent full-record storage",
        len(selected),
    )

    results: list[dict[str, Any]] = []
    for row in tqdm(selected, desc="SigD-Core raw ranges", unit="range"):
        if args.dry_run:
            result = process_dry_run(root, row, source_manifest_version)
        elif args.header_check:
            result = process_header_check(root, row, config, source_manifest_version)
        else:
            result = process_extraction(
                root,
                row,
                config,
                source_manifest_version,
                overwrite=args.overwrite,
                resume=args.resume,
                previous_rows=previous_rows,
            )
        results.append(result)

    history_path: Path | None = None
    if not args.dry_run and not args.header_check:
        output_manifest, current_state_rows = write_actual_extraction_manifest_merged(
            root, results, rows
        )
        failed_csv = write_failed_records_from_current_state(root, current_state_rows)
        history_path = write_extraction_history(
            root, results, extraction_history_label(args)
        )
    else:
        write_csv(output_manifest, results, EXTRACTION_COLUMNS)

    status_counts: dict[str, int] = {}
    for result in results:
        status = str(result.get("extraction_status", ""))
        status_counts[status] = status_counts.get(status, 0) + 1
    logging.info("Extraction status counts: %s", status_counts)
    logging.info("Wrote %s", output_manifest)
    if not args.dry_run and not args.header_check:
        logging.info("Wrote %s", failed_csv)
        logging.info("Wrote extraction history %s", history_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
