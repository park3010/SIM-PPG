"""Validation for the fixed SigD-Core reconstruction snapshot."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np

from common import (
    bool_from_any,
    distribution,
    read_csv_rows,
    read_json,
    resolve_path,
    sha256_file,
    utc_now_iso,
    write_json,
)


REQUIRED_NPZ_FIELDS = [
    "ppg",
    "fs",
    "raw_range_id",
    "subject_id",
    "session_timestamp",
    "channel_name",
    "dataset_name",
    "dataset_version",
]


def parse_sha256s_file(path: Path) -> dict[str, str]:
    """Parse SHA256SUMS.txt into path -> sha256."""

    expected: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        sha, rel = stripped.split(maxsplit=1)
        expected[rel.strip()] = sha
    return expected


def npz_scalar(value: Any) -> Any:
    """Convert scalar NPZ arrays to Python values."""

    array = np.asarray(value)
    if array.shape == ():
        return array.item()
    if array.size == 1:
        return array.reshape(-1)[0].item()
    return array


def snapshot_hash_results(root: Path, config: dict[str, Any]) -> list[dict[str, Any]]:
    """Validate all metadata files listed in SHA256SUMS.txt."""

    sha_path = resolve_path(root, config["input"]["snapshot_sha256_file"])
    expected = parse_sha256s_file(sha_path)
    results = []
    for rel_path, expected_sha in expected.items():
        path = resolve_path(root, rel_path)
        actual_sha = sha256_file(path) if path.exists() else ""
        results.append(
            {
                "path": rel_path,
                "expected_sha256": expected_sha,
                "actual_sha256": actual_sha,
                "exists": path.exists(),
                "sha256_match": path.exists() and actual_sha == expected_sha,
            }
        )
    return results


def available_extraction_rows(config: dict[str, Any], rows: list[dict[str, str]]) -> list[dict[str, str]]:
    """Filter extraction rows to available statuses configured for preprocessing."""

    allowed = set(config["input"].get("allowed_extraction_statuses", ["success", "skipped_existing"]))
    return [row for row in rows if row.get("extraction_status") in allowed]


def select_extraction_rows(
    rows: list[dict[str, str]],
    subject_id: str | None = None,
    limit_raw_ranges: int | None = None,
) -> list[dict[str, str]]:
    """Select rows for smoke validation/preprocessing."""

    selected = [row for row in rows if not subject_id or row.get("subject_id") == subject_id]
    if limit_raw_ranges is not None:
        selected = selected[:limit_raw_ranges]
    return selected


def raw_npz_path(root: Path, row: dict[str, str]) -> Path:
    """Resolve a raw NPZ path from a reconstruction manifest row."""

    rel = Path(row["output_npz_path"])
    if rel.parts and rel.parts[0] == "data":
        return root / "dataset" / "SigD" / rel
    return resolve_path(root, rel)


def validate_npz_file(root: Path, row: dict[str, str], verify_hash: bool) -> dict[str, Any]:
    """Validate one raw NPZ against manifest provenance."""

    path = raw_npz_path(root, row)
    result: dict[str, Any] = {
        "raw_range_id": row.get("raw_range_id", ""),
        "path": str(path.relative_to(root)) if path.exists() else str(path),
        "exists": path.exists(),
        "hash_checked": verify_hash,
        "sha256_match": None,
        "fields_ok": False,
        "failure_reason": "",
    }
    if not path.exists():
        result["failure_reason"] = "missing_npz"
        return result

    if verify_hash:
        actual = sha256_file(path)
        expected = row.get("npz_sha256", "")
        result["actual_sha256"] = actual
        result["expected_sha256"] = expected
        result["sha256_match"] = bool(expected and actual == expected)
        if not result["sha256_match"]:
            result["failure_reason"] = "npz_sha256_mismatch"
            return result

    try:
        with np.load(path, allow_pickle=False) as data:
            missing = [field for field in REQUIRED_NPZ_FIELDS if field not in data]
            if missing:
                result["failure_reason"] = "missing_npz_fields:" + ",".join(missing)
                return result
            raw_range_id = str(npz_scalar(data["raw_range_id"]))
            dataset_name = str(npz_scalar(data["dataset_name"]))
            channel_name = str(npz_scalar(data["channel_name"]))
            if raw_range_id != row.get("raw_range_id"):
                result["failure_reason"] = "raw_range_id_mismatch"
                return result
            if dataset_name != "SigD-Core":
                result["failure_reason"] = "dataset_name_mismatch"
                return result
            if channel_name != "PLETH":
                result["failure_reason"] = "channel_name_mismatch"
                return result
            ppg = np.asarray(data["ppg"])
            if ppg.ndim != 1 or ppg.size == 0:
                result["failure_reason"] = "invalid_ppg_array"
                return result
            result["fields_ok"] = True
            result["fs"] = float(npz_scalar(data["fs"]))
            result["samples"] = int(ppg.size)
            return result
    except Exception as exc:
        result["failure_reason"] = f"npz_load_error:{type(exc).__name__}:{exc}"
        return result


def validate_snapshot(
    root: Path,
    config: dict[str, Any],
    *,
    limit_raw_ranges: int | None = None,
    subject_id: str | None = None,
    verify_all_npz_hashes: bool = False,
    verify_selected_npz_hashes: bool = False,
    output_path: Path | None = None,
) -> dict[str, Any]:
    """Validate fixed snapshot metadata and selected/all NPZ provenance."""

    hash_results = snapshot_hash_results(root, config)
    snapshot_hash_valid = all(item["sha256_match"] for item in hash_results)
    if not snapshot_hash_valid:
        payload = {
            "validation_datetime_utc": utc_now_iso(),
            "snapshot_valid": False,
            "snapshot_hash_results": hash_results,
            "validation_failures": ["snapshot_metadata_sha256_mismatch"],
        }
        if output_path:
            write_json(output_path, payload)
        return payload

    extraction_manifest = resolve_path(root, config["input"]["extraction_manifest"])
    extraction_rows = read_csv_rows(extraction_manifest)
    available_rows = available_extraction_rows(config, extraction_rows)
    selected_rows = select_extraction_rows(available_rows, subject_id, limit_raw_ranges)

    verify_npz_hash = verify_all_npz_hashes or verify_selected_npz_hashes
    rows_for_npz_check = available_rows if verify_all_npz_hashes else selected_rows
    npz_results = [
        validate_npz_file(root, row, verify_npz_hash) for row in rows_for_npz_check
    ]

    audit_path = resolve_path(root, config["input"]["snapshot_dir"]) / "sigd_core_audit_summary.json"
    audit_summary = read_json(audit_path)
    extraction_summary = audit_summary.get("extraction_summary", {})
    future_summary = audit_summary.get("future_window_availability_summary", {})
    status_dist = distribution(row.get("extraction_status", "") for row in extraction_rows)
    failures = [
        item["failure_reason"]
        for item in npz_results
        if item.get("failure_reason")
    ]
    payload = {
        "validation_datetime_utc": utc_now_iso(),
        "snapshot_valid": not failures,
        "snapshot_hash_results": hash_results,
        "extraction_manifest_rows": len(extraction_rows),
        "extraction_status_distribution": status_dist,
        "selected_npz_hash_verified_count": sum(
            1 for item in npz_results if item.get("hash_checked") and item.get("sha256_match")
        ),
        "missing_npz_count": sum(1 for item in npz_results if not item.get("exists")),
        "mismatched_npz_hash_count": sum(
            1 for item in npz_results if item.get("sha256_match") is False
        ),
        "input_dataset_name": config["input"]["dataset_name"],
        "input_dataset_version": "waveform_only_public_reconstruction_v1",
        "raw_available_subjects": extraction_summary.get("successful_subjects_with_available_npz"),
        "raw_available_sessions": extraction_summary.get("successful_sessions_with_available_npz"),
        "raw_available_ranges": extraction_summary.get("successful_raw_ranges_with_available_npz"),
        "raw_10s_eligible_subjects": future_summary.get("10s", {}).get(
            "subjects_eligible_for_future_cross_session_protocol"
        ),
        "raw_10s_candidate_windows": future_summary.get("10s", {}).get(
            "total_possible_nonoverlap_windows"
        ),
        "npz_validation_results": npz_results,
        "validation_failures": failures,
    }
    if output_path:
        write_json(output_path, payload)
    if not payload["snapshot_valid"]:
        logging.error("Snapshot validation failures: %s", failures[:10])
    return payload
