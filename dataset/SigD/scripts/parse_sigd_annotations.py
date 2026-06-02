#!/usr/bin/env python3
"""Parse official SigD annotation ranges into a SigD-Core manifest.

The official ``Extracted_signal_records.pl`` file is treated as a trusted
runtime source only after its SHA256 hash matches ``source_manifest.json``.
The file may be a pickle-like serialized object, so this script never loads an
arbitrary user-provided annotation file without source-lock verification.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import pickle
import re
import statistics
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

try:
    import yaml
except ImportError:  # pragma: no cover - exercised only in minimal envs
    yaml = None


DATASET_NAME = "SigD-Core"
DATASET_FULL_NAME = "SigD-Core (waveform-only public reconstruction)"
DATASET_VERSION = "waveform_only_public_reconstruction_v1"
SOURCE_MANIFEST_NAME = "source_manifest.json"
ANNOTATION_FILENAME = "Extracted_signal_records.pl"

ANNOTATION_COLUMNS = [
    "raw_range_id",
    "subject_id",
    "session_timestamp",
    "session_index_within_subject",
    "annotation_range_index_within_session",
    "offset_text_original",
    "offset_text_normalized",
    "offset_start_text",
    "offset_end_text",
    "offset_start_seconds",
    "offset_end_seconds",
    "requested_duration_seconds",
    "intermediate_dir",
    "record_basename",
    "record_candidate_path_1",
    "record_candidate_path_2",
    "physionet_database",
    "physionet_version",
    "physionet_pn_dir",
    "requested_channel_name",
    "annotation_parse_status",
    "annotation_parse_failure_reason",
]

DEFAULT_CONFIG = {
    "dataset_name": DATASET_NAME,
    "dataset_version": DATASET_VERSION,
    "physionet_database": "mimic3wdb-matched",
    "physionet_version": "1.0",
    "physionet_pn_dir": "mimic3wdb-matched/1.0",
    "signal_channel": "PLETH",
}


class SourceVerificationError(RuntimeError):
    """Raised when the local annotation source does not match the lock file."""


@dataclass(frozen=True)
class ParsedOffsetRange:
    """Normalized representation of an annotated raw waveform range."""

    original_text: str
    normalized_text: str
    start_text: str
    end_text: str
    start_seconds: int
    end_seconds: int
    duration_seconds: int


def utc_now_iso() -> str:
    """Return a UTC ISO-8601 timestamp with seconds precision."""

    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def detect_root(root_arg: str | None = None) -> Path:
    """Resolve the project root from an explicit path, cwd, or script location."""

    if root_arg:
        return Path(root_arg).expanduser().resolve()

    candidates: list[Path] = []
    cwd = Path.cwd().resolve()
    candidates.extend([cwd, *cwd.parents])
    script_root = Path(__file__).resolve().parents[3]
    candidates.extend([script_root, *script_root.parents])

    for candidate in candidates:
        if (candidate / "dataset" / "SigD").exists():
            return candidate
        if (candidate / "docs").exists() and not (candidate / "dataset").exists():
            return candidate

    raise SystemExit(
        "Could not detect sim_ppg root. Pass --root PATH from the project root."
    )


def sigd_dir(root: Path) -> Path:
    """Return the SigD-Core dataset directory under the project root."""

    return root / "dataset" / "SigD"


def setup_logging(root: Path, log_filename: str, verbose: bool = False) -> None:
    """Configure console and file logging for a SigD-Core script."""

    logs_dir = sigd_dir(root) / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)sZ %(levelname)s %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
        handlers=[
            logging.FileHandler(logs_dir / log_filename, encoding="utf-8"),
            logging.StreamHandler(),
        ],
        force=True,
    )
    logging.Formatter.converter = time_gmtime


def time_gmtime(*_: Any) -> Any:
    """Formatter hook that makes logging timestamps UTC."""

    import time

    return time.gmtime()


def load_config(root: Path) -> dict[str, Any]:
    """Load SigD-Core config, falling back to conservative defaults."""

    config_path = sigd_dir(root) / "config" / "sigd_core.yaml"
    if not config_path.exists() or yaml is None:
        return dict(DEFAULT_CONFIG)
    with config_path.open("r", encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle) or {}
    config = dict(DEFAULT_CONFIG)
    config.update(loaded)
    return config


def sha256_file(path: Path) -> str:
    """Compute SHA256 for a local file."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    """Read a UTF-8 JSON object from disk."""

    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def expected_file_hash_from_manifest(
    source_manifest: dict[str, Any], filename: str = ANNOTATION_FILENAME
) -> str:
    """Return the source-locked SHA256 for a required official file."""

    for item in source_manifest.get("official_files", []):
        if item.get("filename") == filename:
            sha = item.get("sha256")
            if not sha:
                break
            return str(sha)
    raise SourceVerificationError(
        f"{filename} is not present in source manifest official_files."
    )


def verify_annotation_source_hash(
    annotation_path: Path,
    source_manifest_path: Path,
    filename: str = ANNOTATION_FILENAME,
) -> str:
    """Verify annotation SHA256 against source_manifest.json.

    Returns the current SHA256 if it matches the source lock.
    """

    if not source_manifest_path.exists():
        raise SourceVerificationError(
            f"Missing source manifest: {source_manifest_path}. "
            "Run setup_sigd_source.py first."
        )
    if not annotation_path.exists():
        raise SourceVerificationError(f"Missing annotation file: {annotation_path}")

    source_manifest = load_json(source_manifest_path)
    expected_sha = expected_file_hash_from_manifest(source_manifest, filename)
    current_sha = sha256_file(annotation_path)
    if current_sha != expected_sha:
        raise SourceVerificationError(
            f"Annotation SHA256 mismatch for {annotation_path}. "
            f"expected={expected_sha} actual={current_sha}"
        )
    return current_sha


def parse_hms_offset(offset_text: str) -> int:
    """Convert an HH:MM:SS offset to seconds, allowing hour values above 24."""

    cleaned = re.sub(r"\s+", "", offset_text)
    match = re.fullmatch(r"(\d+):(\d{1,2}):(\d{1,2})", cleaned)
    if not match:
        raise ValueError(f"invalid_hms_offset:{offset_text}")
    hours, minutes, seconds = (int(part) for part in match.groups())
    if minutes > 59 or seconds > 59:
        raise ValueError(f"minute_or_second_out_of_range:{offset_text}")
    return hours * 3600 + minutes * 60 + seconds


def parse_offset_range(range_text: str) -> ParsedOffsetRange:
    """Normalize and parse an annotated raw-range offset string."""

    original = str(range_text)
    normalized = re.sub(r"\s+", "", original)
    normalized = re.sub(r"(?<=\d)--+(?=\d)", "-", normalized)
    parts = normalized.split("-")
    if len(parts) != 2:
        raise ValueError(f"invalid_range_format:{original}")
    start_text, end_text = parts
    start_seconds = parse_hms_offset(start_text)
    end_seconds = parse_hms_offset(end_text)
    if end_seconds <= start_seconds:
        raise ValueError("end_not_after_start")
    return ParsedOffsetRange(
        original_text=original,
        normalized_text=f"{start_text}-{end_text}",
        start_text=start_text,
        end_text=end_text,
        start_seconds=start_seconds,
        end_seconds=end_seconds,
        duration_seconds=end_seconds - start_seconds,
    )


def subject_intermediate_dir(subject_id: str) -> str:
    """Return the PhysioNet intermediate directory for a subject id."""

    match = re.fullmatch(r"p(\d{2})\d+", subject_id)
    if not match:
        raise ValueError(f"invalid_subject_id:{subject_id}")
    return f"p{match.group(1)}"


def build_record_basename(subject_id: str, session_timestamp: str) -> str:
    """Build the WFDB record basename from subject and surrogate timestamp."""

    return f"{subject_id}-{session_timestamp}"


def build_candidate_record_paths(subject_id: str, session_timestamp: str) -> list[str]:
    """Build both documented candidate remote record paths for WFDB lookup."""

    intermediate_dir = subject_intermediate_dir(subject_id)
    basename = build_record_basename(subject_id, session_timestamp)
    return [
        f"matched/{intermediate_dir}/{subject_id}/{basename}",
        f"{intermediate_dir}/{subject_id}/{basename}",
    ]


def make_raw_range_id(
    subject_id: str, session_timestamp: str, annotation_range_index: int
) -> str:
    """Create a deterministic filename-safe raw_range_id."""

    return f"{subject_id}_{session_timestamp}_r{annotation_range_index:03d}"


def _coerce_range_texts(value: Any) -> list[str]:
    """Coerce common annotation leaf forms into a list of offset strings."""

    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, dict):
        values: list[str] = []
        for key in sorted(value, key=lambda item: str(item)):
            values.extend(_coerce_range_texts(value[key]))
        return values
    if isinstance(value, (list, tuple, set)):
        values = []
        for item in value:
            values.extend(_coerce_range_texts(item))
        return values
    return [str(value)]


def iter_annotation_entries(annotation: Any) -> Iterable[tuple[str, str, list[str]]]:
    """Yield subject, session, raw-range texts from a SigD annotation object."""

    if not isinstance(annotation, dict):
        raise TypeError(
            "Expected top-level annotation object to be a dict keyed by subject_id."
        )

    for subject_key in sorted(annotation, key=lambda item: str(item)):
        subject_id = str(subject_key)
        sessions = annotation[subject_key]
        if not isinstance(sessions, dict):
            logging.warning(
                "Skipping subject %s because its value is %s, not dict",
                subject_id,
                type(sessions).__name__,
            )
            continue
        for session_key in sorted(sessions, key=lambda item: str(item)):
            session_timestamp = str(session_key)
            yield subject_id, session_timestamp, _coerce_range_texts(
                sessions[session_key]
            )


def normalize_annotation_object(
    annotation: Any, config: dict[str, Any] | None = None
) -> list[dict[str, Any]]:
    """Normalize the official annotation object to manifest rows."""

    cfg = dict(DEFAULT_CONFIG)
    if config:
        cfg.update(config)

    rows: list[dict[str, Any]] = []
    by_subject: dict[str, list[tuple[str, list[str]]]] = defaultdict(list)
    for subject_id, session_timestamp, range_texts in iter_annotation_entries(annotation):
        by_subject[subject_id].append((session_timestamp, range_texts))

    for subject_id in sorted(by_subject):
        sessions = sorted(by_subject[subject_id], key=lambda item: item[0])
        for session_index, (session_timestamp, range_texts) in enumerate(sessions):
            try:
                intermediate_dir = subject_intermediate_dir(subject_id)
                record_basename = build_record_basename(subject_id, session_timestamp)
                candidate_paths = build_candidate_record_paths(
                    subject_id, session_timestamp
                )
                path_failure = ""
            except ValueError as exc:
                intermediate_dir = ""
                record_basename = build_record_basename(subject_id, session_timestamp)
                candidate_paths = ["", ""]
                path_failure = str(exc)

            for range_index, offset_text in enumerate(range_texts):
                raw_range_id = make_raw_range_id(
                    subject_id, session_timestamp, range_index
                )
                row: dict[str, Any] = {
                    "raw_range_id": raw_range_id,
                    "subject_id": subject_id,
                    "session_timestamp": session_timestamp,
                    "session_index_within_subject": session_index,
                    "annotation_range_index_within_session": range_index,
                    "offset_text_original": offset_text,
                    "offset_text_normalized": "",
                    "offset_start_text": "",
                    "offset_end_text": "",
                    "offset_start_seconds": "",
                    "offset_end_seconds": "",
                    "requested_duration_seconds": "",
                    "intermediate_dir": intermediate_dir,
                    "record_basename": record_basename,
                    "record_candidate_path_1": candidate_paths[0],
                    "record_candidate_path_2": candidate_paths[1],
                    "physionet_database": cfg["physionet_database"],
                    "physionet_version": str(cfg["physionet_version"]),
                    "physionet_pn_dir": cfg["physionet_pn_dir"],
                    "requested_channel_name": cfg["signal_channel"],
                    "annotation_parse_status": "success",
                    "annotation_parse_failure_reason": "",
                }
                if path_failure:
                    row["annotation_parse_status"] = "failed"
                    row["annotation_parse_failure_reason"] = path_failure
                else:
                    try:
                        parsed = parse_offset_range(offset_text)
                        row.update(
                            {
                                "offset_text_normalized": parsed.normalized_text,
                                "offset_start_text": parsed.start_text,
                                "offset_end_text": parsed.end_text,
                                "offset_start_seconds": parsed.start_seconds,
                                "offset_end_seconds": parsed.end_seconds,
                                "requested_duration_seconds": parsed.duration_seconds,
                            }
                        )
                    except ValueError as exc:
                        row["annotation_parse_status"] = "failed"
                        row["annotation_parse_failure_reason"] = str(exc)
                rows.append(row)
    return rows


def describe_annotation_structure(annotation: Any) -> dict[str, Any]:
    """Return a compact structure summary without dumping the full object."""

    summary: dict[str, Any] = {
        "top_level_object_type": type(annotation).__name__,
        "top_level_key_count": len(annotation) if isinstance(annotation, dict) else None,
        "sample_subjects": [],
    }
    if isinstance(annotation, dict):
        for subject_key in list(sorted(annotation, key=lambda item: str(item)))[:3]:
            sessions = annotation[subject_key]
            sample: dict[str, Any] = {
                "subject_id": str(subject_key),
                "value_type": type(sessions).__name__,
                "session_count": len(sessions) if isinstance(sessions, dict) else None,
                "sample_sessions": [],
            }
            if isinstance(sessions, dict):
                for session_key in list(sorted(sessions, key=lambda item: str(item)))[:3]:
                    ranges = _coerce_range_texts(sessions[session_key])
                    sample["sample_sessions"].append(
                        {
                            "session_timestamp": str(session_key),
                            "raw_range_count": len(ranges),
                            "range_value_type": type(sessions[session_key]).__name__,
                        }
                    )
            summary["sample_subjects"].append(sample)
    return summary


def load_verified_annotation_pickle(annotation_path: Path) -> Any:
    """Load the verified official pickle-like annotation object."""

    with annotation_path.open("rb") as handle:
        try:
            return pickle.load(handle)
        except UnicodeDecodeError:
            handle.seek(0)
            return pickle.load(handle, encoding="latin1")


def write_csv(path: Path, rows: list[dict[str, Any]], columns: list[str]) -> None:
    """Write rows to a UTF-8 CSV with fixed column order."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column, "") for column in columns})


def distribution(values: Iterable[Any]) -> dict[str, int]:
    """Return a JSON-friendly counter keyed by stringified values."""

    return {str(key): int(value) for key, value in sorted(Counter(values).items())}


def build_annotation_summary(
    rows: list[dict[str, Any]],
    annotation: Any,
    annotation_sha256: str,
    source_manifest_path: Path,
) -> dict[str, Any]:
    """Build the annotation summary JSON payload."""

    subjects = {row["subject_id"] for row in rows}
    sessions = {(row["subject_id"], row["session_timestamp"]) for row in rows}
    success_rows = [row for row in rows if row["annotation_parse_status"] == "success"]
    failure_rows = [row for row in rows if row["annotation_parse_status"] != "success"]
    durations = [
        float(row["requested_duration_seconds"])
        for row in success_rows
        if row["requested_duration_seconds"] != ""
    ]
    sessions_by_subject: dict[str, set[str]] = defaultdict(set)
    ranges_by_session: Counter[tuple[str, str]] = Counter()
    for row in rows:
        sessions_by_subject[row["subject_id"]].add(row["session_timestamp"])
        ranges_by_session[(row["subject_id"], row["session_timestamp"])] += 1

    total_duration = float(sum(durations))
    return {
        "dataset_name": DATASET_NAME,
        "dataset_version": DATASET_VERSION,
        "generated_datetime_utc": utc_now_iso(),
        "source_manifest_path": str(source_manifest_path.relative_to(source_manifest_path.parents[1])),
        "annotation_file_sha256": annotation_sha256,
        "top_level_object_type": type(annotation).__name__,
        "total_annotation_subjects": len(subjects),
        "total_annotation_sessions": len(sessions),
        "total_annotation_raw_ranges": len(rows),
        "total_requested_duration_seconds": total_duration,
        "total_requested_duration_hours": total_duration / 3600.0,
        "subjects_with_at_least_2_annotated_sessions": sum(
            1 for subject_sessions in sessions_by_subject.values() if len(subject_sessions) >= 2
        ),
        "session_count_distribution": distribution(
            len(subject_sessions) for subject_sessions in sessions_by_subject.values()
        ),
        "raw_range_count_distribution": distribution(ranges_by_session.values()),
        "min_requested_duration_seconds": min(durations) if durations else None,
        "median_requested_duration_seconds": statistics.median(durations)
        if durations
        else None,
        "max_requested_duration_seconds": max(durations) if durations else None,
        "parsing_success_count": len(success_rows),
        "parsing_failure_count": len(failure_rows),
        "parsing_failure_reason_distribution": distribution(
            row["annotation_parse_failure_reason"] or "none" for row in failure_rows
        ),
    }


def dump_json(path: Path, payload: dict[str, Any]) -> None:
    """Write a UTF-8 JSON file with stable, readable formatting."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(
        description="Parse official SigD annotations into SigD-Core raw ranges."
    )
    parser.add_argument("--root", type=str, default=None, help="sim_ppg root path")
    parser.add_argument(
        "--annotation-path",
        type=str,
        default=None,
        help="Optional path to Extracted_signal_records.pl",
    )
    parser.add_argument("--verbose", action="store_true", help="Enable debug logging")
    return parser.parse_args()


def main() -> int:
    """CLI entry point."""

    args = parse_args()
    root = detect_root(args.root)
    setup_logging(root, "parse_sigd_annotations.log", args.verbose)
    config = load_config(root)

    annotation_path = (
        Path(args.annotation_path).expanduser().resolve()
        if args.annotation_path
        else sigd_dir(root) / "official" / "NasTul_SigD" / ANNOTATION_FILENAME
    )
    source_manifest_path = sigd_dir(root) / "metadata" / SOURCE_MANIFEST_NAME
    output_csv = sigd_dir(root) / "metadata" / "sigd_annotation_manifest.csv"
    summary_json = sigd_dir(root) / "metadata" / "annotation_summary.json"

    logging.info("Verifying annotation source hash: %s", annotation_path)
    annotation_sha256 = verify_annotation_source_hash(
        annotation_path, source_manifest_path
    )
    annotation = load_verified_annotation_pickle(annotation_path)

    structure = describe_annotation_structure(annotation)
    logging.info("Annotation structure: %s", json.dumps(structure, ensure_ascii=False))

    rows = normalize_annotation_object(annotation, config)
    write_csv(output_csv, rows, ANNOTATION_COLUMNS)
    summary = build_annotation_summary(
        rows, annotation, annotation_sha256, source_manifest_path
    )
    dump_json(summary_json, summary)

    logging.info(
        "Parsed %d raw ranges: success=%d failure=%d",
        len(rows),
        summary["parsing_success_count"],
        summary["parsing_failure_count"],
    )
    logging.info("Wrote %s and %s", output_csv, summary_json)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
