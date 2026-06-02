from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

SCRIPT_DIR = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPT_DIR))

from parse_sigd_annotations import (  # noqa: E402
    SourceVerificationError,
    build_candidate_record_paths,
    build_record_basename,
    make_raw_range_id,
    normalize_annotation_object,
    parse_hms_offset,
    parse_offset_range,
    sha256_file,
    subject_intermediate_dir,
    verify_annotation_source_hash,
)


def test_hms_offset_allows_hours_above_24() -> None:
    assert parse_hms_offset("29:40:10") == 29 * 3600 + 40 * 60 + 10


def test_offset_range_with_spaces() -> None:
    parsed = parse_offset_range("29: 40: 10-30: 11: 50")
    assert parsed.normalized_text == "29:40:10-30:11:50"
    assert parsed.start_seconds == 106810
    assert parsed.end_seconds == 108710
    assert parsed.duration_seconds == 1900


def test_offset_range_with_double_hyphen_from_official_notebook() -> None:
    parsed = parse_offset_range("60:16:10--60:19:50")
    assert parsed.normalized_text == "60:16:10-60:19:50"
    assert parsed.duration_seconds == 220


def test_offset_range_rejects_end_before_start() -> None:
    with pytest.raises(ValueError, match="end_not_after_start"):
        parse_offset_range("00:00:10-00:00:05")


def test_hms_offset_rejects_bad_minute_or_second() -> None:
    with pytest.raises(ValueError, match="minute_or_second_out_of_range"):
        parse_offset_range("00:60:00-00:61:00")


def test_subject_path_utilities() -> None:
    subject_id = "p007809"
    session_timestamp = "2136-08-31-17-33"
    assert subject_intermediate_dir(subject_id) == "p00"
    assert (
        build_record_basename(subject_id, session_timestamp)
        == "p007809-2136-08-31-17-33"
    )
    assert build_candidate_record_paths(subject_id, session_timestamp) == [
        "matched/p00/p007809/p007809-2136-08-31-17-33",
        "p00/p007809/p007809-2136-08-31-17-33",
    ]


def test_raw_range_id_is_deterministic_and_unique_by_index() -> None:
    first = make_raw_range_id("p007809", "2136-08-31-17-33", 0)
    second = make_raw_range_id("p007809", "2136-08-31-17-33", 0)
    third = make_raw_range_id("p007809", "2136-08-31-17-33", 1)
    assert first == second
    assert first != third
    assert first == "p007809_2136-08-31-17-33_r000"


def test_annotation_row_normalization_from_readme_mock() -> None:
    annotation = {
        "p007809": {
            "2136-08-31-17-33": [
                "29: 40: 10-30: 11: 50",
                "30: 18: 40-30: 24: 20",
            ],
            "2137-05-28-16-04": ["51: 26: 40-51: 29: 00"],
        }
    }
    rows = normalize_annotation_object(annotation)
    assert len(rows) == 3
    row = rows[0]
    required_columns = {
        "raw_range_id",
        "subject_id",
        "session_timestamp",
        "annotation_range_index_within_session",
        "offset_start_seconds",
        "offset_end_seconds",
        "requested_duration_seconds",
        "record_candidate_path_1",
        "record_candidate_path_2",
    }
    assert required_columns.issubset(row)
    assert row["annotation_parse_status"] == "success"
    assert row["requested_duration_seconds"] == 1900


def test_source_hash_verification_passes_when_locked_hash_matches(tmp_path: Path) -> None:
    annotation = tmp_path / "Extracted_signal_records.pl"
    annotation.write_bytes(b"official annotation bytes")
    manifest = tmp_path / "source_manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "official_files": [
                    {
                        "filename": "Extracted_signal_records.pl",
                        "sha256": sha256_file(annotation),
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    assert verify_annotation_source_hash(annotation, manifest) == sha256_file(annotation)


def test_source_hash_verification_rejects_mismatch(tmp_path: Path) -> None:
    annotation = tmp_path / "Extracted_signal_records.pl"
    annotation.write_bytes(b"changed bytes")
    manifest = tmp_path / "source_manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "official_files": [
                    {
                        "filename": "Extracted_signal_records.pl",
                        "sha256": "0" * 64,
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(SourceVerificationError, match="SHA256 mismatch"):
        verify_annotation_source_hash(annotation, manifest)
