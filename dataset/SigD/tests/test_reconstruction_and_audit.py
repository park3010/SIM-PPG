from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPT_DIR))

import audit_sigd_core as audit  # noqa: E402
import reconstruct_sigd_core as recon  # noqa: E402


def minimal_config() -> dict[str, str]:
    return {
        "signal_channel": "PLETH",
        "physionet_pn_dir": "mimic3wdb-matched/1.0",
        "physionet_database": "mimic3wdb-matched",
        "physionet_version": "1.0",
        "dataset_version": "waveform_only_public_reconstruction_v1",
    }


def annotation_row() -> dict[str, str]:
    return {
        "raw_range_id": "p000001_2100-01-01-00-00_r000",
        "subject_id": "p000001",
        "session_timestamp": "2100-01-01-00-00",
        "session_index_within_subject": "0",
        "annotation_range_index_within_session": "0",
        "requested_channel_name": "PLETH",
        "record_candidate_path_1": "matched/p00/p000001/p000001-2100-01-01-00-00",
        "record_candidate_path_2": "p00/p000001/p000001-2100-01-01-00-00",
        "offset_start_seconds": "0",
        "offset_end_seconds": "4",
        "requested_duration_seconds": "4",
        "annotation_parse_status": "success",
    }


def write_test_npz(root: Path, row: dict[str, str], ppg: np.ndarray, fs: float) -> Path:
    out_path = recon.output_npz_path(root, row)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_path,
        ppg=ppg.astype(np.float32),
        fs=np.float64(fs),
        raw_range_id=np.asarray(row["raw_range_id"]),
        subject_id=np.asarray(row["subject_id"]),
        session_timestamp=np.asarray(row["session_timestamp"]),
        annotation_range_index_within_session=np.int64(0),
        offset_start_seconds=np.int64(0),
        offset_end_seconds=np.int64(4),
        requested_duration_seconds=np.float64(4),
        sampfrom=np.int64(0),
        sampto=np.int64(int(4 * fs)),
        resolved_wfdb_record_name=np.asarray("p00/p000001/p000001-2100-01-01-00-00"),
        channel_name=np.asarray("PLETH"),
        source_database=np.asarray("mimic3wdb-matched"),
        source_version=np.asarray("1.0"),
        dataset_name=np.asarray("SigD-Core"),
        dataset_version=np.asarray("waveform_only_public_reconstruction_v1"),
    )
    return out_path


def test_dry_run_manifest_does_not_overwrite_actual_extraction_manifest(
    tmp_path: Path,
) -> None:
    actual = tmp_path / "dataset" / "SigD" / "metadata" / "sigd_extraction_manifest.csv"
    actual.parent.mkdir(parents=True)
    actual.write_text("actual manifest sentinel\n", encoding="utf-8")

    dry_path = recon.output_manifest_path(tmp_path, dry_run=True, header_check=False)
    recon.write_csv(dry_path, [], recon.EXTRACTION_COLUMNS)

    assert actual.read_text(encoding="utf-8") == "actual manifest sentinel\n"
    assert dry_path.name == "sigd_dry_run_manifest.csv"


def test_header_check_manifest_does_not_overwrite_actual_extraction_manifest(
    tmp_path: Path,
) -> None:
    actual = tmp_path / "dataset" / "SigD" / "metadata" / "sigd_extraction_manifest.csv"
    actual.parent.mkdir(parents=True)
    actual.write_text("actual manifest sentinel\n", encoding="utf-8")

    header_path = recon.output_manifest_path(tmp_path, dry_run=False, header_check=True)
    recon.write_csv(header_path, [], recon.EXTRACTION_COLUMNS)

    assert actual.read_text(encoding="utf-8") == "actual manifest sentinel\n"
    assert header_path.name == "sigd_header_check_manifest.csv"


def test_actual_extraction_manifest_merge_preserves_unselected_rows() -> None:
    existing = [
        {"raw_range_id": "A", "extraction_status": "success", "fs": "125"},
        {"raw_range_id": "B", "extraction_status": "failed", "fs": ""},
        {"raw_range_id": "C", "extraction_status": "success", "fs": "125"},
    ]
    new = [{"raw_range_id": "B", "extraction_status": "success", "fs": "250"}]
    annotation = [
        {"raw_range_id": "A"},
        {"raw_range_id": "B"},
        {"raw_range_id": "C"},
    ]

    merged = recon.merge_actual_extraction_results(existing, new, annotation)

    assert [row["raw_range_id"] for row in merged] == ["A", "B", "C"]
    assert merged[0]["extraction_status"] == "success"
    assert merged[1]["extraction_status"] == "success"
    assert merged[1]["fs"] == "250"
    assert merged[2]["extraction_status"] == "success"


def test_failed_records_csv_uses_merged_current_state(tmp_path: Path) -> None:
    current_state = [
        {
            "raw_range_id": "A",
            "subject_id": "p1",
            "session_timestamp": "s1",
            "extraction_status": "failed",
            "failure_reason": "record_path_not_resolved",
        },
        {
            "raw_range_id": "B",
            "subject_id": "p2",
            "session_timestamp": "s2",
            "extraction_status": "success",
            "failure_reason": "",
        },
    ]
    path = recon.write_failed_records_from_current_state(tmp_path, current_state)
    failed = recon.read_csv_rows(path)
    assert [row["raw_range_id"] for row in failed] == ["A"]

    retried_state = [
        {**current_state[0], "extraction_status": "success", "failure_reason": ""},
        current_state[1],
    ]
    recon.write_failed_records_from_current_state(tmp_path, retried_state)
    assert recon.read_csv_rows(path) == []


def test_existing_npz_stats_are_recovered_without_previous_success_manifest(
    tmp_path: Path,
) -> None:
    row = annotation_row()
    ppg = np.asarray([0.0, 0.0, 1.0, np.nan, np.inf, 2.0, 2.0, 3.0], dtype=np.float32)
    out_path = write_test_npz(tmp_path, row, ppg, fs=2.0)

    result = recon.process_extraction(
        tmp_path,
        row,
        minimal_config(),
        source_manifest_version="test-source",
        overwrite=False,
        resume=True,
        previous_rows={},
    )

    assert out_path.exists()
    assert result["extraction_status"] == "skipped_existing"
    assert result["fs"] == 2.0
    assert result["extracted_samples"] == 8
    assert result["extracted_duration_seconds"] == 4.0
    assert result["nan_count"] == 1
    assert result["inf_count"] == 1
    assert result["npz_sha256"]
    assert result["ppg_mean"] != ""


def test_existing_npz_with_previous_failed_manifest_is_not_reused(tmp_path: Path) -> None:
    row = annotation_row()
    write_test_npz(tmp_path, row, np.asarray([0.0, 1.0], dtype=np.float32), fs=1.0)

    result = recon.process_extraction(
        tmp_path,
        row,
        minimal_config(),
        source_manifest_version="test-source",
        overwrite=False,
        resume=True,
        previous_rows={row["raw_range_id"]: {"extraction_status": "failed"}},
    )

    assert result["extraction_status"] == "failed"
    assert result["failure_reason"] == "existing_npz_with_failed_previous_status"


def test_failed_extraction_created_npz_is_removed(tmp_path: Path, monkeypatch) -> None:
    row = annotation_row()

    class DummyHeader:
        fs = 1.0
        sig_len = 10
        sig_name = ["PLETH"]

    monkeypatch.setattr(
        recon,
        "resolve_remote_record_name",
        lambda candidate_paths, pn_dir: recon.ResolutionResult(
            success=True,
            resolved_wfdb_record_name="p00/p000001/p000001-2100-01-01-00-00",
            wfdb_record_name="p000001-2100-01-01-00-00",
            resolved_pn_dir="mimic3wdb-matched/1.0/p00/p000001",
            header=DummyHeader(),
        ),
    )
    monkeypatch.setattr(
        recon,
        "read_pleth_range",
        lambda *args, **kwargs: (np.asarray([0.1, 0.2], dtype=np.float32), {"sig_name": ["PLETH"]}),
    )

    def write_then_fail(output_path: Path, *args, **kwargs) -> None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(output_path, ppg=np.asarray([0.1], dtype=np.float32))
        raise RuntimeError("forced save failure after write")

    monkeypatch.setattr(recon, "save_raw_range_npz", write_then_fail)

    result = recon.process_extraction(
        tmp_path,
        row,
        minimal_config(),
        source_manifest_version="test-source",
        overwrite=False,
        resume=False,
        previous_rows={},
    )

    assert result["extraction_status"] == "failed"
    assert result["failure_reason"].startswith("save_error")
    assert not recon.output_npz_path(tmp_path, row).exists()


def test_audit_subject_window_eligibility_for_5_10_30_seconds() -> None:
    available = [
        {
            "subject_id": "p1",
            "session_timestamp": "2100-01-01-00-00",
            "extracted_duration_seconds": "10",
            "possible_nonoverlap_windows_5s": 2,
            "possible_nonoverlap_windows_10s": 1,
            "possible_nonoverlap_windows_30s": 0,
        },
        {
            "subject_id": "p1",
            "session_timestamp": "2100-01-02-00-00",
            "extracted_duration_seconds": "12",
            "possible_nonoverlap_windows_5s": 2,
            "possible_nonoverlap_windows_10s": 1,
            "possible_nonoverlap_windows_30s": 0,
        },
    ]
    session_rows = audit.build_session_summary(available, [5, 10, 30])
    subject_rows = audit.build_subject_summary(session_rows, [5, 10, 30])

    assert subject_rows[0]["eligible_for_future_5s_cross_session_protocol"] is True
    assert subject_rows[0]["eligible_for_future_10s_cross_session_protocol"] is True
    assert subject_rows[0]["eligible_for_future_30s_cross_session_protocol"] is False


def test_interval_pair_protocol_flags_follow_session_window_availability() -> None:
    session_rows = [
        {
            "subject_id": "p1",
            "session_timestamp": "2100-01-01-00-00",
            "successful_raw_ranges": 1,
            "total_duration_seconds": 10.0,
            "possible_windows_5s": 2,
            "possible_windows_10s": 1,
            "possible_windows_30s": 0,
        },
        {
            "subject_id": "p1",
            "session_timestamp": "2100-01-02-00-00",
            "successful_raw_ranges": 1,
            "total_duration_seconds": 6.0,
            "possible_windows_5s": 1,
            "possible_windows_10s": 0,
            "possible_windows_30s": 0,
        },
    ]
    pairs = audit.build_interval_pairs(session_rows, [5, 10, 30])

    assert pairs[0]["supports_raw_cross_session_pair"] is True
    assert pairs[0]["supports_future_5s_cross_session_protocol"] is True
    assert pairs[0]["supports_future_10s_cross_session_protocol"] is False
    assert pairs[0]["supports_future_30s_cross_session_protocol"] is False
    assert pairs[0]["supports_any_candidate_cross_session_protocol"] is True
    assert pairs[0]["supports_future_cross_session_verification"] is False


def test_primary_10s_generic_flag_can_be_false_when_only_5s_is_supported() -> None:
    session_rows = [
        {
            "subject_id": "p1",
            "session_timestamp": "2100-01-01-00-00",
            "successful_raw_ranges": 1,
            "total_duration_seconds": 6.0,
            "possible_windows_5s": 1,
            "possible_windows_10s": 0,
            "possible_windows_30s": 0,
        },
        {
            "subject_id": "p1",
            "session_timestamp": "2100-01-02-00-00",
            "successful_raw_ranges": 1,
            "total_duration_seconds": 7.0,
            "possible_windows_5s": 1,
            "possible_windows_10s": 0,
            "possible_windows_30s": 0,
        },
    ]

    pairs = audit.build_interval_pairs(session_rows, [5, 10, 30], 10)

    assert pairs[0]["supports_future_5s_cross_session_protocol"] is True
    assert pairs[0]["supports_future_10s_cross_session_protocol"] is False
    assert pairs[0]["supports_future_cross_session_verification"] is False
    assert pairs[0]["supports_any_candidate_cross_session_protocol"] is True
