from __future__ import annotations

import sys
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))

from window_audit import interval_pair_rows, session_summary_rows, subject_summary_rows  # noqa: E402


def manifest_rows() -> list[dict[str, str]]:
    rows = []
    for session, count in [("2100-01-01-00-00", 3), ("2100-01-02-00-00", 1)]:
        for index in range(count):
            rows.append(
                {
                    "subject_id": "p1",
                    "session_timestamp": session,
                    "common_input_available": "True",
                    "model_input_available": "True",
                    "raw_range_processing_status": "success",
                    "svri_valid_mask": "True",
                    "sqi_valid_mask": "True",
                    "ipa_valid_mask": "False",
                }
            )
    rows.append(
        {
            "subject_id": "p2",
            "session_timestamp": "2100-01-01-00-00",
            "common_input_available": "False",
            "model_input_available": "False",
            "raw_range_processing_status": "success",
            "svri_valid_mask": "False",
            "sqi_valid_mask": "False",
            "ipa_valid_mask": "False",
        }
    )
    return rows


def test_session_available_window_counts_and_k_support() -> None:
    sessions = session_summary_rows(manifest_rows())
    first = [row for row in sessions if row["subject_id"] == "p1" and row["session_timestamp"] == "2100-01-01-00-00"][0]
    second = [row for row in sessions if row["subject_id"] == "p1" and row["session_timestamp"] == "2100-01-02-00-00"][0]
    assert first["common_available_windows"] == 3
    assert first["model_input_available_windows"] == 3
    assert first["supports_common_10s_k3"] is True
    assert first["supports_10s_k3"] is True
    assert second["supports_10s_k3"] is False


def test_subject_requires_two_sessions_for_retention() -> None:
    sessions = session_summary_rows(manifest_rows())
    subjects = subject_summary_rows(
        sessions,
        [{"subject_id": "p1", "eligible_for_future_10s_cross_session_protocol": "True"}],
    )
    p1 = subjects[0]
    assert p1["retained_after_preprocessing_k1"] is True
    assert p1["retained_after_preprocessing_k3"] is False


def test_interval_k_m_flags() -> None:
    sessions = session_summary_rows(manifest_rows())
    pairs = interval_pair_rows(
        sessions,
        [
            {
                "subject_id": "p1",
                "enrollment_session_timestamp": "2100-01-01-00-00",
                "probe_session_timestamp": "2100-01-02-00-00",
                "supports_future_10s_cross_session_protocol": "True",
            }
        ],
    )
    assert pairs[0]["supports_10s_k1_m1"] is True
    assert pairs[0]["supports_10s_k3_m1"] is True
    assert pairs[0]["supports_10s_k3_m3"] is False
    assert pairs[0]["retained_after_preprocessing"] is True


def test_raw_eligible_subject_can_be_lost_after_preprocessing() -> None:
    sessions = session_summary_rows(
        [
            {
                "subject_id": "p3",
                "session_timestamp": "2100-01-01-00-00",
                "common_input_available": "False",
                "model_input_available": "False",
                "raw_range_processing_status": "success",
            }
        ]
    )
    subjects = subject_summary_rows(
        sessions,
        [{"subject_id": "p3", "eligible_for_future_10s_cross_session_protocol": "True"}],
    )
    assert subjects[0]["raw_level_10s_eligible"] is True
    assert subjects[0]["retained_after_preprocessing_k1"] is False


def test_smoke_paths_do_not_equal_full_paths() -> None:
    smoke = Path("preprocessing/SigD/metadata/preprocessing_manifest_10s_smoke.csv")
    full = Path("preprocessing/SigD/metadata/preprocessing_manifest_10s.csv")
    assert smoke != full


def test_morphology_validity_does_not_control_common_eligibility() -> None:
    sessions = session_summary_rows(
        [
            {
                "subject_id": "p4",
                "session_timestamp": "2100-01-01-00-00",
                "common_input_available": "True",
                "model_input_available": "True",
                "raw_range_processing_status": "success",
                "svri_valid_mask": "False",
                "sqi_valid_mask": "False",
                "ipa_valid_mask": "False",
            }
        ]
    )
    assert sessions[0]["common_available_windows"] == 1
    assert sessions[0]["svri_valid_windows"] == 0
    assert sessions[0]["sqi_valid_windows"] == 0
    assert sessions[0]["ipa_valid_windows"] == 0
    assert sessions[0]["supports_common_10s_k1"] is True
