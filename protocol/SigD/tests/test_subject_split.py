from __future__ import annotations

import sys
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

from build_sigd_subject_split import split_summary, stratified_subject_split, subject_features  # noqa: E402


def config() -> dict:
    return {
        "protocol_id": "TEST_PROTOCOL",
        "input_protocol_id": "COMMON_PPG_10S_125HZ_V1",
        "split_seed": 42,
        "expected_subject_counts": {"train": 6, "val": 2, "test": 2},
        "morphology_validity_used_for_protocol": False,
    }


def feature_rows(n: int = 10) -> list[dict]:
    return [
        {
            "subject_id": f"p{i:06d}",
            "session_count": 2 + (i % 3),
            "total_common_available_windows": 50 + i,
            "primary_pair_count": 1 + (i % 2),
            "all_pair_count": 1 + (i % 4),
            "min_gap_days": float(i),
            "median_gap_days": float(i + 1),
            "max_gap_days": float(i + 2),
        }
        for i in range(n)
    ]


def test_subject_split_overlap_is_zero() -> None:
    rows = stratified_subject_split(feature_rows(), config())
    summary = split_summary(rows, config())
    assert summary["subject_overlap_check"]["passed"] is True
    assert summary["subject_overlap_check"]["train_val_overlap"] == 0
    assert summary["subject_overlap_check"]["train_test_overlap"] == 0
    assert summary["subject_overlap_check"]["val_test_overlap"] == 0


def test_subject_split_counts_match_expected() -> None:
    rows = stratified_subject_split(feature_rows(), config())
    counts = {split: sum(1 for row in rows if row["split"] == split) for split in ("train", "val", "test")}
    assert counts == {"train": 6, "val": 2, "test": 2}


def test_subject_features_count_primary_pairs_from_earliest_session() -> None:
    subjects = [
        {
            "subject_id": "p1",
            "successful_preprocessed_sessions": "3",
            "total_common_available_windows": "30",
            "earliest_available_session_timestamp": "2100-01-01-00-00",
            "retained_after_preprocessing_k5": "True",
        }
    ]
    pairs = [
        {
            "subject_id": "p1",
            "enrollment_session_timestamp": "2100-01-01-00-00",
            "probe_session_timestamp": "2100-01-02-00-00",
            "gap_days": "1",
            "supports_common_10s_k5_m1": "True",
        },
        {
            "subject_id": "p1",
            "enrollment_session_timestamp": "2100-01-02-00-00",
            "probe_session_timestamp": "2100-01-03-00-00",
            "gap_days": "1",
            "supports_common_10s_k5_m1": "True",
        },
    ]
    rows = subject_features(config(), subjects, pairs)
    assert rows[0]["primary_pair_count"] == 1
    assert rows[0]["all_pair_count"] == 2
