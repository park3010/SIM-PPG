from __future__ import annotations

import sys
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

from audit_sigd_protocol import audit_protocol  # noqa: E402
from build_sigd_verification_protocol import (  # noqa: E402
    build_genuine_trials,
    build_impostor_trials,
    build_templates,
    gap_bucket,
    later_probe_pool_by_split,
    pair_gap_map,
    session_sort_key,
    windows_by_subject_session,
)


def config() -> dict:
    return {
        "protocol_id": "TEST_PROTOCOL",
        "input_protocol_id": "COMMON_PPG_10S_125HZ_V1",
        "split_seed": 42,
        "enrollment_policy": {
            "session": "earliest_valid_session",
            "k_windows": 5,
            "window_selection": "first_k_common_available_windows",
            "template_aggregation": "mean_embedding_later_model_stage",
        },
        "probe_policy": {
            "session": "later_valid_sessions",
            "m_windows": 1,
            "window_selection": "all_common_available_probe_windows",
            "applies_to": ["genuine", "impostor"],
        },
        "impostor_policy": {
            "sampling": "later_session_only_sampled_ratio",
            "ratio_to_genuine": {"train": 1, "val": 1, "test": 1},
            "seed": 42,
            "exclude_probe_subject_enrollment_session": True,
            "split_internal_only": True,
        },
        "threshold_policy": "validation_only",
        "morphology_validity_used_for_protocol": False,
    }


def split_rows() -> list[dict[str, str]]:
    return [
        {"subject_id": "p1", "split": "train"},
        {"subject_id": "p2", "split": "train"},
        {"subject_id": "p3", "split": "val"},
        {"subject_id": "p4", "split": "val"},
        {"subject_id": "p5", "split": "test"},
        {"subject_id": "p6", "split": "test"},
    ]


def windows() -> list[dict]:
    rows = []
    index = 0
    for subject in ["p1", "p2", "p3", "p4", "p5", "p6"]:
        for session in ["2100-01-01-00-00", "2100-01-02-00-00"]:
            for i in range(6):
                rows.append(
                    {
                        "subject_id": subject,
                        "session_timestamp": session,
                        "parent_raw_range_id": f"{subject}_{session}_r000",
                        "array_index": index,
                        "window_start_sample_in_raw_range": i * 1250,
                        "window_index_within_raw_range": i,
                        "common_input_available": "True",
                        "svri_valid_mask": "False",
                        "sqi_valid_mask": "False",
                        "ipa_valid_mask": "False",
                    }
                )
                index += 1
    return rows


def build_protocol_parts():
    cfg = config()
    grouped = windows_by_subject_session(windows(), {row["subject_id"] for row in split_rows()})
    templates, by_subject = build_templates(cfg, split_rows(), grouped)
    pairs = [
        {
            "subject_id": row["subject_id"],
            "enrollment_session_timestamp": "2100-01-01-00-00",
            "probe_session_timestamp": "2100-01-02-00-00",
            "gap_days": "1",
        }
        for row in split_rows()
    ]
    genuine = build_genuine_trials(cfg, split_rows(), grouped, by_subject, pair_gap_map(pairs))
    gaps = pair_gap_map(pairs)
    impostor = build_impostor_trials(cfg, templates, windows(), genuine, split_rows(), by_subject, gaps)
    verification = genuine + impostor
    return cfg, templates, genuine, impostor, verification


def test_genuine_trials_have_same_subject_different_sessions() -> None:
    _, _, genuine, _, _ = build_protocol_parts()
    assert genuine
    assert all(row["enroll_subject_id"] == row["probe_subject_id"] for row in genuine)
    assert all(row["enroll_session_id"] != row["probe_session_id"] for row in genuine)


def test_impostor_trials_have_different_subjects() -> None:
    _, _, _, impostor, _ = build_protocol_parts()
    assert impostor
    assert all(row["enroll_subject_id"] != row["probe_subject_id"] for row in impostor)


def test_trials_reference_only_subjects_inside_their_split() -> None:
    subject_split = {row["subject_id"]: row["split"] for row in split_rows()}
    _, _, _, _, verification = build_protocol_parts()
    for row in verification:
        assert subject_split[row["enroll_subject_id"]] == row["split"]
        assert subject_split[row["probe_subject_id"]] == row["split"]


def test_enrollment_template_has_exactly_k5_windows() -> None:
    _, templates, _, _, _ = build_protocol_parts()
    assert templates
    assert all(row["k_windows"] == 5 for row in templates)
    assert all(len(__import__("json").loads(row["enrollment_window_indices"])) == 5 for row in templates)


def test_all_window_indices_are_inside_array_bounds() -> None:
    cfg, templates, genuine, impostor, verification = build_protocol_parts()
    audit = audit_protocol(cfg, split_rows(), templates, genuine, impostor, verification, array_rows=len(windows()))
    assert audit["passed"] is True


def test_protocol_generation_does_not_use_morphology_validity() -> None:
    cfg, templates, genuine, impostor, verification = build_protocol_parts()
    assert cfg["morphology_validity_used_for_protocol"] is False
    audit = audit_protocol(cfg, split_rows(), templates, genuine, impostor, verification, array_rows=len(windows()))
    assert audit["morphology_validity_used_for_protocol"] is False


def test_gap_bucket_assignment() -> None:
    assert gap_bucket(10) == "le_30d"
    assert gap_bucket(31) == "31_180d"
    assert gap_bucket(180) == "31_180d"
    assert gap_bucket(181) == "181_365d"
    assert gap_bucket(365) == "181_365d"
    assert gap_bucket(366) == "gt_365d"


def test_impostor_probe_from_probe_subject_enrollment_session_fails() -> None:
    cfg, templates, genuine, impostor, verification = build_protocol_parts()
    bad = dict(impostor[0])
    bad["probe_session_id"] = bad["probe_reference_enrollment_session_id"]
    bad["probe_time_gap_days"] = "1"
    bad["time_gap_days"] = "1"
    bad["probe_time_gap_bucket"] = "le_30d"
    audit = audit_protocol(cfg, split_rows(), templates, genuine, [bad], genuine + [bad], array_rows=len(windows()))
    assert any(
        err.startswith("impostor_probe_not_later_than_probe_subject_enrollment")
        for err in audit["errors"]
    )


def test_impostor_probe_from_probe_subject_later_session_passes() -> None:
    cfg, templates, genuine, impostor, verification = build_protocol_parts()
    audit = audit_protocol(cfg, split_rows(), templates, genuine, impostor, verification, array_rows=len(windows()))
    assert audit["passed"] is True
    assert audit["chronology_checks"]["impostor_later_probe_passed"] is True


def test_genuine_probe_not_later_than_enrollment_fails() -> None:
    cfg, templates, genuine, impostor, _ = build_protocol_parts()
    bad = dict(genuine[0])
    bad["probe_session_id"] = bad["enroll_session_id"]
    bad["probe_time_gap_days"] = "1"
    bad["time_gap_days"] = "1"
    bad["probe_time_gap_bucket"] = "le_30d"
    audit = audit_protocol(cfg, split_rows(), templates, [bad], impostor, [bad] + impostor, array_rows=len(windows()))
    assert any(err.startswith("genuine_probe_not_later_than_enrollment") for err in audit["errors"])


def test_probe_time_gap_days_required_and_positive() -> None:
    cfg, templates, genuine, impostor, _ = build_protocol_parts()
    bad = dict(genuine[0])
    bad["probe_time_gap_days"] = ""
    bad["time_gap_days"] = ""
    bad["probe_time_gap_bucket"] = "unknown"
    audit = audit_protocol(cfg, split_rows(), templates, [bad], impostor, [bad] + impostor, array_rows=len(windows()))
    assert any(err.startswith("probe_time_gap_missing") for err in audit["errors"])


def test_impostor_sampling_uses_later_probe_pool_only() -> None:
    cfg = config()
    grouped = windows_by_subject_session(windows(), {row["subject_id"] for row in split_rows()})
    templates, by_subject = build_templates(cfg, split_rows(), grouped)
    pools = later_probe_pool_by_split(split_rows(), windows(), by_subject)
    for rows in pools.values():
        for row in rows:
            enrollment = by_subject[row["subject_id"]]["enrollment_session_id"]
            assert session_sort_key(row["session_timestamp"]) > session_sort_key(enrollment)

    _, _, _, impostor, _ = build_protocol_parts()
    for row in impostor:
        assert session_sort_key(row["probe_session_id"]) > session_sort_key(row["probe_reference_enrollment_session_id"])
