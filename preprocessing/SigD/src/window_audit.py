"""Post-preprocessing availability audit for SigD windows."""

from __future__ import annotations

from collections import defaultdict
from datetime import datetime
from itertools import combinations
from typing import Any

from common import bool_from_any, distribution, numeric_summary


def parse_session_datetime(value: str) -> datetime | None:
    """Parse a surrogate session timestamp."""

    try:
        return datetime.strptime(value, "%Y-%m-%d-%H-%M")
    except ValueError:
        return None


def common_available(row: dict[str, Any]) -> bool:
    """Return common protocol availability, with legacy manifest fallback."""

    if "common_input_available" in row and row.get("common_input_available") != "":
        return bool_from_any(row.get("common_input_available"))
    return bool_from_any(row.get("model_input_available"))


def session_summary_rows(manifest_rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    """Summarize candidate and available windows by subject/session."""

    grouped: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for row in manifest_rows:
        if row.get("raw_range_processing_status") == "failed":
            continue
        grouped[(row["subject_id"], row["session_timestamp"])].append(row)

    rows = []
    for (subject_id, session_timestamp), items in sorted(grouped.items()):
        available = [row for row in items if common_available(row)]
        item = {
            "subject_id": subject_id,
            "session_timestamp": session_timestamp,
            "total_candidate_windows": len(items),
            "common_available_windows": len(available),
            "model_input_available_windows": len(available),
            "excluded_windows": len(items) - len(available),
            "svri_valid_windows": sum(1 for row in items if bool_from_any(row.get("svri_valid_mask"))),
            "sqi_valid_windows": sum(1 for row in items if bool_from_any(row.get("sqi_valid_mask"))),
            "ipa_valid_windows": sum(1 for row in items if bool_from_any(row.get("ipa_valid_mask"))),
        }
        for k in (1, 3, 5):
            item[f"supports_common_10s_k{k}"] = len(available) >= k
            item[f"supports_10s_k{k}"] = len(available) >= k
        rows.append(item)
    return rows


def subject_summary_rows(
    sessions: list[dict[str, Any]],
    raw_subject_summary: list[dict[str, str]],
) -> list[dict[str, Any]]:
    """Summarize post-QC availability by subject."""

    raw_eligible = {
        row["subject_id"]: bool_from_any(row.get("eligible_for_future_10s_cross_session_protocol"))
        for row in raw_subject_summary
    }
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in sessions:
        grouped[row["subject_id"]].append(row)

    rows = []
    for subject_id, items in sorted(grouped.items()):
        timestamps = sorted(
            [row["session_timestamp"] for row in items],
            key=lambda ts: (parse_session_datetime(ts) is None, parse_session_datetime(ts) or ts),
        )
        parsed = [parse_session_datetime(ts) for ts in timestamps]
        parsed = [dt for dt in parsed if dt is not None]
        item = {
            "subject_id": subject_id,
            "successful_preprocessed_sessions": len(items),
            "total_common_available_windows": sum(int(row["common_available_windows"]) for row in items),
            "total_model_input_available_windows": sum(int(row["model_input_available_windows"]) for row in items),
            "earliest_available_session_timestamp": timestamps[0] if timestamps else "",
            "latest_available_session_timestamp": timestamps[-1] if timestamps else "",
            "max_gap_days": ((max(parsed) - min(parsed)).total_seconds() / 86400.0) if len(parsed) >= 2 else "",
            "raw_level_10s_eligible": raw_eligible.get(subject_id, False),
        }
        for k in (1, 3, 5):
            has_two = sum(1 for row in items if int(row["common_available_windows"]) >= k) >= 2
            item[f"has_at_least_2_common_sessions_k{k}"] = has_two
            item[f"has_at_least_2_sessions_k{k}"] = has_two
            item[f"retained_after_preprocessing_k{k}"] = bool(item["raw_level_10s_eligible"] and has_two)
        rows.append(item)
    return rows


def interval_pair_rows(
    sessions: list[dict[str, Any]],
    raw_interval_pairs: list[dict[str, str]],
) -> list[dict[str, Any]]:
    """Build same-subject post-QC interval pair availability rows."""

    session_map = {
        (row["subject_id"], row["session_timestamp"]): row for row in sessions
    }
    raw_pair_set = {
        (
            row["subject_id"],
            row["enrollment_session_timestamp"],
            row["probe_session_timestamp"],
        )
        for row in raw_interval_pairs
        if bool_from_any(row.get("supports_future_10s_cross_session_protocol", row.get("supports_future_cross_session_verification")))
    }

    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in sessions:
        grouped[row["subject_id"]].append(row)

    pairs = []
    for subject_id, items in sorted(grouped.items()):
        sorted_items = sorted(
            items,
            key=lambda row: (
                parse_session_datetime(row["session_timestamp"]) is None,
                parse_session_datetime(row["session_timestamp"]) or row["session_timestamp"],
            ),
        )
        for enrollment, probe in combinations(sorted_items, 2):
            e_ts = enrollment["session_timestamp"]
            p_ts = probe["session_timestamp"]
            e_dt = parse_session_datetime(e_ts)
            p_dt = parse_session_datetime(p_ts)
            gap_hours = ""
            gap_days = ""
            if e_dt is not None and p_dt is not None:
                seconds = (p_dt - e_dt).total_seconds()
                gap_hours = seconds / 3600.0
                gap_days = seconds / 86400.0
            e_n = int(enrollment["common_available_windows"])
            p_n = int(probe["common_available_windows"])
            raw_level_pair = (subject_id, e_ts, p_ts) in raw_pair_set
            row = {
                "subject_id": subject_id,
                "enrollment_session_timestamp": e_ts,
                "probe_session_timestamp": p_ts,
                "gap_hours": gap_hours,
                "gap_days": gap_days,
                "enrollment_common_available_windows": e_n,
                "probe_common_available_windows": p_n,
                "enrollment_available_windows": e_n,
                "probe_available_windows": p_n,
                "supports_common_10s_k1_m1": e_n >= 1 and p_n >= 1,
                "supports_common_10s_k3_m1": e_n >= 3 and p_n >= 1,
                "supports_common_10s_k5_m1": e_n >= 5 and p_n >= 1,
                "supports_common_10s_k3_m3": e_n >= 3 and p_n >= 3,
                "supports_common_10s_k5_m5": e_n >= 5 and p_n >= 5,
                "supports_10s_k1_m1": e_n >= 1 and p_n >= 1,
                "supports_10s_k3_m1": e_n >= 3 and p_n >= 1,
                "supports_10s_k5_m1": e_n >= 5 and p_n >= 1,
                "supports_10s_k3_m3": e_n >= 3 and p_n >= 3,
                "supports_10s_k5_m5": e_n >= 5 and p_n >= 5,
                "raw_level_pair_was_eligible": raw_level_pair,
            }
            row["retained_after_preprocessing"] = bool(raw_level_pair and row["supports_common_10s_k1_m1"])
            pairs.append(row)
    return pairs


def postqc_summary(
    manifest_rows: list[dict[str, str]],
    sessions: list[dict[str, Any]],
    subjects: list[dict[str, Any]],
    pairs: list[dict[str, Any]],
    raw_audit_summary: dict[str, Any],
    config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build post-QC summary JSON."""

    raw_future = raw_audit_summary.get("future_window_availability_summary", {}).get("10s", {})
    raw_cross = raw_audit_summary.get("cross_session_availability_summary", {})
    primary_pairs = []
    earliest_by_subject: dict[str, str] = {}
    for row in sessions:
        subject = row["subject_id"]
        ts = row["session_timestamp"]
        if subject not in earliest_by_subject or (
            parse_session_datetime(ts) or ts
        ) < (parse_session_datetime(earliest_by_subject[subject]) or earliest_by_subject[subject]):
            earliest_by_subject[subject] = ts
    for row in pairs:
        if row["enrollment_session_timestamp"] == earliest_by_subject.get(row["subject_id"]):
            primary_pairs.append(row)

    k3_subjects = sum(1 for row in subjects if bool_from_any(row.get("retained_after_preprocessing_k3")))
    k5_subjects = sum(1 for row in subjects if bool_from_any(row.get("retained_after_preprocessing_k5")))
    recommendation = "K=3/M=1" if k3_subjects >= max(1, int(0.8 * len(subjects))) else "K=1/M=1"
    if k5_subjects >= max(1, int(0.8 * len(subjects))):
        recommendation = "K=5/M=1"

    protocol = config or {}
    return {
        "input_protocol_id": protocol.get("input_protocol_id", ""),
        "comparison_role": protocol.get("comparison_role", ""),
        "preprocessing_profile": protocol.get("preprocessing_profile", ""),
        "native_or_common_input": protocol.get("native_or_common_input", ""),
        "normalization_policy": protocol.get("normalization_policy", ""),
        "applicable_primary_models": protocol.get("applicable_primary_models", []),
        "native_input_outputs_generated": protocol.get("native_input_outputs_generated", False),
        "raw_level_eligible_subjects_10s": raw_future.get("subjects_eligible_for_future_cross_session_protocol", 231),
        "raw_level_all_session_pairs_10s": raw_cross.get("all_same_subject_successful_session_pair_count", 484),
        "preprocessed_total_windows": len(manifest_rows),
        "common_input_available_windows": sum(1 for row in manifest_rows if common_available(row)),
        "model_input_available_windows": sum(1 for row in manifest_rows if common_available(row)),
        "aux_morphology_annotation_available_windows": sum(
            1 for row in manifest_rows if bool_from_any(row.get("aux_morphology_annotation_available"))
        ),
        "aux_morphology_any_available_windows": sum(
            1 for row in manifest_rows if bool_from_any(row.get("aux_morphology_any_available", row.get("aux_morphology_annotation_available")))
        ),
        "aux_morphology_all_available_windows": sum(
            1 for row in manifest_rows if bool_from_any(row.get("aux_morphology_all_available"))
        ),
        "svri_valid_windows": sum(1 for row in manifest_rows if bool_from_any(row.get("svri_valid_mask"))),
        "sqi_valid_windows": sum(1 for row in manifest_rows if bool_from_any(row.get("sqi_valid_mask"))),
        "ipa_valid_windows": sum(1 for row in manifest_rows if bool_from_any(row.get("ipa_valid_mask"))),
        "postqc_subjects_k1": sum(1 for row in subjects if bool_from_any(row.get("retained_after_preprocessing_k1"))),
        "postqc_subjects_k3": sum(1 for row in subjects if bool_from_any(row.get("retained_after_preprocessing_k3"))),
        "postqc_subjects_k5": sum(1 for row in subjects if bool_from_any(row.get("retained_after_preprocessing_k5"))),
        "postqc_all_session_pairs_k1_m1": sum(1 for row in pairs if bool_from_any(row.get("supports_common_10s_k1_m1"))),
        "postqc_all_session_pairs_k3_m1": sum(1 for row in pairs if bool_from_any(row.get("supports_common_10s_k3_m1"))),
        "postqc_all_session_pairs_k5_m1": sum(1 for row in pairs if bool_from_any(row.get("supports_common_10s_k5_m1"))),
        "postqc_primary_earliest_to_later_pairs_k1_m1": sum(1 for row in primary_pairs if bool_from_any(row.get("supports_common_10s_k1_m1"))),
        "postqc_primary_earliest_to_later_pairs_k3_m1": sum(1 for row in primary_pairs if bool_from_any(row.get("supports_common_10s_k3_m1"))),
        "postqc_primary_earliest_to_later_pairs_k5_m1": sum(1 for row in primary_pairs if bool_from_any(row.get("supports_common_10s_k5_m1"))),
        "time_gap_distribution_for_retained_primary_pairs": numeric_summary(
            row["gap_days"] for row in primary_pairs if bool_from_any(row.get("retained_after_preprocessing"))
        ),
        "recommended_primary_protocol_condition": recommendation,
        "eligibility_basis": "common_input_available_only",
        "morphology_validity_used_for_common_eligibility": False,
        "exclusion_reason_distribution": distribution(
            row.get("exclusion_reason", "") or "none"
            for row in manifest_rows
            if not common_available(row)
        ),
    }
