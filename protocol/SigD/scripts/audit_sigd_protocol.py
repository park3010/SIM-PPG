#!/usr/bin/env python3
"""Audit the SigD subject split and K=5/M=1 verification protocol."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any
from datetime import datetime

import numpy as np
import yaml


DEFAULT_CONFIG = Path("protocol/SigD/config/sigd_protocol_10s_k5m1.yaml")
SPLITS = ("train", "val", "test")


def detect_root(root_arg: str | None) -> Path:
    if root_arg:
        return Path(root_arg).expanduser().resolve()
    for candidate in [Path.cwd().resolve(), *Path.cwd().resolve().parents]:
        if (candidate / "protocol" / "SigD").exists():
            return candidate
    raise SystemExit("Could not detect SIM_PPG root. Pass --root PATH.")


def load_config(root: Path, config_path: str | None) -> dict[str, Any]:
    path = Path(config_path).expanduser().resolve() if config_path else root / DEFAULT_CONFIG
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def resolve(root: Path, path: str | Path) -> Path:
    value = Path(path)
    return value if value.is_absolute() else root / value


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def bool_value(value: Any) -> bool:
    return str(value).strip().lower() in {"true", "1", "yes", "y"}


def as_float(value: Any) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def parse_session_datetime(value: str) -> datetime | None:
    try:
        return datetime.strptime(value, "%Y-%m-%d-%H-%M")
    except (TypeError, ValueError):
        return None


def is_later_session(probe_session: str, reference_session: str) -> bool:
    """Return whether probe_session is chronologically later."""

    probe_dt = parse_session_datetime(probe_session)
    ref_dt = parse_session_datetime(reference_session)
    if probe_dt is None or ref_dt is None:
        return False
    return probe_dt > ref_dt


def gap_bucket(days: Any) -> str:
    parsed = as_float(days)
    if parsed is None:
        return "unknown"
    if parsed <= 30:
        return "le_30d"
    if parsed <= 180:
        return "31_180d"
    if parsed <= 365:
        return "181_365d"
    return "gt_365d"


def parse_index_list(value: str) -> list[int]:
    parsed = json.loads(value)
    if not isinstance(parsed, list):
        raise ValueError("index list is not a JSON list")
    return [int(item) for item in parsed]


def audit_protocol(
    config: dict[str, Any],
    split_rows: list[dict[str, str]],
    templates: list[dict[str, str]],
    genuine: list[dict[str, str]],
    impostor: list[dict[str, str]],
    verification: list[dict[str, str]],
    array_rows: int,
) -> dict[str, Any]:
    """Return protocol audit checks and error details."""

    errors: list[str] = []
    warnings: list[str] = []
    subject_split = {row["subject_id"]: row["split"] for row in split_rows}
    split_subjects = {
        split: {row["subject_id"] for row in split_rows if row["split"] == split}
        for split in SPLITS
    }
    overlaps = {
        "train_val": sorted(split_subjects["train"] & split_subjects["val"]),
        "train_test": sorted(split_subjects["train"] & split_subjects["test"]),
        "val_test": sorted(split_subjects["val"] & split_subjects["test"]),
    }
    for name, subjects in overlaps.items():
        if subjects:
            errors.append(f"subject_overlap:{name}:{subjects[:5]}")

    k = int(config["enrollment_policy"]["k_windows"])
    template_by_id = {row["template_id"]: row for row in templates}
    template_by_subject = {row["subject_id"]: row for row in templates}
    for row in templates:
        try:
            indices = parse_index_list(row["enrollment_window_indices"])
        except Exception as exc:
            errors.append(f"template_index_parse_error:{row.get('template_id')}:{exc}")
            continue
        if len(indices) != k:
            errors.append(f"template_wrong_k:{row['template_id']}:{len(indices)}")
        if any(index < 0 or index >= array_rows for index in indices):
            errors.append(f"template_index_out_of_bounds:{row['template_id']}")
        if subject_split.get(row["subject_id"]) != row["split"]:
            errors.append(f"template_split_mismatch:{row['template_id']}")

    for row in verification:
        template = template_by_id.get(row["template_id"])
        if template is None:
            errors.append(f"missing_template:{row['trial_id']}")
            continue
        split = row["split"]
        enroll_subject = row["enroll_subject_id"]
        probe_subject = row["probe_subject_id"]
        if subject_split.get(enroll_subject) != split or subject_split.get(probe_subject) != split:
            errors.append(f"trial_subject_outside_split:{row['trial_id']}")
        if template["subject_id"] != enroll_subject:
            errors.append(f"trial_template_subject_mismatch:{row['trial_id']}")
        probe_index = int(float(row["probe_window_index"]))
        if probe_index < 0 or probe_index >= array_rows:
            errors.append(f"probe_index_out_of_bounds:{row['trial_id']}")

        same_subject = enroll_subject == probe_subject
        same_session = row["enroll_session_id"] == row["probe_session_id"]
        label = str(row.get("label"))
        if row["trial_type"] == "genuine":
            if label != "1":
                errors.append(f"genuine_label_error:{row['trial_id']}")
            if not same_subject:
                errors.append(f"genuine_subject_mismatch:{row['trial_id']}")
            if same_session:
                errors.append(f"genuine_same_session:{row['trial_id']}")
            if not is_later_session(row["probe_session_id"], row["enroll_session_id"]):
                errors.append(f"genuine_probe_not_later_than_enrollment:{row['trial_id']}")
            if row.get("probe_reference_enrollment_session_id") != row["enroll_session_id"]:
                errors.append(f"genuine_probe_reference_session_mismatch:{row['trial_id']}")
        elif row["trial_type"] == "impostor":
            if label != "0":
                errors.append(f"impostor_label_error:{row['trial_id']}")
            if same_subject:
                errors.append(f"impostor_same_subject:{row['trial_id']}")
            probe_template = template_by_subject.get(probe_subject)
            if probe_template is None:
                errors.append(f"impostor_probe_subject_missing_template:{row['trial_id']}")
            else:
                probe_ref = probe_template["enrollment_session_id"]
                if row.get("probe_reference_enrollment_session_id") != probe_ref:
                    errors.append(f"impostor_probe_reference_session_mismatch:{row['trial_id']}")
                if not is_later_session(row["probe_session_id"], probe_ref):
                    errors.append(f"impostor_probe_not_later_than_probe_subject_enrollment:{row['trial_id']}")
        else:
            errors.append(f"unknown_trial_type:{row['trial_id']}")

        gap = as_float(row.get("probe_time_gap_days"))
        if gap is None:
            errors.append(f"probe_time_gap_missing:{row['trial_id']}")
        elif gap <= 0:
            errors.append(f"probe_time_gap_nonpositive:{row['trial_id']}")
        if row.get("probe_time_gap_bucket") != gap_bucket(row.get("probe_time_gap_days")):
            errors.append(f"probe_time_gap_bucket_mismatch:{row['trial_id']}")

    trial_ids = [row["trial_id"] for row in verification]
    if len(trial_ids) != len(set(trial_ids)):
        errors.append("duplicate_trial_id")

    if len(verification) != len(genuine) + len(impostor):
        errors.append("verification_count_mismatch")
    if bool_value(config.get("morphology_validity_used_for_protocol")):
        errors.append("morphology_validity_used_for_protocol")

    ratio_checks = {}
    ratios = config["impostor_policy"]["ratio_to_genuine"]
    for split in SPLITS:
        g = sum(1 for row in genuine if row["split"] == split)
        i = sum(1 for row in impostor if row["split"] == split)
        expected = g * int(ratios[split])
        ratio_checks[split] = {"genuine": g, "impostor": i, "expected_impostor": expected, "passed": i == expected}
        if i != expected:
            errors.append(f"impostor_ratio_error:{split}:{i}!={expected}")

    genuine_chronology_errors = [
        item for item in errors if item.startswith("genuine_probe_not_later_than_enrollment")
    ]
    impostor_chronology_errors = [
        item for item in errors if item.startswith("impostor_probe_not_later_than_probe_subject_enrollment")
    ]
    gap_errors = [
        item
        for item in errors
        if item.startswith("probe_time_gap_missing")
        or item.startswith("probe_time_gap_nonpositive")
        or item.startswith("probe_time_gap_bucket_mismatch")
    ]
    return {
        "passed": len(errors) == 0,
        "errors": errors,
        "warnings": warnings,
        "array_rows": array_rows,
        "subject_overlap": {
            "train_val": len(overlaps["train_val"]),
            "train_test": len(overlaps["train_test"]),
            "val_test": len(overlaps["val_test"]),
            "details": overlaps,
            "passed": all(len(items) == 0 for items in overlaps.values()),
        },
        "split_subject_counts": {split: len(split_subjects[split]) for split in SPLITS},
        "template_count": len(templates),
        "genuine_trial_count": len(genuine),
        "impostor_trial_count": len(impostor),
        "verification_trial_count": len(verification),
        "impostor_ratio_checks": ratio_checks,
        "morphology_validity_used_for_protocol": False,
        "chronology_checks": {
            "genuine_later_probe_passed": len(genuine_chronology_errors) == 0,
            "impostor_later_probe_passed": len(impostor_chronology_errors) == 0,
            "probe_time_gap_consistency_passed": len(gap_errors) == 0,
            "genuine_later_probe_error_count": len(genuine_chronology_errors),
            "impostor_later_probe_error_count": len(impostor_chronology_errors),
            "probe_time_gap_error_count": len(gap_errors),
        },
        "gap_bucket_counts_by_split": gap_bucket_counts_by_split(verification),
        "later_probe_condition_applies_to": ["genuine", "impostor"],
    }


def gap_bucket_counts_by_split(rows: list[dict[str, str]]) -> dict[str, dict[str, int]]:
    """Count probe-time-gap buckets per split."""

    output = {
        split: {"le_30d": 0, "31_180d": 0, "181_365d": 0, "gt_365d": 0, "unknown": 0}
        for split in SPLITS
    }
    for row in rows:
        split = row.get("split", "")
        if split not in output:
            continue
        bucket = row.get("probe_time_gap_bucket") or "unknown"
        output[split][bucket] = output[split].get(bucket, 0) + 1
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit SigD protocol outputs.")
    parser.add_argument("--root", default=None)
    parser.add_argument("--config", default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = detect_root(args.root)
    config = load_config(root, args.config)
    seed = config["split_seed"]
    out_dir = root / "protocol" / "SigD" / "metadata"
    split_rows = read_csv_rows(out_dir / f"subject_split_seed{seed}.csv")
    templates = read_csv_rows(out_dir / f"enrollment_templates_k5_seed{seed}.csv")
    genuine = read_csv_rows(out_dir / f"genuine_trials_k5m1_seed{seed}.csv")
    impostor = read_csv_rows(out_dir / f"impostor_trials_k5m1_seed{seed}.csv")
    verification = read_csv_rows(out_dir / f"verification_trials_k5m1_seed{seed}.csv")
    array = np.load(resolve(root, config["common_array_path"]), mmap_mode="r")
    audit = audit_protocol(config, split_rows, templates, genuine, impostor, verification, int(array.shape[0]))

    summary_path = out_dir / f"protocol_summary_k5m1_seed{seed}.json"
    summary = read_json(summary_path)
    summary["audit"] = audit
    write_json(summary_path, summary)
    print(f"audit_passed={audit['passed']} errors={len(audit['errors'])}")
    if not audit["passed"]:
        raise SystemExit(1)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
