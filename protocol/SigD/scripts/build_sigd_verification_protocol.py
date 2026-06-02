#!/usr/bin/env python3
"""Build K=5/M=1 verification trials for the SigD common 10s protocol."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import random
import statistics
from datetime import datetime, timezone
from typing import Any, Iterable

import numpy as np
import yaml


DEFAULT_CONFIG = Path("protocol/SigD/config/sigd_protocol_10s_k5m1.yaml")
SPLITS = ("train", "val", "test")

TEMPLATE_COLUMNS = [
    "template_id",
    "split",
    "subject_id",
    "enrollment_session_id",
    "enrollment_window_indices",
    "k_windows",
    "input_protocol_id",
    "protocol_id",
    "template_policy",
]

TRIAL_COLUMNS = [
    "trial_id",
    "split",
    "label",
    "trial_type",
    "template_id",
    "enroll_subject_id",
    "probe_subject_id",
    "enroll_session_id",
    "probe_session_id",
    "probe_window_index",
    "probe_raw_range_id",
    "time_gap_days",
    "probe_reference_enrollment_session_id",
    "probe_time_gap_days",
    "probe_time_gap_bucket",
    "same_subject",
    "same_session",
    "k",
    "m",
    "input_protocol_id",
    "protocol_id",
]


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def detect_root(root_arg: str | None) -> Path:
    if root_arg:
        return Path(root_arg).expanduser().resolve()
    for candidate in [Path.cwd().resolve(), *Path.cwd().resolve().parents]:
        if (candidate / "preprocessing" / "SigD").exists():
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


def write_csv(path: Path, rows: list[dict[str, Any]], columns: list[str], overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise SystemExit(f"Output exists; pass --overwrite: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column, "") for column in columns})


def write_json(path: Path, payload: dict[str, Any], overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise SystemExit(f"Output exists; pass --overwrite: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
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


def numeric_summary(values: Iterable[Any]) -> dict[str, Any]:
    clean = sorted(value for value in (as_float(item) for item in values) if value is not None)
    if not clean:
        return {"count": 0, "min": None, "median": None, "max": None, "mean": None}
    return {
        "count": len(clean),
        "min": clean[0],
        "median": statistics.median(clean),
        "max": clean[-1],
        "mean": sum(clean) / len(clean),
    }


def parse_session_datetime(value: str) -> datetime | None:
    try:
        return datetime.strptime(value, "%Y-%m-%d-%H-%M")
    except ValueError:
        return None


def session_sort_key(value: str) -> tuple[bool, Any]:
    parsed = parse_session_datetime(value)
    return (parsed is None, parsed or value)


def gap_bucket(days: Any) -> str:
    """Bucket a positive day gap for time-gap-stratified evaluation."""

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


def load_common_windows(root: Path, config: dict[str, Any]) -> list[dict[str, Any]]:
    """Load common-input-available manifest rows and verify array index mapping."""

    snapshot = resolve(root, config["preprocessing_snapshot_dir"])
    manifest_rows = read_csv_rows(snapshot / "preprocessing_manifest_10s.csv")
    common_rows = [dict(row) for row in manifest_rows if bool_value(row.get("common_input_available"))]
    for index, row in enumerate(common_rows):
        raw_index = row.get("array_index", "")
        if raw_index == "":
            row["array_index"] = index
        else:
            row["array_index"] = int(float(raw_index))
        row["window_start_sample_in_raw_range"] = int(float(row.get("window_start_sample_in_raw_range", 0) or 0))
        row["window_index_within_raw_range"] = int(float(row.get("window_index_within_raw_range", 0) or 0))

    array = np.load(resolve(root, config["common_array_path"]), mmap_mode="r")
    if array.shape[0] != len(common_rows):
        raise SystemExit(f"Array rows {array.shape[0]} != common manifest rows {len(common_rows)}")
    indices = sorted(int(row["array_index"]) for row in common_rows)
    if indices != list(range(array.shape[0])):
        raise SystemExit("Common manifest array_index values are not contiguous 0..N-1.")
    return common_rows


def split_map(rows: list[dict[str, str]]) -> dict[str, str]:
    return {row["subject_id"]: row["split"] for row in rows}


def windows_by_subject_session(windows: list[dict[str, Any]], subjects: set[str]) -> dict[tuple[str, str], list[dict[str, Any]]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in windows:
        subject = row["subject_id"]
        if subject not in subjects:
            continue
        key = (subject, row["session_timestamp"])
        grouped.setdefault(key, []).append(row)
    for items in grouped.values():
        items.sort(
            key=lambda row: (
                row["session_timestamp"],
                row["parent_raw_range_id"],
                int(row["window_start_sample_in_raw_range"]),
                int(row["array_index"]),
            )
        )
    return grouped


def build_templates(
    config: dict[str, Any],
    split_rows: list[dict[str, str]],
    grouped_windows: dict[tuple[str, str], list[dict[str, Any]]],
) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    """Select earliest-session K-window enrollment templates."""

    k = int(config["enrollment_policy"]["k_windows"])
    subject_to_split = split_map(split_rows)
    templates = []
    template_by_subject: dict[str, dict[str, Any]] = {}
    for subject_id in sorted(subject_to_split):
        sessions = sorted(
            [
                session
                for (subject, session), rows in grouped_windows.items()
                if subject == subject_id and len(rows) >= k
            ],
            key=session_sort_key,
        )
        if not sessions:
            continue
        enrollment_session = sessions[0]
        selected = grouped_windows[(subject_id, enrollment_session)][:k]
        indices = [int(row["array_index"]) for row in selected]
        template = {
            "template_id": f"tmpl_{subject_id}_k{k}",
            "split": subject_to_split[subject_id],
            "subject_id": subject_id,
            "enrollment_session_id": enrollment_session,
            "enrollment_window_indices": json.dumps(indices, separators=(",", ":")),
            "k_windows": k,
            "input_protocol_id": config["input_protocol_id"],
            "protocol_id": config["protocol_id"],
            "template_policy": (
                f"{config['enrollment_policy']['session']}__"
                f"{config['enrollment_policy']['window_selection']}"
            ),
        }
        templates.append(template)
        template_by_subject[subject_id] = template
    return templates, template_by_subject


def pair_gap_map(pair_rows: list[dict[str, str]]) -> dict[tuple[str, str, str], str]:
    return {
        (row["subject_id"], row["enrollment_session_timestamp"], row["probe_session_timestamp"]): row.get("gap_days", "")
        for row in pair_rows
    }


def subject_enrollment_sessions(templates_by_subject: dict[str, dict[str, Any]]) -> dict[str, str]:
    """Return subject -> earliest enrollment session from templates."""

    return {
        subject_id: template["enrollment_session_id"]
        for subject_id, template in templates_by_subject.items()
    }


def build_genuine_trials(
    config: dict[str, Any],
    split_rows: list[dict[str, str]],
    grouped_windows: dict[tuple[str, str], list[dict[str, Any]]],
    templates_by_subject: dict[str, dict[str, Any]],
    pair_gaps: dict[tuple[str, str, str], str],
) -> list[dict[str, Any]]:
    """Create one genuine trial for every later-session probe window."""

    k = int(config["enrollment_policy"]["k_windows"])
    m = int(config["probe_policy"]["m_windows"])
    subject_to_split = split_map(split_rows)
    trials = []
    counter_by_split = {split: 0 for split in SPLITS}
    for subject_id in sorted(templates_by_subject):
        template = templates_by_subject[subject_id]
        enrollment_session = template["enrollment_session_id"]
        later_sessions = sorted(
            [
                session
                for (subject, session), rows in grouped_windows.items()
                if subject == subject_id and session_sort_key(session) > session_sort_key(enrollment_session) and rows
            ],
            key=session_sort_key,
        )
        for probe_session in later_sessions:
            for probe in grouped_windows[(subject_id, probe_session)]:
                split = subject_to_split[subject_id]
                idx = counter_by_split[split]
                counter_by_split[split] += 1
                gap = pair_gaps.get((subject_id, enrollment_session, probe_session), "")
                trials.append(
                    {
                        "trial_id": f"{split}_genuine_{idx:08d}",
                        "split": split,
                        "label": 1,
                        "trial_type": "genuine",
                        "template_id": template["template_id"],
                        "enroll_subject_id": subject_id,
                        "probe_subject_id": subject_id,
                        "enroll_session_id": enrollment_session,
                        "probe_session_id": probe_session,
                        "probe_window_index": int(probe["array_index"]),
                        "probe_raw_range_id": probe["parent_raw_range_id"],
                        "time_gap_days": gap,
                        "probe_reference_enrollment_session_id": enrollment_session,
                        "probe_time_gap_days": gap,
                        "probe_time_gap_bucket": gap_bucket(gap),
                        "same_subject": True,
                        "same_session": enrollment_session == probe_session,
                        "k": k,
                        "m": m,
                        "input_protocol_id": config["input_protocol_id"],
                        "protocol_id": config["protocol_id"],
                    }
                )
    return trials


def later_probe_pool_by_split(
    split_rows: list[dict[str, str]],
    windows: list[dict[str, Any]],
    templates_by_subject: dict[str, dict[str, Any]],
) -> dict[str, list[dict[str, Any]]]:
    """Return split-level probe pools restricted to each probe subject's later sessions."""

    subject_to_split = split_map(split_rows)
    subject_enrollment = subject_enrollment_sessions(templates_by_subject)
    pools = {split: [] for split in SPLITS}
    for row in windows:
        subject_id = row["subject_id"]
        split = subject_to_split.get(subject_id)
        enrollment_session = subject_enrollment.get(subject_id)
        if not split or not enrollment_session:
            continue
        if session_sort_key(row["session_timestamp"]) <= session_sort_key(enrollment_session):
            continue
        pools[split].append(row)
    return pools


def build_impostor_trials(
    config: dict[str, Any],
    templates: list[dict[str, Any]],
    windows: list[dict[str, Any]],
    genuine_trials: list[dict[str, Any]],
    split_rows: list[dict[str, str]],
    templates_by_subject: dict[str, dict[str, Any]],
    pair_gaps: dict[tuple[str, str, str], str],
) -> list[dict[str, Any]]:
    """Sample split-internal impostor trials according to configured ratios."""

    seed = int(config["impostor_policy"].get("seed", config["split_seed"]))
    rng = random.Random(seed)
    k = int(config["enrollment_policy"]["k_windows"])
    m = int(config["probe_policy"]["m_windows"])
    ratios = {split: int(config["impostor_policy"]["ratio_to_genuine"][split]) for split in SPLITS}
    templates_by_split = {split: [row for row in templates if row["split"] == split] for split in SPLITS}
    probe_pools = later_probe_pool_by_split(split_rows, windows, templates_by_subject)
    genuine_count = {split: sum(1 for row in genuine_trials if row["split"] == split) for split in SPLITS}
    trials = []
    seen: set[tuple[str, int]] = set()

    for split in SPLITS:
        target = genuine_count[split] * ratios[split]
        templates_for_split = templates_by_split[split]
        probes_for_split = probe_pools[split]
        attempts = 0
        split_count = 0
        while split_count < target:
            attempts += 1
            if attempts > max(10000, target * 50):
                raise RuntimeError(f"Could not sample enough impostor trials for {split}")
            template = rng.choice(templates_for_split)
            probe = rng.choice(probes_for_split)
            if template["subject_id"] == probe["subject_id"]:
                continue
            key = (template["template_id"], int(probe["array_index"]))
            if key in seen:
                continue
            seen.add(key)
            probe_ref_session = templates_by_subject[probe["subject_id"]]["enrollment_session_id"]
            probe_gap = pair_gaps.get((probe["subject_id"], probe_ref_session, probe["session_timestamp"]), "")
            trials.append(
                {
                    "trial_id": f"{split}_impostor_{split_count:08d}",
                    "split": split,
                    "label": 0,
                    "trial_type": "impostor",
                    "template_id": template["template_id"],
                    "enroll_subject_id": template["subject_id"],
                    "probe_subject_id": probe["subject_id"],
                    "enroll_session_id": template["enrollment_session_id"],
                    "probe_session_id": probe["session_timestamp"],
                    "probe_window_index": int(probe["array_index"]),
                    "probe_raw_range_id": probe["parent_raw_range_id"],
                    "time_gap_days": probe_gap,
                    "probe_reference_enrollment_session_id": probe_ref_session,
                    "probe_time_gap_days": probe_gap,
                    "probe_time_gap_bucket": gap_bucket(probe_gap),
                    "same_subject": False,
                    "same_session": template["enrollment_session_id"] == probe["session_timestamp"],
                    "k": k,
                    "m": m,
                    "input_protocol_id": config["input_protocol_id"],
                    "protocol_id": config["protocol_id"],
                }
            )
            split_count += 1
    return trials


def protocol_summary(
    config: dict[str, Any],
    templates: list[dict[str, Any]],
    genuine: list[dict[str, Any]],
    impostor: list[dict[str, Any]],
    verification: list[dict[str, Any]],
    common_windows: list[dict[str, Any]],
    later_probe_pools: dict[str, list[dict[str, Any]]],
) -> dict[str, Any]:
    """Build protocol generation summary."""

    per_split = {}
    for split in SPLITS:
        split_genuine = [row for row in genuine if row["split"] == split]
        split_impostor = [row for row in impostor if row["split"] == split]
        per_split[split] = {
            "enrollment_templates": sum(1 for row in templates if row["split"] == split),
            "genuine_trials": len(split_genuine),
            "impostor_trials": len(split_impostor),
            "verification_trials": len(split_genuine) + len(split_impostor),
            "genuine_time_gap_days": numeric_summary(row["time_gap_days"] for row in split_genuine),
            "genuine_gap_bucket_distribution": bucket_distribution(split_genuine),
            "impostor_gap_bucket_distribution": bucket_distribution(split_impostor),
            "verification_gap_bucket_distribution": bucket_distribution(split_genuine + split_impostor),
        }
    return {
        "protocol_id": config["protocol_id"],
        "input_protocol_id": config["input_protocol_id"],
        "generated_datetime_utc": utc_now_iso(),
        "split_seed": config["split_seed"],
        "k": int(config["enrollment_policy"]["k_windows"]),
        "m": int(config["probe_policy"]["m_windows"]),
        "enrollment_policy": config["enrollment_policy"],
        "probe_policy": config["probe_policy"],
        "impostor_policy": config["impostor_policy"],
        "evaluation_impostor_policy": config.get("evaluation_impostor_policy", {}),
        "threshold_policy": config["threshold_policy"],
        "morphology_validity_used_for_protocol": config["morphology_validity_used_for_protocol"],
        "common_input_windows": len(common_windows),
        "enrollment_template_count": len(templates),
        "genuine_trial_count": len(genuine),
        "impostor_trial_count": len(impostor),
        "verification_trial_count": len(verification),
        "later_probe_pool_count_by_split": {
            split: len(later_probe_pools.get(split, [])) for split in SPLITS
        },
        "genuine_gap_bucket_distribution_by_split": {
            split: per_split[split]["genuine_gap_bucket_distribution"] for split in SPLITS
        },
        "impostor_gap_bucket_distribution_by_split": {
            split: per_split[split]["impostor_gap_bucket_distribution"] for split in SPLITS
        },
        "verification_gap_bucket_distribution_by_split": {
            split: per_split[split]["verification_gap_bucket_distribution"] for split in SPLITS
        },
        "impostor_probe_policy": "later_session_only",
        "deprecated_previous_protocol": "deprecated_v1_unrestricted_impostor_probe",
        "per_split": per_split,
        "audit": {},
    }


def bucket_distribution(rows: list[dict[str, Any]]) -> dict[str, int]:
    """Count probe time-gap buckets."""

    counts = {"le_30d": 0, "31_180d": 0, "181_365d": 0, "gt_365d": 0, "unknown": 0}
    for row in rows:
        bucket = row.get("probe_time_gap_bucket") or gap_bucket(row.get("probe_time_gap_days"))
        counts[str(bucket)] = counts.get(str(bucket), 0) + 1
    return counts


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build SigD verification protocol.")
    parser.add_argument("--root", default=None)
    parser.add_argument("--config", default=None)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = detect_root(args.root)
    config = load_config(root, args.config)
    out_dir = root / "protocol" / "SigD" / "metadata"
    seed = config["split_seed"]
    split_rows = read_csv_rows(out_dir / f"subject_split_seed{seed}.csv")
    if not split_rows:
        raise SystemExit("Subject split missing. Run build_sigd_subject_split.py first.")

    windows = load_common_windows(root, config)
    subject_ids = {row["subject_id"] for row in split_rows}
    grouped = windows_by_subject_session(windows, subject_ids)
    templates, templates_by_subject = build_templates(config, split_rows, grouped)
    if len(templates) != len(split_rows):
        missing = sorted({row["subject_id"] for row in split_rows} - set(templates_by_subject))
        raise SystemExit(f"Subjects missing enrollment templates: {missing[:10]}")

    pair_rows = read_csv_rows(resolve(root, config["preprocessing_snapshot_dir"]) / "postqc_interval_pairs_10s.csv")
    pair_gaps = pair_gap_map(pair_rows)
    genuine = build_genuine_trials(config, split_rows, grouped, templates_by_subject, pair_gaps)
    later_pools = later_probe_pool_by_split(split_rows, windows, templates_by_subject)
    impostor = build_impostor_trials(config, templates, windows, genuine, split_rows, templates_by_subject, pair_gaps)
    verification = sorted(genuine + impostor, key=lambda row: (row["split"], int(row["label"]), row["trial_id"]))
    summary = protocol_summary(config, templates, genuine, impostor, verification, windows, later_pools)

    write_csv(out_dir / f"enrollment_templates_k5_seed{seed}.csv", templates, TEMPLATE_COLUMNS, args.overwrite)
    write_csv(out_dir / f"genuine_trials_k5m1_seed{seed}.csv", genuine, TRIAL_COLUMNS, args.overwrite)
    write_csv(out_dir / f"impostor_trials_k5m1_seed{seed}.csv", impostor, TRIAL_COLUMNS, args.overwrite)
    write_csv(out_dir / f"verification_trials_k5m1_seed{seed}.csv", verification, TRIAL_COLUMNS, args.overwrite)
    write_json(out_dir / f"protocol_summary_k5m1_seed{seed}.json", summary, args.overwrite)
    print(
        f"templates={len(templates)} genuine={len(genuine)} impostor={len(impostor)} "
        f"verification={len(verification)}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
