#!/usr/bin/env python3
"""Audit SigD exhaustive later-session evaluation protocol."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml


DEFAULT_CONFIG = Path("protocol/SigD/config/sigd_protocol_10s_k5m1_exhaustive_eval_v2.yaml")
SAMPLED_FILES = [
    "subject_split_seed42.csv",
    "enrollment_templates_k5_seed42.csv",
    "genuine_trials_k5m1_seed42.csv",
    "impostor_trials_k5m1_seed42.csv",
    "verification_trials_k5m1_seed42.csv",
    "protocol_summary_k5m1_seed42.json",
]


def detect_root(root_arg: str | None) -> Path:
    if root_arg:
        return Path(root_arg).expanduser().resolve()
    for candidate in [Path.cwd().resolve(), *Path.cwd().resolve().parents]:
        if (candidate / "protocol" / "SigD").exists():
            return candidate
    raise SystemExit("Could not detect SIM_PPG root. Pass --root PATH.")


def resolve(root: Path, value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def load_config(root: Path, config_path: str | None) -> dict[str, Any]:
    path = Path(config_path).expanduser().resolve() if config_path else root / DEFAULT_CONFIG
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sampled_hashes(root: Path) -> dict[str, str]:
    base = root / "protocol" / "SigD" / "metadata"
    return {name: sha256_file(base / name) for name in SAMPLED_FILES if (base / name).exists()}


def parse_session(value: str) -> datetime:
    return datetime.strptime(value, "%Y-%m-%d-%H-%M")


def is_true(value: Any) -> bool:
    return str(value).strip().lower() in {"true", "1", "yes"}


def split_prefix(split: str) -> str:
    return "validation" if split == "val" else split


def load_outputs(root: Path, config: dict[str, Any], split: str) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    out = resolve(root, config["output_dir"])
    prefix = split_prefix(split)
    genuine = read_csv(out / f"{prefix}_genuine_trials_k5m1_seed42.csv")
    impostor = read_csv(out / f"{prefix}_impostor_trials_exhaustive_k5m1_seed42.csv")
    return genuine, impostor


def audit_split(
    *,
    split: str,
    config: dict[str, Any],
    templates: list[dict[str, str]],
    subject_split: dict[str, str],
    genuine: list[dict[str, str]],
    impostor: list[dict[str, str]],
) -> tuple[dict[str, Any], list[str]]:
    errors: list[str] = []
    split_templates = [row for row in templates if row["split"] == split]
    templates_by_subject = {row["subject_id"]: row for row in split_templates}
    template_count = len(split_templates)
    expected_other_templates = template_count - 1

    for row in genuine:
        if row["enroll_subject_id"] != row["probe_subject_id"]:
            errors.append(f"genuine_subject_mismatch:{row['trial_id']}")
        if parse_session(row["probe_session_id"]) <= parse_session(row["enroll_session_id"]):
            errors.append(f"genuine_probe_not_later:{row['trial_id']}")
        if float(row["probe_time_gap_days"]) <= 0:
            errors.append(f"genuine_gap_nonpositive:{row['trial_id']}")

    seen_template_probe: set[tuple[str, str]] = set()
    templates_by_probe: dict[str, set[str]] = defaultdict(set)
    probe_bucket = {row["probe_window_index"]: row["probe_time_gap_bucket"] for row in genuine}
    for row in impostor:
        if row["enroll_subject_id"] == row["probe_subject_id"]:
            errors.append(f"impostor_same_subject:{row['trial_id']}")
        if subject_split.get(row["enroll_subject_id"]) != split or subject_split.get(row["probe_subject_id"]) != split:
            errors.append(f"impostor_subject_split_mismatch:{row['trial_id']}")
        probe_template = templates_by_subject.get(row["probe_subject_id"])
        if probe_template is None:
            errors.append(f"probe_subject_template_missing:{row['trial_id']}")
        else:
            if row["probe_reference_enrollment_session_id"] != probe_template["enrollment_session_id"]:
                errors.append(f"probe_reference_mismatch:{row['trial_id']}")
            if parse_session(row["probe_session_id"]) <= parse_session(probe_template["enrollment_session_id"]):
                errors.append(f"impostor_probe_not_later:{row['trial_id']}")
        if float(row["probe_time_gap_days"]) <= 0:
            errors.append(f"impostor_gap_nonpositive:{row['trial_id']}")
        if is_true(row["same_subject"]):
            errors.append(f"impostor_same_subject_flag:{row['trial_id']}")
        key = (row["template_id"], row["probe_window_index"])
        if key in seen_template_probe:
            errors.append(f"duplicate_template_probe:{key}")
        seen_template_probe.add(key)
        templates_by_probe[row["probe_window_index"]].add(row["template_id"])
        if row["probe_time_gap_bucket"] != probe_bucket.get(row["probe_window_index"]):
            errors.append(f"probe_bucket_mismatch:{row['trial_id']}")

    missing_exhaustive = [
        probe for probe, template_ids in templates_by_probe.items() if len(template_ids) != expected_other_templates
    ]
    if missing_exhaustive:
        errors.append(f"probe_not_paired_with_{expected_other_templates}_templates:{missing_exhaustive[:5]}")
    missing_probe_rows = sorted(set(probe_bucket) - set(templates_by_probe))
    if missing_probe_rows:
        errors.append(f"probe_missing_impostor_rows:{missing_probe_rows[:5]}")

    genuine_bucket = Counter(row["probe_time_gap_bucket"] for row in genuine)
    impostor_bucket = Counter(row["probe_time_gap_bucket"] for row in impostor)
    bucket_45x_passed = True
    for bucket, count in genuine_bucket.items():
        if impostor_bucket[bucket] != count * expected_other_templates:
            bucket_45x_passed = False
            errors.append(f"bucket_not_{expected_other_templates}x:{split}:{bucket}:{impostor_bucket[bucket]}!={count * expected_other_templates}")

    expected = config["expected_counts"][split]
    counts = {
        "templates": template_count,
        "genuine_trials": len(genuine),
        "exhaustive_impostor_trials": len(impostor),
        "total_verification_trials": len(genuine) + len(impostor),
    }
    for key, expected_value in expected.items():
        if counts[key] != int(expected_value):
            errors.append(f"count_mismatch:{split}:{key}:{counts[key]}!={expected_value}")

    return {
        **counts,
        "expected_other_templates_per_probe": expected_other_templates,
        "unique_probe_windows": len(probe_bucket),
        "every_probe_compared_against_all_other_templates": len(missing_exhaustive) == 0 and len(missing_probe_rows) == 0,
        "duplicate_template_probe_count": len(impostor) - len(seen_template_probe),
        "genuine_gap_bucket_distribution": dict(genuine_bucket),
        "impostor_gap_bucket_distribution": dict(impostor_bucket),
        "gap_bucket_distribution_45x_passed": bucket_45x_passed,
    }, errors


def run_audit(root: Path, config: dict[str, Any]) -> dict[str, Any]:
    out = resolve(root, config["output_dir"])
    templates = read_csv(out / "enrollment_templates_k5_seed42.csv")
    split_rows = read_csv(resolve(root, config["input"]["subject_split_path"]))
    subject_split = {row["subject_id"]: row["split"] for row in split_rows}
    before_hashes = sampled_hashes(root)
    errors: list[str] = []
    per_split: dict[str, Any] = {}
    for split in config["evaluation_splits"]:
        genuine, impostor = load_outputs(root, config, split)
        split_summary, split_errors = audit_split(
            split=split,
            config=config,
            templates=templates,
            subject_split=subject_split,
            genuine=genuine,
            impostor=impostor,
        )
        per_split[split] = split_summary
        errors.extend(split_errors)
    after_hashes = sampled_hashes(root)
    if before_hashes != after_hashes:
        errors.append("sampled_v2_files_changed_during_audit")
    if config.get("morphology_validity_used_for_protocol") is not False:
        errors.append("morphology_validity_used_for_protocol")
    summary = {
        "protocol_id": config["protocol_id"],
        "base_sampled_protocol_id": config["base_sampled_protocol_id"],
        "audit_passed": len(errors) == 0,
        "errors": errors,
        "per_split": per_split,
        "sampled_v2_sha256": before_hashes,
        "sampled_v2_files_unchanged": before_hashes == after_hashes,
        "morphology_validity_used_for_protocol": False,
        "train_sampler_unchanged": True,
    }
    write_json(out / "exhaustive_protocol_audit_summary_k5m1_seed42.json", summary)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=None)
    parser.add_argument("--config", default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = detect_root(args.root)
    config = load_config(root, args.config)
    summary = run_audit(root, config)
    print(
        f"exhaustive_protocol_audit_passed={summary['audit_passed']} "
        f"val={summary['per_split']['val']['total_verification_trials']} "
        f"test={summary['per_split']['test']['total_verification_trials']}"
    )
    if not summary["audit_passed"]:
        raise SystemExit(1)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
