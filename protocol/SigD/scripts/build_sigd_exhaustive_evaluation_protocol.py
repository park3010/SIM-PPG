#!/usr/bin/env python3
"""Build later-session-only exhaustive impostor evaluation protocol for SigD."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml


DEFAULT_CONFIG = Path("protocol/SigD/config/sigd_protocol_10s_k5m1_exhaustive_eval_v2.yaml")
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
    "source_trial_protocol_id",
    "impostor_generation_mode",
]
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
SAMPLED_FILES = [
    "subject_split_seed42.csv",
    "enrollment_templates_k5_seed42.csv",
    "genuine_trials_k5m1_seed42.csv",
    "impostor_trials_k5m1_seed42.csv",
    "verification_trials_k5m1_seed42.csv",
    "protocol_summary_k5m1_seed42.json",
]


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


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


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sampled_protocol_hashes(root: Path) -> dict[str, str]:
    base = root / "protocol" / "SigD" / "metadata"
    return {name: sha256_file(base / name) for name in SAMPLED_FILES if (base / name).exists()}


def as_float(value: Any) -> float:
    return float(value)


def parse_session(value: str) -> datetime:
    return datetime.strptime(value, "%Y-%m-%d-%H-%M")


def row_sort_key(row: dict[str, Any]) -> tuple[Any, ...]:
    return (
        row["split"],
        row["trial_type"],
        row["template_id"],
        row["probe_subject_id"],
        row["probe_session_id"],
        int(float(row["probe_window_index"])),
    )


def normalize_template(row: dict[str, str], protocol_id: str) -> dict[str, Any]:
    out = dict(row)
    out["protocol_id"] = protocol_id
    return out


def normalize_genuine(row: dict[str, str], protocol_id: str, source_protocol_id: str) -> dict[str, Any]:
    out: dict[str, Any] = dict(row)
    out["label"] = 1
    out["protocol_id"] = protocol_id
    out["source_trial_protocol_id"] = source_protocol_id
    out["impostor_generation_mode"] = ""
    return out


def build_exhaustive_impostors(
    *,
    split: str,
    templates: list[dict[str, Any]],
    genuine_rows: list[dict[str, Any]],
    config: dict[str, Any],
) -> list[dict[str, Any]]:
    """Pair every later-session probe with every other-subject template."""

    protocol_id = config["protocol_id"]
    source_protocol_id = config["base_sampled_protocol_id"]
    k = int(config["enrollment_policy"]["k_windows"])
    m = int(config["probe_policy"]["m_windows"])
    split_templates = sorted([row for row in templates if row["split"] == split], key=lambda row: row["template_id"])
    counter = 0
    rows: list[dict[str, Any]] = []
    for probe in sorted(genuine_rows, key=lambda row: (row["probe_subject_id"], row["probe_session_id"], int(float(row["probe_window_index"])))):
        for template in split_templates:
            if template["subject_id"] == probe["probe_subject_id"]:
                continue
            rows.append(
                {
                    "trial_id": f"{split}_impostor_exhaustive_{counter:09d}",
                    "split": split,
                    "label": 0,
                    "trial_type": "impostor",
                    "template_id": template["template_id"],
                    "enroll_subject_id": template["subject_id"],
                    "probe_subject_id": probe["probe_subject_id"],
                    "enroll_session_id": template["enrollment_session_id"],
                    "probe_session_id": probe["probe_session_id"],
                    "probe_window_index": int(float(probe["probe_window_index"])),
                    "probe_raw_range_id": probe["probe_raw_range_id"],
                    "time_gap_days": probe["probe_time_gap_days"],
                    "probe_reference_enrollment_session_id": probe["probe_reference_enrollment_session_id"],
                    "probe_time_gap_days": probe["probe_time_gap_days"],
                    "probe_time_gap_bucket": probe["probe_time_gap_bucket"],
                    "same_subject": False,
                    "same_session": False,
                    "k": k,
                    "m": m,
                    "input_protocol_id": config["input_protocol_id"],
                    "protocol_id": protocol_id,
                    "source_trial_protocol_id": source_protocol_id,
                    "impostor_generation_mode": "exhaustive_later_session_only",
                }
            )
            counter += 1
    return rows


def bucket_counts(rows: list[dict[str, Any]]) -> dict[str, int]:
    counts = Counter(str(row.get("probe_time_gap_bucket", "")) for row in rows)
    return dict(sorted(counts.items()))


def validate_genuine_rows(rows: list[dict[str, Any]]) -> list[str]:
    errors = []
    for row in rows:
        if row["enroll_subject_id"] != row["probe_subject_id"]:
            errors.append(f"genuine_subject_mismatch:{row['trial_id']}")
        if parse_session(row["probe_session_id"]) <= parse_session(row["enroll_session_id"]):
            errors.append(f"genuine_probe_not_later:{row['trial_id']}")
        if as_float(row["probe_time_gap_days"]) <= 0:
            errors.append(f"genuine_gap_nonpositive:{row['trial_id']}")
    return errors


def build_protocol(root: Path, config: dict[str, Any], overwrite: bool) -> dict[str, Any]:
    before_hashes = sampled_protocol_hashes(root)
    output_dir = resolve(root, config["output_dir"])
    seed = int(config.get("split_seed", 42))
    templates_source = read_csv_rows(resolve(root, config["input"]["enrollment_templates_path"]))
    sampled_genuine_source = read_csv_rows(resolve(root, config["input"]["sampled_genuine_trials_path"]))
    templates = [normalize_template(row, config["protocol_id"]) for row in templates_source if row["split"] in config["evaluation_splits"]]
    genuine_all = [
        normalize_genuine(row, config["protocol_id"], config["base_sampled_protocol_id"])
        for row in sampled_genuine_source
        if row["split"] in config["evaluation_splits"]
    ]
    errors = validate_genuine_rows(genuine_all)
    if errors:
        raise SystemExit(f"Invalid source genuine rows: {errors[:5]}")

    summary_split: dict[str, Any] = {}
    all_outputs: dict[str, list[dict[str, Any]]] = {}
    for split in config["evaluation_splits"]:
        split_templates = [row for row in templates if row["split"] == split]
        split_genuine = [row for row in genuine_all if row["split"] == split]
        split_impostor = build_exhaustive_impostors(
            split=split,
            templates=templates,
            genuine_rows=split_genuine,
            config=config,
        )
        split_verification = sorted(split_genuine + split_impostor, key=row_sort_key)
        all_outputs[f"{split}_genuine"] = sorted(split_genuine, key=row_sort_key)
        all_outputs[f"{split}_impostor"] = sorted(split_impostor, key=row_sort_key)
        all_outputs[f"{split}_verification"] = split_verification
        summary_split[split] = {
            "templates": len(split_templates),
            "later_probe_windows": len(split_genuine),
            "genuine_trials": len(split_genuine),
            "exhaustive_impostor_trials": len(split_impostor),
            "total_verification_trials": len(split_verification),
            "genuine_gap_bucket_distribution": bucket_counts(split_genuine),
            "impostor_gap_bucket_distribution": bucket_counts(split_impostor),
            "verification_gap_bucket_distribution": bucket_counts(split_verification),
        }
        expected = config["expected_counts"][split]
        for key in ("templates", "genuine_trials", "exhaustive_impostor_trials", "total_verification_trials"):
            if summary_split[split][key] != int(expected[key]):
                raise SystemExit(f"Count mismatch for {split}/{key}: {summary_split[split][key]} != {expected[key]}")

    write_csv(output_dir / f"enrollment_templates_k5_seed{seed}.csv", sorted(templates, key=lambda row: row["template_id"]), TEMPLATE_COLUMNS, overwrite)
    for split in config["evaluation_splits"]:
        write_csv(output_dir / f"{split if split != 'val' else 'validation'}_genuine_trials_k5m1_seed{seed}.csv", all_outputs[f"{split}_genuine"], TRIAL_COLUMNS, overwrite)
        write_csv(output_dir / f"{split if split != 'val' else 'validation'}_impostor_trials_exhaustive_k5m1_seed{seed}.csv", all_outputs[f"{split}_impostor"], TRIAL_COLUMNS, overwrite)
        write_csv(output_dir / f"{split if split != 'val' else 'validation'}_verification_trials_exhaustive_k5m1_seed{seed}.csv", all_outputs[f"{split}_verification"], TRIAL_COLUMNS, overwrite)

    after_hashes = sampled_protocol_hashes(root)
    summary = {
        "protocol_id": config["protocol_id"],
        "base_sampled_protocol_id": config["base_sampled_protocol_id"],
        "input_protocol_id": config["input_protocol_id"],
        "generated_datetime_utc": utc_now_iso(),
        "evaluation_splits": config["evaluation_splits"],
        "enrollment_policy": config["enrollment_policy"],
        "probe_policy": config["probe_policy"],
        "impostor_policy": config["impostor_policy"],
        "threshold_policy": config["threshold_policy"],
        "morphology_validity_used_for_protocol": config["morphology_validity_used_for_protocol"],
        "per_split": summary_split,
        "every_probe_compared_against_all_other_templates": True,
        "train_sampler_unchanged": True,
        "final_scientific_reporting_protocol": True,
        "sampled_v2_sha256_before": before_hashes,
        "sampled_v2_sha256_after": after_hashes,
        "sampled_v2_files_unchanged": before_hashes == after_hashes,
    }
    write_json(output_dir / f"exhaustive_protocol_summary_k5m1_seed{seed}.json", summary, overwrite)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=None)
    parser.add_argument("--config", default=None)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = detect_root(args.root)
    config = load_config(root, args.config)
    summary = build_protocol(root, config, args.overwrite)
    val = summary["per_split"]["val"]
    test = summary["per_split"]["test"]
    print(
        "exhaustive_protocol_built=True "
        f"val={val['total_verification_trials']} test={test['total_verification_trials']} "
        f"sampled_v2_unchanged={summary['sampled_v2_files_unchanged']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
