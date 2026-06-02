#!/usr/bin/env python3
"""Audit the SigD common data pipeline without regenerating inputs."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
import sys
from typing import Any

import numpy as np
from torch.utils.data import DataLoader

SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))

from collate import train_collate_fn  # noqa: E402
from common import (  # noqa: E402
    detect_project_root,
    distribution,
    load_csv_rows,
    load_json,
    load_pipeline_config,
    numeric_summary,
    resolve_from_root,
    utc_now_iso,
    write_json,
)
from common_window_dataset import CommonPPGWindowDataset  # noqa: E402
from manifest_index import ManifestIndex  # noqa: E402
from session_aware_batch_sampler import SessionAwareBatchSampler  # noqa: E402
from train_subject_pool import TrainSubjectPool  # noqa: E402
from transforms import PerWindowZScore  # noqa: E402
from verification_trial_dataset import VerificationTrialDataset  # noqa: E402


def finite_chunk_check(array: np.ndarray, chunk_size: int = 1024) -> dict[str, Any]:
    """Check finite values without copying the full array."""

    nonfinite = 0
    total = int(np.prod(array.shape))
    for start in range(0, array.shape[0], chunk_size):
        chunk = array[start : start + chunk_size]
        nonfinite += int((~np.isfinite(chunk)).sum())
    return {
        "finite": nonfinite == 0,
        "nonfinite_count": nonfinite,
        "total_values": total,
    }


def evaluation_summary(config: dict[str, Any], datasets: dict[str, VerificationTrialDataset]) -> dict[str, Any]:
    """Summarize fixed evaluation datasets."""

    output = {}
    for split, dataset in datasets.items():
        sample = dataset[0]
        labels = [int(row["label"]) for row in dataset.trials]
        output[split] = {
            "length": len(dataset),
            "label_distribution": distribution(labels),
            "sample_enrollment_shape": list(sample["enrollment_windows"].shape),
            "sample_probe_shape": list(sample["probe_window"].shape),
            "sample_protocol_id": sample["protocol_id"],
            "probe_time_gap_days_summary": numeric_summary(row["probe_time_gap_days"] for row in dataset.trials),
        }
    return output


def sampler_batch_summary(batch: dict[str, Any]) -> dict[str, Any]:
    """Summarize one collated train batch."""

    subject_sessions: dict[str, set[str]] = defaultdict(set)
    subject_session_counts: dict[tuple[str, str], int] = defaultdict(int)
    samples_per_subject: dict[str, int] = defaultdict(int)
    indices_by_subject: dict[str, list[int]] = defaultdict(list)
    array_indices = [int(index) for index in batch["array_indices"].tolist()]
    for array_index, subject, session in zip(array_indices, batch["subject_ids"], batch["session_ids"]):
        subject_sessions[subject].add(session)
        subject_session_counts[(subject, session)] += 1
        samples_per_subject[subject] += 1
        indices_by_subject[subject].append(array_index)
    duplicate_array_index_count = len(array_indices) - len(set(array_indices))
    duplicate_array_index_within_subject_count = sum(
        len(indices) - len(set(indices)) for indices in indices_by_subject.values()
    )
    return {
        "batch_size": int(batch["waveforms"].shape[0]),
        "waveform_shape": list(batch["waveforms"].shape),
        "selected_subject_count": len(subject_sessions),
        "sessions_per_subject": {subject: len(sessions) for subject, sessions in subject_sessions.items()},
        "samples_per_subject": dict(samples_per_subject),
        "windows_per_subject_session": {
            f"{subject}|{session}": count for (subject, session), count in subject_session_counts.items()
        },
        "duplicate_array_index_count": duplicate_array_index_count,
        "duplicate_array_index_within_subject_count": duplicate_array_index_within_subject_count,
        "ipa_invalid_samples_retained": int((~batch["ipa_valid_mask"]).sum().item()),
    }


def build_sampler_summary(
    config: dict[str, Any],
    train_pool: TrainSubjectPool,
    dataset: CommonPPGWindowDataset,
    mode: str,
) -> dict[str, Any]:
    """Instantiate a sampler and verify deterministic behavior."""

    sampler_cfg = config["dynamic_sampler"]
    kwargs = {
        "mode": mode,
        "seed": int(sampler_cfg["seed"]),
        "subjects_per_batch": int(sampler_cfg["subjects_per_batch"]),
        "sessions_per_subject": int(sampler_cfg["sessions_per_subject"]),
        "windows_per_session": int(sampler_cfg["windows_per_session"]),
        "num_batches_per_epoch": int(sampler_cfg["num_batches_per_epoch"]),
    }
    sampler_a = SessionAwareBatchSampler(train_pool, **kwargs)
    sampler_a.set_epoch(0)
    batch_indices = next(iter(sampler_a))
    loader = DataLoader(dataset, batch_sampler=[batch_indices], collate_fn=train_collate_fn)
    batch = next(iter(loader))

    sampler_b = SessionAwareBatchSampler(train_pool, **kwargs)
    sampler_b.set_epoch(0)
    same_epoch_same = batch_indices == next(iter(sampler_b))
    sampler_c = SessionAwareBatchSampler(train_pool, **kwargs)
    sampler_c.set_epoch(1)
    different_epoch_differs = batch_indices != next(iter(sampler_c))
    return {
        "mode": mode,
        "deterministic_same_seed_epoch": same_epoch_same,
        "different_epoch_changes_batch": different_epoch_differs,
        "different_epoch_changes_batch_checked": True,
        "batch_indices": batch_indices,
        "batch_summary": sampler_batch_summary(batch),
    }


def main() -> int:
    root = detect_project_root(None)
    config = load_pipeline_config(root)
    metadata_dir = root / "data_pipeline" / "SigD" / "metadata"

    manifest_index = ManifestIndex(root, config)
    manifest_validation = manifest_index.validate()
    array_check = finite_chunk_check(manifest_index.array)
    protocol_summary = load_json(resolve_from_root(root, config["protocol"]["protocol_dir"]) / "protocol_summary_k5m1_seed42.json")

    transform = PerWindowZScore(
        eps=float(config["normalization"]["epsilon"]),
        output_channel_first=bool(config["normalization"]["output_channel_first"]),
    )
    datasets = {
        split: VerificationTrialDataset(root, config, manifest_index, split, transform)
        for split in config["evaluation_dataset"]["allowed_splits"]
    }
    eval_summary = evaluation_summary(config, datasets)
    write_json(metadata_dir / "evaluation_dataset_summary.json", eval_summary)

    train_pool = TrainSubjectPool(root, config, manifest_index)
    train_summary = train_pool.summary()
    write_json(metadata_dir / "train_pool_summary.json", train_summary)

    train_dataset = CommonPPGWindowDataset(manifest_index, transform=transform, index_mode="array_index")
    cross_summary = build_sampler_summary(config, train_pool, train_dataset, "same_subject_cross_session")
    any_summary = build_sampler_summary(config, train_pool, train_dataset, "same_subject_any_session")

    errors: list[str] = []
    expected_shape = tuple(config["input"]["expected_array_shape"])
    if tuple(manifest_index.array.shape) != expected_shape:
        errors.append("array_shape_mismatch")
    if str(manifest_index.array.dtype) != config["input"]["expected_dtype"]:
        errors.append("array_dtype_mismatch")
    if not array_check["finite"]:
        errors.append("array_nonfinite")
    if not manifest_validation["passed"]:
        errors.extend(manifest_validation["errors"])
    expected_lengths = {"train": 14532, "val": 15060, "test": 14370}
    for split, expected in expected_lengths.items():
        if len(datasets[split]) != expected:
            errors.append(f"evaluation_length_mismatch:{split}:{len(datasets[split])}!={expected}")
    if not train_summary["passed"]:
        errors.extend(train_summary["errors"])
    if cross_summary["batch_summary"]["batch_size"] != config["dynamic_sampler"]["batch_size"]:
        errors.append("cross_session_batch_size_mismatch")
    if any_summary["batch_summary"]["batch_size"] != config["dynamic_sampler"]["batch_size"]:
        errors.append("any_session_batch_size_mismatch")
    if cross_summary["batch_summary"]["selected_subject_count"] != config["dynamic_sampler"]["subjects_per_batch"]:
        errors.append("cross_session_subject_count_mismatch")
    if not set(cross_summary["batch_summary"]["samples_per_subject"]).issubset(train_pool.train_subject_set):
        errors.append("cross_session_non_train_subject")
    if not all(value == 2 for value in cross_summary["batch_summary"]["sessions_per_subject"].values()):
        errors.append("cross_session_distinct_session_count_mismatch")
    if not all(value == 2 for value in cross_summary["batch_summary"]["windows_per_subject_session"].values()):
        errors.append("cross_session_windows_per_subject_session_mismatch")
    if cross_summary["batch_summary"]["duplicate_array_index_count"] != 0:
        errors.append("cross_session_duplicate_array_index")
    if not cross_summary["deterministic_same_seed_epoch"]:
        errors.append("cross_session_determinism_failed")
    if not cross_summary["different_epoch_changes_batch"]:
        errors.append("cross_session_epoch_change_failed")
    if any_summary["batch_summary"]["selected_subject_count"] != config["dynamic_sampler"]["subjects_per_batch"]:
        errors.append("any_session_subject_count_mismatch")
    expected_samples_per_subject = config["dynamic_sampler"]["sessions_per_subject"] * config["dynamic_sampler"]["windows_per_session"]
    if not all(value == expected_samples_per_subject for value in any_summary["batch_summary"]["samples_per_subject"].values()):
        errors.append("any_session_samples_per_subject_mismatch")
    if not set(any_summary["batch_summary"]["samples_per_subject"]).issubset(train_pool.train_subject_set):
        errors.append("any_session_non_train_subject")
    if any_summary["batch_summary"]["duplicate_array_index_count"] != 0:
        errors.append("any_session_duplicate_array_index")
    if any_summary["batch_summary"]["duplicate_array_index_within_subject_count"] != 0:
        errors.append("any_session_duplicate_array_index_within_subject")
    if not any_summary["deterministic_same_seed_epoch"]:
        errors.append("any_session_determinism_failed")
    if not any_summary["different_epoch_changes_batch"]:
        errors.append("any_session_epoch_change_failed")
    if protocol_summary["protocol_id"] != config["protocol"]["protocol_id"]:
        errors.append("protocol_id_mismatch")
    if protocol_summary["audit"]["passed"] is not True:
        errors.append("protocol_audit_not_passed")

    summary = {
        "pipeline_id": config["pipeline_id"],
        "generated_datetime_utc": utc_now_iso(),
        "input_protocol_id": config["input"]["input_protocol_id"],
        "protocol_id": config["protocol"]["protocol_id"],
        "array_memmap_mode": "r",
        "array_shape": list(manifest_index.array.shape),
        "array_dtype": str(manifest_index.array.dtype),
        "array_finite_check": array_check,
        "manifest_validation": manifest_validation,
        "evaluation_dataset_summary_path": "data_pipeline/SigD/metadata/evaluation_dataset_summary.json",
        "train_pool_summary_path": "data_pipeline/SigD/metadata/train_pool_summary.json",
        "dynamic_sampler": {
            "same_subject_cross_session": cross_summary,
            "same_subject_any_session": any_summary,
        },
        "morphology": {
            "independent_masks": True,
            "use_for_protocol_eligibility": False,
            "use_for_sampling_filter": False,
            "ipa_invalid_samples_retained_in_cross_session_batch": cross_summary["batch_summary"]["ipa_invalid_samples_retained"],
        },
        "passed": len(errors) == 0,
        "errors": errors,
    }
    write_json(metadata_dir / "data_pipeline_audit_summary.json", summary)
    print(
        f"data_pipeline_audit_passed={summary['passed']} "
        f"train={len(datasets['train'])} val={len(datasets['val'])} test={len(datasets['test'])}"
    )
    if errors:
        raise SystemExit(1)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
