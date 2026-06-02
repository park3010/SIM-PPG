#!/usr/bin/env python3
"""Run evaluator correctness audit with a deterministic MockEncoder."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import torch

SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))

from common import (  # noqa: E402
    add_data_pipeline_src,
    detect_project_root,
    distribution,
    ensure_dir,
    load_csv_rows,
    load_eval_config,
    load_yaml_config,
    numeric_summary,
    resolve_from_root,
    utc_now_iso,
    write_csv_rows,
    write_json,
)
from cosine_verifier import score_trials  # noqa: E402
from embedding_cache import collect_unique_array_indices, compute_embedding_cache, save_embedding_cache  # noqa: E402
from metrics import compute_split_metrics  # noqa: E402
from mock_encoder import DeterministicMockEncoder, mock_encoder_metadata  # noqa: E402
from reporting import prepare_result_root, write_metrics_json, write_run_manifest, write_score_csv  # noqa: E402
from stratified_analysis import macro_summary, session_pair_macro_table, subject_macro_metrics, time_gap_metrics  # noqa: E402
from thresholds import compute_eer_threshold, compute_far_target_threshold  # noqa: E402


def _load_data_pipeline_objects(root: Path, eval_config: dict):
    add_data_pipeline_src(root)
    from manifest_index import ManifestIndex
    from transforms import PerWindowZScore
    from verification_trial_dataset import VerificationTrialDataset

    dp_config_path = resolve_from_root(root, eval_config["input"]["data_pipeline_config_path"])
    assert dp_config_path is not None
    dp_config = load_yaml_config(dp_config_path)
    manifest_index = ManifestIndex(root, dp_config)
    transform = PerWindowZScore(
        eps=float(dp_config["normalization"]["epsilon"]),
        output_channel_first=bool(dp_config["normalization"]["output_channel_first"]),
    )
    datasets = {
        split: VerificationTrialDataset(root, dp_config, manifest_index, split, transform)
        for split in eval_config["input"]["evaluation_splits"]
    }
    return dp_config, manifest_index, transform, datasets


def _assert_config_ids(eval_config: dict, dp_config: dict) -> None:
    if eval_config["input"]["input_protocol_id"] != dp_config["input"]["input_protocol_id"]:
        raise RuntimeError("input_protocol_id mismatch between evaluation and data pipeline configs.")
    if eval_config["input"]["protocol_id"] != dp_config["protocol"]["protocol_id"]:
        raise RuntimeError("protocol_id mismatch between evaluation and data pipeline configs.")


def _device_from_config(config: dict) -> torch.device:
    requested = str(config["encoder"].get("device", "auto"))
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(requested)


def _load_protocol_rows(root: Path, dp_config: dict, split: str) -> tuple[list[dict], list[dict]]:
    """Load enrollment templates and split-specific trials from sampled or exhaustive configs."""

    template_rows = load_csv_rows(resolve_from_root(root, dp_config["protocol"]["enrollment_templates_path"]))
    split_paths = dp_config["protocol"].get("verification_trial_paths_by_split")
    if split_paths:
        if split not in split_paths:
            raise RuntimeError(f"No verification trial path configured for split={split}.")
        trial_rows = load_csv_rows(resolve_from_root(root, split_paths[split]))
    else:
        all_trials = load_csv_rows(resolve_from_root(root, dp_config["protocol"]["verification_trials_path"]))
        trial_rows = [row for row in all_trials if row["split"] == split]
    trial_rows = [row for row in trial_rows if row["split"] == split]
    return template_rows, trial_rows


def _score_split(
    *,
    root: Path,
    eval_config: dict,
    dp_config: dict,
    manifest_index,
    transform,
    encoder: DeterministicMockEncoder,
    split: str,
    result_root: Path,
    device: torch.device,
) -> tuple[list[dict], dict]:
    template_rows, trial_rows = _load_protocol_rows(root, dp_config, split)
    unique_indices = collect_unique_array_indices(template_rows, trial_rows)
    embeddings = compute_embedding_cache(
        manifest_index=manifest_index,
        transform=transform,
        encoder=encoder,
        array_indices=unique_indices,
        batch_size=int(eval_config["encoder"]["batch_size"]),
        device=device,
    )
    cache_dir = ensure_dir(result_root / "cache")
    cache_metadata = {
        "encoder_id": encoder.encoder_id,
        "checkpoint_sha256": "mock_encoder_no_checkpoint",
        "input_protocol_id": eval_config["input"]["input_protocol_id"],
        "protocol_id": eval_config["input"]["protocol_id"],
        "normalization_policy": dp_config["normalization"]["policy"],
        "split": split,
        "scientific_reporting_allowed": False,
    }
    save_embedding_cache(
        path=cache_dir / f"{split}_embeddings.npz",
        manifest_path=cache_dir / f"{split}_cache_manifest.json",
        embeddings=embeddings,
        metadata=cache_metadata,
    )
    scores = score_trials(
        trial_rows=trial_rows,
        template_rows=template_rows,
        embedding_cache=embeddings,
        encoder_id=encoder.encoder_id,
        eps=float(eval_config["template_aggregation"]["epsilon"]),
    )
    score_filename = "validation_scores.csv" if split == "val" else f"{split}_scores.csv"
    write_score_csv(result_root / score_filename, scores)
    return scores, {
        "split": split,
        "trial_count": len(trial_rows),
        "unique_window_count": len(unique_indices),
        "score_summary": numeric_summary(row["score"] for row in scores),
        "score_finite": all(np.isfinite(float(row["score"])) for row in scores),
    }


def run_mock_audit(root: Path, config_path: str | Path | None = None) -> dict:
    """Run MockEncoder audit and write outputs."""

    eval_config = load_eval_config(root, config_path)
    dp_config, manifest_index, transform, datasets = _load_data_pipeline_objects(root, eval_config)
    _assert_config_ids(eval_config, dp_config)
    mock_result_name = (
        "mock_encoder_engine_audit_exhaustive"
        if "EXHAUSTIVE" in eval_config["input"]["protocol_id"]
        else "mock_encoder_engine_audit"
    )
    result_root = prepare_result_root(root / "evaluation" / "SigD" / "results" / mock_result_name)
    encoder = DeterministicMockEncoder(int(eval_config["mock_encoder_audit"]["embedding_dim"]))
    device = _device_from_config(eval_config)

    sample_shapes = {}
    for split, dataset in datasets.items():
        sample = dataset[0]
        sample_shapes[split] = {
            "enrollment_windows": list(sample["enrollment_windows"].shape),
            "probe_window": list(sample["probe_window"].shape),
        }

    split_scores: dict[str, list[dict]] = {}
    split_summaries = {}
    for split in eval_config["input"]["evaluation_splits"]:
        scores, summary = _score_split(
            root=root,
            eval_config=eval_config,
            dp_config=dp_config,
            manifest_index=manifest_index,
            transform=transform,
            encoder=encoder,
            split=split,
            result_root=result_root,
            device=device,
        )
        split_scores[split] = scores
        split_summaries[split] = summary

    val_scores = split_scores["val"]
    val_labels = [int(row["label"]) for row in val_scores]
    val_score_values = [float(row["score"]) for row in val_scores]
    eer_selection = compute_eer_threshold(val_labels, val_score_values)
    far_selection = compute_far_target_threshold(
        val_labels,
        val_score_values,
        float(eval_config["thresholds"]["far_target_threshold"]["far_target"]),
    )
    threshold_payload = {
        "source_split": "val",
        "validation_eer_threshold": eer_selection,
        "validation_far_1pct_threshold": far_selection,
        "test_threshold_tuning_performed": False,
    }
    write_json(result_root / "threshold_selection_from_validation.json", threshold_payload)

    val_metrics = compute_split_metrics(
        split="val",
        labels=val_labels,
        scores=val_score_values,
        validation_eer_threshold=float(eer_selection["threshold"]),
        validation_far_threshold=float(far_selection["threshold"]),
    )
    test_scores = split_scores["test"]
    test_labels = [int(row["label"]) for row in test_scores]
    test_score_values = [float(row["score"]) for row in test_scores]
    test_metrics = compute_split_metrics(
        split="test",
        labels=test_labels,
        scores=test_score_values,
        validation_eer_threshold=float(eer_selection["threshold"]),
        validation_far_threshold=float(far_selection["threshold"]),
    )
    write_metrics_json(result_root / "validation_metrics.json", val_metrics)
    write_metrics_json(result_root / "test_metrics.json", test_metrics)

    gap_rows = time_gap_metrics(
        test_scores,
        float(eer_selection["threshold"]),
        list(eval_config["metrics"]["stratified"]["time_gap_buckets"]),
        threshold_name="validation_eer_threshold",
    ) + time_gap_metrics(
        test_scores,
        float(far_selection["threshold"]),
        list(eval_config["metrics"]["stratified"]["time_gap_buckets"]),
        threshold_name="validation_far_1pct_threshold",
    )
    subject_rows = subject_macro_metrics(
        test_scores,
        float(eer_selection["threshold"]),
        threshold_name="validation_eer_threshold",
    ) + subject_macro_metrics(
        test_scores,
        float(far_selection["threshold"]),
        threshold_name="validation_far_1pct_threshold",
    )
    session_pair_rows = session_pair_macro_table(
        test_scores,
        float(eer_selection["threshold"]),
        threshold_name="validation_eer_threshold",
    ) + session_pair_macro_table(
        test_scores,
        float(far_selection["threshold"]),
        threshold_name="validation_far_1pct_threshold",
    )
    write_csv_rows(result_root / "test_time_gap_metrics.csv", gap_rows)
    write_csv_rows(result_root / "test_subject_macro_metrics.csv", subject_rows)
    write_csv_rows(result_root / "test_session_pair_macro_metrics.csv", session_pair_rows)

    run_manifest = {
        "evaluation_id": eval_config["evaluation_id"],
        "encoder_id": encoder.encoder_id,
        "encoder_variant": "deterministic_mock",
        "encoder_mode": "audit_only",
        "checkpoint_path": None,
        "checkpoint_sha256": None,
        "pretrained_weights_verified": False,
        "scientific_reporting_allowed": False,
        "metrics_valid_for_scientific_reporting": False,
        "input_protocol_id": eval_config["input"]["input_protocol_id"],
        "protocol_id": eval_config["input"]["protocol_id"],
        "normalization_policy": dp_config["normalization"]["policy"],
        "input_setting": "common_input",
        "official_native_preprocessing_reapplied": False,
        "common_transform_source": "data_pipeline/SigD/PerWindowZScore",
        "native_input_supplementary_evaluation_pending": True,
        "val_trial_count": len(val_scores),
        "test_trial_count": len(test_scores),
        "validation_threshold_only": True,
        "impostor_evaluation_mode": eval_config.get("fairness", {}).get("impostor_evaluation_mode", ""),
        "final_headline_reporting_status": eval_config.get("fairness", {}).get("final_headline_reporting_status", ""),
        "source_level_overlap_limitation_note": "MockEncoder audit only; no scientific baseline is reported.",
        "generated_datetime": utc_now_iso(),
    }
    write_run_manifest(result_root / "frozen_baseline_run_manifest.json", run_manifest)

    audit = {
        "audit_datetime_utc": utc_now_iso(),
        "evaluation_id": eval_config["evaluation_id"],
        "mock_encoder": mock_encoder_metadata(),
        "device": str(device),
        "sample_shapes": sample_shapes,
        "split_summaries": split_summaries,
        "threshold_selection": threshold_payload,
        "validation_metrics_path": str(result_root / "validation_metrics.json"),
        "test_metrics_path": str(result_root / "test_metrics.json"),
        "time_gap_metrics_path": str(result_root / "test_time_gap_metrics.csv"),
        "subject_macro_summary": macro_summary(subject_rows),
        "session_pair_macro_summary": macro_summary(session_pair_rows),
        "scientific_reporting_allowed": False,
        "test_threshold_tuning_performed": False,
        "score_range_passed": all(
            -1.000001 <= float(row["score"]) <= 1.000001
            for rows in split_scores.values()
            for row in rows
        ),
        "passed": True,
        "errors": [],
    }
    if not audit["score_range_passed"]:
        audit["passed"] = False
        audit["errors"].append("score_out_of_cosine_range")
    write_json(root / "evaluation" / "SigD" / "metadata" / "evaluation_engine_audit_summary.json", audit)
    return audit


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=None)
    parser.add_argument("--config", default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = detect_project_root(args.root)
    audit = run_mock_audit(root, args.config)
    print(
        f"evaluation_engine_audit_passed={audit['passed']} "
        f"val={audit['split_summaries']['val']['trial_count']} "
        f"test={audit['split_summaries']['test']['trial_count']}"
    )
    if not audit["passed"]:
        raise SystemExit(1)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
