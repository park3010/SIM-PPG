#!/usr/bin/env python3
"""Run the verified PaPaGei-S frozen cosine baseline."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path
import sys

import numpy as np
import torch

SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))

from common import (  # noqa: E402
    add_data_pipeline_src,
    detect_project_root,
    ensure_dir,
    load_csv_rows,
    load_eval_config,
    load_json,
    load_yaml_config,
    numeric_summary,
    resolve_from_root,
    sha256_file,
    utc_now_iso,
    write_csv_rows,
    write_json,
)
from cosine_verifier import score_trials  # noqa: E402
from embedding_cache import collect_unique_array_indices, compute_embedding_cache, save_embedding_cache  # noqa: E402
from encoder_interface import count_trainable_parameters  # noqa: E402
from metrics import compute_split_metrics  # noqa: E402
from papagei_s_adapter import PaPaGeiSFrozenAdapter  # noqa: E402
from reporting import prepare_result_root, write_metrics_json, write_run_manifest, write_score_csv  # noqa: E402
from stratified_analysis import macro_summary, session_pair_macro_table, subject_macro_metrics, time_gap_metrics  # noqa: E402
from thresholds import compute_eer_threshold, compute_far_target_threshold  # noqa: E402


def check_official_ready(root: Path) -> dict:
    """Return official PaPaGei-S readiness status from the local manifest."""

    manifest_path = root / "evaluation" / "SigD" / "metadata" / "papagei_model_reference_manifest.json"
    if not manifest_path.exists():
        return {
            "ready": False,
            "reason": "papagei_model_reference_manifest.json missing; run setup_papagei_model_reference.py --verify first",
        }
    manifest = load_json(manifest_path)
    required = (
        "official_source_verified",
        "checkpoint_verified",
        "pretrained_weights_verified",
        "architecture_verified",
        "loading_api_verified",
        "embedding_dim_verified",
        "ready_for_scientific_frozen_baseline",
    )
    ready = all(bool(manifest.get(key)) for key in required)
    missing = [key for key in required if not bool(manifest.get(key))]
    return {"ready": ready, "manifest": manifest, "reason": None if ready else f"not_ready:{missing}"}


def write_skipped_manifest(root: Path, config: dict, reason: str) -> Path:
    """Record that the official baseline was intentionally skipped."""

    path = root / "evaluation" / "SigD" / "metadata" / "frozen_baseline_run_manifest.json"
    write_json(
        path,
        {
            "evaluation_id": config["evaluation_id"],
            "encoder_id": "papagei_s_frozen_official_common_input_v1",
            "encoder_variant": "PaPaGei-S",
            "encoder_architecture": "ResNet1DMoE",
            "encoder_mode": "frozen",
            "scientific_reporting_allowed": False,
            "official_baseline_status": "skipped_due_to_missing_verified_checkpoint",
            "skip_reason": reason,
            "random_initialization_allowed": False,
            "input_setting": "common_input",
            "official_native_preprocessing_reapplied": False,
            "generated_datetime": utc_now_iso(),
        },
    )
    return path


def official_source_provenance(root: Path) -> dict:
    """Load official source manifest and adapter source hash."""

    source_manifest_path = root / "evaluation" / "SigD" / "metadata" / "papagei_source_snapshot_manifest.json"
    source_manifest = load_json(source_manifest_path)
    adapter_source = root / "evaluation" / "SigD" / "src" / "papagei_s_adapter.py"
    return {
        "official_source_repository": source_manifest.get("source_repository"),
        "official_source_git_commit_sha": source_manifest.get("git_commit_sha"),
        "official_source_archive_sha256": source_manifest.get("archive_sha256"),
        "official_source_manifest_path": "evaluation/SigD/metadata/papagei_source_snapshot_manifest.json",
        "official_source_manifest_sha256": sha256_file(source_manifest_path),
        "adapter_source_sha256": sha256_file(adapter_source),
    }


def ensure_scientific_result_root_available(result_root: Path, overwrite: bool) -> None:
    """Protect existing scientific result files unless --overwrite is explicit."""

    protected = [
        result_root / "frozen_baseline_run_manifest.json",
        result_root / "validation_scores.csv",
        result_root / "test_scores.csv",
    ]
    existing = [path for path in protected if path.exists()]
    if existing and not overwrite:
        existing_text = ", ".join(str(path) for path in existing)
        raise RuntimeError(f"Scientific result files already exist; pass --overwrite to replace them: {existing_text}")
    if existing and overwrite:
        shutil.rmtree(result_root)


def _device_from_config(config: dict) -> torch.device:
    requested = str(config["encoder"].get("device", "auto"))
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(requested)


def _load_data_pipeline(root: Path, eval_config: dict):
    add_data_pipeline_src(root)
    from manifest_index import ManifestIndex
    from transforms import PerWindowZScore

    dp_config_path = resolve_from_root(root, eval_config["input"]["data_pipeline_config_path"])
    assert dp_config_path is not None
    dp_config = load_yaml_config(dp_config_path)
    if eval_config["input"]["input_protocol_id"] != dp_config["input"]["input_protocol_id"]:
        raise RuntimeError("input_protocol_id mismatch.")
    if eval_config["input"]["protocol_id"] != dp_config["protocol"]["protocol_id"]:
        raise RuntimeError("protocol_id mismatch.")
    manifest_index = ManifestIndex(root, dp_config)
    transform = PerWindowZScore(
        eps=float(dp_config["normalization"]["epsilon"]),
        output_channel_first=bool(dp_config["normalization"]["output_channel_first"]),
    )
    return dp_config, manifest_index, transform


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
    encoder: PaPaGeiSFrozenAdapter,
    split: str,
    result_root: Path,
    device: torch.device,
    provenance: dict,
) -> tuple[list[dict], dict]:
    template_rows, trial_rows = _load_protocol_rows(root, dp_config, split)
    unique_indices = collect_unique_array_indices(template_rows, trial_rows)
    encoder_metadata = encoder.get_encoder_metadata()
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
        "checkpoint_sha256": encoder_metadata["checkpoint_sha256"],
        "input_protocol_id": eval_config["input"]["input_protocol_id"],
        "protocol_id": eval_config["input"]["protocol_id"],
        "normalization_policy": dp_config["normalization"]["policy"],
        "split": split,
        "scientific_reporting_allowed": True,
        "official_source_git_commit_sha": provenance.get("official_source_git_commit_sha"),
        "official_source_archive_sha256": provenance.get("official_source_archive_sha256"),
        "adapter_source_sha256": provenance["adapter_source_sha256"],
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
    if not all(np.isfinite(float(row["score"])) for row in scores):
        raise RuntimeError(f"Nonfinite scores detected for split={split}.")
    score_filename = "validation_scores.csv" if split == "val" else f"{split}_scores.csv"
    write_score_csv(result_root / score_filename, scores)
    return scores, {
        "split": split,
        "trial_count": len(trial_rows),
        "unique_window_count": len(unique_indices),
        "score_summary": numeric_summary(row["score"] for row in scores),
    }


def run_official_baseline(root: Path, config_path: str | Path | None = None, overwrite: bool = False) -> dict:
    """Run verified official PaPaGei-S frozen baseline."""

    eval_config = load_eval_config(root, config_path)
    readiness = check_official_ready(root)
    if not readiness["ready"]:
        raise RuntimeError(f"PaPaGei-S official baseline is not ready: {readiness['reason']}")
    dp_config, manifest_index, transform = _load_data_pipeline(root, eval_config)
    result_root = resolve_from_root(root, eval_config["output"]["result_root"])
    assert result_root is not None
    ensure_scientific_result_root_available(result_root, overwrite)
    result_root = prepare_result_root(result_root)
    device = _device_from_config(eval_config)
    encoder = PaPaGeiSFrozenAdapter(root, eval_config)
    encoder.to(device)
    encoder.eval()
    if count_trainable_parameters(encoder) != 0:
        raise RuntimeError("PaPaGei-S adapter is not frozen.")
    encoder_metadata = encoder.get_encoder_metadata()
    provenance = official_source_provenance(root)

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
            provenance=provenance,
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

    buckets = list(eval_config["metrics"]["stratified"]["time_gap_buckets"])
    gap_rows = time_gap_metrics(test_scores, float(eer_selection["threshold"]), buckets, "validation_eer_threshold")
    gap_rows += time_gap_metrics(test_scores, float(far_selection["threshold"]), buckets, "validation_far_1pct_threshold")
    subject_rows = subject_macro_metrics(test_scores, float(eer_selection["threshold"]), "validation_eer_threshold")
    subject_rows += subject_macro_metrics(test_scores, float(far_selection["threshold"]), "validation_far_1pct_threshold")
    session_rows = session_pair_macro_table(test_scores, float(eer_selection["threshold"]), "validation_eer_threshold")
    session_rows += session_pair_macro_table(test_scores, float(far_selection["threshold"]), "validation_far_1pct_threshold")
    write_csv_rows(result_root / "test_time_gap_metrics.csv", gap_rows)
    write_csv_rows(result_root / "test_subject_macro_metrics.csv", subject_rows)
    write_csv_rows(result_root / "test_session_pair_macro_metrics.csv", session_rows)

    run_manifest = {
        "evaluation_id": eval_config["evaluation_id"],
        "encoder_id": encoder.encoder_id,
        "encoder_variant": "PaPaGei-S",
        "encoder_architecture": "ResNet1DMoE",
        "encoder_mode": "frozen",
        "embedding_dim": 512,
        "checkpoint_path": encoder_metadata["checkpoint_path"],
        "checkpoint_md5": encoder_metadata["checkpoint_md5"],
        "checkpoint_sha256": encoder_metadata["checkpoint_sha256"],
        **provenance,
        "pretrained_weights_verified": True,
        "official_source_verified": True,
        "trainable_parameter_count": 0,
        "scientific_reporting_allowed": True,
        "input_protocol_id": eval_config["input"]["input_protocol_id"],
        "protocol_id": eval_config["input"]["protocol_id"],
        "input_setting": "common_input",
        "normalization_policy": dp_config["normalization"]["policy"],
        "official_native_preprocessing_reapplied": False,
        "common_transform_source": "data_pipeline/SigD/PerWindowZScore",
        "native_input_supplementary_evaluation_pending": True,
        "validation_threshold_only": True,
        "impostor_evaluation_mode": eval_config.get("fairness", {}).get(
            "impostor_evaluation_mode",
            "sampled_later_session_only",
        ),
        "final_headline_reporting_status": eval_config.get("fairness", {}).get(
            "final_headline_reporting_status",
            "preliminary_until_exhaustive_eval_added",
        ),
        "sampled_baseline_result_reference": eval_config.get("fairness", {}).get(
            "sampled_baseline_result_reference",
            "",
        ),
        "val_trial_count": len(val_scores),
        "test_trial_count": len(test_scores),
        "split_summaries": split_summaries,
        "source_overlap_limitation": encoder_metadata["source_overlap_limitation"],
        "test_threshold_tuning_performed": False,
        "generated_datetime": utc_now_iso(),
    }
    write_run_manifest(result_root / "frozen_baseline_run_manifest.json", run_manifest)
    return run_manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=None)
    parser.add_argument("--config", default=None)
    parser.add_argument("--allow-skip", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = detect_project_root(args.root)
    config = load_eval_config(root, args.config)
    readiness = check_official_ready(root)
    if not readiness["ready"]:
        path = write_skipped_manifest(root, config, str(readiness["reason"]))
        print(f"official_baseline_skipped={path}")
        if args.allow_skip:
            return 0
        raise SystemExit(2)
    manifest = run_official_baseline(root, args.config, overwrite=args.overwrite)
    print(
        "official_baseline_completed=True "
        f"val={manifest['val_trial_count']} test={manifest['test_trial_count']} "
        f"result_root={resolve_from_root(root, config['output']['result_root'])}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
