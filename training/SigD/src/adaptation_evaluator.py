"""Validation/test evaluator for trainable projection adaptation models."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch

from backbone_feature_cache import ensure_backbone_cache, load_backbone_cache, role_paths
from common import (
    add_data_pipeline_src,
    add_evaluation_src,
    ensure_dir,
    load_csv_rows,
    load_yaml_config,
    numeric_summary,
    resolve_from_root,
    utc_now_iso,
    write_csv_rows,
    write_json,
)


def _load_data_pipeline(root: Path, train_config: dict[str, Any], *, final: bool = True):
    add_data_pipeline_src(root)
    from manifest_index import ManifestIndex
    from transforms import PerWindowZScore

    config_key = "final_evaluation_data_pipeline_config" if final else "common_data_pipeline_config"
    dp_config_path = resolve_from_root(root, train_config["input"][config_key])
    if dp_config_path is None:
        raise RuntimeError("Data pipeline config path missing.")
    dp_config = load_yaml_config(dp_config_path)
    manifest_index = ManifestIndex(root, dp_config)
    transform = PerWindowZScore(
        eps=float(dp_config["normalization"]["epsilon"]),
        output_channel_first=bool(dp_config["normalization"]["output_channel_first"]),
    )
    return dp_config, manifest_index, transform


def _load_protocol_rows(root: Path, dp_config: dict[str, Any], split: str) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    template_rows = load_csv_rows(resolve_from_root(root, dp_config["protocol"]["enrollment_templates_path"]))
    split_paths = dp_config["protocol"].get("verification_trial_paths_by_split")
    if split_paths:
        if split not in split_paths:
            raise RuntimeError(f"No trial file configured for split={split}.")
        trial_rows = load_csv_rows(resolve_from_root(root, split_paths[split]))
    else:
        all_trials = load_csv_rows(resolve_from_root(root, dp_config["protocol"]["verification_trials_path"]))
        trial_rows = [row for row in all_trials if row["split"] == split]
    return template_rows, [row for row in trial_rows if row["split"] == split]


def balanced_trial_subset(rows: list[dict[str, str]], max_trials: int | None) -> list[dict[str, str]]:
    """Return a deterministic balanced-ish subset for smoke validation."""

    if max_trials is None or max_trials >= len(rows):
        return rows
    half = max(1, int(max_trials) // 2)
    genuine = [row for row in rows if str(row["label"]) == "1"][:half]
    impostor = [row for row in rows if str(row["label"]) == "0"][: max(1, int(max_trials) - len(genuine))]
    return genuine + impostor


def collect_unique_array_indices(template_rows: list[dict[str, str]], trial_rows: list[dict[str, str]]) -> list[int]:
    """Collect enrollment and probe array indices needed by trial rows."""

    import json

    template_ids = {row["template_id"] for row in trial_rows}
    indices: set[int] = set()
    for row in template_rows:
        if row["template_id"] in template_ids:
            indices.update(int(item) for item in json.loads(row["enrollment_window_indices"]))
    for row in trial_rows:
        indices.add(int(float(row["probe_window_index"])))
    return sorted(indices)


def compute_projection_cache(
    *,
    model: torch.nn.Module,
    manifest_index: Any,
    transform: Any,
    array_indices: list[int],
    batch_size: int,
    device: torch.device,
) -> dict[int, np.ndarray]:
    """Compute array_index -> 128-d projection embedding cache."""

    model.eval()
    model.to(device)
    cache: dict[int, np.ndarray] = {}
    with torch.inference_mode():
        for start in range(0, len(array_indices), batch_size):
            batch_indices = array_indices[start : start + batch_size]
            waveforms = torch.stack([transform(manifest_index.get_waveform(index)) for index in batch_indices], dim=0).to(device)
            embeddings = model.encode(waveforms).detach().cpu().numpy().astype(np.float32)
            for index, embedding in zip(batch_indices, embeddings):
                cache[int(index)] = embedding
    return cache


def compute_projection_cache_from_backbone_cache(
    *,
    root: Path,
    role: str,
    model: torch.nn.Module,
    array_indices: list[int],
    batch_size: int,
    device: torch.device,
) -> dict[int, np.ndarray]:
    """Compute projection embeddings from cached 512-d frozen backbone embeddings."""

    cached_indices, cached_embeddings, _ = load_backbone_cache(root, role)
    position = {int(index): pos for pos, index in enumerate(cached_indices.tolist())}
    missing = [index for index in array_indices if index not in position]
    if missing:
        raise RuntimeError(f"Backbone cache missing requested indices for {role}: {missing[:10]}")
    model.eval()
    model.to(device)
    cache: dict[int, np.ndarray] = {}
    with torch.inference_mode():
        for start in range(0, len(array_indices), batch_size):
            batch_indices = array_indices[start : start + batch_size]
            backbone = torch.from_numpy(
                np.stack([cached_embeddings[position[index]] for index in batch_indices], axis=0).astype(np.float32)
            ).to(device)
            projected = model.project(backbone).detach().cpu().numpy().astype(np.float32)
            for index, embedding in zip(batch_indices, projected):
                cache[int(index)] = embedding
    return cache


def score_split(
    *,
    root: Path,
    train_config: dict[str, Any],
    model: torch.nn.Module,
    split: str,
    device: torch.device,
    max_trials: int | None = None,
    allow_cache_build: bool = True,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Score one split using projection embeddings."""

    add_evaluation_src(root)
    from cosine_verifier import score_trials

    dp_config, manifest_index, transform = _load_data_pipeline(root, train_config, final=True)
    template_rows, trial_rows = _load_protocol_rows(root, dp_config, split)
    trial_rows = balanced_trial_subset(trial_rows, max_trials)
    unique_indices = collect_unique_array_indices(template_rows, trial_rows)
    role = "validation_exhaustive" if split == "val" else "test_exhaustive"
    cache_path, cache_manifest_path = role_paths(root, role)
    use_cache = max_trials is None and (cache_path.exists() and cache_manifest_path.exists() or allow_cache_build)
    if use_cache:
        if allow_cache_build:
            ensure_backbone_cache(root=root, train_config=train_config, role=role, overwrite=False, device=device)
        embeddings = compute_projection_cache_from_backbone_cache(
            root=root,
            role=role,
            model=model,
            array_indices=unique_indices,
            batch_size=128,
            device=device,
        )
        cache_source = role
    else:
        embeddings = compute_projection_cache(
            model=model,
            manifest_index=manifest_index,
            transform=transform,
            array_indices=unique_indices,
            batch_size=128,
            device=device,
        )
        cache_source = "direct_waveform_forward"
    scores = score_trials(
        trial_rows=trial_rows,
        template_rows=template_rows,
        embedding_cache=embeddings,
        encoder_id=f"{train_config['experiment_id']}_projection_head",
        eps=1.0e-8,
    )
    return scores, {
        "split": split,
        "trial_count": len(trial_rows),
        "unique_window_count": len(unique_indices),
        "backbone_cache_source": cache_source,
        "score_summary": numeric_summary(row["score"] for row in scores),
    }


def evaluate_validation_only(
    *,
    root: Path,
    train_config: dict[str, Any],
    model: torch.nn.Module,
    device: torch.device,
    max_trials: int | None = None,
) -> dict[str, Any]:
    """Evaluate validation split only for checkpoint selection."""

    add_evaluation_src(root)
    from metrics import compute_split_metrics
    from thresholds import compute_far_target_threshold

    scores, summary = score_split(
        root=root,
        train_config=train_config,
        model=model,
        split="val",
        device=device,
        max_trials=max_trials,
        allow_cache_build=max_trials is None,
    )
    labels = [int(row["label"]) for row in scores]
    score_values = [float(row["score"]) for row in scores]
    metrics = compute_split_metrics(split="val", labels=labels, scores=score_values)
    eer = metrics["diagnostic_eer"].get("eer")
    try:
        far_target = compute_far_target_threshold(labels, score_values, target_far=0.01)
        tar_at_far_1pct = far_target.get("validation_tar")
    except ValueError:
        far_target = {"unavailable_reason": "binary labels unavailable"}
        tar_at_far_1pct = None
    return {
        "split_summary": summary,
        "metrics": metrics,
        "validation_far_1pct_threshold": far_target,
        "validation_tar_at_far_1pct": tar_at_far_1pct,
        "validation_exhaustive_eer": eer,
        "test_data_read": False,
    }


def evaluate_final(
    *,
    root: Path,
    train_config: dict[str, Any],
    model: torch.nn.Module,
    device: torch.device,
    result_root: Path,
) -> dict[str, Any]:
    """Evaluate validation/test exhaustive protocol and write outputs."""

    add_evaluation_src(root)
    from metrics import compute_split_metrics
    from stratified_analysis import session_pair_macro_table, subject_macro_metrics, time_gap_metrics
    from thresholds import compute_eer_threshold, compute_far_target_threshold

    result_root = ensure_dir(result_root)
    val_scores, val_summary = score_split(root=root, train_config=train_config, model=model, split="val", device=device)
    test_scores, test_summary = score_split(root=root, train_config=train_config, model=model, split="test", device=device)
    val_labels = [int(row["label"]) for row in val_scores]
    val_values = [float(row["score"]) for row in val_scores]
    eer_selection = compute_eer_threshold(val_labels, val_values)
    far_selection = compute_far_target_threshold(val_labels, val_values, target_far=0.01)
    threshold_payload = {
        "source_split": "val",
        "validation_eer_threshold": eer_selection,
        "validation_far_1pct_threshold": far_selection,
        "test_threshold_tuning_performed": False,
    }
    val_metrics = compute_split_metrics(
        split="val",
        labels=val_labels,
        scores=val_values,
        validation_eer_threshold=float(eer_selection["threshold"]),
        validation_far_threshold=float(far_selection["threshold"]),
    )
    test_labels = [int(row["label"]) for row in test_scores]
    test_values = [float(row["score"]) for row in test_scores]
    test_metrics = compute_split_metrics(
        split="test",
        labels=test_labels,
        scores=test_values,
        validation_eer_threshold=float(eer_selection["threshold"]),
        validation_far_threshold=float(far_selection["threshold"]),
    )
    buckets = ["le_30d", "31_180d", "181_365d", "gt_365d"]
    write_csv_rows(result_root / "validation_scores.csv", val_scores)
    write_csv_rows(result_root / "test_scores.csv", test_scores)
    write_json(result_root / "threshold_selection_from_validation.json", threshold_payload)
    write_json(result_root / "validation_metrics.json", val_metrics)
    write_json(result_root / "test_metrics.json", test_metrics)
    write_csv_rows(
        result_root / "test_time_gap_metrics.csv",
        time_gap_metrics(test_scores, float(eer_selection["threshold"]), buckets, "validation_eer_threshold")
        + time_gap_metrics(test_scores, float(far_selection["threshold"]), buckets, "validation_far_1pct_threshold"),
    )
    write_csv_rows(result_root / "test_subject_macro_metrics.csv", subject_macro_metrics(test_scores, float(eer_selection["threshold"])))
    write_csv_rows(result_root / "test_session_pair_macro_metrics.csv", session_pair_macro_table(test_scores, float(eer_selection["threshold"])))
    return {
        "generated_datetime_utc": utc_now_iso(),
        "val_summary": val_summary,
        "test_summary": test_summary,
        "threshold_selection": threshold_payload,
        "validation_metrics": val_metrics,
        "test_metrics": test_metrics,
        "test_threshold_tuning_performed": False,
    }
