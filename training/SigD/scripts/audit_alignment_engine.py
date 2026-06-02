#!/usr/bin/env python3
"""Audit E6 alignment decomposition without touching test data."""

from __future__ import annotations

import argparse
import copy
import sys
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))

from adaptation_evaluator import evaluate_validation_only  # noqa: E402
from backbone_feature_cache import CachedBackboneEmbeddingDataset, cached_backbone_collate_fn, ensure_backbone_cache, load_backbone_cache  # noqa: E402
from common import (  # noqa: E402
    DEFAULT_CS_CONFIG,
    DEFAULT_E6_A_CONFIG,
    DEFAULT_E6_B_CONFIG,
    DEFAULT_E6_BASE_CONFIG,
    add_data_pipeline_src,
    detect_project_root,
    load_training_config,
    load_yaml_config,
    resolve_from_root,
    set_random_seed,
    sha256_file,
    utc_now_iso,
    write_json,
)
from objective_registry import compute_total_objective, lambda_align_from_config  # noqa: E402
from papagei_projection_model import PaPaGeiProjectionModel  # noqa: E402
from positive_masks import build_positive_mask  # noqa: E402


def snapshot_existing_e4_e5_results(root: Path) -> dict[str, Any]:
    """Hash preserved E4/E5 final-result artifacts if present."""

    base = root / "training" / "SigD" / "metadata" / "final_e4_e5_exhaustive_results_seed42"
    files = sorted(path for path in base.rglob("*") if path.is_file()) if base.exists() else []
    return {
        "base_path": str(base.relative_to(root)),
        "exists": base.exists(),
        "file_count": len(files),
        "sha256_by_file": {str(path.relative_to(root)): sha256_file(path) for path in files},
    }


def compare_e6_configs(configs: dict[str, dict[str, Any]], e5_config: dict[str, Any]) -> dict[str, Any]:
    """Audit E6 fairness-critical config consistency."""

    allowed_prefixes = {
        "experiment_id",
        "experiment_stage",
        "training.positive_mask_mode",
        "loss_components.session_centroid_alignment_weight",
        "loss_components.session_centroid_alignment_weight_candidates",
        "output.result_root",
        "fairness",
    }
    flattened = {name: _flatten(config) for name, config in configs.items()}
    keys = sorted(set().union(*(payload.keys() for payload in flattened.values())))
    differences: dict[str, dict[str, Any]] = {}
    forbidden: list[str] = []
    for key in keys:
        values = {name: flattened[name].get(key) for name in sorted(configs)}
        if len({repr(value) for value in values.values()}) > 1:
            differences[key] = values
            if not any(key == prefix or key.startswith(f"{prefix}.") for prefix in allowed_prefixes):
                forbidden.append(key)

    base = configs["e6_base"]
    e6_a = configs["e6_a"]
    e6_b = configs["e6_b"]
    structural = {
        "all_e6_sampler_cross_session": all(
            config["training"]["sampler_mode"] == "same_subject_cross_session" for config in configs.values()
        ),
        "e6_base_and_e6_b_positive_mask_identical": base["training"]["positive_mask_mode"] == e6_b["training"]["positive_mask_mode"],
        "e5_and_e6_a_sampler_identical": e5_config["training"]["sampler_mode"] == e6_a["training"]["sampler_mode"],
        "e5_and_e6_a_positive_mask_identical": e5_config["training"]["positive_mask_mode"] == e6_a["training"]["positive_mask_mode"],
        "shared_backbone_cache_root_identical": len({config["input"]["backbone_cache_root"] for config in configs.values()}) == 1,
    }
    return {
        "passed": not forbidden and all(structural.values()),
        "differences": differences,
        "differences_outside_allowed_fields": forbidden,
        "allowed_difference_prefixes": sorted(allowed_prefixes),
        "structural_checks": structural,
    }


def _flatten(value: Any, prefix: str = "") -> dict[str, Any]:
    if isinstance(value, dict):
        result: dict[str, Any] = {}
        for key, item in value.items():
            child = f"{prefix}.{key}" if prefix else str(key)
            result.update(_flatten(item, child))
        return result
    return {prefix: value}


def build_cached_cross_session_batch(root: Path, config: dict[str, Any]) -> tuple[dict[str, Any], Any]:
    """Return one cached cross-session training batch and its train pool."""

    add_data_pipeline_src(root)
    from manifest_index import ManifestIndex
    from session_aware_batch_sampler import SessionAwareBatchSampler
    from train_subject_pool import TrainSubjectPool

    dp_path = resolve_from_root(root, config["input"]["common_data_pipeline_config"])
    assert dp_path is not None
    dp_config = load_yaml_config(dp_path)
    manifest_index = ManifestIndex(root, dp_config)
    train_pool = TrainSubjectPool(root, dp_config, manifest_index)
    dataset = CachedBackboneEmbeddingDataset(root, "train", manifest_index, index_mode="array_index")
    sampler = SessionAwareBatchSampler(
        train_pool,
        mode="same_subject_cross_session",
        seed=int(config["seed"]),
        subjects_per_batch=int(config["training"]["subjects_per_batch"]),
        sessions_per_subject=2,
        windows_per_session=2,
        num_batches_per_epoch=1,
    )
    sampler.set_epoch(0)
    loader = DataLoader(dataset, batch_sampler=sampler, collate_fn=cached_backbone_collate_fn)
    return next(iter(loader)), train_pool


def batch_structure(batch: dict[str, Any]) -> dict[str, Any]:
    """Summarize cross-session batch composition."""

    subjects = batch["subject_ids"]
    sessions = batch["session_ids"]
    per_subject: dict[str, dict[str, int]] = {}
    for subject_id, session_id in zip(subjects, sessions):
        per_subject.setdefault(subject_id, {}).setdefault(session_id, 0)
        per_subject[subject_id][session_id] += 1
    array_indices = [int(index) for index in batch["array_indices"].tolist()]
    return {
        "batch_size": len(subjects),
        "selected_subject_count": len(per_subject),
        "sessions_per_subject": {subject: len(counts) for subject, counts in per_subject.items()},
        "samples_per_subject_session": {subject: counts for subject, counts in per_subject.items()},
        "duplicate_array_index_count": len(array_indices) - len(set(array_indices)),
    }


def _parameter_snapshot(module: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {name: parameter.detach().cpu().clone() for name, parameter in module.named_parameters()}


def _max_parameter_delta(before: dict[str, torch.Tensor], module: torch.nn.Module) -> float:
    deltas = []
    for name, parameter in module.named_parameters():
        if name in before:
            deltas.append(float((before[name] - parameter.detach().cpu()).abs().max().item()))
    return max(deltas) if deltas else 0.0


def audit_config_one_step(root: Path, config: dict[str, Any], batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    """Run one cached optimizer step for one E6 config."""

    set_random_seed(int(config["seed"]))
    model = PaPaGeiProjectionModel(root, config).to(device)
    model.train(True)
    backbone_before = _parameter_snapshot(model.backbone)
    projection_before = _parameter_snapshot(model.projection_head)
    cached_before = batch["backbone_embeddings"].detach().clone()
    embeddings = model.project(batch["backbone_embeddings"].to(device))
    loss, diagnostics = compute_total_objective(embeddings, batch["subject_ids"], batch["session_ids"], config)
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=float(config["training"]["learning_rate"]),
        weight_decay=float(config["training"]["weight_decay"]),
    )
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
    positive_mask = build_positive_mask(config["training"]["positive_mask_mode"], batch["subject_ids"], batch["session_ids"])
    validation = evaluate_validation_only(root=root, train_config=config, model=model, device=device, max_trials=128)
    return {
        "experiment_id": config["experiment_id"],
        "lambda_align": lambda_align_from_config(config),
        "positive_mask_mode": config["training"]["positive_mask_mode"],
        "anchor_positive_counts": sorted(set(int(value) for value in positive_mask.sum(dim=1).tolist())),
        "positive_pair_count": int(positive_mask.sum().item()),
        "total_loss": diagnostics["total_loss"],
        "supcon_loss": diagnostics["supcon_loss"],
        "alignment_loss": diagnostics["alignment_loss"],
        "alignment_diagnostics": diagnostics["alignment_diagnostics"],
        "mean_centroid_cosine": diagnostics["mean_centroid_cosine"],
        "loss_finite": torch.isfinite(loss).item(),
        "projection_parameter_max_delta_after_step": _max_parameter_delta(projection_before, model.projection_head),
        "backbone_parameter_max_delta_after_step": _max_parameter_delta(backbone_before, model.backbone),
        "cached_backbone_embedding_max_delta_after_step": float((cached_before - batch["backbone_embeddings"]).abs().max().item()),
        "trainable_parameter_count": model.trainable_parameter_count(),
        "validation_only_smoke": {
            "trial_count": validation["split_summary"]["trial_count"],
            "validation_exhaustive_eer": validation["validation_exhaustive_eer"],
            "test_data_read": validation["test_data_read"],
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=None)
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = detect_project_root(args.root)
    requested = args.device
    device = torch.device("cuda" if requested == "auto" and torch.cuda.is_available() else ("cpu" if requested == "auto" else requested))
    before = snapshot_existing_e4_e5_results(root)
    configs = {
        "e6_base": load_training_config(root, root / DEFAULT_E6_BASE_CONFIG),
        "e6_a": load_training_config(root, root / DEFAULT_E6_A_CONFIG),
        "e6_b": load_training_config(root, root / DEFAULT_E6_B_CONFIG),
    }
    e5 = load_training_config(root, root / DEFAULT_CS_CONFIG)
    for config in configs.values():
        ensure_backbone_cache(root=root, train_config=config, role="train", device=device)
        ensure_backbone_cache(root=root, train_config=config, role="validation_exhaustive", device=device)
    batch, train_pool = build_cached_cross_session_batch(root, configs["e6_base"])
    structure = batch_structure(batch)
    train_indices, train_embeddings, _ = load_backbone_cache(root, "train")
    train_subject_leak_count = sum(
        1
        for index in train_indices.tolist()
        if train_pool.manifest_index.get_metadata(int(index))["subject_id"] not in train_pool.train_subject_set
    )
    config_audit = compare_e6_configs(configs, e5)
    one_step = {name: audit_config_one_step(root, copy.deepcopy(config), batch, device) for name, config in configs.items()}
    after = snapshot_existing_e4_e5_results(root)
    errors: list[str] = []
    if before["sha256_by_file"] != after["sha256_by_file"]:
        errors.append("e4_e5_result_hash_changed")
    if not config_audit["passed"]:
        errors.append("e6_config_fairness_failed")
    if structure["batch_size"] != 32 or structure["selected_subject_count"] != 8:
        errors.append("cross_session_batch_shape_failed")
    if structure["duplicate_array_index_count"] != 0:
        errors.append("duplicate_array_index")
    if set(structure["sessions_per_subject"].values()) != {2}:
        errors.append("sessions_per_subject_failed")
    if any(set(counts.values()) != {2} for counts in structure["samples_per_subject_session"].values()):
        errors.append("samples_per_subject_session_failed")
    if train_subject_leak_count != 0:
        errors.append("train_cache_subject_leakage")
    expected_counts = {"e6_base": 3, "e6_a": 2, "e6_b": 3}
    expected_pairs = {"e6_base": 96, "e6_a": 64, "e6_b": 96}
    trainable_counts = {payload["trainable_parameter_count"] for payload in one_step.values()}
    if len(trainable_counts) != 1 or next(iter(trainable_counts)) != 164736:
        errors.append("trainable_parameter_count_mismatch")
    for name, payload in one_step.items():
        if payload["anchor_positive_counts"] != [expected_counts[name]]:
            errors.append(f"{name}_positive_count_mismatch")
        if payload["positive_pair_count"] != expected_pairs[name]:
            errors.append(f"{name}_positive_pair_count_mismatch")
        if not payload["loss_finite"]:
            errors.append(f"{name}_loss_nonfinite")
        if payload["projection_parameter_max_delta_after_step"] <= 0.0:
            errors.append(f"{name}_projection_not_updated")
        if payload["backbone_parameter_max_delta_after_step"] != 0.0:
            errors.append(f"{name}_backbone_changed")
        if payload["cached_backbone_embedding_max_delta_after_step"] != 0.0:
            errors.append(f"{name}_cached_embedding_changed")
        if payload["validation_only_smoke"]["test_data_read"]:
            errors.append(f"{name}_read_test_data")
    if abs(one_step["e6_base"]["total_loss"] - one_step["e6_base"]["supcon_loss"]) > 1.0e-8:
        errors.append("e6_base_total_not_equal_supcon")
    for name in ("e6_a", "e6_b"):
        payload = one_step[name]
        if not payload["alignment_diagnostics"]["centroid_pair_count"] == 8:
            errors.append(f"{name}_centroid_pair_count_mismatch")
        if payload["alignment_loss"] < 0:
            errors.append(f"{name}_alignment_loss_invalid")

    summary = {
        "generated_datetime_utc": utc_now_iso(),
        "device": str(device),
        "e4_e5_result_preservation": {
            "before": before,
            "after": after,
            "unchanged": before["sha256_by_file"] == after["sha256_by_file"],
        },
        "config_fairness_audit": config_audit,
        "cross_session_batch_structure": structure,
        "train_cache": {
            "array_index_count": int(len(train_indices)),
            "embedding_shape": [int(x) for x in train_embeddings.shape],
            "train_subject_leak_count": int(train_subject_leak_count),
        },
        "one_step_audit": one_step,
        "test_accessed_during_audit": False,
        "passed": len(errors) == 0,
        "errors": errors,
    }
    output = root / "training" / "SigD" / "metadata" / "alignment_engine_audit_summary.json"
    write_json(output, summary)
    print(
        f"alignment_engine_audit_passed={summary['passed']} "
        f"e6_base_pairs={one_step['e6_base']['positive_pair_count']} "
        f"e6_a_pairs={one_step['e6_a']['positive_pair_count']} "
        f"e6_b_pairs={one_step['e6_b']['positive_pair_count']}"
    )
    if errors:
        raise SystemExit(1)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

