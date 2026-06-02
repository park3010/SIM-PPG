#!/usr/bin/env python3
"""Audit E7 sVRI/SQI morphology preservation without test access."""

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
from backbone_feature_cache import CachedBackboneEmbeddingDataset, cached_backbone_collate_fn, ensure_backbone_cache  # noqa: E402
from common import (  # noqa: E402
    DEFAULT_E6_BASE_CONFIG,
    DEFAULT_E7_A_CONFIG,
    DEFAULT_E7_B_CONFIG,
    DEFAULT_GENERIC_CONFIG,
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
from morphology_objective import compute_total_loss_with_morphology  # noqa: E402
from papagei_projection_model import PaPaGeiProjectionModel  # noqa: E402
from positive_masks import build_positive_mask  # noqa: E402


PRESERVED_RESULT_DIRS = [
    "evaluation/SigD/metadata/final_papagei_s_frozen_exhaustive_baseline_snapshot_v2",
    "training/SigD/metadata/final_e4_e5_exhaustive_results_seed42",
    "training/SigD/metadata/e6_validation_only_selection_snapshot_seed42",
    "training/SigD/metadata/final_e6_validation_and_test_snapshot_seed42",
    "training/SigD/results/papagei_s_e6_base_generic_cs_batch_noalign",
    "training/SigD/results/papagei_s_e6_a_cs_supcon_alignment",
    "training/SigD/results/papagei_s_e6_b_generic_supcon_alignment_cs_batch",
]


def snapshot_preserved_results(root: Path) -> dict[str, Any]:
    """Hash existing frozen/E4/E5/E6 result artifacts."""

    payload: dict[str, Any] = {}
    for rel in PRESERVED_RESULT_DIRS:
        base = root / rel
        files = sorted(path for path in base.rglob("*") if path.is_file()) if base.exists() else []
        payload[rel] = {
            "exists": base.exists(),
            "file_count": len(files),
            "sha256_by_file": {str(path.relative_to(root)): sha256_file(path) for path in files},
        }
    return payload


def _flatten(value: Any, prefix: str = "") -> dict[str, Any]:
    if isinstance(value, dict):
        result: dict[str, Any] = {}
        for key, item in value.items():
            child = f"{prefix}.{key}" if prefix else str(key)
            result.update(_flatten(item, child))
        return result
    return {prefix: value}


def config_fairness_audit(e4: dict[str, Any], e6_base: dict[str, Any], e7_a: dict[str, Any], e7_b: dict[str, Any]) -> dict[str, Any]:
    """Check E7 branches differ from their bases only by morphology-related settings."""

    checks = {
        "e7_a_sampler_matches_e4": e7_a["training"]["sampler_mode"] == e4["training"]["sampler_mode"],
        "e7_a_positive_mask_matches_e4": e7_a["training"]["positive_mask_mode"] == e4["training"]["positive_mask_mode"],
        "e7_b_sampler_matches_e6_base": e7_b["training"]["sampler_mode"] == e6_base["training"]["sampler_mode"],
        "e7_b_positive_mask_matches_e6_base": e7_b["training"]["positive_mask_mode"] == e6_base["training"]["positive_mask_mode"],
        "projection_architecture_shared": e7_a["model"]["projection_head"] == e7_b["model"]["projection_head"] == e4["model"]["projection_head"],
        "optimizer_shared": e7_a["training"]["optimizer"] == e7_b["training"]["optimizer"] == e4["training"]["optimizer"],
        "learning_rate_shared": e7_a["training"]["learning_rate"] == e7_b["training"]["learning_rate"] == e4["training"]["learning_rate"],
        "final_protocol_shared": e7_a["input"]["final_protocol_id"] == e7_b["input"]["final_protocol_id"] == e4["input"]["final_protocol_id"],
        "morphology_targets_svri_sqi_only": e7_a["model"]["morphology_heads"]["targets"] == ["svri", "sqi"]
        and e7_b["model"]["morphology_heads"]["targets"] == ["svri", "sqi"],
        "ipa_disabled": not e7_a["loss_components"]["use_ipa"] and not e7_b["loss_components"]["use_ipa"],
        "sqi_weighting_disabled": not e7_a["loss_components"]["sqi_weighting_enabled"]
        and not e7_b["loss_components"]["sqi_weighting_enabled"],
    }
    e7_diffs = {
        key: values
        for key, values in _diff_configs(e7_a, e7_b).items()
        if key
        not in {
            "experiment_id",
            "experiment_stage",
            "training.sampler_mode",
            "training.sessions_per_subject",
            "training.windows_per_session",
            "output.result_root",
            "fairness.base_branch",
        }
    }
    return {"passed": all(checks.values()) and not e7_diffs, "checks": checks, "unexpected_e7_a_b_differences": e7_diffs}


def _diff_configs(a: dict[str, Any], b: dict[str, Any]) -> dict[str, dict[str, Any]]:
    fa = _flatten(a)
    fb = _flatten(b)
    diff: dict[str, dict[str, Any]] = {}
    for key in sorted(set(fa) | set(fb)):
        if fa.get(key) != fb.get(key):
            diff[key] = {"a": fa.get(key), "b": fb.get(key)}
    return diff


def build_cached_batch(root: Path, config: dict[str, Any]) -> tuple[dict[str, Any], Any]:
    """Build one cached training batch according to config sampler."""

    add_data_pipeline_src(root)
    from manifest_index import ManifestIndex
    from session_aware_batch_sampler import SessionAwareBatchSampler
    from train_subject_pool import TrainSubjectPool

    dp_config = load_yaml_config(resolve_from_root(root, config["input"]["common_data_pipeline_config"]))
    manifest_index = ManifestIndex(root, dp_config)
    train_pool = TrainSubjectPool(root, dp_config, manifest_index)
    dataset = CachedBackboneEmbeddingDataset(root, "train", manifest_index, index_mode="array_index")
    sampler = SessionAwareBatchSampler(
        train_pool,
        mode=config["training"]["sampler_mode"],
        seed=int(config["seed"]),
        subjects_per_batch=int(config["training"]["subjects_per_batch"]),
        sessions_per_subject=int(config["training"].get("sessions_per_subject", 2)),
        windows_per_session=int(config["training"].get("windows_per_session", 2)),
        num_batches_per_epoch=1,
    )
    sampler.set_epoch(0)
    return next(iter(DataLoader(dataset, batch_sampler=sampler, collate_fn=cached_backbone_collate_fn))), train_pool


def _parameter_snapshot(module: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {name: parameter.detach().cpu().clone() for name, parameter in module.named_parameters()}


def _max_delta(before: dict[str, torch.Tensor], module: torch.nn.Module) -> float:
    values = []
    for name, parameter in module.named_parameters():
        if name in before:
            values.append(float((before[name] - parameter.detach().cpu()).abs().max().item()))
    return max(values) if values else 0.0


def audit_one_branch(root: Path, config: dict[str, Any], device: torch.device) -> dict[str, Any]:
    """Run one morphology one-step audit."""

    set_random_seed(int(config["seed"]))
    batch, _ = build_cached_batch(root, config)
    model = PaPaGeiProjectionModel(root, config).to(device)
    model.train(True)
    projection_before = _parameter_snapshot(model.projection_head)
    morphology_before = _parameter_snapshot(model.morphology_heads)
    backbone_before = _parameter_snapshot(model.backbone)
    cached_before = batch["backbone_embeddings"].detach().clone()
    embeddings = model.project(batch["backbone_embeddings"].to(device))
    predictions = model.predict_morphology(embeddings)
    loss, diagnostics = compute_total_loss_with_morphology(
        embeddings,
        batch["subject_ids"],
        batch["session_ids"],
        predictions,
        batch,
        config,
    )
    zero_config = copy.deepcopy(config)
    zero_config["loss_components"]["lambda_svri"] = 0.0
    zero_config["loss_components"]["lambda_sqi"] = 0.0
    zero_loss, zero_diag = compute_total_loss_with_morphology(
        embeddings,
        batch["subject_ids"],
        batch["session_ids"],
        predictions,
        batch,
        zero_config,
    )
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
        "sampler_mode": config["training"]["sampler_mode"],
        "positive_mask_mode": config["training"]["positive_mask_mode"],
        "anchor_positive_counts": sorted(set(int(value) for value in positive_mask.sum(dim=1).tolist())),
        "positive_pair_count": int(positive_mask.sum().item()),
        "batch_contains_morphology": all(key in batch for key in ("svri", "sqi", "svri_valid_mask", "sqi_valid_mask", "ipa", "ipa_valid_mask")),
        "svri_valid_count": diagnostics["svri_valid_count"],
        "sqi_valid_count": diagnostics["sqi_valid_count"],
        "ipa_present_but_unused": "ipa" in batch and diagnostics["ipa_used"] is False,
        "loss_diagnostics": diagnostics,
        "zero_lambda_total_equals_supcon": abs(float(zero_loss.detach().cpu()) - float(zero_diag["supcon_loss"])) < 1.0e-6,
        "projection_parameter_max_delta_after_step": _max_delta(projection_before, model.projection_head),
        "morphology_head_parameter_max_delta_after_step": _max_delta(morphology_before, model.morphology_heads),
        "backbone_parameter_max_delta_after_step": _max_delta(backbone_before, model.backbone),
        "cached_backbone_embedding_max_delta_after_step": float((cached_before - batch["backbone_embeddings"]).abs().max().item()),
        "trainable_parameter_count": model.trainable_parameter_count(),
        "morphology_heads_parameter_count": model.morphology_heads.parameter_count(),
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
    before = snapshot_preserved_results(root)
    e4 = load_training_config(root, root / DEFAULT_GENERIC_CONFIG)
    e6_base = load_training_config(root, root / DEFAULT_E6_BASE_CONFIG)
    e7_a = load_training_config(root, root / DEFAULT_E7_A_CONFIG)
    e7_b = load_training_config(root, root / DEFAULT_E7_B_CONFIG)
    for config in (e7_a, e7_b):
        ensure_backbone_cache(root=root, train_config=config, role="train", device=device)
        ensure_backbone_cache(root=root, train_config=config, role="validation_exhaustive", device=device)
    config_audit = config_fairness_audit(e4, e6_base, e7_a, e7_b)
    branches = {
        "e7_a": audit_one_branch(root, e7_a, device),
        "e7_b": audit_one_branch(root, e7_b, device),
    }
    after = snapshot_preserved_results(root)
    errors: list[str] = []
    if before != after:
        errors.append("preserved_result_hash_changed")
    if not config_audit["passed"]:
        errors.append("config_fairness_failed")
    expected_positive = {"e7_a": 3, "e7_b": 3}
    for name, payload in branches.items():
        if payload["anchor_positive_counts"] != [expected_positive[name]]:
            errors.append(f"{name}_positive_count_mismatch")
        if payload["positive_pair_count"] != 96:
            errors.append(f"{name}_positive_pair_count_mismatch")
        if not payload["batch_contains_morphology"]:
            errors.append(f"{name}_missing_morphology_fields")
        if payload["svri_valid_count"] <= 0 or payload["sqi_valid_count"] <= 0:
            errors.append(f"{name}_morphology_valid_count_zero")
        if not payload["ipa_present_but_unused"]:
            errors.append(f"{name}_ipa_usage_policy_failed")
        if not payload["zero_lambda_total_equals_supcon"]:
            errors.append(f"{name}_zero_lambda_supcon_mismatch")
        if payload["projection_parameter_max_delta_after_step"] <= 0:
            errors.append(f"{name}_projection_not_updated")
        if payload["morphology_head_parameter_max_delta_after_step"] <= 0:
            errors.append(f"{name}_morphology_heads_not_updated")
        if payload["backbone_parameter_max_delta_after_step"] != 0.0:
            errors.append(f"{name}_backbone_changed")
        if payload["cached_backbone_embedding_max_delta_after_step"] != 0.0:
            errors.append(f"{name}_cached_embedding_changed")
        if payload["trainable_parameter_count"] <= 164736:
            errors.append(f"{name}_trainable_parameter_count_not_increased")
        if payload["validation_only_smoke"]["test_data_read"]:
            errors.append(f"{name}_read_test_data")
    summary = {
        "generated_datetime_utc": utc_now_iso(),
        "device": str(device),
        "preserved_results": {"before": before, "after": after, "unchanged": before == after},
        "config_fairness_audit": config_audit,
        "branches": branches,
        "test_accessed_during_audit": False,
        "passed": len(errors) == 0,
        "errors": errors,
    }
    write_json(root / "training" / "SigD" / "metadata" / "morphology_engine_audit_summary.json", summary)
    print(
        f"morphology_engine_audit_passed={summary['passed']} "
        f"e7_a_svri_valid={branches['e7_a']['svri_valid_count']} "
        f"e7_b_svri_valid={branches['e7_b']['svri_valid_count']}"
    )
    if errors:
        raise SystemExit(1)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
