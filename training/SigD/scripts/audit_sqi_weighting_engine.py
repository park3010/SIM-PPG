#!/usr/bin/env python3
"""Audit E8 SQI-weighted SupCon without test access."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))

from adaptation_evaluator import evaluate_validation_only  # noqa: E402
from backbone_feature_cache import CachedBackboneEmbeddingDataset, cached_backbone_collate_fn, ensure_backbone_cache  # noqa: E402
from common import DEFAULT_E8_CONFIG, add_data_pipeline_src, detect_project_root, load_training_config, load_yaml_config, resolve_from_root, sha256_file, utc_now_iso, write_json  # noqa: E402
from morphology_objective import compute_total_loss_with_morphology  # noqa: E402
from papagei_projection_model import PaPaGeiProjectionModel  # noqa: E402
from positive_masks import build_positive_mask  # noqa: E402
from sqi_weighting import compute_sqi_weights  # noqa: E402
from supervised_contrastive_loss import supervised_contrastive_loss  # noqa: E402
from weighted_supcon_loss import weighted_supervised_contrastive_loss  # noqa: E402


PRESERVED_RESULT_DIRS = [
    "training/SigD/metadata/final_e7_a_morphology_exhaustive_snapshot_seed42",
    "training/SigD/results/papagei_s_e7_a_generic_supcon_morph_e4_branch",
    "training/SigD/results/papagei_s_e7_b_generic_supcon_morph_cs_batch_branch",
]


def snapshot_preserved_results(root: Path) -> dict[str, Any]:
    """Hash existing E7 artifacts if present."""

    output: dict[str, Any] = {}
    for rel in PRESERVED_RESULT_DIRS:
        base = root / rel
        files = sorted(path for path in base.rglob("*") if path.is_file()) if base.exists() else []
        output[rel] = {
            "exists": base.exists(),
            "file_count": len(files),
            "sha256_by_file": {str(path.relative_to(root)): sha256_file(path) for path in files},
        }
    return output


def build_cached_batch(root: Path, config: dict[str, Any]) -> dict[str, Any]:
    """Build one E8 cached training batch."""

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
        sessions_per_subject=2,
        windows_per_session=2,
        num_batches_per_epoch=1,
    )
    sampler.set_epoch(0)
    return next(iter(DataLoader(dataset, batch_sampler=sampler, collate_fn=cached_backbone_collate_fn)))


def _parameter_snapshot(module: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {name: parameter.detach().cpu().clone() for name, parameter in module.named_parameters()}


def _max_delta(before: dict[str, torch.Tensor], module: torch.nn.Module) -> float:
    values = []
    for name, parameter in module.named_parameters():
        if name in before:
            values.append(float((before[name] - parameter.detach().cpu()).abs().max().item()))
    return max(values) if values else 0.0


def audit_sqi_weights(batch: dict[str, Any]) -> dict[str, Any]:
    """Exercise SQI weighting modes."""

    sqi = batch["sqi"]
    mask = batch["sqi_valid_mask"]
    modes = ["mild_linear", "clipped_linear", "strong_linear", "rank_bottom20_downweight"]
    mode_payload = {}
    for mode in modes:
        weights, diagnostics = compute_sqi_weights(sqi, mask, mode)
        mode_payload[mode] = {
            **diagnostics,
            "finite": bool(torch.isfinite(weights).all().item()),
            "invalid_weight_neutral": bool((weights[~mask] == 1.0).all().item()) if (~mask).any() else True,
        }
    all_invalid, all_invalid_diag = compute_sqi_weights(sqi, torch.zeros_like(mask, dtype=torch.bool), "mild_linear")
    return {
        "modes": mode_payload,
        "all_invalid_returns_ones": bool(torch.allclose(all_invalid, torch.ones_like(all_invalid))),
        "all_invalid_diagnostics": all_invalid_diag,
    }


def audit_weighted_supcon(batch: dict[str, Any], config: dict[str, Any], embeddings: torch.Tensor) -> dict[str, Any]:
    """Check weighted loss against unweighted controls."""

    mask = build_positive_mask(config["training"]["positive_mask_mode"], batch["subject_ids"], batch["session_ids"]).to(embeddings.device)
    ones = torch.ones(embeddings.shape[0], device=embeddings.device)
    weighted_ones, diag_ones = weighted_supervised_contrastive_loss(
        embeddings,
        mask,
        ones,
        float(config["training"]["temperature"]),
        return_diagnostics=True,
    )
    unweighted = supervised_contrastive_loss(embeddings, mask, float(config["training"]["temperature"]))
    sqi_weights, sqi_diag = compute_sqi_weights(
        batch["sqi"].to(embeddings.device),
        batch["sqi_valid_mask"].to(embeddings.device),
        str(config["loss_components"]["sqi_weighting_mode"]),
    )
    weighted, weighted_diag = weighted_supervised_contrastive_loss(
        embeddings,
        mask,
        sqi_weights,
        float(config["training"]["temperature"]),
        return_diagnostics=True,
    )
    return {
        "ones_equals_unweighted": abs(float(weighted_ones.detach().cpu()) - float(unweighted.detach().cpu())) < 1.0e-6,
        "nonuniform_weighted_loss": float(weighted.detach().cpu()),
        "unweighted_loss": float(unweighted.detach().cpu()),
        "loss_changed_with_nonuniform_weights": abs(float(weighted.detach().cpu()) - float(unweighted.detach().cpu())) > 1.0e-8,
        "ones_diagnostics": diag_ones,
        "sqi_weight_diagnostics": sqi_diag,
        "weighted_diagnostics": weighted_diag,
    }


def audit_training_step(root: Path, config: dict[str, Any], batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    """Run one E8 optimizer step."""

    model = PaPaGeiProjectionModel(root, config).to(device)
    model.train(True)
    before_projection = _parameter_snapshot(model.projection_head)
    before_morphology = _parameter_snapshot(model.morphology_heads)
    before_backbone = _parameter_snapshot(model.backbone)
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
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=float(config["training"]["learning_rate"]),
        weight_decay=float(config["training"]["weight_decay"]),
    )
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
    validation = evaluate_validation_only(root=root, train_config=config, model=model, device=device, max_trials=128)
    return {
        "loss_finite": bool(torch.isfinite(loss).item()),
        "loss_diagnostics": diagnostics,
        "projection_parameter_max_delta_after_step": _max_delta(before_projection, model.projection_head),
        "morphology_head_parameter_max_delta_after_step": _max_delta(before_morphology, model.morphology_heads),
        "backbone_parameter_max_delta_after_step": _max_delta(before_backbone, model.backbone),
        "cached_backbone_embedding_max_delta_after_step": float((cached_before - batch["backbone_embeddings"]).abs().max().item()),
        "verification_uses_projection_only": True,
        "sqi_weighting_used_for_verification": False,
        "morphology_used_for_verification": False,
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
    config = load_training_config(root, root / DEFAULT_E8_CONFIG)
    ensure_backbone_cache(root=root, train_config=config, role="train", device=device)
    ensure_backbone_cache(root=root, train_config=config, role="validation_exhaustive", device=device)
    batch = build_cached_batch(root, config)
    model_for_loss = PaPaGeiProjectionModel(root, config).to(device)
    with torch.no_grad():
        embeddings = model_for_loss.project(batch["backbone_embeddings"].to(device))
    weights_audit = audit_sqi_weights(batch)
    weighted_audit = audit_weighted_supcon(batch, config, embeddings)
    step_audit = audit_training_step(root, config, batch, device)
    after = snapshot_preserved_results(root)
    errors: list[str] = []
    if before != after:
        errors.append("preserved_result_hash_changed")
    for mode, payload in weights_audit["modes"].items():
        if not payload["finite"]:
            errors.append(f"{mode}_weights_nonfinite")
        if not payload["invalid_weight_neutral"]:
            errors.append(f"{mode}_invalid_not_neutral")
    if not weights_audit["all_invalid_returns_ones"]:
        errors.append("all_invalid_not_ones")
    if not weighted_audit["ones_equals_unweighted"]:
        errors.append("ones_weight_not_unweighted")
    if not step_audit["loss_finite"]:
        errors.append("e8_loss_nonfinite")
    if step_audit["projection_parameter_max_delta_after_step"] <= 0:
        errors.append("projection_not_updated")
    if step_audit["morphology_head_parameter_max_delta_after_step"] <= 0:
        errors.append("morphology_not_updated")
    if step_audit["backbone_parameter_max_delta_after_step"] != 0.0:
        errors.append("backbone_changed")
    if step_audit["cached_backbone_embedding_max_delta_after_step"] != 0.0:
        errors.append("cached_embedding_changed")
    if step_audit["validation_only_smoke"]["test_data_read"]:
        errors.append("test_data_read")
    summary = {
        "generated_datetime_utc": utc_now_iso(),
        "preserved_results": {"before": before, "after": after, "unchanged": before == after},
        "sqi_weighting": weights_audit,
        "weighted_supcon": weighted_audit,
        "training_step": step_audit,
        "test_accessed_during_audit": False,
        "passed": len(errors) == 0,
        "errors": errors,
    }
    write_json(root / "training" / "SigD" / "metadata" / "sqi_weighting_engine_audit_summary.json", summary)
    print(
        f"sqi_weighting_engine_audit_passed={summary['passed']} "
        f"mode={config['loss_components']['sqi_weighting_mode']} "
        f"weighted_loss={step_audit['loss_diagnostics']['weighted_supcon_loss']:.6f}"
    )
    if errors:
        raise SystemExit(1)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

