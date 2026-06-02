"""SupCon plus sVRI/SQI morphology preservation objective for E7."""

from __future__ import annotations

from typing import Any, Sequence

import torch

from morphology_losses import compute_morphology_losses
from positive_masks import build_positive_mask
from sqi_weighting import compute_sqi_weights
from supervised_contrastive_loss import supervised_contrastive_loss
from weighted_supcon_loss import weighted_supervised_contrastive_loss


def compute_total_loss_with_morphology(
    embeddings: torch.Tensor,
    subject_ids: Sequence[str],
    session_ids: Sequence[str],
    morphology_predictions: dict[str, torch.Tensor],
    batch: dict[str, Any],
    config: dict[str, Any],
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Compute E7 total loss from SupCon and train-time morphology heads."""

    loss_name = str(config["training"].get("loss"))
    if loss_name not in {"supervised_contrastive_with_morphology", "sqi_weighted_supcon_with_morphology"}:
        raise ValueError("Morphology objective requires supervised_contrastive_with_morphology or sqi_weighted_supcon_with_morphology.")
    components = config.get("loss_components", {})
    if float(components.get("session_centroid_alignment_weight", 0.0)) != 0.0:
        raise ValueError("E7 morphology objective does not include session alignment.")
    if bool(components.get("use_ipa", False)):
        raise ValueError("E7/E8 morphology objectives do not use IPA loss.")

    positive_mask = build_positive_mask(config["training"]["positive_mask_mode"], subject_ids, session_ids).to(embeddings.device)
    sqi_weight_diag = {
        "mode": components.get("sqi_weighting_mode"),
        "weight_min": None,
        "weight_max": None,
        "weight_mean": None,
        "weight_std": None,
        "effective_weighted_anchor_count": None,
    }
    if loss_name == "sqi_weighted_supcon_with_morphology":
        if not bool(components.get("sqi_weighting_enabled", False)):
            raise ValueError("E8 SQI-weighted objective requires sqi_weighting_enabled=true.")
        sqi = _batch_tensor(batch, "sqi", embeddings)
        sqi_mask = _batch_mask(batch, "sqi_valid_mask", embeddings)
        weights, weight_diag = compute_sqi_weights(
            sqi,
            sqi_mask,
            str(components.get("sqi_weighting_mode", "mild_linear")),
        )
        supcon_loss, supcon_diag = weighted_supervised_contrastive_loss(
            embeddings,
            positive_mask,
            weights,
            float(config["training"]["temperature"]),
            return_diagnostics=True,
        )
        sqi_weight_diag = {**weight_diag, **{key: supcon_diag.get(key) for key in (
            "weight_min",
            "weight_max",
            "weight_mean",
            "weight_std",
            "effective_weighted_anchor_count",
        )}}
    else:
        if bool(components.get("sqi_weighting_enabled", False)):
            raise ValueError("E7 objective does not enable SQI weighting.")
        supcon_loss, supcon_diag = supervised_contrastive_loss(
            embeddings,
            positive_mask,
            float(config["training"]["temperature"]),
            return_diagnostics=True,
        )
    morph_loss, morph_diag = compute_morphology_losses(morphology_predictions, batch, config)
    supcon_weight = float(components.get("supervised_contrastive_weight", 1.0))
    morph_weight = float(components.get("morphology_weight", 1.0))
    total = supcon_weight * supcon_loss + morph_weight * morph_loss
    if not torch.isfinite(total):
        raise RuntimeError("E7 total objective is nonfinite.")
    diagnostics = {
        "objective_name": "sqi_weighted_supcon_with_svri_sqi_morphology"
        if loss_name == "sqi_weighted_supcon_with_morphology"
        else "generic_supcon_with_svri_sqi_morphology",
        "total_loss": float(total.detach().cpu()),
        "supcon_loss": float(supcon_loss.detach().cpu()),
        "weighted_supcon_loss": supcon_diag.get("weighted_supcon_loss", float(supcon_loss.detach().cpu())),
        "unweighted_supcon_loss": supcon_diag.get("unweighted_supcon_loss", float(supcon_loss.detach().cpu())),
        "alignment_loss": 0.0,
        "svri_loss": morph_diag["svri_loss"],
        "sqi_loss": morph_diag["sqi_loss"],
        "morphology_loss": float(morph_loss.detach().cpu()),
        "lambda_svri": morph_diag["lambda_svri"],
        "lambda_sqi": morph_diag["lambda_sqi"],
        "svri_valid_count": morph_diag["svri_valid_count"],
        "sqi_valid_count": morph_diag["sqi_valid_count"],
        "svri_pred_mean": morph_diag["svri_pred_mean"],
        "sqi_pred_mean": morph_diag["sqi_pred_mean"],
        "svri_target_mean": morph_diag["svri_target_mean"],
        "sqi_target_mean": morph_diag["sqi_target_mean"],
        "mean_positive_similarity": supcon_diag["mean_positive_similarity"],
        "mean_negative_similarity": supcon_diag["mean_negative_similarity"],
        "positive_pair_count": supcon_diag["positive_pair_count"],
        "positive_count_min": supcon_diag["positive_count_min"],
        "positive_count_max": supcon_diag["positive_count_max"],
        "morphology_loss_active": True,
        "sqi_weighting_active": loss_name == "sqi_weighted_supcon_with_morphology",
        "sqi_weighting_mode": components.get("sqi_weighting_mode"),
        "sqi_weight_min": sqi_weight_diag.get("weight_min"),
        "sqi_weight_max": sqi_weight_diag.get("weight_max"),
        "sqi_weight_mean": sqi_weight_diag.get("weight_mean"),
        "sqi_weight_std": sqi_weight_diag.get("weight_std"),
        "effective_weighted_anchor_count": sqi_weight_diag.get("effective_weighted_anchor_count"),
        "ipa_loss_active": False,
        "ipa_used": False,
    }
    return total, diagnostics


def _batch_tensor(batch: dict[str, Any], key: str, reference: torch.Tensor) -> torch.Tensor:
    value = batch[key]
    if isinstance(value, torch.Tensor):
        return value.to(device=reference.device, dtype=reference.dtype)
    return torch.as_tensor(value, device=reference.device, dtype=reference.dtype)


def _batch_mask(batch: dict[str, Any], key: str, reference: torch.Tensor) -> torch.Tensor:
    value = batch[key]
    if isinstance(value, torch.Tensor):
        return value.to(device=reference.device, dtype=torch.bool)
    return torch.as_tensor(value, device=reference.device, dtype=torch.bool)
