"""Config-driven objective composition for SigD adaptation stages."""

from __future__ import annotations

from typing import Any, Sequence

import torch

from positive_masks import build_positive_mask
from session_alignment_loss import SessionCentroidAlignmentLoss
from supervised_contrastive_loss import supervised_contrastive_loss


SUPPORTED_OBJECTIVES = {
    ("same_subject_different_sample", 0.0): "generic_supcon_noalign",
    ("same_subject_different_session", 0.0): "cs_supcon_noalign",
}


def lambda_align_from_config(config: dict[str, Any]) -> float:
    """Return the configured session-centroid alignment weight."""

    components = config.get("loss_components", {})
    if "session_centroid_alignment_weight" in components:
        return float(components.get("session_centroid_alignment_weight", 0.0))
    return float(components.get("centroid_alignment_weight", 0.0))


def infer_objective_name(config: dict[str, Any]) -> str:
    """Infer objective family from positive-mask mode and alignment weight."""

    mode = str(config["training"]["positive_mask_mode"])
    weight = lambda_align_from_config(config)
    if weight < 0:
        raise ValueError("session_centroid_alignment_weight must be nonnegative.")
    if mode == "same_subject_different_sample":
        return "generic_supcon_with_alignment" if weight > 0 else "generic_supcon_noalign"
    if mode == "same_subject_different_session":
        return "cs_supcon_with_alignment" if weight > 0 else "cs_supcon_noalign"
    raise ValueError(f"Unsupported positive_mask_mode for objective: {mode}")


def compute_total_objective(
    embeddings: torch.Tensor,
    subject_ids: Sequence[str],
    session_ids: Sequence[str],
    config: dict[str, Any],
    *,
    return_diagnostics: bool = True,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Compute SupCon plus optional session-centroid alignment."""

    if str(config["training"].get("loss", "supervised_contrastive")) != "supervised_contrastive":
        raise ValueError(f"Unsupported training loss: {config['training'].get('loss')}")
    objective_name = infer_objective_name(config)
    components = config.get("loss_components", {})
    supcon_weight = float(components.get("supervised_contrastive_weight", 1.0))
    if supcon_weight <= 0:
        raise ValueError("supervised_contrastive_weight must be positive.")
    lambda_align = lambda_align_from_config(config)
    positive_mask = build_positive_mask(config["training"]["positive_mask_mode"], subject_ids, session_ids).to(embeddings.device)
    supcon_loss, supcon_diag = supervised_contrastive_loss(
        embeddings,
        positive_mask,
        float(config["training"]["temperature"]),
        return_diagnostics=True,
    )
    alignment_loss = embeddings.new_tensor(0.0)
    alignment_diag: dict[str, Any] = {
        "subject_count": None,
        "centroid_pair_count": 0,
        "mean_centroid_cosine": None,
        "min_centroid_cosine": None,
        "max_centroid_cosine": None,
    }
    if lambda_align > 0:
        alignment_loss_module = SessionCentroidAlignmentLoss()
        alignment_loss, alignment_diag = alignment_loss_module(
            embeddings,
            subject_ids,
            session_ids,
            return_diagnostics=True,
        )
    total_loss = supcon_weight * supcon_loss + float(lambda_align) * alignment_loss
    if not torch.isfinite(total_loss):
        raise RuntimeError("Total adaptation objective is nonfinite.")

    diagnostics = {
        "objective_name": objective_name,
        "total_loss": float(total_loss.detach().cpu()),
        "supcon_loss": float(supcon_loss.detach().cpu()),
        "alignment_loss": float(alignment_loss.detach().cpu()),
        "lambda_align": float(lambda_align),
        "supervised_contrastive_weight": float(supcon_weight),
        "mean_positive_similarity": supcon_diag["mean_positive_similarity"],
        "mean_negative_similarity": supcon_diag["mean_negative_similarity"],
        "positive_pair_count": supcon_diag["positive_pair_count"],
        "positive_count_min": supcon_diag["positive_count_min"],
        "positive_count_max": supcon_diag["positive_count_max"],
        "mean_centroid_cosine": alignment_diag.get("mean_centroid_cosine"),
        "alignment_diagnostics": alignment_diag,
        "morphology_loss_active": False,
        "sqi_weighting_active": False,
        "ipa_loss_active": False,
    }
    return total_loss, diagnostics

