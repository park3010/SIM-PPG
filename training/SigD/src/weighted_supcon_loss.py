"""Anchor-level weighted supervised contrastive loss."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F


def weighted_supervised_contrastive_loss(
    embeddings: torch.Tensor,
    positive_mask: torch.Tensor,
    anchor_weights: torch.Tensor,
    temperature: float = 0.07,
    *,
    eps: float = 1.0e-8,
    return_diagnostics: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, dict[str, Any]]:
    """Compute SupCon with anchor-level reliability weights."""

    per_anchor_loss, diagnostics = per_anchor_supcon_loss(
        embeddings,
        positive_mask,
        temperature,
    )
    weights = anchor_weights.to(device=embeddings.device, dtype=embeddings.dtype)
    if weights.shape != per_anchor_loss.shape:
        raise ValueError("anchor_weights must be [B].")
    if not torch.isfinite(weights).all():
        raise ValueError("anchor_weights contain nonfinite values.")
    if (weights < 0).any().item():
        raise ValueError("anchor_weights must be nonnegative.")
    denominator = weights.sum() + float(eps)
    if float(denominator.detach().cpu()) <= float(eps):
        raise ValueError("anchor_weights sum is zero.")
    weighted_loss = torch.sum(weights * per_anchor_loss) / denominator
    if not torch.isfinite(weighted_loss):
        raise RuntimeError("Weighted SupCon loss is nonfinite.")
    if not return_diagnostics:
        return weighted_loss
    output = {
        **diagnostics,
        "weighted_supcon_loss": float(weighted_loss.detach().cpu()),
        "unweighted_supcon_loss": float(per_anchor_loss.mean().detach().cpu()),
        "weight_mean": float(weights.mean().detach().cpu()),
        "weight_min": float(weights.min().detach().cpu()),
        "weight_max": float(weights.max().detach().cpu()),
        "weight_std": float(weights.std(unbiased=False).detach().cpu()),
        "effective_weighted_anchor_count": float(weights.sum().detach().cpu()),
    }
    return weighted_loss, output


def per_anchor_supcon_loss(
    embeddings: torch.Tensor,
    positive_mask: torch.Tensor,
    temperature: float,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Return stable per-anchor SupCon losses."""

    if temperature <= 0:
        raise ValueError("temperature must be positive.")
    if embeddings.ndim != 2:
        raise ValueError("embeddings must be [B, D].")
    batch_size = embeddings.shape[0]
    if positive_mask.shape != (batch_size, batch_size):
        raise ValueError("positive_mask shape must match [B, B].")
    positive_mask = positive_mask.to(device=embeddings.device, dtype=torch.bool)
    if torch.diagonal(positive_mask).any().item():
        raise ValueError("positive_mask diagonal must be false.")
    positive_counts = positive_mask.sum(dim=1)
    if (positive_counts == 0).any().item():
        anchors = torch.where(positive_counts == 0)[0].cpu().tolist()
        raise ValueError(f"Every anchor must have at least one positive; empty anchors={anchors[:10]}")
    if not torch.isfinite(embeddings).all():
        raise ValueError("embeddings contain nonfinite values.")

    embeddings = F.normalize(embeddings, p=2, dim=1, eps=1.0e-8)
    logits = embeddings @ embeddings.T / float(temperature)
    self_mask = torch.eye(batch_size, dtype=torch.bool, device=embeddings.device)
    logits = logits.masked_fill(self_mask, float("-inf"))
    log_denominator = torch.logsumexp(logits, dim=1)
    log_prob = logits - log_denominator.unsqueeze(1)
    positive_log_prob = torch.where(positive_mask, log_prob, torch.zeros_like(log_prob)).sum(dim=1) / positive_counts
    per_anchor = -positive_log_prob
    if not torch.isfinite(per_anchor).all():
        raise RuntimeError("Per-anchor SupCon loss is nonfinite.")
    with torch.no_grad():
        sim = embeddings @ embeddings.T
        negative_mask = (~positive_mask) & (~self_mask)
        diagnostics = {
            "positive_pair_count": int(positive_mask.sum().item()),
            "mean_positive_similarity": float(sim[positive_mask].mean().detach().cpu()),
            "mean_negative_similarity": float(sim[negative_mask].mean().detach().cpu()) if negative_mask.any() else None,
            "positive_count_min": int(positive_counts.min().item()),
            "positive_count_max": int(positive_counts.max().item()),
        }
    return per_anchor, diagnostics

