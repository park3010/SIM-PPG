"""Supervised contrastive loss with explicit positive masks."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F


def supervised_contrastive_loss(
    embeddings: torch.Tensor,
    positive_mask: torch.Tensor,
    temperature: float = 0.07,
    *,
    return_diagnostics: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, dict[str, Any]]:
    """Compute SupCon loss over L2-normalized embeddings and a positive mask."""

    if temperature <= 0:
        raise ValueError("temperature must be positive.")
    if embeddings.ndim != 2:
        raise ValueError("embeddings must be [B, D].")
    batch_size = embeddings.shape[0]
    if positive_mask.shape != (batch_size, batch_size):
        raise ValueError("positive_mask shape must match [B, B].")
    if positive_mask.dtype is not torch.bool:
        positive_mask = positive_mask.to(dtype=torch.bool)
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
    positive_log_prob = torch.where(
        positive_mask.to(log_prob.device),
        log_prob,
        torch.zeros_like(log_prob),
    ).sum(dim=1) / positive_counts.to(log_prob.device)
    loss = -positive_log_prob.mean()
    if not torch.isfinite(loss).all():
        raise RuntimeError("SupCon loss is nonfinite.")

    if not return_diagnostics:
        return loss

    with torch.no_grad():
        sim = embeddings @ embeddings.T
        negative_mask = (~positive_mask.to(embeddings.device)) & (~self_mask)
        diagnostics = {
            "loss": float(loss.detach().cpu()),
            "positive_pair_count": int(positive_mask.sum().item()),
            "mean_positive_similarity": float(sim[positive_mask.to(embeddings.device)].mean().detach().cpu()),
            "mean_negative_similarity": float(sim[negative_mask].mean().detach().cpu()) if negative_mask.any() else None,
            "positive_count_min": int(positive_counts.min().item()),
            "positive_count_max": int(positive_counts.max().item()),
        }
    return loss, diagnostics
