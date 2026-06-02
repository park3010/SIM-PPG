"""Batch-level SQI reliability weights for E8 training-only SupCon weighting."""

from __future__ import annotations

from typing import Any

import torch


SUPPORTED_SQI_WEIGHTING_MODES = {
    "mild_linear",
    "clipped_linear",
    "strong_linear",
    "rank_bottom20_downweight",
}


def robust_normalize_sqi(sqi: torch.Tensor, sqi_valid_mask: torch.Tensor, eps: float = 1.0e-8) -> tuple[torch.Tensor, dict[str, Any]]:
    """Normalize SQI to [0, 1] using valid-sample q05/q95; invalid samples become neutral high reliability."""

    sqi = sqi.to(dtype=torch.float32)
    mask = sqi_valid_mask.to(device=sqi.device, dtype=torch.bool)
    if sqi.ndim != 1 or mask.shape != sqi.shape:
        raise ValueError("sqi and sqi_valid_mask must be [B].")
    if not torch.isfinite(sqi[mask]).all():
        raise ValueError("Valid SQI values contain nonfinite entries.")
    valid_count = int(mask.sum().detach().cpu().item())
    normalized = torch.ones_like(sqi, dtype=torch.float32)
    if valid_count < 4:
        return normalized, {"valid_count": valid_count, "q05": None, "q95": None, "constant_or_insufficient": True}
    valid = sqi[mask]
    q05 = torch.quantile(valid, 0.05)
    q95 = torch.quantile(valid, 0.95)
    spread = q95 - q05
    if not torch.isfinite(spread) or float(spread.detach().cpu()) < float(eps):
        return normalized, {
            "valid_count": valid_count,
            "q05": float(q05.detach().cpu()),
            "q95": float(q95.detach().cpu()),
            "constant_or_insufficient": True,
        }
    normalized_valid = torch.clamp((valid - q05) / (spread + float(eps)), min=0.0, max=1.0)
    normalized = normalized.clone()
    normalized[mask] = normalized_valid
    return normalized, {
        "valid_count": valid_count,
        "q05": float(q05.detach().cpu()),
        "q95": float(q95.detach().cpu()),
        "constant_or_insufficient": False,
    }


def compute_sqi_weights(
    sqi: torch.Tensor,
    sqi_valid_mask: torch.Tensor,
    mode: str,
    eps: float = 1.0e-8,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Return per-anchor SQI weights and diagnostics."""

    if mode not in SUPPORTED_SQI_WEIGHTING_MODES:
        raise ValueError(f"Unsupported SQI weighting mode: {mode}")
    sqi = sqi.to(dtype=torch.float32)
    mask = sqi_valid_mask.to(device=sqi.device, dtype=torch.bool)
    if mode == "rank_bottom20_downweight":
        weights = _rank_bottom20_weights(sqi, mask)
        norm_info = {"valid_count": int(mask.sum().detach().cpu().item()), "q05": None, "q95": None}
    else:
        normalized, norm_info = robust_normalize_sqi(sqi, mask, eps)
        if norm_info.get("constant_or_insufficient"):
            weights = torch.ones_like(sqi, dtype=torch.float32)
        elif mode == "mild_linear":
            weights = 0.5 + 0.5 * normalized
        elif mode == "clipped_linear":
            weights = torch.clamp(normalized, min=0.5, max=1.0)
        elif mode == "strong_linear":
            weights = 0.25 + 0.75 * normalized
        else:  # pragma: no cover - guarded above
            raise ValueError(mode)
        weights = weights.to(device=sqi.device, dtype=torch.float32)
        weights[~mask] = 1.0
    if not torch.isfinite(weights).all():
        raise RuntimeError("SQI weights are nonfinite.")
    diagnostics = {
        **norm_info,
        "mode": mode,
        "weight_min": float(weights.min().detach().cpu()),
        "weight_max": float(weights.max().detach().cpu()),
        "weight_mean": float(weights.mean().detach().cpu()),
        "weight_std": float(weights.std(unbiased=False).detach().cpu()),
    }
    return weights, diagnostics


def _rank_bottom20_weights(sqi: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    valid_count = int(mask.sum().detach().cpu().item())
    weights = torch.ones_like(sqi, dtype=torch.float32)
    if valid_count < 5:
        return weights
    if not torch.isfinite(sqi[mask]).all():
        raise ValueError("Valid SQI values contain nonfinite entries.")
    valid_indices = torch.where(mask)[0]
    valid_values = sqi[valid_indices]
    bottom_count = max(1, int(valid_count * 0.2))
    order = torch.argsort(valid_values)
    weights[valid_indices[order[:bottom_count]]] = 0.5
    return weights

