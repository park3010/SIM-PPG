"""Masked sVRI/SQI morphology preservation losses."""

from __future__ import annotations

from typing import Any

import torch


def masked_mse_loss(pred: torch.Tensor, target: torch.Tensor, valid_mask: torch.Tensor) -> tuple[torch.Tensor, int]:
    """Compute MSE only over valid samples, returning zero if none are valid."""

    if pred.ndim != 1:
        raise ValueError("pred must be [B].")
    target = target.to(device=pred.device, dtype=pred.dtype)
    valid_mask = valid_mask.to(device=pred.device, dtype=torch.bool)
    if target.shape != pred.shape or valid_mask.shape != pred.shape:
        raise ValueError("target/mask shape must match pred.")
    valid_count = int(valid_mask.sum().detach().cpu().item())
    if valid_count == 0:
        return pred.sum() * 0.0, 0
    if not torch.isfinite(target[valid_mask]).all():
        raise ValueError("Morphology target contains nonfinite values on valid samples.")
    loss = torch.mean((pred[valid_mask] - target[valid_mask]) ** 2)
    if not torch.isfinite(loss):
        raise RuntimeError("Masked morphology MSE is nonfinite.")
    return loss, valid_count


def compute_morphology_losses(
    predictions: dict[str, torch.Tensor],
    batch: dict[str, Any],
    config: dict[str, Any],
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Compute weighted E7 sVRI/SQI losses; IPA is ignored."""

    components = config.get("loss_components", {})
    lambda_svri = float(components.get("lambda_svri", 0.0))
    lambda_sqi = float(components.get("lambda_sqi", 0.0))
    if bool(components.get("use_ipa", False)):
        raise ValueError("IPA is not enabled for E7 morphology preservation.")

    svri_pred = predictions["svri_pred"]
    sqi_pred = predictions["sqi_pred"]
    svri_target = _tensor_from_batch(batch, "svri", svri_pred)
    sqi_target = _tensor_from_batch(batch, "sqi", sqi_pred)
    svri_mask = _mask_from_batch(batch, "svri_valid_mask", svri_pred)
    sqi_mask = _mask_from_batch(batch, "sqi_valid_mask", sqi_pred)
    svri_loss, svri_count = masked_mse_loss(svri_pred, svri_target, svri_mask)
    sqi_loss, sqi_count = masked_mse_loss(sqi_pred, sqi_target, sqi_mask)
    total = lambda_svri * svri_loss + lambda_sqi * sqi_loss
    if not torch.isfinite(total):
        raise RuntimeError("Total morphology loss is nonfinite.")
    diagnostics = {
        "svri_loss": float(svri_loss.detach().cpu()),
        "sqi_loss": float(sqi_loss.detach().cpu()),
        "svri_valid_count": svri_count,
        "sqi_valid_count": sqi_count,
        "svri_pred_mean": _masked_mean(svri_pred, svri_mask),
        "sqi_pred_mean": _masked_mean(sqi_pred, sqi_mask),
        "svri_target_mean": _masked_mean(svri_target.to(svri_pred.device), svri_mask),
        "sqi_target_mean": _masked_mean(sqi_target.to(sqi_pred.device), sqi_mask),
        "lambda_svri": lambda_svri,
        "lambda_sqi": lambda_sqi,
        "ipa_used": False,
    }
    return total, diagnostics


def _tensor_from_batch(batch: dict[str, Any], key: str, reference: torch.Tensor) -> torch.Tensor:
    if key not in batch:
        raise KeyError(f"Batch missing morphology target: {key}")
    value = batch[key]
    if isinstance(value, torch.Tensor):
        return value.to(device=reference.device, dtype=reference.dtype)
    return torch.as_tensor(value, device=reference.device, dtype=reference.dtype)


def _mask_from_batch(batch: dict[str, Any], key: str, reference: torch.Tensor) -> torch.Tensor:
    if key not in batch:
        raise KeyError(f"Batch missing morphology valid mask: {key}")
    value = batch[key]
    if isinstance(value, torch.Tensor):
        return value.to(device=reference.device, dtype=torch.bool)
    return torch.as_tensor(value, device=reference.device, dtype=torch.bool)


def _masked_mean(values: torch.Tensor, mask: torch.Tensor) -> float | None:
    mask = mask.to(device=values.device, dtype=torch.bool)
    if not mask.any():
        return None
    return float(values[mask].detach().mean().cpu())

