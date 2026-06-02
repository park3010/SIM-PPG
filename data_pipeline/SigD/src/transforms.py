"""Waveform transforms shared by all common-input SigD models."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch


class PerWindowZScore:
    """Apply per-window z-score normalization and return [1, T] float32."""

    def __init__(self, eps: float = 1.0e-8, output_channel_first: bool = True) -> None:
        self.eps = float(eps)
        self.output_channel_first = bool(output_channel_first)

    def __call__(self, values: Any) -> torch.Tensor:
        tensor = torch.as_tensor(np.array(values, copy=True), dtype=torch.float32)
        if tensor.ndim == 2 and tensor.shape[0] == 1:
            tensor = tensor.squeeze(0)
        if tensor.ndim != 1:
            raise ValueError(f"Expected waveform shape [T] or [1,T], got {tuple(tensor.shape)}")
        if not torch.isfinite(tensor).all():
            raise ValueError("Nonfinite waveform values cannot be z-score normalized.")
        mean = tensor.mean()
        std = tensor.std(unbiased=False)
        normalized = (tensor - mean) / (std + self.eps)
        if self.output_channel_first:
            normalized = normalized.unsqueeze(0)
        return normalized.to(dtype=torch.float32)


class IdentityTransform:
    """Return a [1, T] float32 tensor without normalization."""

    def __call__(self, values: Any) -> torch.Tensor:
        tensor = torch.as_tensor(np.array(values, copy=True), dtype=torch.float32)
        if tensor.ndim == 1:
            tensor = tensor.unsqueeze(0)
        if tensor.ndim != 2 or tensor.shape[0] != 1:
            raise ValueError(f"Expected waveform shape [T] or [1,T], got {tuple(tensor.shape)}")
        if not torch.isfinite(tensor).all():
            raise ValueError("Nonfinite waveform values are not allowed.")
        return tensor
