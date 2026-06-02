"""Encoder interface and validation helpers for PPG verification."""

from __future__ import annotations

from abc import abstractmethod

import torch
from torch import nn


class PPGEncoderInterface(nn.Module):
    """Common interface for frozen and trainable PPG encoders."""

    encoder_id: str
    embedding_dim: int

    @abstractmethod
    def encode(self, waveforms: torch.Tensor) -> torch.Tensor:
        """Return embeddings for waveforms shaped [B, 1, 1250]."""

    def forward(self, waveforms: torch.Tensor) -> torch.Tensor:
        return self.encode(waveforms)


def validate_waveform_input(waveforms: torch.Tensor, samples: int = 1250) -> None:
    """Validate common PPG input tensor shape and finiteness."""

    if waveforms.ndim != 3:
        raise ValueError(f"Expected waveforms [B, 1, {samples}], got ndim={waveforms.ndim}")
    if waveforms.shape[1] != 1:
        raise ValueError(f"Expected one channel, got {waveforms.shape[1]}")
    if waveforms.shape[2] != samples:
        raise ValueError(f"Expected {samples} samples, got {waveforms.shape[2]}")
    if not torch.isfinite(waveforms).all():
        raise ValueError("Waveform input contains nonfinite values.")


def validate_embedding_output(waveforms: torch.Tensor, embeddings: torch.Tensor) -> None:
    """Validate encoder embedding output."""

    if embeddings.ndim != 2:
        raise ValueError(f"Expected embeddings [B, D], got ndim={embeddings.ndim}")
    if embeddings.shape[0] != waveforms.shape[0]:
        raise ValueError("Embedding batch dimension does not match input batch.")
    if not torch.isfinite(embeddings).all():
        raise ValueError("Encoder produced nonfinite embeddings.")


def freeze_encoder(encoder: nn.Module) -> nn.Module:
    """Switch encoder to frozen inference mode."""

    encoder.eval()
    for parameter in encoder.parameters():
        parameter.requires_grad_(False)
    return encoder


def count_trainable_parameters(module: nn.Module) -> int:
    """Return the number of trainable parameters."""

    return sum(parameter.numel() for parameter in module.parameters() if parameter.requires_grad)
