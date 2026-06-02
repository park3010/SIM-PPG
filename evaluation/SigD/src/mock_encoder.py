"""Deterministic mock encoder for evaluation-engine correctness tests."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from encoder_interface import PPGEncoderInterface, validate_embedding_output, validate_waveform_input


class DeterministicMockEncoder(PPGEncoderInterface):
    """Parameter-free deterministic encoder for audit-only evaluation."""

    encoder_id = "deterministic_mock_encoder_engine_audit_only"

    def __init__(self, embedding_dim: int = 64) -> None:
        super().__init__()
        if embedding_dim != 64:
            raise ValueError("DeterministicMockEncoder currently emits exactly 64 features.")
        self.embedding_dim = int(embedding_dim)

    @torch.inference_mode()
    def encode(self, waveforms: torch.Tensor) -> torch.Tensor:
        """Encode waveforms by deterministic pooled summary statistics."""

        validate_waveform_input(waveforms)
        x = waveforms.to(dtype=torch.float32)
        pooled_mean = F.adaptive_avg_pool1d(x, 32).squeeze(1)
        centered = x - x.mean(dim=-1, keepdim=True)
        pooled_std = torch.sqrt(F.adaptive_avg_pool1d(centered.square(), 32).squeeze(1) + 1.0e-8)
        embeddings = torch.cat([pooled_mean, pooled_std], dim=1)
        validate_embedding_output(waveforms, embeddings)
        return embeddings


def mock_encoder_metadata() -> dict[str, object]:
    """Return audit-only metadata for the deterministic mock encoder."""

    return {
        "encoder_id": DeterministicMockEncoder.encoder_id,
        "metrics_valid_for_scientific_reporting": False,
        "purpose": "evaluation_engine_correctness_only",
        "pretrained_weights_verified": False,
    }
