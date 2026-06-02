"""Train-time morphology prediction heads for E7 preservation objectives."""

from __future__ import annotations

from typing import Any, Iterable

import torch
from torch import nn


SUPPORTED_E7_TARGETS = {"svri", "sqi"}


class MorphologyHeads(nn.Module):
    """Independent small regression heads on top of projected embeddings."""

    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__()
        self.input_dim = int(config.get("input_dim", 128))
        self.hidden_dim = int(config.get("hidden_dim", 64))
        targets = [str(target) for target in config.get("targets", ["svri", "sqi"])]
        unsupported = sorted(set(targets) - SUPPORTED_E7_TARGETS)
        if unsupported:
            raise ValueError(f"Unsupported E7 morphology targets: {unsupported}")
        if "ipa" in targets:
            raise ValueError("IPA head is intentionally not created in E7.")
        self.enabled_targets = list(targets)
        self.heads = nn.ModuleDict(
            {
                target: nn.Sequential(
                    nn.Linear(self.input_dim, self.hidden_dim),
                    nn.ReLU(),
                    nn.Linear(self.hidden_dim, 1),
                )
                for target in self.enabled_targets
            }
        )

    def forward(self, projected_embeddings: torch.Tensor) -> dict[str, torch.Tensor]:
        """Return target_name_pred -> [B] finite tensors."""

        if projected_embeddings.ndim != 2 or projected_embeddings.shape[1] != self.input_dim:
            raise ValueError(f"Expected projected embeddings [B, {self.input_dim}].")
        predictions: dict[str, torch.Tensor] = {}
        for target, head in self.heads.items():
            pred = head(projected_embeddings).squeeze(-1)
            if pred.ndim != 1:
                raise RuntimeError(f"{target} morphology prediction must be [B].")
            if not torch.isfinite(pred).all():
                raise RuntimeError(f"{target} morphology head produced nonfinite predictions.")
            predictions[f"{target}_pred"] = pred
        return predictions

    def parameter_count(self) -> int:
        """Return trainable parameter count in morphology heads."""

        return sum(parameter.numel() for parameter in self.parameters())

    def metadata(self) -> dict[str, Any]:
        """Return architecture metadata."""

        return {
            "enabled_targets": list(self.enabled_targets),
            "input_dim": self.input_dim,
            "hidden_dim": self.hidden_dim,
            "parameter_count": self.parameter_count(),
            "ipa_head_created": False,
        }


def morphology_heads_enabled(model_config: dict[str, Any]) -> bool:
    """Return whether morphology heads are enabled in model config."""

    return bool(model_config.get("morphology_heads", {}).get("enabled", False))

