"""Frozen PaPaGei-S backbone with a trainable projection head."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch import nn
import torch.nn.functional as F

from common import add_evaluation_src, load_json, resolve_from_root
from morphology_heads import MorphologyHeads, morphology_heads_enabled


SOURCE_OVERLAP_LIMITATION = {
    "overlap_risk_present": True,
    "reason": "PaPaGei pretraining includes MIMIC-III, while SigD-Core is reconstructed from MIMIC-III waveform records.",
    "interpretation_policy": "Report adaptation results with explicit source-level overlap limitation; use same-backbone ablations for method-level claims.",
}


class ProjectionHead(nn.Module):
    """Small MLP projection head shared by E4/E5."""

    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__()
        input_dim = int(config["input_dim"])
        hidden_dim = int(config["hidden_dim"])
        output_dim = int(config["output_dim"])
        norm_name = str(config.get("normalization_layer", "layer_norm")).lower()
        if norm_name == "layer_norm":
            norm = nn.LayerNorm(hidden_dim)
        elif norm_name == "batch_norm":
            norm = nn.BatchNorm1d(hidden_dim)
        else:
            raise ValueError(f"Unsupported normalization layer: {norm_name}")
        activation = nn.ReLU() if str(config.get("activation", "relu")).lower() == "relu" else None
        if activation is None:
            raise ValueError(f"Unsupported activation: {config.get('activation')}")
        self.output_l2_normalization = bool(config.get("output_l2_normalization", True))
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            norm,
            activation,
            nn.Dropout(float(config.get("dropout", 0.0))),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        projected = self.net(embeddings)
        if self.output_l2_normalization:
            projected = F.normalize(projected, p=2, dim=1, eps=1.0e-8)
        if not torch.isfinite(projected).all():
            raise RuntimeError("Projection head produced nonfinite embeddings.")
        return projected


class PaPaGeiProjectionModel(nn.Module):
    """Encode common PPG windows with frozen PaPaGei-S and train a projection head."""

    def __init__(
        self,
        root: Path,
        config: dict[str, Any],
        backbone_adapter: nn.Module | None = None,
    ) -> None:
        super().__init__()
        self.root = root
        self.config = config
        self.backbone = backbone_adapter if backbone_adapter is not None else self._load_verified_backbone()
        self.backbone_embedding_dim = int(config["model"]["backbone_embedding_dim"])
        self.projection_head = ProjectionHead(config["model"]["projection_head"])
        if morphology_heads_enabled(config["model"]):
            self.morphology_heads = MorphologyHeads(config["model"]["morphology_heads"])
        else:
            self.morphology_heads = None
        self.output_embedding_dim = int(config["model"]["projection_head"]["output_dim"])
        self._freeze_backbone()

    def _load_verified_backbone(self) -> nn.Module:
        add_evaluation_src(self.root)
        from papagei_s_adapter import PaPaGeiSFrozenAdapter

        eval_config_path = resolve_from_root(self.root, self.config["input"]["frozen_eval_config_reference"])
        if eval_config_path is None:
            raise RuntimeError("Frozen evaluation config path missing.")
        import yaml

        with eval_config_path.open("r", encoding="utf-8") as handle:
            eval_config = yaml.safe_load(handle) or {}
        return PaPaGeiSFrozenAdapter(self.root, eval_config)

    def _freeze_backbone(self) -> None:
        self.backbone.eval()
        for parameter in self.backbone.parameters():
            parameter.requires_grad_(False)

    def train(self, mode: bool = True):  # type: ignore[override]
        super().train(mode)
        self._freeze_backbone()
        self.projection_head.train(mode)
        if self.morphology_heads is not None:
            self.morphology_heads.train(mode)
        return self

    def encode_backbone(self, waveforms: torch.Tensor) -> torch.Tensor:
        """Return frozen 512-d PaPaGei-S embeddings without gradient tracking."""

        if waveforms.ndim != 3 or waveforms.shape[1] != 1 or waveforms.shape[2] != 1250:
            raise ValueError(f"Expected waveforms [B, 1, 1250], got {tuple(waveforms.shape)}")
        self._freeze_backbone()
        with torch.no_grad():
            embeddings = self.backbone.encode(waveforms).detach().clone()
        if embeddings.ndim != 2 or embeddings.shape[1] != self.backbone_embedding_dim:
            raise RuntimeError(f"Unexpected backbone embedding shape: {tuple(embeddings.shape)}")
        if not torch.isfinite(embeddings).all():
            raise RuntimeError("Backbone produced nonfinite embeddings.")
        return embeddings

    def project(self, backbone_embeddings: torch.Tensor) -> torch.Tensor:
        """Project backbone embeddings to normalized 128-d verification embeddings."""

        if backbone_embeddings.ndim != 2 or backbone_embeddings.shape[1] != self.backbone_embedding_dim:
            raise ValueError(f"Expected backbone embeddings [B, {self.backbone_embedding_dim}].")
        return self.projection_head(backbone_embeddings.to(dtype=torch.float32))

    def encode(self, waveforms: torch.Tensor) -> torch.Tensor:
        """Encode common-input waveforms into trainable projection embeddings."""

        return self.project(self.encode_backbone(waveforms))

    def predict_morphology(self, projected_embeddings: torch.Tensor) -> dict[str, torch.Tensor]:
        """Return train-time morphology predictions from projected embeddings."""

        if self.morphology_heads is None:
            raise RuntimeError("Morphology heads are not enabled for this model.")
        return self.morphology_heads(projected_embeddings)

    def trainable_parameter_count(self) -> int:
        """Return number of trainable parameters."""

        return sum(parameter.numel() for parameter in self.parameters() if parameter.requires_grad)

    def get_model_metadata(self) -> dict[str, Any]:
        """Return model provenance and architecture metadata."""

        manifest_path = self.root / "evaluation" / "SigD" / "metadata" / "papagei_model_reference_manifest.json"
        manifest = load_json(manifest_path) if manifest_path.exists() else {}
        return {
            "backbone_variant": "PaPaGei-S",
            "backbone_architecture": "ResNet1DMoE",
            "backbone_frozen": True,
            "backbone_checkpoint_sha256": manifest.get("checkpoint_sha256"),
            "backbone_checkpoint_md5": manifest.get("checkpoint_md5"),
            "pretrained_weights_verified": bool(manifest.get("pretrained_weights_verified")),
            "projection_head_architecture": self.config["model"]["projection_head"],
            "morphology_heads_enabled": self.morphology_heads is not None,
            "morphology_heads": self.morphology_heads.metadata() if self.morphology_heads is not None else None,
            "output_embedding_dim": self.output_embedding_dim,
            "trainable_parameter_count": self.trainable_parameter_count(),
            "input_setting": "common_input",
            "official_native_preprocessing_reapplied": False,
            "source_overlap_limitation": SOURCE_OVERLAP_LIMITATION,
        }
