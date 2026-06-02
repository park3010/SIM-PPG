"""Verified PaPaGei-S frozen adapter for common-input SigD evaluation."""

from __future__ import annotations

import importlib
import sys
from pathlib import Path
from typing import Any

import torch

from common import load_json, resolve_from_root
from encoder_interface import (
    PPGEncoderInterface,
    count_trainable_parameters,
    validate_embedding_output,
    validate_waveform_input,
)


MODEL_CONFIG = {
    "base_filters": 32,
    "kernel_size": 3,
    "stride": 2,
    "groups": 1,
    "n_block": 18,
    "n_classes": 512,
    "n_experts": 3,
}
SOURCE_OVERLAP_LIMITATION = {
    "overlap_risk_present": True,
    "reason": "PaPaGei pretraining includes MIMIC-III, while SigD-Core is reconstructed from MIMIC-III waveform records.",
    "interpretation_policy": "Report frozen performance with explicit source-level overlap limitation; use controlled same-backbone adaptation ablations for method-level claims.",
}
COMMON_INPUT_POLICY = {
    "input_setting": "common_input",
    "official_native_preprocessing_reapplied": False,
    "common_transform_source": "data_pipeline/SigD/PerWindowZScore",
    "native_input_supplementary_evaluation_pending": True,
}


class PaPaGeiSFrozenAdapter(PPGEncoderInterface):
    """Official PaPaGei-S frozen encoder for common-input evaluation."""

    encoder_id = "papagei_s_frozen_official_common_input_v1"
    embedding_dim = 512

    def __init__(self, root: Path, config: dict[str, Any]) -> None:
        super().__init__()
        self.root = root
        self.config = config
        self.reference_manifest = self._load_reference_manifest()
        self._require_verified_reference()
        self.source_dir = resolve_from_root(root, config["encoder"]["official_source_dir"])
        self.checkpoint_path = resolve_from_root(root, config["encoder"]["checkpoint_path"])
        assert self.source_dir is not None and self.checkpoint_path is not None
        self.model = self._load_model()
        self.model.eval()
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)
        trainable = count_trainable_parameters(self.model)
        if trainable != 0:
            raise RuntimeError(f"PaPaGei-S frozen adapter has trainable parameters: {trainable}")

    def _load_reference_manifest(self) -> dict[str, Any]:
        path = self.root / "evaluation" / "SigD" / "metadata" / "papagei_model_reference_manifest.json"
        if not path.exists():
            raise RuntimeError("PaPaGei model reference manifest missing. Run setup_papagei_model_reference.py --verify.")
        return load_json(path)

    def _require_verified_reference(self) -> None:
        required_flags = {
            "ready_for_scientific_frozen_baseline": True,
            "checkpoint_verified": True,
            "official_source_verified": True,
            "pretrained_weights_verified": True,
            "architecture_verified": True,
            "loading_api_verified": True,
            "embedding_dim_verified": True,
        }
        failed = [key for key, expected in required_flags.items() if self.reference_manifest.get(key) is not expected]
        if failed:
            raise RuntimeError(f"PaPaGei-S reference is not verified for scientific baseline: {failed}")

    def _load_model(self) -> torch.nn.Module:
        sys.path.insert(0, str(self.source_dir))
        ResNet1DMoE = importlib.import_module("models.resnet").ResNet1DMoE
        importlib.import_module("linearprobing.utils").load_model_without_module_prefix
        model = ResNet1DMoE(in_channels=1, **MODEL_CONFIG)
        checkpoint = _safe_torch_load(self.checkpoint_path, checkpoint_verified=True)
        state_dict = _strip_known_prefixes(_extract_state_dict(checkpoint))
        incompatible = model.load_state_dict(state_dict, strict=False)
        missing = list(incompatible.missing_keys)
        unexpected = list(incompatible.unexpected_keys)
        if missing or unexpected:
            raise RuntimeError(
                "Strict PaPaGei-S checkpoint load failed: "
                f"missing={len(missing)} unexpected={len(unexpected)} "
                f"missing_preview={missing[:10]} unexpected_preview={unexpected[:10]}"
            )
        return model

    @torch.inference_mode()
    def encode(self, waveforms: torch.Tensor) -> torch.Tensor:
        """Encode common-input tensors shaped [B, 1, 1250]."""

        validate_waveform_input(waveforms)
        outputs = self.model(waveforms.to(dtype=torch.float32, device=next(self.model.parameters()).device))
        if not isinstance(outputs, (tuple, list)):
            raise RuntimeError("PaPaGei-S forward output is expected to be tuple/list; outputs[0] must be embedding.")
        embeddings = outputs[0].to(dtype=torch.float32)
        if embeddings.shape[1] != self.embedding_dim:
            raise RuntimeError(f"Unexpected PaPaGei-S embedding dim: {embeddings.shape}")
        validate_embedding_output(waveforms, embeddings)
        return embeddings

    def to(self, *args: Any, **kwargs: Any):  # type: ignore[override]
        super().to(*args, **kwargs)
        self.model.to(*args, **kwargs)
        return self

    def eval(self):  # type: ignore[override]
        super().eval()
        if hasattr(self, "model"):
            self.model.eval()
        return self

    def parameters(self, recurse: bool = True):  # type: ignore[override]
        return self.model.parameters(recurse=recurse)

    def get_encoder_metadata(self) -> dict[str, Any]:
        """Return reproducibility metadata for the verified encoder."""

        return {
            "encoder_id": self.encoder_id,
            "model_variant": "PaPaGei-S",
            "model_architecture": "ResNet1DMoE",
            "model_config": MODEL_CONFIG,
            "embedding_dim": self.embedding_dim,
            "checkpoint_path": self.reference_manifest.get("checkpoint_path"),
            "checkpoint_md5": self.reference_manifest.get("checkpoint_md5"),
            "checkpoint_sha256": self.reference_manifest.get("checkpoint_sha256"),
            "pretrained_weights_verified": True,
            "official_source_verified": True,
            "frozen": True,
            "trainable_parameter_count": count_trainable_parameters(self.model),
            **COMMON_INPUT_POLICY,
            "source_overlap_limitation": SOURCE_OVERLAP_LIMITATION,
        }


def _extract_state_dict(checkpoint: Any) -> dict[str, torch.Tensor]:
    if isinstance(checkpoint, dict):
        for key in ("state_dict", "model_state_dict", "model", "net", "encoder"):
            value = checkpoint.get(key)
            if isinstance(value, dict) and value and all(torch.is_tensor(v) for v in value.values()):
                return value
        if checkpoint and all(torch.is_tensor(v) for v in checkpoint.values()):
            return checkpoint
    raise RuntimeError("Could not extract tensor state_dict from PaPaGei checkpoint.")


def _strip_known_prefixes(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    cleaned = {}
    for key, value in state_dict.items():
        new_key = key
        if new_key.startswith("module."):
            new_key = new_key[len("module.") :]
        cleaned[new_key] = value
    return cleaned


def _safe_torch_load(checkpoint_path: Path, checkpoint_verified: bool) -> Any:
    """Prefer weights_only loading, falling back only after manifest verification."""

    try:
        return torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    except TypeError:
        if not checkpoint_verified:
            raise RuntimeError("weights_only torch.load unsupported and checkpoint is not verified; refusing fallback.")
        return torch.load(checkpoint_path, map_location="cpu")
    except Exception as exc:
        if not checkpoint_verified:
            raise RuntimeError("weights_only torch.load failed and checkpoint is not verified; refusing fallback.") from exc
        try:
            return torch.load(checkpoint_path, map_location="cpu")
        except Exception:
            raise exc


def papagei_checkpoint_metadata(root: Path, config: dict[str, Any]) -> dict[str, Any]:
    """Return checkpoint availability metadata without loading a model."""

    checkpoint_path = resolve_from_root(root, config["encoder"].get("checkpoint_path"))
    manifest_path = root / "evaluation" / "SigD" / "metadata" / "papagei_checkpoint_manifest.json"
    manifest = load_json(manifest_path) if manifest_path.exists() else {}
    return {
        "checkpoint_path": str(checkpoint_path) if checkpoint_path else None,
        "checkpoint_available": bool(checkpoint_path and checkpoint_path.exists() and checkpoint_path.is_file()),
        "checkpoint_md5": manifest.get("observed_md5"),
        "checkpoint_sha256": manifest.get("sha256"),
        "checkpoint_verified": bool(manifest.get("verified")),
    }
