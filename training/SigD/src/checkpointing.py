"""Checkpoint and early stopping helpers for SigD adaptation."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from common import ensure_dir, write_json


class EarlyStopping:
    """Track best validation metric with patience."""

    def __init__(self, patience: int, min_delta: float = 0.0, mode: str = "min") -> None:
        if mode not in {"min", "max"}:
            raise ValueError("mode must be min or max")
        self.patience = int(patience)
        self.min_delta = float(min_delta)
        self.mode = mode
        self.best: float | None = None
        self.best_epoch: int | None = None
        self.bad_epochs = 0

    def update(self, value: float, epoch: int) -> bool:
        """Return True when value improves."""

        if self.best is None:
            improved = True
        elif self.mode == "min":
            improved = value < self.best - self.min_delta
        else:
            improved = value > self.best + self.min_delta
        if improved:
            self.best = float(value)
            self.best_epoch = int(epoch)
            self.bad_epochs = 0
            return True
        self.bad_epochs += 1
        return False

    @property
    def should_stop(self) -> bool:
        return self.bad_epochs >= self.patience

    def state_dict(self) -> dict[str, Any]:
        return {
            "patience": self.patience,
            "min_delta": self.min_delta,
            "mode": self.mode,
            "best": self.best,
            "best_epoch": self.best_epoch,
            "bad_epochs": self.bad_epochs,
            "should_stop": self.should_stop,
        }


def save_projection_checkpoint(path: Path, model: torch.nn.Module, metadata: dict[str, Any], overwrite: bool = True) -> None:
    """Save projection head weights and metadata."""

    if path.exists() and not overwrite:
        raise FileExistsError(f"Checkpoint exists: {path}")
    ensure_dir(path.parent)
    torch.save(
        {
            "projection_head_state_dict": model.projection_head.state_dict(),
            "morphology_heads_state_dict": model.morphology_heads.state_dict() if getattr(model, "morphology_heads", None) is not None else None,
            "metadata": metadata,
        },
        path,
    )


def load_projection_checkpoint(path: Path, model: torch.nn.Module) -> dict[str, Any]:
    """Load projection head weights into an initialized model."""

    payload = torch.load(path, map_location="cpu", weights_only=False)
    model.projection_head.load_state_dict(payload["projection_head_state_dict"], strict=True)
    morphology_state = payload.get("morphology_heads_state_dict")
    if morphology_state is not None:
        if getattr(model, "morphology_heads", None) is None:
            raise RuntimeError("Checkpoint contains morphology heads but model config does not enable them.")
        model.morphology_heads.load_state_dict(morphology_state, strict=True)
    elif getattr(model, "morphology_heads", None) is not None:
        raise RuntimeError("Model expects morphology heads but checkpoint does not contain them.")
    return dict(payload.get("metadata", {}))


def write_manifest(path: Path, payload: dict[str, Any]) -> None:
    """Write a checkpoint/run manifest."""

    write_json(path, payload)
