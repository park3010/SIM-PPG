from __future__ import annotations

from pathlib import Path
import sys

import pytest
import torch

SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))

from morphology_heads import MorphologyHeads  # noqa: E402


def config(targets=None):
    return {"input_dim": 128, "hidden_dim": 64, "targets": targets or ["svri", "sqi"]}


def test_morphology_head_outputs_shape_and_finite() -> None:
    heads = MorphologyHeads(config())
    outputs = heads(torch.randn(8, 128))
    assert set(outputs) == {"svri_pred", "sqi_pred"}
    assert outputs["svri_pred"].shape == (8,)
    assert outputs["sqi_pred"].shape == (8,)
    assert torch.isfinite(outputs["svri_pred"]).all()
    assert torch.isfinite(outputs["sqi_pred"]).all()


def test_morphology_head_parameter_count_positive() -> None:
    heads = MorphologyHeads(config())
    assert heads.parameter_count() > 0


def test_ipa_head_not_created_in_e7() -> None:
    with pytest.raises(ValueError, match="Unsupported|IPA"):
        MorphologyHeads(config(["svri", "ipa"]))

