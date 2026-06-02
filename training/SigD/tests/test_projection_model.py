from __future__ import annotations

from pathlib import Path
import sys

import torch

SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))
TESTS = Path(__file__).resolve().parent
sys.path.insert(0, str(TESTS))

from helpers import FakeBackbone, minimal_config  # noqa: E402
from papagei_projection_model import PaPaGeiProjectionModel  # noqa: E402


def build_model() -> PaPaGeiProjectionModel:
    return PaPaGeiProjectionModel(Path(__file__).resolve().parents[3], minimal_config(), backbone_adapter=FakeBackbone())


def test_projection_model_output_shape_and_norm() -> None:
    model = build_model()
    output = model.encode(torch.randn(4, 1, 1250))
    assert output.shape == (4, 128)
    assert torch.allclose(output.norm(dim=1), torch.ones(4), atol=1.0e-5)


def test_backbone_frozen_projection_trainable() -> None:
    model = build_model()
    assert all(not parameter.requires_grad for parameter in model.backbone.parameters())
    assert any(parameter.requires_grad for parameter in model.projection_head.parameters())


def test_train_keeps_backbone_eval() -> None:
    model = build_model()
    model.train(True)
    assert model.projection_head.training is True
    assert model.backbone.training is False


def test_common_input_policy_and_provenance_retained() -> None:
    model = build_model()
    metadata = model.get_model_metadata()
    assert metadata["backbone_frozen"] is True
    assert metadata["input_setting"] == "common_input"
    assert metadata["official_native_preprocessing_reapplied"] is False
    assert metadata["output_embedding_dim"] == 128
