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
from positive_masks import build_cross_session_positive_mask, build_generic_supcon_positive_mask  # noqa: E402
from supervised_contrastive_loss import supervised_contrastive_loss  # noqa: E402


def ids():
    subjects = []
    sessions = []
    for subject in range(8):
        for session in ("s1", "s2"):
            for _ in range(2):
                subjects.append(f"p{subject}")
                sessions.append(session)
    return subjects, sessions


def one_step(mask_builder):
    config = minimal_config()
    model = PaPaGeiProjectionModel(Path(__file__).resolve().parents[3], config, backbone_adapter=FakeBackbone())
    before_backbone = [p.detach().clone() for p in model.backbone.parameters()]
    before_projection = [p.detach().clone() for p in model.projection_head.parameters()]
    waveforms = torch.randn(32, 1, 1250)
    subjects, sessions = ids()
    embeddings = model.encode(waveforms)
    loss = supervised_contrastive_loss(embeddings, mask_builder(subjects, sessions))
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1.0e-3)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    backbone_delta = max((a - b).abs().max().item() for a, b in zip(before_backbone, model.backbone.parameters()))
    projection_delta = max((a - b).abs().max().item() for a, b in zip(before_projection, model.projection_head.parameters()))
    return loss, backbone_delta, projection_delta, model.trainable_parameter_count()


def test_generic_one_step_loss_finite_and_updates_projection_only() -> None:
    loss, backbone_delta, projection_delta, _ = one_step(build_generic_supcon_positive_mask)
    assert torch.isfinite(loss).item()
    assert backbone_delta == 0.0
    assert projection_delta > 0.0


def test_cs_one_step_loss_finite_and_updates_projection_only() -> None:
    loss, backbone_delta, projection_delta, _ = one_step(build_cross_session_positive_mask)
    assert torch.isfinite(loss).item()
    assert backbone_delta == 0.0
    assert projection_delta > 0.0


def test_e4_e5_trainable_parameter_count_identical() -> None:
    generic = one_step(build_generic_supcon_positive_mask)[3]
    cs = one_step(build_cross_session_positive_mask)[3]
    assert generic == cs


def test_morphology_values_not_required_for_loss() -> None:
    subjects, sessions = ids()
    embeddings = torch.randn(32, 128, requires_grad=True)
    mask = build_cross_session_positive_mask(subjects, sessions)
    loss = supervised_contrastive_loss(embeddings, mask)
    loss.backward()
    assert embeddings.grad is not None


def test_cached_training_does_not_update_or_call_backbone() -> None:
    config = minimal_config()
    fake = FakeBackbone()
    model = PaPaGeiProjectionModel(Path(__file__).resolve().parents[3], config, backbone_adapter=fake)
    backbone_before = [p.detach().clone() for p in model.backbone.parameters()]
    cached_backbone = torch.randn(32, 512)
    subjects, sessions = ids()
    embeddings = model.project(cached_backbone)
    loss = supervised_contrastive_loss(embeddings, build_cross_session_positive_mask(subjects, sessions))
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1.0e-3)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    assert fake.call_count == 0
    assert max((a - b).abs().max().item() for a, b in zip(backbone_before, model.backbone.parameters())) == 0.0
