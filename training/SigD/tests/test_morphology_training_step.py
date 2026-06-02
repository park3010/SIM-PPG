from __future__ import annotations

from pathlib import Path
import sys

import torch

SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))
TESTS = Path(__file__).resolve().parent
sys.path.insert(0, str(TESTS))

from helpers import FakeBackbone, minimal_config  # noqa: E402
from morphology_objective import compute_total_loss_with_morphology  # noqa: E402
from papagei_projection_model import PaPaGeiProjectionModel  # noqa: E402


def ids():
    subjects = []
    sessions = []
    for subject in range(8):
        for session in ("s1", "s2"):
            for _ in range(2):
                subjects.append(f"p{subject}")
                sessions.append(session)
    return subjects, sessions


def cfg(stage: str) -> dict:
    config = minimal_config()
    config["model"]["morphology_heads"] = {"enabled": True, "input_dim": 128, "hidden_dim": 64, "targets": ["svri", "sqi"]}
    config["training"].update(
        {
            "loss": "supervised_contrastive_with_morphology",
            "positive_mask_mode": "same_subject_different_sample",
            "sampler_mode": "same_subject_any_session" if stage == "e7_a" else "same_subject_cross_session",
            "temperature": 0.07,
        }
    )
    config["loss_components"] = {
        "supervised_contrastive_weight": 1.0,
        "session_centroid_alignment_weight": 0.0,
        "morphology_weight": 1.0,
        "lambda_svri": 0.05,
        "lambda_sqi": 0.05,
        "use_ipa": False,
        "sqi_weighting_enabled": False,
    }
    return config


def batch():
    return {
        "subject_ids": ids()[0],
        "session_ids": ids()[1],
        "backbone_embeddings": torch.randn(32, 512),
        "svri": torch.randn(32),
        "sqi": torch.randn(32),
        "svri_valid_mask": torch.ones(32, dtype=torch.bool),
        "sqi_valid_mask": torch.ones(32, dtype=torch.bool),
        "ipa": torch.full((32,), float("nan")),
        "ipa_valid_mask": torch.zeros(32, dtype=torch.bool),
    }


def run_step(stage: str):
    config = cfg(stage)
    fake = FakeBackbone()
    model = PaPaGeiProjectionModel(Path(__file__).resolve().parents[3], config, backbone_adapter=fake)
    before_backbone = [p.detach().clone() for p in model.backbone.parameters()]
    before_projection = [p.detach().clone() for p in model.projection_head.parameters()]
    before_morphology = [p.detach().clone() for p in model.morphology_heads.parameters()]
    item = batch()
    cached_before = item["backbone_embeddings"].clone()
    embeddings = model.project(item["backbone_embeddings"])
    predictions = model.predict_morphology(embeddings)
    loss, diagnostics = compute_total_loss_with_morphology(
        embeddings,
        item["subject_ids"],
        item["session_ids"],
        predictions,
        item,
        config,
    )
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1.0e-3)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    backbone_delta = max((a - b).abs().max().item() for a, b in zip(before_backbone, model.backbone.parameters()))
    projection_delta = max((a - b).abs().max().item() for a, b in zip(before_projection, model.projection_head.parameters()))
    morphology_delta = max((a - b).abs().max().item() for a, b in zip(before_morphology, model.morphology_heads.parameters()))
    cached_delta = float((cached_before - item["backbone_embeddings"]).abs().max().item())
    return loss, diagnostics, backbone_delta, projection_delta, morphology_delta, cached_delta, fake.call_count, model


def test_e7_a_one_step_loss_finite() -> None:
    loss, *_ = run_step("e7_a")
    assert torch.isfinite(loss).item()


def test_e7_b_one_step_loss_finite() -> None:
    loss, *_ = run_step("e7_b")
    assert torch.isfinite(loss).item()


def test_projection_and_morphology_update_but_backbone_unchanged() -> None:
    _, _, backbone_delta, projection_delta, morphology_delta, cached_delta, call_count, model = run_step("e7_a")
    assert projection_delta > 0
    assert morphology_delta > 0
    assert backbone_delta == 0.0
    assert cached_delta == 0.0
    assert call_count == 0
    assert model.trainable_parameter_count() > 164736


def test_morphology_heads_not_used_in_verification_projection_path() -> None:
    _, _, _, _, _, _, _, model = run_step("e7_b")
    embeddings = model.project(torch.randn(4, 512))
    assert embeddings.shape == (4, 128)

