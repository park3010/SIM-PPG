from __future__ import annotations

from pathlib import Path
import sys

import torch

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


def minimal_config() -> dict:
    return {
        "experiment_id": "TEST_EXPERIMENT",
        "experiment_stage": "TEST",
        "seed": 42,
        "input": {
            "input_protocol_id": "COMMON_PPG_10S_125HZ_V1",
            "final_protocol_id": "SIGD_COMMON_10S_K5M1_EARLIEST_ENROLL_LATER_PROBE_EXHAUSTIVE_EVAL_V2",
            "frozen_eval_config_reference": "evaluation/SigD/config/papagei_s_frozen_cosine_exhaustive_eval.yaml",
        },
        "model": {
            "backbone_embedding_dim": 512,
            "projection_head": {
                "input_dim": 512,
                "hidden_dim": 256,
                "output_dim": 128,
                "normalization_layer": "layer_norm",
                "activation": "relu",
                "dropout": 0.1,
                "output_l2_normalization": True,
            },
        },
        "training": {
            "temperature": 0.07,
            "positive_mask_mode": "same_subject_different_sample",
        },
    }


class FakeBackbone(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear = torch.nn.Linear(1250, 512)
        self.call_count = 0
        torch.nn.init.normal_(self.linear.weight, std=0.01)
        torch.nn.init.zeros_(self.linear.bias)

    def encode(self, waveforms: torch.Tensor) -> torch.Tensor:
        self.call_count += 1
        return self.linear(waveforms[:, 0, :])

    def get_encoder_metadata(self) -> dict:
        return {
            "checkpoint_sha256": "fake",
            "source_overlap_limitation": {"overlap_risk_present": True},
        }
