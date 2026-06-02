from __future__ import annotations

from pathlib import Path
import sys

SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))

from e8_selector import select_e8_candidate  # noqa: E402


def write_candidate(root: Path, name: str, eer: float, tar: float, with_test: bool = False) -> Path:
    root.mkdir(parents=True)
    (root / "checkpoints").mkdir()
    (root / "checkpoints" / "best_projection_head.pt").write_bytes(b"fake")
    (root / "manifest.json").write_text(
        "{"
        f'"experiment_id":"e8","candidate_name":"{name}",'
        f'"loss_components":{{"candidate_name":"{name}","sqi_weighting_mode":"{name}"}}'
        "}",
        encoding="utf-8",
    )
    (root / "validation_history.csv").write_text(
        "epoch,validation_exhaustive_eer,validation_tar_at_far_1pct\n"
        f"1,{eer},{tar}\n",
        encoding="utf-8",
    )
    if with_test:
        (root / "test_metrics.json").write_text("{}", encoding="utf-8")
    return root


def test_no_candidate_beats_e7_selects_reference(tmp_path: Path) -> None:
    root = write_candidate(tmp_path / "c1", "sqi_mild_linear", 0.40, 0.1)
    payload = select_e8_candidate(candidate_roots=[root], e7_reference_path=None, fallback_validation_eer=0.35)
    assert payload["selected_model"] == "E7_A_REFERENCE"
    assert payload["final_e8_test_evaluation_allowed"] is False


def test_candidate_beats_e7_selected(tmp_path: Path) -> None:
    root = write_candidate(tmp_path / "c1", "sqi_mild_linear", 0.30, 0.1)
    payload = select_e8_candidate(candidate_roots=[root], e7_reference_path=None, fallback_validation_eer=0.35)
    assert payload["selected_model"] == "E8_CANDIDATE"
    assert payload["selected_candidate_name"] == "sqi_mild_linear"
    assert payload["final_e8_test_evaluation_allowed"] is True


def test_tie_breaker_uses_tar_then_priority(tmp_path: Path) -> None:
    roots = [
        write_candidate(tmp_path / "c1", "sqi_strong_linear", 0.30, 0.2),
        write_candidate(tmp_path / "c2", "sqi_mild_linear", 0.30, 0.2),
    ]
    payload = select_e8_candidate(candidate_roots=roots, e7_reference_path=None, fallback_validation_eer=0.35)
    assert payload["selected_candidate_name"] == "sqi_mild_linear"


def test_selector_does_not_read_test_artifacts(tmp_path: Path) -> None:
    root = write_candidate(tmp_path / "c1", "sqi_mild_linear", 0.30, 0.1, with_test=True)
    payload = select_e8_candidate(candidate_roots=[root], e7_reference_path=None, fallback_validation_eer=0.35)
    assert payload["test_data_accessed"] is False
    assert payload["test_artifacts_present_but_not_read"]

