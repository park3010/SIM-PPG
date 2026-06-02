from __future__ import annotations

from pathlib import Path
import sys

SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))

from morphology_selector import select_morphology_candidate  # noqa: E402


def write_candidate(root: Path, name: str, lambda_svri: float, lambda_sqi: float, eer: float, tar: float | None = None, with_test: bool = False) -> Path:
    root.mkdir(parents=True)
    (root / "checkpoints").mkdir()
    (root / "checkpoints" / "best_projection_head.pt").write_bytes(b"fake")
    (root / "manifest.json").write_text(
        "{"
        f'"experiment_id":"candidate","experiment_stage":"E7_A","candidate_name":"{name}",'
        f'"lambda_svri":{lambda_svri},"lambda_sqi":{lambda_sqi}'
        "}",
        encoding="utf-8",
    )
    tar_value = "" if tar is None else str(tar)
    (root / "validation_history.csv").write_text(
        "epoch,validation_exhaustive_eer,validation_tar_at_far_1pct,lambda_svri,lambda_sqi\n"
        f"1,{eer},{tar_value},{lambda_svri},{lambda_sqi}\n",
        encoding="utf-8",
    )
    if with_test:
        (root / "test_scores.csv").write_text("trial_id,score\n", encoding="utf-8")
    return root


def test_lowest_validation_eer_selected() -> None:
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        roots = [
            write_candidate(root / "a", "a", 0.01, 0.01, 0.4),
            write_candidate(root / "b", "b", 0.05, 0.05, 0.3),
        ]
        payload = select_morphology_candidate(experiment_family="fam", candidate_roots=roots)
        assert payload["selected_candidate_name"] == "b"


def test_tie_breaker_higher_tar_then_lower_weight() -> None:
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        roots = [
            write_candidate(root / "a", "a", 0.10, 0.10, 0.3, 0.2),
            write_candidate(root / "b", "b", 0.01, 0.01, 0.3, 0.2),
        ]
        payload = select_morphology_candidate(experiment_family="fam", candidate_roots=roots)
        assert payload["selected_candidate_name"] == "b"


def test_selector_does_not_read_test_artifacts() -> None:
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        candidate = write_candidate(root / "a", "a", 0.01, 0.01, 0.3, 0.2, with_test=True)
        payload = select_morphology_candidate(experiment_family="fam", candidate_roots=[candidate])
        assert payload["test_data_accessed"] is False
        assert payload["test_artifacts_present_but_not_read"]


def test_selected_checkpoint_path_exists() -> None:
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        candidate = write_candidate(root / "a", "a", 0.01, 0.01, 0.3, 0.2)
        payload = select_morphology_candidate(experiment_family="fam", candidate_roots=[candidate])
        assert Path(payload["selected_checkpoint_path"]).exists()

