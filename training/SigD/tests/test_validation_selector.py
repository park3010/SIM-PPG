from __future__ import annotations

from pathlib import Path
import sys

SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))

from validation_selector import select_alignment_candidate  # noqa: E402


def write_candidate(root: Path, lambda_value: float, eer: float, tar: float | None = None, with_test_artifact: bool = False) -> Path:
    root.mkdir(parents=True)
    (root / "manifest.json").write_text(
        f'{{"experiment_id":"candidate","lambda_align":{lambda_value},"best_epoch":1}}',
        encoding="utf-8",
    )
    tar_value = "" if tar is None else str(tar)
    (root / "validation_history.csv").write_text(
        "epoch,validation_exhaustive_eer,validation_tar_at_far_1pct\n"
        f"1,{eer},{tar_value}\n",
        encoding="utf-8",
    )
    if with_test_artifact:
        (root / "test_metrics.json").write_text("{}", encoding="utf-8")
    return root


def test_lowest_validation_eer_lambda_selected(tmp_path: Path) -> None:
    roots = [
        write_candidate(tmp_path / "lambda_0p01/seed42", 0.01, 0.4, 0.1),
        write_candidate(tmp_path / "lambda_0p05/seed42", 0.05, 0.3, 0.1),
    ]
    payload = select_alignment_candidate(experiment_family="fam", candidate_roots=roots)
    assert payload["selected_lambda"] == 0.05


def test_eer_tie_uses_higher_tar(tmp_path: Path) -> None:
    roots = [
        write_candidate(tmp_path / "lambda_0p01/seed42", 0.01, 0.3, 0.1),
        write_candidate(tmp_path / "lambda_0p05/seed42", 0.05, 0.3, 0.2),
    ]
    payload = select_alignment_candidate(experiment_family="fam", candidate_roots=roots)
    assert payload["selected_lambda"] == 0.05


def test_second_tie_uses_smaller_lambda(tmp_path: Path) -> None:
    roots = [
        write_candidate(tmp_path / "lambda_0p20/seed42", 0.20, 0.3, 0.2),
        write_candidate(tmp_path / "lambda_0p05/seed42", 0.05, 0.3, 0.2),
    ]
    payload = select_alignment_candidate(experiment_family="fam", candidate_roots=roots)
    assert payload["selected_lambda"] == 0.05


def test_selector_does_not_read_test_artifacts(tmp_path: Path) -> None:
    root = write_candidate(tmp_path / "lambda_0p01/seed42", 0.01, 0.3, 0.1, with_test_artifact=True)
    payload = select_alignment_candidate(experiment_family="fam", candidate_roots=[root])
    assert payload["test_data_accessed"] is False
    assert payload["test_artifacts_present_but_not_read"]


def test_output_manifest_records_no_test_access(tmp_path: Path) -> None:
    root = write_candidate(tmp_path / "lambda_0p01/seed42", 0.01, 0.3, 0.1)
    output = tmp_path / "selection.json"
    payload = select_alignment_candidate(experiment_family="fam", candidate_roots=[root], output_path=output)
    assert output.exists()
    assert payload["test_data_accessed"] is False

