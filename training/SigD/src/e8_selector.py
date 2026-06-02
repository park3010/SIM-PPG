"""Validation-only selector for E8 SQI-weighted candidates."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

from common import load_csv_rows, load_json, utc_now_iso, write_json


MODE_PRIORITY = {
    "sqi_mild_linear": 0,
    "sqi_clipped_linear": 1,
    "sqi_rank_bottom20_downweight": 2,
    "sqi_strong_linear": 3,
}


def select_e8_candidate(
    *,
    candidate_roots: Iterable[Path],
    e7_reference_path: Path | None,
    output_path: Path | None = None,
    fallback_validation_eer: float = 0.3542762284196547,
    tie_tolerance: float = 1.0e-12,
) -> dict[str, Any]:
    """Select E8 only if a candidate beats E7-A validation EER."""

    reference = _load_e7_reference(e7_reference_path, fallback_validation_eer)
    candidates = [_load_candidate(Path(root)) for root in candidate_roots]
    test_artifacts_seen = [artifact for row in candidates for artifact in row.pop("test_artifacts_present")]
    improving = [
        row
        for row in candidates
        if float(row["validation_exhaustive_eer"]) < float(reference["e7_a_reference_validation_eer"]) - tie_tolerance
    ]
    if not improving:
        selected = {
            "selected_model": "E7_A_REFERENCE",
            "selected_candidate_name": None,
            "selected_validation_eer": reference["e7_a_reference_validation_eer"],
            "selected_checkpoint_path": reference.get("e7_a_reference_checkpoint"),
            "e8_improves_e7_a": False,
            "final_e8_test_evaluation_allowed": False,
            "final_test_evaluation_pending": False,
            "recommendation": "Keep E7-A as final seed42 model",
        }
    else:
        best = _select_best(improving, tie_tolerance)
        selected = {
            "selected_model": "E8_CANDIDATE",
            "selected_candidate_name": best["candidate_name"],
            "selected_validation_eer": best["validation_exhaustive_eer"],
            "selected_checkpoint_path": best["checkpoint_path"],
            "e8_improves_e7_a": True,
            "final_e8_test_evaluation_allowed": True,
            "final_test_evaluation_pending": True,
            "recommendation": "E8 candidate may proceed to one frozen final test evaluation",
        }
    payload = {
        "generated_datetime_utc": utc_now_iso(),
        **reference,
        "candidates": candidates,
        **selected,
        "test_data_accessed": False,
        "test_artifacts_present_but_not_read": test_artifacts_seen,
    }
    if output_path is not None:
        write_json(output_path, payload)
    return payload


def _load_e7_reference(path: Path | None, fallback_validation_eer: float) -> dict[str, Any]:
    if path is not None and path.exists():
        payload = load_json(path)
        return {
            "e7_a_reference_validation_eer": float(
                payload.get("selected_validation_eer")
                or payload.get("e7_a_reference_validation_eer")
                or fallback_validation_eer
            ),
            "e7_a_reference_checkpoint": payload.get("selected_checkpoint_path") or payload.get("e7_a_reference_checkpoint"),
        }
    return {
        "e7_a_reference_validation_eer": float(fallback_validation_eer),
        "e7_a_reference_checkpoint": None,
    }


def _load_candidate(root: Path) -> dict[str, Any]:
    manifest = load_json(root / "manifest.json")
    rows = load_csv_rows(root / "validation_history.csv")
    if not rows:
        raise ValueError(f"No validation rows found: {root}")
    best = min(rows, key=lambda row: (_float(row.get("validation_exhaustive_eer")), int(row.get("epoch", 0))))
    candidate_name = str(manifest.get("candidate_name") or manifest.get("loss_components", {}).get("candidate_name"))
    return {
        "candidate_name": candidate_name,
        "sqi_weighting_mode": manifest.get("sqi_weighting_mode") or manifest.get("loss_components", {}).get("sqi_weighting_mode"),
        "result_root": str(root),
        "validation_exhaustive_eer": _float(best.get("validation_exhaustive_eer")),
        "validation_tar_at_far_1pct": _optional_float(best.get("validation_tar_at_far_1pct")),
        "checkpoint_path": str(root / "checkpoints" / "best_projection_head.pt"),
        "test_artifacts_present": [str(path) for path in _existing_test_artifacts(root)],
    }


def _select_best(candidates: list[dict[str, Any]], tie_tolerance: float) -> dict[str, Any]:
    best = candidates[0]
    for candidate in candidates[1:]:
        if _is_better(candidate, best, tie_tolerance):
            best = candidate
    return best


def _is_better(candidate: dict[str, Any], best: dict[str, Any], tie_tolerance: float) -> bool:
    cand_eer = float(candidate["validation_exhaustive_eer"])
    best_eer = float(best["validation_exhaustive_eer"])
    if cand_eer < best_eer - tie_tolerance:
        return True
    if cand_eer > best_eer + tie_tolerance:
        return False
    cand_tar = candidate.get("validation_tar_at_far_1pct")
    best_tar = best.get("validation_tar_at_far_1pct")
    cand_tar_num = float(cand_tar) if cand_tar is not None else float("-inf")
    best_tar_num = float(best_tar) if best_tar is not None else float("-inf")
    if cand_tar_num > best_tar_num + tie_tolerance:
        return True
    if cand_tar_num < best_tar_num - tie_tolerance:
        return False
    return MODE_PRIORITY.get(str(candidate["candidate_name"]), 99) < MODE_PRIORITY.get(str(best["candidate_name"]), 99)


def _existing_test_artifacts(root: Path) -> list[Path]:
    names = [
        "test_metrics.json",
        "test_scores.csv",
        "final_exhaustive_evaluation/test_metrics.json",
        "final_exhaustive_evaluation/test_scores.csv",
    ]
    return [root / name for name in names if (root / name).exists()]


def _float(value: Any) -> float:
    parsed = float(value)
    if parsed != parsed:
        raise ValueError("NaN validation value is not selectable.")
    return parsed


def _optional_float(value: Any) -> float | None:
    if value in {None, ""}:
        return None
    return float(value)

