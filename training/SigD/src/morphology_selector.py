"""Validation-only selector for E7 morphology candidates."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

from common import load_csv_rows, load_json, utc_now_iso, write_json


def select_morphology_candidate(
    *,
    experiment_family: str,
    candidate_roots: Iterable[Path],
    output_path: Path | None = None,
    prefer_branch: str | None = None,
    tie_tolerance: float = 1.0e-12,
) -> dict[str, Any]:
    """Select the best E7 candidate using validation histories only."""

    candidates: list[dict[str, Any]] = []
    test_artifacts_seen: list[str] = []
    for root in candidate_roots:
        root = Path(root)
        manifest = load_json(root / "manifest.json")
        rows = load_csv_rows(root / "validation_history.csv")
        if not rows:
            raise ValueError(f"No validation rows found: {root}")
        test_artifacts_seen.extend(str(path) for path in _existing_test_artifacts(root))
        best_row = min(rows, key=lambda row: (_float(row.get("validation_exhaustive_eer")), int(row.get("epoch", 0))))
        lambda_svri = _candidate_float(manifest, rows, "lambda_svri")
        lambda_sqi = _candidate_float(manifest, rows, "lambda_sqi")
        candidate_name = str(manifest.get("candidate_name") or _candidate_name(lambda_svri, lambda_sqi))
        candidates.append(
            {
                "result_root": str(root),
                "experiment_id": manifest.get("experiment_id"),
                "experiment_stage": manifest.get("experiment_stage"),
                "candidate_name": candidate_name,
                "lambda_svri": lambda_svri,
                "lambda_sqi": lambda_sqi,
                "total_morphology_weight": lambda_svri + lambda_sqi,
                "best_epoch": int(best_row.get("epoch", manifest.get("best_epoch") or 0)),
                "validation_exhaustive_eer": _float(best_row.get("validation_exhaustive_eer")),
                "validation_tar_at_far_1pct": _optional_float(best_row.get("validation_tar_at_far_1pct")),
                "checkpoint_path": str(root / "checkpoints" / "best_projection_head.pt"),
            }
        )
    if not candidates:
        raise ValueError("No morphology candidate roots were provided.")
    selected = _select(candidates, prefer_branch, tie_tolerance)
    payload = {
        "generated_datetime_utc": utc_now_iso(),
        "experiment_family": experiment_family,
        "candidates": candidates,
        "selection_metric": "validation_exhaustive_eer",
        "tie_breaker": [
            "lower_validation_exhaustive_eer",
            "higher_validation_tar_at_far_1pct",
            "lower_total_morphology_weight",
            "preferred_branch_if_exact_tie",
        ],
        "selected_candidate_name": selected["candidate_name"],
        "selected_lambda_svri": selected["lambda_svri"],
        "selected_lambda_sqi": selected["lambda_sqi"],
        "selected_best_epoch": selected["best_epoch"],
        "selected_validation_eer": selected["validation_exhaustive_eer"],
        "selected_validation_tar_at_far_1pct": selected["validation_tar_at_far_1pct"],
        "selected_checkpoint_path": selected["checkpoint_path"],
        "test_data_accessed": False,
        "test_artifacts_present_but_not_read": test_artifacts_seen,
        "final_test_evaluation_pending": True,
        "post_e4_e5_policy_applies": True,
    }
    if output_path is not None:
        write_json(output_path, payload)
    return payload


def _select(candidates: list[dict[str, Any]], prefer_branch: str | None, tie_tolerance: float) -> dict[str, Any]:
    best = candidates[0]
    for candidate in candidates[1:]:
        if _is_better(candidate, best, prefer_branch, tie_tolerance):
            best = candidate
    return best


def _is_better(candidate: dict[str, Any], best: dict[str, Any], prefer_branch: str | None, tie_tolerance: float) -> bool:
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
    if float(candidate["total_morphology_weight"]) < float(best["total_morphology_weight"]) - tie_tolerance:
        return True
    if float(candidate["total_morphology_weight"]) > float(best["total_morphology_weight"]) + tie_tolerance:
        return False
    if prefer_branch:
        return str(candidate.get("experiment_stage")) == prefer_branch and str(best.get("experiment_stage")) != prefer_branch
    return False


def _candidate_float(manifest: dict[str, Any], rows: list[dict[str, str]], key: str) -> float:
    if manifest.get(key) is not None:
        return float(manifest[key])
    if manifest.get("loss_components", {}).get(key) is not None:
        return float(manifest["loss_components"][key])
    for row in rows:
        if row.get(key) not in {None, ""}:
            return float(row[key])
    return 0.0


def _candidate_name(lambda_svri: float, lambda_sqi: float) -> str:
    return f"svri{lambda_svri:.2f}_sqi{lambda_sqi:.2f}".replace(".", "p")


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

