"""Validation-only lambda selector for E6 alignment runs."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

from common import load_csv_rows, load_json, utc_now_iso, write_json


def select_alignment_candidate(
    *,
    experiment_family: str,
    candidate_roots: Iterable[Path],
    output_path: Path | None = None,
    tie_tolerance: float = 1.0e-12,
) -> dict[str, Any]:
    """Select an alignment candidate without reading any test artifacts."""

    candidates: list[dict[str, Any]] = []
    test_artifacts_seen: list[str] = []
    for root in candidate_roots:
        root = Path(root)
        test_artifacts_seen.extend(str(path) for path in _existing_test_artifacts(root))
        manifest = load_json(root / "manifest.json")
        rows = load_csv_rows(root / "validation_history.csv")
        if not rows:
            raise ValueError(f"No validation rows found: {root}")
        lambda_align = _candidate_lambda(manifest, rows)
        best_row = min(rows, key=lambda row: (_float(row.get("validation_exhaustive_eer")), int(row.get("epoch", 0))))
        candidates.append(
            {
                "result_root": str(root),
                "experiment_id": manifest.get("experiment_id"),
                "lambda_align": lambda_align,
                "best_epoch": int(best_row.get("epoch", manifest.get("best_epoch") or 0)),
                "validation_exhaustive_eer": _float(best_row.get("validation_exhaustive_eer")),
                "validation_tar_at_far_1pct": _optional_float(best_row.get("validation_tar_at_far_1pct")),
                "checkpoint_path": str(root / "checkpoints" / "best_projection_head.pt"),
            }
        )
    if not candidates:
        raise ValueError("No candidate roots were provided.")
    selected = _select(candidates, tie_tolerance)
    payload = {
        "generated_datetime_utc": utc_now_iso(),
        "experiment_family": experiment_family,
        "candidate_lambdas": [row["lambda_align"] for row in candidates],
        "candidates": candidates,
        "selection_metric": "validation_exhaustive_eer",
        "tie_breaker": [
            "lower_validation_exhaustive_eer",
            "higher_validation_tar_at_far_1pct",
            "smaller_lambda_align",
        ],
        "selected_lambda": selected["lambda_align"],
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


def _select(candidates: list[dict[str, Any]], tie_tolerance: float) -> dict[str, Any]:
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
    return float(candidate["lambda_align"]) < float(best["lambda_align"])


def _candidate_lambda(manifest: dict[str, Any], rows: list[dict[str, str]]) -> float:
    if manifest.get("lambda_align") is not None:
        return float(manifest["lambda_align"])
    if manifest.get("loss_components", {}).get("session_centroid_alignment_weight") is not None:
        return float(manifest["loss_components"]["session_centroid_alignment_weight"])
    for row in rows:
        if row.get("lambda_align") not in {None, ""}:
            return float(row["lambda_align"])
    return 0.0


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

