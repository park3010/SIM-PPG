#!/usr/bin/env python3
"""Summarize final E4/E7-A multi-seed exhaustive evaluation results."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
from pathlib import Path
from typing import Any

SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))

from common import detect_project_root, ensure_dir, load_csv_rows, load_json, write_json  # noqa: E402

DEFAULT_SEEDS = "42 52 123 777 2026"
E4_MODEL = "E4_Generic_SupCon"
E7A_MODEL = "E7_A_SIM_PPG"
E7A_CANDIDATE = "svri0p05_sqi0p05"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=None)
    parser.add_argument("--seeds", default=DEFAULT_SEEDS)
    parser.add_argument("--output-prefix", default="multiseed_final_summary_seed42_52_123_777_2026")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = detect_project_root(args.root)
    seeds = [int(item) for item in str(args.seeds).split()]
    metadata_dir = ensure_dir(root / "training" / "SigD" / "metadata")
    rows: list[dict[str, Any]] = []
    time_gap_rows: list[dict[str, Any]] = []
    missing: list[str] = []
    for seed in seeds:
        for model in [E4_MODEL, E7A_MODEL]:
            result_root, legacy_eval_root = result_roots(root, model, seed)
            try:
                rows.append(read_seed_result(model, seed, result_root, legacy_eval_root))
            except FileNotFoundError as exc:
                missing.append(str(exc))
                continue
            time_gap_rows.extend(read_time_gap_rows(model, seed, result_root, legacy_eval_root))
    aggregate = aggregate_rows(rows)
    pairwise = paired_deltas(rows)
    payload = {
        "seeds_requested": seeds,
        "models": [E4_MODEL, E7A_MODEL],
        "e7_a_fixed_candidate": E7A_CANDIDATE,
        "e8_excluded_reason": "validation_selector_did_not_beat_e7_a",
        "missing_results": missing,
        "per_seed": rows,
        "aggregate": aggregate,
        "paired_deltas": pairwise,
    }
    csv_path = metadata_dir / f"{args.output_prefix}.csv"
    json_path = metadata_dir / f"{args.output_prefix}.json"
    time_gap_path = metadata_dir / "multiseed_time_gap_summary_seed42_52_123_777_2026.csv"
    write_rows(csv_path, rows)
    write_json(json_path, payload)
    write_rows(time_gap_path, time_gap_rows)
    print(
        f"multiseed_summary_written=True rows={len(rows)} missing={len(missing)} "
        f"json={json_path}"
    )
    return 0


def result_roots(root: Path, model: str, seed: int) -> tuple[Path, Path | None]:
    if model == E4_MODEL:
        return root / f"training/SigD/results/papagei_s_generic_supcon_head_only/seed{seed}", None
    result_root = root / f"training/SigD/results/papagei_s_e7_a_generic_supcon_morph_e4_branch/{E7A_CANDIDATE}/seed{seed}"
    legacy = root / "training/SigD/results/papagei_s_e7_a_generic_supcon_morph_e4_branch/seed42" if seed == 42 else None
    return result_root, legacy


def read_seed_result(model: str, seed: int, result_root: Path, legacy_eval_root: Path | None) -> dict[str, Any]:
    manifest_path = result_root / "manifest.json"
    history_path = result_root / "validation_history.csv"
    eval_root = eval_root_for(result_root, legacy_eval_root)
    metrics_path = eval_root / "test_metrics.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Missing manifest: {manifest_path}")
    if not history_path.exists():
        raise FileNotFoundError(f"Missing validation_history: {history_path}")
    if not metrics_path.exists():
        raise FileNotFoundError(f"Missing test_metrics: {metrics_path}")
    manifest = load_json(manifest_path)
    metrics = load_json(metrics_path)
    history = load_csv_rows(history_path)
    best = best_validation_row(history)
    far1 = metrics["validation_fixed_far_1pct_threshold"]
    eer_thr = metrics["validation_fixed_eer_threshold"]
    checkpoint = result_root / "checkpoints" / "best_projection_head.pt"
    return {
        "model": model,
        "seed": seed,
        "result_root": relative(result_root),
        "evaluation_root": relative(eval_root),
        "legacy_seed42_evaluation_reused": bool(
            legacy_eval_root and eval_root == legacy_eval_root / "final_exhaustive_evaluation"
        ),
        "experiment_id": manifest.get("experiment_id"),
        "candidate_name": manifest.get("candidate_name"),
        "best_epoch": int(best.get("epoch", manifest.get("best_epoch") or 0)),
        "best_validation_eer": float(best.get("validation_exhaustive_eer", manifest.get("best_validation_exhaustive_eer"))),
        "test_roc_auc": float(metrics["roc_auc"]),
        "test_diagnostic_eer": float(metrics["diagnostic_eer"]["eer"]),
        "test_tar_at_val_far_1pct": float(far1["tar"]),
        "test_far_at_val_far_1pct": float(far1["far"]),
        "test_tar_at_val_eer_threshold": float(eer_thr["tar"]),
        "test_far_at_val_eer_threshold": float(eer_thr["far"]),
        "checkpoint_path": relative(checkpoint),
    }


def read_time_gap_rows(model: str, seed: int, result_root: Path, legacy_eval_root: Path | None) -> list[dict[str, Any]]:
    eval_root = eval_root_for(result_root, legacy_eval_root)
    path = eval_root / "test_time_gap_metrics.csv"
    if not path.exists():
        return []
    rows = []
    for row in load_csv_rows(path):
        enriched = {"model": model, "seed": seed}
        enriched.update(row)
        rows.append(enriched)
    return rows


def eval_root_for(result_root: Path, legacy_eval_root: Path | None) -> Path:
    candidate = result_root / "final_exhaustive_evaluation"
    if candidate.exists():
        return candidate
    if legacy_eval_root is not None and (legacy_eval_root / "final_exhaustive_evaluation").exists():
        return legacy_eval_root / "final_exhaustive_evaluation"
    return candidate


def best_validation_row(rows: list[dict[str, str]]) -> dict[str, str]:
    if not rows:
        return {}
    return min(rows, key=lambda row: float(row["validation_exhaustive_eer"]))


def aggregate_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    metrics = [
        "test_roc_auc",
        "test_diagnostic_eer",
        "test_tar_at_val_far_1pct",
        "test_far_at_val_far_1pct",
        "best_validation_eer",
    ]
    payload: dict[str, Any] = {}
    for model in sorted({row["model"] for row in rows}):
        model_rows = [row for row in rows if row["model"] == model]
        payload[model] = {"seed_count": len(model_rows), "seeds": [row["seed"] for row in model_rows]}
        for metric in metrics:
            values = [float(row[metric]) for row in model_rows]
            payload[model][f"{metric}_mean"] = mean(values)
            payload[model][f"{metric}_std"] = std(values)
    return payload


def paired_deltas(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_key = {(row["model"], int(row["seed"])): row for row in rows}
    seeds = sorted({int(row["seed"]) for row in rows})
    deltas: list[dict[str, Any]] = []
    for seed in seeds:
        e4 = by_key.get((E4_MODEL, seed))
        e7a = by_key.get((E7A_MODEL, seed))
        if not e4 or not e7a:
            continue
        deltas.append(
            {
                "seed": seed,
                "delta_eer_e7a_minus_e4": float(e7a["test_diagnostic_eer"]) - float(e4["test_diagnostic_eer"]),
                "delta_auc_e7a_minus_e4": float(e7a["test_roc_auc"]) - float(e4["test_roc_auc"]),
                "delta_tar_far1_e7a_minus_e4": float(e7a["test_tar_at_val_far_1pct"]) - float(e4["test_tar_at_val_far_1pct"]),
            }
        )
    return {
        "paired_seed_count": len(deltas),
        "per_seed": deltas,
        "e7a_improves_eer_count": sum(1 for row in deltas if row["delta_eer_e7a_minus_e4"] < 0),
        "e7a_improves_tar_far1_count": sum(1 for row in deltas if row["delta_tar_far1_e7a_minus_e4"] > 0),
    }


def write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    ensure_dir(path.parent)
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def mean(values: list[float]) -> float | None:
    return float(statistics.mean(values)) if values else None


def std(values: list[float]) -> float | None:
    if not values:
        return None
    if len(values) == 1:
        return 0.0
    return float(statistics.stdev(values))


def relative(path: Path) -> str:
    try:
        root = detect_project_root(None)
        return str(path.relative_to(root))
    except Exception:
        return str(path)


if __name__ == "__main__":
    raise SystemExit(main())
