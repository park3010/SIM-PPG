#!/usr/bin/env python3
"""Audit final E4/E7-A multi-seed result completeness and guardrails."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))

from common import detect_project_root, load_json, write_json  # noqa: E402

DEFAULT_SEEDS = "42 52 123 777 2026"
E7A_CANDIDATE = "svri0p05_sqi0p05"
FINAL_PROTOCOL_ID = "SIGD_COMMON_10S_K5M1_EARLIEST_ENROLL_LATER_PROBE_EXHAUSTIVE_EVAL_V2"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=None)
    parser.add_argument("--seeds", default=DEFAULT_SEEDS)
    parser.add_argument("--allow-missing", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = detect_project_root(args.root)
    seeds = [int(item) for item in str(args.seeds).split()]
    entries: list[dict[str, Any]] = []
    errors: list[str] = []
    warnings: list[str] = []
    for seed in seeds:
        entries.append(audit_model_seed(root, "E4_Generic_SupCon", seed, args.allow_missing, errors, warnings))
        entries.append(audit_model_seed(root, "E7_A_SIM_PPG", seed, args.allow_missing, errors, warnings))
    summary_files = summary_file_status(root)
    if not args.allow_missing and not all(summary_files.values()):
        errors.append("Multi-seed summary files are missing; run summarize_multiseed_results.py.")
    checkpoint_hashes = sorted(
        {
            entry.get("backbone_checkpoint_sha256")
            for entry in entries
            if entry.get("present") and entry.get("backbone_checkpoint_sha256")
        }
    )
    protocol_ids = sorted({entry.get("final_protocol_id") for entry in entries if entry.get("present") and entry.get("final_protocol_id")})
    if len(checkpoint_hashes) > 1:
        errors.append(f"Multiple backbone checkpoint hashes found: {checkpoint_hashes}")
    if any(protocol_id != FINAL_PROTOCOL_ID for protocol_id in protocol_ids):
        errors.append(f"Unexpected final protocol IDs: {protocol_ids}")
    missing = [entry for entry in entries if not entry.get("present")]
    passed = not errors and (args.allow_missing or not missing)
    output = {
        "audit_id": "MULTISEED_FINAL_E4_E7A_AUDIT_V1",
        "seeds": seeds,
        "allow_missing": bool(args.allow_missing),
        "passed": bool(passed),
        "strict_ready": bool(not missing and not errors),
        "entries": entries,
        "missing_count": len(missing),
        "errors": errors,
        "warnings": warnings,
        "summary_files": summary_files,
        "checkpoint_hashes": checkpoint_hashes,
        "protocol_ids": protocol_ids,
        "e7_a_fixed_candidate": E7A_CANDIDATE,
        "e8_excluded": True,
        "e8_exclusion_reason": "E8 validation selector did not beat E7-A; final E8 test evaluation is forbidden.",
    }
    out_path = root / "training" / "SigD" / "metadata" / "multiseed_audit_summary_seed42_52_123_777_2026.json"
    write_json(out_path, output)
    print(
        f"multiseed_audit_passed={output['passed']} strict_ready={output['strict_ready']} "
        f"missing_count={output['missing_count']} output={out_path}"
    )
    return 0 if output["passed"] else 1


def audit_model_seed(
    root: Path,
    model: str,
    seed: int,
    allow_missing: bool,
    errors: list[str],
    warnings: list[str],
) -> dict[str, Any]:
    result_root, legacy_eval_root = result_roots(root, model, seed)
    eval_root = eval_root_for(result_root, legacy_eval_root)
    required = {
        "manifest": result_root / "manifest.json",
        "validation_history": result_root / "validation_history.csv",
        "best_checkpoint": result_root / "checkpoints" / "best_projection_head.pt",
        "eval_manifest": eval_root / "adapted_model_run_manifest.json",
        "test_metrics": eval_root / "test_metrics.json",
        "test_scores": eval_root / "test_scores.csv",
    }
    missing = [str(path) for path in required.values() if not path.exists()]
    entry: dict[str, Any] = {
        "model": model,
        "seed": seed,
        "result_root": str(result_root.relative_to(root)),
        "evaluation_root": str(eval_root.relative_to(root)),
        "legacy_seed42_evaluation_reused": bool(
            legacy_eval_root and eval_root == legacy_eval_root / "final_exhaustive_evaluation"
        ),
        "present": not missing,
        "missing": missing,
    }
    if missing:
        if not allow_missing:
            errors.append(f"Missing {model} seed{seed}: {missing[:3]}")
        return entry
    manifest = load_json(required["manifest"])
    eval_manifest = load_json(required["eval_manifest"])
    entry.update(
        {
            "training_test_accessed": bool(manifest.get("test_accessed_during_training") or manifest.get("test_data_read_during_training")),
            "validation_threshold_only": bool(eval_manifest.get("validation_threshold_only")),
            "test_threshold_tuning_performed": bool(eval_manifest.get("test_threshold_tuning_performed")),
            "final_protocol_id": eval_manifest.get("final_protocol_id"),
            "input_protocol_id": eval_manifest.get("input_protocol_id"),
            "backbone_checkpoint_sha256": manifest.get("model_metadata", {}).get("backbone_checkpoint_sha256")
            or eval_manifest.get("model_metadata", {}).get("backbone_checkpoint_sha256"),
        }
    )
    if entry["training_test_accessed"]:
        errors.append(f"Training manifest indicates test access for {model} seed{seed}.")
    if not entry["validation_threshold_only"]:
        errors.append(f"Evaluation manifest missing validation_threshold_only=true for {model} seed{seed}.")
    if entry["test_threshold_tuning_performed"]:
        errors.append(f"Evaluation manifest indicates test threshold tuning for {model} seed{seed}.")
    if model == "E7_A_SIM_PPG":
        candidate = manifest.get("candidate_name") or manifest.get("loss_components", {}).get("candidate_name")
        lambda_svri = float(manifest.get("lambda_svri", manifest.get("loss_components", {}).get("lambda_svri", -1.0)))
        lambda_sqi = float(manifest.get("lambda_sqi", manifest.get("loss_components", {}).get("lambda_sqi", -1.0)))
        entry.update(
            {
                "candidate_name": candidate,
                "lambda_svri": lambda_svri,
                "lambda_sqi": lambda_sqi,
                "sqi_weighting_enabled": bool(manifest.get("sqi_weighting_enabled", manifest.get("loss_components", {}).get("sqi_weighting_enabled", False))),
                "use_ipa": bool(manifest.get("use_ipa", manifest.get("loss_components", {}).get("use_ipa", False))),
            }
        )
        if candidate != E7A_CANDIDATE:
            errors.append(f"E7-A candidate is not fixed for seed{seed}: {candidate}")
        if abs(lambda_svri - 0.05) > 1e-12 or abs(lambda_sqi - 0.05) > 1e-12:
            errors.append(f"E7-A lambda mismatch for seed{seed}: svri={lambda_svri} sqi={lambda_sqi}")
        if entry["sqi_weighting_enabled"] or entry["use_ipa"]:
            errors.append(f"E7-A forbidden option enabled for seed{seed}.")
    else:
        morph = bool(manifest.get("morphology_heads_enabled", False))
        entry["morphology_heads_enabled"] = morph
        if morph:
            errors.append(f"E4 morphology heads should be disabled for seed{seed}.")
    if seed == 42 and model == "E7_A_SIM_PPG" and entry["legacy_seed42_evaluation_reused"]:
        warnings.append("Reused existing legacy seed42 E7-A final evaluation root; new seeds use candidate-specific roots.")
    return entry


def result_roots(root: Path, model: str, seed: int) -> tuple[Path, Path | None]:
    if model == "E4_Generic_SupCon":
        return root / f"training/SigD/results/papagei_s_generic_supcon_head_only/seed{seed}", None
    result_root = root / f"training/SigD/results/papagei_s_e7_a_generic_supcon_morph_e4_branch/{E7A_CANDIDATE}/seed{seed}"
    legacy = root / "training/SigD/results/papagei_s_e7_a_generic_supcon_morph_e4_branch/seed42" if seed == 42 else None
    return result_root, legacy


def eval_root_for(result_root: Path, legacy_eval_root: Path | None) -> Path:
    candidate = result_root / "final_exhaustive_evaluation"
    if candidate.exists():
        return candidate
    if legacy_eval_root is not None and (legacy_eval_root / "final_exhaustive_evaluation").exists():
        return legacy_eval_root / "final_exhaustive_evaluation"
    return candidate


def summary_file_status(root: Path) -> dict[str, bool]:
    meta = root / "training" / "SigD" / "metadata"
    return {
        "summary_csv": (meta / "multiseed_final_summary_seed42_52_123_777_2026.csv").exists(),
        "summary_json": (meta / "multiseed_final_summary_seed42_52_123_777_2026.json").exists(),
        "time_gap_csv": (meta / "multiseed_time_gap_summary_seed42_52_123_777_2026.csv").exists(),
    }


if __name__ == "__main__":
    raise SystemExit(main())
