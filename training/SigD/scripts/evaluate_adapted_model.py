#!/usr/bin/env python3
"""Evaluate a trained projection head on exhaustive validation/test protocol."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))

from adaptation_evaluator import evaluate_final  # noqa: E402
from checkpointing import load_projection_checkpoint  # noqa: E402
from common import detect_project_root, ensure_dir, load_training_config, rewrite_result_root_seed, utc_now_iso, write_json  # noqa: E402
from papagei_projection_model import PaPaGeiProjectionModel  # noqa: E402
from trainer import resolve_training_result_root  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=None)
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--candidate-name", default=None)
    parser.add_argument("--effective-result-root", default=None)
    parser.add_argument("--lambda-svri", type=float, default=None)
    parser.add_argument("--lambda-sqi", type=float, default=None)
    parser.add_argument("--sqi-weighting-mode", default=None)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = detect_project_root(args.root)
    config = load_training_config(root, args.config)
    runtime = {
        "base_config_path": str(args.config),
        "multi_seed_run": args.seed is not None,
        "fixed_e7a_candidate": False,
        "exact_result_root": False,
    }
    if args.seed is not None:
        config["seed"] = int(args.seed)
        config.setdefault("output", {})["result_root"] = rewrite_result_root_seed(
            config["output"]["result_root"],
            int(args.seed),
        )
    if args.candidate_name is not None:
        config.setdefault("loss_components", {})["candidate_name"] = str(args.candidate_name)
    if args.lambda_svri is not None:
        config.setdefault("loss_components", {})["lambda_svri"] = float(args.lambda_svri)
    if args.lambda_sqi is not None:
        config.setdefault("loss_components", {})["lambda_sqi"] = float(args.lambda_sqi)
    if args.sqi_weighting_mode is not None:
        config.setdefault("loss_components", {})["sqi_weighting_mode"] = str(args.sqi_weighting_mode)
    if args.effective_result_root is not None:
        config.setdefault("output", {})["result_root"] = str(args.effective_result_root)
        runtime["effective_result_root_override"] = str(args.effective_result_root)
        runtime["exact_result_root"] = True
    runtime["fixed_e7a_candidate"] = (
        bool(runtime["multi_seed_run"])
        and config.get("experiment_id") == "PAPAGEI_S_GENERIC_SUPCON_MORPH_E4_BRANCH_SIGD_V1"
        and config.get("loss_components", {}).get("candidate_name") == "svri0p05_sqi0p05"
        and float(config.get("loss_components", {}).get("lambda_svri", 0.0)) == 0.05
        and float(config.get("loss_components", {}).get("lambda_sqi", 0.0)) == 0.05
    )
    config["_runtime"] = runtime
    requested = args.device
    device = torch.device("cuda" if requested == "auto" and torch.cuda.is_available() else ("cpu" if requested == "auto" else requested))
    result_root = resolve_training_result_root(root, config, smoke_test=False)
    checkpoint = Path(args.checkpoint) if args.checkpoint else result_root / "checkpoints" / "best_projection_head.pt"
    if not checkpoint.exists():
        raise FileNotFoundError(f"Projection checkpoint missing: {checkpoint}")
    model = PaPaGeiProjectionModel(root, config).to(device)
    checkpoint_metadata = load_projection_checkpoint(checkpoint, model)
    eval_root = ensure_dir(result_root / "final_exhaustive_evaluation")
    protected = [
        eval_root / "adapted_model_run_manifest.json",
        eval_root / "validation_scores.csv",
        eval_root / "test_scores.csv",
    ]
    existing = [path for path in protected if path.exists()]
    if existing and not args.overwrite:
        raise RuntimeError(f"Final evaluation output exists; pass --overwrite to replace: {existing[:3]}")
    summary = evaluate_final(root=root, train_config=config, model=model, device=device, result_root=eval_root)
    run_manifest = {
        "experiment_id": config["experiment_id"],
        "experiment_stage": config["experiment_stage"],
        "seed": config["seed"],
        "base_config_path": runtime["base_config_path"],
        "effective_result_root": _relative_to_root(root, result_root),
        "multi_seed_run": bool(runtime["multi_seed_run"]),
        "fixed_e7a_candidate": bool(runtime["fixed_e7a_candidate"]),
        "checkpoint_path": str(checkpoint),
        "checkpoint_metadata": checkpoint_metadata,
        "model_metadata": model.get_model_metadata(),
        "loss_type": config["training"]["loss"],
        "sampler_mode": config["training"]["sampler_mode"],
        "positive_mask_mode": config["training"]["positive_mask_mode"],
        "session_centroid_alignment_weight": config.get("loss_components", {}).get("session_centroid_alignment_weight", 0.0),
        "morphology_heads_loaded": model.morphology_heads is not None,
        "morphology_used_for_verification": False,
        "morphology_targets": config.get("model", {}).get("morphology_heads", {}).get("targets", []),
        "lambda_svri": config.get("loss_components", {}).get("lambda_svri", 0.0),
        "lambda_sqi": config.get("loss_components", {}).get("lambda_sqi", 0.0),
        "sqi_weighting_enabled": bool(config.get("loss_components", {}).get("sqi_weighting_enabled", False)),
        "sqi_weighting_mode": config.get("loss_components", {}).get("sqi_weighting_mode"),
        "sqi_weighting_used_for_verification": False,
        "checkpoint_selection_metric": "validation_exhaustive_eer",
        "final_protocol_id": config["input"]["final_protocol_id"],
        "input_protocol_id": config["input"]["input_protocol_id"],
        "validation_threshold_only": True,
        "final_test_evaluation_runs": 1,
        "test_threshold_tuning_performed": False,
        "selected_lambda_source": "validation_only" if config.get("loss_components", {}).get("session_centroid_alignment_weight_candidates") else None,
        "post_e4_e5_policy_applies": bool(config.get("fairness", {}).get("post_e4_e5_policy_applies", False)),
        "frozen_exhaustive_baseline_reference": "evaluation/SigD/results/papagei_s_frozen_cosine_exhaustive_eval/seed42",
        "generated_datetime_utc": utc_now_iso(),
        "summary": summary,
    }
    write_json(eval_root / "adapted_model_run_manifest.json", run_manifest)
    print(
        f"adapted_model_evaluation_completed=True experiment_id={config['experiment_id']} "
        f"result_root={eval_root}"
    )
    return 0


def _relative_to_root(root: Path, path: Path) -> str:
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


if __name__ == "__main__":
    raise SystemExit(main())
