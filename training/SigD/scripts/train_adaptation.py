#!/usr/bin/env python3
"""Train PaPaGei-S projection-head adaptation without touching test data."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))

from common import detect_project_root, load_training_config, rewrite_result_root_seed  # noqa: E402
from trainer import AdaptationTrainer  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=None)
    parser.add_argument("--config", required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--max-epochs", type=int, default=None)
    parser.add_argument("--num-batches-per-epoch", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--effective-result-root", default=None)
    parser.add_argument("--lambda-align", type=float, default=None)
    parser.add_argument("--lambda-svri", type=float, default=None)
    parser.add_argument("--lambda-sqi", type=float, default=None)
    parser.add_argument("--candidate-name", default=None)
    parser.add_argument("--sqi-weighting-mode", default=None)
    parser.add_argument("--smoke-test", action="store_true")
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
    if args.effective_result_root is not None:
        config.setdefault("output", {})["result_root"] = str(args.effective_result_root)
        runtime["effective_result_root_override"] = str(args.effective_result_root)
        runtime["exact_result_root"] = True
    if args.lambda_align is not None:
        config.setdefault("loss_components", {})["session_centroid_alignment_weight"] = float(args.lambda_align)
    if args.lambda_svri is not None:
        config.setdefault("loss_components", {})["lambda_svri"] = float(args.lambda_svri)
    if args.lambda_sqi is not None:
        config.setdefault("loss_components", {})["lambda_sqi"] = float(args.lambda_sqi)
    if args.candidate_name is not None:
        config.setdefault("loss_components", {})["candidate_name"] = str(args.candidate_name)
    if args.sqi_weighting_mode is not None:
        config.setdefault("loss_components", {})["sqi_weighting_mode"] = str(args.sqi_weighting_mode)
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
    trainer = AdaptationTrainer(root, config, device=device)
    manifest = trainer.fit(
        max_epochs=args.max_epochs,
        num_batches_per_epoch=args.num_batches_per_epoch,
        smoke_test=args.smoke_test,
        overwrite=args.overwrite,
    )
    print(
        f"adaptation_training_completed=True experiment_id={manifest['experiment_id']} "
        f"smoke_test={manifest['smoke_test']} best_epoch={manifest['best_epoch']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
