#!/usr/bin/env python3
"""Build and verify frozen PaPaGei-S backbone embedding caches."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))

from backbone_feature_cache import (  # noqa: E402
    compute_backbone_embeddings,
    load_backbone_cache,
    role_paths,
    save_backbone_cache,
    verify_cache_payload,
)
from common import DEFAULT_GENERIC_CONFIG, detect_project_root, load_training_config, write_json  # noqa: E402


def build_role(root: Path, config: dict, role: str, overwrite: bool, device: torch.device) -> dict:
    """Build one role cache."""

    data_path, manifest_path = role_paths(root, role)
    if data_path.exists() and manifest_path.exists() and not overwrite:
        verification = verify_role(root, role)
        return {"manifest": {}, "verification": verification, "reused_existing": True}
    indices, embeddings, provenance = compute_backbone_embeddings(
        root=root,
        train_config=config,
        role=role,
        device=device,
    )
    manifest = save_backbone_cache(root, role, indices, embeddings, provenance, overwrite=overwrite)
    verification = verify_cache_payload(role, indices, embeddings, manifest)
    return {"manifest": manifest, "verification": verification}


def verify_role(root: Path, role: str) -> dict:
    """Verify one existing cache role."""

    indices, embeddings, manifest = load_backbone_cache(root, role)
    return verify_cache_payload(role, indices, embeddings, manifest)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=None)
    parser.add_argument("--config", default=None)
    parser.add_argument("--build-train", action="store_true")
    parser.add_argument("--build-validation", action="store_true")
    parser.add_argument("--build-test", action="store_true")
    parser.add_argument("--verify", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = detect_project_root(args.root)
    config = load_training_config(root, args.config or root / DEFAULT_GENERIC_CONFIG)
    requested = args.device
    device = torch.device("cuda" if requested == "auto" and torch.cuda.is_available() else ("cpu" if requested == "auto" else requested))
    role_flags = {
        "train": args.build_train,
        "validation_exhaustive": args.build_validation,
        "test_exhaustive": args.build_test,
    }
    outputs: dict[str, dict] = {}
    for role, enabled in role_flags.items():
        if enabled:
            outputs[role] = build_role(root, config, role, args.overwrite, device)
            if args.verbose:
                print(f"built_{role}=True count={outputs[role]['verification']['array_index_count']}")
    if args.verify:
        for role in role_flags:
            data_path, manifest_path = role_paths(root, role)
            if data_path.exists() and manifest_path.exists():
                outputs.setdefault(role, {})["verification"] = verify_role(root, role)
                if args.verbose:
                    print(f"verified_{role}={outputs[role]['verification']['passed']} count={outputs[role]['verification']['array_index_count']}")
    summary = {
        "cache_build_summary": outputs,
        "test_cache_built": bool(args.build_test),
    }
    write_json(root / "training" / "SigD" / "metadata" / "backbone_cache_build_summary.json", summary)
    failed = [role for role, payload in outputs.items() if payload.get("verification", {}).get("passed") is False]
    if failed:
        raise SystemExit(f"Cache verification failed: {failed}")
    print(
        "backbone_cache_done=True "
        + " ".join(
            f"{role}={payload.get('verification', {}).get('array_index_count')}"
            for role, payload in outputs.items()
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
