#!/usr/bin/env python3
"""Validate the fixed SigD-Core reconstruction snapshot."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

SCRIPT_DIR = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SCRIPT_DIR))

from common import detect_root, load_config, preprocessing_dir, setup_logging  # noqa: E402
from snapshot_validation import validate_snapshot  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate SigD-Core final reconstruction snapshot.")
    parser.add_argument("--root", type=str, default=None)
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--verify-all-npz-hashes", action="store_true")
    parser.add_argument("--limit-raw-ranges", type=int, default=None)
    parser.add_argument("--subject-id", type=str, default=None)
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = detect_root(args.root)
    setup_logging(root, "verify_sigd_snapshot.log", args.verbose)
    config = load_config(root, args.config)
    out = preprocessing_dir(root) / "metadata" / "input_snapshot_validation.json"
    payload = validate_snapshot(
        root,
        config,
        limit_raw_ranges=args.limit_raw_ranges,
        subject_id=args.subject_id,
        verify_all_npz_hashes=args.verify_all_npz_hashes,
        verify_selected_npz_hashes=not args.verify_all_npz_hashes,
        output_path=out,
    )
    print(
        "snapshot_valid={snapshot_valid} rows={extraction_manifest_rows} "
        "verified_npz={selected_npz_hash_verified_count} failures={n_failures}".format(
            n_failures=len(payload.get("validation_failures", [])),
            **payload,
        )
    )
    return 0 if payload["snapshot_valid"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
