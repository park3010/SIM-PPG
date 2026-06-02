#!/usr/bin/env python3
"""Select E8 SQI-weighting candidate using validation only."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))

from common import detect_project_root  # noqa: E402
from e8_selector import select_e8_candidate  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=None)
    parser.add_argument("--e7-reference", default=None)
    parser.add_argument("--candidate-root", action="append", required=True)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = detect_project_root(args.root)
    roots = [root / Path(path) if not Path(path).is_absolute() else Path(path) for path in args.candidate_root]
    output = root / Path(args.output) if not Path(args.output).is_absolute() else Path(args.output)
    reference = root / Path(args.e7_reference) if args.e7_reference and not Path(args.e7_reference).is_absolute() else (Path(args.e7_reference) if args.e7_reference else None)
    payload = select_e8_candidate(candidate_roots=roots, e7_reference_path=reference, output_path=output)
    print(
        f"e8_candidate_selection_completed=True selected_model={payload['selected_model']} "
        f"allowed={payload['final_e8_test_evaluation_allowed']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

