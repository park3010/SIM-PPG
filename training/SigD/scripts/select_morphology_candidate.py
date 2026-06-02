#!/usr/bin/env python3
"""Select E7 morphology candidate from validation histories only."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))

from common import detect_project_root  # noqa: E402
from morphology_selector import select_morphology_candidate  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=None)
    parser.add_argument("--experiment-family", required=True)
    parser.add_argument("--candidate-root", action="append", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--prefer-branch", default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = detect_project_root(args.root)
    roots = [root / Path(path) if not Path(path).is_absolute() else Path(path) for path in args.candidate_root]
    output = root / Path(args.output) if not Path(args.output).is_absolute() else Path(args.output)
    payload = select_morphology_candidate(
        experiment_family=args.experiment_family,
        candidate_roots=roots,
        output_path=output,
        prefer_branch=args.prefer_branch,
    )
    print(
        f"morphology_candidate_selected=True family={payload['experiment_family']} "
        f"candidate={payload['selected_candidate_name']} eer={payload['selected_validation_eer']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

