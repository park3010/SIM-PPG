#!/usr/bin/env python3
"""Download and source-lock official PaPaGei reference files."""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import subprocess
import tempfile
import urllib.request
from pathlib import Path
from typing import Any

import sys

SCRIPT_DIR = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SCRIPT_DIR))

from common import detect_root, load_config, preprocessing_dir, setup_logging, sha256_file, utc_now_iso, write_json  # noqa: E402


REQUIRED_FILES = [
    "README.md",
    "LICENSE",
    "preprocessing/ppg.py",
    "preprocessing/flatline.py",
    "segmentations.py",
    "morphology.py",
    "dataset.py",
]


def run_git(args: list[str], cwd: Path | None = None) -> subprocess.CompletedProcess[str]:
    """Run git and capture stdout/stderr."""

    return subprocess.run(
        ["git", *args],
        cwd=str(cwd) if cwd else None,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )


def reference_dir(root: Path) -> Path:
    """Return runtime PaPaGei reference directory."""

    return preprocessing_dir(root) / "official_reference" / "PaPaGei"


def copy_required_files(source: Path, destination: Path) -> None:
    """Copy required reference files preserving relative paths."""

    destination.mkdir(parents=True, exist_ok=True)
    for rel in REQUIRED_FILES:
        src = source / rel
        if not src.exists():
            raise FileNotFoundError(f"Official PaPaGei source missing {rel}")
        dst = destination / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)


def clone_reference(repo_url: str, branch: str, destination: Path) -> str | None:
    """Clone the official repo and copy the required files."""

    with tempfile.TemporaryDirectory(prefix="papagei_reference_") as tmp:
        repo_path = Path(tmp) / "repo"
        result = run_git(["clone", "--depth", "1", "--branch", branch, repo_url, str(repo_path)])
        if result.returncode != 0:
            logging.warning("Branch clone failed, trying default branch: %s", result.stderr.strip())
            result = run_git(["clone", "--depth", "1", repo_url, str(repo_path)])
        if result.returncode != 0:
            raise RuntimeError(result.stderr.strip())
        copy_required_files(repo_path, destination)
        commit = run_git(["rev-parse", "HEAD"], cwd=repo_path)
        return commit.stdout.strip() if commit.returncode == 0 else None


def raw_url(repo_url: str, ref: str, rel_path: str) -> str:
    """Build a raw GitHub URL."""

    slug = repo_url.rstrip("/").removeprefix("https://github.com/")
    return f"https://raw.githubusercontent.com/{slug}/{ref}/{rel_path}"


def download_raw_reference(repo_url: str, ref: str, destination: Path) -> None:
    """Download required files from GitHub raw URLs."""

    destination.mkdir(parents=True, exist_ok=True)
    for rel in REQUIRED_FILES:
        url = raw_url(repo_url, ref, rel)
        logging.info("Downloading %s", url)
        with urllib.request.urlopen(url, timeout=60) as response:
            data = response.read()
        out = destination / rel
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_bytes(data)


def file_entries(root: Path) -> list[dict[str, Any]]:
    """Return source file hash manifest entries."""

    base = preprocessing_dir(root)
    entries = []
    for rel in REQUIRED_FILES:
        path = reference_dir(root) / rel
        entries.append(
            {
                "path": rel,
                "relative_path": str(path.relative_to(base)),
                "sha256": sha256_file(path),
                "size_bytes": path.stat().st_size,
            }
        )
    return entries


def build_manifest(root: Path, config: dict[str, Any], commit: str | None) -> dict[str, Any]:
    """Build the PaPaGei source lock manifest."""

    papagei = config["papagei_alignment"]
    return {
        "source_name": "PaPaGei",
        "official_repo_url": papagei["official_repo_url"],
        "official_repo_branch": papagei["official_repo_branch"],
        "official_repo_commit": commit,
        "retrieval_datetime_utc": utc_now_iso(),
        "license_file": "LICENSE",
        "source_files": file_entries(root),
        "referenced_behavior": {
            "filtering_function": "preprocess_one_ppg_signal",
            "filter_backend": "pyPPG",
            "filter_lowcut_hz": papagei["filter_lowcut_hz"],
            "filter_highcut_hz": papagei["filter_highcut_hz"],
            "filter_order": papagei["filter_order"],
            "segmentation_seconds": 10,
            "target_fs_hz": 125,
            "normalization_in_dataset_loader": "z-score",
            "morphology_functions": ["extract_svri", "skewness_sqi", "compute_ipa"],
        },
        "local_adaptations": [
            "nonfinite_interpolation_for_SigD_raw_ranges",
            "validity_masks_for_morphology_targets",
            "no_quality_threshold_rejection_at_preprocessing_stage",
        ],
    }


def validate_existing(root: Path, manifest_path: Path) -> bool:
    """Validate runtime reference files against an existing manifest."""

    if not manifest_path.exists():
        return False
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected = {entry["path"]: entry["sha256"] for entry in manifest.get("source_files", [])}
    for rel in REQUIRED_FILES:
        path = reference_dir(root) / rel
        if not path.exists() or sha256_file(path) != expected.get(rel):
            return False
    return True


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Source-lock PaPaGei reference files.")
    parser.add_argument("--root", type=str, default=None)
    parser.add_argument("--refresh-source", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = detect_root(args.root)
    setup_logging(root, "setup_papagei_reference.log", args.verbose)
    config = load_config(root)
    manifest_path = preprocessing_dir(root) / "metadata" / "papagei_reference_manifest.json"
    refresh = args.refresh_source or args.force
    if manifest_path.exists() and not refresh and validate_existing(root, manifest_path):
        logging.info("PaPaGei reference source lock verified.")
        return 0

    papagei = config["papagei_alignment"]
    commit: str | None = None
    try:
        commit = clone_reference(
            papagei["official_repo_url"],
            papagei["official_repo_branch"],
            reference_dir(root),
        )
    except Exception as exc:
        logging.warning("git clone failed; using raw download fallback: %s", exc)
        download_raw_reference(
            papagei["official_repo_url"],
            papagei["official_repo_branch"],
            reference_dir(root),
        )
    manifest = build_manifest(root, config, commit)
    write_json(manifest_path, manifest)
    logging.info("Wrote %s", manifest_path)
    logging.info("Official commit: %s", commit)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
