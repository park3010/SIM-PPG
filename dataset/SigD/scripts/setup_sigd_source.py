#!/usr/bin/env python3
"""Acquire and source-lock official SigD runtime annotation files."""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import subprocess
import tempfile
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from parse_sigd_annotations import (
    DATASET_NAME,
    DATASET_VERSION,
    SOURCE_MANIFEST_NAME,
    detect_root,
    load_config,
    setup_logging,
    sha256_file,
    sigd_dir,
)


REQUIRED_FILES = [
    "README.md",
    "Extracted_signal_records.pl",
    "GetMIMIC-IIIdata.ipynb",
]


def utc_now_iso() -> str:
    """Return a UTC ISO-8601 timestamp with seconds precision."""

    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def source_dir(root: Path) -> Path:
    """Return the runtime official source directory."""

    return sigd_dir(root) / "official" / "NasTul_SigD"


def source_manifest_path(root: Path) -> Path:
    """Return the source manifest path."""

    return sigd_dir(root) / "metadata" / SOURCE_MANIFEST_NAME


def run_git(args: list[str], cwd: Path | None = None) -> subprocess.CompletedProcess[str]:
    """Run a git command and capture text output."""

    return subprocess.run(
        ["git", *args],
        cwd=str(cwd) if cwd else None,
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )


def copy_required_files(src: Path, dst: Path) -> None:
    """Copy required official files from a checkout into the runtime directory."""

    dst.mkdir(parents=True, exist_ok=True)
    for filename in REQUIRED_FILES:
        source_file = src / filename
        if not source_file.exists():
            raise FileNotFoundError(f"Official repository is missing {filename}")
        shutil.copy2(source_file, dst / filename)


def clone_official_repo(repo_url: str, branch: str, dst: Path) -> str | None:
    """Clone the official repository shallowly and copy required files."""

    with tempfile.TemporaryDirectory(prefix="sigd_source_") as tmp:
        tmp_path = Path(tmp) / "repo"
        clone_args = ["clone", "--depth", "1", "--branch", branch, repo_url, str(tmp_path)]
        result = run_git(clone_args)
        if result.returncode != 0:
            logging.warning(
                "git clone with branch %s failed; trying repository default branch. stderr=%s",
                branch,
                result.stderr.strip(),
            )
            result = run_git(["clone", "--depth", "1", repo_url, str(tmp_path)])
        if result.returncode != 0:
            raise RuntimeError(f"git clone failed: {result.stderr.strip()}")
        copy_required_files(tmp_path, dst)
        commit = run_git(["rev-parse", "HEAD"], cwd=tmp_path)
        if commit.returncode == 0:
            return commit.stdout.strip()
    return None


def raw_github_url(repo_url: str, ref: str, filename: str) -> str:
    """Build a raw.githubusercontent.com URL for a GitHub repository file."""

    slug = repo_url.rstrip("/").removeprefix("https://github.com/")
    return f"https://raw.githubusercontent.com/{slug}/{ref}/{filename}"


def download_official_files_raw(repo_url: str, ref: str, dst: Path) -> None:
    """Download required official files from GitHub raw URLs."""

    dst.mkdir(parents=True, exist_ok=True)
    for filename in REQUIRED_FILES:
        url = raw_github_url(repo_url, ref, filename)
        logging.info("Downloading %s", url)
        with urllib.request.urlopen(url, timeout=60) as response:
            data = response.read()
        (dst / filename).write_bytes(data)


def acquire_official_files(
    root: Path,
    repo_url: str,
    branch: str,
    prefer_ref: str | None = None,
) -> str | None:
    """Acquire official source files using git first, then raw downloads."""

    dst = source_dir(root)
    ref = prefer_ref or branch
    try:
        return clone_official_repo(repo_url, branch, dst)
    except Exception as exc:
        logging.warning("git acquisition failed; falling back to raw files: %s", exc)
        download_official_files_raw(repo_url, ref, dst)
        return prefer_ref


def file_manifest_entries(root: Path) -> list[dict[str, Any]]:
    """Build manifest entries for required files."""

    base = sigd_dir(root)
    entries: list[dict[str, Any]] = []
    for filename in REQUIRED_FILES:
        path = source_dir(root) / filename
        entries.append(
            {
                "filename": filename,
                "relative_path": str(path.relative_to(base)),
                "sha256": sha256_file(path),
                "size_bytes": path.stat().st_size,
            }
        )
    return entries


def build_source_manifest(
    root: Path,
    config: dict[str, Any],
    commit: str | None,
) -> dict[str, Any]:
    """Build the source-lock manifest payload."""

    return {
        "dataset_name": config["dataset_name"],
        "dataset_version": config["dataset_version"],
        "source_locked": True,
        "official_repo_url": config["official_repo_url"],
        "official_repo_branch": config["official_repo_branch"],
        "official_repo_commit": commit,
        "retrieval_datetime_utc": utc_now_iso(),
        "official_files": file_manifest_entries(root),
        "physionet_database": config["physionet_database"],
        "physionet_version": str(config["physionet_version"]),
        "physionet_pn_dir": config["physionet_pn_dir"],
        "signal_channel": config["signal_channel"],
        "clinical_metadata_used": False,
        "demographic_metadata_enabled": False,
        "notes": [
            "Official SigD files are downloaded for local runtime reconstruction only.",
            "Extracted_signal_records.pl is loaded only after SHA256 source-lock verification.",
            "SigD-Core stores waveform-only public reconstruction provenance and no clinical or demographic metadata.",
        ],
    }


def read_manifest(path: Path) -> dict[str, Any]:
    """Read the existing source manifest."""

    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_manifest(path: Path, payload: dict[str, Any]) -> None:
    """Write the source manifest with stable formatting."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def expected_hashes(manifest: dict[str, Any]) -> dict[str, str]:
    """Return expected hashes keyed by filename."""

    return {
        str(item["filename"]): str(item["sha256"])
        for item in manifest.get("official_files", [])
    }


def validate_against_manifest(root: Path, manifest: dict[str, Any]) -> tuple[bool, list[str]]:
    """Validate local official files against an existing source lock."""

    failures: list[str] = []
    expected = expected_hashes(manifest)
    for filename in REQUIRED_FILES:
        path = source_dir(root) / filename
        if not path.exists():
            failures.append(f"missing:{filename}")
            continue
        actual = sha256_file(path)
        if actual != expected.get(filename):
            failures.append(f"sha256_mismatch:{filename}")
    return not failures, failures


def restore_locked_files_if_missing(
    root: Path,
    manifest: dict[str, Any],
    config: dict[str, Any],
) -> None:
    """Re-download locked files when local runtime copies are absent."""

    failures = [
        filename
        for filename in REQUIRED_FILES
        if not (source_dir(root) / filename).exists()
    ]
    if not failures:
        return
    commit = manifest.get("official_repo_commit")
    ref = str(commit or config["official_repo_branch"])
    logging.info("Restoring missing source files from locked ref %s", ref)
    download_official_files_raw(config["official_repo_url"], ref, source_dir(root))


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(
        description="Download and source-lock official SigD files for SigD-Core."
    )
    parser.add_argument("--root", type=str, default=None, help="sim_ppg root path")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Refresh source files and overwrite the existing source manifest.",
    )
    parser.add_argument(
        "--refresh-source",
        action="store_true",
        help="Explicitly refresh official source files and update hashes.",
    )
    parser.add_argument("--verbose", action="store_true", help="Enable debug logging")
    return parser.parse_args()


def main() -> int:
    """CLI entry point."""

    args = parse_args()
    root = detect_root(args.root)
    setup_logging(root, "setup_sigd_source.log", args.verbose)
    config = load_config(root)

    manifest_path = source_manifest_path(root)
    src_dir = source_dir(root)
    refresh = args.force or args.refresh_source

    logging.info(
        "SigD-Core source setup: range-limited PLETH extraction without persistent full-record storage"
    )
    logging.info("Runtime official source directory: %s", src_dir)

    if manifest_path.exists() and not refresh:
        manifest = read_manifest(manifest_path)
        restore_locked_files_if_missing(root, manifest, config)
        ok, failures = validate_against_manifest(root, manifest)
        if not ok:
            raise SystemExit(
                "Existing source manifest lock does not match local files: "
                + ", ".join(failures)
                + ". Use --refresh-source or --force only if you intend to update the lock."
            )
        logging.info("Source lock verified; no refresh performed.")
        return 0

    if refresh:
        logging.info("Refreshing official source files and source manifest.")

    commit = acquire_official_files(
        root,
        repo_url=config["official_repo_url"],
        branch=config["official_repo_branch"],
    )
    manifest = build_source_manifest(root, config, commit)
    write_manifest(manifest_path, manifest)
    logging.info("Wrote source manifest: %s", manifest_path)
    for entry in manifest["official_files"]:
        logging.info(
            "locked %s sha256=%s size=%s",
            entry["filename"],
            entry["sha256"],
            entry["size_bytes"],
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
