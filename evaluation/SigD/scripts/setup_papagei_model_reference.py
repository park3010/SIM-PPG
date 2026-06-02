#!/usr/bin/env python3
"""Inspect, optionally download, and verify official PaPaGei-S assets."""

from __future__ import annotations

import argparse
import importlib
import json
import shutil
import subprocess
import sys
import tempfile
import urllib.request
from pathlib import Path
from typing import Any

import torch

SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))

from common import (  # noqa: E402
    detect_project_root,
    ensure_dir,
    load_eval_config,
    load_json,
    md5_file,
    resolve_from_root,
    sha256_file,
    utc_now_iso,
    write_json,
)


SOURCE_REPOSITORY = "https://github.com/Nokia-Bell-Labs/papagei-foundation-model"
SOURCE_BRANCH = "main"
ZENODO_RECORD_ID = "13983110"
CHECKPOINT_FILENAME = "papagei_s.pt"
EXPECTED_MD5 = "a4cdb32392e2a7b25999128af92813b5"
MODEL_CONFIG = {
    "base_filters": 32,
    "kernel_size": 3,
    "stride": 2,
    "groups": 1,
    "n_block": 18,
    "n_classes": 512,
    "n_experts": 3,
}
SOURCE_OVERLAP_LIMITATION = {
    "overlap_risk_present": True,
    "reason": "PaPaGei pretraining includes MIMIC-III, while SigD-Core is reconstructed from MIMIC-III waveform records.",
    "interpretation_policy": "Report frozen performance with explicit source-level overlap limitation; use controlled same-backbone adaptation ablations for method-level claims.",
}


def asset_paths(root: Path, config: dict[str, Any]) -> dict[str, Path]:
    """Return canonical local source/checkpoint paths."""

    source_dir = resolve_from_root(root, config["encoder"]["official_source_dir"])
    checkpoint_path = resolve_from_root(root, config["encoder"]["checkpoint_path"])
    assert source_dir is not None and checkpoint_path is not None
    return {
        "source_dir": source_dir,
        "checkpoint_path": checkpoint_path,
        "asset_root": root / "evaluation" / "SigD" / "official_reference" / "PaPaGei_Model",
        "metadata_dir": root / "evaluation" / "SigD" / "metadata",
    }


def download_source(source_dir: Path, overwrite: bool) -> dict[str, Any]:
    """Download official source by git clone, falling back to GitHub archive."""

    if source_dir.exists() and any(source_dir.iterdir()) and not overwrite:
        return {"downloaded": False, "method": "existing_local_source", "git_commit_sha": git_commit_sha(source_dir)}
    if source_dir.exists() and overwrite:
        shutil.rmtree(source_dir)
    ensure_dir(source_dir.parent)
    try:
        subprocess.run(
            ["git", "clone", "--depth", "1", "--branch", SOURCE_BRANCH, SOURCE_REPOSITORY, str(source_dir)],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        return {"downloaded": True, "method": "git_clone_depth_1", "git_commit_sha": git_commit_sha(source_dir)}
    except Exception as exc:
        archive_url = f"{SOURCE_REPOSITORY}/archive/refs/heads/{SOURCE_BRANCH}.zip"
        with tempfile.TemporaryDirectory() as tmpdir:
            archive_path = Path(tmpdir) / "papagei-main.zip"
            urllib.request.urlretrieve(archive_url, archive_path)
            shutil.unpack_archive(str(archive_path), tmpdir)
            extracted = next(Path(tmpdir).glob("papagei-foundation-model-*"))
            shutil.move(str(extracted), str(source_dir))
            return {
                "downloaded": True,
                "method": "github_archive_zip",
                "archive_url": archive_url,
                "archive_sha256": sha256_file(archive_path),
                "git_clone_error": str(exc),
            }


def download_checkpoint(checkpoint_path: Path, overwrite: bool) -> dict[str, Any]:
    """Download papagei_s.pt from the official Zenodo record."""

    if checkpoint_path.exists() and not overwrite:
        return {"downloaded": False, "method": "existing_local_checkpoint"}
    ensure_dir(checkpoint_path.parent)
    api_url = f"https://zenodo.org/api/records/{ZENODO_RECORD_ID}"
    with urllib.request.urlopen(api_url, timeout=60) as response:
        record = json.loads(response.read().decode("utf-8"))
    target = None
    for file_info in record.get("files", []):
        if file_info.get("key") == CHECKPOINT_FILENAME:
            target = file_info
            break
    if target is None:
        raise RuntimeError(f"{CHECKPOINT_FILENAME} not found in Zenodo record {ZENODO_RECORD_ID}.")
    download_url = target.get("links", {}).get("download") or target.get("links", {}).get("self")
    if not download_url:
        raise RuntimeError("Zenodo file download link missing.")
    if checkpoint_path.exists() and overwrite:
        checkpoint_path.unlink()
    urllib.request.urlretrieve(download_url, checkpoint_path)
    return {
        "downloaded": True,
        "method": "zenodo_api_file_download",
        "zenodo_record_id": ZENODO_RECORD_ID,
        "download_url": download_url,
    }


def git_commit_sha(source_dir: Path) -> str | None:
    """Return git commit SHA when source_dir is a git clone."""

    try:
        result = subprocess.run(
            ["git", "-C", str(source_dir), "rev-parse", "HEAD"],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
    except Exception:
        return None
    return result.stdout.strip()


def source_manifest(root: Path, source_dir: Path, acquisition: dict[str, Any], verify: bool) -> dict[str, Any]:
    """Build source snapshot manifest."""

    key_files = {
        "README.md": source_dir / "README.md",
        "LICENSE": source_dir / "LICENSE",
        "models/resnet.py": source_dir / "models" / "resnet.py",
        "linearprobing/utils.py": source_dir / "linearprobing" / "utils.py",
    }
    hashes = []
    for relative, path in key_files.items():
        if path.exists() and path.is_file():
            hashes.append({"relative_path": relative, "sha256": sha256_file(path), "size_bytes": path.stat().st_size})
    import_result = verify_source_import(source_dir) if verify else {"performed": False}
    verified = bool(
        source_dir.exists()
        and key_files["README.md"].exists()
        and key_files["LICENSE"].exists()
        and key_files["models/resnet.py"].exists()
        and key_files["linearprobing/utils.py"].exists()
        and (not verify or import_result.get("source_import_verified") is True)
    )
    return {
        "source_name": "PaPaGei",
        "source_repository": SOURCE_REPOSITORY,
        "source_branch": SOURCE_BRANCH,
        "acquisition_datetime_utc": utc_now_iso(),
        "acquisition_method": acquisition.get("method", "local_inspection"),
        "local_source_dir": str(source_dir.relative_to(root)),
        "git_commit_sha": git_commit_sha(source_dir) or acquisition.get("git_commit_sha"),
        "archive_sha256": acquisition.get("archive_sha256"),
        "license_present": key_files["LICENSE"].exists(),
        "readme_present": key_files["README.md"].exists(),
        "model_definition_present": key_files["models/resnet.py"].exists(),
        "loading_utility_present": key_files["linearprobing/utils.py"].exists(),
        "source_files_sha256": hashes,
        "import_verification": import_result,
        "verified": verified,
    }


def checkpoint_manifest(root: Path, checkpoint_path: Path) -> dict[str, Any]:
    """Build checkpoint manifest."""

    available = checkpoint_path.exists() and checkpoint_path.is_file()
    observed_md5 = md5_file(checkpoint_path) if available else None
    return {
        "model_variant": "PaPaGei-S",
        "checkpoint_filename": CHECKPOINT_FILENAME,
        "local_checkpoint_path": str(checkpoint_path.relative_to(root)),
        "zenodo_record_id": ZENODO_RECORD_ID,
        "expected_md5": EXPECTED_MD5,
        "observed_md5": observed_md5,
        "md5_verified": observed_md5 == EXPECTED_MD5 if observed_md5 else False,
        "sha256": sha256_file(checkpoint_path) if available else None,
        "size_bytes": checkpoint_path.stat().st_size if available else None,
        "verified": observed_md5 == EXPECTED_MD5 if observed_md5 else False,
    }


def local_inspection_manifest(
    root: Path,
    source: dict[str, Any],
    checkpoint: dict[str, Any],
    network_download_performed: bool,
) -> dict[str, Any]:
    """Build non-authoritative local inspection status without readiness downgrade."""

    return {
        "inspection_datetime_utc": utc_now_iso(),
        "source_name": "PaPaGei",
        "source_repository": SOURCE_REPOSITORY,
        "local_source_dir": source.get("local_source_dir"),
        "local_source_present": bool(source.get("model_definition_present")),
        "license_present": bool(source.get("license_present")),
        "readme_present": bool(source.get("readme_present")),
        "model_definition_present": bool(source.get("model_definition_present")),
        "loading_utility_present": bool(source.get("loading_utility_present")),
        "local_checkpoint_path": checkpoint.get("local_checkpoint_path"),
        "checkpoint_available": checkpoint.get("size_bytes") is not None,
        "checkpoint_expected_md5": checkpoint.get("expected_md5"),
        "checkpoint_observed_md5": checkpoint.get("observed_md5"),
        "checkpoint_md5_matches_expected": bool(checkpoint.get("md5_verified")),
        "checkpoint_sha256": checkpoint.get("sha256"),
        "checkpoint_size_bytes": checkpoint.get("size_bytes"),
        "network_download_performed": network_download_performed,
        "readiness_manifest_authoritative": False,
        "notes": [
            "This inspection manifest does not change scientific readiness.",
            "Run with --verify to refresh papagei_model_reference_manifest.json.",
        ],
    }


def verify_source_import(source_dir: Path) -> dict[str, Any]:
    """Import official model architecture/loading utility."""

    output: dict[str, Any] = {"performed": True}
    sys.path.insert(0, str(source_dir))
    try:
        importlib.import_module("models.resnet").ResNet1DMoE
        output["resnet_imported"] = True
    except Exception as exc:
        output["resnet_imported"] = False
        output["resnet_import_error"] = repr(exc)
    try:
        importlib.import_module("linearprobing.utils").load_model_without_module_prefix
        output["loading_utility_imported"] = True
    except Exception as exc:
        output["loading_utility_imported"] = False
        output["loading_utility_import_error"] = repr(exc)
    output["source_import_verified"] = bool(output.get("resnet_imported") and output.get("loading_utility_imported"))
    return output


def instantiate_official_model(source_dir: Path):
    """Instantiate official ResNet1DMoE from source snapshot."""

    sys.path.insert(0, str(source_dir))
    ResNet1DMoE = importlib.import_module("models.resnet").ResNet1DMoE
    return ResNet1DMoE(in_channels=1, **MODEL_CONFIG)


def extract_state_dict(checkpoint: Any) -> dict[str, torch.Tensor]:
    """Extract a tensor state_dict from common checkpoint structures."""

    if isinstance(checkpoint, dict):
        for key in ("state_dict", "model_state_dict", "model", "net", "encoder"):
            value = checkpoint.get(key)
            if isinstance(value, dict) and value and all(torch.is_tensor(v) for v in value.values()):
                return value
        if checkpoint and all(torch.is_tensor(v) for v in checkpoint.values()):
            return checkpoint
    raise RuntimeError("Could not extract tensor state_dict from PaPaGei checkpoint.")


def strip_known_prefixes(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Remove common wrapper prefixes while preserving strict key matching."""

    cleaned = {}
    for key, value in state_dict.items():
        new_key = key
        for prefix in ("module.",):
            if new_key.startswith(prefix):
                new_key = new_key[len(prefix) :]
        cleaned[new_key] = value
    return cleaned


def verify_checkpoint_loading_and_forward(source_dir: Path, checkpoint_path: Path) -> dict[str, Any]:
    """Strictly load checkpoint and run a forward contract smoke test."""

    result: dict[str, Any] = {"performed": True, "model_config": MODEL_CONFIG}
    try:
        model = instantiate_official_model(source_dir)
        result["architecture_instantiated"] = True
    except Exception as exc:
        result["architecture_instantiated"] = False
        result["architecture_error"] = repr(exc)
        return result
    try:
        checkpoint, load_method = safe_torch_load(checkpoint_path, checkpoint_verified=True)
        result["torch_load_method"] = load_method
        state_dict = strip_known_prefixes(extract_state_dict(checkpoint))
        incompatible = model.load_state_dict(state_dict, strict=False)
        missing = list(incompatible.missing_keys)
        unexpected = list(incompatible.unexpected_keys)
        result["missing_keys_count"] = len(missing)
        result["unexpected_keys_count"] = len(unexpected)
        result["missing_keys_preview"] = missing[:10]
        result["unexpected_keys_preview"] = unexpected[:10]
        result["strict_checkpoint_loaded"] = len(missing) == 0 and len(unexpected) == 0
    except Exception as exc:
        result["strict_checkpoint_loaded"] = False
        result["checkpoint_load_error"] = repr(exc)
        return result
    if not result["strict_checkpoint_loaded"]:
        return result
    try:
        model.eval()
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        signal = torch.linspace(-1.0, 1.0, 1250, dtype=torch.float32).reshape(1, 1, 1250)
        with torch.inference_mode():
            outputs = model(signal)
        result["forward_output_is_tuple_or_list"] = isinstance(outputs, (tuple, list))
        embedding = outputs[0] if isinstance(outputs, (tuple, list)) else outputs
        result["forward_embedding_shape"] = list(embedding.shape)
        result["embedding_dim_verified"] = list(embedding.shape) == [1, 512]
        result["forward_embedding_finite"] = bool(torch.isfinite(embedding).all().item())
        result["trainable_parameter_count"] = sum(p.numel() for p in model.parameters() if p.requires_grad)
        result["frozen_verified"] = result["trainable_parameter_count"] == 0
    except Exception as exc:
        result["forward_error"] = repr(exc)
    return result


def safe_torch_load(checkpoint_path: Path, checkpoint_verified: bool) -> tuple[Any, str]:
    """Prefer weights_only checkpoint loading, falling back only after checksum verification."""

    try:
        return torch.load(checkpoint_path, map_location="cpu", weights_only=True), "weights_only_true"
    except TypeError:
        if not checkpoint_verified:
            raise RuntimeError("weights_only torch.load unsupported and checkpoint is not verified; refusing fallback.")
        return torch.load(checkpoint_path, map_location="cpu"), "fallback_full_load_after_md5_verification"
    except Exception as exc:
        if not checkpoint_verified:
            raise RuntimeError("weights_only torch.load failed and checkpoint is not verified; refusing fallback.") from exc
        try:
            return torch.load(checkpoint_path, map_location="cpu"), "fallback_full_load_after_md5_verification"
        except Exception:
            raise exc


def combined_manifest(
    *,
    source: dict[str, Any],
    checkpoint: dict[str, Any],
    load_forward: dict[str, Any],
    network_download_performed: bool,
) -> dict[str, Any]:
    """Build combined PaPaGei model reference manifest."""

    ready = bool(
        source.get("verified")
        and checkpoint.get("verified")
        and load_forward.get("strict_checkpoint_loaded")
        and load_forward.get("forward_output_is_tuple_or_list")
        and load_forward.get("embedding_dim_verified")
        and load_forward.get("frozen_verified")
        and load_forward.get("forward_embedding_finite")
    )
    missing_items = []
    if not source.get("verified"):
        missing_items.append("verified official PaPaGei source snapshot")
    if not checkpoint.get("verified"):
        missing_items.append("verified papagei_s.pt checkpoint with expected MD5")
    if checkpoint.get("verified") and not load_forward.get("strict_checkpoint_loaded"):
        missing_items.append("strict checkpoint load without missing/unexpected keys")
    if load_forward.get("strict_checkpoint_loaded") and not load_forward.get("forward_output_is_tuple_or_list"):
        missing_items.append("PaPaGei-S forward output tuple/list contract")
    if load_forward.get("strict_checkpoint_loaded") and not load_forward.get("embedding_dim_verified"):
        missing_items.append("forward smoke embedding shape [1, 512]")
    return {
        "model_variant": "PaPaGei-S",
        "generated_datetime_utc": utc_now_iso(),
        "official_source_available": source.get("model_definition_present", False),
        "official_source_verified": bool(source.get("verified")),
        "checkpoint_available": checkpoint.get("size_bytes") is not None,
        "checkpoint_verified": bool(checkpoint.get("verified")),
        "pretrained_weights_verified": bool(checkpoint.get("verified")),
        "architecture_verified": bool(load_forward.get("architecture_instantiated")),
        "loading_api_verified": bool(source.get("import_verification", {}).get("loading_utility_imported")),
        "embedding_dim_verified": bool(load_forward.get("embedding_dim_verified")),
        "ready_for_scientific_frozen_baseline": ready,
        "missing_items": missing_items,
        "source_manifest_path": "evaluation/SigD/metadata/papagei_source_snapshot_manifest.json",
        "checkpoint_manifest_path": "evaluation/SigD/metadata/papagei_checkpoint_manifest.json",
        "checkpoint_path": checkpoint.get("local_checkpoint_path"),
        "checkpoint_md5": checkpoint.get("observed_md5"),
        "checkpoint_sha256": checkpoint.get("sha256"),
        "model_config": MODEL_CONFIG,
        "forward_verification": load_forward,
        "source_overlap_limitation": SOURCE_OVERLAP_LIMITATION,
        "network_download_performed": network_download_performed,
    }


def write_manifests(root: Path, manifests: dict[str, dict[str, Any]]) -> None:
    """Write all setup manifests."""

    metadata_dir = root / "evaluation" / "SigD" / "metadata"
    ensure_dir(metadata_dir)
    write_json(metadata_dir / "papagei_source_snapshot_manifest.json", manifests["source"])
    write_json(metadata_dir / "papagei_checkpoint_manifest.json", manifests["checkpoint"])
    write_json(metadata_dir / "papagei_model_reference_manifest.json", manifests["combined"])


def existing_ready_manifest(root: Path) -> dict[str, Any] | None:
    """Return existing readiness manifest if it is already scientific-ready."""

    path = root / "evaluation" / "SigD" / "metadata" / "papagei_model_reference_manifest.json"
    if not path.exists():
        return None
    try:
        manifest = load_json(path)
    except Exception:
        return None
    return manifest if manifest.get("ready_for_scientific_frozen_baseline") is True else None


def write_local_inspection(root: Path, manifest: dict[str, Any]) -> None:
    """Write non-authoritative local inspection manifest."""

    metadata_dir = root / "evaluation" / "SigD" / "metadata"
    ensure_dir(metadata_dir)
    write_json(metadata_dir / "papagei_local_inspection_manifest.json", manifest)


def inspect_or_setup(root: Path, config: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    """Perform requested inspection/download/verification workflow."""

    paths = asset_paths(root, config)
    source_acquisition = {"method": "local_inspection"}
    checkpoint_acquisition = {"method": "local_inspection"}
    network_download_performed = False
    if args.download_official_assets:
        source_acquisition = download_source(paths["source_dir"], args.overwrite)
        checkpoint_acquisition = download_checkpoint(paths["checkpoint_path"], args.overwrite)
        network_download_performed = bool(source_acquisition.get("downloaded") or checkpoint_acquisition.get("downloaded"))
    source = source_manifest(root, paths["source_dir"], source_acquisition, args.verify)
    checkpoint = checkpoint_manifest(root, paths["checkpoint_path"])
    inspection = local_inspection_manifest(root, source, checkpoint, network_download_performed)
    write_local_inspection(root, inspection)
    can_forward_verify = bool(args.verify and source.get("verified") and checkpoint.get("verified"))
    load_forward = (
        verify_checkpoint_loading_and_forward(paths["source_dir"], paths["checkpoint_path"])
        if can_forward_verify
        else {"performed": False, "reason": "source_or_checkpoint_not_verified"}
    )
    source["checkpoint_acquisition"] = checkpoint_acquisition
    combined = combined_manifest(
        source=source,
        checkpoint=checkpoint,
        load_forward=load_forward,
        network_download_performed=network_download_performed,
    )
    manifests = {"source": source, "checkpoint": checkpoint, "combined": combined}
    readiness_path = root / "evaluation" / "SigD" / "metadata" / "papagei_model_reference_manifest.json"
    preserve_ready = bool(not args.verify and existing_ready_manifest(root) is not None)
    if preserve_ready:
        manifests["combined"] = existing_ready_manifest(root) or combined
        manifests["verified_readiness_manifest_preserved"] = {"value": True}
    elif args.verify or not readiness_path.exists():
        write_manifests(root, manifests)
        manifests["verified_readiness_manifest_preserved"] = {"value": False}
    else:
        manifests["verified_readiness_manifest_preserved"] = {"value": False}
    return manifests


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=None)
    parser.add_argument("--config", default=None)
    parser.add_argument("--download-official-assets", action="store_true")
    parser.add_argument("--verify", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = detect_project_root(args.root)
    config = load_eval_config(root, args.config)
    manifests = inspect_or_setup(root, config, args)
    combined = manifests["combined"]
    if args.verbose:
        print(f"official_source_verified={combined['official_source_verified']}")
        print(f"checkpoint_verified={combined['checkpoint_verified']}")
        print(f"ready_for_scientific_frozen_baseline={combined['ready_for_scientific_frozen_baseline']}")
        print(f"missing_items={combined['missing_items']}")
        print(
            "verified_readiness_manifest_preserved="
            f"{manifests.get('verified_readiness_manifest_preserved', {}).get('value', False)}"
        )
        print(f"local_inspection_manifest={root / 'evaluation' / 'SigD' / 'metadata' / 'papagei_local_inspection_manifest.json'}")
        print(f"manifest={root / 'evaluation' / 'SigD' / 'metadata' / 'papagei_model_reference_manifest.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
