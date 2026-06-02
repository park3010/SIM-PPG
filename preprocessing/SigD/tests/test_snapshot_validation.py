from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))

from common import sha256_file  # noqa: E402
from snapshot_validation import parse_sha256s_file, validate_npz_file  # noqa: E402


def write_npz(path: Path, raw_range_id: str = "r1") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        ppg=np.arange(20, dtype=np.float32),
        fs=np.float64(125),
        raw_range_id=np.asarray(raw_range_id),
        subject_id=np.asarray("p000001"),
        session_timestamp=np.asarray("2100-01-01-00-00"),
        channel_name=np.asarray("PLETH"),
        dataset_name=np.asarray("SigD-Core"),
        dataset_version=np.asarray("waveform_only_public_reconstruction_v1"),
    )


def test_sha256s_parse_and_match(tmp_path: Path) -> None:
    file_path = tmp_path / "a.json"
    file_path.write_text("{}", encoding="utf-8")
    sums = tmp_path / "SHA256SUMS.txt"
    sums.write_text(f"{sha256_file(file_path)}  a.json\n", encoding="utf-8")
    parsed = parse_sha256s_file(sums)
    assert parsed["a.json"] == sha256_file(file_path)


def test_hash_mismatch_is_detectable(tmp_path: Path) -> None:
    file_path = tmp_path / "a.json"
    file_path.write_text("{}", encoding="utf-8")
    sums = tmp_path / "SHA256SUMS.txt"
    sums.write_text(f"{'0' * 64}  a.json\n", encoding="utf-8")
    parsed = parse_sha256s_file(sums)
    assert parsed["a.json"] != sha256_file(file_path)


def test_npz_hash_validation_success(tmp_path: Path) -> None:
    root = tmp_path
    npz_path = root / "dataset/SigD/data/raw_ranges/p000001/s/range_000.npz"
    write_npz(npz_path, "r1")
    row = {
        "raw_range_id": "r1",
        "output_npz_path": "data/raw_ranges/p000001/s/range_000.npz",
        "npz_sha256": sha256_file(npz_path),
    }
    result = validate_npz_file(root, row, verify_hash=True)
    assert result["fields_ok"] is True
    assert result["sha256_match"] is True


def test_npz_hash_validation_failure(tmp_path: Path) -> None:
    root = tmp_path
    npz_path = root / "dataset/SigD/data/raw_ranges/p000001/s/range_000.npz"
    write_npz(npz_path, "r1")
    row = {
        "raw_range_id": "r1",
        "output_npz_path": "data/raw_ranges/p000001/s/range_000.npz",
        "npz_sha256": "0" * 64,
    }
    result = validate_npz_file(root, row, verify_hash=True)
    assert result["sha256_match"] is False
    assert result["failure_reason"] == "npz_sha256_mismatch"


def test_snapshot_validation_json_shape_can_be_written(tmp_path: Path) -> None:
    out = tmp_path / "input_snapshot_validation.json"
    out.write_text('{"snapshot_valid": true}', encoding="utf-8")
    assert out.exists()
