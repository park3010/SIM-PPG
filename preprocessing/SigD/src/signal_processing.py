"""Common standardized PPG filtering and fixed-window generation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from scipy import signal

from common import sha256_file
from morphology_targets import compute_morphology_targets


@dataclass
class RawRangeData:
    """Loaded raw SigD-Core range and provenance."""

    ppg: np.ndarray
    fs: float
    raw_range_id: str
    subject_id: str
    session_timestamp: str


def npz_scalar(value: Any) -> Any:
    """Convert a scalar NPZ field to a Python value."""

    array = np.asarray(value)
    if array.shape == ():
        return array.item()
    if array.size == 1:
        return array.reshape(-1)[0].item()
    return array


def load_raw_range_npz(path: Path, expected_raw_range_id: str | None = None) -> RawRangeData:
    """Safely load one SigD-Core raw range NPZ."""

    with np.load(path, allow_pickle=False) as data:
        raw_range_id = str(npz_scalar(data["raw_range_id"]))
        if expected_raw_range_id and raw_range_id != expected_raw_range_id:
            raise ValueError(
                f"raw_range_id mismatch: expected {expected_raw_range_id}, got {raw_range_id}"
            )
        return RawRangeData(
            ppg=np.asarray(data["ppg"], dtype=np.float32).reshape(-1),
            fs=float(npz_scalar(data["fs"])),
            raw_range_id=raw_range_id,
            subject_id=str(npz_scalar(data["subject_id"])),
            session_timestamp=str(npz_scalar(data["session_timestamp"])),
        )


def flatline_ratio(values: np.ndarray) -> float:
    """Return the fraction of adjacent finite differences equal to zero."""

    finite = np.asarray(values)[np.isfinite(values)]
    if finite.size < 2:
        return 0.0
    return float(np.mean(np.diff(finite) == 0))


def basic_stats(values: np.ndarray, prefix: str) -> dict[str, Any]:
    """Return basic finite-value statistics."""

    finite = np.asarray(values)[np.isfinite(values)]
    return {
        f"{prefix}_mean": float(np.mean(finite)) if finite.size else "",
        f"{prefix}_std": float(np.std(finite)) if finite.size else "",
        f"{prefix}_min": float(np.min(finite)) if finite.size else "",
        f"{prefix}_max": float(np.max(finite)) if finite.size else "",
    }


def raw_integrity(values: np.ndarray) -> dict[str, Any]:
    """Compute raw range integrity stats before interpolation/filtering."""

    x = np.asarray(values)
    nonfinite = ~np.isfinite(x)
    stats = {
        "total_samples": int(x.size),
        "finite_samples": int(np.isfinite(x).sum()),
        "nan_count": int(np.isnan(x).sum()),
        "inf_count": int(np.isinf(x).sum()),
        "nonfinite_ratio": float(nonfinite.mean()) if x.size else 1.0,
        "raw_flatline_ratio": flatline_ratio(x),
    }
    stats.update(basic_stats(x, "raw"))
    return stats


def interpolate_nonfinite(values: np.ndarray) -> tuple[np.ndarray, np.ndarray, bool]:
    """Linearly interpolate NaN/Inf values, extending edge values as needed."""

    x = np.asarray(values, dtype=np.float64)
    mask = ~np.isfinite(x)
    if mask.all():
        raise ValueError("all_nonfinite")
    if not mask.any():
        return x.astype(np.float32), mask, False
    idx = np.arange(x.size)
    finite_idx = idx[~mask]
    finite_values = x[~mask]
    repaired = np.interp(idx, finite_idx, finite_values)
    return repaired.astype(np.float32), mask, True


def pypg_filter(values: np.ndarray, fs: float, lowcut: float, highcut: float, order: int) -> np.ndarray:
    """Filter using pyPPG's Preprocess backend."""

    try:
        from dotmap import DotMap
        from pyPPG.preproc import Preprocess
    except ImportError as exc:
        raise RuntimeError(
            "pyPPG backend is required by config but pyPPG/dotmap is unavailable"
        ) from exc
    processor = Preprocess(fL=lowcut, fH=highcut, order=order)
    ppg, _, _, _ = processor.get_signals(
        DotMap({"v": np.asarray(values, dtype=float), "fs": fs, "filtering": True})
    )
    return np.asarray(ppg, dtype=np.float32)


def scipy_test_filter(values: np.ndarray, fs: float, lowcut: float, highcut: float, order: int) -> np.ndarray:
    """A test-only scipy bandpass path used by unit tests."""

    sos = signal.butter(order, [lowcut, highcut], btype="bandpass", fs=fs, output="sos")
    return signal.sosfiltfilt(sos, np.asarray(values, dtype=float)).astype(np.float32)


def filter_ppg(values: np.ndarray, fs: float, config: dict[str, Any]) -> np.ndarray:
    """Apply the configured standardized PPG filter.

    The default pyPPG parameters are source-locked through the PaPaGei
    reference because PaPaGei uses that preprocessing definition. The output
    remains the common 10s/125Hz PPG protocol, not a PaPaGei-S native tensor.
    """

    papagei = config["papagei_alignment"]
    lowcut = float(papagei["filter_lowcut_hz"])
    highcut = float(papagei["filter_highcut_hz"])
    order = int(papagei["filter_order"])
    backend = str(papagei.get("filter_backend", "pyPPG"))
    if backend == "pyPPG":
        return pypg_filter(values, fs, lowcut, highcut, order)
    if backend == "scipy_test":
        return scipy_test_filter(values, fs, lowcut, highcut, order)
    raise ValueError(f"unsupported_filter_backend:{backend}")


def resample_to_target(window: np.ndarray, source_fs: float, target_fs: float, seconds: int) -> tuple[np.ndarray, bool]:
    """Resample one window to target fs if needed."""

    expected = int(round(target_fs * seconds))
    if abs(source_fs - target_fs) < 1.0e-9 and window.size == expected:
        return np.asarray(window, dtype=np.float32), False
    resampled = signal.resample(np.asarray(window, dtype=float), expected)
    return np.asarray(resampled, dtype=np.float32), True


def iter_window_slices(total_samples: int, fs: float, window_seconds: int) -> tuple[list[tuple[int, int]], int]:
    """Return non-overlapping window slices and trailing remainder samples."""

    window_samples = int(round(fs * window_seconds))
    usable = (total_samples // window_samples) * window_samples
    slices = [(start, start + window_samples) for start in range(0, usable, window_samples)]
    return slices, total_samples - usable


def model_input_validity(
    filtered_window: np.ndarray,
    original_nonfinite_ratio: float,
    expected_samples: int,
    config: dict[str, Any],
) -> tuple[bool, str]:
    """Apply hard validity rules for model input availability."""

    hard = config["hard_validity"]
    nonfinite = config["nonfinite_handling"]
    if original_nonfinite_ratio > float(nonfinite["max_window_original_nonfinite_ratio_for_model_input"]):
        return False, "window_original_nonfinite_ratio_too_high"
    if filtered_window.size != expected_samples:
        return False, "unexpected_output_length"
    if not np.all(np.isfinite(filtered_window)):
        return False, "filtered_output_nonfinite"
    if float(np.std(filtered_window)) <= float(hard["minimum_finite_std"]):
        return False, "near_constant_filtered_output"
    return True, ""


def make_window_id(raw_range_id: str, index: int, window_seconds: int) -> str:
    """Create a deterministic window id."""

    return f"{raw_range_id}_w{window_seconds:02d}_{index:04d}"


def process_raw_range(
    root: Path,
    manifest_row: dict[str, str],
    raw_npz_path: Path,
    config: dict[str, Any],
    window_seconds: int,
    preprocessing_config_sha256: str,
    input_snapshot_reference: str,
) -> tuple[list[dict[str, Any]], list[np.ndarray], dict[str, Any]]:
    """Process one raw range into candidate window rows and available arrays."""

    rows: list[dict[str, Any]] = []
    arrays: list[np.ndarray] = []
    source_hash = sha256_file(raw_npz_path)
    raw = load_raw_range_npz(raw_npz_path, manifest_row["raw_range_id"])
    raw_stats = raw_integrity(raw.ppg)
    max_nonfinite = float(
        config["nonfinite_handling"]["max_raw_range_nonfinite_ratio_for_processing"]
    )
    expected_output_samples = int(round(float(config["signal"]["target_fs_hz"]) * window_seconds))

    if raw_stats["nonfinite_ratio"] > max_nonfinite:
        summary = {"status": "failed", "failure_reason": "raw_range_nonfinite_ratio_too_high"}
        return rows, arrays, summary

    try:
        repaired, original_nonfinite_mask, interpolated = interpolate_nonfinite(raw.ppg)
        filtered = filter_ppg(repaired, raw.fs, config)
    except Exception as exc:
        summary = {"status": "failed", "failure_reason": f"filter_or_interpolation_failed:{type(exc).__name__}:{exc}"}
        return rows, arrays, summary

    slices, trailing = iter_window_slices(filtered.size, raw.fs, window_seconds)
    target_fs = float(config["signal"]["target_fs_hz"])
    for window_index, (start, end) in enumerate(slices):
        filtered_slice = filtered[start:end]
        repaired_slice = repaired[start:end]
        original_mask_slice = original_nonfinite_mask[start:end]
        try:
            processed_window, resampled = resample_to_target(
                filtered_slice, raw.fs, target_fs, window_seconds
            )
            window_status = "success"
            resampling_failed = False
        except Exception:
            processed_window = np.asarray([], dtype=np.float32)
            resampled = False
            window_status = "failed"
            resampling_failed = True

        original_ratio = float(original_mask_slice.mean()) if original_mask_slice.size else 1.0
        valid, exclusion_reason = model_input_validity(
            processed_window, original_ratio, expected_output_samples, config
        )
        if resampling_failed:
            valid = False
            exclusion_reason = "resampling_failure"
        morphology = compute_morphology_targets(processed_window, target_fs) if processed_window.size else {
            "svri": np.nan,
            "svri_valid_mask": False,
            "svri_failure_reason": "window_unavailable",
            "sqi_skewness": np.nan,
            "sqi_valid_mask": False,
            "sqi_failure_reason": "window_unavailable",
            "ipa": np.nan,
            "ipa_valid_mask": False,
            "ipa_failure_reason": "window_unavailable",
        }
        aux_morphology_any_available = bool(
            morphology.get("svri_valid_mask")
            or morphology.get("sqi_valid_mask")
            or morphology.get("ipa_valid_mask")
        )
        aux_morphology_all_available = bool(
            morphology.get("svri_valid_mask")
            and morphology.get("sqi_valid_mask")
            and morphology.get("ipa_valid_mask")
        )

        array_index: int | str = ""
        if valid:
            array_index = len(arrays)
            arrays.append(processed_window.astype(np.float32, copy=False))

        row = {
            "window_id": make_window_id(raw.raw_range_id, window_index, window_seconds),
            "parent_raw_range_id": raw.raw_range_id,
            "subject_id": raw.subject_id,
            "session_timestamp": raw.session_timestamp,
            "session_index_within_subject": manifest_row.get("session_index_within_subject", ""),
            "raw_range_index_within_session": manifest_row.get("annotation_range_index_within_session", ""),
            "raw_npz_path": str(raw_npz_path.relative_to(root)),
            "raw_npz_sha256": source_hash,
            "preprocessing_version": config["preprocessing_version"],
            "input_protocol_id": config["input_protocol_id"],
            "comparison_role": config["comparison_role"],
            "preprocessing_profile": config["preprocessing_profile"],
            "native_or_common_input": config["native_or_common_input"],
            "normalization_policy": config["normalization_policy"],
            "preprocessing_config_sha256": preprocessing_config_sha256,
            "input_snapshot_sha256_reference": input_snapshot_reference,
            "window_index_within_raw_range": window_index,
            "window_start_sample_in_raw_range": start,
            "window_end_sample_in_raw_range": end,
            "window_start_seconds_in_raw_range": start / raw.fs,
            "window_end_seconds_in_raw_range": end / raw.fs,
            "source_fs": raw.fs,
            "target_fs": target_fs,
            "window_seconds": window_seconds,
            "expected_output_samples": expected_output_samples,
            "actual_output_samples": int(processed_window.size),
            "resampled": resampled,
            "trailing_remainder_samples_in_parent_range": trailing,
            "raw_range_processing_status": "success",
            "window_processing_status": window_status,
            "model_input_available": valid,
            "common_input_available": valid,
            "exclusion_reason": exclusion_reason,
            "array_index": array_index,
            "original_window_nonfinite_count": int(original_mask_slice.sum()),
            "original_window_nonfinite_ratio": original_ratio,
            "original_window_interpolated": bool(interpolated and original_mask_slice.any()),
            "raw_flatline_ratio_window": flatline_ratio(repaired_slice),
            "filtered_flatline_ratio_window": flatline_ratio(processed_window),
            "filtered_has_nonfinite": not bool(np.all(np.isfinite(processed_window))) if processed_window.size else True,
            **basic_stats(processed_window, "filtered"),
            **morphology,
            "aux_morphology_annotation_available": aux_morphology_any_available,
            "aux_morphology_any_available": aux_morphology_any_available,
            "aux_morphology_all_available": aux_morphology_all_available,
            "raw_subject_eligible_for_10s_cross_session_protocol": "",
            "raw_parent_session_supports_10s_protocol": "",
        }
        rows.append(row)

    summary = {
        "status": "success",
        "failure_reason": "",
        "candidate_windows": len(rows),
        "available_windows": len(arrays),
        "interpolated": interpolated,
        "trailing_remainder_samples": trailing,
        **raw_stats,
    }
    return rows, arrays, summary
