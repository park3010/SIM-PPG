"""Window-level morphology targets with validity masks.

The metric names follow PaPaGei's official morphology targets
(`extract_svri`, `skewness_sqi`, `compute_ipa`). This local implementation adds
validity masks and failure reasons so auxiliary losses can ignore invalid
targets without dropping otherwise usable model-input windows.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy.signal import argrelmin
from scipy.stats import skew


def invalid_targets(reason: str, prefix: str) -> dict[str, Any]:
    """Return a NaN target with a False validity mask."""

    return {
        prefix: np.nan,
        f"{prefix}_valid_mask": False,
        f"{prefix}_failure_reason": reason,
    }


def extract_svri(window: np.ndarray, eps: float = 1.0e-8) -> dict[str, Any]:
    """Compute a PaPaGei-style systolic/vascular ratio target."""

    x = np.asarray(window, dtype=float)
    finite = x[np.isfinite(x)]
    if finite.size < 3:
        return invalid_targets("too_few_finite_samples", "svri")
    lo = float(np.min(finite))
    hi = float(np.max(finite))
    if hi - lo <= eps:
        return invalid_targets("constant_signal", "svri")
    scaled = (x - lo) / (hi - lo)
    peak_idx = int(np.nanargmax(scaled))
    if peak_idx <= 0 or peak_idx >= scaled.size - 1:
        return invalid_targets("peak_at_boundary", "svri")
    before = scaled[:peak_idx]
    after = scaled[peak_idx:]
    denom = float(np.nanmean(before))
    if not np.isfinite(denom) or abs(denom) <= eps:
        return invalid_targets("invalid_denominator", "svri")
    value = float(np.nanmean(after) / denom)
    if not np.isfinite(value):
        return invalid_targets("nonfinite_output", "svri")
    return {"svri": value, "svri_valid_mask": True, "svri_failure_reason": ""}


def skewness_sqi(window: np.ndarray) -> dict[str, Any]:
    """Compute skewness SQI with a validity mask."""

    x = np.asarray(window, dtype=float)
    finite = x[np.isfinite(x)]
    if finite.size < 3:
        return invalid_targets("too_few_finite_samples", "sqi")
    if float(np.std(finite)) <= 1.0e-12:
        return invalid_targets("constant_signal", "sqi")
    value = float(skew(finite, bias=True))
    if not np.isfinite(value):
        return invalid_targets("nonfinite_output", "sqi")
    return {"sqi": value, "sqi_valid_mask": True, "sqi_failure_reason": ""}


def compute_ipa(window: np.ndarray, fs: float, eps: float = 1.0e-12) -> dict[str, Any]:
    """Compute PaPaGei-compatible IPA with failure masks.

    PaPaGei's official ``compute_ipa`` identifies the first beat using the
    first two relative minima, then splits that beat at the first internal
    relative minimum and returns the systolic/diastolic area ratio. The
    official function returns 0 on an IndexError. This wrapper preserves the
    same target meaning but exposes failure reasons and masks instead of using
    0 as an ambiguous sentinel value.
    """

    x = np.asarray(window, dtype=float)
    if not np.all(np.isfinite(x)):
        return invalid_targets("nonfinite_input", "ipa")
    if x.size < max(8, int(fs * 1.5)):
        return invalid_targets("too_short", "ipa")
    if float(np.std(x)) <= 1.0e-12:
        return invalid_targets("constant_signal", "ipa")

    order = max(1, int(fs) // 5)
    minima = argrelmin(x, order=order)[0]
    if minima.size < 2:
        return invalid_targets("missing_cycle_boundaries", "ipa")

    single_beat = x[int(minima[0]) : int(minima[1])]
    if single_beat.size < 4:
        return invalid_targets("cycle_too_short", "ipa")
    internal_minima = argrelmin(single_beat)[0]
    if internal_minima.size == 0:
        return invalid_targets("missing_internal_minimum", "ipa")

    split = int(internal_minima[0])
    if split <= 0 or split >= single_beat.size - 1:
        return invalid_targets("invalid_internal_minimum", "ipa")

    sys_values = single_beat[:split]
    dias_values = single_beat[split:]
    if sys_values.size < 2 or dias_values.size < 2:
        return invalid_targets("phase_too_short", "ipa")

    sys_x = np.linspace(0, sys_values.size - 1, sys_values.size)
    dias_x = np.linspace(0, dias_values.size - 1, dias_values.size)
    systolic = float(np.trapz(y=sys_values, x=sys_x))
    diastolic = float(np.trapz(y=dias_values, x=dias_x))
    if not np.isfinite(systolic) or not np.isfinite(diastolic):
        return invalid_targets("nonfinite_area", "ipa")
    if abs(diastolic) <= eps:
        return invalid_targets("invalid_diastolic_area", "ipa")
    value = float(systolic / diastolic)
    if not np.isfinite(value):
        return invalid_targets("nonfinite_output", "ipa")
    if abs(value) <= eps:
        return invalid_targets("zero_ipa_value", "ipa")
    return {"ipa": value, "ipa_valid_mask": True, "ipa_failure_reason": ""}


def compute_morphology_targets(window: np.ndarray, fs: float) -> dict[str, Any]:
    """Compute all morphology/SQI targets for one filtered window."""

    svri = extract_svri(window)
    sqi = skewness_sqi(window)
    ipa = compute_ipa(window, fs)
    return {
        "svri": svri["svri"],
        "svri_valid_mask": svri["svri_valid_mask"],
        "svri_failure_reason": svri["svri_failure_reason"],
        "sqi_skewness": sqi["sqi"],
        "sqi_valid_mask": sqi["sqi_valid_mask"],
        "sqi_failure_reason": sqi["sqi_failure_reason"],
        "ipa": ipa["ipa"],
        "ipa_valid_mask": ipa["ipa_valid_mask"],
        "ipa_failure_reason": ipa["ipa_failure_reason"],
    }
