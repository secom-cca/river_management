"""Explicit, shared assumptions for the to7 breeding response (no PySD dependency)."""

from __future__ import annotations

import hashlib
import math
from pathlib import Path

MODEL_REVISION = "to7-calendar-breeding-v2"
# Adopted on 2026-10-08: effects begin only after the investment target is met.
DEFAULT_MODE = "threshold"
TARGET_YEN = 500_000_000.0
TOLERANCE_YEN = 0.01
MIDPOINT_YEN = 300_000_000.0
SCALE_YEN = 70_000_000.0
MAX_REDUCTION = 0.5


def logistic_effect_ratio(investment, midpoint=MIDPOINT_YEN, scale=SCALE_YEN):
    """Zero-anchored progress, tending to one; this is not a success probability."""
    investment, midpoint, scale = map(float, (investment, midpoint, scale))
    if not all(math.isfinite(x) for x in (investment, midpoint, scale)):
        raise ValueError("Breeding response arguments must be finite")
    if midpoint < 0 or scale <= 0:
        raise ValueError("Breeding midpoint must be nonnegative and scale positive")
    if investment <= 0:
        return 0.0

    def sigmoid(value):
        if value >= 0:
            return 1.0 / (1.0 + math.exp(-value))
        exp_value = math.exp(value)
        return exp_value / (1.0 + exp_value)

    initial = sigmoid(-midpoint / scale)
    return min(1.0, max(0.0, (sigmoid((investment - midpoint) / scale) - initial) / (1.0 - initial)))


def model_parameters(mode=DEFAULT_MODE, *, midpoint=MIDPOINT_YEN, scale=SCALE_YEN, max_reduction=MAX_REDUCTION):
    if mode not in ("threshold", "logistic"):
        raise ValueError("Breeding mode must be threshold or logistic")
    logistic_effect_ratio(0.0, midpoint, scale)
    if not math.isfinite(max_reduction) or not 0 <= max_reduction <= 1:
        raise ValueError("Maximum chalky-kernel reduction must be between zero and one")
    return {
        "breeding_effect_mode": int(mode == "logistic"),
        "breeding_investment_target": TARGET_YEN,
        "breeding_investment_tolerance": TOLERANCE_YEN,
        "breeding_logistic_midpoint": float(midpoint),
        "breeding_logistic_scale": float(scale),
        "breeding_max_quality_reduction": float(max_reduction),
    }


def model_configuration():
    """Prevent resuming trials produced by a different model or aggregation."""
    root = Path(__file__).resolve().parent
    names = ("River_management_xls_to7.py", "run_vensim_with_pysd_to7_opt.py", "breeding_response_to7.py")
    return {
        "revision": MODEL_REVISION,
        "breeding_mode": DEFAULT_MODE,
        "parameters": model_parameters(),
        "year_end_convention": "next January 1 after integrating December 31",
        "source_sha256": {name: hashlib.sha256((root / name).read_bytes()).hexdigest() for name in names},
    }
