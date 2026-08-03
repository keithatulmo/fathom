"""The SD3 metric estimators, each written fresh from its definition and honesty condition."""

from __future__ import annotations

from .bootstrap import BootstrapResult, cluster_bootstrap
from .calibration import CalibrationResult, ReliabilityBin, expected_calibration_error
from .coverage import CoverageResult, coverage_fraction, coverage_from_mask
from .custody import label_consistency, loss_count, swap_count, time_to_reacquire
from .ospa import ospa, windowed_ospa2
from .reject_miss import (
    RejectMissPoint,
    miss_rate_upper_bound,
    operating_point,
    reject_miss_curve,
    rule_of_three,
)

__all__ = [
    "BootstrapResult",
    "CalibrationResult",
    "CoverageResult",
    "ReliabilityBin",
    "RejectMissPoint",
    "cluster_bootstrap",
    "coverage_fraction",
    "coverage_from_mask",
    "expected_calibration_error",
    "label_consistency",
    "loss_count",
    "miss_rate_upper_bound",
    "operating_point",
    "ospa",
    "reject_miss_curve",
    "rule_of_three",
    "swap_count",
    "time_to_reacquire",
    "windowed_ospa2",
]
