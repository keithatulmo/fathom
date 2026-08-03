"""Hand-computed tests for equal-mass expected calibration error."""

from __future__ import annotations

import numpy as np
import pytest

from fathom.metrics import expected_calibration_error


def test_two_bin_hand_example() -> None:
    # Sorted into two equal-mass bins: {0.2,0.4} accuracy 0, {0.6,0.8} accuracy 1.
    # ECE = 0.5*|0-0.3| + 0.5*|1-0.7| = 0.3.
    conf = np.array([0.2, 0.4, 0.6, 0.8])
    correct = np.array([False, False, True, True])
    result = expected_calibration_error(conf, correct, n_bins=2, min_bin_count=1)
    assert result.ece == pytest.approx(0.3)
    assert result.n_bins_effective == 2
    assert result.bins[0].mean_confidence == pytest.approx(0.3)
    assert result.bins[0].accuracy == pytest.approx(0.0)
    assert result.bins[1].mean_confidence == pytest.approx(0.7)
    assert result.bins[1].accuracy == pytest.approx(1.0)


def test_perfect_calibration_is_zero() -> None:
    # Confidence equals accuracy within every bin.
    conf = np.array([0.0, 0.0, 1.0, 1.0])
    correct = np.array([False, False, True, True])
    result = expected_calibration_error(conf, correct, n_bins=2, min_bin_count=1)
    assert result.ece == pytest.approx(0.0)


def test_effective_bins_respects_floor() -> None:
    # Ten samples with a floor of five per bin admit at most two bins.
    conf = np.linspace(0.0, 1.0, 10)
    correct = np.array([False] * 5 + [True] * 5)
    result = expected_calibration_error(conf, correct, n_bins=15, min_bin_count=5)
    assert result.n_bins_effective == 2


def test_single_bin_when_data_scarce() -> None:
    conf = np.array([0.4, 0.6])
    correct = np.array([True, False])
    result = expected_calibration_error(conf, correct, n_bins=15, min_bin_count=5)
    assert result.n_bins_effective == 1
    # One bin: mean confidence 0.5, accuracy 0.5, gap 0.
    assert result.ece == pytest.approx(0.0)


def test_empty_input_is_zero() -> None:
    result = expected_calibration_error(
        np.array([], dtype=np.float64), np.array([], dtype=np.bool_)
    )
    assert result.ece == 0.0
    assert result.n_bins_effective == 0


def test_out_of_range_confidence_raises() -> None:
    with pytest.raises(ValueError):
        expected_calibration_error(np.array([1.5]), np.array([True]))
