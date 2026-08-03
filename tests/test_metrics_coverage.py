"""Tests for coverage accounting."""

from __future__ import annotations

import numpy as np
import pytest

from fathom.metrics import coverage_fraction, coverage_from_mask


def test_full_coverage_is_one() -> None:
    result = coverage_fraction(8, 8)
    assert result.fraction == pytest.approx(1.0)
    assert result.channels_under_search == 8
    assert result.total_channels == 8


def test_partial_coverage() -> None:
    assert coverage_fraction(6, 8).fraction == pytest.approx(0.75)


def test_coverage_from_mask() -> None:
    mask = np.array([True, True, False, True])
    result = coverage_from_mask(mask)
    assert result.channels_under_search == 3
    assert result.total_channels == 4
    assert result.fraction == pytest.approx(0.75)


def test_coverage_rejects_bad_inputs() -> None:
    with pytest.raises(ValueError):
        coverage_fraction(1, 0)
    with pytest.raises(ValueError):
        coverage_fraction(9, 8)
