"""Hand-computed tests for the reject-versus-miss curve and the exact miss bound."""

from __future__ import annotations

import math

import numpy as np
import pytest

from fathom.metrics import (
    miss_rate_upper_bound,
    operating_point,
    reject_miss_curve,
    rule_of_three,
)


def test_zero_miss_bound_matches_closed_form() -> None:
    # With zero misses the exact upper bound is 1 - alpha ** (1 / n).
    for n in (30, 59, 100, 300):
        expected = 1.0 - 0.05 ** (1.0 / n)
        assert miss_rate_upper_bound(0, n) == pytest.approx(expected, rel=1e-9)


def test_zero_miss_bound_is_below_rule_of_three() -> None:
    # The rule-of-three 3/n is a slightly conservative stand-in for the exact zero-miss bound.
    for n in (30, 100, 300):
        assert miss_rate_upper_bound(0, n) < rule_of_three(n)
    assert rule_of_three(30) == pytest.approx(0.1)
    assert rule_of_three(100) == pytest.approx(0.03)
    assert rule_of_three(300) == pytest.approx(0.01)


def test_rule_of_three_tracks_exact_bound_asymptotically() -> None:
    # Both the exact bound and 3/n share the -ln(alpha)/n leading term; -ln(0.05) ~ 2.996.
    n = 100000
    assert miss_rate_upper_bound(0, n) == pytest.approx(-math.log(0.05) / n, rel=1e-3)


def test_all_miss_bound_is_one() -> None:
    assert miss_rate_upper_bound(10, 10) == 1.0


def test_bound_increases_with_observed_misses() -> None:
    values = [miss_rate_upper_bound(k, 50) for k in range(0, 6)]
    assert values == sorted(values)
    assert all(0.0 <= v <= 1.0 for v in values)


def test_bound_specific_value_k1_n10() -> None:
    # Clopper-Pearson upper for 1 miss in 10 trials at 95%: the 0.95 quantile of Beta(2, 9),
    # independently the p that solves the binomial CDF P(X <= 1; n=10, p) = 0.05.
    assert miss_rate_upper_bound(1, 10) == pytest.approx(0.39416330, abs=1e-7)


def test_bound_rejects_bad_inputs() -> None:
    with pytest.raises(ValueError):
        miss_rate_upper_bound(0, 0)
    with pytest.raises(ValueError):
        miss_rate_upper_bound(3, 2)
    with pytest.raises(ValueError):
        rule_of_three(0)


def test_operating_point_counts_misses_and_rejections() -> None:
    # Two present targets (scores 0.9, 0.2) and two clutter (scores 0.1, 0.8) at threshold 0.5.
    present = np.array([True, True, False, False])
    scores = np.array([0.9, 0.2, 0.1, 0.8])
    point = operating_point(present, scores, 0.5)
    assert point.present_trials == 2
    assert point.misses == 1  # the present target scoring 0.2 is missed
    assert point.miss_rate == pytest.approx(0.5)
    assert point.clutter_trials == 2
    assert point.rejection_fraction == pytest.approx(0.5)  # one clutter rejected, one accepted
    assert point.miss_rate_upper == pytest.approx(miss_rate_upper_bound(1, 2))


def test_operating_point_perfect_separation() -> None:
    present = np.array([True, True, True, False, False])
    scores = np.array([0.9, 0.8, 0.95, 0.1, 0.2])
    point = operating_point(present, scores, 0.5)
    assert point.misses == 0
    assert point.miss_rate == 0.0
    assert point.rejection_fraction == pytest.approx(1.0)
    assert point.miss_rate_upper == pytest.approx(1.0 - 0.05 ** (1.0 / 3.0))


def test_curve_sweeps_thresholds_in_order() -> None:
    present = np.array([True, True, False, False])
    scores = np.array([0.9, 0.2, 0.1, 0.8])
    curve = reject_miss_curve(present, scores)
    thresholds = [p.threshold for p in curve]
    assert thresholds == sorted(thresholds)
    assert len(curve) == len(np.unique(scores))
