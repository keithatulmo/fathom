"""Tests for the cluster bootstrap over the exchangeable unit."""

from __future__ import annotations

import numpy as np
import pytest

from fathom.metrics import cluster_bootstrap


def _mean(values: np.ndarray) -> float:
    return float(np.mean(values))


def test_point_estimate_is_pooled_statistic() -> None:
    units = [np.array([0.0]), np.array([1.0]), np.array([1.0]), np.array([1.0])]
    result = cluster_bootstrap(units, _mean, seed=1, n_resamples=100)
    assert result.estimate == pytest.approx(0.75)
    assert result.ci_low <= result.estimate <= result.ci_high


def test_seed_makes_interval_reproducible() -> None:
    units = [np.array([0.0, 0.1]), np.array([0.9, 1.0]), np.array([0.5])]
    a = cluster_bootstrap(units, _mean, seed=7, n_resamples=250)
    b = cluster_bootstrap(units, _mean, seed=7, n_resamples=250)
    assert a == b


def test_different_seeds_can_differ() -> None:
    units = [np.array([0.0]), np.array([1.0]), np.array([2.0]), np.array([3.0])]
    a = cluster_bootstrap(units, _mean, seed=1, n_resamples=250)
    b = cluster_bootstrap(units, _mean, seed=2, n_resamples=250)
    assert (a.ci_low, a.ci_high) != (b.ci_low, b.ci_high)


def test_degenerate_units_collapse_interval() -> None:
    units = [np.array([2.0]), np.array([2.0]), np.array([2.0])]
    result = cluster_bootstrap(units, _mean, seed=3, n_resamples=100)
    assert result.ci_low == pytest.approx(2.0)
    assert result.ci_high == pytest.approx(2.0)


def test_resampling_is_by_unit_not_item() -> None:
    # Two units of very different means; resampling whole units yields a wide interval that
    # includes both unit means, which clip-level resampling would understate.
    units = [np.zeros(50), np.ones(50)]
    result = cluster_bootstrap(units, _mean, seed=5, n_resamples=500)
    assert result.ci_low == pytest.approx(0.0)
    assert result.ci_high == pytest.approx(1.0)


def test_invalid_inputs_raise() -> None:
    with pytest.raises(ValueError):
        cluster_bootstrap([], _mean, seed=1)
    with pytest.raises(ValueError):
        cluster_bootstrap([np.array([1.0])], _mean, seed=1, n_resamples=0)
    with pytest.raises(ValueError):
        cluster_bootstrap([np.array([1.0])], _mean, seed=1, alpha=1.5)
