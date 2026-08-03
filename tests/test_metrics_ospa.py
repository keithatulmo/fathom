"""Hand-computed tests for OSPA and windowed OSPA(2)."""

from __future__ import annotations

import numpy as np
import pytest

from fathom.metrics import ospa, windowed_ospa2


def test_ospa_single_pair_is_euclidean() -> None:
    x = np.array([[0.0, 0.0]])
    y = np.array([[3.0, 4.0]])
    assert ospa(x, y, c=100.0) == pytest.approx(5.0)


def test_ospa_truncates_at_cutoff() -> None:
    x = np.array([[0.0, 0.0]])
    y = np.array([[3.0, 4.0]])
    assert ospa(x, y, c=2.0) == pytest.approx(2.0)


def test_ospa_both_empty_is_zero() -> None:
    empty = np.zeros((0, 2))
    assert ospa(empty, empty, c=10.0) == 0.0


def test_ospa_one_empty_is_cutoff() -> None:
    assert ospa(np.array([[0.0]]), np.zeros((0, 1)), c=10.0) == pytest.approx(10.0)


def test_ospa_cardinality_penalty() -> None:
    # One matched pair at distance 0 and one unmatched point penalized at c: (0 + 5) / 2.
    x = np.array([[0.0], [10.0]])
    y = np.array([[0.0]])
    assert ospa(x, y, c=5.0) == pytest.approx(2.5)


def test_ospa_order_two() -> None:
    x = np.array([[0.0, 0.0]])
    y = np.array([[3.0, 4.0]])
    assert ospa(x, y, c=100.0, p=2.0) == pytest.approx(5.0)


def test_ospa2_matched_track_over_window() -> None:
    gt = {0: {0: np.array([0.0]), 1: np.array([0.0])}}
    est = {7: {0: np.array([1.0]), 1: np.array([1.0])}}
    # Distance 1 at each of two frames, averaged to 1; single track each, no cardinality penalty.
    assert windowed_ospa2(gt, est, [0, 1], c=10.0) == pytest.approx(1.0)


def test_ospa2_charges_cutoff_for_missing_existence() -> None:
    gt = {0: {0: np.array([0.0]), 1: np.array([0.0])}}
    est = {7: {0: np.array([0.0])}}  # exists only at frame 0
    # Frame 0 distance 0, frame 1 charges the cutoff 10; average is 5.
    assert windowed_ospa2(gt, est, [0, 1], c=10.0) == pytest.approx(5.0)


def test_ospa2_empty_cases() -> None:
    gt: dict[int, dict[int, np.ndarray]] = {}
    est = {0: {0: np.array([0.0])}}
    assert windowed_ospa2(gt, gt, [0], c=5.0) == 0.0
    assert windowed_ospa2(gt, est, [0], c=5.0) == pytest.approx(5.0)
