"""Tests for line-statistic extraction and the train-audio fit (WO-4)."""

from __future__ import annotations

import math

import numpy as np
import pytest

from fathom.determinism import rng
from fathom.surrogate.extract import extract_line_stats, fit_from_segments
from fathom.surrogate.fit import ForbiddenSplitError

_BAND = (4.0, 150.0)


def _segment(sample_rate: float, duration_s: float, lines: list[tuple[float, float]]) -> np.ndarray:
    n = int(sample_rate * duration_s)
    t = np.arange(n) / sample_rate
    signal = 0.02 * rng(0).standard_normal(n)
    for freq, amp in lines:
        signal = signal + amp * np.sin(2.0 * math.pi * freq * t)
    return np.ascontiguousarray(signal, dtype=np.float64)


def test_extraction_recovers_lines_and_rolloff() -> None:
    # Three lines at 20/40/60 Hz with falling amplitude, so the roll-off is below one.
    seg = _segment(2000.0, 8.0, [(20.0, 1.0), (40.0, 0.5), (60.0, 0.25)])
    stats = extract_line_stats(seg, 2000.0, _BAND)
    assert stats.n_lines >= 2
    assert 0.3 <= stats.rolloff < 1.0
    assert stats.width_hz > 0.0
    assert len(stats.peak_amplitudes) == stats.n_lines


def test_extraction_on_pure_noise_finds_few_lines() -> None:
    seg = _segment(2000.0, 8.0, [])
    stats = extract_line_stats(seg, 2000.0, _BAND)
    assert stats.n_lines <= 3  # no strong narrowband structure to find


def _entry(vessel_id: str, split: str, lines: list[tuple[float, float]]) -> dict[str, object]:
    return {
        "vessel_id": vessel_id,
        "split": split,
        "sample_rate": 2000.0,
        "samples": _segment(2000.0, 8.0, lines),
        "is_quiet_tail": True,
    }


def test_fit_from_train_audio_sets_provenance_and_roster() -> None:
    train = [
        _entry("T0", "train", [(20.0, 1.0), (40.0, 0.5)]),
        _entry("T1", "train", [(25.0, 0.8), (50.0, 0.3)]),
        _entry("T2", "train", [(30.0, 0.6), (60.0, 0.2)]),
    ]
    dist = fit_from_segments(train, _BAND)
    assert dist.provenance["amplitude_statistics"] == "train_audio"
    assert dist.train_vessel_count == 3
    assert dist.train_vessel_ids == ("T0", "T1", "T2")
    # The fitted amplitude ranges are ordered shaft >= aux, so quiet draws stay low.
    assert dist.shaft_amp_range[1] >= dist.aux_amp_range[0]


def test_fit_refuses_a_held_out_segment() -> None:
    # The leak guard on the audio fit: a test-split segment raises rather than being read.
    with pytest.raises(ForbiddenSplitError):
        fit_from_segments([_entry("H0", "test", [(20.0, 1.0)])], _BAND)
    with pytest.raises(ForbiddenSplitError):
        fit_from_segments(
            [_entry("T0", "train", [(20.0, 1.0)]), _entry("H0", "test", [(20.0, 1.0)])], _BAND
        )
