"""Tests for in-band SNR at closest approach, on synthetic recordings."""

from __future__ import annotations

import math

import numpy as np

from fathom.corpus.decimate import encode_wav
from fathom.corpus.snr import measure_inband_snr


def _recording(sample_rate: float, duration_s: float, tone_center_s: float | None) -> bytes:
    rng = np.random.default_rng(0)
    frames = int(sample_rate * duration_s)
    signal = rng.normal(0.0, 0.01, size=frames)
    if tone_center_s is not None:
        t = np.arange(frames) / sample_rate
        # A strong in-band (50 Hz) tone, on only within +/- 30 s of the closest-approach time.
        mask = np.abs(t - tone_center_s) <= 30.0
        signal = signal + mask * 0.3 * np.sin(2.0 * math.pi * 50.0 * t)
    return encode_wav(signal[np.newaxis, :], sample_rate)


def test_loud_close_passage_has_high_snr() -> None:
    blob = _recording(2000.0, 300.0, tone_center_s=150.0)
    result = measure_inband_snr(blob, closest_offset_s=150.0)
    assert result.reference_count > 5
    assert result.snr_db > 12.0  # a clearly audible passage


def test_flat_noise_has_low_snr() -> None:
    blob = _recording(2000.0, 300.0, tone_center_s=None)
    result = measure_inband_snr(blob, closest_offset_s=150.0)
    assert math.isfinite(result.snr_db)
    assert result.snr_db < 6.0  # nothing to hear: near the ambient floor


def test_short_recording_is_unverified_nan() -> None:
    # Too short to form an analysis segment (well under 64 samples), so the result is unverified.
    blob = _recording(2000.0, 0.02, tone_center_s=None)
    result = measure_inband_snr(blob, closest_offset_s=0.01)
    assert math.isnan(result.snr_db)
