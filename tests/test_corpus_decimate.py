"""Tests for the decimation derivative and audio codec helpers."""

from __future__ import annotations

import numpy as np
import pytest

from fathom.corpus.decimate import decimate, encode_wav, read_audio


def test_decimation_reduces_rate_and_length() -> None:
    rate_in = 4000.0
    n = 4000
    signal = np.sin(2 * np.pi * 50.0 * np.arange(n) / rate_in)[None, :]
    out = decimate(signal, rate_in, target_rate=1000.0)
    assert out.sample_rate_hz == 1000.0
    assert out.samples.shape[0] == 1
    assert out.samples.shape[1] == 1000
    assert out.params["method"] == "resample_poly"
    assert out.params["antialias_cutoff_hz"] == 500.0
    assert out.params["up"] == 1
    assert out.params["down"] == 4


def test_passthrough_when_rate_at_or_below_target() -> None:
    signal = np.ones((2, 100))
    out = decimate(signal, 500.0, target_rate=1000.0)
    assert out.params["method"] == "passthrough"
    assert out.sample_rate_hz == 500.0
    # A 500 Hz source has a 250 Hz Nyquist, below the 300 Hz band top, so coverage is partial.
    assert out.params["band_partial"] is True
    assert out.duration_s == pytest.approx(100 / 500.0)


def test_full_band_not_partial() -> None:
    out = decimate(np.ones((1, 100)), 2000.0, target_rate=1000.0)
    assert out.params["band_partial"] is False


def test_wav_roundtrip() -> None:
    rate = 1000.0
    signal = (0.5 * np.sin(2 * np.pi * 40.0 * np.arange(500) / rate))[None, :]
    blob = encode_wav(signal, rate)
    restored, restored_rate = read_audio(blob)
    assert restored_rate == rate
    assert restored.shape == signal.shape
    assert np.allclose(restored, signal, atol=1e-6)
