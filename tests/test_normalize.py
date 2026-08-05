"""Tests for the split-window and temporal-median normalizers (SD4 / WO-6)."""

from __future__ import annotations

import numpy as np

from fathom.frontend import FrontEndParams, lofar_gram
from fathom.normalize import normalize_gram

_RATE = 1000.0
_BAND = (4.0, 150.0)


def _params(normalizer: str = "split_window") -> FrontEndParams:
    return FrontEndParams(
        window_length_s=4.0,
        overlap=0.5,
        integration_count=1,
        taper="hann",
        n_tapers=5,
        nw=3.0,
        normalizer=normalizer,
        sw_half_window_hz=6.0,
        sw_guard_hz=1.0,
        sw_truncation=3,
        tm_window_frames=8,
    )


def _red_noise_with_comb(seed: int, combs: tuple[float, ...]) -> np.ndarray:
    n = int(_RATE * 25)
    gen = np.random.default_rng(seed)
    white = gen.standard_normal(n)
    spec = np.fft.rfft(white)
    freqs = np.fft.rfftfreq(n, 1.0 / _RATE)
    freqs[0] = freqs[1]
    spec *= (freqs / freqs[len(freqs) // 2]) ** (-1.0)  # red background
    background = np.fft.irfft(spec, n=n)
    background /= background.std()
    t = np.arange(n) / _RATE
    comb = sum(0.4 * np.sin(2.0 * np.pi * f * t) for f in combs)
    return np.ascontiguousarray(0.3 * background + 0.15 * comb, dtype=np.float64)


def test_split_window_flattens_colored_background() -> None:
    background = _red_noise_with_comb(1, combs=())
    gram, freqs = lofar_gram(background, _RATE, _params("split_window"))
    normalized, _ = normalize_gram(gram, freqs, _BAND, _params("split_window"))
    # The colored background normalizes to a flat unit level: median near one, modest spread.
    assert 0.7 < float(np.median(normalized)) < 1.4
    assert float(np.median(normalized.mean(axis=1))) < 2.0


def test_split_window_preserves_a_line() -> None:
    signal = _red_noise_with_comb(2, combs=(20.0, 40.0))
    gram, freqs = lofar_gram(signal, _RATE, _params("split_window"))
    normalized, nf = normalize_gram(gram, freqs, _BAND, _params("split_window"))
    line_bin = int(np.argmin(np.abs(nf - 20.0)))
    off_bin = int(np.argmin(np.abs(nf - 27.0)))
    # A real tonal survives the normalizer and stands well above the flattened background.
    assert float(np.max(normalized[line_bin])) > 4.0 * float(np.median(normalized[off_bin]))


def test_split_window_handles_the_one_sided_floor() -> None:
    signal = _red_noise_with_comb(3, combs=(4.0, 8.0))
    gram, freqs = lofar_gram(signal, _RATE, _params("split_window"))
    normalized, nf = normalize_gram(gram, freqs, _BAND, _params("split_window"))
    # The lowest in-band bins, where the split window is one-sided, still produce finite output.
    assert np.all(np.isfinite(normalized))
    assert nf[0] <= 5.0


def test_temporal_median_flattens_and_is_deterministic() -> None:
    background = _red_noise_with_comb(4, combs=())
    gram, freqs = lofar_gram(background, _RATE, _params("temporal_median"))
    a, _ = normalize_gram(gram, freqs, _BAND, _params("temporal_median"))
    b, _ = normalize_gram(gram, freqs, _BAND, _params("temporal_median"))
    assert np.array_equal(a, b)
    assert 0.5 < float(np.median(a)) < 1.6
