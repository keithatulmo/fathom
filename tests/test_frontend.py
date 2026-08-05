"""Tests for the sweepable linear front-end family (SD4 / WO-6)."""

from __future__ import annotations

import numpy as np

from fathom.frontend import FrontEndParams, compute_cost_proxy, lofar_gram

_RATE = 1000.0


def _params(
    window_length_s: float = 4.0, taper: str = "hann", integration: int = 1
) -> FrontEndParams:
    return FrontEndParams(
        window_length_s=window_length_s,
        overlap=0.5,
        integration_count=integration,
        taper=taper,
        n_tapers=5,
        nw=3.0,
        normalizer="split_window",
        sw_half_window_hz=6.0,
        sw_guard_hz=1.0,
        sw_truncation=3,
        tm_window_frames=8,
    )


def _tone(freq: float, n: int, seed: int) -> np.ndarray:
    t = np.arange(n) / _RATE
    gen = np.random.default_rng(seed)
    return np.ascontiguousarray(
        0.3 * gen.standard_normal(n) + 0.5 * np.sin(2.0 * np.pi * freq * t), dtype=np.float64
    )


def test_window_length_sets_resolution() -> None:
    signal = _tone(30.0, int(_RATE * 20), seed=1)
    _, freqs2 = lofar_gram(signal, _RATE, _params(window_length_s=2.0))
    _, freqs4 = lofar_gram(signal, _RATE, _params(window_length_s=4.0))
    # A longer window gives a finer frequency bin: about the reciprocal of the window length.
    assert abs((freqs2[1] - freqs2[0]) - 0.5) < 1e-6
    assert abs((freqs4[1] - freqs4[0]) - 0.25) < 1e-6


def test_gram_is_bit_level_deterministic() -> None:
    signal = _tone(30.0, int(_RATE * 15), seed=2)
    for taper in ("hann", "multitaper"):
        a, _ = lofar_gram(signal, _RATE, _params(taper=taper))
        b, _ = lofar_gram(signal, _RATE, _params(taper=taper))
        assert np.array_equal(a, b)


def test_multitaper_reduces_variance() -> None:
    noise = np.ascontiguousarray(np.random.default_rng(3).standard_normal(int(_RATE * 15)))
    hann, _ = lofar_gram(noise, _RATE, _params(taper="hann"))
    multi, _ = lofar_gram(noise, _RATE, _params(taper="multitaper"))
    # The multitaper estimate has lower relative variance than the single-taper periodogram.
    assert multi.std() / multi.mean() < hann.std() / hann.mean()


def test_integration_reduces_frame_count() -> None:
    signal = _tone(30.0, int(_RATE * 20), seed=4)
    one, _ = lofar_gram(signal, _RATE, _params(integration=1))
    four, _ = lofar_gram(signal, _RATE, _params(integration=4))
    assert four.shape[1] < one.shape[1]
    assert four.shape[0] == one.shape[0]  # same frequency axis


def test_compute_cost_grows_with_tapers() -> None:
    assert compute_cost_proxy(_params(taper="multitaper"), _RATE, 20.0) > compute_cost_proxy(
        _params(taper="hann"), _RATE, 20.0
    )
