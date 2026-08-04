"""Tests for kinematics, propagation, synthesis, and signal-to-noise leveling."""

from __future__ import annotations

import numpy as np

from fathom.determinism import rng
from fathom.surrogate.kinematics import KinematicParams, evaluate_track
from fathom.surrogate.level import inject_at_snr
from fathom.surrogate.lines import MachineryParams, build_lines
from fathom.surrogate.propagation import PropagationParams, range_envelope, spectral_gain
from fathom.surrogate.synth import synthesize

_SAMPLE_RATE = 2048.0
_DURATION = 2.0
_BAND = (2.0, 400.0)


def _machinery() -> MachineryParams:
    return MachineryParams(
        shaft_hz=4.0,
        shaft_harmonics=3,
        blade_count=5,
        blade_harmonics=2,
        electrical_hz=60.0,
        electrical_harmonics=2,
        aux_hz=(35.0, 120.0),
        shaft_amp=1.0,
        blade_amp=0.8,
        electrical_amp=0.5,
        aux_amp=0.3,
        harmonic_rolloff=0.6,
        line_width_hz=0.3,
        wander_hz=0.1,
    )


def _kinematics() -> KinematicParams:
    return KinematicParams(
        bearing0_deg=90.0,
        closest_proxy=1.0,
        speed_proxy=0.5,
        t_cpa_s=_DURATION / 2.0,
        doppler_peak=0.02,
    )


def _propagation() -> PropagationParams:
    return PropagationParams(
        absorption_coeff=0.001,
        multipath_depth=0.3,
        multipath_spacing_hz=90.0,
        spreading_exponent=1.0,
    )


def _times() -> np.ndarray:
    n = int(round(_DURATION * _SAMPLE_RATE))
    return np.arange(n, dtype=np.float64) / _SAMPLE_RATE


def test_track_is_unity_doppler_and_closest_at_cpa() -> None:
    track = evaluate_track(_kinematics(), _times())
    cpa_index = int(np.argmin(track.range_proxy))
    assert np.isclose(track.range_proxy[cpa_index], 1.0, atol=1e-6)
    assert np.isclose(track.doppler_factor[cpa_index], 1.0, atol=1e-3)
    # Approaching, the line is shifted up (doppler > 1); receding, down (< 1).
    assert track.doppler_factor[0] > 1.0
    assert track.doppler_factor[-1] < 1.0
    assert track.bearings_deg[-1] > track.bearings_deg[0]


def test_propagation_rolls_off_and_range_envelope_peaks_at_cpa() -> None:
    prop = _propagation()
    gains = spectral_gain(np.array([10.0, 400.0]), prop)
    assert gains[0] > gains[1]  # absorption attenuates the higher line more
    track = evaluate_track(_kinematics(), _times())
    envelope = range_envelope(track.range_proxy, float(track.range_proxy.min()), prop)
    assert np.isclose(envelope.max(), 1.0, atol=1e-6)
    assert envelope[int(np.argmin(track.range_proxy))] == envelope.max()


def test_synthesis_places_energy_at_line_frequencies() -> None:
    lines = build_lines(_machinery())
    track = evaluate_track(_kinematics(), _times())
    signal = synthesize(lines, track, _propagation(), _SAMPLE_RATE, _DURATION, seed=7)
    spectrum = np.abs(np.fft.rfft(signal))
    freqs = np.fft.rfftfreq(signal.shape[0], d=1.0 / _SAMPLE_RATE)
    peak_hz = freqs[int(np.argmax(spectrum))]
    # The strongest line is the shaft fundamental at 4 Hz; the peak sits within a bin or two of it.
    assert abs(peak_hz - 4.0) < 2.0


def test_synthesis_is_deterministic_under_seed() -> None:
    lines = build_lines(_machinery())
    track = evaluate_track(_kinematics(), _times())
    a = synthesize(lines, track, _propagation(), _SAMPLE_RATE, _DURATION, seed=11)
    b = synthesize(lines, track, _propagation(), _SAMPLE_RATE, _DURATION, seed=11)
    assert np.array_equal(a, b)


def test_achieved_snr_within_tolerance_of_request() -> None:
    lines = build_lines(_machinery())
    track = evaluate_track(_kinematics(), _times())
    signal = synthesize(lines, track, _propagation(), _SAMPLE_RATE, _DURATION, seed=3)
    background = np.asarray(rng(99).standard_normal(signal.shape[0]) * 0.5, dtype=np.float64)
    for requested in (-6.0, 0.0, 6.0, 12.0):
        _, achieved = inject_at_snr(signal, background, _BAND, _SAMPLE_RATE, requested)
        assert abs(achieved - requested) < 1.0, (requested, achieved)
