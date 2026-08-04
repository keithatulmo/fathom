"""Tests for the WO-5 hybrid surrogate: the broadband continuum fit, synthesis, and bounding.

These cover the WO-5 acceptance points that can be checked without the ledger or R2: the continuum
is a non-empty between-line stream with the line-to-broadband ratio a fitted distribution, the fit
is leak-guarded to the train side, the quiet-target mode keeps the continuum while dropping the
propulsion lines, no absolute level appears, and the continuum bounds the surrogate's line
prominence where the pure-narrowband surrogate's climbed without bound.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from fathom.determinism import derive_seed, rng
from fathom.e1 import channel_responses
from fathom.surrogate.broadband import (
    BroadbandParams,
    extract_broadband_stats,
)
from fathom.surrogate.extract import fit_from_segments
from fathom.surrogate.fit import (
    ForbiddenSplitError,
    SurrogateDistributions,
    draw_broadband,
    draw_machinery,
)
from fathom.surrogate.kinematics import KinematicParams, evaluate_track
from fathom.surrogate.level import inband_power, inject_at_snr
from fathom.surrogate.lines import build_lines
from fathom.surrogate.propagation import PropagationParams
from fathom.surrogate.synth import add_broadband, synthesize, synthesize_continuum

_RATE = 1000.0
_BAND = (4.0, 150.0)


def _noise_with_lines(seed: int, lines: tuple[float, ...], line_gain: float) -> np.ndarray:
    """A 1 kHz segment of red-ish broadband noise plus a few strong tonal lines."""
    n = int(_RATE * 8.0)
    t = np.arange(n) / _RATE
    gen = rng(seed)
    signal = 0.3 * gen.standard_normal(n)
    for i, freq in enumerate(lines):
        signal = signal + line_gain * (0.7**i) * np.sin(2.0 * math.pi * freq * t)
    return np.ascontiguousarray(signal, dtype=np.float64)


def test_continuum_is_band_limited_shaped_and_unit_power() -> None:
    continuum = synthesize_continuum(8000, _RATE, _BAND, exponent=-1.0, seed=3)
    in_power = inband_power(continuum, _RATE, _BAND)
    full_power = float(np.sum(np.fft.rfft(continuum).real ** 2 + np.fft.rfft(continuum).imag ** 2))
    assert in_power == pytest.approx(1.0, rel=1e-6)  # returned at unit in-band power
    # Almost all energy is in-band: the out-of-band leakage is a small fraction of the total.
    freqs = np.fft.rfftfreq(8000, d=1.0 / _RATE)
    spec = np.fft.rfft(continuum)
    out = (freqs < _BAND[0]) | (freqs > _BAND[1])
    out_power = float(np.sum(spec[out].real ** 2 + spec[out].imag ** 2))
    assert out_power / full_power < 0.02


def test_extract_broadband_ratio_orders_clean_above_quiet() -> None:
    clean = extract_broadband_stats(_noise_with_lines(1, (20.0, 40.0, 60.0), 3.0), _RATE, _BAND)
    quiet = extract_broadband_stats(_noise_with_lines(1, (20.0, 40.0, 60.0), 0.3), _RATE, _BAND)
    # A strong-line vessel has a higher line-to-broadband ratio than a broadband-dominated one.
    assert clean.lbr_db > quiet.lbr_db
    # The continuum exponent is finite and inside the clamped physical band.
    assert -3.0 <= clean.continuum_exponent <= 0.0
    assert clean.n_floor_bins > 0


def test_extract_broadband_no_lines_gives_broadband_default() -> None:
    pure_noise = np.ascontiguousarray(0.3 * rng(5).standard_normal(8000), dtype=np.float64)
    stats = extract_broadband_stats(pure_noise, _RATE, _BAND)
    assert stats.lbr_db == 0.0  # broadband-dominated default when no lines are detected


def _train(vessel_id: str, seed: int) -> dict[str, object]:
    return {
        "vessel_id": vessel_id,
        "split": "train",
        "sample_rate": _RATE,
        "samples": _noise_with_lines(seed, (20.0, 40.0, 60.0), 2.0),
    }


def test_fit_carries_broadband_ranges_and_provenance() -> None:
    train = [_train(f"T{i}", seed=i) for i in range(5)]
    dist = fit_from_segments(train, _BAND)
    assert dist.provenance["broadband_statistics"] == "train_audio"
    fit = dist.provenance["broadband_fit"]
    assert len(fit["per_vessel"]) == 5
    assert set(fit["per_vessel"][0]) == {
        "vessel_id",
        "continuum_exponent",
        "line_to_broadband_db",
        "n_floor_bins",
    }
    assert "ForbiddenSplitError" in fit["leak_guard"]
    lo, hi = dist.lbr_db_range
    assert hi >= lo  # a real fitted distribution, not a point
    e_lo, e_hi = dist.continuum_exponent_range
    assert -3.0 <= e_lo <= e_hi <= 0.0


def test_broadband_fit_is_leak_guarded() -> None:
    # A held-out segment mixed into the fit raises before any statistic is used, line or broadband.
    train = [_train("T0", 0), {**_train("H0", 1), "split": "test"}]
    with pytest.raises(ForbiddenSplitError):
        fit_from_segments(train, _BAND)


def _hybrid_signal(dist: SurrogateDistributions, n: int, seed: int, quiet_only: bool) -> np.ndarray:
    prop = PropagationParams(
        absorption_coeff=0.001,
        multipath_depth=0.3,
        multipath_spacing_hz=90.0,
        spreading_exponent=1.0,
    )
    kin = KinematicParams(
        bearing0_deg=90.0,
        closest_proxy=1.0,
        speed_proxy=0.5,
        t_cpa_s=0.5 * n / _RATE,
        doppler_peak=0.02,
    )
    track = evaluate_track(kin, np.arange(n, dtype=np.float64) / _RATE)
    machinery = draw_machinery(dist, derive_seed(seed, "m"))
    if quiet_only:
        machinery = type(machinery)(
            **{**machinery.to_dict(), "quiet_mode": True, "propulsion_absent": True}
        )
    line_signal = synthesize(
        build_lines(machinery), track, prop, _RATE, n / _RATE, derive_seed(seed, "s")
    )
    bb = draw_broadband(dist, derive_seed(seed, "b"))
    continuum = synthesize_continuum(n, _RATE, _BAND, bb.continuum_exponent, derive_seed(seed, "c"))
    return add_broadband(line_signal, continuum, _BAND, _RATE, bb.lbr_db)


def test_quiet_mode_retains_broadband_continuum() -> None:
    # A quiet target with the propulsion lines dropped still carries a broadband continuum: its
    # in-band power is not concentrated in a handful of line bins but spread across the band.
    dist = SurrogateDistributions(
        quiet_fraction=1.0, train_vessel_count=3, train_vessel_ids=("A", "B", "C")
    )
    n = 8000
    sig = _hybrid_signal(dist, n, seed=2, quiet_only=True)
    spec = np.fft.rfft(sig)
    freqs = np.fft.rfftfreq(n, d=1.0 / _RATE)
    in_band = (freqs >= _BAND[0]) & (freqs <= _BAND[1])
    per_bin = spec[in_band].real ** 2 + spec[in_band].imag ** 2
    # The strongest few bins hold well under the whole in-band power: a broadband floor is present,
    # not a near-silent line-only object.
    top = float(np.sort(per_bin)[-5:].sum())
    assert top / float(per_bin.sum()) < 0.9
    assert inband_power(sig, _RATE, _BAND) > 0.0


def test_hybrid_bounds_line_prominence_where_narrowband_diverged() -> None:
    dist = SurrogateDistributions(
        quiet_fraction=0.3, train_vessel_count=5, train_vessel_ids=tuple("ABCDE")
    )
    n = 8000
    background = np.ascontiguousarray(0.3 * rng(11).standard_normal(n), dtype=np.float64)
    prop = PropagationParams(
        absorption_coeff=0.001,
        multipath_depth=0.3,
        multipath_spacing_hz=90.0,
        spreading_exponent=1.0,
    )
    kin = KinematicParams(
        bearing0_deg=90.0,
        closest_proxy=1.0,
        speed_proxy=0.5,
        t_cpa_s=0.5 * n / _RATE,
        doppler_peak=0.02,
    )
    track = evaluate_track(kin, np.arange(n, dtype=np.float64) / _RATE)

    def rejector_median(hybrid: bool, snr: float) -> float:
        values = []
        for j in range(16):
            machinery = draw_machinery(dist, derive_seed(j, "m"))
            line_signal = synthesize(
                build_lines(machinery), track, prop, _RATE, n / _RATE, derive_seed(j, "s")
            )
            if hybrid:
                bb = draw_broadband(dist, derive_seed(j, "b"))
                cont = synthesize_continuum(
                    n, _RATE, _BAND, bb.continuum_exponent, derive_seed(j, "c")
                )
                signal = add_broadband(line_signal, cont, _BAND, _RATE, bb.lbr_db)
            else:
                signal = line_signal
            injected, _ = inject_at_snr(signal, background, _BAND, _RATE, snr)
            _, rej = channel_responses(injected[np.newaxis, :], _RATE, 256, 128, _BAND, 3.0)
            values.append(float(rej[0]))
        return float(np.median(values))

    lines_lo, lines_hi = rejector_median(False, 0.0), rejector_median(False, 24.0)
    hybrid_lo, hybrid_hi = rejector_median(True, 0.0), rejector_median(True, 24.0)
    # At a high injected level the pure-narrowband prominence runs away; the hybrid stays bounded.
    assert hybrid_hi < 0.5 * lines_hi
    # And the hybrid's growth from low to high level is far smaller than the narrowband's runaway.
    assert (hybrid_hi / hybrid_lo) < 0.5 * (lines_hi / lines_lo)


def test_broadband_params_carry_no_absolute_level() -> None:
    bb = BroadbandParams(continuum_exponent=-1.0, lbr_db=6.0)
    keys = set(bb.__dict__)
    assert not any("source_level" in k or "absolute" in k or "spl" in k for k in keys)
    # The parameters are a dimensionless exponent and a decibel ratio only.
    assert set(keys) == {"continuum_exponent", "lbr_db"}
