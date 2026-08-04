"""Time-domain synthesis of the surrogate line set.

Each line is realised as a narrowband tone whose instantaneous frequency is the line frequency
scaled by the kinematic Doppler factor, plus a slow wander and a fast finite-width jitter, and whose
amplitude follows the propagation spectral gain and the kinematic range envelope. The phase is the
running integral of the instantaneous frequency, so the tone is coherent and its spectral line sits
where the kinematics place it. Every random draw comes from a seed derived from the run seed and the
line index, so the synthesis is deterministic and regenerable. The output amplitude scale is
arbitrary and dimensionless; the injected level is set later by signal-to-noise leveling, so nothing
here asserts an absolute level.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

from ..determinism import derive_seed, rng
from .kinematics import Track
from .level import inband_power
from .lines import LineSet
from .propagation import PropagationParams, range_envelope, spectral_gain

# The wander process is smoothed over roughly this many seconds so it is a slow drift rather than
# broadband noise; the finite line width comes from the separate fast jitter term.
_WANDER_SMOOTH_S = 1.0
# Lines are dropped above this fraction of Nyquist so a Doppler-shifted tone never aliases.
_NYQUIST_FRACTION = 0.95


def _slow_process(
    generator: np.random.Generator, n: int, sample_rate: float
) -> npt.NDArray[np.float64]:
    """Return a unit-variance slow random process for line wander, smoothed over ~1 second."""
    white = generator.standard_normal(n)
    width = max(1, int(round(sample_rate * _WANDER_SMOOTH_S)))
    kernel = np.ones(width, dtype=np.float64) / float(width)
    slow = np.convolve(white, kernel, mode="same")
    std = float(slow.std())
    return slow / std if std > 0.0 else slow


def synthesize(
    lines: LineSet,
    track: Track,
    propagation: PropagationParams,
    sample_rate: float,
    duration_s: float,
    seed: int,
) -> npt.NDArray[np.float64]:
    """Synthesise the surrogate waveform from its line set, kinematic track, and propagation."""
    n = int(round(duration_s * sample_rate))
    if n <= 0:
        raise ValueError("duration must be positive")
    if track.doppler_factor.shape[0] != n or track.range_proxy.shape[0] != n:
        raise ValueError("track arrays must match the synthesized sample count")
    nyquist = sample_rate / 2.0
    max_freq = _NYQUIST_FRACTION * nyquist
    doppler_max = float(track.doppler_factor.max())

    spec = spectral_gain(lines.freqs, propagation)
    envelope = range_envelope(track.range_proxy, float(track.range_proxy.min()), propagation)

    signal = np.zeros(n, dtype=np.float64)
    for index in range(lines.freqs.shape[0]):
        centre = float(lines.freqs[index])
        if centre * doppler_max >= max_freq:
            continue  # a Doppler-shifted line above Nyquist cannot be represented; drop it
        line_rng = rng(derive_seed(seed, f"line-{index}"))
        phase0 = float(line_rng.uniform(0.0, 2.0 * np.pi))
        fast = float(lines.widths[index]) * line_rng.standard_normal(n)
        slow = float(lines.wanders[index]) * _slow_process(line_rng, n, sample_rate)
        inst_freq = centre * track.doppler_factor + slow + fast
        inst_freq = np.clip(inst_freq, 0.0, max_freq)
        phase = phase0 + 2.0 * np.pi * np.cumsum(inst_freq) / sample_rate
        amplitude = float(lines.amplitudes[index]) * float(spec[index]) * envelope
        signal += amplitude * np.cos(phase)
    return np.ascontiguousarray(signal, dtype=np.float64)


def synthesize_continuum(
    n: int, sample_rate: float, band: tuple[float, float], exponent: float, seed: int
) -> npt.NDArray[np.float64]:
    """Synthesise a band-limited broadband continuum with a power-law spectral shape.

    White noise is shaped in the frequency domain so its in-band power spectral density follows a
    power law in frequency with the given exponent and rolls to zero outside the working band, then
    returned at unit in-band power so the caller sets its level relative to the lines. This is the
    between-line energy real quiet vessels carry, generated as coloured noise rather than tones, so
    it fills the spectrum between the lines rather than adding structure of its own. The draw is
    seeded, so the continuum is deterministic and regenerable.
    """
    if n <= 0:
        raise ValueError("continuum length must be positive")
    generator = rng(seed)
    spectrum = np.fft.rfft(generator.standard_normal(n))
    freqs = np.fft.rfftfreq(n, d=1.0 / sample_rate)
    in_band = (freqs >= band[0]) & (freqs <= band[1])
    if not np.any(in_band):
        raise ValueError("continuum band contains no frequency bins")
    f_ref = 0.5 * (band[0] + band[1])
    shape = np.zeros(freqs.shape[0], dtype=np.float64)
    # A power-law PSD proportional to (f / f_ref) ** exponent is a spectral amplitude proportional
    # to (f / f_ref) ** (exponent / 2); bins outside the band stay zero, so it is band-limited.
    shape[in_band] = (freqs[in_band] / f_ref) ** (exponent / 2.0)
    continuum = np.fft.irfft(spectrum * shape, n=n)
    power = inband_power(continuum, sample_rate, band)
    if power <= 0.0:
        raise ValueError("continuum has no in-band power")
    return np.ascontiguousarray(continuum / np.sqrt(power), dtype=np.float64)


def add_broadband(
    line_signal: npt.NDArray[np.float64],
    continuum: npt.NDArray[np.float64],
    band: tuple[float, float],
    sample_rate: float,
    lbr_db: float,
) -> npt.NDArray[np.float64]:
    """Add the continuum to the line signal at the requested line-to-broadband power ratio.

    The continuum is scaled so the line in-band power over the continuum in-band power equals the
    requested ratio, so a low ratio yields a broadband-dominated surrogate and a high ratio a
    line-dominated one. This sets the surrogate's internal line prominence before injection; the
    injection then scales the whole surrogate uniformly to the requested signal-to-noise ratio,
    which preserves the ratio, so the surrogate's line prominence is bounded by the broadband it now
    carries rather than climbing without bound as the pure-narrowband surrogate's did.
    """
    line_power = inband_power(line_signal, sample_rate, band)
    continuum_power = inband_power(continuum, sample_rate, band)
    if line_power <= 0.0:
        raise ValueError("line signal has no in-band power")
    if continuum_power <= 0.0:
        raise ValueError("continuum has no in-band power")
    target_continuum_power = line_power / 10.0 ** (lbr_db / 10.0)
    gain = float(np.sqrt(target_continuum_power / continuum_power))
    return np.ascontiguousarray(line_signal + gain * continuum, dtype=np.float64)
