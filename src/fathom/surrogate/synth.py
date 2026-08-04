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
