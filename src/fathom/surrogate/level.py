"""Signal-to-noise leveling against the measured in-band background.

WO-3 deliverable 2 sets the injected level relative to the measured in-band background of the
target channel, with no absolute level anywhere in the interface, and asks that the achieved ratio
be verifiable by re-measuring it after injection. That is what this module does: it measures the
in-band power of the surrogate and of the background, scales the surrogate so their ratio equals
the requested signal-to-noise ratio, adds it to the background, and re-measures the achieved ratio
on the combined channel. Because the ratio is dimensionless and the measurement is relative, the
quietness of a target is a low signal-to-noise ratio, never an asserted decibel level, per SD2
Section 2.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

_TINY = 1e-30


def inband_power(
    samples: npt.NDArray[np.float64], sample_rate: float, band: tuple[float, float]
) -> float:
    """Return the summed spectral power of a signal within a frequency band.

    The measure is relative: it is used only in ratios, so its normalisation cancels and no absolute
    level is implied.
    """
    n = samples.shape[0]
    spectrum = np.fft.rfft(samples)
    power = spectrum.real**2 + spectrum.imag**2
    freqs = np.fft.rfftfreq(n, d=1.0 / sample_rate)
    mask = (freqs >= band[0]) & (freqs <= band[1])
    if not np.any(mask):
        raise ValueError("in-band power band contains no frequency bins")
    return float(power[mask].sum())


def scale_to_snr(
    signal: npt.NDArray[np.float64],
    background: npt.NDArray[np.float64],
    band: tuple[float, float],
    sample_rate: float,
    requested_snr_db: float,
) -> npt.NDArray[np.float64]:
    """Scale a surrogate so its in-band power over the background's equals the requested ratio."""
    signal_power = inband_power(signal, sample_rate, band)
    background_power = inband_power(background, sample_rate, band)
    if signal_power <= 0.0:
        raise ValueError("surrogate has no in-band power to level")
    if background_power <= 0.0:
        raise ValueError("background has no in-band power to level against")
    target_ratio = 10.0 ** (requested_snr_db / 10.0)
    gain = float(np.sqrt(target_ratio * background_power / signal_power))
    return np.ascontiguousarray(gain * signal, dtype=np.float64)


def injected_snr_db(
    injected: npt.NDArray[np.float64],
    background: npt.NDArray[np.float64],
    band: tuple[float, float],
    sample_rate: float,
) -> float:
    """Re-measure the achieved in-band signal-to-noise ratio on the injected channel, in decibels.

    The achieved ratio is the excess in-band power of the injected channel over the background,
    divided by the background power, which recovers the requested ratio up to the cross-term of the
    finite sample and is the quantity the acceptance tolerance is stated against.
    """
    background_power = inband_power(background, sample_rate, band)
    excess = inband_power(injected, sample_rate, band) - background_power
    return 10.0 * float(np.log10(max(excess, _TINY) / max(background_power, _TINY)))


def inject_at_snr(
    signal: npt.NDArray[np.float64],
    background: npt.NDArray[np.float64],
    band: tuple[float, float],
    sample_rate: float,
    requested_snr_db: float,
) -> tuple[npt.NDArray[np.float64], float]:
    """Level a surrogate to the requested ratio, add it to the background, and re-measure.

    Returns the injected channel and its re-measured achieved signal-to-noise ratio in decibels.
    """
    scaled = scale_to_snr(signal, background, band, sample_rate, requested_snr_db)
    injected = np.ascontiguousarray(background + scaled, dtype=np.float64)
    return injected, injected_snr_db(injected, background, band, sample_rate)
