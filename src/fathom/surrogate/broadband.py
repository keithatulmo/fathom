"""Broadband continuum extraction and the line-to-broadband ratio, fit from train-side audio.

WO-5 executes the SD2 Section 8 flip. The pure-narrowband surrogate of Option A carries tonal lines
over noise with nothing between them, so its in-band line prominence climbs without bound as the
injected level rises, which the WO-4 E1 real-data run exposed as a fail against the real held-out
quiet class: at the signal-to-noise where the surrogate becomes as detectable as the real class it
is far more line-prominent than any real quiet vessel. The fix is a broadband continuum beneath and
between the lines. This module measures, from a real train-side segment, the two quantities that
specify that continuum: the power-law spectral shape of the between-line floor across the working
band, and the line-to-broadband power ratio, the ratio of the summed line excess to the summed
continuum. Real quiet vessels carry a broadband continuum from cavitation, flow, and propulsion that
bounds their line prominence, and the surrogate must carry the same.

Both quantities are dimensionless: a spectral exponent and a power ratio in decibels, so no absolute
level is introduced, per SD2 Section 2. The fit is train-only; the leak guard lives in the caller
(:func:`fathom.surrogate.extract.fit_from_segments`) exactly as it does for the line fit.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
from scipy.signal import find_peaks, welch

# The continuum power-law exponent is clamped to a physically sensible band: real vessel and ambient
# broadband is red, its power falling with frequency, so the exponent is negative or mild, and an
# unbounded negative exponent would over-weight the lowest bin, so the fit is clamped to this range.
_EXPONENT_RANGE = (-3.0, 0.0)
_DEFAULT_EXPONENT = -1.0
# The broadband-dominated default used when a segment shows no detected lines: the surrogate is then
# almost entirely continuum, the quiet extreme the held-out class occupies.
_DEFAULT_LBR_DB = 0.0


@dataclass(frozen=True)
class BroadbandStats:
    """Dimensionless broadband statistics measured from one real segment."""

    continuum_exponent: float  # power-law exponent b: continuum PSD proportional to f ** b
    lbr_db: float  # line-to-broadband in-band power ratio, in decibels
    n_floor_bins: int


@dataclass(frozen=True)
class BroadbandParams:
    """One drawn broadband continuum: its spectral exponent and its line-to-broadband ratio."""

    continuum_exponent: float
    lbr_db: float


def extract_broadband_stats(
    samples: npt.NDArray[np.float64], sample_rate: float, band: tuple[float, float]
) -> BroadbandStats:
    """Measure the between-line continuum shape and the line-to-broadband ratio of a segment.

    The segment is decimated to the working-band analysis rate and its average spectrum is estimated
    exactly as the line fit does, so the two fits see the same spectrum. The detected lines are
    masked out; the continuum exponent is fit to the remaining floor, and the line-to-broadband
    ratio is the summed line excess over the summed continuum.
    """
    from .extract import (  # lazy import keeps the surrogate import graph acyclic
        _FLOOR,
        _PEAK_HEIGHT_DB,
        _PEAK_PROMINENCE_DB,
        _WELCH_SEGMENT,
        _decimate_to_analysis,
    )

    signal, rate = _decimate_to_analysis(samples, sample_rate)
    freqs, psd = welch(signal, fs=rate, nperseg=min(_WELCH_SEGMENT, signal.shape[0]))
    in_band = (freqs >= band[0]) & (freqs <= band[1])
    if not np.any(in_band):
        raise ValueError("analysis band contains no frequency bins")
    band_freqs = np.asarray(freqs[in_band], dtype=np.float64)
    band_psd = np.asarray(psd[in_band], dtype=np.float64)
    background = float(np.median(band_psd))
    over_background_db = 10.0 * np.log10(np.maximum(band_psd, _FLOOR) / max(background, _FLOOR))
    peaks, _ = find_peaks(
        over_background_db, height=_PEAK_HEIGHT_DB, prominence=_PEAK_PROMINENCE_DB, width=0.0
    )
    floor_mask = np.ones(band_psd.shape[0], dtype=bool)
    floor_mask[peaks] = False
    exponent = _fit_exponent(band_freqs[floor_mask], band_psd[floor_mask])
    lbr_db = _line_to_broadband_db(band_psd, peaks, band_psd[floor_mask], _FLOOR)
    return BroadbandStats(
        continuum_exponent=exponent, lbr_db=lbr_db, n_floor_bins=int(floor_mask.sum())
    )


def _fit_exponent(
    floor_freqs: npt.NDArray[np.float64], floor_psd: npt.NDArray[np.float64]
) -> float:
    """Fit the continuum power-law exponent: the slope of log10(PSD) on log10(f), clamped."""
    positive = (floor_freqs > 0.0) & (floor_psd > 0.0)
    if int(positive.sum()) < 3:
        return _DEFAULT_EXPONENT
    log_f = np.log10(floor_freqs[positive])
    log_p = np.log10(floor_psd[positive])
    slope = float(np.polyfit(log_f, log_p, 1)[0])
    return float(np.clip(slope, _EXPONENT_RANGE[0], _EXPONENT_RANGE[1]))


def _line_to_broadband_db(
    band_psd: npt.NDArray[np.float64],
    peaks: npt.NDArray[np.intp],
    floor_psd: npt.NDArray[np.float64],
    floor: float,
) -> float:
    """Return the in-band line-to-broadband power ratio in decibels.

    The line power is the excess of the detected peak bins over the local continuum level; the
    broadband power is the summed continuum floor extrapolated across the band. A vessel with strong
    clean lines over a low floor has a high ratio; a broadband-heavy quiet vessel has a low one.
    With no detected lines, or no positive line excess, the ratio is the broadband default.
    """
    continuum_level = float(np.median(floor_psd)) if floor_psd.size else floor
    broadband_power = continuum_level * float(band_psd.shape[0])
    if peaks.size == 0 or broadband_power <= 0.0:
        return _DEFAULT_LBR_DB
    line_excess = float(np.sum(np.maximum(band_psd[peaks] - continuum_level, 0.0)))
    if line_excess <= 0.0:
        return _DEFAULT_LBR_DB
    return 10.0 * float(np.log10(line_excess / broadband_power))
