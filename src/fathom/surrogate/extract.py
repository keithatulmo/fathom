"""Line-statistic extraction from real train-side quiet-tail audio.

SD2 Section 4 draws the surrogate's line amplitudes, harmonic roll-off, widths, and wander from
distributions fit to train-side real vessels, and WO-4 makes that the real-data refinement that
replaces the first-principles priors, with the injection-leak rule of SD3 Section 5.4 enforcing
train-only reads. This module measures those statistics from a real closest-approach segment: it
band-limits the segment to the working band, estimates the average spectrum, finds the prominent
narrowband peaks, and reports their amplitudes relative to the local background, the roll-off of
amplitude with frequency, the peak width, and the frequency wander across time. The line placement
itself stays first-principles kinematics; only the amplitude, roll-off, width, and wander
distributions are fit here, exactly as SD2 Section 4 specifies. Every statistic is dimensionless or
in hertz, so no absolute level is introduced.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
from scipy.signal import find_peaks, welch

from .fit import SurrogateDistributions

_ANALYSIS_RATE_HZ = 1000.0
_WELCH_SEGMENT = 1024
# A line must exceed the local in-band background by this many decibels to count, which rejects the
# random peaks of a noise spectrum while admitting a real narrowband tonal.
_PEAK_HEIGHT_DB = 6.0
_PEAK_PROMINENCE_DB = 3.0
_FLOOR = 1e-20


@dataclass(frozen=True)
class LineStats:
    """Dimensionless line statistics measured from one real segment."""

    peak_amplitudes: tuple[float, ...]  # prominent peak power over local background, in dB
    rolloff: float  # per-harmonic amplitude ratio implied by the amplitude-vs-frequency slope
    width_hz: float  # median prominent-peak width
    wander_hz: float  # frequency wander of the strongest line across time
    n_lines: int


def _decimate_to_analysis(
    samples: npt.NDArray[np.float64], sample_rate: float
) -> tuple[npt.NDArray[np.float64], float]:
    """Decimate a mono segment to the working-band analysis rate, returning it and the new rate."""
    from ..corpus.decimate import decimate  # local import keeps the module import light

    out = decimate(samples[np.newaxis, :], sample_rate, target_rate=_ANALYSIS_RATE_HZ)
    return out.samples[0], out.sample_rate_hz


def extract_line_stats(
    samples: npt.NDArray[np.float64], sample_rate: float, band: tuple[float, float]
) -> LineStats:
    """Measure the in-band line statistics of a real closest-approach segment."""
    signal, rate = _decimate_to_analysis(samples, sample_rate)
    freqs, psd = welch(signal, fs=rate, nperseg=min(_WELCH_SEGMENT, signal.shape[0]))
    in_band = (freqs >= band[0]) & (freqs <= band[1])
    if not np.any(in_band):
        raise ValueError("analysis band contains no frequency bins")
    band_freqs = freqs[in_band]
    band_psd = psd[in_band]
    background = float(np.median(band_psd))
    over_background_db = np.asarray(
        10.0 * np.log10(np.maximum(band_psd, _FLOOR) / max(background, _FLOOR))
    )
    peaks, props = find_peaks(
        over_background_db, height=_PEAK_HEIGHT_DB, prominence=_PEAK_PROMINENCE_DB, width=0.0
    )
    if peaks.size == 0:
        return LineStats(peak_amplitudes=(), rolloff=0.6, width_hz=0.3, wander_hz=0.05, n_lines=0)

    peak_freqs = band_freqs[peaks]
    peak_db = over_background_db[peaks]
    # Roll-off: the slope of peak amplitude in dB per hertz, mapped to a per-harmonic amplitude
    # ratio at a nominal harmonic spacing; clamped to a sensible band.
    if peak_freqs.size >= 2:
        slope_db_per_hz = float(np.polyfit(peak_freqs, peak_db, 1)[0])
    else:
        slope_db_per_hz = -0.05
    spacing = np.diff(np.sort(peak_freqs))
    nominal_spacing_hz = float(np.median(spacing)) if spacing.size else 20.0
    rolloff = float(np.clip(10.0 ** (slope_db_per_hz * nominal_spacing_hz / 20.0), 0.3, 0.95))
    # Peak width in hertz from the find_peaks widths (in bins) times the bin spacing.
    bin_hz = float(band_freqs[1] - band_freqs[0]) if band_freqs.size >= 2 else 1.0
    widths = props.get("widths")
    width_hz = float(np.median(widths) * bin_hz) if widths is not None and len(widths) else bin_hz
    dominant_hz = float(peak_freqs[int(np.argmax(peak_db))])
    wander_hz = _dominant_line_wander(signal, rate, band, dominant_hz)
    return LineStats(
        peak_amplitudes=tuple(float(v) for v in peak_db),
        rolloff=rolloff,
        width_hz=max(width_hz, 0.05),
        wander_hz=wander_hz,
        n_lines=int(peaks.size),
    )


def _dominant_line_wander(
    signal: npt.NDArray[np.float64], rate: float, band: tuple[float, float], centre_hz: float
) -> float:
    """Estimate the frequency wander of the dominant line as the std of its per-frame peak."""
    frame = int(rate)  # ~1 second frames
    if signal.shape[0] < 2 * frame:
        return 0.05
    centres = []
    for start in range(0, signal.shape[0] - frame + 1, frame):
        block = signal[start : start + frame]
        spectrum = np.abs(np.fft.rfft(block))
        block_freqs = np.fft.rfftfreq(block.shape[0], d=1.0 / rate)
        near = (block_freqs >= centre_hz - 5.0) & (block_freqs <= centre_hz + 5.0)
        if np.any(near):
            centres.append(float(block_freqs[near][int(np.argmax(spectrum[near]))]))
    return float(np.std(centres)) if len(centres) >= 2 else 0.05


def fit_from_segments(
    train_segments: list[dict[str, object]], band: tuple[float, float]
) -> SurrogateDistributions:
    """Fit surrogate amplitude, roll-off, width, and wander distributions from train-side audio.

    Each entry must carry ``split == "train"``, ``vessel_id``, and ``samples`` with ``sample_rate``;
    a non-train entry raises through the fit's leak guard, so the held-out class cannot contribute.
    The structural frequency ranges stay first-principles; the amplitude, roll-off, width, and
    wander ranges are the tenth-to-ninetieth-percentile spread measured across the train segments.
    """
    from .fit import TRAIN_SPLIT, ForbiddenSplitError

    stats: list[LineStats] = []
    roster: list[str] = []
    quiet_flags: list[bool] = []
    for entry in train_segments:
        if str(entry.get("split")) != TRAIN_SPLIT:
            raise ForbiddenSplitError(
                f"segment for {entry.get('vessel_id')!r} is split {entry.get('split')!r}, not "
                f"{TRAIN_SPLIT!r}; surrogate statistics fit from train-side audio only (SD3 5.4)"
            )
        samples = np.asarray(entry["samples"], dtype=np.float64)
        stats.append(extract_line_stats(samples, float(entry["sample_rate"]), band))  # type: ignore[arg-type]
        roster.append(str(entry["vessel_id"]))
        quiet_flags.append(bool(entry.get("is_quiet_tail", True)))
    if not stats:
        raise ValueError("no train-side segments to fit from")

    all_amps = [a for s in stats for a in s.peak_amplitudes]
    rolloffs = [s.rolloff for s in stats]
    widths = [s.width_hz for s in stats]
    wanders = [s.wander_hz for s in stats]
    # Amplitudes measured in dB above background are mapped to dimensionless family weights by a
    # decibel-to-ratio conversion, then the family ranges are set from their spread.
    amp_ratios = [float(10.0 ** (a / 20.0)) for a in all_amps] or [1.0]
    amp_lo, amp_hi = _percentile_range(amp_ratios)
    amp_span = max(amp_hi - amp_lo, 1e-3)
    quiet_fraction = sum(quiet_flags) / len(quiet_flags) if quiet_flags else 0.5

    # Normalise the measured amplitude spread onto the family weight ranges: the shaft carries the
    # top of the spread, the auxiliary the bottom, so quiet-target draws stay sparse and low.
    return SurrogateDistributions(
        quiet_fraction=quiet_fraction,
        train_vessel_count=len(roster),
        train_vessel_ids=tuple(sorted(roster)),
        shaft_amp_range=(amp_lo + 0.5 * amp_span, amp_hi),
        blade_amp_range=(amp_lo + 0.25 * amp_span, amp_lo + 0.85 * amp_span),
        electrical_amp_range=(amp_lo + 0.1 * amp_span, amp_lo + 0.6 * amp_span),
        aux_amp_range=(amp_lo, amp_lo + 0.4 * amp_span),
        rolloff_range=_percentile_range(rolloffs),
        width_hz_range=_percentile_range(widths),
        wander_hz_range=_percentile_range(wanders),
        provenance={"structural_ranges": "first_principles", "amplitude_statistics": "train_audio"},
    )


def _percentile_range(values: list[float]) -> tuple[float, float]:
    """Return the tenth-to-ninetieth percentile range of a list, with a small floor on the span."""
    if not values:
        return (0.1, 0.4)
    lo = float(np.percentile(values, 10))
    hi = float(np.percentile(values, 90))
    if hi - lo < 1e-6:
        hi = lo + max(abs(lo) * 0.1, 1e-3)
    return (lo, hi)
