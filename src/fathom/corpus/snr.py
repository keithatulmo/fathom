"""In-band signal-to-noise at a passage's closest approach.

The adequacy note defines a quiet-tail proxy by measured in-band signal-to-noise at close passage,
which only the audio can settle: a vessel flagged quiet from AIS kinematics counts toward the
cohort only if its closest-approach passage is actually audible in the working band. This module
measures that. It decodes a short window of the recording around the closest-approach time and a
spread of reference windows for the ambient floor, band-limits each to the working band via the
recorded decimation and a Butterworth pass, and reports the ratio of in-band power at closest
approach to the low-percentile ambient floor in decibels. Only short segments are decoded, never the
whole multi-hour file, so the measurement is cheap even at 48 or 96 kHz.
"""

from __future__ import annotations

import io
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
from scipy.signal import butter, sosfiltfilt

from .decimate import decimate

WORKING_BAND_HZ = (4.0, 150.0)
SIGNAL_WINDOW_S = 60.0
REFERENCE_WINDOW_S = 20.0
REFERENCE_COUNT = 30
# The ambient floor is a low percentile of the reference windows, a standard noise-floor estimator:
# the quiet baseline, not the mean, which at a busy site is lifted by constant distant traffic.
NOISE_FLOOR_PERCENTILE = 10.0
_DECIMATED_RATE_HZ = 1000.0
_BUTTER_ORDER = 4


@dataclass(frozen=True)
class SnrResult:
    """The in-band SNR at closest approach and the powers behind it."""

    snr_db: float
    signal_power: float
    noise_floor: float
    reference_count: int


def _inband_power(
    segment: npt.NDArray[np.float64], sample_rate: float, band: tuple[float, float]
) -> float:
    """Return the mean in-band power of a mono segment, decimating then band-passing to the band."""
    if segment.size < 64:
        return float("nan")
    decimated = decimate(segment[np.newaxis, :], sample_rate, target_rate=_DECIMATED_RATE_HZ)
    signal = decimated.samples[0]
    nyquist = decimated.sample_rate_hz / 2.0
    low = band[0] / nyquist
    high = min(band[1], nyquist * 0.99) / nyquist
    if not 0.0 < low < high < 1.0 or signal.size < 3 * _BUTTER_ORDER * 4:
        return float("nan")
    sos = butter(_BUTTER_ORDER, [low, high], btype="band", output="sos")
    filtered = sosfiltfilt(sos, signal)
    return float(np.mean(filtered * filtered))


def measure_inband_snr(
    blob: bytes,
    closest_offset_s: float,
    *,
    band: tuple[float, float] = WORKING_BAND_HZ,
    signal_window_s: float = SIGNAL_WINDOW_S,
    reference_window_s: float = REFERENCE_WINDOW_S,
    reference_count: int = REFERENCE_COUNT,
) -> SnrResult:
    """Measure in-band SNR at ``closest_offset_s`` seconds into an encoded WAV/FLAC recording.

    The signal is the in-band power in a window centred on the closest-approach time; the ambient
    floor is the low percentile of the in-band power of reference windows spread across the whole
    recording, so a vessel that is loud only at closest approach lifts the signal above the floor.
    The result is NaN when the recording is too short or cannot be decoded, which the caller treats
    as unverified rather than as a passing measurement.
    """
    import soundfile as sf

    with sf.SoundFile(io.BytesIO(blob)) as handle:
        sample_rate = float(handle.samplerate)
        total_frames = int(handle.frames)
        duration_s = total_frames / sample_rate if sample_rate > 0 else 0.0

        def read_segment(center_s: float, window_s: float) -> npt.NDArray[np.float64]:
            start = max(0, int((center_s - window_s / 2.0) * sample_rate))
            count = min(int(window_s * sample_rate), total_frames - start)
            if count <= 0:
                return np.zeros(0, dtype=np.float64)
            handle.seek(start)
            data = handle.read(count, dtype="float64", always_2d=True)
            return np.ascontiguousarray(data[:, 0], dtype=np.float64)

        signal_segment = read_segment(closest_offset_s, signal_window_s)
        signal_power = _inband_power(signal_segment, sample_rate, band)
        references: list[float] = []
        for index in range(reference_count):
            center = (index + 0.5) * duration_s / reference_count
            power = _inband_power(read_segment(center, reference_window_s), sample_rate, band)
            if np.isfinite(power) and power > 0.0:
                references.append(power)

    noise_floor = (
        float(np.percentile(references, NOISE_FLOOR_PERCENTILE)) if references else float("nan")
    )
    if (
        np.isfinite(signal_power)
        and signal_power > 0.0
        and np.isfinite(noise_floor)
        and noise_floor > 0.0
    ):
        snr_db = float(10.0 * np.log10(signal_power / noise_floor))
    else:
        snr_db = float("nan")
    return SnrResult(snr_db, signal_power, noise_floor, len(references))
