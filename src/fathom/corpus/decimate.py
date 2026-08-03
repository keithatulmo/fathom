"""Decimation of working derivatives.

WO-2 Section 4 fixes the processing policy: originals are stored as acquired, and working
derivatives are decimated to a 1 kHz sample rate covering the 3 to 300 Hz processing span with
margin, with anti-alias filter parameters recorded in lineage. A 1 kHz rate has a 500 Hz Nyquist,
so it covers the 300 Hz top of the span with margin; the low end is preserved because no high-pass
is applied here. The polyphase resampler applies the anti-alias low-pass, and its parameters are
returned so they are recorded rather than implied. Band coverage is reported honestly, including the
partial coverage of a source whose own rate is below what the span needs.
"""

from __future__ import annotations

import io
from dataclasses import dataclass
from math import gcd
from typing import Any

import numpy as np
import numpy.typing as npt
from scipy.signal import resample_poly

DEFAULT_TARGET_RATE_HZ = 1000.0
DEFAULT_BAND_HZ = (3.0, 300.0)
_RESAMPLE_WINDOW = ("kaiser", 5.0)


@dataclass(frozen=True)
class DecimationOutput:
    """A decimated multichannel signal and the parameters that produced it."""

    samples: npt.NDArray[np.float64]  # shape [channels, frames]
    sample_rate_hz: float
    duration_s: float
    params: dict[str, Any]


def decimate(
    samples: npt.NDArray[np.float64],
    sample_rate_in: float,
    *,
    target_rate: float = DEFAULT_TARGET_RATE_HZ,
    band: tuple[float, float] = DEFAULT_BAND_HZ,
) -> DecimationOutput:
    """Decimate a signal to the target rate with a recorded polyphase anti-alias filter.

    The input is shaped ``[channels, frames]`` or ``[frames]``. When the source rate is already at
    or below the target, the signal is passed through and the band coverage is marked partial if the
    source Nyquist does not reach the top of the processing span.
    """
    signal = np.atleast_2d(samples).astype(np.float64)
    frames = signal.shape[1]
    duration = frames / sample_rate_in

    source_nyquist = sample_rate_in / 2.0
    band_partial = source_nyquist < band[1]

    if sample_rate_in <= target_rate:
        params = {
            "method": "passthrough",
            "sample_rate_in": sample_rate_in,
            "sample_rate_out": sample_rate_in,
            "target_band_hz": list(band),
            "antialias_cutoff_hz": source_nyquist,
            "band_partial": band_partial,
        }
        return DecimationOutput(signal, sample_rate_in, duration, params)

    ratio = gcd(int(round(sample_rate_in)), int(round(target_rate)))
    up = int(round(target_rate)) // ratio
    down = int(round(sample_rate_in)) // ratio
    resampled = resample_poly(signal, up, down, axis=1, window=_RESAMPLE_WINDOW)

    params = {
        "method": "resample_poly",
        "up": up,
        "down": down,
        "window": ["kaiser", 5.0],
        "sample_rate_in": sample_rate_in,
        "sample_rate_out": target_rate,
        "target_band_hz": list(band),
        "antialias_cutoff_hz": target_rate / 2.0,
        "band_partial": band_partial,
    }
    return DecimationOutput(
        np.ascontiguousarray(resampled, dtype=np.float64),
        target_rate,
        duration,
        params,
    )


def read_audio(blob: bytes) -> tuple[npt.NDArray[np.float64], float]:
    """Decode WAV or FLAC bytes into a ``[channels, frames]`` array and its sample rate."""
    import soundfile as sf

    data, sample_rate = sf.read(io.BytesIO(blob), dtype="float64", always_2d=True)
    return np.ascontiguousarray(data.T, dtype=np.float64), float(sample_rate)


def encode_wav(samples: npt.NDArray[np.float64], sample_rate: float) -> bytes:
    """Encode a ``[channels, frames]`` array as floating-point WAV bytes."""
    import soundfile as sf

    buffer = io.BytesIO()
    sf.write(buffer, samples.T, int(round(sample_rate)), format="WAV", subtype="FLOAT")
    return buffer.getvalue()
