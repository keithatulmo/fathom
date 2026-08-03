"""The band-limited spectral front end.

This stage constructs a power gram per channel with a short-time Fourier transform under a Hann
window, optionally flattening the background by dividing each frequency bin by its median over
time. It is a placeholder front end for the acceptance path: the real front end, its transform
parameters, and its normalizer family are selected later at experiment E2 under SD4. The transform
is written fresh from the definition and uses only the NumPy array seam SD10 Section 7 requires.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import numpy.typing as npt

from ..artifacts import AudioCorpus, Gram
from ..config import FrontEndConfig
from .base import Stage, register

_MEDIAN_FLOOR = 1e-12


def _hann(nfft: int) -> npt.NDArray[np.float64]:
    # Periodic Hann window, defined here rather than imported so the seam stays explicit.
    n = np.arange(nfft, dtype=np.float64)
    return 0.5 - 0.5 * np.cos(2.0 * np.pi * n / nfft)


def stft_power(
    signal: npt.NDArray[np.float64], nfft: int, hop: int, window: npt.NDArray[np.float64]
) -> npt.NDArray[np.float64]:
    """Return the one-sided power spectrogram of a signal, shaped ``[freq_bins, frames]``."""
    if signal.shape[0] < nfft:
        signal = np.pad(signal, (0, nfft - signal.shape[0]))
    starts = range(0, signal.shape[0] - nfft + 1, hop)
    frames = np.stack([signal[start : start + nfft] * window for start in starts], axis=0)
    spectrum = np.fft.rfft(frames, n=nfft, axis=1)
    power = (spectrum.real**2 + spectrum.imag**2) / float(nfft)
    return np.asarray(power.T, dtype=np.float64)


class FrontEndStage(Stage):
    """Transform an audio corpus into a per-channel power gram."""

    NAME = "front_end"
    VERSION = "1"
    OUTPUTS = {"gram": "gram"}

    def parse_config(self, raw: dict[str, Any]) -> FrontEndConfig:
        return FrontEndConfig.model_validate(raw)

    def run(self, inputs: dict[str, bytes], config: Any, seed: int) -> dict[str, bytes]:
        assert isinstance(config, FrontEndConfig)
        if config.window != "hann":
            raise ValueError(f"unsupported window {config.window!r}")
        audio = AudioCorpus.from_blob(inputs["audio"])
        window = _hann(config.nfft)
        freqs = np.fft.rfftfreq(config.nfft, d=1.0 / audio.sample_rate).astype(np.float64)

        grams = [
            stft_power(audio.samples[row], config.nfft, config.hop, window)
            for row in range(audio.samples.shape[0])
        ]
        gram = np.stack(grams, axis=0)

        if config.normalizer == "median_over_frequency":
            # Flatten the broadband floor per frame so a narrowband tonal stands out; this is the
            # axis a standing tonal survives, unlike a time normalizer, which would erase it.
            median = np.median(gram, axis=1, keepdims=True)
            gram = gram / np.maximum(median, _MEDIAN_FLOOR)
        elif config.normalizer != "none":
            raise ValueError(f"unsupported normalizer {config.normalizer!r}")

        artifact = Gram(
            gram=np.ascontiguousarray(gram, dtype=np.float64),
            freqs=freqs,
            channel_ids=audio.channel_ids,
            sample_rate=audio.sample_rate,
            nfft=config.nfft,
            hop=config.hop,
        )
        return {"gram": artifact.to_blob()}


register(FrontEndStage())
