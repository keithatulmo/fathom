"""The sweepable linear-frequency spectral front end (SD4 / WO-6).

SD4 selects a linear-frequency short-time Fourier gram, the LOFAR gram, over a constant-Q axis,
because the target is a constant-hertz harmonic comb (a shaft fundamental near 4 Hz and its integer
harmonics across the 4-150 Hz working band) that a log axis cannot hold at one resolution across the
band. This module generalises the WO-3 reference front end (``fathom.stages.dsp.stft_power``, kept
unchanged for comparison) into the sweepable family the SD4 memo hands to E2: the window length and
therefore the frequency resolution, the overlap, the incoherent integration (the averaging count and
dwell), and the taper (a single Hann by default, a Thomson multitaper on discrete prolate spheroidal
sequences as the alternative) are all parameters. The normalizer family lives in
:mod:`fathom.normalize`. Every quantity here is a ratio or a count; no absolute level is set,
and the gram is bit-level deterministic given its inputs.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import numpy.typing as npt

_FLOOR = 1e-30


@dataclass(frozen=True)
class FrontEndParams:
    """One point in the E2 front-end grid: transform, integration, taper, and normalizer.

    ``window_length_s`` sets the frequency resolution (roughly its reciprocal in hertz);
    ``overlap`` is the fractional frame overlap; ``integration_count`` is the number of consecutive
    frames averaged incoherently, the dwell that trades time resolution for variance. The taper is a
    single Hann or a Thomson multitaper of ``n_tapers`` sequences at time-bandwidth ``nw``. The
    normalizer fields are consumed by :mod:`fathom.normalize`.
    """

    window_length_s: float
    overlap: float
    integration_count: int
    taper: str
    n_tapers: int
    nw: float
    normalizer: str
    sw_half_window_hz: float
    sw_guard_hz: float
    sw_truncation: int
    tm_window_frames: int

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable record of the configuration for the lineage."""
        return asdict(self)

    def key(self) -> str:
        """A short stable identifier for the configuration, for lineage and logging."""
        res = f"{1.0 / self.window_length_s:.3g}Hz"
        tap = "hann" if self.taper == "hann" else f"mt{self.n_tapers}"
        return (
            f"res{res}_ov{self.overlap:.2f}_int{self.integration_count}_{tap}_{self.normalizer}"
            f"_hw{self.sw_half_window_hz:g}_g{self.sw_guard_hz:g}_t{self.sw_truncation}"
        )


def _hann(nfft: int) -> npt.NDArray[np.float64]:
    """Periodic Hann window, defined here so the transform seam stays explicit (SD10 Section 7)."""
    n = np.arange(nfft, dtype=np.float64)
    return 0.5 - 0.5 * np.cos(2.0 * np.pi * n / nfft)


def _tapers(params: FrontEndParams, nfft: int) -> npt.NDArray[np.float64]:
    """Return the taper set, shaped ``[n_tapers, nfft]``: one Hann, or the DPSS multitaper set."""
    if params.taper == "hann":
        return _hann(nfft)[np.newaxis, :]
    if params.taper == "multitaper":
        from scipy.signal.windows import dpss  # local import keeps the module import light

        return np.asarray(dpss(nfft, params.nw, Kmax=params.n_tapers, norm=2), dtype=np.float64)
    raise ValueError(f"unsupported taper {params.taper!r}")


def _frame_starts(n_samples: int, nfft: int, hop: int) -> range:
    if n_samples < nfft:
        return range(0, 1)
    return range(0, n_samples - nfft + 1, hop)


def lofar_gram(
    signal: npt.NDArray[np.float64], sample_rate: float, params: FrontEndParams
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Return the linear-frequency power gram ``[freq_bins, frames]`` and its frequency axis.

    The window length sets the transform size and thus the resolution; the taper is a single Hann or
    the averaged multitaper estimate, which reduces the variance of the estimate at the cost
    of a wider main lobe and a per-taper cost; the frames are then averaged in groups of
    ``integration_count`` to set the dwell. The result is deterministic given the signal and params.
    """
    nfft = max(8, int(round(params.window_length_s * sample_rate)))
    if not (0.0 <= params.overlap < 1.0):
        raise ValueError("overlap must be in [0, 1)")
    hop = max(1, int(round(nfft * (1.0 - params.overlap))))
    freqs = np.fft.rfftfreq(nfft, d=1.0 / sample_rate).astype(np.float64)

    sig = signal if signal.shape[0] >= nfft else np.pad(signal, (0, nfft - signal.shape[0]))
    starts = list(_frame_starts(sig.shape[0], nfft, hop))
    tapers = _tapers(params, nfft)

    frame_power = np.empty((len(starts), freqs.shape[0]), dtype=np.float64)
    for f_index, start in enumerate(starts):
        block = sig[start : start + nfft]
        # Average the power over the taper set: one taper for Hann, K for multitaper. Each taper is
        # unit-energy (norm=2), so the estimate is a proper average periodogram.
        tapered = tapers * block[np.newaxis, :]
        spectra = np.fft.rfft(tapered, n=nfft, axis=1)
        power = (spectra.real**2 + spectra.imag**2).mean(axis=0) / float(nfft)
        frame_power[f_index] = power

    integrated = _integrate(frame_power, params.integration_count)
    return np.ascontiguousarray(integrated.T, dtype=np.float64), freqs


def _integrate(frame_power: npt.NDArray[np.float64], count: int) -> npt.NDArray[np.float64]:
    """Incoherently average consecutive frames in groups of ``count`` (the dwell)."""
    if count <= 1 or frame_power.shape[0] <= 1:
        return frame_power
    groups = frame_power.shape[0] // count
    if groups == 0:
        return frame_power.mean(axis=0, keepdims=True)
    trimmed = frame_power[: groups * count]
    return trimmed.reshape(groups, count, frame_power.shape[1]).mean(axis=1)


def compute_cost_proxy(params: FrontEndParams, sample_rate: float, duration_s: float) -> float:
    """A deterministic per-second compute proxy: taper FFTs times their size, over the hop rate.

    The proxy is the number of tapered fast Fourier transforms per second of audio times the
    transform work per FFT (n log2 n), the quantity that grows with resolution, overlap, and
    the multitaper taper count, and is what the all-beam compute budget is charged against.
    It is a relative proxy, not wall-clock, so it is deterministic across machines.
    """
    nfft = max(8, int(round(params.window_length_s * sample_rate)))
    hop = max(1, int(round(nfft * (1.0 - params.overlap))))
    frames_per_s = sample_rate / hop
    n_tapers = 1 if params.taper == "hann" else params.n_tapers
    work_per_fft = nfft * np.log2(max(nfft, 2))
    return float(frames_per_s * n_tapers * work_per_fft)
