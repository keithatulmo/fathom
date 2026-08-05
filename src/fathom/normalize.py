"""Background normalizers for the spectral front end (SD4 / WO-6).

SD4 selects a two-pass split-window order-truncated-average normalizer, which estimates the
background for each frequency bin from the flanking bins on either side, excludes a guard around
the cell under test, truncates the largest flanking values so a dense harmonic comb cannot estimate
its background from neighbouring teeth and null its own lines, and divides it out so the gram has
a flat background against which one detector threshold holds across sites. The named alternative,
carried as the second sweep arm, is a per-bin temporal-median background, robust to intermittent and
wandering lines because a moving line spends little time in one bin. Near the 4 Hz floor the band
runs out of bins below the line, so the split window becomes one-sided, which is handled explicitly.

Both normalizers flatten a colored, site-varying background per frame from the data itself with no
per-site tuning, which is the whole of the flatness criterion. Every operation is a ratio; no
absolute level is expressible, and the output is bit-level deterministic given its input.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

from .frontend import FrontEndParams

_FLOOR = 1e-30
# A pass-one normalized value above this is treated as a candidate line and censored from the
# pass-two background estimate, so a real tonal does not inflate its own local background.
_PEAK_CENSOR_RATIO = 2.0


def normalize_gram(
    gram: npt.NDArray[np.float64],
    freqs: npt.NDArray[np.float64],
    band: tuple[float, float],
    params: FrontEndParams,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Normalize the in-band gram by the configured background estimator; return it and freqs."""
    in_band = np.where((freqs >= band[0]) & (freqs <= band[1]))[0]
    if in_band.size == 0:
        raise ValueError("normalizer band contains no frequency bins")
    if params.normalizer == "split_window":
        normalized = _split_window_ota(gram, freqs, in_band, params)
    elif params.normalizer == "temporal_median":
        normalized = _temporal_median(gram[in_band], params.tm_window_frames)
    else:
        raise ValueError(f"unsupported normalizer {params.normalizer!r}")
    return np.ascontiguousarray(normalized, dtype=np.float64), freqs[in_band]


def _flank_indices(center: int, guard: int, half: int, n_bins: int) -> npt.NDArray[np.intp]:
    """Return the flanking-bin indices for a cell: two sidebands minus the guard, clipped to range.

    When the low sideband runs off the bottom of the band, as it does near the 4 Hz floor, only the
    bins that exist are returned, so the estimate becomes one-sided rather than past the edge.
    """
    lower = range(max(0, center - half), max(0, center - guard))
    upper = range(min(n_bins, center + guard + 1), min(n_bins, center + half + 1))
    return np.array([i for i in list(lower) + list(upper) if i != center], dtype=np.intp)


def _ota_background(values: npt.NDArray[np.float64], truncation: int) -> npt.NDArray[np.float64]:
    """Order-truncated average over the flanking axis (axis 0): drop the largest, mean the rest.

    ``values`` is ``[n_flank, n_frames]`` and may carry NaN for censored bins. Sorting sends
    NaNs to the end of each column, so keeping the lowest ``finite_count - truncation`` positions
    drops the largest finite values and every NaN at once. Fully vectorised across frames so the
    normalizer stays affordable at sweep scale; a censored or one-sided cell degrades gracefully.
    """
    if values.shape[0] == 0:
        return np.full(values.shape[1], _FLOOR, dtype=np.float64)
    ordered = np.sort(values, axis=0)  # NaNs sort to the end of each column
    finite_count = np.isfinite(ordered).sum(axis=0)
    drop = np.minimum(truncation, np.maximum(finite_count - 1, 0))
    keep_count = np.maximum(finite_count - drop, 0)
    positions = np.arange(values.shape[0])[:, np.newaxis]
    keep = positions < keep_count[np.newaxis, :]
    kept = np.where(keep, np.nan_to_num(ordered, nan=0.0), 0.0)
    with np.errstate(invalid="ignore", divide="ignore"):
        background = kept.sum(axis=0) / np.maximum(keep_count, 1)
    return np.asarray(np.where(keep_count > 0, background, _FLOOR), dtype=np.float64)


def _split_window_ota(
    gram: npt.NDArray[np.float64],
    freqs: npt.NDArray[np.float64],
    in_band: npt.NDArray[np.intp],
    params: FrontEndParams,
) -> npt.NDArray[np.float64]:
    """Two-pass split-window order-truncated-average normalizer over the in-band bins."""
    bin_hz = float(freqs[1] - freqs[0]) if freqs.shape[0] >= 2 else 1.0
    guard = max(1, int(round(params.sw_guard_hz / bin_hz)))
    half = max(guard + 1, int(round(params.sw_half_window_hz / bin_hz)))
    n_bins = gram.shape[0]

    normalized = np.empty((in_band.shape[0], gram.shape[1]), dtype=np.float64)
    # Pass one: a straight order-truncated-average background per in-band cell.
    flanks = [_flank_indices(int(c), guard, half, n_bins) for c in in_band]
    pass1 = np.empty_like(normalized)
    for row, center in enumerate(in_band):
        idx = flanks[row]
        bg = (
            _ota_background(gram[idx], params.sw_truncation)
            if idx.size
            else np.full(gram.shape[1], np.median(gram[int(center)]))
        )
        pass1[row] = gram[int(center)] / np.maximum(bg, _FLOOR)

    # Pass two: censor flanking bins that pass one flagged as candidate lines, so a tonal does not
    # estimate its background from its own comb teeth, then re-estimate and re-normalize. The peak
    # mask is carried on the full frequency axis so a flank gather can censor it without a Python
    # loop over flanking bins; out-of-band bins are never peaks.
    peak_full = np.zeros((n_bins, gram.shape[1]), dtype=bool)
    peak_full[in_band] = pass1 > _PEAK_CENSOR_RATIO
    for row, center in enumerate(in_band):
        idx = flanks[row]
        if not idx.size:
            normalized[row] = pass1[row]
            continue
        values = np.where(peak_full[idx], np.nan, gram[idx])
        bg2 = _ota_background(values, params.sw_truncation)
        normalized[row] = gram[int(center)] / np.maximum(bg2, _FLOOR)
    return normalized


def _temporal_median(
    gram_in_band: npt.NDArray[np.float64], window_frames: int
) -> npt.NDArray[np.float64]:
    """Per-bin temporal-median background over a centered frame window; divide it out.

    A moving or intermittent line spends little time in any one bin, so its per-bin temporal median
    stays near the background and the line survives normalization. When the window covers the whole
    record the estimate is the global per-bin median, the natural fallback near the one-sided floor.
    """
    n_frames = gram_in_band.shape[1]
    if window_frames >= n_frames or n_frames <= 1:
        background = np.median(gram_in_band, axis=1, keepdims=True)
        return np.asarray(gram_in_band / np.maximum(background, _FLOOR), dtype=np.float64)
    half = window_frames // 2
    out = np.empty_like(gram_in_band)
    for t in range(n_frames):
        lo = max(0, t - half)
        hi = min(n_frames, t + half + 1)
        background = np.median(gram_in_band[:, lo:hi], axis=1)
        out[:, t] = gram_in_band[:, t] / np.maximum(background, _FLOOR)
    return out
