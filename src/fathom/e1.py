"""The E1 realism analysis: the reference front end, detector, rejector, and the comparison.

SD2 Section 7 designs E1 to convert the surrogate selection from provisional to closed. It injects
surrogates onto real in-band background across a swept signal-to-noise ladder, processes them and
the held-out quietest real vessel class through an identical documented reference front end and
detector, and compares the detectability-versus-signal-to-noise response and the rejector-response
distribution. The realism criterion is met when the two are statistically indistinguishable across
the operating band, judged by overlapping confidence intervals on the detectability curve and a
two-sample distance below a stated tolerance on the rejector response. This module holds the pure
analysis: the reference front end and detector are the same STFT power gram and in-band energy ratio
the acceptance path uses, the reference rejector is a line-prominence statistic standing in until
SD6 closes, and the comparison is a bootstrap over channels. It asserts no absolute level; every
input is a dimensionless response.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

from .determinism import rng

_FLOOR = 1e-12


def channel_responses(
    samples: npt.NDArray[np.float64],
    sample_rate: float,
    nfft: int,
    hop: int,
    band: tuple[float, float],
    energy_scale: float,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Return the detector confidence and rejector response for each channel of a sample block.

    The front end is the STFT power gram; the detector confidence is the saturating map of the
    in-band over out-of-band power ratio, the same reference detector the acceptance path uses; the
    rejector response is the in-band line prominence, the ratio of the strongest in-band bin to the
    median in-band bin, which is large for a narrowband target and near one for broadband clutter.
    """
    from .stages.dsp import _hann, stft_power  # local import avoids a stage-registry import cycle

    window = _hann(nfft)
    freqs = np.fft.rfftfreq(nfft, d=1.0 / sample_rate).astype(np.float64)
    in_band = (freqs >= band[0]) & (freqs <= band[1])
    out_band = ~in_band
    if not np.any(in_band) or not np.any(out_band):
        raise ValueError("band must select some in-band and some out-of-band frequency bins")

    confidences = np.empty(samples.shape[0], dtype=np.float64)
    rejectors = np.empty(samples.shape[0], dtype=np.float64)
    for row in range(samples.shape[0]):
        gram = stft_power(samples[row], nfft, hop, window)
        band_mean = float(gram[in_band, :].mean())
        out_mean = float(gram[out_band, :].mean())
        ratio = band_mean / max(out_mean, _FLOOR)
        confidences[row] = 1.0 - np.exp(-max(0.0, ratio - 1.0) / energy_scale)
        per_bin = gram[in_band, :].mean(axis=1)
        rejectors[row] = float(per_bin.max()) / max(float(np.median(per_bin)), _FLOOR)
    return confidences, rejectors


def detection_rate(confidences: npt.NDArray[np.float64], threshold: float) -> float:
    """Return the fraction of channels whose confidence reaches the operating threshold."""
    if confidences.shape[0] == 0:
        return 0.0
    return float(np.mean(confidences >= threshold))


def bootstrap_rate_interval(
    confidences: npt.NDArray[np.float64],
    threshold: float,
    resamples: int,
    alpha: float,
    seed: int,
) -> tuple[float, float]:
    """Return a percentile bootstrap confidence interval on the detection rate over channels."""
    n = confidences.shape[0]
    if n == 0:
        return (0.0, 0.0)
    generator = rng(seed)
    rates = np.empty(resamples, dtype=np.float64)
    for i in range(resamples):
        draw = generator.integers(0, n, size=n)
        rates[i] = float(np.mean(confidences[draw] >= threshold))
    lo = float(np.quantile(rates, alpha / 2.0))
    hi = float(np.quantile(rates, 1.0 - alpha / 2.0))
    return (lo, hi)


def intervals_overlap(a: tuple[float, float], b: tuple[float, float]) -> bool:
    """Return whether two closed intervals overlap."""
    return not (a[1] < b[0] or b[1] < a[0])


def ks_two_sample(a: npt.NDArray[np.float64], b: npt.NDArray[np.float64]) -> float:
    """Return the two-sample Kolmogorov-Smirnov distance between two response samples.

    The distance is the greatest gap between the two empirical distribution functions, bounded in
    [0, 1], zero when the samples are identical and one when they are fully separated; it is the
    documented two-sample distance on the rejector response until a rejector is selected at SD6.
    """
    if a.shape[0] == 0 or b.shape[0] == 0:
        return 1.0
    grid = np.concatenate([a, b])
    grid.sort()
    cdf_a = np.searchsorted(np.sort(a), grid, side="right") / a.shape[0]
    cdf_b = np.searchsorted(np.sort(b), grid, side="right") / b.shape[0]
    return float(np.max(np.abs(cdf_a - cdf_b)))


def realism_judgment(
    overlaps: list[bool], operating_indices: list[int], rejector_distance: float, tolerance: float
) -> dict[str, object]:
    """Score realism per SD2 Section 3 and decide the pass against the operating-band criterion.

    The criterion of SD2 Section 7 is met when the detectability intervals overlap across the
    operating band and the rejector distance is below tolerance. The score is five when they also
    overlap in the tails, three when they agree in the operating region but diverge in the tails or
    the distance is within twice tolerance, and one when they diverge inside the operating region.
    """
    all_overlap = all(overlaps)
    operating_overlap = all(overlaps[i] for i in operating_indices) if operating_indices else False
    within_tol = rejector_distance < tolerance
    if all_overlap and within_tol:
        score = 5
    elif operating_overlap and rejector_distance < 2.0 * tolerance:
        score = 3
    else:
        score = 1
    return {
        "score": score,
        "passed": bool(operating_overlap and within_tol),
        "all_overlap": all_overlap,
        "operating_overlap": operating_overlap,
        "rejector_distance": rejector_distance,
        "rejector_tolerance": tolerance,
    }
