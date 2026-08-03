"""Expected calibration error under equal-mass binning, with reliability data.

SD3 Section 5.5 reports calibration as expected calibration error under equal-mass binning,
fifteen bins by default with a floor on per-bin count, accompanied by reliability diagrams. ECE
under a binning choice is a biased estimator, so the binning is fixed in configuration, recorded
in lineage, and never tuned per result. Equal-mass binning places an equal number of samples in
each bin, which keeps every bin's estimate at comparable precision. Written fresh.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt


@dataclass(frozen=True)
class ReliabilityBin:
    """One bin of a reliability diagram: mean confidence, accuracy, and sample count."""

    mean_confidence: float
    accuracy: float
    count: int


@dataclass(frozen=True)
class CalibrationResult:
    """The scalar ECE and the reliability-diagram bins that produced it."""

    ece: float
    n_bins_effective: int
    bins: tuple[ReliabilityBin, ...]


def expected_calibration_error(
    confidences: npt.NDArray[np.float64],
    correct: npt.NDArray[np.bool_],
    *,
    n_bins: int = 15,
    min_bin_count: int = 1,
) -> CalibrationResult:
    """Compute equal-mass ECE and its reliability bins.

    The samples are sorted by confidence and split into as many equal-mass bins as the data
    supports without any bin falling below ``min_bin_count``; the effective bin count is returned
    so a small sample does not silently fabricate empty bins. ECE is the sample-weighted mean of
    the absolute gap between each bin's mean confidence and its accuracy.
    """
    if confidences.shape != correct.shape:
        raise ValueError("confidences and correct must share a shape")
    if confidences.ndim != 1:
        raise ValueError("confidences must be one-dimensional")
    n = confidences.shape[0]
    if n == 0:
        return CalibrationResult(ece=0.0, n_bins_effective=0, bins=())
    if np.any((confidences < 0.0) | (confidences > 1.0)):
        raise ValueError("confidences must lie in [0, 1]")

    effective = max(1, min(n_bins, n // max(1, min_bin_count)))
    order = np.argsort(confidences, kind="stable")
    conf_sorted = confidences[order]
    correct_sorted = correct[order].astype(np.float64)

    bins: list[ReliabilityBin] = []
    ece = 0.0
    for chunk_conf, chunk_correct in zip(
        np.array_split(conf_sorted, effective),
        np.array_split(correct_sorted, effective),
        strict=True,
    ):
        count = int(chunk_conf.shape[0])
        if count == 0:  # pragma: no cover - array_split yields non-empty chunks when n>=bins
            continue
        mean_conf = float(np.mean(chunk_conf))
        accuracy = float(np.mean(chunk_correct))
        bins.append(ReliabilityBin(mean_confidence=mean_conf, accuracy=accuracy, count=count))
        ece += (count / n) * abs(accuracy - mean_conf)

    return CalibrationResult(ece=ece, n_bins_effective=len(bins), bins=tuple(bins))
