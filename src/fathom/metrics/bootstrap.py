"""Cluster bootstrap over the exchangeable unit.

SD3 Section 5.5 requires that uncertainty on every reported comparison use a cluster bootstrap
resampling vessels, not clips, because the vessel is the exchangeable unit and clip-level
resampling understates variance through intra-vessel correlation. The default resample count is
one thousand, recorded in configuration rather than buried here. The resampling is seeded, so the
interval is reproducible. Written fresh.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

Statistic = Callable[[npt.NDArray[np.float64]], float]


@dataclass(frozen=True)
class BootstrapResult:
    """A point estimate and a percentile confidence interval from a cluster bootstrap."""

    estimate: float
    ci_low: float
    ci_high: float
    n_resamples: int
    alpha: float


def cluster_bootstrap(
    units: Sequence[npt.NDArray[np.float64]],
    statistic: Statistic,
    *,
    seed: int,
    n_resamples: int = 1000,
    alpha: float = 0.05,
) -> BootstrapResult:
    """Bootstrap a statistic by resampling whole units with replacement.

    Each element of ``units`` holds the per-item values of one exchangeable unit, for example one
    vessel's per-clip scores. The point estimate is the statistic over the pooled items. Each
    resample draws ``len(units)`` units with replacement, pools their items, and recomputes the
    statistic; the confidence interval is the central ``1 - alpha`` percentile band of that
    distribution. Resampling units rather than items keeps intra-unit correlation from deflating
    the interval.
    """
    if len(units) == 0:
        raise ValueError("at least one unit is required")
    if n_resamples <= 0:
        raise ValueError("n_resamples must be positive")
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must lie in (0, 1)")

    pooled = np.concatenate(list(units))
    estimate = float(statistic(pooled))

    rng = np.random.default_rng(seed)
    n_units = len(units)
    draws = np.empty(n_resamples, dtype=np.float64)
    for i in range(n_resamples):
        choice = rng.integers(0, n_units, size=n_units)
        resampled = np.concatenate([units[k] for k in choice])
        draws[i] = statistic(resampled)

    lower = float(np.quantile(draws, alpha / 2.0))
    upper = float(np.quantile(draws, 1.0 - alpha / 2.0))
    return BootstrapResult(
        estimate=estimate,
        ci_low=lower,
        ci_high=upper,
        n_resamples=n_resamples,
        alpha=alpha,
    )
