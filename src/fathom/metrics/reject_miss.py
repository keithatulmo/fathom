"""The reject-versus-miss curve and the exact miss-rate bound.

SD3 Section 5.5 makes the miss axis the statistical crux: near-zero-miss claims are bounded by
trial count, and with zero observed misses in N independent present-target trials the one-sided
ninety-five percent exact upper bound on the miss rate is approximately three divided by N. The
harness therefore reports every miss estimate with its exact one-sided Clopper-Pearson bound, so
a small denominator cannot masquerade as a strong result. Written fresh from the definitions.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
from scipy.stats import beta


def miss_rate_upper_bound(misses: int, trials: int, alpha: float = 0.05) -> float:
    """Return the exact one-sided upper confidence bound on the miss rate.

    This is the Clopper-Pearson upper limit: for ``k`` misses in ``n`` present-target trials the
    ``1 - alpha`` upper bound is the ``1 - alpha`` quantile of a Beta(k + 1, n - k) distribution.
    With zero misses it reduces to ``1 - alpha ** (1 / n)``, whose leading behaviour is the
    rule-of-three ``-ln(alpha) / n`` and which is close to ``3 / n`` at ``alpha = 0.05``.
    """
    if trials <= 0:
        raise ValueError("trials must be positive")
    if not 0 <= misses <= trials:
        raise ValueError("misses must lie in [0, trials]")
    if misses == trials:
        return 1.0
    return float(beta.ppf(1.0 - alpha, misses + 1, trials - misses))


def rule_of_three(trials: int) -> float:
    """Return the rule-of-three approximation ``3 / n`` for the zero-miss upper bound."""
    if trials <= 0:
        raise ValueError("trials must be positive")
    return 3.0 / trials


@dataclass(frozen=True)
class RejectMissPoint:
    """A single operating point on the reject-versus-miss trade.

    ``miss_rate`` is the point estimate on present targets, ``miss_rate_upper`` its exact
    one-sided upper bound, and ``rejection_fraction`` the fraction of clutter trials the detector
    correctly rejects at this operating point. The two axes are reported together so the reader
    sees the honesty bound beside the achieved rejection.
    """

    site: str
    present_trials: int
    misses: int
    miss_rate: float
    miss_rate_upper: float
    clutter_trials: int
    rejection_fraction: float
    threshold: float
    alpha: float


def operating_point(
    truth_present: npt.NDArray[np.bool_],
    scores: npt.NDArray[np.float64],
    threshold: float,
    *,
    site: str = "all",
    alpha: float = 0.05,
) -> RejectMissPoint:
    """Compute the reject-versus-miss operating point at a decision threshold.

    A trial is decided present when its score is at least the threshold. A miss is a present
    target decided absent; a rejection is a clutter trial decided absent. Present and clutter
    trials are counted separately because the two axes of the trade are measured on disjoint
    populations.
    """
    if truth_present.shape != scores.shape:
        raise ValueError("truth_present and scores must share a shape")
    decided_present = scores >= threshold
    present = truth_present
    clutter = ~truth_present

    present_trials = int(np.count_nonzero(present))
    misses = int(np.count_nonzero(present & ~decided_present))
    clutter_trials = int(np.count_nonzero(clutter))
    rejections = int(np.count_nonzero(clutter & ~decided_present))

    miss_rate = misses / present_trials if present_trials else 0.0
    miss_upper = miss_rate_upper_bound(misses, present_trials, alpha) if present_trials else 0.0
    rejection_fraction = rejections / clutter_trials if clutter_trials else 0.0

    return RejectMissPoint(
        site=site,
        present_trials=present_trials,
        misses=misses,
        miss_rate=miss_rate,
        miss_rate_upper=miss_upper,
        clutter_trials=clutter_trials,
        rejection_fraction=rejection_fraction,
        threshold=float(threshold),
        alpha=alpha,
    )


def reject_miss_curve(
    truth_present: npt.NDArray[np.bool_],
    scores: npt.NDArray[np.float64],
    *,
    alpha: float = 0.05,
) -> list[RejectMissPoint]:
    """Sweep every distinct score threshold, returning the reject-versus-miss curve.

    The curve is the set of operating points obtained as the decision threshold ranges over the
    distinct observed scores, ordered by increasing threshold, so a reader can read the miss rate
    achieved for any required rejection fraction and the exact bound at each point.
    """
    thresholds = np.unique(scores)
    return [operating_point(truth_present, scores, float(t), alpha=alpha) for t in thresholds]
