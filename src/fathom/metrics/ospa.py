"""OSPA and windowed OSPA(2), the supplementary tracking diagnostics.

SD3 Section 5.5 carries OSPA and windowed OSPA(2) as supplementary diagnostics per their
published definitions. OSPA (Schuhmacher, Vo, and Vo, 2008) is a metric between two finite sets
that combines a localization term, computed under the optimal assignment, with a cardinality
penalty at cutoff ``c``. OSPA(2) (Beard, Vo, and Vo, 2017) applies OSPA over a time window using
a time-averaged base distance between whole tracks, so it scores label continuity and not only
per-frame position. Both are implemented fresh from those definitions.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping

import numpy as np
import numpy.typing as npt
from scipy.optimize import linear_sum_assignment

# A track is a mapping from time index to a position vector; a label indexes a track.
Track = Mapping[int, npt.NDArray[np.float64]]


def ospa(
    x: npt.NDArray[np.float64],
    y: npt.NDArray[np.float64],
    c: float,
    p: float = 1.0,
) -> float:
    """Return the OSPA distance between two finite sets of points.

    ``x`` and ``y`` are arrays of shape ``(m, d)`` and ``(n, d)``. Two empty sets are at distance
    zero; if exactly one set is empty the distance is the cutoff ``c``. Otherwise the localization
    cost is the minimum-assignment sum of per-pair distances truncated at ``c``, the cardinality
    penalty charges ``c`` for each unmatched element, and the total is normalized by the larger
    cardinality and taken to the ``1 / p`` power.
    """
    if c <= 0:
        raise ValueError("cutoff c must be positive")
    if p <= 0:
        raise ValueError("order p must be positive")
    m = 0 if x.size == 0 else x.shape[0]
    n = 0 if y.size == 0 else y.shape[0]
    if m == 0 and n == 0:
        return 0.0
    if m == 0 or n == 0:
        return float(c)

    diff = x[:, None, :] - y[None, :, :]
    dist = np.linalg.norm(diff, axis=2)
    truncated = np.minimum(c, dist) ** p

    row, col = linear_sum_assignment(truncated)
    card_max = max(m, n)
    card_min = min(m, n)
    localization = float(truncated[row, col].sum())
    cardinality = (c**p) * (card_max - card_min)
    return float(((localization + cardinality) / card_max) ** (1.0 / p))


def _track_distance(gt: Track, est: Track, window: tuple[int, ...], c: float, p: float) -> float:
    """Time-averaged truncated distance between two tracks over a window (the OSPA(2) base)."""
    total = 0.0
    denom = 0
    for t in window:
        xi = gt.get(t)
        yj = est.get(t)
        if xi is not None and yj is not None:
            total += float(min(c, float(np.linalg.norm(xi - yj)))) ** p
            denom += 1
        elif xi is not None or yj is not None:
            total += c**p
            denom += 1
    if denom == 0:
        return 0.0
    return float((total / denom) ** (1.0 / p))


def windowed_ospa2(
    gt_tracks: Mapping[int, Track],
    est_tracks: Mapping[int, Track],
    window: Iterable[int],
    c: float,
    p: float = 1.0,
) -> float:
    """Return the windowed OSPA(2) distance between labeled ground-truth and estimated tracks.

    Each track is a mapping from time to position. A track is present in the window if it exists
    at any time in it. The base distance between a ground-truth track and an estimated track is
    the time-averaged truncated distance over the window, charging the cutoff at times where only
    one of the pair exists. OSPA is then taken over the sets of present tracks using that base
    distance, so a track that keeps its identity across the window scores better than one that
    fragments.
    """
    if c <= 0:
        raise ValueError("cutoff c must be positive")
    if p <= 0:
        raise ValueError("order p must be positive")
    win = tuple(window)

    gt_labels = [label for label, tr in gt_tracks.items() if any(t in tr for t in win)]
    est_labels = [label for label, tr in est_tracks.items() if any(t in tr for t in win)]
    m = len(gt_labels)
    n = len(est_labels)
    if m == 0 and n == 0:
        return 0.0
    if m == 0 or n == 0:
        return float(c)

    cost = np.empty((m, n), dtype=np.float64)
    for i, gt_label in enumerate(gt_labels):
        for j, est_label in enumerate(est_labels):
            base = _track_distance(gt_tracks[gt_label], est_tracks[est_label], win, c, p)
            cost[i, j] = base**p

    row, col = linear_sum_assignment(cost)
    card_max = max(m, n)
    card_min = min(m, n)
    localization = float(cost[row, col].sum())
    cardinality = (c**p) * (card_max - card_min)
    return float(((localization + cardinality) / card_max) ** (1.0 / p))
