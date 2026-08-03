"""The custody metric set: label consistency, swaps, losses, and time-to-reacquire.

SD3 Section 5.5 fixes these to the surviving trade-study definitions, implemented fresh: label
consistency is the fraction of held time a target keeps its majority label; swaps and losses are
counted per run; and time-to-reacquire is the interval from truth reappearance after a fade to
correct re-association, reported as a distribution with no threshold asserted because none is
ratified. A track's assigned label is a non-negative integer, and the sentinel ``-1`` marks a
frame in which the target is not held.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

UNASSOCIATED = -1


def label_consistency(assigned: npt.NDArray[np.int64]) -> float:
    """Return the fraction of held time a target keeps its majority label.

    Held time is the set of frames in which the target is associated to some track, that is where
    the assigned label is non-negative. Frames in which the target is not held do not count toward
    the denominator. A target that is never held has no consistency defined and raises.
    """
    if assigned.ndim != 1:
        raise ValueError("assigned must be one-dimensional")
    held = assigned[assigned >= 0]
    if held.size == 0:
        raise ValueError("target is never held; label consistency is undefined")
    _, counts = np.unique(held, return_counts=True)
    return float(counts.max() / held.size)


def swap_count(assigned: npt.NDArray[np.int64]) -> int:
    """Count label swaps: frames where a held target's label differs from the previous frame's.

    Both the current and previous frame must be held for a change to count as a swap; a transition
    into or out of the held state is a reacquire or a loss, not a swap.
    """
    if assigned.ndim != 1:
        raise ValueError("assigned must be one-dimensional")
    if assigned.size < 2:
        return 0
    previous = assigned[:-1]
    current = assigned[1:]
    both_held = (previous >= 0) & (current >= 0)
    return int(np.count_nonzero(both_held & (previous != current)))


def loss_count(assigned: npt.NDArray[np.int64]) -> int:
    """Count losses: transitions from a held frame to an unassociated frame."""
    if assigned.ndim != 1:
        raise ValueError("assigned must be one-dimensional")
    if assigned.size < 2:
        return 0
    previous = assigned[:-1]
    current = assigned[1:]
    return int(np.count_nonzero((previous >= 0) & (current < 0)))


def time_to_reacquire(
    present: npt.NDArray[np.bool_],
    associated: npt.NDArray[np.bool_],
) -> list[int]:
    """Return the distribution of frames from truth reappearance to correct re-association.

    A reappearance is a frame in which the target is present and was absent the frame before. For
    each reappearance the time-to-reacquire is the number of frames until the target is next
    correctly associated; reappearances never followed by an association within the record are
    censored and omitted, because no threshold or timeout is asserted.
    """
    if present.shape != associated.shape:
        raise ValueError("present and associated must share a shape")
    if present.ndim != 1:
        raise ValueError("present and associated must be one-dimensional")
    n = present.shape[0]
    times: list[int] = []
    for r in range(1, n):
        if present[r] and not present[r - 1]:
            follow = np.nonzero(associated[r:])[0]
            if follow.size:
                times.append(int(follow[0]))
    return times
