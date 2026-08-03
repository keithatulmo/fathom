"""Coverage accounting.

SD3 Section 5.5 requires coverage accounting to report the fraction of beams and bins under
continuous un-cued search for every run, because target A5 requires that fraction to be one in a
proof-grade run, and a number the harness computes cannot be quietly asserted. The fraction is
reported as data so that a shortfall is visible rather than assumed away.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt


@dataclass(frozen=True)
class CoverageResult:
    """The coverage fraction and the channel counts that produced it."""

    channels_under_search: int
    total_channels: int
    fraction: float


def coverage_fraction(channels_under_search: int, total_channels: int) -> CoverageResult:
    """Return the fraction of channels under continuous un-cued search.

    A proof-grade run has every channel under search, so the fraction is one; any smaller value
    is a measured shortfall that the report carries rather than hides.
    """
    if total_channels <= 0:
        raise ValueError("total_channels must be positive")
    if not 0 <= channels_under_search <= total_channels:
        raise ValueError("channels_under_search must lie in [0, total_channels]")
    return CoverageResult(
        channels_under_search=channels_under_search,
        total_channels=total_channels,
        fraction=channels_under_search / total_channels,
    )


def coverage_from_mask(mask: npt.NDArray[np.bool_]) -> CoverageResult:
    """Return coverage accounting from a boolean per-channel search mask."""
    if mask.ndim != 1:
        raise ValueError("mask must be one-dimensional")
    total = int(mask.shape[0])
    under = int(np.count_nonzero(mask))
    return coverage_fraction(under, total)
