"""Propagation conditioning at proof scope.

SD2 Section 4 and WO-3 deliverable 3 condition the surrogate by propagation at proof scope:
spreading loss, frequency-dependent absorption, and a simple multipath or modal surrogate,
explicitly short of the track-two world model. Two relative effects are applied, both dimensionless
so no absolute level or range is expressed. The first is a frequency shaping: absorption attenuates
higher lines more than lower ones, and a multipath comb adds a frequency-selective interference
ripple, which together reshape the relative line spectrum at the reference range. The second is a
range envelope from spreading, which makes the surrogate loudest at the closest point of approach
and quieter away from it, evaluated against the kinematic range proxy. The absolute level is never
set here but by the signal-to-noise leveling downstream, so a stronger or weaker propagation only
reshapes the spectrum and the time envelope, never asserts a decibel.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import numpy.typing as npt


@dataclass(frozen=True)
class PropagationParams:
    """Dimensionless propagation parameters.

    ``absorption_coeff`` scales the exponential roll-off with frequency (the reference range is
    folded into it, so it carries no absolute range); ``multipath_depth`` in [0, 1) and
    ``multipath_spacing_hz`` set the interference ripple; ``spreading_exponent`` sets how fast the
    range envelope falls with the range proxy.
    """

    absorption_coeff: float
    multipath_depth: float
    multipath_spacing_hz: float
    spreading_exponent: float

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable record of the parameters for the lineage ledger."""
        return asdict(self)


def spectral_gain(
    freqs: npt.NDArray[np.float64], params: PropagationParams
) -> npt.NDArray[np.float64]:
    """Return the per-frequency relative gain from absorption and a multipath comb.

    Absorption is an exponential roll-off with frequency, and the multipath term is a bounded cosine
    ripple; the product is non-negative because the multipath depth is below one.
    """
    if not 0.0 <= params.multipath_depth < 1.0:
        raise ValueError("multipath_depth must be in [0, 1)")
    absorption = np.exp(-params.absorption_coeff * freqs)
    if params.multipath_spacing_hz <= 0.0:
        multipath = np.ones_like(freqs)
    else:
        multipath = 1.0 + params.multipath_depth * np.cos(
            2.0 * np.pi * freqs / params.multipath_spacing_hz
        )
    return np.ascontiguousarray(absorption * multipath, dtype=np.float64)


def range_envelope(
    range_proxy: npt.NDArray[np.float64], closest_proxy: float, params: PropagationParams
) -> npt.NDArray[np.float64]:
    """Return the per-time relative level from spreading, unity at the closest point of approach.

    The envelope is the closest-approach distance over the instantaneous range proxy, raised to the
    spreading exponent, so it peaks at one at the closest point and falls as the target recedes.
    """
    if closest_proxy <= 0.0:
        raise ValueError("closest_proxy must be positive")
    return np.ascontiguousarray(
        (closest_proxy / range_proxy) ** params.spreading_exponent, dtype=np.float64
    )
