"""The kinematic bearing track that makes the surrogate a moving target.

SD2 Section 4 and WO-3 deliverable 4 require the surrogate to sit on a bearing and bearing-rate
track consistent with the custody scenario, so injected targets move rather than sitting at fixed
bearing. This module models a straight-line constant-speed pass in dimensionless proxy units: the
target closes to a closest point of approach and recedes. That single geometry produces the three
quantities the injector needs, all without an absolute range: the bearing over time for the custody
scenario, a range proxy that rises and falls so the surrogate is loudest near the closest point, and
a Doppler factor that shifts the lines up while approaching and down while receding and passes
through unity at the closest point, because there the motion is tangential. The Doppler peak is a
dimensionless speed-to-sound-speed ratio, never an absolute speed.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import numpy.typing as npt


@dataclass(frozen=True)
class KinematicParams:
    """A straight-line pass in dimensionless proxy units.

    ``closest_proxy`` is the closest-point-of-approach distance and ``speed_proxy`` the along-track
    speed, both in the same arbitrary distance unit; only their ratio to the track length matters,
    so no absolute range is expressed. ``doppler_peak`` is the dimensionless speed-to-sound-speed
    ratio that scales the frequency shift.
    """

    bearing0_deg: float
    closest_proxy: float
    speed_proxy: float
    t_cpa_s: float
    doppler_peak: float

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable record of the parameters for the lineage ledger."""
        return asdict(self)


@dataclass(frozen=True)
class Track:
    """The evaluated pass: bearing, range proxy, and Doppler factor over time."""

    bearings_deg: npt.NDArray[np.float64]
    range_proxy: npt.NDArray[np.float64]
    doppler_factor: npt.NDArray[np.float64]


def evaluate_track(params: KinematicParams, times: npt.NDArray[np.float64]) -> Track:
    """Evaluate the bearing, range proxy, and Doppler factor of the pass over the given times.

    The along-track position is speed times elapsed time relative to the closest-approach time, the
    range proxy is the hypotenuse of that position and the closest-approach distance, the bearing is
    the arctangent of the two plus the initial bearing, and the Doppler factor is one minus the peak
    ratio times the radial velocity normalised to the along-track speed, so it is unity at the
    closest point and bounded by the peak ratio away from it.
    """
    if params.closest_proxy <= 0.0 or params.speed_proxy <= 0.0:
        raise ValueError("closest_proxy and speed_proxy must be positive")
    along = params.speed_proxy * (times - params.t_cpa_s)
    range_proxy = np.sqrt(params.closest_proxy**2 + along**2)
    bearings = params.bearing0_deg + np.degrees(np.arctan2(along, params.closest_proxy))
    # Radial velocity is the along-track speed projected onto the line of sight; normalised to the
    # along-track speed it is bounded in [-1, 1] and is zero at the closest point.
    radial_fraction = along / range_proxy
    doppler_factor = 1.0 - params.doppler_peak * radial_fraction
    return Track(
        bearings_deg=np.ascontiguousarray(bearings, dtype=np.float64),
        range_proxy=np.ascontiguousarray(range_proxy, dtype=np.float64),
        doppler_factor=np.ascontiguousarray(doppler_factor, dtype=np.float64),
    )
