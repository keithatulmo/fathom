"""The injection orchestrator: background plus surrogate at a controlled ratio, per channel.

WO-3 deliverable 6 makes injection a pipeline step that consumes real or fixture background and
emits an injected channel with its full parameter set and seed recorded, regenerable from the seed.
This module is the pure core the stage wraps. For each background channel it draws a machinery
parameter set from the distributions, builds the line families, conditions them by a moving-target
kinematic track and propagation, synthesises the waveform, and levels it onto the background at the
requested signal-to-noise ratio. Every draw is seeded from the run seed and the channel identity, so
the same seed regenerates the same injected channel byte for byte. No absolute level is set; the
level is only the requested ratio, re-measured and recorded.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..artifacts import AudioCorpus
from ..determinism import derive_seed
from .fit import SurrogateDistributions, draw_machinery
from .kinematics import KinematicParams, evaluate_track
from .level import inject_at_snr
from .lines import build_lines
from .propagation import PropagationParams
from .synth import synthesize


@dataclass(frozen=True)
class InjectionSpec:
    """The dimensionless injection parameters shared across a batch's channels."""

    band: tuple[float, float]
    requested_snr_db: float
    bearing0_deg: float
    closest_proxy: float
    speed_proxy: float
    doppler_peak: float
    t_cpa_fraction: float
    absorption_coeff: float
    multipath_depth: float
    multipath_spacing_hz: float
    spreading_exponent: float


def inject_background(
    background: AudioCorpus,
    distributions: SurrogateDistributions,
    spec: InjectionSpec,
    seed: int,
) -> tuple[AudioCorpus, list[dict[str, object]]]:
    """Inject one surrogate per background channel; return the injected corpus and the records.

    Each channel's machinery and synthesis draw from a seed derived from the run seed and the
    channel identity, so the injection is deterministic and regenerable. The track and propagation
    are shared across channels; the per-channel record carries every parameter and the achieved
    signal-to-noise ratio re-measured after injection.
    """
    sample_rate = background.sample_rate
    n_samples = background.samples.shape[1]
    duration_s = n_samples / sample_rate
    times = np.arange(n_samples, dtype=np.float64) / sample_rate
    kinematics = KinematicParams(
        bearing0_deg=spec.bearing0_deg,
        closest_proxy=spec.closest_proxy,
        speed_proxy=spec.speed_proxy,
        t_cpa_s=spec.t_cpa_fraction * duration_s,
        doppler_peak=spec.doppler_peak,
    )
    track = evaluate_track(kinematics, times)
    propagation = PropagationParams(
        absorption_coeff=spec.absorption_coeff,
        multipath_depth=spec.multipath_depth,
        multipath_spacing_hz=spec.multipath_spacing_hz,
        spreading_exponent=spec.spreading_exponent,
    )

    injected = np.empty_like(background.samples)
    records: list[dict[str, object]] = []
    for row, channel_id in enumerate(background.channel_ids):
        channel_seed = derive_seed(seed, str(channel_id))
        machinery = draw_machinery(distributions, derive_seed(channel_seed, "machinery"))
        lines = build_lines(machinery)
        signal = synthesize(
            lines, track, propagation, sample_rate, duration_s, derive_seed(channel_seed, "synth")
        )
        channel_injected, achieved = inject_at_snr(
            signal, background.samples[row], spec.band, sample_rate, spec.requested_snr_db
        )
        injected[row] = channel_injected
        records.append(
            {
                "channel_id": str(channel_id),
                "seed": channel_seed,
                "requested_snr_db": spec.requested_snr_db,
                "achieved_snr_db": achieved,
                "band_hz": [spec.band[0], spec.band[1]],
                "machinery": machinery.to_dict(),
                "kinematics": kinematics.to_dict(),
                "propagation": propagation.to_dict(),
            }
        )
    corpus = AudioCorpus(
        samples=np.ascontiguousarray(injected, dtype=np.float64),
        sample_rate=sample_rate,
        channel_ids=background.channel_ids,
    )
    return corpus, records
