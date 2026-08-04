"""The E1 realism harness stage.

This stage injects surrogates across the signal-to-noise ladder onto background channels, processes
them and a held-out class through the identical reference front end and detector, and compares the
detectability-versus-signal-to-noise response and the rejector-response distribution, emitting the
E1 realism report of SD2 Section 7. On synthetic fixtures the held-out arm is a placeholder class,
proving the comparison machinery end to end (WO-3 AC4); the real held-out quietest vessel class is
read at AC7 and is never consulted in surrogate fitting, per the SD3 Section 5.4 leak rule. The
report carries no absolute level; every quantity is a dimensionless response or a signal-to-noise
ratio.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from ..artifacts import AudioCorpus, E1Report, SurrogateDist, Truth
from ..config import E1Config
from ..determinism import derive_seed
from ..e1 import (
    bootstrap_rate_interval,
    channel_responses,
    detection_rate,
    intervals_overlap,
    ks_two_sample,
    realism_judgment,
)
from ..surrogate.fit import SurrogateDistributions
from ..surrogate.inject import InjectionSpec, inject_background
from .base import Stage, register

_MIN_BACKGROUND = 4


def _subset(audio: AudioCorpus, rows: list[int]) -> AudioCorpus:
    return AudioCorpus(
        samples=np.ascontiguousarray(audio.samples[rows], dtype=np.float64),
        sample_rate=audio.sample_rate,
        channel_ids=tuple(audio.channel_ids[r] for r in rows),
    )


def _run_e1(
    audio: AudioCorpus,
    truth: Truth,
    distributions: SurrogateDistributions,
    config: E1Config,
    seed: int,
) -> dict[str, Any]:
    clutter = [i for i, ch in enumerate(truth.channels) if ch.get("kind") == "clutter"]
    if len(clutter) < _MIN_BACKGROUND:
        raise ValueError(
            f"E1 needs at least {_MIN_BACKGROUND} background channels, got {len(clutter)}"
        )
    half = len(clutter) // 2
    background_surrogate = _subset(audio, clutter[:half])
    background_held_out = _subset(audio, clutter[half:])
    band = (config.band_low_hz, config.band_high_hz)
    ladder = list(config.snr_ladder_db)

    def spec(snr: float) -> InjectionSpec:
        return InjectionSpec(
            band=band,
            requested_snr_db=snr,
            bearing0_deg=config.bearing0_deg,
            closest_proxy=config.closest_proxy,
            speed_proxy=config.speed_proxy,
            doppler_peak=config.doppler_peak,
            t_cpa_fraction=config.t_cpa_fraction,
            absorption_coeff=config.absorption_coeff,
            multipath_depth=config.multipath_depth,
            multipath_spacing_hz=config.multipath_spacing_hz,
            spreading_exponent=config.spreading_exponent,
        )

    surrogate_rate: list[float] = []
    surrogate_ci: list[tuple[float, float]] = []
    held_rate: list[float] = []
    held_ci: list[tuple[float, float]] = []
    surrogate_rejectors: list[float] = []
    held_rejectors: list[float] = []
    for index, snr in enumerate(ladder):
        injected_s, _ = inject_background(
            background_surrogate, distributions, spec(snr), derive_seed(seed, f"surrogate-{index}")
        )
        conf_s, rej_s = channel_responses(
            injected_s.samples,
            audio.sample_rate,
            config.nfft,
            config.hop,
            band,
            config.energy_scale,
        )
        injected_h, _ = inject_background(
            background_held_out, distributions, spec(snr), derive_seed(seed, f"heldout-{index}")
        )
        conf_h, rej_h = channel_responses(
            injected_h.samples,
            audio.sample_rate,
            config.nfft,
            config.hop,
            band,
            config.energy_scale,
        )
        surrogate_rate.append(detection_rate(conf_s, config.threshold))
        surrogate_ci.append(
            bootstrap_rate_interval(
                conf_s,
                config.threshold,
                config.bootstrap_resamples,
                config.bootstrap_alpha,
                derive_seed(seed, f"bs-surrogate-{index}"),
            )
        )
        held_rate.append(detection_rate(conf_h, config.threshold))
        held_ci.append(
            bootstrap_rate_interval(
                conf_h,
                config.threshold,
                config.bootstrap_resamples,
                config.bootstrap_alpha,
                derive_seed(seed, f"bs-heldout-{index}"),
            )
        )
        surrogate_rejectors.extend(rej_s.tolist())
        held_rejectors.extend(rej_h.tolist())

    overlaps = [intervals_overlap(surrogate_ci[k], held_ci[k]) for k in range(len(ladder))]
    operating = list(range(1, len(ladder) - 1)) if len(ladder) > 2 else list(range(len(ladder)))
    distance = ks_two_sample(np.array(surrogate_rejectors), np.array(held_rejectors))
    judgment = realism_judgment(overlaps, operating, distance, config.rejector_tolerance)
    return {
        "snr_ladder_db": ladder,
        "detectability": {
            "surrogate": {"rate": surrogate_rate, "ci": [list(c) for c in surrogate_ci]},
            "held_out": {"rate": held_rate, "ci": [list(c) for c in held_ci]},
            "overlap_per_rung": overlaps,
            "operating_indices": operating,
        },
        "rejector_response": {
            "two_sample_ks": distance,
            "tolerance": config.rejector_tolerance,
            "surrogate_n": len(surrogate_rejectors),
            "held_out_n": len(held_rejectors),
        },
        "realism": judgment,
        "held_out_source": "placeholder_synthetic",
        "note": (
            "The held-out arm is a placeholder synthetic class on fixtures (AC4); the real "
            "held-out quietest vessel class is read at AC7 and is never consulted in fitting."
        ),
        "owner_certified": config.owner_certified.model_dump(mode="json"),
    }


class E1Stage(Stage):
    """Run the E1 realism comparison and emit the realism report."""

    NAME = "e1"
    VERSION = "1"
    OUTPUTS = {"e1_report": "e1_report"}

    def parse_config(self, raw: dict[str, Any]) -> E1Config:
        return E1Config.model_validate(raw)

    def run(self, inputs: dict[str, bytes], config: Any, seed: int) -> dict[str, bytes]:
        assert isinstance(config, E1Config)
        audio = AudioCorpus.from_blob(inputs["audio"])
        truth = Truth.from_blob(inputs["truth"])
        distributions = SurrogateDistributions.from_dict(
            SurrogateDist.from_blob(inputs["distributions"]).payload
        )
        payload = _run_e1(audio, truth, distributions, config, seed)
        return {"e1_report": E1Report(payload=payload).to_blob()}


register(E1Stage())
