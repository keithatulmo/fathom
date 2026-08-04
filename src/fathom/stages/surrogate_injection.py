"""The surrogate-injection stage.

This stage consumes a background audio corpus and the fitted surrogate distributions and injects one
surrogate onto each background channel at the requested signal-to-noise ratio, emitting the injected
corpus and the per-channel parameter record. The record carries every drawn parameter and the seed,
so any injected scenario regenerates byte for byte through the reproduction command, which is the
injection-reproducibility criterion satisfied by construction. No absolute level is set; the level
is the requested ratio, re-measured after injection and recorded.
"""

from __future__ import annotations

from typing import Any

from ..artifacts import AudioCorpus, SurrogateDist, SurrogateParams
from ..config import InjectionConfig
from ..surrogate.fit import SurrogateDistributions
from ..surrogate.inject import InjectionSpec, inject_background
from .base import Stage, register


class SurrogateInjectionStage(Stage):
    """Inject a surrogate onto each background channel at a controlled signal-to-noise ratio."""

    NAME = "surrogate_injection"
    VERSION = "1"
    OUTPUTS = {"audio": "audio_corpus", "surrogate": "surrogate_params"}

    def parse_config(self, raw: dict[str, Any]) -> InjectionConfig:
        return InjectionConfig.model_validate(raw)

    def run(self, inputs: dict[str, bytes], config: Any, seed: int) -> dict[str, bytes]:
        assert isinstance(config, InjectionConfig)
        background = AudioCorpus.from_blob(inputs["background"])
        distributions = SurrogateDistributions.from_dict(
            SurrogateDist.from_blob(inputs["distributions"]).payload
        )
        spec = InjectionSpec(
            band=(config.band_low_hz, config.band_high_hz),
            requested_snr_db=config.snr_db,
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
        injected, records = inject_background(background, distributions, spec, seed)
        return {
            "audio": injected.to_blob(),
            "surrogate": SurrogateParams(records=tuple(records)).to_blob(),
        }


register(SurrogateInjectionStage())
