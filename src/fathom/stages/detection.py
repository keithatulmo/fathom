"""The trivial band-limited energy-threshold detector.

SD3 Section 5.7 calls for a deliberately trivial band-limited energy-threshold detector to drive
the acceptance path. This stage measures mean power in a fixed frequency band per channel and maps
it to a confidence in the unit interval with a saturating exponential, so a present target's
in-band energy yields a confidence near one and a clutter channel's yields a confidence near zero.
It is not a real detector, and it holds no learned parameters; the real detection layer is selected
later at experiment E3 under SD5.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from ..artifacts import Detections, Gram
from ..config import DetectionConfig
from .base import Stage, register


class DetectorStage(Stage):
    """Score each channel by its in-band power, expressed as a confidence."""

    NAME = "detector"
    VERSION = "1"
    OUTPUTS = {"detections": "detections"}

    def parse_config(self, raw: dict[str, Any]) -> DetectionConfig:
        return DetectionConfig.model_validate(raw)

    def run(self, inputs: dict[str, bytes], config: Any, seed: int) -> dict[str, bytes]:
        assert isinstance(config, DetectionConfig)
        gram = Gram.from_blob(inputs["gram"])

        band = (gram.freqs >= config.band_low_hz) & (gram.freqs <= config.band_high_hz)
        out_of_band = ~band
        if not np.any(band):
            raise ValueError("detection band contains no frequency bins")
        if not np.any(out_of_band):
            raise ValueError("detection band leaves no out-of-band reference bins")

        # A crude scale-invariant SNR: mean in-band power over mean out-of-band power. A standing
        # tonal drives the ratio well above one; broadband clutter sits near one.
        band_energy = gram.gram[:, band, :].mean(axis=(1, 2))
        reference = gram.gram[:, out_of_band, :].mean(axis=(1, 2))
        ratio = band_energy / np.maximum(reference, 1e-12)
        confidence = 1.0 - np.exp(-np.maximum(0.0, ratio - 1.0) / config.energy_scale)

        detections = Detections(
            channel_ids=gram.channel_ids,
            confidence=np.ascontiguousarray(confidence, dtype=np.float64),
            band_energy=np.ascontiguousarray(band_energy, dtype=np.float64),
            band_low_hz=config.band_low_hz,
            band_high_hz=config.band_high_hz,
        )
        return {"detections": detections.to_blob()}


register(DetectorStage())
