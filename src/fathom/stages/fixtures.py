"""The seeded synthetic fixture generator.

SD3 Section 5.7 requires a seeded synthetic fixture generator that produces a small corpus with
known truth at run time, so no binary data lives in the repository. Present-target channels carry
an in-band tonal above noise; clutter channels carry noise alone. Vessels are the split unit, one
vessel is quarantined to exercise the roster, and clutter splits by site-and-time rather than by
vessel per SD3 Section 5.1. Evaluation truth is tier one by construction.

The corpus is entirely synthetic and carries no owner-certified value of any kind.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from ..artifacts import AudioCorpus, Registry, Truth
from ..config import FixtureConfig
from ..determinism import derive_seed
from .base import Stage, register

_SPLITS = ("train", "calibration", "test")


class FixtureStage(Stage):
    """Generate a seeded synthetic acoustic corpus with known, tiered truth."""

    NAME = "fixtures"
    VERSION = "1"
    OUTPUTS = {"audio": "audio_corpus", "registry": "registry", "truth": "truth"}

    def parse_config(self, raw: dict[str, Any]) -> FixtureConfig:
        return FixtureConfig.model_validate(raw)

    def run(self, inputs: dict[str, bytes], config: Any, seed: int) -> dict[str, bytes]:
        assert isinstance(config, FixtureConfig)
        vessels = self._vessels(config)
        channels = self._channels(config, vessels)
        samples = self._synthesize(config, channels, seed)

        channel_ids = tuple(channel["channel_id"] for channel in channels)
        audio = AudioCorpus(
            samples=samples,
            sample_rate=config.sample_rate,
            channel_ids=channel_ids,
        )
        registry = Registry(
            vessels=tuple(vessels),
            guard_hours=config.guard_hours,
            calibration_target_events=config.calibration_target_events,
        )
        truth = Truth(channels=tuple(channels))
        return {
            "audio": audio.to_blob(),
            "registry": registry.to_blob(),
            "truth": truth.to_blob(),
        }

    def _vessels(self, config: FixtureConfig) -> list[dict[str, Any]]:
        vessels: list[dict[str, Any]] = []
        index = 0
        for split in _SPLITS:
            for i in range(config.vessels_per_split):
                vessels.append(
                    {
                        "vessel_id": f"V-{split[:2]}-{i:02d}",
                        "site": f"site{index % config.sites}",
                        "split": split,
                        "truth_tier": 1,
                        "quarantined": False,
                    }
                )
                index += 1
        for q in range(config.quarantined_vessels):
            # Quarantined identities are usable only as unattributed training background at
            # tier three, never in evaluation, per SD3 Section 5.1.
            vessels.append(
                {
                    "vessel_id": f"V-qz-{q:02d}",
                    "site": f"site{q % config.sites}",
                    "split": "train",
                    "truth_tier": 3,
                    "quarantined": True,
                }
            )
        return vessels

    def _channels(
        self, config: FixtureConfig, vessels: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        channels: list[dict[str, Any]] = []
        index = 0
        for vessel in vessels:
            if vessel["quarantined"]:
                # One background channel, present=False, tier three, training only.
                channels.append(
                    {
                        "channel_id": f"ch{index:04d}_{vessel['split']}_qzbg",
                        "vessel_id": vessel["vessel_id"],
                        "site": vessel["site"],
                        "split": vessel["split"],
                        "present": False,
                        "truth_tier": 3,
                        "kind": "quarantine_background",
                    }
                )
                index += 1
                continue
            for _ in range(config.recordings_per_present_vessel):
                channels.append(
                    {
                        "channel_id": f"ch{index:04d}_{vessel['split']}_tgt",
                        "vessel_id": vessel["vessel_id"],
                        "site": vessel["site"],
                        "split": vessel["split"],
                        "present": True,
                        "truth_tier": 1,
                        "kind": "target",
                    }
                )
                index += 1
        for j in range(config.clutter_channels):
            split = _SPLITS[j % len(_SPLITS)]
            # Clutter carries no vessel identity, so it splits by site-and-time block, and a
            # verified quiet interval is tier-one truth per SD3 Section 5.3.
            channels.append(
                {
                    "channel_id": f"ch{index:04d}_{split}_clut",
                    "vessel_id": None,
                    "site": f"site{j % config.sites}",
                    "split": split,
                    "present": False,
                    "truth_tier": 1,
                    "kind": "clutter",
                }
            )
            index += 1
        return channels

    def _synthesize(
        self, config: FixtureConfig, channels: list[dict[str, Any]], seed: int
    ) -> np.ndarray:
        n_samples = int(round(config.duration_s * config.sample_rate))
        times = np.arange(n_samples, dtype=np.float64) / config.sample_rate
        tonal = np.sin(2.0 * np.pi * config.target_tonal_hz * times)
        samples = np.empty((len(channels), n_samples), dtype=np.float64)
        for row, channel in enumerate(channels):
            # Each channel draws from its own stream keyed by its identity, so the corpus is
            # deterministic and channels are independent regardless of ordering.
            channel_rng = np.random.default_rng(derive_seed(seed, str(channel["channel_id"])))
            noise = channel_rng.standard_normal(n_samples) * config.noise_amplitude
            if channel["present"]:
                samples[row] = config.target_amplitude * tonal + noise
            else:
                samples[row] = noise
        return samples


register(FixtureStage())
