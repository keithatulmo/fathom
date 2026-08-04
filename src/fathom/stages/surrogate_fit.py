"""The surrogate parameter-fitting stage.

This stage reads the vessel registry, selects the train split, and fits the surrogate line-statistic
distributions from those vessels only, refusing any calibration, test, or held-out vessel by
construction per SD3 Section 5.4. Its output is the fitted distributions, carrying the train roster
so the train-side-only derivation is auditable downstream.
"""

from __future__ import annotations

from typing import Any

from ..artifacts import Registry, SurrogateDist
from ..config import FitConfig
from ..surrogate.fit import fit_distributions, select_train_vessels
from .base import Stage, register


class SurrogateFitStage(Stage):
    """Fit surrogate distributions from the registry's train-split vessels only."""

    NAME = "surrogate_fit"
    VERSION = "1"
    OUTPUTS = {"distributions": "surrogate_dist"}

    def parse_config(self, raw: dict[str, Any]) -> FitConfig:
        return FitConfig.model_validate(raw)

    def run(self, inputs: dict[str, bytes], config: Any, seed: int) -> dict[str, bytes]:
        assert isinstance(config, FitConfig)
        registry = Registry.from_blob(inputs["registry"])
        train = select_train_vessels([dict(v) for v in registry.vessels])
        distributions = fit_distributions(train)
        return {"distributions": SurrogateDist(payload=distributions.to_dict()).to_blob()}


register(SurrogateFitStage())
