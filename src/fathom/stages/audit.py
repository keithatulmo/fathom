"""The split-integrity audit stage.

This stage wraps :func:`fathom.split.build_audit` as a pipeline node, so the audit is computed on
every run, emitted as a first-class artifact, and carried in lineage like any other output, per
SD3 Section 5.6. Whether the audit passed is read downstream by the scoring stage, which refuses
to report metrics for a run whose audit failed.
"""

from __future__ import annotations

from typing import Any

from ..artifacts import Registry, Truth
from ..config import AuditConfig
from ..split import build_audit
from .base import Stage, register


class SplitAuditStage(Stage):
    """Compute the split-integrity audit from a run's registry and truth."""

    NAME = "split_audit"
    VERSION = "1"
    OUTPUTS = {"audit": "split_audit"}

    def parse_config(self, raw: dict[str, Any]) -> AuditConfig:
        return AuditConfig.model_validate(raw)

    def run(self, inputs: dict[str, bytes], config: Any, seed: int) -> dict[str, bytes]:
        assert isinstance(config, AuditConfig)
        registry = Registry.from_blob(inputs["registry"])
        truth = Truth.from_blob(inputs["truth"])
        audit = build_audit(
            registry,
            truth,
            guard_hours=config.guard_hours,
            require_tier_one_eval=config.require_tier_one_eval,
        )
        return {"audit": audit.to_blob()}


register(SplitAuditStage())
