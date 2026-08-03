"""The scoring stage.

This stage reads the detections, the truth, and the split-integrity audit, and assembles the
scored report per SD3 Section 5.5. It is pure and deterministic: it derives its bootstrap
randomness from the seed the runner supplies, which is fixed by the manifest, so re-execution
reproduces the report byte for byte. When the audit did not pass, the report withholds metrics
because a run whose audit failed is invalid by construction.
"""

from __future__ import annotations

from typing import Any

from ..artifacts import Detections, Report, SplitAudit, Truth
from ..config import ScoringConfig
from ..report import build_report_payload
from .base import Stage, register


class ScoringStage(Stage):
    """Assemble the scored report from detections, truth, and the split-integrity audit."""

    NAME = "scoring"
    VERSION = "1"
    OUTPUTS = {"report": "report"}

    def parse_config(self, raw: dict[str, Any]) -> ScoringConfig:
        return ScoringConfig.model_validate(raw)

    def run(self, inputs: dict[str, bytes], config: Any, seed: int) -> dict[str, bytes]:
        assert isinstance(config, ScoringConfig)
        detections = Detections.from_blob(inputs["detections"])
        truth = Truth.from_blob(inputs["truth"])
        audit = SplitAudit.from_blob(inputs["audit"])
        payload = build_report_payload(detections, truth, audit, config, seed)
        return {"report": Report(payload=payload).to_blob()}


register(ScoringStage())
