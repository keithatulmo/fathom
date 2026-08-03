"""Tests for runner behaviours: cycle detection, content-addressed reuse, and the CLI stub."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from fathom.cli import main
from fathom.manifest import Manifest
from fathom.runner import Runner, _topological_order

MANIFEST_PATH = Path(__file__).resolve().parents[1] / "manifests" / "acceptance.json"


def _manifest() -> Manifest:
    return Manifest.model_validate(json.loads(MANIFEST_PATH.read_text(encoding="utf-8")))


def test_cycle_is_rejected() -> None:
    manifest = Manifest(
        name="cyclic",
        seed=1,
        nodes=(
            {"id": "a", "stage": "front_end", "inputs": {"audio": "b:x"}, "config": {}},
            {"id": "b", "stage": "front_end", "inputs": {"audio": "a:x"}, "config": {}},
        ),
    )
    with pytest.raises(ValueError, match="cycle"):
        _topological_order(manifest)


def test_second_run_reuses_content_addressed_outputs(tmp_path: Path) -> None:
    manifest = _manifest()
    runner = Runner(tmp_path)
    try:
        runner.run(manifest)
        second = runner.run(manifest)
        # The second run's stage executions should all be cache reuses, since inputs, config,
        # code version, environment, and seeds are identical.
        executions = runner.ledger.get_stage_executions(second.run_id)
        assert executions
        assert all(execution.reused for execution in executions)
    finally:
        runner.close()


def test_topological_order_respects_dependencies() -> None:
    order = _topological_order(_manifest())
    assert order.index("fixtures") < order.index("frontend")
    assert order.index("frontend") < order.index("detect")
    assert order.index("detect") < order.index("score")
    assert order.index("audit") < order.index("score")


def test_demo_stub_returns_zero(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["demo"]) == 0
    out = capsys.readouterr().out
    assert "not yet implemented" in out
