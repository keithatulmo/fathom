"""End-to-end acceptance tests: the run, the lineage, and the reproduction hash-equality check.

These tests exercise the acceptance path in an isolated state directory, so they assert the work
order's acceptance criteria directly: a scored report with a lineage row for every artifact (AC1),
a reproduction with zero hash mismatches (AC2), and the absence of PyTorch from the executed path
(AC7). Everything runs from a seed with no network access (AC5).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from fathom.manifest import Manifest
from fathom.runner import Runner

MANIFEST_PATH = Path(__file__).resolve().parents[1] / "manifests" / "acceptance.json"


@pytest.fixture
def manifest() -> Manifest:
    return Manifest.model_validate(json.loads(MANIFEST_PATH.read_text(encoding="utf-8")))


def test_acceptance_run_is_reportable_and_scored(tmp_path: Path, manifest: Manifest) -> None:
    runner = Runner(tmp_path)
    try:
        result = runner.run(manifest)
    finally:
        runner.close()

    assert result.reportable
    assert result.report_path is not None and result.report_path.is_file()

    document = json.loads(result.report_path.read_text(encoding="utf-8"))
    report = document["report"]
    point = report["reject_versus_miss"]["operating_point"]
    assert point["misses"] == 0
    assert point["miss_rate"] == 0.0
    # Zero misses over N present trials give an exact upper bound near the rule-of-three 3/N.
    assert 0.0 < point["miss_rate_upper"] < 0.2
    assert point["rejection_fraction"] == pytest.approx(1.0)
    assert report["calibration"]["ece"] < 0.05
    assert report["coverage"]["fraction"] == pytest.approx(1.0)
    assert report["audit"]["passed"] is True
    assert report["owner_certified"] == {"detection_range_r": None, "confirmer_pd": None}


def test_every_artifact_has_a_lineage_row(tmp_path: Path, manifest: Manifest) -> None:
    runner = Runner(tmp_path)
    try:
        result = runner.run(manifest)
        produced_hashes = {
            digest for outputs in result.node_outputs.values() for digest in outputs.values()
        }
        for digest in produced_hashes:
            assert runner.ledger.artifact_exists(digest), digest
        # The ledger's per-run artifact listing covers every produced artifact (AC1).
        run_artifacts = set(runner.ledger.list_run_artifacts(result.run_id))
        assert produced_hashes <= run_artifacts
    finally:
        runner.close()


def test_reproduction_has_zero_mismatches(tmp_path: Path, manifest: Manifest) -> None:
    runner = Runner(tmp_path)
    try:
        result = runner.run(manifest)
    finally:
        runner.close()

    # A fresh runner over the same state re-executes and verifies hashes (AC2).
    verifier = Runner(tmp_path)
    try:
        repro = verifier.reproduce(result.run_id)
    finally:
        verifier.close()

    assert repro.ok
    assert repro.checked > 0
    assert repro.mismatches == ()


def test_two_runs_produce_identical_report_hash(tmp_path: Path, manifest: Manifest) -> None:
    runner = Runner(tmp_path)
    try:
        first = runner.run(manifest)
        second = runner.run(manifest)
    finally:
        runner.close()
    assert first.report_hash == second.report_hash


def test_pytorch_absent_from_executed_path(tmp_path: Path, manifest: Manifest) -> None:
    runner = Runner(tmp_path)
    try:
        runner.run(manifest)
    finally:
        runner.close()
    # PyTorch is confined to training extras and must not be pulled into the acceptance path (AC7).
    assert "torch" not in sys.modules
