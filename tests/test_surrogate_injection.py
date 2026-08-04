"""Injection through the pipeline: lineage, determinism, and byte-identical regeneration."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from fathom.artifacts import AudioCorpus, SurrogateParams
from fathom.manifest import Manifest
from fathom.runner import Runner

MANIFEST_PATH = Path(__file__).resolve().parents[1] / "manifests" / "injection_fixtures.json"


def _manifest() -> Manifest:
    return Manifest.model_validate(json.loads(MANIFEST_PATH.read_text(encoding="utf-8")))


def test_injection_records_params_and_changes_the_background(tmp_path: Path) -> None:
    runner = Runner(tmp_path)
    try:
        result = runner.run(_manifest())
        background_hash = result.node_outputs["fixtures"]["audio"]
        injected_hash = result.node_outputs["inject"]["audio"]
        params_hash = result.node_outputs["inject"]["surrogate"]
        # Injection changes the audio, so its content hash differs from the background's.
        assert injected_hash != background_hash
        background = AudioCorpus.from_blob(runner._store.get(background_hash))
        injected = AudioCorpus.from_blob(runner._store.get(injected_hash))
        assert injected.samples.shape == background.samples.shape
        assert not np.array_equal(injected.samples, background.samples)
        # Every channel carries a recorded parameter set with an achieved (not absolute) level.
        params = SurrogateParams.from_blob(runner._store.get(params_hash))
        assert len(params.records) == background.samples.shape[0]
        record = params.records[0]
        assert "machinery" in record and "kinematics" in record and "propagation" in record
        assert "achieved_snr_db" in record
        assert abs(float(record["achieved_snr_db"]) - float(record["requested_snr_db"])) < 1.5
        # And there is no absolute-level field anywhere in the record.
        assert not any("source_level" in k or "absolute" in k for k in record)
    finally:
        runner.close()


def test_injection_regenerates_byte_identically(tmp_path: Path) -> None:
    runner = Runner(tmp_path)
    try:
        result = runner.run(_manifest())
    finally:
        runner.close()
    verifier = Runner(tmp_path)
    try:
        repro = verifier.reproduce(result.run_id)
    finally:
        verifier.close()
    assert repro.ok
    assert repro.checked > 0
    assert repro.mismatches == ()


def test_two_injection_runs_match_hashes(tmp_path: Path) -> None:
    runner = Runner(tmp_path)
    try:
        first = runner.run(_manifest())
        second = runner.run(_manifest())
    finally:
        runner.close()
    assert first.node_outputs["inject"]["audio"] == second.node_outputs["inject"]["audio"]
    assert first.node_outputs["inject"]["surrogate"] == second.node_outputs["inject"]["surrogate"]
