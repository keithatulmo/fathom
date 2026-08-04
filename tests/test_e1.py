"""Tests for the E1 realism analysis and the end-to-end harness on fixtures (AC4)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from fathom.artifacts import E1Report
from fathom.determinism import rng
from fathom.e1 import (
    channel_responses,
    held_out_null_and_test,
    intervals_overlap,
    ks_two_sample,
    null_verdict,
    realism_judgment,
    wasserstein_distance,
)
from fathom.manifest import Manifest
from fathom.runner import Runner

MANIFEST_PATH = Path(__file__).resolve().parents[1] / "manifests" / "e1_fixtures.json"


def test_intervals_overlap() -> None:
    assert intervals_overlap((0.0, 1.0), (0.5, 2.0))
    assert not intervals_overlap((0.0, 1.0), (2.0, 3.0))


def test_ks_two_sample_bounds() -> None:
    a = np.linspace(0.0, 1.0, 50)
    assert ks_two_sample(a, a) == 0.0
    assert ks_two_sample(a, a + 10.0) == 1.0


def test_wasserstein_distance_is_tail_aware() -> None:
    a = np.linspace(0.0, 1.0, 50)
    assert wasserstein_distance(a, a) == 0.0
    # A pure shift moves the whole distribution; Wasserstein equals the shift, where KS saturates.
    assert wasserstein_distance(a, a + 5.0) == pytest.approx(5.0, abs=0.05)


def test_null_verdict_passes_when_surrogate_matches_the_class() -> None:
    generator = rng(7)
    held_out = generator.normal(1.0, 0.3, size=13)
    surrogate = generator.normal(1.0, 0.3, size=500)  # same distribution as the class
    null, test = held_out_null_and_test(surrogate, held_out, resamples=300, seed=1)
    verdict = null_verdict(null, test)
    assert verdict["passed"] is True
    assert float(verdict["test_median"]) <= float(verdict["tolerance"])  # type: ignore[arg-type]


def test_null_verdict_fails_when_surrogate_is_shifted() -> None:
    generator = rng(7)
    held_out = generator.normal(1.0, 0.3, size=13)
    surrogate = generator.normal(3.0, 0.3, size=500)  # materially different from the class
    null, test = held_out_null_and_test(surrogate, held_out, resamples=300, seed=1)
    verdict = null_verdict(null, test)
    assert verdict["passed"] is False
    assert float(verdict["test_median"]) > float(verdict["tolerance"])  # type: ignore[arg-type]


def test_realism_judgment_passes_when_overlapping_and_close() -> None:
    overlaps = [True, True, True, True]
    passed = realism_judgment(overlaps, [1, 2], rejector_distance=0.1, tolerance=0.35)
    assert passed["passed"] is True
    assert passed["score"] == 5
    failed = realism_judgment(
        [True, False, False, True], [1, 2], rejector_distance=0.9, tolerance=0.35
    )
    assert failed["passed"] is False
    assert failed["score"] == 1


def test_channel_responses_separate_tonal_from_noise() -> None:
    sample_rate, n = 2048.0, 4096
    times = np.arange(n) / sample_rate
    tonal = np.sin(2.0 * np.pi * 60.0 * times) + 0.1 * rng(1).standard_normal(n)
    noise = 0.5 * rng(2).standard_normal(n)
    samples = np.stack([tonal, noise], axis=0)
    conf, rej = channel_responses(samples, sample_rate, 256, 128, (4.0, 300.0), 3.0)
    # The tonal channel is more confident and more line-prominent than the noise channel.
    assert conf[0] > conf[1]
    assert rej[0] > rej[1]


def _manifest() -> Manifest:
    return Manifest.model_validate(json.loads(MANIFEST_PATH.read_text(encoding="utf-8")))


def test_e1_harness_runs_end_to_end_on_fixtures(tmp_path: Path) -> None:
    runner = Runner(tmp_path)
    try:
        result = runner.run(_manifest())
        report_hash = result.node_outputs["e1"]["e1_report"]
        report = E1Report.from_blob(runner._store.get(report_hash)).payload
    finally:
        runner.close()
    # The comparison machinery produced both response distributions across the ladder.
    ladder = report["snr_ladder_db"]
    assert len(report["detectability"]["surrogate"]["rate"]) == len(ladder)
    assert len(report["detectability"]["held_out"]["rate"]) == len(ladder)
    assert len(report["detectability"]["overlap_per_rung"]) == len(ladder)
    # A rejector two-sample distance and a realism verdict are rendered (pass or fail).
    assert 0.0 <= report["rejector_response"]["two_sample_ks"] <= 1.0
    assert report["realism"]["score"] in (1, 3, 5)
    assert isinstance(report["realism"]["passed"], bool)
    # The held-out arm is the placeholder, and no owner-certified value appears.
    assert report["held_out_source"] == "placeholder_synthetic"
    assert report["owner_certified"] == {"detection_range_r": None, "confirmer_pd": None}


def test_injector_path_pulls_no_torch_or_network(tmp_path: Path) -> None:
    # AC6: the injector and E1 harness add no network dependency to the offline path, and PyTorch
    # stays confined to the training extras. Run in a fresh interpreter so earlier tests importing
    # boto3 or soundfile do not pollute the check through the shared module table.
    import subprocess
    import sys
    import textwrap

    script = textwrap.dedent(
        f"""
        import json, sys
        from pathlib import Path
        from fathom.manifest import Manifest
        from fathom.runner import Runner
        manifest = Manifest.model_validate(json.loads(Path({str(MANIFEST_PATH)!r}).read_text()))
        runner = Runner(Path({str(tmp_path / "state")!r}))
        try:
            runner.run(manifest)
        finally:
            runner.close()
        for module in ("torch", "boto3", "soundfile"):
            assert module not in sys.modules, module
        print("OK")
        """
    )
    result = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, check=False
    )
    assert result.returncode == 0, result.stderr
    assert "OK" in result.stdout


def test_e1_report_regenerates_byte_identically(tmp_path: Path) -> None:
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
    assert repro.mismatches == ()
