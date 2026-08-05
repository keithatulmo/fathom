"""Tests for the E3 detection bake-off: scoring, selection, the flip clause, and the verifier."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np

from fathom.config import E3Config
from fathom.e3 import run_e3_bakeoff
from fathom.surrogate.fit import SurrogateDistributions

_VERIFIER = Path(__file__).resolve().parents[1] / "scripts" / "verify_e3.py"
_RATE = 1000.0


def _red(n: int, seed: int, slope: float, scale: float) -> np.ndarray:
    gen = np.random.default_rng(seed)
    spec = np.fft.rfft(gen.standard_normal(n))
    freqs = np.fft.rfftfreq(n, 1.0 / _RATE)
    freqs[0] = freqs[1]
    spec *= (freqs / freqs[len(freqs) // 2]) ** (slope / 2.0)
    return np.ascontiguousarray(scale * np.fft.irfft(spec, n=n), dtype=np.float64)


def _config() -> E3Config:
    return E3Config(
        snr_ladder_db=(0.0, 6.0, 12.0),
        eval_backgrounds_per_regime=2,
        train_backgrounds_per_regime=1,
        max_subwindows_per_background=8,
    )


def _backgrounds() -> tuple[
    dict[str, list[np.ndarray]], dict[str, list[np.ndarray]], dict[str, list[str]]
]:
    n = int(_RATE * 64)
    train = {
        "quiet": [_red(n, 1, -1.4, 0.3)],
        "nominal": [_red(n, 11, -1.1, 0.4)],
        "busy": [_red(n, 21, -0.8, 0.6)],
    }
    ev = {
        "quiet": [_red(n, 100 + i, -1.4, 0.3) for i in range(2)],
        "nominal": [_red(n, 110 + i, -1.1, 0.4) for i in range(2)],
        "busy": [_red(n, 120 + i, -0.8, 0.6) for i in range(2)],
    }
    sites = {"quiet": ["adeon"], "nominal": ["mbari"], "busy": ["onc"]}
    return train, ev, sites


def _distributions() -> SurrogateDistributions:
    lineage = json.loads(Path("docs/e1_realdata_lineage.json").read_text())
    return SurrogateDistributions.from_dict(lineage["fitted_distributions"])


def test_bakeoff_produces_three_candidates_and_a_selection() -> None:
    train, ev, sites = _backgrounds()
    payload = run_e3_bakeoff(train, ev, sites, _distributions(), _config(), seed=5)
    assert {c["name"] for c in payload["candidates"]} == {"cfar", "integration", "learned"}
    for candidate in payload["candidates"]:
        assert set(candidate) >= {
            "operating_point_sensitivity",
            "floor_4hz_sensitivity",
            "sensitivity_curve",
            "per_site_far",
            "false_alarm_stable",
            "latency_s",
            "implementation_risk",
            "soft_score",
        }
        assert candidate["soft_score"]["graded"] is True
    # A selection is made, or the no-stable-candidate flip fires; both are well-formed outcomes, and
    # the SD5-close verdict tracks whether a detector was selected. The real run makes the call.
    assert payload["selected"] is not None or (
        payload["flip_clause"]["branch"] == "no_stable_candidate"
    )
    assert payload["verdict"]["closes_sd5_on_measurement"] == (payload["selected"] is not None)
    assert payload["owner_certified"] == {"detection_range_r": None, "confirmer_pd": None}
    assert "train-side backgrounds only" in payload["leak_rule"]


def test_sensitivity_rises_with_snr() -> None:
    # The bake-off measures which detector wins, not a fixed ordering; the guaranteed property
    # a detector's sensitivity curve does not fall as the injected level rises. Check the learned
    # detector, whose curve is smooth enough to test on the small synthetic sample.
    train, ev, sites = _backgrounds()
    payload = run_e3_bakeoff(train, ev, sites, _distributions(), _config(), seed=5)
    learned = next(c for c in payload["candidates"] if c["name"] == "learned")
    regimes = list(learned["sensitivity_curve"])
    low = float(np.mean([learned["sensitivity_curve"][r]["0"] for r in regimes]))
    high = float(np.mean([learned["sensitivity_curve"][r]["12"] for r in regimes]))
    assert high >= low


def test_committed_lineage_passes_the_stdlib_verifier(tmp_path: Path) -> None:
    train, ev, sites = _backgrounds()
    payload = run_e3_bakeoff(train, ev, sites, _distributions(), _config(), seed=5)
    payload["run_id"] = "e3-detection-test"
    payload["seed"] = 5
    lineage = tmp_path / "e3_detection_lineage.json"
    lineage.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    result = subprocess.run(
        [sys.executable, str(_VERIFIER), str(lineage)], capture_output=True, text=True
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "AUDIT OK" in result.stdout
