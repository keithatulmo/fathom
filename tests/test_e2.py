"""Tests for the E2 front-end sweep: scoring, selection, the flip clause, and the verifier."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np

from fathom.config import E2Config
from fathom.e2 import build_grid, recovered_prominence, run_e2_sweep
from fathom.frontend import FrontEndParams, lofar_gram
from fathom.normalize import normalize_gram
from fathom.surrogate.fit import SurrogateDistributions

_RATE = 1000.0
_VERIFIER = Path(__file__).resolve().parents[1] / "scripts" / "verify_e2.py"


def _small_config() -> E2Config:
    return E2Config(
        window_lengths_s=(2.0, 4.0),
        overlaps=(0.5,),
        integration_counts=(1,),
        tapers=("hann", "multitaper"),
        normalizers=("split_window", "temporal_median"),
        backgrounds_per_regime=2,
        injection_draws=2,
    )


def _red(n: int, seed: int, slope: float, scale: float) -> np.ndarray:
    gen = np.random.default_rng(seed)
    spec = np.fft.rfft(gen.standard_normal(n))
    freqs = np.fft.rfftfreq(n, 1.0 / _RATE)
    freqs[0] = freqs[1]
    spec *= (freqs / freqs[len(freqs) // 2]) ** (slope / 2.0)
    return np.ascontiguousarray(scale * np.fft.irfft(spec, n=n), dtype=np.float64)


def _backgrounds() -> tuple[dict[str, list[np.ndarray]], dict[str, list[str]]]:
    n = int(_RATE * 20)
    bgs = {
        "quiet": [_red(n, i, -1.4, 0.3) for i in range(2)],
        "nominal": [_red(n, 10 + i, -1.1, 0.4) for i in range(2)],
        "busy": [_red(n, 20 + i, -0.8, 0.6) for i in range(2)],
    }
    sites = {"quiet": ["adeon"], "nominal": ["mbari"], "busy": ["onc"]}
    return bgs, sites


def _distributions() -> SurrogateDistributions:
    return SurrogateDistributions(
        quiet_fraction=0.3, train_vessel_count=5, train_vessel_ids=tuple("ABCDE")
    )


def test_build_grid_is_the_axis_product() -> None:
    config = _small_config()
    assert len(build_grid(config)) == 2 * 1 * 1 * 2 * 2


def test_recovered_prominence_lifts_a_known_comb() -> None:
    t = np.arange(int(_RATE * 20)) / _RATE
    comb = sum(0.5 * np.sin(2.0 * np.pi * f * t) for f in (8.0, 16.0, 24.0))
    signal = np.ascontiguousarray(0.2 * np.random.default_rng(1).standard_normal(t.shape[0]) + comb)
    params = FrontEndParams(4.0, 0.5, 1, "hann", 5, 3.0, "split_window", 6.0, 1.0, 3, 8)
    gram, freqs = lofar_gram(signal, _RATE, params)
    normalized, nf = normalize_gram(gram, freqs, (4.0, 150.0), params)
    value = recovered_prominence(normalized, nf, np.array([8.0, 16.0, 24.0]), (4.0, 150.0))
    assert value > 6.0  # a clean comb is recovered well above the floor


def test_sweep_selects_holds_flatness_and_asserts_no_absolute_level() -> None:
    bgs, sites = _backgrounds()
    payload = run_e2_sweep(bgs, sites, _distributions(), _small_config(), seed=7)
    assert len(payload["per_config"]) == 8
    assert payload["owner_certified"] == {"detection_range_r": None, "confirmer_pd": None}
    # The operating point is one value across sites, never per site.
    assert "operating_snr_db" in payload
    for config in payload["per_config"]:
        assert config["constraints"]["determinism_cleared"] is True
        assert set(config["separability"]["per_regime_db"]) == {"quiet", "nominal", "busy"}
    # A selection (or a fired flip clause) is always rendered.
    assert payload["selected"] is not None or payload["flip_clause"]["fired"]
    assert "reference_baseline" in payload and "band_endpoint_finding" in payload


def test_closed_front_end_is_the_split_window_config() -> None:
    from fathom.frontend import closed_front_end

    params = closed_front_end()
    # The SD4-closed default is the split-window config the operating-point objective picks.
    assert params.key() == "res0.5Hz_ov0.75_int1_hann_split_window_hw6_g1_t3"
    assert params.normalizer == "split_window"
    assert params.taper == "hann"
    assert abs(1.0 / params.window_length_s - 0.5) < 1e-9


def test_operating_point_objective_prefers_the_operating_site() -> None:
    from fathom.e2 import _select_and_flip, operating_separability

    def row(key: str, quiet: float, nominal: float, busy: float) -> dict[str, Any]:
        avg = float(np.median([quiet, nominal, busy]))
        return {
            "key": key,
            "params": {"taper": "hann"},
            "candidate": True,
            "separability": {
                "separability_db": avg,
                "per_regime_db": {"quiet": quiet, "nominal": nominal, "busy": busy},
            },
            "flatness": {"flatness_holds": True, "flatness_score": 0.9},
            "constraints": {"compute_cleared": True, "determinism_cleared": True},
        }

    # A wins the site-average by manufacturing its lead at the busy site; B wins the quiet operating
    # point. The corrected objective must select B, not the site-averaged argmax A.
    site_avg_winner = row("A_siteavg", quiet=9.9, nominal=10.5, busy=11.9)
    operating_winner = row("B_quiet", quiet=10.3, nominal=10.2, busy=10.1)
    config = E2Config(operating_point_regime="quiet")
    selected, flip = _select_and_flip([site_avg_winner, operating_winner], config)
    assert selected is not None and selected["key"] == "B_quiet"
    assert flip["fired"] is False
    assert operating_separability(site_avg_winner, "quiet") == 9.9


def test_committed_lineage_passes_the_stdlib_verifier(tmp_path: Path) -> None:
    bgs, sites = _backgrounds()
    payload = run_e2_sweep(bgs, sites, _distributions(), _small_config(), seed=7)
    payload["run_id"] = "e2-frontend-test"
    payload["seed"] = 7
    lineage = tmp_path / "e2_front_end_lineage.json"
    lineage.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    result = subprocess.run(
        [sys.executable, str(_VERIFIER), str(lineage)], capture_output=True, text=True
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "AUDIT OK" in result.stdout
