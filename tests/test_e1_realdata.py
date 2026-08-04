"""Tests for the E1 real-data run over synthetic segments (no ledger or R2)."""

from __future__ import annotations

import math

import numpy as np
import pytest

from fathom.config import E1RealDataConfig
from fathom.determinism import rng
from fathom.e1_realdata import run_e1_realdata
from fathom.surrogate.fit import ForbiddenSplitError

_RATE = 1000.0


def _tonal(
    seed: int, duration_s: float = 4.0, lines: tuple[float, ...] = (20.0, 40.0, 60.0)
) -> np.ndarray:
    n = int(_RATE * duration_s)
    t = np.arange(n) / _RATE
    gen = rng(seed)
    signal = 0.05 * gen.standard_normal(n)
    for i, freq in enumerate(lines):
        signal = signal + (0.6**i) * np.sin(2.0 * math.pi * freq * t + gen.uniform(0, 2 * math.pi))
    return np.ascontiguousarray(signal, dtype=np.float64)


def _segment(vessel_id: str, split: str, seed: int) -> dict[str, object]:
    return {"vessel_id": vessel_id, "split": split, "sample_rate": _RATE, "samples": _tonal(seed)}


def _config() -> E1RealDataConfig:
    return E1RealDataConfig(
        snr_ladder_db=(0.0, 6.0, 12.0),
        analysis_rate_hz=_RATE,
        surrogate_draws=12,
        bootstrap_resamples=60,
        null_resamples=60,
    )


def test_realdata_run_produces_per_rung_verdict() -> None:
    train = [_segment(f"T{i}", "train", seed=i) for i in range(6)]
    held = [_segment(f"H{i}", "test", seed=100 + i) for i in range(6)]
    backgrounds = [0.3 * rng(200 + i).standard_normal(4000) for i in range(6)]
    payload, dist = run_e1_realdata(train, held, backgrounds, _config(), seed=1)

    assert dist.provenance["amplitude_statistics"] == "train_audio"
    assert len(payload["per_rung"]) == 3
    for rung in payload["per_rung"]:
        assert {
            "snr_db",
            "surrogate",
            "held_out",
            "detectability_overlap",
            "wasserstein",
            "ks",
            "null_tolerance",
            "test_median",
            "rejector_inside_null",
        } <= set(rung)
        assert rung["wasserstein"] >= 0.0
        assert isinstance(rung["rejector_inside_null"], bool)
    assert payload["distance"] == "wasserstein"
    assert payload["tolerance_method"] == "held_out_null_distribution"
    assert payload["held_out_response"]["processed"] == "as_is"
    assert isinstance(payload["verdict"]["passed"], bool)
    assert payload["owner_certified"] == {"detection_range_r": None, "confirmer_pd": None}


def test_realdata_leak_guard_refuses_held_out_in_training() -> None:
    # A held-out segment mixed into the training set must raise, never contribute to the fit.
    train = [_segment("T0", "train", seed=0), _segment("H0", "test", seed=1)]
    held = [_segment(f"H{i}", "test", seed=100 + i) for i in range(3)]
    backgrounds = [0.3 * rng(200 + i).standard_normal(4000) for i in range(3)]
    with pytest.raises(ForbiddenSplitError):
        run_e1_realdata(train, held, backgrounds, _config(), seed=1)
