"""Tests for physics-parametric line placement and harmonic families."""

from __future__ import annotations

import numpy as np

from fathom.surrogate.lines import MachineryParams, build_lines


def _params(**over: object) -> MachineryParams:
    base = dict(
        shaft_hz=4.0,
        shaft_harmonics=3,
        blade_count=5,
        blade_harmonics=2,
        electrical_hz=60.0,
        electrical_harmonics=2,
        aux_hz=(35.0, 120.0),
        shaft_amp=1.0,
        blade_amp=0.8,
        electrical_amp=0.5,
        aux_amp=0.3,
        harmonic_rolloff=0.5,
        line_width_hz=0.2,
        wander_hz=0.05,
    )
    base.update(over)
    return MachineryParams(**base)  # type: ignore[arg-type]


def test_lines_fall_at_kinematic_frequencies() -> None:
    lines = build_lines(_params())
    by_family = {
        f: lines.freqs[[i for i, fam in enumerate(lines.families) if fam == f]]
        for f in set(lines.families)
    }
    # Shaft fundamental and harmonics at k * shaft_hz.
    assert np.allclose(sorted(by_family["shaft"]), [4.0, 8.0, 12.0])
    # Blade-rate line at shaft_hz * blade_count = 20 Hz, plus its second harmonic.
    assert np.allclose(sorted(by_family["blade"]), [20.0, 40.0])
    # Electrical line at the supply frequency and its harmonic.
    assert np.allclose(sorted(by_family["electrical"]), [60.0, 120.0])
    # Auxiliary lines placed individually.
    assert np.allclose(sorted(by_family["auxiliary"]), [35.0, 120.0])


def test_harmonic_amplitudes_roll_off() -> None:
    lines = build_lines(_params(shaft_harmonics=3, harmonic_rolloff=0.5, shaft_amp=1.0))
    idx = [i for i, fam in enumerate(lines.families) if fam == "shaft"]
    order = np.argsort(lines.freqs[idx])
    amps = lines.amplitudes[idx][order]
    # Fundamental 1.0, then 0.5, then 0.25 under a 0.5 roll-off.
    assert np.allclose(amps, [1.0, 0.5, 0.25])


def test_quiet_mode_drops_propulsion_and_keeps_auxiliary() -> None:
    lines = build_lines(_params(quiet_mode=True, propulsion_absent=True))
    assert "shaft" not in lines.families
    assert "blade" not in lines.families
    # The auxiliary lines carry the signature in the quiet-target case.
    assert "auxiliary" in lines.families
    assert "electrical" in lines.families


def test_line_arrays_are_consistent_length() -> None:
    lines = build_lines(_params())
    n = lines.freqs.shape[0]
    assert lines.amplitudes.shape[0] == n
    assert lines.widths.shape[0] == n
    assert lines.wanders.shape[0] == n
    assert len(lines.families) == n
