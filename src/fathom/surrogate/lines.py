"""Narrowband line families from machinery kinematics.

SD2 Section 4 Option A places the surrogate's tonal set from first-principles machinery structure: a
shaft-rate fundamental and its harmonics, a blade-rate line at the shaft rate times the blade count
with harmonics, an electrical line at the supply frequency with harmonics, and a small set of
auxiliary-machinery lines. Higher harmonics roll off, which is the first-principles structure a
physics model supplies. The quiet-target case is the hard one the workflow matrix names: the
propulsion lines (shaft and blade) may be absent, and the auxiliary lines carry the signature at low
level. Every amplitude here is dimensionless and relative; the absolute injected level is never set
in this module but by the signal-to-noise leveling of :mod:`fathom.surrogate.level`, so no absolute
source level is expressible here, per SD2 Section 2.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import numpy.typing as npt


@dataclass(frozen=True)
class MachineryParams:
    """The machinery structure of one surrogate target, all frequencies in hertz.

    Amplitudes are dimensionless family weights; the harmonic roll-off multiplies each successive
    harmonic's amplitude, so a value below one makes higher harmonics weaker as real machinery does.
    In the quiet-target mode the propulsion lines are optionally dropped and their weight is set
    low, leaving the auxiliary lines to carry the signature.
    """

    shaft_hz: float
    shaft_harmonics: int
    blade_count: int
    blade_harmonics: int
    electrical_hz: float
    electrical_harmonics: int
    aux_hz: tuple[float, ...]
    shaft_amp: float
    blade_amp: float
    electrical_amp: float
    aux_amp: float
    harmonic_rolloff: float
    line_width_hz: float
    wander_hz: float
    quiet_mode: bool = False
    propulsion_absent: bool = False

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable record of the parameters for the lineage ledger."""
        record = asdict(self)
        record["aux_hz"] = list(self.aux_hz)
        return record


@dataclass(frozen=True)
class LineSet:
    """A constructed set of narrowband lines: frequency, amplitude, width, wander, and family."""

    freqs: npt.NDArray[np.float64]
    amplitudes: npt.NDArray[np.float64]
    widths: npt.NDArray[np.float64]
    wanders: npt.NDArray[np.float64]
    families: tuple[str, ...]

    def __post_init__(self) -> None:
        n = self.freqs.shape[0]
        if not (
            self.amplitudes.shape[0]
            == self.widths.shape[0]
            == self.wanders.shape[0]
            == len(self.families)
            == n
        ):
            raise ValueError("line arrays and family labels must all have the same length")


def _harmonic_family(
    fundamental_hz: float, count: int, amplitude: float, rolloff: float, family: str
) -> list[tuple[float, float, str]]:
    """Return a fundamental and its harmonics as (frequency, amplitude, family) triples.

    The k-th harmonic sits at k times the fundamental and carries the family amplitude scaled by the
    roll-off raised to k minus one, so the fundamental is strongest and higher harmonics fall off.
    """
    if count <= 0 or amplitude <= 0.0 or fundamental_hz <= 0.0:
        return []
    return [
        (fundamental_hz * k, amplitude * rolloff ** (k - 1), family)
        for k in range(1, count + 1)
    ]


def build_lines(params: MachineryParams) -> LineSet:
    """Construct the surrogate's line set from its machinery parameters.

    The propulsion lines are the shaft-rate family and the blade-rate family at the shaft rate times
    the blade count; the electrical family sits at the supply frequency; the auxiliary lines are
    placed individually. In the quiet-target mode the propulsion families are dropped when
    ``propulsion_absent`` is set, and the auxiliary lines carry the signature.
    """
    triples: list[tuple[float, float, str]] = []

    include_propulsion = not (params.quiet_mode and params.propulsion_absent)
    if include_propulsion:
        triples += _harmonic_family(
            params.shaft_hz, params.shaft_harmonics, params.shaft_amp, params.harmonic_rolloff,
            "shaft",
        )
        blade_hz = params.shaft_hz * params.blade_count
        triples += _harmonic_family(
            blade_hz, params.blade_harmonics, params.blade_amp, params.harmonic_rolloff, "blade",
        )
    triples += _harmonic_family(
        params.electrical_hz, params.electrical_harmonics, params.electrical_amp,
        params.harmonic_rolloff, "electrical",
    )
    for aux in params.aux_hz:
        if aux > 0.0 and params.aux_amp > 0.0:
            triples.append((aux, params.aux_amp, "auxiliary"))

    if not triples:
        raise ValueError("machinery parameters produced no lines")

    freqs = np.array([t[0] for t in triples], dtype=np.float64)
    amplitudes = np.array([t[1] for t in triples], dtype=np.float64)
    families = tuple(t[2] for t in triples)
    widths = np.full(freqs.shape, float(params.line_width_hz), dtype=np.float64)
    wanders = np.full(freqs.shape, float(params.wander_hz), dtype=np.float64)
    return LineSet(freqs=freqs, amplitudes=amplitudes, widths=widths, wanders=wanders,
                   families=families)
