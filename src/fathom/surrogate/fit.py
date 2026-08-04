"""Train-only parameter fitting with a mechanical injection-leak guard.

SD3 Section 5.4 and WO-3 deliverable 5 bind the surrogate hard: line-statistic distributions derive
from train-side vessels and first-principles structure only, and the held-out quietest real vessel
class that E1 measures against is never consulted in surrogate design, tuning, or level-setting.
This module enforces that by construction. The core fitting routine refuses any vessel not in the
train split by raising :class:`ForbiddenSplitError`, so no code path can let a calibration, test, or
held-out vessel reach surrogate construction; a leak is a crash, not a silent contamination.

At proof scope the structural ranges are first-principles machinery priors and the quiet-target
fraction is read from the train roster, because fitting the amplitude and roll-off statistics to
real train-side audio is the refinement the SD2 Section 8 flip condition and the E1 real-data run
owns.
What is fit from the train side here is which vessels contribute and how often the quiet-target case
is drawn; the held-out class contributes nothing, which is the property that matters. All quantities
are dimensionless or in hertz, and no absolute level is expressible.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import TYPE_CHECKING, Any

from ..determinism import rng
from .lines import MachineryParams

if TYPE_CHECKING:
    from .broadband import BroadbandParams

TRAIN_SPLIT = "train"

# First-principles machinery priors, dimensionless or in hertz, documented as priors rather than fit
# to real audio at proof scope. Shaft rates are a few hertz, blade counts small integers, supply
# frequencies the two standards, auxiliary lines in the tens to low hundreds of hertz.
_SHAFT_HZ_RANGE = (2.0, 8.0)
_BLADE_COUNT_CHOICES = (3, 4, 5, 6, 7)
_ELECTRICAL_HZ_CHOICES = (50.0, 60.0)
_AUX_HZ_RANGE = (20.0, 150.0)
_AUX_COUNT_RANGE = (1, 4)
_SHAFT_AMP_RANGE = (0.6, 1.0)
_BLADE_AMP_RANGE = (0.4, 0.9)
_ELECTRICAL_AMP_RANGE = (0.2, 0.6)
_AUX_AMP_RANGE = (0.1, 0.4)
_ROLLOFF_RANGE = (0.4, 0.8)
_WIDTH_HZ_RANGE = (0.1, 0.5)
_WANDER_HZ_RANGE = (0.02, 0.2)
# First-principles broadband continuum priors, refined from train audio by the WO-5 fit. The
# continuum is red (its exponent negative), and the line-to-broadband ratio spans the
# broadband-dominated quiet extreme to a moderately line-dominated case; both are dimensionless.
_CONTINUUM_EXPONENT_RANGE = (-2.0, -0.5)
_LBR_DB_RANGE = (0.0, 10.0)
_SHAFT_HARMONICS = 4
_BLADE_HARMONICS = 3
_ELECTRICAL_HARMONICS = 2
# In the quiet-target mode this fraction of quiet draws also drops the propulsion lines entirely,
# the hard case where the auxiliary lines alone carry the signature.
_PROPULSION_ABSENT_FRACTION = 0.5
_DEFAULT_QUIET_FRACTION = 0.5


class ForbiddenSplitError(RuntimeError):
    """Raised when a vessel outside the train split reaches surrogate parameter fitting."""


@dataclass(frozen=True)
class SurrogateDistributions:
    """Fitted line-statistic distributions and the train roster they derive from.

    The structural ranges are first-principles priors; ``quiet_fraction`` and the roster are read
    from the train side. The roster and the count make the train-side-only derivation auditable.
    """

    quiet_fraction: float
    train_vessel_count: int
    train_vessel_ids: tuple[str, ...]
    shaft_hz_range: tuple[float, float] = _SHAFT_HZ_RANGE
    blade_count_choices: tuple[int, ...] = _BLADE_COUNT_CHOICES
    electrical_hz_choices: tuple[float, ...] = _ELECTRICAL_HZ_CHOICES
    aux_hz_range: tuple[float, float] = _AUX_HZ_RANGE
    aux_count_range: tuple[int, int] = _AUX_COUNT_RANGE
    shaft_amp_range: tuple[float, float] = _SHAFT_AMP_RANGE
    blade_amp_range: tuple[float, float] = _BLADE_AMP_RANGE
    electrical_amp_range: tuple[float, float] = _ELECTRICAL_AMP_RANGE
    aux_amp_range: tuple[float, float] = _AUX_AMP_RANGE
    rolloff_range: tuple[float, float] = _ROLLOFF_RANGE
    width_hz_range: tuple[float, float] = _WIDTH_HZ_RANGE
    wander_hz_range: tuple[float, float] = _WANDER_HZ_RANGE
    shaft_harmonics: int = _SHAFT_HARMONICS
    blade_harmonics: int = _BLADE_HARMONICS
    electrical_harmonics: int = _ELECTRICAL_HARMONICS
    propulsion_absent_fraction: float = _PROPULSION_ABSENT_FRACTION
    continuum_exponent_range: tuple[float, float] = _CONTINUUM_EXPONENT_RANGE
    lbr_db_range: tuple[float, float] = _LBR_DB_RANGE
    provenance: dict[str, Any] = field(
        default_factory=lambda: {"structural_ranges": "first_principles"}
    )

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable record for the artifact and the lineage ledger."""
        record = asdict(self)
        record["train_vessel_ids"] = list(self.train_vessel_ids)
        record["blade_count_choices"] = list(self.blade_count_choices)
        record["electrical_hz_choices"] = list(self.electrical_hz_choices)
        return record

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SurrogateDistributions:
        """Reconstruct distributions from a serialised record, restoring tuple types."""
        kwargs = dict(data)
        kwargs["train_vessel_ids"] = tuple(kwargs["train_vessel_ids"])
        for key in (
            "blade_count_choices",
            "electrical_hz_choices",
            "shaft_hz_range",
            "aux_hz_range",
            "aux_count_range",
            "shaft_amp_range",
            "blade_amp_range",
            "electrical_amp_range",
            "aux_amp_range",
            "rolloff_range",
            "width_hz_range",
            "wander_hz_range",
            "continuum_exponent_range",
            "lbr_db_range",
        ):
            if key in kwargs:
                kwargs[key] = tuple(kwargs[key])
        return cls(**kwargs)


def _assert_train_only(vessels: list[dict[str, Any]]) -> None:
    """Raise :class:`ForbiddenSplitError` if any vessel is outside the train split."""
    for vessel in vessels:
        split = str(vessel.get("split"))
        if split != TRAIN_SPLIT:
            raise ForbiddenSplitError(
                f"vessel {vessel.get('vessel_id')!r} is in split {split!r}, not {TRAIN_SPLIT!r}; "
                "surrogate parameters derive from train-side vessels only (SD3 Section 5.4)"
            )


def select_train_vessels(vessels: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return the train-split, non-quarantined vessels, the only ones fitting may read."""
    return [
        v
        for v in vessels
        if str(v.get("split")) == TRAIN_SPLIT and not bool(v.get("quarantined", False))
    ]


def fit_distributions(train_vessels: list[dict[str, Any]]) -> SurrogateDistributions:
    """Fit the surrogate distributions from train-side vessels, refusing any other split.

    The guard runs first: a vessel outside the train split raises rather than being read, so no
    held-out or evaluation vessel can reach surrogate construction. The quiet-target fraction is the
    fraction of the train roster flagged quiet-tail where that flag is present, else a first-
    principles default; the structural ranges are first-principles priors recorded as such.
    """
    _assert_train_only(train_vessels)
    if not train_vessels:
        raise ValueError("no train-side vessels to fit from")
    flags = [bool(v.get("is_quiet_tail")) for v in train_vessels if "is_quiet_tail" in v]
    quiet_fraction = (sum(flags) / len(flags)) if flags else _DEFAULT_QUIET_FRACTION
    roster = tuple(sorted(str(v["vessel_id"]) for v in train_vessels))
    return SurrogateDistributions(
        quiet_fraction=quiet_fraction,
        train_vessel_count=len(train_vessels),
        train_vessel_ids=roster,
    )


def draw_machinery(distributions: SurrogateDistributions, seed: int) -> MachineryParams:
    """Draw one machinery parameter set from the fitted distributions, deterministically."""
    generator = rng(seed)
    quiet = bool(generator.random() < distributions.quiet_fraction)
    propulsion_absent = quiet and bool(
        generator.random() < distributions.propulsion_absent_fraction
    )
    aux_count = int(
        generator.integers(distributions.aux_count_range[0], distributions.aux_count_range[1] + 1)
    )
    aux_hz = tuple(
        sorted(float(generator.uniform(*distributions.aux_hz_range)) for _ in range(aux_count))
    )
    return MachineryParams(
        shaft_hz=float(generator.uniform(*distributions.shaft_hz_range)),
        shaft_harmonics=distributions.shaft_harmonics,
        blade_count=int(generator.choice(distributions.blade_count_choices)),
        blade_harmonics=distributions.blade_harmonics,
        electrical_hz=float(generator.choice(distributions.electrical_hz_choices)),
        electrical_harmonics=distributions.electrical_harmonics,
        aux_hz=aux_hz,
        shaft_amp=float(generator.uniform(*distributions.shaft_amp_range)),
        blade_amp=float(generator.uniform(*distributions.blade_amp_range)),
        electrical_amp=float(generator.uniform(*distributions.electrical_amp_range)),
        aux_amp=float(generator.uniform(*distributions.aux_amp_range)),
        harmonic_rolloff=float(generator.uniform(*distributions.rolloff_range)),
        line_width_hz=float(generator.uniform(*distributions.width_hz_range)),
        wander_hz=float(generator.uniform(*distributions.wander_hz_range)),
        quiet_mode=quiet,
        propulsion_absent=propulsion_absent,
    )


def draw_broadband(distributions: SurrogateDistributions, seed: int) -> BroadbandParams:
    """Draw one broadband continuum parameter set from the fitted distributions, deterministically.

    The continuum exponent and the line-to-broadband ratio are drawn independently of the machinery
    draw and of the quiet-target flag, so the ratio spans the full range the train cohort shows,
    from the broadband-dominated quiet extreme to the line-dominated case, rather than sitting at a
    single clean point. The continuum is drawn for every target, quiet or not, so a quiet surrogate
    is broadband plus auxiliary lines rather than the near-silent line-only object it was.
    """
    from .broadband import BroadbandParams

    generator = rng(seed)
    return BroadbandParams(
        continuum_exponent=float(generator.uniform(*distributions.continuum_exponent_range)),
        lbr_db=float(generator.uniform(*distributions.lbr_db_range)),
    )
