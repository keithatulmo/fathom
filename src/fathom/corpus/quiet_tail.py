"""Selecting which recordings to fetch to give the quiet-tail cohort audio.

The window pre-correlation flags quiet-tail passages from AIS kinematics against recordings whose
audio was never downloaded. To make the cohort real, the audio for those passages has to be
fetched, but not every candidate file is worth the bytes: a single recording often carries several
distinct quiet-tail vessels, and the aim is to back at least the CA2 minimum of distinct held-out
and train-side vessels. This module selects a near-minimal set of files by greedily taking the file
that adds the most not-yet-covered distinct vessels, so a bounded download covers the target rather
than fetching every flagged window. The selection is pure and deterministic, so it is reviewable and
testable apart from the fetch itself.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class QuietTailFile:
    """A candidate recording to fetch: its source, origin, site, rate, and the vessels it covers.

    ``min_range_m`` is the closest-approach range of the closest quiet-tail passage in the file,
    used to prioritise files whose passages are near enough to be audible in the working band.
    """

    source_id: str
    origin_url: str
    site: str
    sample_rate_hz: float | None
    vessels: frozenset[str]
    min_range_m: float = 0.0


def select_files(
    candidates: list[QuietTailFile], target_vessels: int, *, max_files: int | None = None
) -> list[QuietTailFile]:
    """Greedily choose files covering the most distinct quiet-tail vessels up to a target.

    Each step takes the candidate adding the most vessels not already covered, stopping when the
    target distinct-vessel count is reached, when no remaining file adds a new vessel, or when an
    optional file-count budget is hit. Ties break on origin URL so the choice is deterministic.
    """
    covered: set[str] = set()
    chosen: list[QuietTailFile] = []
    pool = list(candidates)
    while pool and len(covered) < target_vessels:
        best = max(pool, key=lambda f: (len(f.vessels - covered), f.origin_url))
        gain = len(best.vessels - covered)
        if gain == 0:
            break
        chosen.append(best)
        covered |= best.vessels
        pool.remove(best)
        if max_files is not None and len(chosen) >= max_files:
            break
    return chosen


def select_files_by_closeness(
    candidates: list[QuietTailFile],
    target_vessels: int,
    *,
    already_covered: frozenset[str] = frozenset(),
    max_files: int | None = None,
) -> list[QuietTailFile]:
    """Choose files with the closest passages first, covering new distinct vessels up to a target.

    Audibility falls off with range, so to add audio-backed vessels efficiently the files are taken
    in order of their closest quiet-tail passage, each one contributing the vessels it covers that
    are not already accounted for. Vessels in ``already_covered`` (for example those already
    audio-backed) do not count toward the target, so a top-up fetch does not re-fetch them.
    """
    covered = set(already_covered)
    chosen: list[QuietTailFile] = []
    fresh = target_vessels
    for candidate in sorted(candidates, key=lambda f: (f.min_range_m, f.origin_url)):
        new = candidate.vessels - covered
        if not new:
            continue
        chosen.append(candidate)
        covered |= candidate.vessels
        fresh -= len(new)
        if fresh <= 0 or (max_files is not None and len(chosen) >= max_files):
            break
    return chosen


def covered_vessels(files: list[QuietTailFile]) -> set[str]:
    """Return the union of distinct vessels covered by a set of files."""
    out: set[str] = set()
    for f in files:
        out |= f.vessels
    return out
