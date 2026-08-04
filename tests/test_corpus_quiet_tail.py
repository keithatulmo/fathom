"""Tests for greedy quiet-tail file selection."""

from __future__ import annotations

from fathom.corpus.quiet_tail import (
    QuietTailFile,
    covered_vessels,
    select_files,
    select_files_by_closeness,
)


def _f(url: str, vessels: set[str], min_range_m: float = 0.0) -> QuietTailFile:
    return QuietTailFile(
        source_id="s",
        origin_url=url,
        site="site",
        sample_rate_hz=48000.0,
        vessels=frozenset(vessels),
        min_range_m=min_range_m,
    )


def test_greedy_covers_target_with_fewest_files() -> None:
    files = [
        _f("a", {"V1", "V2", "V3"}),
        _f("b", {"V3", "V4"}),
        _f("c", {"V5"}),
        _f("d", {"V1", "V2"}),  # fully redundant with a
    ]
    chosen = select_files(files, target_vessels=4)
    # 'a' (3 new) is taken first, then one more 1-gain file reaches the target of 4 distinct.
    assert chosen[0].origin_url == "a"
    assert len(chosen) == 2
    assert len(covered_vessels(chosen)) == 4
    assert {"V1", "V2", "V3"} <= covered_vessels(chosen)


def test_stops_when_no_file_adds_new_vessels() -> None:
    files = [_f("a", {"V1", "V2"}), _f("b", {"V1", "V2"})]
    chosen = select_files(files, target_vessels=10)
    assert covered_vessels(chosen) == {"V1", "V2"}
    assert len(chosen) == 1  # the second file is fully redundant


def test_max_files_budget_caps_selection() -> None:
    files = [_f(chr(97 + i), {f"V{i}"}) for i in range(10)]
    chosen = select_files(files, target_vessels=10, max_files=3)
    assert len(chosen) == 3


def test_selection_is_deterministic() -> None:
    files = [_f("a", {"V1"}), _f("b", {"V2"}), _f("c", {"V3"})]
    assert select_files(files, 2) == select_files(files, 2)


def test_closeness_prefers_near_passages() -> None:
    files = [
        _f("far", {"V1", "V2", "V3"}, min_range_m=4000.0),
        _f("near", {"V4"}, min_range_m=300.0),
        _f("mid", {"V5"}, min_range_m=1500.0),
    ]
    chosen = select_files_by_closeness(files, target_vessels=2)
    # Nearest passages first, even though 'far' would win a count-greedy selection.
    assert [f.origin_url for f in chosen] == ["near", "mid"]


def test_closeness_excludes_already_covered() -> None:
    files = [_f("a", {"V1", "V2"}, min_range_m=200.0), _f("b", {"V3"}, min_range_m=500.0)]
    chosen = select_files_by_closeness(
        files, target_vessels=1, already_covered=frozenset({"V1", "V2"})
    )
    # V1/V2 already covered, so 'a' adds nothing new and only 'b' is taken.
    assert [f.origin_url for f in chosen] == ["b"]
