"""Formatting rules for the post-load performance stat line (load_stats.py).

Every case here is about how the line READS, not about measurement: the
clock and the psutil reading are taken at the call site in api.py, and
these functions turn whatever they produced — including nothing at all —
into a line that never states a wrong number.
"""
from __future__ import annotations

from ml_data_view import load_stats
from ml_data_view.load_stats import (
    build_load_stats,
    format_cache_clause,
    format_duration,
    format_peak_memory,
    format_stat_line,
)


# --- duration ---------------------------------------------------------


def test_format_duration_uses_one_decimal_second():
    assert format_duration(3.44) == "3.4s"


def test_format_duration_keeps_one_decimal_for_sub_second():
    assert format_duration(0.21) == "0.2s"


def test_format_duration_stays_in_seconds_just_under_the_minute_switch():
    assert format_duration(89.9) == "89.9s"


def test_format_duration_switches_to_minutes_and_seconds_at_the_threshold():
    assert format_duration(90) == "1m 30s"


def test_format_duration_zero_pads_the_seconds_part():
    assert format_duration(125) == "2m 05s"


def test_format_duration_rounds_seconds_in_minute_form():
    assert format_duration(113.4) == "1m 53s"


# --- peak memory ------------------------------------------------------


def test_format_peak_memory_uses_mebibytes_below_a_gibibyte():
    assert format_peak_memory(842 * 1024 * 1024) == "842.0 MiB"


def test_format_peak_memory_switches_to_gibibytes():
    assert format_peak_memory(int(1.94 * 1024 * 1024 * 1024)) == "1.9 GiB"


def test_format_peak_memory_uses_kibibytes_for_small_values():
    assert format_peak_memory(2048) == "2.0 KiB"


def test_format_peak_memory_reports_bytes_without_a_decimal():
    assert format_peak_memory(512) == "512 B"


# --- the assembled line -----------------------------------------------


def test_format_stat_line_renders_every_part():
    line = format_stat_line(
        row_count=195_800_000, file_count=250, elapsed_seconds=113.4, peak_bytes=2067791872
    )

    assert line == (
        "Profiled 195,800,000 rows across 250 files in 1m 53s "
        "— peak memory 1.9 GiB (process)"
    )


def test_format_stat_line_separates_thousands_in_row_counts():
    line = format_stat_line(
        row_count=1_529_035, file_count=1, elapsed_seconds=2.0, peak_bytes=None
    )

    assert "1,529,035 rows" in line


def test_format_stat_line_says_one_file_singular():
    line = format_stat_line(
        row_count=200, file_count=1, elapsed_seconds=0.3, peak_bytes=None
    )

    assert "across 1 file " in line
    assert "1 files" not in line


def test_format_stat_line_says_one_row_singular():
    line = format_stat_line(
        row_count=1, file_count=1, elapsed_seconds=0.3, peak_bytes=None
    )

    assert "Profiled 1 row across" in line


# --- omission rather than a wrong number ------------------------------


def test_format_stat_line_omits_memory_clause_when_unavailable():
    line = format_stat_line(
        row_count=100, file_count=1, elapsed_seconds=1.0, peak_bytes=None
    )

    assert line == "Profiled 100 rows across 1 file in 1.0s"
    assert "memory" not in line


def test_format_stat_line_omits_file_clause_when_file_count_unknown():
    line = format_stat_line(
        row_count=100, file_count=None, elapsed_seconds=1.0, peak_bytes=None
    )

    assert line == "Profiled 100 rows in 1.0s"


def test_format_stat_line_omits_row_clause_when_row_count_unknown():
    """A .csv folder group has no metadata row count (see folder_loader) —
    the line still has something true to say about time."""
    line = format_stat_line(
        row_count=None, file_count=3, elapsed_seconds=1.0, peak_bytes=None
    )

    assert line == "Profiled 3 files in 1.0s"


def test_format_stat_line_omits_duration_clause_when_unavailable():
    line = format_stat_line(
        row_count=100, file_count=2, elapsed_seconds=None, peak_bytes=None
    )

    assert line == "Profiled 100 rows across 2 files"


def test_format_stat_line_keeps_memory_when_only_memory_is_known():
    line = format_stat_line(
        row_count=None, file_count=None, elapsed_seconds=None, peak_bytes=1024 * 1024 * 1024
    )

    assert line == "Peak memory 1.0 GiB (process)"


def test_format_stat_line_returns_none_when_nothing_is_known():
    """Nothing usable means no line at all — never a row of zeroes."""
    assert (
        format_stat_line(
            row_count=None, file_count=None, elapsed_seconds=None, peak_bytes=None
        )
        is None
    )


def test_format_stat_line_treats_zero_rows_as_a_real_value_not_missing():
    """An empty dataset genuinely has 0 rows; only None means unknown."""
    line = format_stat_line(
        row_count=0, file_count=1, elapsed_seconds=0.1, peak_bytes=None
    )

    assert "0 rows" in line


# --- the assembled payload the API hands the frontend -------------------


def test_build_load_stats_bundles_the_line_with_its_raw_parts():
    stats = build_load_stats(
        row_count=5, file_count=2, elapsed_seconds=1.25, peak_bytes=1024 * 1024
    )

    assert stats == {
        "line": "Profiled 5 rows across 2 files in 1.2s — peak memory 1.0 MiB (process)",
        "row_count": 5,
        "file_count": 2,
        "elapsed_seconds": 1.25,
        "peak_bytes": 1024 * 1024,
        # None for an in-process load: there was no ceiling to measure
        # against, so the line keeps the "(process)" phrasing.
        "ceiling_bytes": None,
        # None when the caller did not go through the profile cache — the
        # clause about it is then left out entirely.
        "files_scanned": None,
        "files_from_cache": None,
    }


def test_build_load_stats_reads_peak_memory_itself_when_not_given(monkeypatch):
    monkeypatch.setattr(load_stats, "peak_memory_bytes", lambda: 3 * 1024 * 1024)

    stats = build_load_stats(row_count=1, file_count=1, elapsed_seconds=0.5)

    assert stats["peak_bytes"] == 3 * 1024 * 1024
    assert "peak memory 3.0 MiB (process)" in stats["line"]


def test_build_load_stats_drops_the_memory_clause_when_the_reading_fails(monkeypatch):
    """peak_memory_bytes() returns None off Windows — the line must lose
    the clause rather than report a zero."""
    monkeypatch.setattr(load_stats, "peak_memory_bytes", lambda: None)

    stats = build_load_stats(row_count=1, file_count=1, elapsed_seconds=0.5)

    assert stats["peak_bytes"] is None
    assert stats["line"] == "Profiled 1 row across 1 file in 0.5s"


def test_build_load_stats_line_is_none_when_nothing_is_known(monkeypatch):
    monkeypatch.setattr(load_stats, "peak_memory_bytes", lambda: None)

    stats = build_load_stats(row_count=None, file_count=None, elapsed_seconds=None)

    assert stats["line"] is None


# --- the ceiling clause --------------------------------------------------
#
# The memory clause answers a different question when a load ran as a
# capped background job, so it reads differently. Without a ceiling the
# figure is the PROCESS high-water mark, which Windows offers no way to
# reset; with one it is that single job's own peak, measured by the Job
# Object against the limit it ran under.


def test_a_capped_load_reports_its_peak_against_the_ceiling():
    line = format_stat_line(
        row_count=195_800_000,
        file_count=250,
        elapsed_seconds=113.4,
        peak_bytes=int(1.4 * 1024**3),
        ceiling_bytes=8 * 1024**3,
    )

    assert line == (
        "Profiled 195,800,000 rows across 250 files in 1m 53s "
        "— peak 1.4 GB of 8 GB ceiling"
    )


def test_a_capped_load_does_not_claim_to_be_a_process_high_water_mark():
    """"(process)" says the figure covers everything since launch. A job's
    peak does not, and carrying the qualifier over would make a more
    precise number read as a vaguer one."""
    line = format_stat_line(5, 2, 1.0, 1024**3, ceiling_bytes=8 * 1024**3)

    assert "(process)" not in line


def test_an_uncapped_load_keeps_the_process_wording():
    line = format_stat_line(5, 2, 1.0, 1024**3, ceiling_bytes=None)

    assert "peak memory 1.0 GiB (process)" in line


def test_a_ceiling_with_no_peak_reading_drops_the_whole_clause():
    """Same omission rule as every other figure here: unknown means the
    clause goes, not that a zero appears."""
    line = format_stat_line(5, 2, 1.0, peak_bytes=None, ceiling_bytes=8 * 1024**3)

    assert line == "Profiled 5 rows across 2 files in 1.0s"

# --- the cache clause -----------------------------------------------------
#
# "12 of 279 files profiled, 267 from cache". It replaces the plain file
# count rather than joining it: two file counts in one sentence make a
# reader stop and check whether they are the same number.


def test_a_partly_cached_load_says_how_much_it_actually_read():
    line = format_stat_line(
        row_count=195_800_000,
        file_count=279,
        elapsed_seconds=4.1,
        peak_bytes=None,
        files_scanned=12,
        files_from_cache=267,
    )

    assert line == "Profiled 195,800,000 rows in 4.1s \u2014 12 of 279 files profiled, 267 from cache"


def test_a_fully_cached_load_says_nothing_was_read():
    line = format_stat_line(5, 3, 0.2, None, files_scanned=0, files_from_cache=3)

    assert "0 of 3 files profiled, 3 from cache" in line


def test_a_first_open_keeps_the_ordinary_wording():
    """"279 of 279 files profiled, 0 from cache" is a sentence about an
    absence; "across 279 files" already says it, and better."""
    line = format_stat_line(5, 3, 0.2, None, files_scanned=3, files_from_cache=0)

    assert line == "Profiled 5 rows across 3 files in 0.2s"


def test_a_load_that_never_consulted_the_cache_keeps_the_ordinary_wording():
    line = format_stat_line(5, 3, 0.2, None, files_scanned=None, files_from_cache=None)

    assert line == "Profiled 5 rows across 3 files in 0.2s"


def test_the_cache_clause_sits_between_the_subject_and_the_memory_clause():
    line = format_stat_line(
        5, 3, 0.2, 1024**3, ceiling_bytes=8 * 1024**3, files_scanned=1, files_from_cache=2
    )

    assert line == (
        "Profiled 5 rows in 0.2s \u2014 1 of 3 files profiled, 2 from cache "
        "\u2014 peak 1 GB of 8 GB ceiling"
    )


def test_the_clause_alone_is_none_when_nothing_came_from_cache():
    assert format_cache_clause(12, 0) is None
    assert format_cache_clause(0, 0) is None
    assert format_cache_clause(12, 267) == "12 of 279 files profiled, 267 from cache"
