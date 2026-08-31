"""Formatting for the post-load performance stat line.

A quiet single line under the top strip saying how much was profiled, how
long it took, and how much memory the process peaked at — the same figures
the diagnostics in PROJECT_BRIEF.md quote by hand ("~113s at a 1972 MiB
peak"), surfaced in the app instead of a log.

Pure formatting plus one psutil reading. Nothing here touches the loaded
dataset: the row/file counts are already known by the time a load finishes
(row_count from a streaming aggregation or parquet metadata, file count
from the schema group), so this adds no work to the lazy/streaming data
path — see PROJECT_BRIEF.md's architecture notes.

The guiding rule, shared with the partial-load and filter banners: a
figure that isn't known is left out entirely rather than shown as a zero
or a guess. Every clause below is independently omittable, and when
nothing is known there is no line at all.
"""
from __future__ import annotations

from typing import Any

# Seconds at or above which the duration reads as minutes+seconds rather
# than a decimal number of seconds. Past a minute and a half, "113.4s" is
# arithmetic the reader has to do themselves.
MINUTES_THRESHOLD_SECONDS = 90

_MEMORY_UNITS = ("KiB", "MiB", "GiB", "TiB")


def format_duration(seconds: float) -> str:
    """Wall time as "3.4s", switching to "1m 53s" at MINUTES_THRESHOLD_SECONDS."""
    if seconds < MINUTES_THRESHOLD_SECONDS:
        return f"{seconds:.1f}s"
    whole = round(seconds)
    minutes, remainder = divmod(whole, 60)
    return f"{minutes}m {remainder:02d}s"


def format_peak_memory(byte_count: int) -> str:
    """Bytes as a binary-unit string — "842.0 MiB", "1.9 GiB".

    Whole bytes below 1 KiB (a decimal there would be noise); one decimal
    everywhere above, matching how the memory measurements in
    PROJECT_BRIEF.md are quoted.
    """
    if byte_count < 1024:
        return f"{byte_count} B"
    value = float(byte_count)
    for unit in _MEMORY_UNITS:
        value /= 1024
        if value < 1024 or unit == _MEMORY_UNITS[-1]:
            return f"{value:.1f} {unit}"
    raise AssertionError("unreachable — the loop always returns on the last unit")


def peak_memory_bytes() -> int | None:
    """This process's peak working set, or None if it can't be read.

    Windows exposes a true high-water mark (`peak_wset`); nothing
    equivalent is wired up for other platforms here, and an ordinary
    current-RSS reading is NOT a substitute — it would silently understate
    the peak, which is worse than saying nothing. So this returns None
    off Windows and the caller drops the clause.

    Note this is the peak since the process started, not since the current
    load began — hence the "(process)" qualifier format_stat_line() adds.
    Windows offers no way to reset the high-water mark, so a small file
    opened after a large folder would otherwise appear to have used the
    folder's memory.
    """
    try:
        import psutil
    except ImportError:
        return None
    try:
        info = psutil.Process().memory_info()
    except Exception:
        return None
    peak = getattr(info, "peak_wset", None)
    return int(peak) if peak is not None else None


def _plural(count: int, noun: str) -> str:
    return f"{count:,} {noun}" if count == 1 else f"{count:,} {noun}s"


def format_stat_line(
    row_count: int | None,
    file_count: int | None,
    elapsed_seconds: float | None,
    peak_bytes: int | None,
) -> str | None:
    """The whole line, or None when no part of it is known.

    Each argument is independently optional — a .csv folder group has no
    metadata row count (folder_loader.py reports it as unknown rather than
    forcing a full scan), and peak memory is unavailable off Windows. Only
    None means unknown: a genuinely empty dataset still reads "0 rows".
    """
    subjects = []
    if row_count is not None:
        subjects.append(_plural(row_count, "row"))
    if file_count is not None:
        subjects.append(_plural(file_count, "file"))

    profiled = ""
    if subjects:
        profiled = f"Profiled {' across '.join(subjects)}"
        if elapsed_seconds is not None:
            profiled += f" in {format_duration(elapsed_seconds)}"
    elif elapsed_seconds is not None:
        profiled = f"Profiled in {format_duration(elapsed_seconds)}"

    if peak_bytes is None:
        return profiled or None

    memory = f"peak memory {format_peak_memory(peak_bytes)} (process)"
    if not profiled:
        # Only the first character — .capitalize() would lowercase the
        # unit suffix ("GiB" -> "gib").
        return memory[0].upper() + memory[1:]
    return f"{profiled} — {memory}"


_UNSET = object()


def build_load_stats(
    row_count: int | None,
    file_count: int | None,
    elapsed_seconds: float | None,
    peak_bytes: int | None | object = _UNSET,
) -> dict[str, Any]:
    """The payload the frontend renders: the ready-made `line` plus the
    raw parts behind it.

    `peak_bytes` defaults to reading peak_memory_bytes() here, so the three
    call sites (load_file, load_folder_group, FolderLoadJob) don't each
    repeat it. The frontend only needs `line`; the raw parts are kept so a
    caller that wants the numbers isn't forced to parse the sentence back
    apart.
    """
    if peak_bytes is _UNSET:
        peak_bytes = peak_memory_bytes()
    return {
        "line": format_stat_line(row_count, file_count, elapsed_seconds, peak_bytes),
        "row_count": row_count,
        "file_count": file_count,
        "elapsed_seconds": elapsed_seconds,
        "peak_bytes": peak_bytes,
    }
