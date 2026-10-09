"""A unit's channels over time, at whatever resolution the window needs.

APPLICABILITY
A detected timestamp column, and nothing else. A single CSV with a time
column gets a timeline; a series (hierarchy.py) just means the viewer has
several chunks to choose between rather than one. Nothing here asks what
the directory tree looked like, and nothing here needs a series level.

THE ONE RULE THAT MAKES THIS AFFORDABLE
A fetch reads the ROW GROUPS whose own time range overlaps the requested
window — and only when their combined row count is within a stated budget
(`timeline_max_rows`). Otherwise it answers from the per-chunk buckets
that are already in the profile cache, at whatever resolution those give,
and SAYS SO.

Both halves matter. Because the cached buckets are always an acceptable
answer, the viewer opens instantly on a recording of any size: a
200-million-row chunk hands back its twenty-four cached buckets rather
than being read. Because a read is planned from parquet footer statistics
before a byte of data is decoded, zooming into that same chunk reads the
handful of row groups the window actually touches — so detail appears as
you zoom, and no request ever reads more rows than the budget names. RAW
ROWS NEVER LEAVE PYTHON at any zoom: what goes to the frontend is a
bounded number of buckets, each carrying min, max, mean and a count.

WHY BUCKETS AND NOT A DECIMATION
A min/max envelope per bucket is the only downsample that cannot hide a
spike. Picking every Nth row, or the mean alone, makes a one-sample
excursion vanish at exactly the zoom level where somebody is looking for
it — and the whole point of this view is finding the moment a channel did
something it should not have.

WHAT IS DRAWN OVER IT IS DECIDED HERE
Gaps, missing chunks, dropouts, flatlines and out-of-range extremes are
turned into marks in this module, from the same merged artifact the
findings read (ts_checks.py). A second copy of "what counts as a gap" in
JS is how the timeline and the finding beside it would end up disagreeing
— the same rule timeseries.coverage_strip already follows.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ml_data_view.data_loader import BATCH_ROWS
from ml_data_view.timeseries import out_of_range_extremes

# What the frontend asks for by default, and the most it may ask for. The
# cap is not politeness: a bucket is six numbers per channel, and a caller
# that asked for a bucket per row would have asked for the raw rows this
# module exists to avoid sending.
DEFAULT_BUCKETS = 600
MAX_BUCKETS = 2000

# Channels drawn at once. A strip per channel is a chart; forty strips is
# a wall, and the read cost is per channel as well.
DEFAULT_MAX_CHANNELS = 6

# Rows a single fetch may decode, across all the row groups it selected.
# Above this the answer comes from the cache instead — see the module
# docstring. Overridable from Settings.
DEFAULT_MAX_ROWS = 2_000_000

# Overlay marks kept, worst first. Bounded for the same reason every list
# that reaches the frontend is.
MAX_MARKS = 200


@dataclass
class Chunk:
    """One chunk, and what is already known about it without reading it."""

    path: Path
    first_ts: float | None = None
    last_ts: float | None = None
    # The cached per-chunk time-series artifact, when there is one. Its
    # per-channel `buckets` are the whole no-read answer.
    artifact: dict[str, Any] = field(default_factory=dict)

    def overlaps(self, window: tuple[float, float] | None) -> bool:
        """True unless this chunk is KNOWN to fall outside the window.

        A chunk whose bounds are unknown always overlaps. That is not
        caution for its own sake: excluding it would silently drop rows
        from an answer presented as complete, and the only honest thing to
        do with "we do not know when this file covers" is to look.
        """
        if window is None or self.first_ts is None or self.last_ts is None:
            return True
        return self.last_ts >= window[0] and self.first_ts <= window[1]


def chunks_from_artifacts(
    paths: list[Path], artifacts: dict[str, dict[str, Any]]
) -> list[Chunk]:
    """Pair each chunk path with its cached artifact, in the given order.

    The order is the caller's, and it is already the right one:
    hierarchy.py ordered a series' chunks by name and then verified that
    against their timestamps.
    """
    chunks = []
    for path in paths:
        artifact = artifacts.get(str(path)) or {}
        chunks.append(
            Chunk(
                path=Path(path),
                first_ts=artifact.get("first_ts"),
                last_ts=artifact.get("last_ts"),
                artifact=artifact,
            )
        )
    return chunks


def unit_window(chunks: list[Chunk]) -> tuple[float, float] | None:
    """The whole span the unit covers, or None when no chunk said."""
    firsts = [c.first_ts for c in chunks if c.first_ts is not None]
    lasts = [c.last_ts for c in chunks if c.last_ts is not None]
    if not firsts or not lasts:
        return None
    return min(firsts), max(lasts)


# --- planning a read -----------------------------------------------------


@dataclass
class ReadPlan:
    """Which row groups a window needs, decided from footers alone."""

    # (path, [row group indices]) — an empty index list means "the whole
    # file", which is what a format with no row groups gets.
    groups: list[tuple[Path, list[int] | None]] = field(default_factory=list)
    rows: int = 0
    chunks_considered: int = 0

    @property
    def chunks(self) -> int:
        return len(self.groups)


def plan_read(
    chunks: list[Chunk], timestamp_column: str, window: tuple[float, float] | None
) -> ReadPlan:
    """What reading `window` would cost, before reading anything.

    Parquet stores per-row-group min/max for every column in its footer,
    so a window narrower than a chunk is answerable by decoding a few of
    its row groups rather than all of them. That is what makes zooming
    into a single enormous chunk affordable — and knowing the row count in
    advance is what lets the caller decline.
    """
    plan = ReadPlan()
    for chunk in chunks:
        if not chunk.overlaps(window):
            continue
        plan.chunks_considered += 1
        selected, rows = _row_groups(chunk.path, timestamp_column, window)
        if selected is not None and not selected:
            continue  # every row group ruled out by its own statistics
        plan.groups.append((chunk.path, selected))
        plan.rows += rows
    return plan


def _row_groups(
    path: Path, timestamp_column: str, window: tuple[float, float] | None
) -> tuple[list[int] | None, int]:
    """(row group indices, their row count) for one file.

    None for the indices means "no row groups to choose between" — a CSV
    or an arrow file, which is read whole. The row count is then the best
    estimate available, and for a format with no metadata it is unknown,
    reported as 0 rather than guessed at; the caller's budget therefore
    never blocks a single small CSV, which is the only shape that reaches
    that path.
    """
    if Path(path).suffix.lower() != ".parquet":
        return None, 0
    try:
        import pyarrow.parquet as pq

        parquet = pq.ParquetFile(path)
        metadata = parquet.metadata
    except Exception:
        return None, 0

    try:
        index = parquet.schema_arrow.names.index(timestamp_column)
    except ValueError:
        return None, int(metadata.num_rows)

    from ml_data_view.hierarchy import _timestamp_value

    selected: list[int] = []
    rows = 0
    for number in range(metadata.num_row_groups):
        group = metadata.row_group(number)
        statistics = group.column(index).statistics
        if window is not None and statistics is not None and statistics.has_min_max:
            low = _timestamp_value(statistics.min)
            high = _timestamp_value(statistics.max)
            if low is not None and high is not None and (high < window[0] or low > window[1]):
                continue
        selected.append(number)
        rows += group.num_rows
    return selected, rows


# --- reading -------------------------------------------------------------


def read_series(
    plan: ReadPlan,
    timestamp_column: str,
    channels: list[str],
    window: tuple[float, float],
    buckets: int,
) -> dict[str, dict[str, list]]:
    """Bucketed min/max/mean per channel, over equal time buckets.

    One channel at a time, one batch at a time — the same bounded pattern
    timeseries._analyse_channel uses, and for the same reason: peak memory
    is a batch of one column, whatever the window turned out to contain.

    Rows outside the window are dropped here rather than in the plan: a
    row group is selected because it OVERLAPS the window, so most of the
    ones at either end of it are half outside.
    """
    import numpy as np

    start, end = window
    width = (end - start) / buckets if end > start else 0.0
    state = {name: _Accumulator(buckets) for name in channels}

    for path, row_groups in plan.groups:
        for name in channels:
            for values, stamps in _iter_window(
                Path(path), [name, timestamp_column], row_groups
            ):
                if len(values) == 0:
                    continue
                inside = (stamps >= start) & (stamps <= end) & ~np.isnan(values)
                if not np.any(inside):
                    continue
                index = (
                    np.zeros(int(np.count_nonzero(inside)), dtype="int64")
                    if width <= 0
                    else np.clip(
                        ((stamps[inside] - start) / width).astype("int64"), 0, buckets - 1
                    )
                )
                state[name].feed(index, values[inside])

    edges = [start + width * i for i in range(buckets + 1)]
    return {name: state[name].finish(edges) for name in channels}


class _Accumulator:
    """Per-bucket count, sum, min and max, filled in place.

    `np.minimum.at` rather than a sort-and-reduce: the bucket index is
    only non-decreasing while time is, and a recording whose timestamps
    step backwards is precisely one somebody is looking at this view to
    understand. Unbuffered ufunc.at is slower than reduceat, and it is
    correct for any ordering, which reduceat is not.
    """

    __slots__ = ("count", "total", "low", "high")

    def __init__(self, buckets: int) -> None:
        import numpy as np

        self.count = np.zeros(buckets, dtype="int64")
        self.total = np.zeros(buckets, dtype="float64")
        self.low = np.full(buckets, np.inf, dtype="float64")
        self.high = np.full(buckets, -np.inf, dtype="float64")

    def feed(self, index, values) -> None:
        import numpy as np

        np.add.at(self.count, index, 1)
        np.add.at(self.total, index, values)
        np.minimum.at(self.low, index, values)
        np.maximum.at(self.high, index, values)

    def finish(self, edges: list[float]) -> dict[str, list]:
        empty = self.count == 0
        return {
            "t0": edges[:-1],
            "t1": edges[1:],
            "n": [int(value) for value in self.count],
            # An empty bucket is None, not zero. A channel with no rows in
            # a stretch of time and a channel reading zero there are
            # different things, and an envelope that draws them the same
            # way invents data.
            "min": [None if blank else float(value) for blank, value in zip(empty, self.low)],
            "max": [None if blank else float(value) for blank, value in zip(empty, self.high)],
            "mean": [
                None if blank else float(total / count)
                for blank, total, count in zip(empty, self.total, self.count)
            ],
        }


def _iter_window(path: Path, columns: list[str], row_groups: list[int] | None):
    """The requested columns, in bounded pieces, from selected row groups.

    A thin extension of timeseries._iter_columns: same batching, same
    float conversion, but able to skip the row groups a window does not
    touch. That skip is the difference between zooming into a large chunk
    and re-reading it.
    """
    from ml_data_view.timeseries import _iter_columns, _to_float_array

    if Path(path).suffix.lower() != ".parquet" or row_groups is None:
        yield from _iter_columns(Path(path), columns)
        return

    import pyarrow.parquet as pq

    parquet = pq.ParquetFile(path)
    if not row_groups:
        return
    for batch in parquet.iter_batches(
        batch_size=BATCH_ROWS, columns=columns, row_groups=row_groups
    ):
        yield [_to_float_array(batch.column(index)) for index in range(len(columns))]


# --- answering from the cache -------------------------------------------


def cached_series(
    chunks: list[Chunk],
    channels: list[str],
    window: tuple[float, float] | None,
    buckets: int,
) -> dict[str, dict[str, list]]:
    """The same shape, folded out of the per-chunk buckets already stored.

    Not an approximation of the read: each cached bucket carries the exact
    min, max, count and sum of the rows it covers, and folding two
    adjacent ones is exact for all four. The only thing coarser here is
    RESOLUTION — where the bucket edges fall — which is why a caller that
    wants more of it zooms in and pays for a read.
    """
    from ml_data_view.timeseries import fold_channel_buckets

    result: dict[str, dict[str, list]] = {}
    for name in channels:
        rows: list[tuple] = []
        for chunk in chunks:
            stored = ((chunk.artifact.get("channels") or {}).get(name) or {}).get("buckets")
            if not stored:
                continue
            for entry in zip(
                stored.get("t0") or [],
                stored.get("t1") or [],
                stored.get("n") or [],
                stored.get("min") or [],
                stored.get("max") or [],
                stored.get("sum") or [],
            ):
                if window is not None and (entry[1] < window[0] or entry[0] > window[1]):
                    continue
                rows.append(entry)
        rows.sort(key=lambda row: row[0])
        folded = fold_channel_buckets(rows, buckets)
        result[name] = {
            "t0": [row[0] for row in folded],
            "t1": [row[1] for row in folded],
            "n": [int(row[2]) for row in folded],
            "min": [row[3] for row in folded],
            "max": [row[4] for row in folded],
            "mean": [
                (row[5] / row[2]) if row[2] else None for row in folded
            ],
        }
    return result


def cached_resolution(
    chunks: list[Chunk], channels: list[str], window: tuple[float, float] | None
) -> int:
    """How many cached buckets fall inside the window, at best.

    What the caller compares against what was asked for, to decide whether
    a read would tell it anything it does not already know.
    """
    best = 0
    for name in channels:
        count = 0
        for chunk in chunks:
            stored = ((chunk.artifact.get("channels") or {}).get(name) or {}).get("buckets")
            if not stored:
                continue
            for t0, t1 in zip(stored.get("t0") or [], stored.get("t1") or []):
                if window is None or (t1 >= window[0] and t0 <= window[1]):
                    count += 1
        best = max(best, count)
    return best


# --- the decision --------------------------------------------------------


def build_timeline(
    chunks: list[Chunk],
    timestamp_column: str,
    channels: list[str],
    *,
    window: tuple[float, float] | None = None,
    buckets: int = DEFAULT_BUCKETS,
    max_rows: int = DEFAULT_MAX_ROWS,
) -> dict[str, Any]:
    """One unit's channels over `window`, read or recalled as the cost says.

    The decision, in full: if the cached buckets already answer at the
    resolution asked for, fold them and open nothing. Otherwise plan the
    read from footers and take it when its row count is within budget,
    falling back to the cached buckets — with a note saying so — when it
    is not. Either way the caller is told which happened, how many chunks
    were read and how many rows, because "instant because it was cached"
    and "instant because there was nothing there" look identical from
    outside.
    """
    buckets = max(1, min(int(buckets), MAX_BUCKETS))
    span = unit_window(chunks)
    if span is None:
        return {
            "available": False,
            "reason": "no chunk in this selection reported a time range",
            "channels": [],
        }
    window = _clamp(window, span)
    selected = [chunk for chunk in chunks if chunk.overlaps(window)]
    have = cached_resolution(selected, channels, window)

    # A read is worth its cost only if it beats what is already stored.
    # Zooming a long way into one chunk is the case this exists for: the
    # cache holds twenty-four buckets for the whole chunk and perhaps one
    # of them touches the window, while a read of two row groups answers
    # at full resolution. Asked coarsely enough, not even the FOOTERS are
    # opened — there is nothing a plan could tell us that would change the
    # answer.
    worthwhile = have < buckets
    plan = plan_read(selected, timestamp_column, window) if worthwhile else ReadPlan()
    affordable = plan.rows <= max_rows if plan.rows else bool(plan.groups)

    if worthwhile and affordable:
        series = read_series(plan, timestamp_column, channels, window, buckets)
        source, note = "scan", ""
    else:
        series = cached_series(selected, channels, window, buckets)
        source = "cache"
        note = (
            ""
            if not worthwhile
            else (
                f"Showing {have} cached bucket(s): reading this window would decode "
                f"{plan.rows:,} rows, past the {max_rows:,}-row budget. Zoom in for detail."
            )
        )

    return {
        "available": True,
        "timestamp_column": timestamp_column,
        "channels": [{"column": name, **series.get(name, {})} for name in channels],
        "window": list(window),
        "span": list(span),
        "buckets": buckets,
        # How the answer was produced, always. See the docstring.
        "source": source,
        "note": note,
        "chunks_total": len(chunks),
        "chunks_in_window": len(selected),
        "chunks_read": plan.chunks if source == "scan" else 0,
        "rows_read": plan.rows if source == "scan" else 0,
        "max_rows": max_rows,
        "cached_buckets": have,
    }


def _clamp(
    window: tuple[float, float] | None, span: tuple[float, float]
) -> tuple[float, float]:
    if window is None:
        return span
    start, end = float(window[0]), float(window[1])
    if end <= start:
        return span
    return max(start, span[0]), min(end, span[1])


# --- what is drawn over it ----------------------------------------------


def segment_span(segment: dict[str, Any], dt_median: float | None) -> tuple[float, float] | None:
    """A run's start and end in TIME, from its start and its length in rows.

    Run lengths are counted in rows, because that is what a run is. The
    conversion to a duration uses the unit's own median interval — the
    only one available without re-reading — and every caller states that
    rather than presenting the end as measured. timeseries.channel_strips
    makes the same conversion; this is the one implementation of it.
    """
    start = segment.get("at")
    if start is None:
        return None
    length = float(segment.get("length") or 0)
    return float(start), float(start) + length * float(dt_median or 0.0)


def overlay(
    unit: dict[str, Any],
    channels: list[str],
    ranges: dict[str, Any] | None = None,
    range_spread_factor: float = 10.0,
) -> list[dict[str, Any]]:
    """The findings, as marks on the time axis.

    Built from the merged per-unit artifact — the same dict ts_checks.py
    reads — so a mark and the finding that describes it can never disagree
    about where something happened. Each carries `channel` (None for one
    about the recording as a whole), a kind, a span, and the text the
    tooltip shows.
    """
    dt_median = unit.get("dt_median")
    marks: list[dict[str, Any]] = []

    for gap in unit.get("gaps") or []:
        start = float(gap["at"])
        boundary = bool(gap.get("at_boundary"))
        marks.append(
            {
                "kind": "missing" if boundary else "gap",
                "channel": None,
                "t0": start,
                "t1": start + float(gap["duration"]),
                "severity": "critical" if boundary else "warning",
                "label": (
                    f"Missing chunk after {gap.get('after_chunk')}"
                    if boundary
                    else "Gap in time"
                ),
                "detail": f"{_seconds(gap['duration'])} with no rows",
            }
        )

    for overlap in unit.get("overlaps") or []:
        at = float(overlap["at"])
        marks.append(
            {
                "kind": "overlap",
                "channel": None,
                "t0": at,
                "t1": at + float(overlap.get("overlap") or 0.0),
                "severity": "critical",
                "label": f"Chunks overlap: {overlap.get('after_chunk')} / {overlap.get('next_chunk')}",
                "detail": f"{_seconds(overlap.get('overlap'))} covered twice",
            }
        )

    for name in channels:
        channel = (unit.get("channels") or {}).get(name)
        if not channel:
            continue
        for segment in channel.get("segments") or []:
            span = segment_span(segment, dt_median)
            if span is None:
                continue
            kind = segment.get("kind")
            marks.append(
                {
                    "kind": "nan" if kind == "nan" else "flat",
                    "channel": name,
                    "t0": span[0],
                    "t1": span[1],
                    "severity": "warning",
                    "label": "Dropout" if kind == "nan" else "Flatline",
                    "detail": (
                        f"{segment.get('length', 0):,} rows"
                        + ("" if dt_median else " (no median interval to convert to a duration)")
                    ),
                }
            )
        marks.extend(_range_marks(name, channel, ranges, range_spread_factor))

    # Worst first, then earliest. A capped list has to keep the marks
    # somebody would act on, not the ones that happen to sort first.
    order = {"critical": 0, "warning": 1, "info": 2}
    marks.sort(key=lambda mark: (order.get(mark["severity"], 3), mark["t0"]))
    return marks[:MAX_MARKS]


def _range_marks(
    name: str, channel: dict[str, Any], ranges: dict[str, Any] | None, factor: float
) -> list[dict[str, Any]]:
    """Where a channel reached an extreme worth pointing at.

    The verdict comes from timeseries.out_of_range_extremes — the same
    function the finding uses — so a mark on the timeline and the finding
    that describes it can never disagree about whether a spike counts. Two
    rules, labelled differently: exact against a documented range,
    HEURISTIC without one, in those words.
    """
    documented = (ranges or {}).get(name)
    marks = []
    for extreme in out_of_range_extremes(channel, documented, factor):
        at, value = extreme.get("at"), extreme.get("value")
        if at is None or value is None:
            continue
        exact = extreme["rule"] == "documented"
        marks.append(
            {
                "kind": "out_of_range",
                "channel": name,
                "t0": float(at),
                "t1": float(at),
                "severity": "critical" if exact else "warning",
                "label": (
                    f"{name} outside its documented range"
                    if exact
                    else f"{name} reaches an extreme value"
                ),
                "detail": (
                    f"{extreme['edge']} {value:.6g}, documented range "
                    f"{documented[0]:.6g} to {documented[1]:.6g}"
                    if exact
                    else (
                        f"HEURISTIC — {extreme['edge']} {value:.6g}, "
                        f"{extreme['distance']:.0f}x the p0.1–p99.9 spread beyond it"
                    )
                ),
            }
        )
    return marks


def _seconds(value: Any) -> str:
    """A duration as a person reads it — ts_checks._seconds' shape."""
    if value is None:
        return "unknown"
    value = float(value)
    if value < 1:
        return f"{value * 1000:.0f}ms"
    if value < 90:
        return f"{value:.2f}s"
    minutes, seconds = divmod(value, 60)
    return f"{int(minutes)}m {seconds:04.1f}s"


# --- a selected window ---------------------------------------------------


def window_stats(
    timeline: dict[str, Any], window: tuple[float, float]
) -> dict[str, Any]:
    """Summary statistics for a lassoed stretch, from the buckets on screen.

    Read off the buckets rather than by scanning again, which is what
    makes a selection instant — and which also decides what the numbers
    mean. A bucket is either wholly inside the selection or not counted,
    so the figures describe whole buckets, and the payload says whether
    the selection happened to land on bucket edges (`exact`). At a scanned
    zoom the buckets are narrow and the difference is invisible; at a
    cached one it can be a whole chunk's worth, and quietly reporting the
    wrong rows would be worse than saying which rows these are.
    """
    start, end = float(window[0]), float(window[1])
    channels = []
    covered_from: float | None = None
    covered_to: float | None = None
    for channel in timeline.get("channels") or []:
        count = 0
        total = 0.0
        low: float | None = None
        high: float | None = None
        for t0, t1, n, minimum, maximum, mean in zip(
            channel.get("t0") or [],
            channel.get("t1") or [],
            channel.get("n") or [],
            channel.get("min") or [],
            channel.get("max") or [],
            channel.get("mean") or [],
        ):
            if t0 < start or t1 > end or not n:
                continue
            count += int(n)
            total += float(mean) * int(n)
            low = float(minimum) if low is None else min(low, float(minimum))
            high = float(maximum) if high is None else max(high, float(maximum))
            covered_from = t0 if covered_from is None else min(covered_from, t0)
            covered_to = t1 if covered_to is None else max(covered_to, t1)
        channels.append(
            {
                "column": channel.get("column"),
                "n": count,
                "min": low,
                "max": high,
                "mean": (total / count) if count else None,
            }
        )
    return {
        "start": start,
        "end": end,
        "duration": end - start,
        "channels": channels,
        # The stretch the figures actually describe, and whether that is
        # the stretch that was asked for.
        "covered": [covered_from, covered_to] if covered_from is not None else None,
        "exact": covered_from is not None
        and math.isclose(covered_from, start, rel_tol=0, abs_tol=(end - start) * 1e-6)
        and math.isclose(covered_to, end, rel_tol=0, abs_tol=(end - start) * 1e-6),
        "source": timeline.get("source"),
    }
