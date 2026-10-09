"""Time-series health: what a recorded signal looks like when it goes wrong.

Everything here answers questions about ORDER — is time going forwards, is
it evenly spaced, did the sensor stop moving — which is the one kind of
question the rest of this app cannot answer, because every other statistic
it computes is order-free. A null count does not care which rows the nulls
were in; a flatline is nothing but which rows.

APPLICABILITY
These checks hang off a DETECTED TIMESTAMP COLUMN, not off a directory
structure. A single CSV with a time column gets the timestamp and channel
checks. Only the chunk-boundary ones — missing chunk, overlap, chunk
ordering — additionally need a series level, because a unit that is one
file has no boundaries between chunks. See findings.py for the predicates
and hierarchy.py for where "unit" comes from.

MEMORY: BOUNDED BY A BATCH, NOT BY A RECORDING
A chunk is the cacheable unit, but it is not the memory unit. Each column
is read through pyarrow's `iter_batches`, which splits WITHIN a row group,
so peak memory here is BATCH_ROWS x one column regardless of how large a
chunk is or how many chunks a recording has — the same technique, and the
same reason, as data_loader._bounded_parquet_missing_counts. Run lengths,
dt statistics and boundary state are carried incrementally across batches
and then across chunks.

WHY NOT POLARS STREAMING
`diff`, `shift` and `rle` are all accepted under `collect(engine=
"streaming")` — they do not raise — but every one of them is a
whole-column, order-dependent operation, so the engine materialises the
column to answer it. That would make peak memory a function of chunk size
rather than batch size, and it would give no way to carry a run across a
chunk boundary, which is the entire difficulty here. So the analysis is
numpy over pyarrow batches; polars is still used for the bounded per-chunk
quantiles, where it genuinely does stream.

MERGEABILITY
Per-chunk results are cached (profile_cache's timeseries section) and
merged in time order. Sums are sums; histograms add bin-wise; runs join
across a boundary using each chunk's LEADING and TRAILING run lengths,
which is why those are recorded separately from the longest interior run.
The recording-level answers that only exist AT a boundary — a gap that
falls exactly between two chunks, an overlap, a backwards step from one
chunk to the next — are computed during the merge from the neighbouring
chunks' first/last timestamps, never by re-reading anything.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ml_data_view.data_loader import BATCH_ROWS

# --- tunables, all overridable from Settings -----------------------------

# A dt this many times the median is a gap rather than jitter. Five is
# loose enough that ordinary scheduling jitter never trips it and tight
# enough to catch a dropped second at 100Hz.
DEFAULT_GAP_FACTOR = 5.0
# Identical consecutive values beyond this are a stuck sensor rather than a
# quiet one. At 100Hz this is five seconds of a signal that does not move.
DEFAULT_FLATLINE_ROWS = 500
# A unit shorter than this share of the median unit duration is suspicious.
DEFAULT_SHORT_UNIT_RATIO = 0.5
# Jitter (MAD/median) above this is worth an info finding.
DEFAULT_JITTER_RATIO = 0.25
# Segments whose median dt differs from the recording's by more than this
# are a rate CHANGE rather than jitter.
DEFAULT_RATE_CHANGE_RATIO = 0.5
# How far beyond the inter-percentile spread a value has to sit before the
# heuristic out-of-range rule speaks. Deliberately generous: this fires
# without a data dictionary and has to earn its interruption.
DEFAULT_RANGE_SPREAD_FACTOR = 10.0

# Largest gaps kept per chunk. Bounded, and far above the number of real
# gaps any recording has — a chunk that fills it is itself the finding.
MAX_LARGE_DTS = 200
# Coverage/NaN/flatline segments kept per chunk per channel.
MAX_SEGMENTS = 50
# Time buckets emitted per chunk. The frontend never receives raw rows;
# these are the downsampled aggregates the coverage strip and the
# dt-over-time plot are drawn from.
BUCKETS_PER_CHUNK = 24
# Buckets in the merged payload, after adjacent ones are folded together.
MAX_MERGED_BUCKETS = 400

# Rows a channel needs before the percentile HEURISTIC for out-of-range is
# allowed to speak. With only a few hundred values, p99.9 lands on the
# outlier itself and the rule cannot see what it is looking for — so it
# stays silent rather than reporting an absence of evidence as evidence of
# absence. The documented-range path has no such floor: it is a comparison
# against a stated rule and is exact at any size.
MIN_ROWS_FOR_RANGE_HEURISTIC = 1_000

# dt histogram: log-spaced, ten bins per decade from 1e-9 to 1e6 seconds.
# Log-spaced because sample intervals span nanoseconds to minutes and a
# linear histogram would put every real dt in one bin. Summing bins across
# chunks is exact, which is what makes the merged median well defined.
DT_DECADES = (-9, 6)
DT_BINS_PER_DECADE = 10
DT_BIN_COUNT = (DT_DECADES[1] - DT_DECADES[0]) * DT_BINS_PER_DECADE


def dt_bin_index(value: float) -> int:
    """Which log-spaced bin a positive dt falls in."""
    if value <= 0:
        return 0
    order = math.log10(value)
    index = int((order - DT_DECADES[0]) * DT_BINS_PER_DECADE)
    return max(0, min(DT_BIN_COUNT - 1, index))


def dt_bin_value(index: int) -> float:
    """The representative dt for a bin — its geometric centre."""
    order = DT_DECADES[0] + (index + 0.5) / DT_BINS_PER_DECADE
    return float(10.0**order)


def histogram_quantile(bins: list[int], quantile: float) -> float | None:
    """A quantile read off the merged dt histogram.

    Bin-resolution rather than exact: with ten bins per decade the answer
    is within about 12% of the true value, which is far finer than any
    threshold here needs — a gap is 5x the median, not 1.1x.
    """
    total = sum(bins)
    if total <= 0:
        return None
    target = total * quantile
    seen = 0
    for index, count in enumerate(bins):
        seen += count
        if seen >= target:
            return dt_bin_value(index)
    return dt_bin_value(len(bins) - 1)


@dataclass
class Thresholds:
    """Everything a user can tune, in one record.

    Passed into the analysis rather than read from Settings inside it, so
    the module stays importable and testable without an app around it —
    the same rule chart_data.py and findings.py follow.
    """

    gap_factor: float = DEFAULT_GAP_FACTOR
    flatline_rows: int = DEFAULT_FLATLINE_ROWS
    short_unit_ratio: float = DEFAULT_SHORT_UNIT_RATIO
    jitter_ratio: float = DEFAULT_JITTER_RATIO
    rate_change_ratio: float = DEFAULT_RATE_CHANGE_RATIO
    range_spread_factor: float = DEFAULT_RANGE_SPREAD_FACTOR
    # {column: [low, high]} from a data dictionary, when one is configured.
    # With it, out-of-range is an exact count; without it, a heuristic on
    # the extremes. The two are labelled differently for that reason.
    ranges: dict[str, list[float]] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "gap_factor": self.gap_factor,
            "flatline_rows": self.flatline_rows,
            "short_unit_ratio": self.short_unit_ratio,
            "jitter_ratio": self.jitter_ratio,
            "rate_change_ratio": self.rate_change_ratio,
            "range_spread_factor": self.range_spread_factor,
            "ranges": dict(self.ranges),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> Thresholds:
        data = data or {}
        defaults = cls()
        ranges = data.get("ranges") or {}
        return cls(
            gap_factor=float(data.get("gap_factor") or defaults.gap_factor),
            flatline_rows=int(data.get("flatline_rows") or defaults.flatline_rows),
            short_unit_ratio=float(data.get("short_unit_ratio") or defaults.short_unit_ratio),
            jitter_ratio=float(data.get("jitter_ratio") or defaults.jitter_ratio),
            rate_change_ratio=float(
                data.get("rate_change_ratio") or defaults.rate_change_ratio
            ),
            range_spread_factor=float(
                data.get("range_spread_factor") or defaults.range_spread_factor
            ),
            ranges={
                str(name): [float(low), float(high)]
                for name, (low, high) in ranges.items()
            },
        )


# --- reading one chunk ---------------------------------------------------


_FLOAT = None  # set on first use; polars is imported lazily everywhere here


def _float_dtype():
    global _FLOAT
    if _FLOAT is None:
        import polars as pl

        _FLOAT = pl.Float64
    return _FLOAT


def _to_float_array(array) -> Any:
    """One pyarrow column as a float64 numpy array, nulls as NaN.

    Nulls and NaN are treated as the same thing here for run detection —
    both mean "no usable value at this instant", which is what a coverage
    strip is about. The counts keep them apart where it matters.
    """
    import numpy as np

    values = array.to_numpy(zero_copy_only=False)
    if values.dtype.kind in "mM":  # datetime / timedelta
        values = values.astype("datetime64[ns]").astype("int64").astype("float64") / 1e9
    elif values.dtype == object:
        values = np.array([float("nan") if v is None else float(v) for v in values])
    else:
        values = values.astype("float64", copy=False)
    return values


# Values kept per channel per chunk to estimate p0.1/p99.9 from. Bounded,
# and taken by a stride that doubles as the chunk grows, so the sample stays
# uniform over the whole chunk rather than being its first N rows — and
# stays deterministic, which a cached statistic has to be.
SAMPLE_MAX = 20_000


class _Sampler:
    """A bounded, deterministic, uniform sample of a column.

    Keeps every value until SAMPLE_MAX, then halves what it holds and takes
    every other value from then on, repeatedly. No randomness: the same
    chunk always yields the same sample, which matters because the result
    is cached and compared against thresholds.
    """

    def __init__(self, limit: int = SAMPLE_MAX) -> None:
        self.limit = limit
        self.stride = 1
        self.offset = 0
        self.kept: list = []

    def feed(self, values) -> None:
        import numpy as np

        if len(values) == 0:
            return
        taken = values[self.offset :: self.stride]
        self.offset = (self.offset - len(values)) % self.stride
        self.kept.append(taken)
        held = sum(len(part) for part in self.kept)
        while held > self.limit:
            self.stride *= 2
            self.kept = [part[::2] for part in self.kept]
            held = sum(len(part) for part in self.kept)

    def quantiles(self) -> tuple[float | None, float | None]:
        import numpy as np

        if not self.kept:
            return None, None
        values = np.concatenate(self.kept) if len(self.kept) > 1 else self.kept[0]
        if len(values) == 0:
            return None, None
        return float(np.quantile(values, 0.001)), float(np.quantile(values, 0.999))


@dataclass
class _RunTracker:
    """Longest run of a predicate, carried across batches and chunks.

    Leading and trailing runs are tracked separately from the longest
    interior one because a run touching an edge might continue into the
    next batch or the next chunk — and joining those at the merge is what
    makes a per-chunk result mergeable at all.
    """

    longest: int = 0
    longest_at: float | None = None
    leading: int = 0
    leading_open: bool = True
    trailing: int = 0
    current: int = 0
    current_at: float | None = None

    def feed(self, flags, stamps) -> None:
        import numpy as np

        if len(flags) == 0:
            return
        marks = flags.astype("int8")
        changes = np.flatnonzero(np.diff(marks))
        starts = np.concatenate(([0], changes + 1))
        ends = np.concatenate((changes + 1, [len(flags)]))

        for start, end in zip(starts, ends):
            length = int(end - start)
            if not flags[start]:
                self._close()
                self.leading_open = False
                continue
            if self.current == 0:
                self.current_at = float(stamps[start]) if stamps is not None else None
            self.current += length
            if self.leading_open:
                self.leading += length
        # A run still open at the end of a batch may continue into the next
        # one, so it is deliberately NOT closed here.

    def _close(self) -> None:
        if self.current > self.longest:
            self.longest = self.current
            self.longest_at = self.current_at
        self.current = 0
        self.current_at = None

    def finish(self) -> None:
        """Called once per chunk. The trailing run is recorded before it is
        closed, so the merge can join it to the next chunk's leading run."""
        self.trailing = self.current
        self._close()


def analyse_chunk(
    path: str | Path,
    timestamp_column: str,
    channels: list[str],
    thresholds: Thresholds | None = None,
) -> dict[str, Any]:
    """Everything time-ordered about one chunk, from bounded reads.

    Two kinds of pass, both one column at a time:

      the TIMESTAMP column, giving row count, first/last, backwards steps,
      duplicate timestamps, a dt histogram, the largest dts (gap
      candidates) and the time buckets the charts are drawn from;

      each CHANNEL, giving NaN and flatline runs with their locations,
      saturation counts at the chunk's own extremes, a bounded sample for
      the percentile heuristic, and — when a data dictionary is configured
      — an exact out-of-range count.

    The result is JSON-safe and small: a fixed-size histogram, bounded
    lists, and a scalar per question. It is what gets cached per chunk.

    Non-parquet sources take a different reader — see _iter_columns — but
    the same state machines, so there is one implementation of every
    question rather than one per file format.
    """
    thresholds = thresholds or Thresholds()
    path = Path(path)
    available = _column_names(path)
    if timestamp_column not in available:
        return {"rows": 0, "channels": {}}

    stamps_state = _TimestampState()
    for arrays in _iter_columns(path, [timestamp_column]):
        stamps_state.feed(arrays[0])
    result = stamps_state.finish()

    channel_results: dict[str, Any] = {}
    for name in channels:
        if name not in available or name == timestamp_column:
            continue
        channel_results[name] = _analyse_channel(path, name, timestamp_column, thresholds)
    result["channels"] = channel_results
    return result


def _column_names(path: Path) -> set[str]:
    if path.suffix.lower() == ".parquet":
        import pyarrow.parquet as pq

        return set(pq.ParquetFile(path).schema_arrow.names)
    from ml_data_view.data_loader import _dispatch_scanner

    return set(_dispatch_scanner(path, path.suffix.lower()).collect_schema())


def _iter_columns(path: Path, columns: list[str]):
    """Yield the requested columns as float arrays, in bounded pieces.

    PARQUET reads through `iter_batches`, which splits WITHIN a row group,
    so peak memory is BATCH_ROWS x len(columns) however large the file is.

    ANYTHING ELSE — a CSV or arrow file opened directly — has no such
    seekable structure, so it is read in ONE streaming pass over just these
    columns. Memory is then bounded by those columns rather than by a
    batch, which is a real difference and the reason it is stated here: a
    series of chunks is parquet-only (hierarchy.CHUNK_SUFFIX), so this path
    is only ever reached for a single file someone opened on purpose, not
    for a folder of hundreds.
    """
    if path.suffix.lower() == ".parquet":
        import pyarrow.parquet as pq

        parquet = pq.ParquetFile(path)
        for batch in parquet.iter_batches(batch_size=BATCH_ROWS, columns=columns):
            yield [_to_float_array(batch.column(index)) for index in range(len(columns))]
        return

    import numpy as np

    from ml_data_view.data_loader import _dispatch_scanner

    frame = (
        _dispatch_scanner(path, path.suffix.lower())
        .select(columns)
        .collect(engine="streaming")
    )
    yield [
        np.asarray(
            frame[name].cast(_float_dtype(), strict=False).to_numpy(), dtype="float64"
        )
        for name in columns
    ]


class _TimestampState:
    """Incremental statistics over one chunk's timestamp column."""

    def __init__(self) -> None:
        self.rows = 0
        self.first: float | None = None
        self.last: float | None = None
        self.backwards = 0
        self.duplicates = 0
        self.worst_duplicate_run = 0
        self.dt_bins = [0] * DT_BIN_COUNT
        self.large: list[tuple[float, float]] = []
        self._buckets: list[tuple] = []

    def feed(self, values) -> None:
        import numpy as np

        if len(values) == 0:
            return
        if self.first is None:
            self.first = float(values[0])
        previous = self.last
        self.last = float(values[-1])
        self.rows += len(values)

        # The dt ACROSS a batch boundary counts too: the seam is one the
        # reader chose, not one the data did.
        series = values if previous is None else np.concatenate(([previous], values))
        deltas = np.diff(series)
        if len(deltas) == 0:
            self._add_buckets(values, np.zeros(len(values)))
            return

        self.backwards += int(np.count_nonzero(deltas < 0))
        zeros = deltas == 0
        self.duplicates += int(np.count_nonzero(zeros))
        self.worst_duplicate_run = max(self.worst_duplicate_run, _longest_true_run(zeros))

        positive = deltas > 0
        if np.any(positive):
            orders = np.log10(deltas[positive])
            indices = np.clip(
                ((orders - DT_DECADES[0]) * DT_BINS_PER_DECADE).astype("int64"),
                0,
                DT_BIN_COUNT - 1,
            )
            for index, count in enumerate(np.bincount(indices, minlength=DT_BIN_COUNT)):
                if count:
                    self.dt_bins[index] += int(count)

            # Gap candidates: the largest dts and where each started.
            starts = series[:-1]
            keep = min(MAX_LARGE_DTS, len(deltas))
            order = (
                np.argpartition(-deltas, keep - 1)[:keep] if keep < len(deltas) else range(len(deltas))
            )
            for position in order:
                if deltas[position] > 0:
                    self.large.append((float(starts[position]), float(deltas[position])))
            self.large.sort(key=lambda item: -item[1])
            del self.large[MAX_LARGE_DTS:]

        # One dt per row, aligned to `values`, so a bucket describes the
        # rows it actually covers.
        row_deltas = deltas if len(deltas) == len(values) else np.concatenate(([0.0], deltas))
        self._add_buckets(values, row_deltas)

    def _add_buckets(self, values, row_deltas) -> None:
        """Downsampled aggregates: the frontend never sees a raw row."""
        import numpy as np

        rows = len(values)
        if rows == 0:
            return
        edges = np.linspace(0, rows, min(BUCKETS_PER_CHUNK, rows) + 1).astype("int64")
        for start, end in zip(edges[:-1], edges[1:]):
            if end <= start:
                continue
            window = row_deltas[start:end]
            positive = window[window > 0]
            median = float(np.median(positive)) if len(positive) else 0.0
            largest = float(np.max(positive)) if len(positive) else 0.0
            self._buckets.append(
                (float(values[start]), float(values[end - 1]), int(end - start), [median, largest])
            )

    def finish(self) -> dict[str, Any]:
        buckets = _fold_buckets(self._buckets, BUCKETS_PER_CHUNK)
        return {
            "rows": self.rows,
            "first_ts": self.first,
            "last_ts": self.last,
            "backwards": self.backwards,
            "duplicates": self.duplicates,
            "worst_duplicate_run": self.worst_duplicate_run,
            "dt_bins": self.dt_bins,
            "large_dts": [[start, delta] for start, delta in self.large],
            "buckets": [
                {"t0": t0, "t1": t1, "rows": rows, "dt_median": stats[0], "dt_max": stats[1]}
                for t0, t1, rows, stats in buckets
            ],
        }


def _fold_buckets(rows: list[tuple], limit: int) -> list[tuple]:
    """Fold adjacent buckets until there are at most `limit`.

    Row-weighted, so folding never invents or loses rows — a merged bucket
    describes exactly the rows its parts did.
    """
    if len(rows) <= limit:
        return rows
    factor = math.ceil(len(rows) / limit)
    folded = []
    for start in range(0, len(rows), factor):
        group = rows[start : start + factor]
        total = sum(item[2] for item in group) or 1
        median = sum(item[3][0] * item[2] for item in group) / total
        largest = max(item[3][1] for item in group)
        folded.append(
            (group[0][0], group[-1][1], sum(item[2] for item in group), [median, largest])
        )
    return folded


class _ChannelBuckets:
    """Downsampled min/max/mean of one channel, per time bucket.

    The same shape and the same reasoning as _TimestampState._add_buckets:
    row-equal slices within each batch, folded down to BUCKETS_PER_CHUNK
    at the end, so a chunk of 200 million rows contributes exactly as many
    numbers as one of 200. It costs nothing to collect — the channel pass
    is already reading these values with their timestamps — and it is what
    lets the timeline viewer draw a whole recording without reading it.

    Stored as six parallel ARRAYS rather than a list of dicts: this lands
    in the profile cache as JSON, where repeating six key names per bucket
    per channel per chunk is most of the bytes.
    """

    __slots__ = ("t0", "t1", "n", "min", "max", "sum")

    def __init__(self) -> None:
        self.t0: list[float] = []
        self.t1: list[float] = []
        self.n: list[int] = []
        self.min: list[float | None] = []
        self.max: list[float | None] = []
        self.sum: list[float] = []

    def feed(self, values, stamps, bad) -> None:
        import numpy as np

        rows = len(values)
        if rows == 0 or stamps is None or len(stamps) != rows:
            return
        edges = np.linspace(0, rows, min(BUCKETS_PER_CHUNK, rows) + 1).astype("int64")
        for start, end in zip(edges[:-1], edges[1:]):
            if end <= start:
                continue
            window = values[start:end]
            clean = window[~bad[start:end]]
            self.t0.append(float(stamps[start]))
            self.t1.append(float(stamps[end - 1]))
            self.n.append(int(len(clean)))
            self.min.append(float(np.min(clean)) if len(clean) else None)
            self.max.append(float(np.max(clean)) if len(clean) else None)
            self.sum.append(float(np.sum(clean)) if len(clean) else 0.0)

    def finish(self) -> dict[str, list]:
        folded = fold_channel_buckets(
            list(zip(self.t0, self.t1, self.n, self.min, self.max, self.sum)),
            BUCKETS_PER_CHUNK,
        )
        return {
            "t0": [row[0] for row in folded],
            "t1": [row[1] for row in folded],
            "n": [row[2] for row in folded],
            "min": [row[3] for row in folded],
            "max": [row[4] for row in folded],
            "sum": [row[5] for row in folded],
        }


def fold_channel_buckets(rows: list[tuple], limit: int) -> list[tuple]:
    """Fold adjacent channel buckets until there are at most `limit`.

    Exact for every field it carries: a merged bucket's min is the min of
    its parts', its count and sum are theirs added, so the mean read off
    it is the true mean of exactly the rows it covers. That is what makes
    the cached per-chunk buckets a legitimate answer at a coarse zoom
    rather than an approximation of one — the only thing lost by folding
    is resolution, never accuracy.
    """
    if len(rows) <= limit:
        return list(rows)
    factor = math.ceil(len(rows) / limit)
    folded = []
    for start in range(0, len(rows), factor):
        group = rows[start : start + factor]
        minima = [row[3] for row in group if row[3] is not None]
        maxima = [row[4] for row in group if row[4] is not None]
        folded.append(
            (
                group[0][0],
                group[-1][1],
                sum(row[2] for row in group),
                min(minima) if minima else None,
                max(maxima) if maxima else None,
                sum(row[5] for row in group),
            )
        )
    return folded


def _longest_true_run(flags) -> int:
    import numpy as np

    if len(flags) == 0:
        return 0
    padded = np.concatenate(([False], flags, [False])).astype("int8")
    changes = np.flatnonzero(np.diff(padded))
    if len(changes) < 2:
        return 0
    return int(np.max(changes[1::2] - changes[0::2]))


def _stamp_of(values, stamps, target: float) -> float | None:
    """The timestamp of the first row holding `target`, within one batch."""
    import numpy as np

    if stamps is None or len(stamps) != len(values):
        return None
    where = np.flatnonzero(values == target)
    return float(stamps[where[0]]) if len(where) else None


def _analyse_channel(
    path: Path, name: str, timestamp_column: str, thresholds: Thresholds
) -> dict[str, Any]:
    """One channel's runs, extremes and saturation, from bounded reads.

    Read WITH the timestamp column so every run can say WHERE it was — two
    columns per pass rather than one, still a long way from the
    all-columns-at-once pattern that measured 1,154MB against 623MB per
    column.
    """
    import numpy as np

    missing = _RunTracker()
    flat = _RunTracker()
    sampler = _Sampler()
    buckets = _ChannelBuckets()
    nulls = 0
    not_a_number = 0
    minimum = math.inf
    maximum = -math.inf
    min_count = 0
    max_count = 0
    # WHEN each extreme was reached. Two numbers, and the only thing that
    # lets a finding about an extreme value point at a place on a
    # timeline — a count says a spike happened, not where to look.
    min_at: float | None = None
    max_at: float | None = None
    out_of_range = 0
    total = 0
    first_value: float | None = None
    last_value: float | None = None
    previous: float | None = None
    configured = thresholds.ranges.get(name)

    for values, stamps in _iter_columns(path, [name, timestamp_column]):
        if len(values) == 0:
            continue
        # A null arrives as NaN from _to_float_array, so the two are
        # counted apart here only where the distinction is available.
        nulls += int(np.count_nonzero(np.isnan(values)))
        total += len(values)
        if first_value is None:
            first_value = float(values[0])
        last_value = float(values[-1])

        # A null and a NaN are both "no usable value at this instant",
        # which is what a coverage strip is about — so runs treat them the
        # same and the counts keep them apart.
        bad = np.isnan(values)
        not_a_number += int(np.count_nonzero(bad))
        missing.feed(bad, stamps)
        buckets.feed(values, stamps, bad)

        # Flatline: identical CONSECUTIVE values, compared back into the
        # previous batch so a run is never cut in half by a read boundary.
        same = np.zeros(len(values), dtype=bool)
        if previous is None:
            same[1:] = (values[1:] == values[:-1]) & ~bad[1:]
        else:
            series = np.concatenate(([previous], values))
            same[:] = (series[1:] == series[:-1]) & ~bad
        flat.feed(same, stamps)
        tail = values[-1]
        previous = float(tail) if not math.isnan(tail) else previous

        clean = values[~bad]
        if len(clean):
            sampler.feed(clean)
            batch_min, batch_max = float(np.min(clean)), float(np.max(clean))
            # Counts are kept only for the extreme seen SO FAR: a lower
            # minimum resets the count, because no earlier value equalled
            # it. That is what makes the merge exact.
            if batch_min < minimum:
                minimum, min_count = batch_min, 0
                min_at = _stamp_of(values, stamps, batch_min)
            if batch_max > maximum:
                maximum, max_count = batch_max, 0
                max_at = _stamp_of(values, stamps, batch_max)
            if batch_min == minimum:
                min_count += int(np.count_nonzero(clean == minimum))
            if batch_max == maximum:
                max_count += int(np.count_nonzero(clean == maximum))
            if configured:
                low, high = configured
                out_of_range += int(np.count_nonzero((clean < low) | (clean > high)))

    missing.finish()
    flat.finish()
    low_q, high_q = sampler.quantiles()

    segments: list[dict[str, Any]] = []
    if missing.longest:
        segments.append({"kind": "nan", "at": missing.longest_at, "length": missing.longest})
    if flat.longest:
        segments.append({"kind": "flat", "at": flat.longest_at, "length": flat.longest})

    return {
        "rows": total,
        # `nan` is nulls AND NaN together — the number the coverage strip
        # draws; `null` is the subset that were absent rather than broken.
        "nan": not_a_number,
        "null": nulls,
        "longest_nan": missing.longest,
        "longest_nan_at": missing.longest_at,
        "lead_nan": missing.leading,
        "trail_nan": missing.trailing,
        "longest_flat": flat.longest,
        "longest_flat_at": flat.longest_at,
        "lead_flat": flat.leading,
        "trail_flat": flat.trailing,
        "first_value": first_value,
        "last_value": last_value,
        "min": None if minimum == math.inf else minimum,
        "max": None if maximum == -math.inf else maximum,
        "min_count": min_count,
        "max_count": max_count,
        "min_at": min_at,
        "max_at": max_at,
        "out_of_range": out_of_range,
        "p001": low_q,
        "p999": high_q,
        "segments": segments[:MAX_SEGMENTS],
        # The timeline viewer's whole overview, and free: this pass was
        # already reading these values beside their timestamps.
        "buckets": buckets.finish(),
    }


# --- merging chunks into one unit ----------------------------------------


def merge_unit(
    chunks: list[dict[str, Any]],
    names: list[str],
    thresholds: Thresholds | None = None,
) -> dict[str, Any]:
    """One unit's health, merged from its chunks IN TIME ORDER.

    `chunks` are the per-chunk artifacts; `names` their filenames, used
    only to say where something was. Order matters and is the caller's
    (hierarchy.py already ordered them, by timestamp where filenames
    disagreed).

    Three kinds of answer come out of here:

      SUMS — rows, NaN counts, duplicate counts. Trivially mergeable.

      HISTOGRAM-DERIVED — the median dt and its spread, read off the
      summed dt histogram. Summing bins is exact; reading a quantile off
      them is bin-resolution, which is far finer than any threshold here.

      BOUNDARY-ONLY — a gap that falls exactly BETWEEN two chunks, an
      overlap, a backwards step from one chunk to the next, a NaN run that
      spans a chunk seam. None of these exist inside any chunk, and none
      needs a re-read: they are arithmetic on neighbouring chunks'
      first/last values, which is the whole reason those are cached.
    """
    thresholds = thresholds or Thresholds()
    usable = [(chunk, name) for chunk, name in zip(chunks, names) if chunk.get("rows")]
    if not usable:
        return {"rows": 0, "channels": {}, "gaps": [], "overlaps": [], "chunks": []}

    rows = sum(chunk["rows"] for chunk, _ in usable)
    first_ts = usable[0][0].get("first_ts")
    last_ts = usable[-1][0].get("last_ts")

    dt_bins = [0] * DT_BIN_COUNT
    for chunk, _name in usable:
        for index, count in enumerate(chunk.get("dt_bins") or []):
            dt_bins[index] += int(count)

    backwards = sum(chunk.get("backwards", 0) for chunk, _ in usable)
    duplicates = sum(chunk.get("duplicates", 0) for chunk, _ in usable)
    worst_duplicate_run = max(
        (chunk.get("worst_duplicate_run", 0) for chunk, _ in usable), default=0
    )

    # --- the seams ---
    boundary_dts: list[tuple[float, float, str, str]] = []
    overlaps: list[dict[str, Any]] = []
    for (previous, previous_name), (current, current_name) in zip(usable, usable[1:]):
        end = previous.get("last_ts")
        start = current.get("first_ts")
        if end is None or start is None:
            continue
        delta = start - end
        if delta < 0:
            backwards += 1
            overlaps.append(
                {
                    "after_chunk": previous_name,
                    "next_chunk": current_name,
                    "overlap": abs(delta),
                    "at": start,
                }
            )
        elif delta == 0:
            duplicates += 1
        else:
            dt_bins[dt_bin_index(delta)] += 1
            boundary_dts.append((end, delta, previous_name, current_name))

    median = histogram_quantile(dt_bins, 0.5)
    q25 = histogram_quantile(dt_bins, 0.25)
    q75 = histogram_quantile(dt_bins, 0.75)
    # Half the interquartile range: a robust spread, which is what the
    # brief asks for and what a handful of gaps cannot distort the way a
    # standard deviation would.
    spread = (q75 - q25) / 2 if (q25 is not None and q75 is not None) else None
    jitter = (spread / median) if (spread is not None and median) else None

    # --- gaps: the largest dts, judged against the MERGED median ---
    gaps: list[dict[str, Any]] = []
    if median:
        limit = median * thresholds.gap_factor
        boundary_at = {round(start, 9): (after, nxt) for start, _d, after, nxt in boundary_dts}
        candidates: list[tuple[float, float]] = []
        for chunk, _name in usable:
            candidates.extend(
                (float(start), float(delta)) for start, delta in (chunk.get("large_dts") or [])
            )
        candidates.extend((start, delta) for start, delta, _a, _b in boundary_dts)
        for start, delta in sorted(set(candidates)):
            if delta <= limit:
                continue
            seam = boundary_at.get(round(start, 9))
            gaps.append(
                {
                    "at": start,
                    "duration": delta,
                    # A gap that lands exactly on a chunk seam is a
                    # different claim from one inside a chunk: the rows are
                    # not late, a whole file is probably absent.
                    "at_boundary": seam is not None,
                    "after_chunk": seam[0] if seam else None,
                    "next_chunk": seam[1] if seam else None,
                }
            )
    gaps.sort(key=lambda item: -item["duration"])
    gap_total = sum(gap["duration"] for gap in gaps)
    duration = (last_ts - first_ts) if (first_ts is not None and last_ts is not None) else None

    buckets = _fold_buckets(
        [
            (b["t0"], b["t1"], b["rows"], [b["dt_median"], b["dt_max"]])
            for chunk, _ in usable
            for b in (chunk.get("buckets") or [])
        ],
        MAX_MERGED_BUCKETS,
    )

    return {
        "rows": rows,
        "first_ts": first_ts,
        "last_ts": last_ts,
        "duration": duration,
        "backwards": backwards,
        "duplicates": duplicates,
        "worst_duplicate_run": worst_duplicate_run,
        "dt_median": median,
        "dt_spread": spread,
        "jitter": jitter,
        "hz": (1.0 / median) if median else None,
        "gaps": gaps[:MAX_LARGE_DTS],
        "gap_total": gap_total,
        "gap_share": (gap_total / duration) if duration else None,
        "overlaps": overlaps,
        "rate_segments": _rate_segments(buckets, median, thresholds.rate_change_ratio),
        "buckets": [
            {"t0": t0, "t1": t1, "rows": n, "dt_median": stats[0], "dt_max": stats[1]}
            for t0, t1, n, stats in buckets
        ],
        "chunks": [
            {
                "name": name,
                "rows": chunk["rows"],
                "first_ts": chunk.get("first_ts"),
                "last_ts": chunk.get("last_ts"),
            }
            for chunk, name in usable
        ],
        "channels": _merge_channels(usable, thresholds),
    }


def _rate_segments(
    buckets: list[tuple], median: float | None, ratio: float
) -> list[dict[str, Any]]:
    """Contiguous stretches whose sample rate differs from the unit's.

    Read off the downsampled buckets rather than the rows — which is the
    only thing that makes this affordable, and is also the right
    resolution: a rate CHANGE lasts long enough to span buckets, while
    anything shorter is jitter and is reported as jitter.
    """
    if not median or not buckets:
        return []
    segments: list[dict[str, Any]] = []
    current: dict[str, Any] | None = None
    for t0, t1, rows, stats in buckets:
        bucket_median = stats[0]
        off = bool(bucket_median) and abs(bucket_median - median) / median > ratio
        if off and current is None:
            current = {"t0": t0, "t1": t1, "rows": rows, "dt_median": bucket_median}
        elif off:
            current["t1"] = t1
            current["rows"] += rows
            current["dt_median"] = (current["dt_median"] + bucket_median) / 2
        elif current is not None:
            segments.append(current)
            current = None
    if current is not None:
        segments.append(current)
    return segments


def _merge_channels(
    usable: list[tuple[dict[str, Any], str]], thresholds: Thresholds
) -> dict[str, Any]:
    """Per-channel merge, joining runs across chunk seams."""
    names: list[str] = []
    for chunk, _name in usable:
        for name in (chunk.get("channels") or {}):
            if name not in names:
                names.append(name)

    merged: dict[str, Any] = {}
    for name in names:
        parts = [
            (chunk["channels"][name], chunk, chunk_name)
            for chunk, chunk_name in usable
            if name in (chunk.get("channels") or {})
        ]
        if not parts:
            continue
        rows = sum(part["rows"] for part, _c, _n in parts)
        nan = sum(part["nan"] for part, _c, _n in parts)
        nulls = sum(part["null"] for part, _c, _n in parts)
        out_of_range = sum(part.get("out_of_range", 0) for part, _c, _n in parts)

        minima = [part["min"] for part, _c, _n in parts if part["min"] is not None]
        maxima = [part["max"] for part, _c, _n in parts if part["max"] is not None]
        minimum = min(minima) if minima else None
        maximum = max(maxima) if maxima else None
        # EXACT: a chunk whose own minimum is above the global one holds no
        # value equal to the global one, so it contributes nothing.
        min_count = sum(
            part["min_count"] for part, _c, _n in parts if part["min"] == minimum
        )
        max_count = sum(
            part["max_count"] for part, _c, _n in parts if part["max"] == maximum
        )
        # The FIRST chunk that reached the extreme is where it happened —
        # later chunks that tie it hold the same value at a later time,
        # and a marker has to point at one place.
        min_at = next(
            (part.get("min_at") for part, _c, _n in parts if part["min"] == minimum), None
        )
        max_at = next(
            (part.get("max_at") for part, _c, _n in parts if part["max"] == maximum), None
        )

        weighted = [(part.get("p001"), part.get("p999"), part["rows"]) for part, _c, _n in parts]
        total = sum(w for _l, _h, w in weighted if _l is not None) or 0
        p001 = (
            sum(low * w for low, _h, w in weighted if low is not None) / total if total else None
        )
        p999 = (
            sum(high * w for _l, high, w in weighted if high is not None) / total
            if total
            else None
        )

        longest_nan, nan_at = _join_runs(parts, "nan")
        longest_flat, flat_at = _join_runs(parts, "flat")

        merged[name] = {
            "rows": rows,
            "nan": nan,
            "null": nulls,
            "nan_share": (nan / rows) if rows else 0.0,
            "longest_nan": longest_nan,
            "longest_nan_at": nan_at,
            "longest_flat": longest_flat,
            "longest_flat_at": flat_at,
            "min": minimum,
            "max": maximum,
            "min_count": min_count,
            "max_count": max_count,
            "min_at": min_at,
            "max_at": max_at,
            "min_share": (min_count / rows) if rows else 0.0,
            "max_share": (max_count / rows) if rows else 0.0,
            "out_of_range": out_of_range,
            "p001": p001,
            "p999": p999,
            "segments": [
                dict(segment, chunk=chunk_name)
                for part, _c, chunk_name in parts
                for segment in (part.get("segments") or [])
            ][:MAX_SEGMENTS],
        }
    return merged


def _join_runs(parts: list[tuple], kind: str) -> tuple[int, float | None]:
    """The longest run of `kind`, with runs spanning chunk seams joined.

    A NaN run that ends a chunk and continues at the start of the next is
    ONE run; reporting two shorter ones would understate exactly the case
    that matters, since a sensor that dies rarely does it inside one file.
    A flatline additionally requires the values on either side of the seam
    to actually match — two chunks stuck at different values are two
    flatlines, not one.
    """
    longest_key = f"longest_{kind}"
    at_key = f"longest_{kind}_at"
    lead_key = f"lead_{kind}"
    trail_key = f"trail_{kind}"

    best = 0
    best_at: float | None = None
    running = 0
    running_at: float | None = None

    for part, chunk, _name in parts:
        rows = part["rows"]
        lead = part.get(lead_key, 0)
        joins = running > 0 and lead > 0
        if joins and kind == "flat":
            joins = part.get("first_value") is not None and part.get("first_value") == _previous_value(
                parts, part
            )
        if joins:
            candidate = running + lead
            if candidate > best:
                best, best_at = candidate, running_at
        if part.get(longest_key, 0) > best:
            best, best_at = part[longest_key], part.get(at_key)
        if lead == rows and rows:
            # The whole chunk is one run, so it may continue further still.
            running = running + rows if running else rows
            if running_at is None:
                running_at = chunk.get("first_ts")
            continue
        running = part.get(trail_key, 0)
        running_at = chunk.get("last_ts") if running else None

    if running > best:
        best, best_at = running, running_at
    return best, best_at


def _previous_value(parts: list[tuple], part: dict[str, Any]) -> Any:
    """The last value of the chunk before `part`'s, for the flatline join."""
    for index, (candidate, _chunk, _name) in enumerate(parts):
        if candidate is part and index:
            return parts[index - 1][0].get("last_value")
    return None


# --- what the charts draw ------------------------------------------------
#
# Built here rather than in JS for the reason every other payload in this
# app is: the rules about what counts as a gap, and which gap is a missing
# chunk, are the content — and a second copy of them in the frontend is
# how the chart and the finding beside it end up disagreeing. The frontend
# receives segments with a kind, and draws them.


def coverage_strip(unit: dict[str, Any]) -> list[dict[str, Any]]:
    """The unit's time axis, cut into covered and uncovered stretches.

    Three kinds. DATA is where chunks actually span. GAP is a stretch
    inside a chunk with no rows. MISSING is a gap that falls exactly on a
    chunk boundary — the same distinction the findings make, drawn in the
    same colours, so the strip and the tab cannot tell different stories.

    Bounded by construction: one segment per chunk plus one per gap, and
    gaps are already capped.
    """
    first, last = unit.get("first_ts"), unit.get("last_ts")
    if first is None or last is None:
        return []

    cuts: list[tuple[float, float, str]] = []
    for gap in unit.get("gaps") or []:
        start = float(gap["at"])
        cuts.append(
            (start, start + float(gap["duration"]), "missing" if gap.get("at_boundary") else "gap")
        )
    cuts.sort()

    segments: list[dict[str, Any]] = []
    cursor = float(first)
    for start, end, kind in cuts:
        if start > cursor:
            segments.append({"t0": cursor, "t1": start, "kind": "data"})
        segments.append({"t0": max(start, cursor), "t1": end, "kind": kind})
        cursor = max(cursor, end)
    if cursor < last:
        segments.append({"t0": cursor, "t1": float(last), "kind": "data"})
    return segments


def channel_strips(unit: dict[str, Any], limit: int = 8) -> list[dict[str, Any]]:
    """Per-channel coverage: where each channel was missing or stuck.

    One row per channel, capped — a strip with forty rows is a wall, not a
    chart, and the channels worth looking at are the ones with something
    on them, so they are ordered by how much.
    """
    rows = []
    for name, channel in (unit.get("channels") or {}).items():
        segments = [
            {
                "t0": segment.get("at"),
                # Segment lengths are in ROWS; converted to a time span
                # with the unit's own median interval, which is the only
                # honest conversion available and is stated as such in the
                # chart's note.
                "t1": (segment.get("at") or 0) + (segment.get("length") or 0) * (unit.get("dt_median") or 0),
                "kind": segment.get("kind"),
            }
            for segment in (channel.get("segments") or [])
            if segment.get("at") is not None
        ]
        if not segments:
            continue
        rows.append(
            {
                "column": name,
                "segments": segments,
                "nan_share": channel.get("nan_share") or 0.0,
                "worst": max(
                    (channel.get("longest_nan") or 0, channel.get("longest_flat") or 0)
                ),
            }
        )
    rows.sort(key=lambda row: -row["worst"])
    return rows[:limit]


def unit_charts(unit: dict[str, Any]) -> dict[str, Any]:
    """Everything the Health tab draws for one unit, already downsampled.

    Never a raw row: the dt series is the merged buckets (min/max/median
    per bucket), and the strips are segments. A recording of 200 million
    rows produces the same few hundred numbers as one of 200.
    """
    return {
        "coverage": coverage_strip(unit),
        "dt": [
            {"t": bucket["t0"], "dt_median": bucket["dt_median"], "dt_max": bucket["dt_max"]}
            for bucket in (unit.get("buckets") or [])
        ],
        "channels": channel_strips(unit),
        "chunks": unit.get("chunks") or [],
        "summary": {
            "rows": unit.get("rows"),
            "duration": unit.get("duration"),
            "hz": unit.get("hz"),
            "dt_median": unit.get("dt_median"),
            "jitter": unit.get("jitter"),
            "gap_total": unit.get("gap_total"),
            "gap_share": unit.get("gap_share"),
            "gap_count": len(unit.get("gaps") or []),
            "missing_chunks": sum(1 for gap in (unit.get("gaps") or []) if gap.get("at_boundary")),
            "overlaps": len(unit.get("overlaps") or []),
            "backwards": unit.get("backwards"),
            "duplicates": unit.get("duplicates"),
        },
    }


def out_of_range_extremes(
    channel: dict[str, Any], documented: Any, factor: float
) -> list[dict[str, Any]]:
    """Which of a channel's extremes sit outside its range, and by which rule.

    ONE implementation of the two rules, read by the check that reports
    them and by the timeline that marks them. A second copy would be how
    the tooltip on a spike and the finding about the same spike end up
    disagreeing about whether it is one.

    With a DOCUMENTED range the verdict is exact and needs no floor: it is
    a comparison against a stated rule. WITHOUT one it is a heuristic on
    how far the extreme sits beyond the inter-percentile spread, and it
    stays silent below MIN_ROWS_FOR_RANGE_HEURISTIC rows, where p99.9
    lands on the outlier itself and the rule cannot see what it is looking
    for.

    Each entry carries `at` — when the extreme was reached — which is what
    lets a finding about a value point at a place on a time axis.
    """
    minimum, maximum = channel.get("min"), channel.get("max")
    edges = (
        ("maximum", maximum, channel.get("max_at")),
        ("minimum", minimum, channel.get("min_at")),
    )
    if documented:
        low, high = documented
        return [
            {"edge": edge, "value": value, "at": at, "rule": "documented", "distance": None}
            for edge, value, at in edges
            if value is not None and not (low <= value <= high)
        ]

    if (channel.get("rows") or 0) < MIN_ROWS_FOR_RANGE_HEURISTIC:
        return []
    low, high = channel.get("p001"), channel.get("p999")
    if None in (low, high, minimum, maximum):
        return []
    spread = high - low
    if spread <= 0:
        return []
    found = []
    if maximum > high + spread * factor:
        found.append(
            {
                "edge": "maximum",
                "value": maximum,
                "at": channel.get("max_at"),
                "rule": "heuristic",
                "distance": abs(maximum - high) / spread,
                "bound": high,
            }
        )
    if minimum < low - spread * factor:
        found.append(
            {
                "edge": "minimum",
                "value": minimum,
                "at": channel.get("min_at"),
                "rule": "heuristic",
                "distance": abs(minimum - low) / spread,
                "bound": low,
            }
        )
    return found
