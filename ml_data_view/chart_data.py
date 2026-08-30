"""Headless chart-data computation for Phase 1 profiling charts.

Pure functions over a polars LazyFrame — no pywebview or Plotly imports
here. Each function returns a small JSON-safe dict; the frontend (app.js)
turns that into a Plotly.js figure. Every function below collect()s only a
bounded slice or a streaming aggregation, never the full frame — see
PROJECT_BRIEF.md's architecture note and data_loader.py's module docstring.
"""
from __future__ import annotations

import math
from collections.abc import Callable
from pathlib import Path
from typing import Any

import polars as pl

from ml_data_view.data_loader import row_count
from ml_data_view.embedding_data import detect_embedding_columns, embedding_projection_data
from ml_data_view.image_data import detect_image_columns, image_previews_data
from ml_data_view.null_patterns import correlated_missingness

# Whole-dataset steps that run after the per-column loop, named so the
# progress bar can label them rather than appearing to stall.
_TRAILING_STEPS = (
    "Detecting embedding columns…",
    "Detecting image columns…",
    "Checking correlated missingness…",
    "Building missing-value heatmap…",
)

NUMERIC_HISTOGRAM_BINS = 30
TEXT_LENGTH_HISTOGRAM_BINS = 30
# Skew detection for log-scale histograms (see _is_skewed). A column whose
# full range is this many times wider than its own p50-to-p99 spread has a
# tail long enough to make equal-width bins useless. Reference points:
# uniform ~2.0, normal ~2.6 — so 10 leaves clear separation.
SKEW_RANGE_TO_SPREAD_RATIO = 10.0
# Below this range a log axis compresses nothing worth the confusion, and
# it keeps near-binary columns (mostly 0, a few 1s) on a linear axis.
LOG_SCALE_MIN_RANGE = 100.0

# Above this many distinct values, a top-N bar stops being informative — it
# names a handful of arbitrary IDs whose share of rows is a rounding error.
# Such columns get a value-frequency histogram instead (see
# value_frequency_histogram_data). Real example: a track-ID column with
# ~515k distinct values in two files, top bar ~0.3% of rows.
CATEGORICAL_BAR_MAX_DISTINCT = 200
# Per-file shortlist size for the bounded top-N path. Far above
# TOP_N_CATEGORIES so a globally-frequent value is not missed for being
# just outside one file's shortlist.
CATEGORICAL_PER_FILE_TOP_K = 200
# Detections-per-value buckets. Left-closed upper bounds, so labels are one
# longer than breaks. Log-ish because these counts pile up at 1.
VALUE_FREQUENCY_BREAKS = [1.0, 3.0, 10.0, 50.0, 200.0, 1000.0]
VALUE_FREQUENCY_LABELS = ["1", "2-3", "4-10", "11-50", "51-200", "201-1000", "1000+"]
TOP_N_CATEGORIES = 10
NULL_HEATMAP_MAX_ROWS = 500
NULL_HEATMAP_MAX_DISPLAY_ROWS = 100
CATEGORICAL_MAX_DISTINCT = 50
CATEGORICAL_MAX_DISTINCT_RATIO = 0.5

NUMERIC_DTYPE_NAMES = {
    "Int8", "Int16", "Int32", "Int64", "Int128",
    "UInt8", "UInt16", "UInt32", "UInt64",
    "Float32", "Float64",
}
STRING_DTYPE_NAMES = {"String", "Utf8"}


def _scan_filtered(file: Path, predicate: pl.Expr | None) -> pl.LazyFrame:
    """Scan one file, applying the active chart filter if there is one.

    Every per-file scan in this module routes through here rather than
    calling pl.scan_parquet directly, so a filter cannot end up applied in
    some chart computations and silently forgotten in others — which would
    put numbers describing different row sets on the same screen.
    """
    scan = pl.scan_parquet(file)
    return scan.filter(predicate) if predicate is not None else scan


def _is_missing(value: Any) -> bool:
    """True for None or float NaN.

    polars treats NaN as a real (non-null) float value distinct from null,
    so `.drop_nulls()` alone lets NaN through — but for profiling purposes
    a NaN is just as "no usable value" as a null, so every place that
    decides what counts as missing routes through this helper.
    """
    if value is None:
        return True
    return isinstance(value, float) and value != value  # NaN is the only value != itself


def classify_column(
    lf: pl.LazyFrame,
    column: str,
    files: list[Path] | None = None,
    predicate: pl.Expr | None = None,
) -> str:
    """Classify a column for charting purposes.

    Returns "numeric", "categorical", "text", or "unsupported" (no chart
    type defined yet, e.g. Date/Datetime columns). collect_schema() reads
    schema metadata only; a string column's categorical-vs-text call needs
    one small streaming aggregation (see _classify_string_column).

    `files`, when the LazyFrame is a concat of several parquet files,
    makes that aggregation run per file rather than over the concat —
    required for folders, see _classify_string_column.
    """
    return classify_column_with_cardinality(lf, column, files=files, predicate=predicate)[0]


def classify_column_with_cardinality(
    lf: pl.LazyFrame,
    column: str,
    files: list[Path] | None = None,
    predicate: pl.Expr | None = None,
) -> tuple[str, int | None]:
    """classify_column(), plus the distinct-value estimate it already had
    to compute. Returned together so build_chart_payload can choose between
    a top-N bar and a value-frequency histogram without paying for a second
    cardinality pass. None means "not applicable" (numeric/unsupported)."""
    dtype_name = str(lf.collect_schema()[column])
    if dtype_name in NUMERIC_DTYPE_NAMES:
        return "numeric", None
    if dtype_name == "Boolean":
        return "categorical", 2
    if dtype_name in STRING_DTYPE_NAMES:
        return _classify_string_column(lf, column, files=files, predicate=predicate)
    return "unsupported", None


def _classify_string_column_per_file(
    files: list[Path], column: str, predicate: pl.Expr | None = None
) -> tuple[str, int]:
    """Categorical-vs-text from per-file distinct/row ratios, never
    querying the multi-file concat.

    A single query over a concat costs ~28-42MB per file, linearly —
    measured on the real Criteo folder: 10 files 976MB, 25 files 1383MB,
    50 files 1956MB, 100 files over a 2GB cap, for ONE column. The same
    work done file-by-file ran all 250 files in 1011MB and 2.2s. So this
    scans one file at a time and combines scalars.

    Distinct counts cannot simply be summed across files (the same value
    recurs in many files), so this combines RATIOS rather than counts.
    That is sound for this heuristic because the distinct/row ratio falls
    monotonically as data grows — n grows linearly while distinct values
    saturate — whereas free text stays near 1.0 at any scale. Measured on
    categorical_feature_1: 0.233 (2 files), 0.182 (10), 0.144 (25),
    0.130 (50).

    Consequence, stated plainly: this can only ever be wrong in one
    direction — calling something "text" that the full-folder ratio would
    call "categorical". It cannot invent a "categorical" verdict for a
    genuinely free-text column.
    """
    non_null = pl.col(column).drop_nulls()
    total_rows = 0
    weighted_ratio = 0.0
    max_distinct = 0

    for file in files:
        result = (
            _scan_filtered(file, predicate)
            .select(non_null.count().alias("n"), non_null.approx_n_unique().alias("nu"))
            .collect(engine="streaming")
        )
        n, n_unique = result[0, "n"], result[0, "nu"]
        if n == 0:
            continue
        total_rows += n
        weighted_ratio += (n_unique / n) * n
        max_distinct = max(max_distinct, n_unique)

    if total_rows == 0:
        return "categorical", 0
    # A column whose per-file distinct count never exceeds the threshold is
    # low-cardinality everywhere, so it is categorical regardless of ratio.
    if max_distinct <= CATEGORICAL_MAX_DISTINCT:
        return "categorical", max_distinct
    mean_ratio = weighted_ratio / total_rows
    kind = "categorical" if mean_ratio <= CATEGORICAL_MAX_DISTINCT_RATIO else "text"
    # max_distinct is the largest PER-FILE estimate, a lower bound on the
    # folder-wide count — enough to decide bar-vs-histogram, which only
    # needs to know whether cardinality is small.
    return kind, max_distinct


def _classify_string_column(
    lf: pl.LazyFrame,
    column: str,
    files: list[Path] | None = None,
    predicate: pl.Expr | None = None,
) -> tuple[str, int]:
    # approx_n_unique (HyperLogLog), not n_unique: an exact distinct count
    # builds a hash set of every distinct value, which on a large
    # high-cardinality column is unbounded — measured at 535MB for one
    # column over two real Criteo files, versus 162MB approximate, for the
    # same classification. HLL is constant-memory, so this does not grow
    # with row or file count.
    #
    # Estimated rather than sampled deliberately: this needs the
    # FULL-data distinct/row ratio. A bounded sample of a
    # high-cardinality-but-repeating column reports a near-1.0 ratio and
    # would misclassify it as "text" (see the
    # high_cardinality_but_repeating test). The estimate is only ever used
    # against the thresholds below, which is already a heuristic.
    non_null = pl.col(column).drop_nulls()
    if files:
        return _classify_string_column_per_file(files, column, predicate)
    result = lf.select(
        non_null.count().alias("n"), non_null.approx_n_unique().alias("nu")
    ).collect(engine="streaming")
    n, n_unique = result[0, "n"], result[0, "nu"]
    if n == 0:
        return "categorical", 0
    if n_unique <= CATEGORICAL_MAX_DISTINCT or n_unique / n <= CATEGORICAL_MAX_DISTINCT_RATIO:
        return "categorical", int(n_unique)
    return "text", int(n_unique)


def _clean_numeric_expr(column: str) -> pl.Expr:
    """The column cast to Float64 with nulls and NaN both filtered out (see
    _is_missing's docstring for why NaN counts as missing here too)."""
    casted = pl.col(column).cast(pl.Float64)
    return casted.filter(casted.is_not_null() & ~casted.is_nan())


def _histogram_from_clean_expr(lf: pl.LazyFrame, clean: pl.Expr, bins: int) -> tuple[list[int], list[float]]:
    """Shared by numeric_histogram_data/text_length_histogram_data: one
    streaming query gets the non-missing count, minimum, and per-bin
    histogram (via polars' native Expr.hist(), vectorized — see
    numeric_histogram_data's docstring for why that matters at scale) all
    at once. Returns (counts, bin_edges), or ([], []) if there's no data.
    """
    result = lf.select(
        clean.count().alias("__n"),
        clean.min().alias("__min"),
        clean.hist(bin_count=bins, include_breakpoint=True).alias("__hist"),
    ).collect(engine="streaming")

    if result[0, "__n"] == 0:
        return [], []

    hist = result.unnest("__hist")
    counts = [int(c) for c in hist["count"].to_list()]
    bin_edges = [float(result[0, "__min"])] + [float(b) for b in hist["breakpoint"].to_list()]
    return counts, bin_edges


def _is_skewed(minimum: float, maximum: float, p25: float, p75: float) -> bool:
    """Whether a column's tail is long enough that equal-width bins would
    collapse nearly every row into the first bar.

    Compares the FULL range against the interquartile spread: a tail-heavy
    column has a range many times wider than the middle half of its own
    distribution. Reference values this has to separate — uniform 2.0,
    normal ~4.4 — sit well below the threshold, while a long tail sends
    the ratio to thousands or to infinity.

    Two heuristics were tried and rejected first, both worth recording:
    p99/median misses the worst case outright (for a mostly-zero column
    with a rare huge tail, p99 AND the median are both zero, so the ratio
    is 1.0), and range/(p99 - p50) silently fails as soon as the tail is
    more than 1% of rows, because p99 then lands INSIDE the tail and the
    spread balloons to match the range. The IQR is robust up to a 25%
    tail, which covers anything worth calling skewed.
    """
    spread = p75 - p25
    value_range = maximum - minimum
    if value_range < LOG_SCALE_MIN_RANGE:
        # Too narrow for a log axis to compress anything usefully — this
        # also keeps near-binary columns (mostly 0, a few 1s) on a linear
        # axis, where the range/spread test alone would flag them.
        return False
    if spread <= 0:
        return True  # bulk is a single value, everything else is tail
    return (value_range / spread) >= SKEW_RANGE_TO_SPREAD_RATIO


def _numeric_stats_per_file(
    files: list[Path], column: str, predicate: pl.Expr | None = None
) -> tuple[int, float, float, float, float] | None:
    """(count, min, max, p25, p75) across `files`, one file at a time.

    min/max/count combine exactly. The quantiles are a row-weighted mean of
    each file's own quantile — not the true global quantile, which cannot
    be merged from per-file values. That is acceptable here because they
    feed only _is_skewed's range-vs-spread heuristic, whose threshold (10)
    sits an order of magnitude away from the reference distributions.
    """
    clean = _clean_numeric_expr(column)
    total = 0
    minimum = math.inf
    maximum = -math.inf
    weighted_p25 = 0.0
    weighted_p75 = 0.0

    for file in files:
        result = (
            _scan_filtered(file, predicate)
            .select(
                clean.count().alias("n"),
                clean.min().alias("mn"),
                clean.max().alias("mx"),
                clean.quantile(0.25).alias("p25"),
                clean.quantile(0.75).alias("p75"),
            )
            .collect(engine="streaming")
        )
        n = int(result[0, "n"])
        if n == 0:
            continue
        total += n
        minimum = min(minimum, float(result[0, "mn"]))
        maximum = max(maximum, float(result[0, "mx"]))
        weighted_p25 += float(result[0, "p25"]) * n
        weighted_p75 += float(result[0, "p75"]) * n

    if total == 0:
        return None
    return total, minimum, maximum, weighted_p25 / total, weighted_p75 / total


def _histogram_per_file(
    files: list[Path], expr: pl.Expr, edges: list[float], predicate: pl.Expr | None = None
) -> list[int]:
    """Histogram over `files` against FIXED bin edges, summed.

    Fixed edges are what make this exact: each file is binned identically,
    so per-file counts add up to precisely the counts a single pass over
    everything would produce. (Letting each file pick its own edges would
    not be summable at all.)
    """
    totals = [0] * (len(edges) - 1)
    for file in files:
        result = (
            _scan_filtered(file, predicate)
            .select(expr.hist(bins=edges, include_breakpoint=False).alias("__h"))
            .collect(engine="streaming")
        )
        counts = result["__h"].to_list()
        for index, count in enumerate(counts[: len(totals)]):
            totals[index] += int(count)
    return totals


def _numeric_histogram_per_file(
    files: list[Path], column: str, bins: int, predicate: pl.Expr | None = None
) -> dict[str, Any]:
    stats = _numeric_stats_per_file(files, column, predicate)
    if stats is None:
        return {"column": column, "bin_edges": [], "counts": [], "x_scale": "linear"}
    _total, minimum, maximum, p25, p75 = stats

    if maximum <= minimum:  # every value identical — one bin is the honest answer
        return {
            "column": column,
            "bin_edges": [minimum, maximum],
            "counts": [_total],
            "x_scale": "linear",
        }

    clean = _clean_numeric_expr(column)
    if not _is_skewed(minimum, maximum, p25, p75):
        edges = [minimum + (maximum - minimum) * i / bins for i in range(bins + 1)]
        return {
            "column": column,
            "bin_edges": edges,
            "counts": _histogram_per_file(files, clean, edges, predicate),
            "x_scale": "linear",
        }

    offset = -minimum if minimum < 0 else 0.0
    low, high = math.log1p(minimum + offset), math.log1p(maximum + offset)
    log_edges = [low + (high - low) * i / bins for i in range(bins + 1)]
    counts = _histogram_per_file(files, (clean + offset).log1p(), log_edges, predicate)
    return {
        "column": column,
        "bin_edges": [math.expm1(edge) - offset for edge in log_edges],
        "counts": counts,
        "x_scale": "log",
    }


def numeric_histogram_data(
    lf: pl.LazyFrame,
    column: str,
    bins: int = NUMERIC_HISTOGRAM_BINS,
    files: list[Path] | None = None,
    predicate: pl.Expr | None = None,
) -> dict[str, Any]:
    """Equal-width histogram over a numeric column, computed lazily via
    polars' native streaming Expr.hist() (Rust-vectorized) rather than
    collecting the column and looping in Python — for a multi-million-row
    column this is roughly 50-60x faster than a naive Python approach
    (measured on the old eager Series.hist() path: ~0.6s -> ~0.01s per
    column on 5M rows), which directly matters for how long a user waits
    after opening a large file — and, unlike Series.hist(), never requires
    the column to be fully in memory first.

    Returns `x_scale`: "log" for a skewed column (see _is_skewed), else
    "linear". For "log" the values are binned in log1p space so the bins
    themselves are log-spaced — a log AXIS over linear bins would not
    help, since the bins would still each hold the same crammed mass.
    `bin_edges` is always reported in the column's ORIGINAL units, so
    consumers label axes and tooltips without knowing about the transform.

    The extra work is one small streaming aggregation for the quantiles —
    the same lazy path as everything else here, never a collect.

    `files` computes the whole thing per file against fixed bin edges and
    sums, which is exact and keeps memory flat; querying the concatenated
    frame instead grows ~28-42MB per file (measured repeatedly).
    """
    if files:
        return _numeric_histogram_per_file(files, column, bins, predicate)
    clean = _clean_numeric_expr(column)
    stats = lf.select(
        clean.count().alias("__n"),
        clean.min().alias("__min"),
        clean.max().alias("__max"),
        clean.quantile(0.25).alias("__p25"),
        clean.quantile(0.75).alias("__p75"),
    ).collect(engine="streaming")

    if stats[0, "__n"] == 0:
        return {"column": column, "bin_edges": [], "counts": [], "x_scale": "linear"}

    minimum = float(stats[0, "__min"])
    maximum = float(stats[0, "__max"])
    if not _is_skewed(minimum, maximum, float(stats[0, "__p25"]), float(stats[0, "__p75"])):
        counts, bin_edges = _histogram_from_clean_expr(lf, clean, bins)
        return {"column": column, "bin_edges": bin_edges, "counts": counts, "x_scale": "linear"}

    # Shift so the smallest value sits at 0 before log1p: log1p is only
    # defined above -1, and a raw log would break at 0. Applying the
    # offset BEFORE the transform is what makes zeros and negatives safe.
    offset = -minimum if minimum < 0 else 0.0
    counts, log_edges = _histogram_from_clean_expr(lf, (clean + offset).log1p(), bins)
    bin_edges = [math.expm1(edge) - offset for edge in log_edges]
    return {"column": column, "bin_edges": bin_edges, "counts": counts, "x_scale": "log"}


def _categorical_bar_per_file(
    files: list[Path], column: str, top_n: int, predicate: pl.Expr | None = None
) -> dict[str, Any]:
    """Top-N categories without ever holding every distinct value at once.

    The single-frame path below materializes the full value_counts table.
    On the real Criteo folder that is unbounded: two files alone hold ~2.2M
    distinct values across the categorical columns, and the loop over them
    exceeded a 2GB cap — for a chart that shows ten bars.

    Each file is instead reduced to its own top-K inside polars (sort +
    head before collect, so only K rows ever reach Python), and those are
    merged. Approximate by construction: a value ranked below K in every
    single file but high overall would be missed. K is set far above
    top_n to make that vanishingly unlikely for the skewed distributions a
    top-N bar is meaningful for in the first place.
    """
    merged: dict[str, int] = {}
    total = 0
    for file in files:
        scan = _scan_filtered(file, predicate).select(column).drop_nulls()
        total += int(scan.select(pl.len()).collect(engine="streaming")[0, 0])
        top = (
            scan.group_by(column)
            .len()
            .sort("len", descending=True)
            .head(CATEGORICAL_PER_FILE_TOP_K)
            .collect(engine="streaming")
        )
        for value, count in zip(top[column].to_list(), top["len"].to_list()):
            key = str(value)
            merged[key] = merged.get(key, 0) + int(count)

    ordered = sorted(merged.items(), key=lambda item: item[1], reverse=True)[:top_n]
    categories = [value for value, _ in ordered]
    counts = [count for _, count in ordered]
    other = total - sum(counts)
    if other > 0:
        categories.append("Other")
        counts.append(other)
    return {"column": column, "categories": categories, "counts": counts}


def categorical_bar_data(
    lf: pl.LazyFrame,
    column: str,
    top_n: int = TOP_N_CATEGORIES,
    files: list[Path] | None = None,
    predicate: pl.Expr | None = None,
) -> dict[str, Any]:
    if files:
        return _categorical_bar_per_file(files, column, top_n, predicate)
    non_null = pl.col(column).drop_nulls()
    result = lf.select(non_null.value_counts(sort=True).alias("__vc")).collect(engine="streaming")
    value_counts = result.unnest("__vc")

    categories = [str(v) for v in value_counts[column].to_list()]
    counts = value_counts["count"].to_list()

    if len(categories) > top_n:
        other_count = sum(counts[top_n:])
        categories = categories[:top_n] + ["Other"]
        counts = counts[:top_n] + [other_count]

    return {"column": column, "categories": categories, "counts": counts}


def value_frequency_histogram_data(
    lf: pl.LazyFrame,
    column: str,
    files: list[Path] | None = None,
    predicate: pl.Expr | None = None,
) -> dict[str, Any]:
    """How many times each distinct value occurs, bucketed.

    For an ID-like column — track IDs, where each row is one detection —
    naming the ten most frequent IDs says nothing (the top one can be 0.3%
    of rows). What matters is the SHAPE: how many tracks were seen once,
    how many were seen repeatedly. x is detections-per-value, y is how
    many values fall in that bucket.

    Buckets are log-ish (1, 2-3, 4-10, ...) because these counts are always
    heavily skewed toward 1.

    The bucketing happens inside polars — the big per-value count table
    never leaves the query engine, only the ~7 bucket totals do.

    `files` counts each file separately and sums the buckets, which keeps
    memory flat across a folder. A value appearing in several files is then
    counted once PER FILE rather than once overall, which shifts the
    distribution left; `per_file` is returned so the UI can say so.
    """
    sources: list[pl.LazyFrame] = (
        [_scan_filtered(file, predicate) for file in files] if files else [lf]
    )
    totals = dict.fromkeys(VALUE_FREQUENCY_LABELS, 0)

    for source in sources:
        # Group by a 64-bit HASH of the value, not the value itself. Only
        # the count per distinct value matters here — the values are never
        # displayed — and grouping ~500k strings directly cost ~400MB per
        # column and accumulated ~150-190MB per further column, overrunning
        # a 2GB cap by the fifth. Hashing collapses each key to 8 fixed
        # bytes before the grouping.
        #
        # A hash collision would merge two values' counts into one, moving
        # a single entry one bucket right. At 64 bits and ~10^6 distinct
        # values the expected number of collisions is ~10^-7, so this
        # cannot meaningfully change the shape being charted.
        bucketed = (
            source.select(pl.col(column).drop_nulls().hash().alias("__key"))
            .group_by("__key")
            .len()
            .with_columns(
                pl.col("len")
                .cast(pl.Float64)
                .cut(VALUE_FREQUENCY_BREAKS, labels=VALUE_FREQUENCY_LABELS)
                .alias("__bucket")
            )
            .group_by("__bucket")
            .len()
            .collect(engine="streaming")
        )
        for bucket, count in zip(bucketed["__bucket"].to_list(), bucketed["len"].to_list()):
            if bucket in totals:
                totals[bucket] += int(count)

    return {
        "column": column,
        "buckets": list(VALUE_FREQUENCY_LABELS),
        "counts": [totals[label] for label in VALUE_FREQUENCY_LABELS],
        "per_file": bool(files),
    }


def text_length_histogram_data(
    lf: pl.LazyFrame, column: str, bins: int = TEXT_LENGTH_HISTOGRAM_BINS
) -> dict[str, Any]:
    lengths = pl.col(column).drop_nulls().str.len_chars().cast(pl.Float64)
    counts, bin_edges = _histogram_from_clean_expr(lf, lengths, bins)
    return {"column": column, "bin_edges": bin_edges, "counts": counts}


def null_heatmap_data(
    lf: pl.LazyFrame,
    total_rows: int | None = None,
    max_rows: int = NULL_HEATMAP_MAX_ROWS,
    max_display_rows: int = NULL_HEATMAP_MAX_DISPLAY_ROWS,
) -> dict[str, Any]:
    """Build the null/missing heatmap's row x column grid.

    Plotly's heatmap trace silently rasterizes to a 100%-transparent image
    once the z-matrix has more rows than the plot has pixels of height
    (confirmed empirically: renders fine at 150 rows in a ~230px-tall plot,
    produces zero opaque pixels at 230+ rows in that same plot). To stay
    well clear of that cliff regardless of window size, sampled rows are
    grouped into at most `max_display_rows` bins, and each cell becomes the
    *fraction* of missing values in that bin for that column (0.0-1.0)
    rather than a strict 0/1 — which also reads as a more informative
    "density of missingness" view for large datasets. For datasets small
    enough that no binning is needed, each bin is exactly one row, so the
    fraction is always 0.0 or 1.0 and the output is unchanged from a plain
    per-row grid.

    `sample_df` is a bounded `.head(max_rows).collect()` — at most
    max_rows (500) rows are ever pulled into memory here, regardless of
    the source file's size. `total_rows`, if the caller already has it, is
    reused instead of triggering another count.
    """
    sample_df = lf.head(max_rows).collect()
    columns = sample_df.columns
    column_values = {column: sample_df[column].to_list() for column in columns}
    sampled_rows = sample_df.height

    if total_rows is None:
        total_rows = row_count(lf)

    if sampled_rows == 0:
        return {
            "columns": columns,
            "sampled_rows": 0,
            "display_rows": 0,
            "total_rows": total_rows,
            "z": [],
        }

    display_rows = min(sampled_rows, max_display_rows)
    bin_size = sampled_rows / display_rows

    z = []
    for bin_idx in range(display_rows):
        start = int(bin_idx * bin_size)
        is_last_bin = bin_idx == display_rows - 1
        end = sampled_rows if is_last_bin else max(int((bin_idx + 1) * bin_size), start + 1)
        bin_length = end - start
        z.append(
            [
                sum(1 for row_idx in range(start, end) if _is_missing(column_values[column][row_idx]))
                / bin_length
                for column in columns
            ]
        )

    return {
        "columns": columns,
        "sampled_rows": sampled_rows,
        "display_rows": display_rows,
        "total_rows": total_rows,
        "z": z,
    }


def build_chart_payload(
    lf: pl.LazyFrame,
    total_rows: int | None = None,
    dataset_dir: Path | None = None,
    null_counts: dict[str, int] | None = None,
    files: list[Path] | None = None,
    on_progress: Callable[[float, str], None] | None = None,
    predicate: pl.Expr | None = None,
) -> dict[str, Any]:
    """Compute every Phase 1+2+3 chart's data for the given LazyFrame.

    `total_rows`, if the caller already computed it (api.py shares one
    row_count() call across dataset info + chart data), is threaded down
    to null_heatmap_data/embedding_projection_data/image_previews_data
    instead of each re-counting; otherwise each computes it itself when
    needed (still one streaming count, never a full collect).

    `dataset_dir` (the loaded file's own directory) is used to resolve
    relative file paths for path-sourced image columns; without it, such
    columns are still detected but their thumbnails come back empty.
    """
    schema = lf.collect_schema()
    # The filter is applied to the frame here AND passed to every per-file
    # helper below, because the two take different routes to the data: the
    # per-file helpers re-scan each file themselves and never see this frame.
    if predicate is not None:
        lf = lf.filter(predicate)
    # Progress is reported over the per-column loop plus the four
    # whole-dataset steps that follow it, so the fraction advances
    # smoothly instead of stalling at the end. Measured at ~0.54s per
    # file for the whole payload, so on a large folder this runs for
    # minutes and the caller needs something to show.
    total_units = len(schema) + len(_TRAILING_STEPS)
    completed = 0

    def report(label: str) -> None:
        if on_progress is not None:
            on_progress(min(completed / total_units, 1.0) if total_units else 1.0, label)

    report("Preparing charts…")
    numeric_histograms = []
    categorical_bars = []
    value_frequency_histograms = []
    text_length_histograms = []
    categorical_columns = []

    for index, column in enumerate(schema):
        report(f"Charting columns — {index + 1} of {len(schema)}")
        chart_type, cardinality = classify_column_with_cardinality(
            lf, column, files=files, predicate=predicate
        )
        if chart_type == "numeric":
            numeric_histograms.append(
                numeric_histogram_data(lf, column, files=files, predicate=predicate)
            )
        elif chart_type == "categorical":
            # Too many distinct values for a top-N bar to say anything (an
            # ID-like column: track IDs, hashes). Chart the SHAPE of how
            # often values recur instead of naming ten arbitrary ones.
            if cardinality is not None and cardinality > CATEGORICAL_BAR_MAX_DISTINCT:
                value_frequency_histograms.append(
                    value_frequency_histogram_data(lf, column, files=files, predicate=predicate)
                )
            else:
                categorical_bars.append(
                    categorical_bar_data(lf, column, files=files, predicate=predicate)
                )
                categorical_columns.append(column)
        elif chart_type == "text":
            text_length_histograms.append(text_length_histogram_data(lf, column))
        completed += 1
        # "unsupported" columns (e.g. Date/Datetime) get no per-column chart
        # yet, but are still included in the null heatmap below.

    # Phase 2: embedding columns (list-of-float, consistent length > 8 dims)
    # get a PCA-projected scatter plot, colorable by any categorical column
    # found above. Detection is cheap (schema + a bounded length check); the
    # PCA itself is bounded by embedding_projection_data's row sampling, so
    # computing it eagerly here for every detected column stays cheap too.
    report(_TRAILING_STEPS[0])
    embedding_columns = detect_embedding_columns(lf)
    embedding_projections = [
        embedding_projection_data(
            lf, embedding["column"], categorical_columns=categorical_columns, total_rows=total_rows
        )
        for embedding in embedding_columns
    ]

    # Phase 3: image columns (bytes with image magic numbers, or strings
    # that look like image file paths) get a lazy-loaded thumbnail grid for
    # their first 50 rows. Detection is cheap (a small bounded sample per
    # column); thumbnail generation is capped at 50 rows so it stays cheap too.
    completed += 1
    report(_TRAILING_STEPS[1])
    image_columns = detect_image_columns(lf, files=files)
    image_previews = [
        image_previews_data(
            lf, image["column"], image["source"], total_rows=total_rows, dataset_dir=dataset_dir
        )
        for image in image_columns
    ]

    # Correlated missingness reuses the null counts the schema table
    # already has (api.py passes them in) — that reuse IS the cheap first
    # pass; see null_patterns.py. Without them, there is nothing to filter
    # candidates with, so the check is skipped rather than paying for a
    # fresh null-count scan here.
    completed += 1
    report(_TRAILING_STEPS[2])
    correlated: list[dict[str, Any]] = []
    if null_counts:
        # `total_rows or 0` would be wrong here: total_rows is None when the
        # caller didn't precompute it, and 0 trips correlated_missingness's
        # empty-dataset guard, silently disabling the whole feature.
        rows = total_rows if total_rows is not None else row_count(lf)
        correlated = correlated_missingness(
            lf, null_counts, rows, files=files, predicate=predicate
        )

    completed += 1
    report(_TRAILING_STEPS[3])
    heatmap = null_heatmap_data(lf, total_rows=total_rows)
    completed += 1
    report("Charts ready.")

    return {
        "correlated_missingness": correlated,
        "value_frequency_histograms": value_frequency_histograms,
        "null_heatmap": heatmap,
        "numeric_histograms": numeric_histograms,
        "categorical_bars": categorical_bars,
        "text_length_histograms": text_length_histograms,
        "embedding_columns": embedding_columns,
        "embedding_projections": embedding_projections,
        "image_columns": image_columns,
        "image_previews": image_previews,
    }
