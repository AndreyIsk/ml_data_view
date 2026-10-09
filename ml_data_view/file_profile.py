"""Per-file profiling, and how those profiles MERGE into a folder profile.

This is the half of the cache that has to be right. Storing numbers is
easy; the design problem is that a folder profile must be assembled by
MERGING per-file statistics rather than by rescanning, and not every
statistic merges.

THE THREE KINDS OF STATISTIC

  EXACT MERGE — the merged value is identical to a full recompute:
    row count (sum), null counts (sum), NaN counts (sum), missing counts
    (sum), numeric count/min/max/sum (sum/min/max), value-frequency
    buckets (sum), and histograms — numeric and text-length alike —
    PROVIDED every file was binned against the same edges. That proviso is
    the whole reason a histogram spec is stored beside its counts below.

  APPROXIMATE MERGE — merged deliberately, with a stated error:
    categorical top-K (a value below K in every single file but high
    overall is missed; flagged when any file was truncated), the p25/p75
    used only by the log-scale skew heuristic (a row-weighted mean of
    per-file quantiles, not the true global quantile), and the
    categorical-vs-text verdict (a row-weighted mean of per-file
    distinct/row RATIOS — see chart_data._classify_string_column_per_file,
    which this reproduces exactly from cached scalars).

  NOT MERGEABLE — recomputed on every load, never cached:
    the null heatmap (a sample of actual ROWS, which no merge of
    per-file summaries can reconstruct), embedding projections (a PCA
    basis is fitted to the data it projects; two files' bases are not
    combinable), image thumbnails (decoded bytes, not statistics — and
    cheap, being a bounded read of the first rows), and correlated
    missingness' candidate GROUPS (they depend on merged null counts, so
    which pairs to check is not known until every file has been merged —
    though the per-file co-null COUNTS for a pair, once known, are an
    exact sum and are cached; see null_patterns.CoNullCache).

  All four are bounded reads by construction, so recomputing them costs a
  fraction of a cold profile rather than undoing the cache.

WHY HISTOGRAMS CARRY THEIR SPEC
Per-file bin counts only add up if the bins are the same bins. The global
min/max is not known until every file has been read, so the edges are
derived from the MERGED min/max and each file's counts are stored with the
spec that produced them. Re-opening an unchanged folder derives the same
spec and every file hits. Adding a file that widens the range derives a
different spec, so the affected column's counts miss for every file and are
recomputed — for that COLUMN only. Every other column, and every other
statistic of that column, still comes from the cache. That is recorded in
the payload (`range_expanded`) rather than happening silently.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import polars as pl

from ml_data_view import chart_data as cd
from ml_data_view import timeseries as ts
from ml_data_view import fleet
from ml_data_view.catalog import build_catalog, collect_units, unit_label_key
from ml_data_view.hierarchy import identity_columns
from ml_data_view.profile_cache import (
    SECTION_CHART,
    SECTION_CO_NULL,
    SECTION_HIST,
    SECTION_LOAD,
    SECTION_TIMESERIES,
    MemoryProfileCache,
    ProfileCache,
)

# Per-file shortlist size for cached categorical top-K. Far above the ten
# bars actually drawn, so a value that is globally frequent is only missed
# if it ranked below 1000 in EVERY file — which, for the skewed
# distributions a top-N bar is meaningful for at all, does not happen. The
# cost is bounded: at most 1000 (value, count) pairs per column per file
# reach Python, and nothing larger ever leaves the query engine.
CACHE_CATEGORICAL_TOP_K = cd.CATEGORICAL_PER_FILE_TOP_K

# String columns whose distinct/row scalars are collected in one query.
# Bounded for the same reason every other batch size in this codebase is:
# decoding N column chunks at once costs N chunks of memory.
STRING_STATS_BATCH = 8


# --- histogram specs ----------------------------------------------------


@dataclass(frozen=True)
class HistogramSpec:
    """The parameters that GENERATE a set of bin edges.

    Stored instead of the edges themselves: 31 floats per column per file
    across 250 files is megabytes of JSON to say something five numbers
    already say, and two specs comparing equal is a cheaper and stricter
    test than two float lists comparing equal.

    `low`/`high` are in the space the binning happens in — original units
    for a linear scale, log1p space for a log one. `offset` is the shift
    applied before log1p so zeros and negatives are safe (see
    chart_data.numeric_histogram_data).
    """

    low: float
    high: float
    bins: int
    scale: str
    offset: float = 0.0

    def edges(self) -> list[float]:
        """Bin edges in binning space — what polars' hist(bins=) takes.

        Written as `low + (high - low) * i / bins`, not as a precomputed
        step times i: the two differ in the last bit, and chart_data
        computes it the first way, so anything else makes two payloads
        describing the same histogram compare unequal.
        """
        span = self.high - self.low
        return [self.low + span * index / self.bins for index in range(self.bins + 1)]

    def display_edges(self) -> list[float]:
        """Bin edges in the column's ORIGINAL units, which is what the
        chart labels its axis with regardless of how it was binned."""
        if self.scale == "log":
            return [math.expm1(edge) - self.offset for edge in self.edges()]
        return self.edges()

    def expression(self, base: pl.Expr) -> pl.Expr:
        return (base + self.offset).log1p() if self.scale == "log" else base

    def to_json(self) -> list[Any]:
        return [self.low, self.high, self.bins, self.scale, self.offset]

    @classmethod
    def from_json(cls, data: Any) -> HistogramSpec | None:
        if not isinstance(data, list) or len(data) != 5:
            return None
        low, high, bins, scale, offset = data
        return cls(float(low), float(high), int(bins), str(scale), float(offset))


def numeric_histogram_spec(
    minimum: float, maximum: float, p25: float, p75: float, bins: int
) -> HistogramSpec | None:
    """The spec a numeric column's merged min/max/quantiles imply.

    None means "no histogram": every value is identical, so there is no
    range to bin. Mirrors chart_data._numeric_histogram_per_file's choice
    of linear vs log exactly — the two must agree, or the cached path and
    the direct path would draw different charts from the same data.
    """
    if maximum <= minimum:
        return None
    if not cd._is_skewed(minimum, maximum, p25, p75):
        return HistogramSpec(minimum, maximum, bins, "linear", 0.0)
    offset = -minimum if minimum < 0 else 0.0
    return HistogramSpec(
        math.log1p(minimum + offset), math.log1p(maximum + offset), bins, "log", offset
    )


def text_length_histogram_spec(minimum: float, maximum: float, bins: int) -> HistogramSpec | None:
    """Text-length histograms are always linear over the global length
    range — which is exactly the uniform binning polars' own
    hist(bin_count=n) produces, so a merged result equals the single-frame
    one bin for bin."""
    if maximum <= minimum:
        return None
    return HistogramSpec(minimum, maximum, bins, "linear", 0.0)


# --- reading one file ---------------------------------------------------


def _scan_file(file: Path) -> pl.LazyFrame:
    """Every read of a source file in this module goes through here.

    One seam, so a test can record exactly which files were touched and
    prove an incremental open read only the new one — the same reason
    chart_data routes its scans through _scan_filtered.
    """
    from ml_data_view.data_loader import _dispatch_scanner

    return _dispatch_scanner(Path(file), Path(file).suffix.lower())


def _file_row_count(file: Path) -> int:
    """Rows in one file, from format-native metadata where there is any.

    Parquet keeps it in the footer, so this reads no column data at all —
    the same rule folder discovery already follows.
    """
    if Path(file).suffix.lower() == ".parquet":
        import pyarrow.parquet as pq

        return int(pq.ParquetFile(file).metadata.num_rows)
    return int(_scan_file(file).select(pl.len()).collect(engine="streaming")[0, 0])


def _file_missing_counts(
    file: Path, schema: pl.Schema
) -> tuple[dict[str, int], dict[str, int], dict[str, int]]:
    """(missing, nulls, nans) for one file.

    "Missing" is nulls plus NaN for floats, which is what the schema table
    and every chart in this app mean by missing. Nulls and NaN are also
    kept apart, because they have different causes — a null is an absent
    value, a NaN is usually a computation that went wrong upstream — and
    the brief asks for both.

    Parquet footer statistics answer the null half without decoding
    anything. Floats still need a real read for NaN, and a column whose
    writer omitted statistics needs one too; both go through the same
    bounded pyarrow batch decode data_loader already uses, scoped to just
    those columns of just this file.
    """
    from ml_data_view.data_loader import (
        _bounded_parquet_missing_counts,
        _null_count_expr,
        _parquet_null_counts_from_metadata,
    )

    columns = list(schema)
    if Path(file).suffix.lower() != ".parquet":
        # No footer statistics to read: one streaming aggregation over this
        # file gives missing counts, and null-vs-NaN is then a second
        # bounded expression on the same scan.
        exprs = [_null_count_expr(name, schema[name]) for name in columns]
        exprs += [pl.col(name).null_count().alias(f"__null__{name}") for name in columns]
        result = _scan_file(file).select(exprs).collect(engine="streaming")
        missing = {name: int(result[0, name]) for name in columns}
        nulls = {name: int(result[0, f"__null__{name}"]) for name in columns}
        return missing, nulls, {name: missing[name] - nulls[name] for name in columns}

    metadata_nulls, unavailable = _parquet_null_counts_from_metadata([Path(file)], schema)
    float_columns = {
        name for name in columns if schema[name] in (pl.Float32, pl.Float64)
    }
    decode_columns = unavailable | float_columns
    decoded = _bounded_parquet_missing_counts([Path(file)], schema, decode_columns)

    missing: dict[str, int] = {}
    nulls: dict[str, int] = {}
    nans: dict[str, int] = {}
    for name in columns:
        if name in decode_columns:
            missing[name] = int(decoded.get(name, 0))
            # A column with no footer statistics has no separately-known
            # null count; reporting the whole missing count as nulls would
            # be a guess, so NaN is reported as 0 and the two are only
            # split where the metadata actually said so.
            nulls[name] = int(metadata_nulls.get(name, missing[name]))
            nans[name] = max(0, missing[name] - nulls[name])
        else:
            missing[name] = int(metadata_nulls.get(name, 0))
            nulls[name] = missing[name]
            nans[name] = 0
    return missing, nulls, nans


def profile_load_section(
    file: Path, schema: pl.Schema, timestamp_column: str | None = None
) -> dict[str, Any]:
    """Everything a FOLDER LOAD needs from one file.

    Deliberately not everything the charts need: the load has to finish
    before the user sees a schema table, and the chart statistics depend
    on a classification verdict that is not known until every file's
    scalars have been merged. They land in the same cache row later (see
    profile_chart_section), so nothing is read twice.
    """
    columns = list(schema)
    missing, nulls, nans = _file_missing_counts(file, schema)
    strings = {
        name for name in columns if str(schema[name]) in cd.STRING_DTYPE_NAMES
    }
    string_stats: dict[str, dict[str, int]] = {}
    # approx_n_unique (HyperLogLog), not n_unique — an exact distinct count
    # builds a hash set of every distinct value, measured at 535MB for one
    # column over two real files. HyperLogLog is constant memory, so it
    # does not grow with rows.
    #
    # Batched rather than one query over every string column: the HLL
    # registers are tiny, but DECODING is not — 26 string columns at once
    # measured 1,154MB against 623MB per column, the same reason
    # data_loader decodes one column at a time and null_patterns caps its
    # aggregations per query.
    ordered_strings = [name for name in columns if name in strings]
    for start in range(0, len(ordered_strings), STRING_STATS_BATCH):
        batch = ordered_strings[start : start + STRING_STATS_BATCH]
        result = (
            _scan_file(file)
            .select(
                [pl.col(name).drop_nulls().count().alias(f"n__{name}") for name in batch]
                + [
                    pl.col(name).drop_nulls().approx_n_unique().alias(f"u__{name}")
                    for name in batch
                ]
            )
            .collect(engine="streaming")
        )
        for name in batch:
            string_stats[name] = {
                "n": int(result[0, f"n__{name}"]),
                "distinct": int(result[0, f"u__{name}"]),
            }

    return {
        "row_count": _file_row_count(file),
        "schema": {name: str(schema[name]) for name in columns},
        "missing_counts": missing,
        "null_counts": nulls,
        "nan_counts": nans,
        "string_stats": string_stats,
        "time_bounds": _time_bounds(file, timestamp_column),
    }


def _time_bounds(file: Path, column: str | None) -> dict[str, list[float]]:
    """This chunk's first and last timestamp, from the footer.

    Cached WITH the chunk because the chunk is the cache unit: a
    series-level boundary check — do its chunks overlap, is there a gap
    between them — is then a comparison of numbers already in hand rather
    than a re-read of every file in the series.

    Keyed BY COLUMN rather than stored as two bare numbers, so changing the
    timestamp column in the structure mapping adds an entry instead of
    invalidating the whole profile. Each entry costs one footer read, which
    is what discovery already pays.
    """
    if not column:
        return {}
    from ml_data_view.hierarchy import read_footer

    footer = read_footer(Path(file))
    if footer is None or column not in footer.stats:
        return {}
    from ml_data_view.hierarchy import _timestamp_value

    low, high = footer.stats[column]
    low_value, high_value = _timestamp_value(low), _timestamp_value(high)
    if low_value is None or high_value is None:
        return {}
    return {column: [low_value, high_value]}


# --- merging the load section -------------------------------------------


@dataclass
class MergedLoad:
    row_count: int = 0
    # (first, last) across every merged file, for the timestamp column the
    # mapping chose. None when no chunk carried bounds — a merge of nothing
    # known is not zero.
    time_bounds: tuple[float, float] | None = None
    missing_counts: dict[str, int] = field(default_factory=dict)
    null_counts: dict[str, int] = field(default_factory=dict)
    nan_counts: dict[str, int] = field(default_factory=dict)
    # (kind, cardinality) per column, reproducing
    # chart_data.classify_column_with_cardinality's folder verdict from
    # cached per-file scalars rather than by re-reading anything.
    classification: dict[str, tuple[str, int | None]] = field(default_factory=dict)


def merge_load(sections: list[dict[str, Any]], schema: pl.Schema) -> MergedLoad:
    """Fold per-file load sections into one folder-level answer.

    Every count here is an EXACT merge: a row belongs to exactly one file,
    so sums are sums. The only approximation is the categorical-vs-text
    verdict, which combines RATIOS rather than distinct counts — distinct
    values recur across files and cannot be added. See
    chart_data._classify_string_column_per_file for why combining ratios
    is sound for this heuristic and which direction it can err in.
    """
    merged = MergedLoad()
    columns = list(schema)
    merged.missing_counts = dict.fromkeys(columns, 0)
    merged.null_counts = dict.fromkeys(columns, 0)
    merged.nan_counts = dict.fromkeys(columns, 0)

    weighted_ratio: dict[str, float] = {}
    string_rows: dict[str, int] = {}
    max_distinct: dict[str, int] = {}

    lows: list[float] = []
    highs: list[float] = []

    for section in sections:
        merged.row_count += int(section.get("row_count") or 0)
        for bounds in (section.get("time_bounds") or {}).values():
            if isinstance(bounds, list) and len(bounds) == 2:
                lows.append(float(bounds[0]))
                highs.append(float(bounds[1]))
        for name in columns:
            merged.missing_counts[name] += int((section.get("missing_counts") or {}).get(name, 0))
            merged.null_counts[name] += int((section.get("null_counts") or {}).get(name, 0))
            merged.nan_counts[name] += int((section.get("nan_counts") or {}).get(name, 0))
        for name, stats in (section.get("string_stats") or {}).items():
            n = int(stats.get("n") or 0)
            if n == 0:
                continue
            string_rows[name] = string_rows.get(name, 0) + n
            weighted_ratio[name] = weighted_ratio.get(name, 0.0) + (
                int(stats["distinct"]) / n
            ) * n
            max_distinct[name] = max(max_distinct.get(name, 0), int(stats["distinct"]))

    for name in columns:
        dtype_name = str(schema[name])
        if dtype_name in cd.NUMERIC_DTYPE_NAMES:
            merged.classification[name] = ("numeric", None)
        elif dtype_name == "Boolean":
            merged.classification[name] = ("categorical", 2)
        elif dtype_name in cd.STRING_DTYPE_NAMES:
            merged.classification[name] = _string_verdict(
                string_rows.get(name, 0), weighted_ratio.get(name, 0.0), max_distinct.get(name, 0)
            )
        else:
            merged.classification[name] = ("unsupported", None)
    # min of the firsts, max of the lasts — an exact merge, like every
    # other extent here.
    if lows and highs:
        merged.time_bounds = (min(lows), max(highs))
    return merged


def _string_verdict(total_rows: int, weighted_ratio: float, max_distinct: int) -> tuple[str, int]:
    """chart_data._classify_string_column_per_file's decision, taken from
    merged scalars instead of by scanning. Kept in lockstep with it: if
    that heuristic changes, this must change and SCHEMA_VERSION must be
    bumped, or a cached folder would keep the old verdict."""
    if total_rows == 0:
        return "categorical", 0
    if max_distinct <= cd.CATEGORICAL_MAX_DISTINCT:
        return "categorical", max_distinct
    mean_ratio = weighted_ratio / total_rows
    kind = "categorical" if mean_ratio <= cd.CATEGORICAL_MAX_DISTINCT_RATIO else "text"
    return kind, max_distinct


# --- the chart section --------------------------------------------------


def _plan_kind(kind: str, cardinality: int | None) -> str | None:
    """Which per-column statistic a column needs, given its verdict.

    Mirrors build_chart_payload's branch: a categorical column with more
    distinct values than a top-N bar can say anything about gets a
    value-frequency histogram instead.
    """
    if kind == "numeric":
        return "numeric"
    if kind == "categorical":
        if cardinality is not None and cardinality > cd.CATEGORICAL_BAR_MAX_DISTINCT:
            return "value_freq"
        return "categorical"
    if kind == "text":
        return "text"
    return None


def profile_chart_column(file: Path, column: str, kind: str, dtype: pl.DataType) -> dict[str, Any]:
    """One column's cacheable chart statistics, for one file."""
    scan = _scan_file(file)
    if kind == "numeric":
        clean = cd._clean_numeric_expr(column)
        result = scan.select(
            clean.count().alias("n"),
            clean.min().alias("mn"),
            clean.max().alias("mx"),
            clean.sum().alias("sm"),
            clean.quantile(0.25).alias("p25"),
            clean.quantile(0.75).alias("p75"),
        ).collect(engine="streaming")
        n = int(result[0, "n"])
        if n == 0:
            return {"kind": "numeric", "n": 0}
        return {
            "kind": "numeric",
            "n": n,
            "min": float(result[0, "mn"]),
            "max": float(result[0, "mx"]),
            # Not used by any chart today. Cached because it is the one
            # aggregate that makes a merged MEAN exact (sum/count), and it
            # costs nothing to collect in a query already running.
            "sum": float(result[0, "sm"]),
            "p25": float(result[0, "p25"]),
            "p75": float(result[0, "p75"]),
        }

    if kind == "categorical":
        non_null = scan.select(column).drop_nulls()
        total = int(non_null.select(pl.len()).collect(engine="streaming")[0, 0])
        top = (
            non_null.group_by(column)
            .len()
            .sort("len", descending=True)
            .head(CACHE_CATEGORICAL_TOP_K)
            .collect(engine="streaming")
        )
        pairs = [
            [str(value), int(count)]
            for value, count in zip(top[column].to_list(), top["len"].to_list())
        ]
        return {
            "kind": "categorical",
            "total": total,
            "top": pairs,
            # Fewer rows than the shortlist asked for means the file's
            # distinct values fit inside it, so its contribution to a
            # merged count is complete rather than an underestimate.
            "complete": len(pairs) < CACHE_CATEGORICAL_TOP_K,
        }

    if kind == "value_freq":
        return {
            "kind": "value_freq",
            "buckets": _value_frequency_buckets(scan, column),
        }

    if kind == "text":
        lengths = pl.col(column).drop_nulls().str.len_chars().cast(pl.Float64)
        result = scan.select(
            lengths.count().alias("n"), lengths.min().alias("mn"), lengths.max().alias("mx")
        ).collect(engine="streaming")
        n = int(result[0, "n"])
        if n == 0:
            return {"kind": "text", "n": 0}
        return {
            "kind": "text",
            "n": n,
            "min": float(result[0, "mn"]),
            "max": float(result[0, "mx"]),
        }

    return {"kind": kind}


def _value_frequency_buckets(scan: pl.LazyFrame, column: str) -> dict[str, int]:
    """One file's detections-per-value buckets.

    Identical query to chart_data.value_frequency_histogram_data's per-file
    branch, including the 64-bit hash grouping that keeps a 500k-distinct
    column from costing ~400MB. Summing these across files is EXACT for
    the statistic as defined: the folder version already counts a value
    once per file it appears in, and says so (`per_file`).
    """
    totals = dict.fromkeys(cd.VALUE_FREQUENCY_LABELS, 0)
    bucketed = (
        scan.select(pl.col(column).drop_nulls().hash().alias("__key"))
        .group_by("__key")
        .len()
        .with_columns(
            pl.col("len")
            .cast(pl.Float64)
            .cut(cd.VALUE_FREQUENCY_BREAKS, labels=cd.VALUE_FREQUENCY_LABELS)
            .alias("__bucket")
        )
        .group_by("__bucket")
        .len()
        .collect(engine="streaming")
    )
    for bucket, count in zip(bucketed["__bucket"].to_list(), bucketed["len"].to_list()):
        if bucket in totals:
            totals[bucket] += int(count)
    return totals


# --- the orchestration --------------------------------------------------


@dataclass
class FileProfile:
    path: Path
    payload: dict[str, Any]
    # True when the section this pass needed was already cached. Counted
    # for the stat line ("12 of 279 files profiled, 267 from cache"), so it
    # has to mean "no data was read", not "a row existed".
    from_cache: bool


@dataclass
class FolderProfile:
    """A merged folder profile plus an honest account of how it was made."""

    profiles: list[FileProfile]
    merged: MergedLoad
    files_from_cache: int
    files_scanned: int
    # Columns whose histograms had to be recomputed because a file widened
    # the global range past the cached bin edges. Recorded rather than
    # silent: it is the one case where re-opening a cached folder does real
    # work, and a user watching the clock deserves to know which column.
    range_expanded: list[str] = field(default_factory=list)


def load_folder_profile(
    files: list[Path],
    schema: pl.Schema,
    cache: ProfileCache,
    on_file: Callable[[int, Path], None] | None = None,
    should_cancel: Callable[[], bool] | None = None,
    timestamp_column: str | None = None,
) -> FolderProfile:
    """Load (or compute) every file's load section and merge them.

    A cache hit reads no column data at all. A miss scans exactly one file.
    Files that have been removed from the folder are simply not in `files`,
    so they drop out of the merge without anything having to notice.

    `should_cancel` is checked at a FILE boundary only, so a stopped load
    covers whole files — the guarantee folder_load_job.py relies on to
    present a stopped load as a smaller dataset rather than an
    incomplete one.
    """
    profiles: list[FileProfile] = []

    for index, file in enumerate(files):
        if should_cancel is not None and should_cancel():
            break
        payload = cache.get(file) or {}
        cached = _load_section_usable(
            payload.get(SECTION_LOAD), schema, timestamp_column=timestamp_column
        )
        if not cached:
            payload = cache.merge_sections(
                file,
                {SECTION_LOAD: profile_load_section(Path(file), schema, timestamp_column)},
            )
        profiles.append(FileProfile(Path(file), payload, from_cache=cached))
        if on_file is not None:
            on_file(index, Path(file))

    merged = merge_load([p.payload[SECTION_LOAD] for p in profiles], schema)
    return FolderProfile(
        profiles=profiles,
        merged=merged,
        files_from_cache=sum(1 for p in profiles if p.from_cache),
        files_scanned=sum(1 for p in profiles if not p.from_cache),
    )


def _load_section_usable(
    section: Any, schema: pl.Schema, timestamp_column: str | None = None
) -> bool:
    """A cached load section is only usable if it describes THIS schema.

    A file can legitimately be in the cache under a schema group that is
    not the one being loaded (a folder holding two shapes of file), and a
    section missing a column would merge as a silent zero.

    A section that predates the chosen timestamp column is a miss too — but
    only a cheap one: re-profiling the chunk re-reads a footer it was going
    to read anyway, and the alternative is series-level time boundaries
    silently covering some chunks and not others.
    """
    if not isinstance(section, dict):
        return False
    cached_schema = section.get("schema")
    if not isinstance(cached_schema, dict):
        return False
    if not set(cached_schema) >= set(schema):
        return False
    if timestamp_column and timestamp_column not in (section.get("time_bounds") or {}):
        return False
    return True


# --- merging the chart section ------------------------------------------


@dataclass
class NumericStats:
    n: int = 0
    minimum: float = math.inf
    maximum: float = -math.inf
    total: float = 0.0
    weighted_p25: float = 0.0
    weighted_p75: float = 0.0

    @property
    def p25(self) -> float:
        return self.weighted_p25 / self.n if self.n else 0.0

    @property
    def p75(self) -> float:
        return self.weighted_p75 / self.n if self.n else 0.0


def _merge_numeric(sections: list[dict[str, Any]]) -> NumericStats:
    """count/min/max/sum are exact merges. p25/p75 are NOT — they are a
    row-weighted mean of each file's own quantile, which is not the global
    quantile and cannot be made into one from per-file summaries. That is
    acceptable only because they feed nothing but _is_skewed's
    range-vs-spread test, whose threshold sits an order of magnitude away
    from the reference distributions; chart_data._numeric_stats_per_file
    has always made the same trade."""
    merged = NumericStats()
    for section in sections:
        n = int(section.get("n") or 0)
        if n == 0:
            continue
        merged.n += n
        merged.minimum = min(merged.minimum, float(section["min"]))
        merged.maximum = max(merged.maximum, float(section["max"]))
        merged.total += float(section.get("sum") or 0.0)
        merged.weighted_p25 += float(section["p25"]) * n
        merged.weighted_p75 += float(section["p75"]) * n
    return merged


def _merge_categorical(sections: list[dict[str, Any]], top_n: int) -> dict[str, Any]:
    """Sum each value's per-file counts, then take the top N.

    APPROXIMATE, in one specific way: a value that ranked below
    CACHE_CATEGORICAL_TOP_K in every file it appears in contributes
    nothing to the merge even though its true total might be large. The
    `truncated` flag says when that is possible at all — it is set only
    when some file actually filled its shortlist — so the UI can mark the
    chart rather than presenting an approximation as a count.

    "Other" absorbs the difference between the total non-null count (an
    exact sum) and the bars shown, so the chart still adds up to the real
    number of rows even when the shortlist missed something.
    """
    merged: dict[str, int] = {}
    total = 0
    truncated = False
    for section in sections:
        total += int(section.get("total") or 0)
        if not section.get("complete", True):
            truncated = True
        for value, count in section.get("top") or []:
            merged[value] = merged.get(value, 0) + int(count)

    ordered = cd._ranked(merged, top_n)
    categories = [value for value, _ in ordered]
    counts = [count for _, count in ordered]
    other = total - sum(counts)
    if other > 0:
        categories.append("Other")
        counts.append(other)
    return {"categories": categories, "counts": counts, "truncated": truncated}


def _merge_value_freq(sections: list[dict[str, Any]]) -> list[int]:
    """Exact sum. See _value_frequency_buckets for why per-file counting is
    the definition of this statistic rather than an approximation of it."""
    totals = dict.fromkeys(cd.VALUE_FREQUENCY_LABELS, 0)
    for section in sections:
        for label, count in (section.get("buckets") or {}).items():
            if label in totals:
                totals[label] += int(count)
    return [totals[label] for label in cd.VALUE_FREQUENCY_LABELS]


def _merge_extent(sections: list[dict[str, Any]]) -> tuple[int, float, float]:
    """(n, min, max) for a text-length column — all three exact."""
    n = 0
    minimum = math.inf
    maximum = -math.inf
    for section in sections:
        count = int(section.get("n") or 0)
        if count == 0:
            continue
        n += count
        minimum = min(minimum, float(section["min"]))
        maximum = max(maximum, float(section["max"]))
    return n, minimum, maximum


# --- filling the chart and histogram sections ---------------------------


def _fill_chart_sections(
    profiles: list[FileProfile],
    schema: pl.Schema,
    plan: dict[str, str],
    cache: ProfileCache,
    on_file: Callable[[int, Path], None] | None = None,
) -> None:
    """Make sure every file has the per-column statistic its column's
    verdict calls for, computing only what is missing.

    A cached entry whose `kind` differs from the plan is a miss, not a
    reusable value: a column that was categorical last time and is text now
    (a new file pushed its distinct/row ratio over the threshold) needs a
    different statistic entirely, and the old one would merge into nonsense.
    """
    for index, profile in enumerate(profiles):
        section = dict(profile.payload.get(SECTION_CHART) or {})
        dirty = False
        for column, kind in plan.items():
            cached = section.get(column)
            if isinstance(cached, dict) and cached.get("kind") == kind:
                continue
            section[column] = profile_chart_column(
                profile.path, column, kind, schema[column]
            )
            dirty = True
        if dirty:
            profile.payload = cache.merge_sections(profile.path, {SECTION_CHART: section})
            profile.from_cache = False
        if on_file is not None:
            on_file(index, profile.path)


def _fill_histograms(
    profiles: list[FileProfile],
    specs: dict[str, HistogramSpec],
    plan: dict[str, str],
    cache: ProfileCache,
) -> tuple[dict[str, list[int]], list[str]]:
    """Per-file bin counts against the merged specs, summed.

    Returns (counts by column, columns whose cached histogram had to be
    thrown away because the global range moved). The second is the point
    of the whole design: a new file that widens one column's range costs a
    rescan of THAT COLUMN across the folder, and nothing else — not the
    other columns, not the other statistics of the same column, and not a
    re-derivation of anything already merged.
    """
    totals: dict[str, list[int]] = {
        column: [0] * spec.bins for column, spec in specs.items()
    }
    expanded: set[str] = set()

    for profile in profiles:
        section = dict(profile.payload.get(SECTION_HIST) or {})
        dirty = False
        for column, spec in specs.items():
            cached = section.get(column)
            cached_spec = HistogramSpec.from_json((cached or {}).get("spec")) if cached else None
            if cached_spec == spec and cached is not None:
                counts = [int(c) for c in cached["counts"]]
            else:
                if cached_spec is not None:
                    # There WAS a usable histogram here; it is only being
                    # discarded because the bins moved under it.
                    expanded.add(column)
                counts = _file_histogram(profile.path, column, plan[column], spec)
                section[column] = {"spec": spec.to_json(), "counts": counts}
                dirty = True
            for index, count in enumerate(counts[: len(totals[column])]):
                totals[column][index] += count
        if dirty:
            profile.payload = cache.merge_sections(profile.path, {SECTION_HIST: section})
            profile.from_cache = False

    return totals, sorted(expanded)


def _file_histogram(file: Path, column: str, kind: str, spec: HistogramSpec) -> list[int]:
    """One file's bin counts against FIXED edges.

    Fixed edges are what make the sum exact: every file is binned
    identically, so per-file counts add up to precisely the counts one pass
    over everything would produce. Letting each file pick its own edges
    would not be summable at all.
    """
    base = (
        pl.col(column).drop_nulls().str.len_chars().cast(pl.Float64)
        if kind == "text"
        else cd._clean_numeric_expr(column)
    )
    result = (
        _scan_file(file)
        .select(spec.expression(base).hist(bins=spec.edges(), include_breakpoint=False).alias("__h"))
        .collect(engine="streaming")
    )
    return [int(count) for count in result["__h"].to_list()[: spec.bins]]


# --- assembling the chart payload ---------------------------------------


def build_chart_payload_from_cache(
    lf: pl.LazyFrame,
    files: list[Path],
    cache: ProfileCache,
    total_rows: int | None = None,
    dataset_dir: Path | None = None,
    null_counts: dict[str, int] | None = None,
    on_progress: Callable[[float, str], None] | None = None,
    folder: FolderProfile | None = None,
    identity: list[dict[str, Any]] | None = None,
    timestamp_column: str | None = None,
    thresholds: ts.Thresholds | None = None,
    max_channels: int = 12,
    labels: dict[str, Any] | None = None,
    grouping_column: str | None = None,
    fleet_thresholds: fleet.FleetThresholds | None = None,
    fleet_max_channels: int = 12,
) -> dict[str, Any]:
    """chart_data.build_chart_payload's cached, incremental equivalent.

    Produces the same payload, key for key, but assembles the mergeable
    parts from per-file statistics — read from the cache where they are
    already known — and recomputes only what cannot be merged. A test pins
    the two against each other on a synthetic multi-file fixture; they are
    meant to be interchangeable, and the direct path stays the reference
    implementation rather than dead code.

    No `predicate` parameter, deliberately. A chart filter changes every
    number a file contributes, so a cached per-file statistic is simply
    not about the same rows — and keying the cache on a filter the user
    edits constantly would cache nothing useful while making every entry
    look valid. Filtered builds go down the direct path.
    """
    # The FILE schema, not the frame's: derived columns (hierarchy.py) are
    # literals attached per chunk and exist in no file, so profiling a file
    # against a schema that includes them asks for a column that is not
    # there. Their statistics are synthesized from row counts below.
    frame_schema = lf.collect_schema()
    derived = set(identity_columns(identity))
    schema = pl.Schema(
        {name: dtype for name, dtype in frame_schema.items() if name not in derived}
    )
    if folder is None:
        folder = load_folder_profile(files, schema, cache)
    merged = folder.merged

    plan = {
        column: kind
        for column, (verdict, cardinality) in merged.classification.items()
        # Derived columns are not IN any file, so they cannot be profiled
        # from one. Their statistics are synthesized below from row counts
        # the cache already holds — no read at all.
        if column not in derived
        and (kind := _plan_kind(verdict, cardinality)) is not None
    }

    steps = len(files) + len(cd._TRAILING_STEPS)
    completed = 0

    def report(label: str) -> None:
        if on_progress is not None:
            on_progress(min(completed / steps, 1.0) if steps else 1.0, label)

    report("Preparing charts…")

    def note_file(index: int, _path: Path) -> None:
        nonlocal completed
        completed = index + 1
        report(f"Profiling files — {index + 1} of {len(files)}")

    _fill_chart_sections(folder.profiles, schema, plan, cache, on_file=note_file)

    sections_by_column: dict[str, list[dict[str, Any]]] = {
        column: [
            (profile.payload.get(SECTION_CHART) or {}).get(column) or {}
            for profile in folder.profiles
        ]
        for column in plan
    }

    # Specs come from the MERGED extents, so every file is binned against
    # the same edges and the per-file counts below are summable.
    specs: dict[str, HistogramSpec] = {}
    numeric_merged: dict[str, NumericStats] = {}
    text_extent: dict[str, tuple[int, float, float]] = {}
    for column, kind in plan.items():
        if kind == "numeric":
            stats = _merge_numeric(sections_by_column[column])
            numeric_merged[column] = stats
            if stats.n:
                spec = numeric_histogram_spec(
                    stats.minimum, stats.maximum, stats.p25, stats.p75, cd.NUMERIC_HISTOGRAM_BINS
                )
                if spec is not None:
                    specs[column] = spec
        elif kind == "text":
            extent = _merge_extent(sections_by_column[column])
            text_extent[column] = extent
            if extent[0]:
                spec = text_length_histogram_spec(
                    extent[1], extent[2], cd.TEXT_LENGTH_HISTOGRAM_BINS
                )
                if spec is not None:
                    specs[column] = spec

    histogram_counts, range_expanded = _fill_histograms(folder.profiles, specs, plan, cache)
    folder.range_expanded = range_expanded
    folder.files_from_cache = sum(1 for p in folder.profiles if p.from_cache)
    folder.files_scanned = sum(1 for p in folder.profiles if not p.from_cache)

    numeric_histograms: list[dict[str, Any]] = []
    categorical_bars: list[dict[str, Any]] = []
    value_frequency_histograms: list[dict[str, Any]] = []
    text_length_histograms: list[dict[str, Any]] = []
    categorical_columns: list[str] = []

    # Column ORDER matches build_chart_payload's single pass over the
    # schema, so the two payloads compare equal as whole lists rather than
    # needing to be sorted first.
    # Frame order, so the payload lists columns the way the schema table
    # does — derived ones included, even though they are charted below
    # rather than here.
    for column in frame_schema:
        kind = plan.get(column)
        if kind == "numeric":
            numeric_histograms.append(
                _numeric_chart(column, numeric_merged[column], specs.get(column), histogram_counts)
            )
        elif kind == "categorical":
            categorical_bars.append(
                {
                    "column": column,
                    **_merge_categorical(sections_by_column[column], cd.TOP_N_CATEGORIES),
                }
            )
            categorical_columns.append(column)
        elif kind == "value_freq":
            value_frequency_histograms.append(
                {
                    "column": column,
                    "buckets": list(cd.VALUE_FREQUENCY_LABELS),
                    "counts": _merge_value_freq(sections_by_column[column]),
                    "per_file": True,
                }
            )
        elif kind == "text":
            text_length_histograms.append(
                _text_chart(column, text_extent[column], specs.get(column), histogram_counts)
            )

    # Rows per group and per unit: the charts a mapped directory
    # structure makes possible that a bag of files does not. Computed
    # entirely from the per-chunk row counts already merged above — a
    # chunk's group and unit are constants known from its path, so there
    # is nothing to read and nothing to approximate. Empty, and therefore
    # absent, for any input with no structure.
    categorical_bars.extend(derived_column_bars(identity, folder))

    # --- everything below is RECOMPUTED, never merged ---------------------
    #
    # Each is either a sample of actual rows or a fit over them, so no
    # combination of per-file summaries reconstructs it. All are bounded
    # reads, which is why recomputing them does not undo the cache.
    completed = len(files)
    report(cd._TRAILING_STEPS[0])
    embedding_columns = cd.detect_embedding_columns(lf)
    embedding_projections = [
        cd.embedding_projection_data(
            lf, embedding["column"], categorical_columns=categorical_columns, total_rows=total_rows
        )
        for embedding in embedding_columns
    ]

    completed += 1
    report(cd._TRAILING_STEPS[1])
    image_columns = cd.detect_image_columns(cd.drop_derived(lf, list(derived)), files=files)
    image_previews = [
        cd.image_previews_data(
            lf, image["column"], image["source"], total_rows=total_rows, dataset_dir=dataset_dir
        )
        for image in image_columns
    ]

    completed += 1
    report(cd._TRAILING_STEPS[2])
    correlated: list[dict[str, Any]] = []
    if null_counts:
        rows = total_rows if total_rows is not None else merged.row_count
        correlated = cd.correlated_missingness(
            lf, null_counts, rows, files=files, co_null_cache=CachedCoNulls(cache)
        )

    completed += 1
    report(cd._TRAILING_STEPS[3])
    heatmap = cd.null_heatmap_data(lf, total_rows=total_rows)

    # Time-series health. Runs only with a detected timestamp column —
    # which is the applicability predicate for every check that reads this,
    # stated once here.
    report("Checking time-series health…")
    numeric = [
        name
        for name, (kind, _cardinality) in merged.classification.items()
        if kind == "numeric" and name != timestamp_column
    ]
    timeseries = build_timeseries_payload(
        files,
        timestamp_column,
        numeric,
        thresholds or ts.Thresholds(),
        cache=cache,
        identity=identity,
        max_channels=max_channels,
    )
    # The catalogue and the group comparison. Both are arithmetic over
    # what is already in hand — cached row counts and schemas, the
    # per-file histograms every file was binned into against the SAME
    # merged spec, and the per-unit time-series merge above. Neither reads
    # anything, which is why they run unconditionally rather than behind a
    # button.
    report("Building the catalogue…")
    catalog = build_catalog_payload(
        files,
        identity=identity,
        row_counts=_load_row_counts(folder),
        schemas=_load_schemas(folder),
        timeseries=timeseries,
        labels=labels,
    )
    fleet_payload = build_fleet_payload(
        {str(profile.path): profile.payload for profile in folder.profiles},
        identity=identity,
        edges=numeric_edges(specs, plan),
        catalog=catalog,
        grouping_column=grouping_column,
        labels=labels,
        timestamp_column=timestamp_column,
        thresholds=fleet_thresholds,
        max_channels=fleet_max_channels,
    )
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
        "timeseries": timeseries,
        "catalog": catalog,
        "fleet": fleet_payload,
        # How this payload was actually produced. Reported rather than
        # inferred: "cached" and "recomputed because the range moved" look
        # identical from the outside, and only one of them is instant.
        "cache_stats": {
            "files_total": len(files),
            "files_from_cache": folder.files_from_cache,
            "files_scanned": folder.files_scanned,
            "range_expanded_columns": range_expanded,
        },
    }


def derived_column_bars(
    identity: list[dict[str, Any]] | None, folder: FolderProfile
) -> list[dict[str, Any]]:
    """Rows-per-group and rows-per-unit bars, from metadata alone.

    Charted: every GROUP column and the UNIT column — "which partition
    contributed how much" is the question a structure exists to make
    askable. Not charted: `chunk_index` and `chunk_file`, which are
    identifiers of position and of file rather than measurements; a bar
    naming ten arbitrary chunk filenames says nothing the unit bar does
    not. They stay on the frame for filtering, search and SQL, which is
    where an identifier is actually useful. That is why the last two
    columns are dropped here rather than a fixed list being consulted —
    the names in front of them depend on the mapping.

    A column with more distinct values than a top-N bar can say anything
    about is skipped entirely, the same threshold the rest of the app uses.

    Exact, not approximate: a chunk belongs to exactly one group and one
    unit, and its row count is a footer number, so these counts equal what
    a full scan would produce.
    """
    if not identity:
        return []
    chartable = identity_columns(identity)[:-2]
    rows_by_file = {
        str(profile.path): int((profile.payload.get(SECTION_LOAD) or {}).get("row_count") or 0)
        for profile in folder.profiles
    }
    bars = []
    for column in chartable:
        totals: dict[str, int] = {}
        for row in identity:
            rows = rows_by_file.get(row["file"])
            if rows is None:
                continue
            totals[str(row[column])] = totals.get(str(row[column]), 0) + rows
        if not totals or len(totals) > cd.CATEGORICAL_BAR_MAX_DISTINCT:
            continue
        ordered = cd._ranked(totals, cd.TOP_N_CATEGORIES)
        categories = [value for value, _ in ordered]
        counts = [count for _, count in ordered]
        other = sum(totals.values()) - sum(counts)
        if other > 0:
            categories.append("Other")
            counts.append(other)
        bars.append(
            {
                "column": column,
                "categories": categories,
                "counts": counts,
                "truncated": False,
                # Marks this as a chart of the structure rather than of the
                # data, so the UI can say where the numbers came from.
                "derived": True,
            }
        )
    return bars


def _numeric_chart(
    column: str,
    stats: NumericStats,
    spec: HistogramSpec | None,
    counts_by_column: dict[str, list[int]],
) -> dict[str, Any]:
    """The same three shapes chart_data._numeric_histogram_per_file
    produces: nothing to chart, one bin, or a real histogram."""
    if stats.n == 0:
        return {"column": column, "bin_edges": [], "counts": [], "x_scale": "linear"}
    if spec is None:  # every value identical — one bin is the honest answer
        return {
            "column": column,
            "bin_edges": [stats.minimum, stats.maximum],
            "counts": [stats.n],
            "x_scale": "linear",
        }
    return {
        "column": column,
        "bin_edges": spec.display_edges(),
        "counts": counts_by_column.get(column, [0] * spec.bins),
        "x_scale": spec.scale,
    }


def _text_chart(
    column: str,
    extent: tuple[int, float, float],
    spec: HistogramSpec | None,
    counts_by_column: dict[str, list[int]],
) -> dict[str, Any]:
    n, minimum, maximum = extent
    if n == 0:
        return {"column": column, "bin_edges": [], "counts": []}
    if spec is None:  # every value the same length
        return {"column": column, "bin_edges": [minimum, maximum], "counts": [n]}
    return {
        "column": column,
        "bin_edges": spec.display_edges(),
        "counts": counts_by_column.get(column, [0] * spec.bins),
    }


# --- correlated missingness' per-file co-null counts ---------------------


class CachedCoNulls:
    """Per-file co-null counts, cached per (file, anchor, other) pair.

    Correlated missingness is listed above as NOT mergeable, and the group
    STRUCTURE genuinely is not: which pairs are worth checking depends on
    the merged null counts, so it cannot be known file by file. But once a
    pair is chosen, the count of rows where both are null is a plain sum
    over files — a co-null row belongs to exactly one file — so the
    per-file halves cache and merge exactly like everything else.

    Without this, re-opening a cached folder would still pay for the one
    real scan in the chart build, and "near-instant" would be false for
    precisely the datasets that have correlated missingness in them.
    """

    def __init__(self, cache: ProfileCache) -> None:
        self._cache = cache

    @staticmethod
    def _key(anchor: str, other: str) -> str:
        return f"{anchor}\u0000{other}"

    def get(self, file: Path, anchor: str, other: str) -> int | None:
        payload = self._cache.get(file) or {}
        value = (payload.get(SECTION_CO_NULL) or {}).get(self._key(anchor, other))
        return int(value) if value is not None else None

    def put(self, file: Path, anchor: str, counts: dict[str, int]) -> None:
        payload = self._cache.get(file) or {}
        section = dict(payload.get(SECTION_CO_NULL) or {})
        for other, count in counts.items():
            section[self._key(anchor, other)] = int(count)
        self._cache.merge_sections(file, {SECTION_CO_NULL: section})


# --- time-series health --------------------------------------------------


def unit_groups(
    files: list[Path], identity: list[dict[str, Any]] | None
) -> list[tuple[tuple[str, ...], list[Path]]]:
    """The units, as (key, ordered chunk paths).

    With a structure, a unit is whatever the mapping says — its chunks
    already ordered by hierarchy.py, by timestamp where the filenames
    disagreed. Without one, a unit is a single FILE, which is the same
    fallback every other part of this app makes and is why a lone CSV gets
    the timestamp and channel checks just as a mapped tree does.

    The KEY is the group values then the unit id, as a tuple. Everything
    that needs to name a unit — the time-series payload, the catalogue, a
    finding — joins `catalog.unit_label_key` onto it, so there is one
    spelling of "123 / rec_a" in the codebase rather than three.
    """
    if not identity:
        return [((Path(file).name,), [Path(file)]) for file in files]

    names = identity_columns(identity)
    # [groups..., unit, chunk_index, chunk_file] — the unit column is third
    # from the end, whatever the mapping called it, and the group columns
    # are everything before it.
    group_columns = names[:-3]
    unit_column = names[-3]
    units: dict[tuple[str, ...], list[tuple[int, Path]]] = {}
    for row in identity:
        # Keyed by the group values AS WELL as the unit id. Two groups can
        # legitimately hold a unit of the same name — `day=1/part0.parquet`
        # and `day=2/part0.parquet` are two units, and merging them would
        # invent a series where there is none.
        key = tuple(str(row[name]) for name in group_columns) + (str(row[unit_column]),)
        units.setdefault(key, []).append((int(row[names[-2]]), Path(row["file"])))
    return [
        (key, [path for _index, path in sorted(entries)]) for key, entries in units.items()
    ]


def build_timeseries_payload(
    files: list[Path],
    timestamp_column: str | None,
    channels: list[str],
    thresholds: ts.Thresholds,
    cache: ProfileCache | MemoryProfileCache | None = None,
    identity: list[dict[str, Any]] | None = None,
    max_channels: int = 12,
) -> dict[str, Any] | None:
    """Per-unit time-series health, from per-chunk artifacts.

    None when there is no timestamp column — which is the applicability
    predicate, expressed once here so every consumer downstream inherits
    it rather than restating it.

    The per-chunk artifact is cached like every other per-chunk statistic,
    so re-opening reads nothing and adding one chunk analyses one chunk.
    A cached artifact computed against a DIFFERENT timestamp column, or a
    different set of documented ranges, is a miss: both change the numbers,
    and a stale answer here would be wrong rather than merely old.

    `max_channels` bounds the expensive half. The timestamp pass is one
    read per chunk; the channel passes are one per channel per chunk, so a
    40-column folder would multiply a first profile by forty. Analysed in
    schema order and reported, so what was left out is visible rather than
    silently absent.
    """
    if not timestamp_column or not files:
        return None

    wanted = [name for name in channels if name != timestamp_column][:max_channels]
    skipped = [name for name in channels if name != timestamp_column][max_channels:]
    signature = timeseries_signature(timestamp_column, thresholds)

    artifacts: dict[str, dict[str, Any]] = {}
    scanned = 0
    for file in files:
        payload = (cache.get(file) or {}) if cache is not None else {}
        section = payload.get(SECTION_TIMESERIES)
        if (
            isinstance(section, dict)
            and section.get("signature") == signature
            and set(section.get("result", {}).get("channels") or {}) >= set(wanted)
        ):
            artifacts[str(file)] = section["result"]
            continue
        result = ts.analyse_chunk(file, timestamp_column, wanted, thresholds)
        artifacts[str(file)] = result
        scanned += 1
        if cache is not None:
            cache.merge_sections(
                file, {SECTION_TIMESERIES: {"signature": signature, "result": result}}
            )

    units: dict[str, Any] = {}
    for key, paths in unit_groups(files, identity):
        chunk_results = [artifacts.get(str(path), {"rows": 0}) for path in paths]
        units[unit_label_key(key)] = ts.merge_unit(
            chunk_results, [p.name for p in paths], thresholds
        )

    return {
        "timestamp_column": timestamp_column,
        "thresholds": thresholds.to_dict(),
        # The drawn form, built here so the chart and the finding beside it
        # cannot tell different stories about what counts as a gap.
        "charts": {unit_id: ts.unit_charts(unit) for unit_id, unit in units.items()},
        "channels_analysed": wanted,
        "channels_skipped": skipped,
        "chunks_scanned": scanned,
        "chunks_total": len(files),
        # Whether chunk-boundary questions mean anything here. A unit that
        # is one file has no boundary between chunks, so the checks that
        # need one skip rather than reporting that nothing is wrong.
        "has_series": bool(identity)
        and any(len(paths) > 1 for _key, paths in unit_groups(files, identity)),
        "units": units,
    }


# --- the catalogue and the group comparison ------------------------------
#
# Both are pure arithmetic over statistics already in hand: the per-file
# load sections (row counts, schemas), the per-file histograms (binned
# against the MERGED spec, so every file in every group shares its edges)
# and the merged per-unit time-series artifact. Neither reads a byte, on a
# first profile or on any after it.


def build_catalog_payload(
    files: list[Path],
    *,
    identity: list[dict[str, Any]] | None = None,
    row_counts: dict[str, int] | None = None,
    schemas: dict[str, dict[str, str]] | None = None,
    timeseries: dict[str, Any] | None = None,
    labels: dict[str, Any] | None = None,
) -> dict[str, Any] | None:
    """One catalogue row per unit — a series, or a file where there is no
    series level. None when there is nothing loaded to catalogue."""
    if not files:
        return None
    return build_catalog(
        collect_units(
            unit_groups(files, identity),
            row_counts=row_counts,
            schemas=schemas,
            numeric_dtypes=cd.NUMERIC_DTYPE_NAMES,
        ),
        timeseries=timeseries,
        labels=labels,
    )


def _load_row_counts(folder: FolderProfile) -> dict[str, int]:
    return {
        str(profile.path): int((profile.payload.get(SECTION_LOAD) or {}).get("row_count") or 0)
        for profile in folder.profiles
        if (profile.payload.get(SECTION_LOAD) or {}).get("row_count") is not None
    }


def _load_schemas(folder: FolderProfile) -> dict[str, dict[str, str]]:
    return {
        str(profile.path): dict((profile.payload.get(SECTION_LOAD) or {}).get("schema") or {})
        for profile in folder.profiles
    }


def build_fleet_payload(
    payloads: dict[str, dict[str, Any]],
    *,
    identity: list[dict[str, Any]] | None,
    edges: dict[str, list[float]],
    catalog: dict[str, Any] | None,
    grouping_column: str | None,
    labels: dict[str, Any] | None = None,
    timestamp_column: str | None = None,
    thresholds: fleet.FleetThresholds | None = None,
    max_channels: int = 12,
) -> dict[str, Any]:
    """Per-channel divergence from the typical group, from the cache alone.

    `payloads` is the profile cache's own content per file, and `edges` the
    merged display edges per numeric column. Both are already in hand
    wherever this is called from, which is what keeps this a pure
    rearrangement of stored numbers rather than a second pass over data.

    The grouping value has to be a property of the FILE, which is what the
    identity rows carry — a directory role, a hive key, or any derived
    column. An ordinary data column varies within a file, and no merge of
    per-file histograms can be split by it; that reports itself
    unavailable with the reason rather than quietly producing nothing.
    See fleet.py's module docstring.
    """
    labels = dict(labels or {})
    if not grouping_column:
        return fleet.unavailable("no grouping column")
    names = identity_columns(identity)
    if grouping_column not in names:
        return fleet.unavailable(
            f"{grouping_column!r} is a data column, not a property of each file — "
            "a group comparison from cached per-file statistics needs a grouping "
            "that is constant within a file (a directory level, a hive key)"
        )

    file_groups = [(str(row["file"]), str(row[grouping_column])) for row in identity or []]
    per_file = {
        path: {
            "hist": {
                column: [int(count) for count in (entry or {}).get("counts") or []]
                for column, entry in (payload.get(SECTION_HIST) or {}).items()
            },
            "numeric": {
                column: entry
                for column, entry in (payload.get(SECTION_CHART) or {}).items()
                if isinstance(entry, dict) and entry.get("kind") == "numeric"
            },
        }
        for path, payload in payloads.items()
    }
    return fleet.build_fleet(
        group_column=grouping_column,
        group_label=_grouping_label(grouping_column, labels),
        unit_label=str(labels.get("unit_label") or "File"),
        file_groups=file_groups,
        per_file=per_file,
        edges=edges,
        catalog=catalog,
        exclude={timestamp_column} if timestamp_column else set(),
        max_channels=max_channels,
        thresholds=thresholds,
    )


def numeric_edges(specs: dict[str, HistogramSpec], plan: dict[str, str]) -> dict[str, list[float]]:
    """The bin edges a group comparison charts against.

    Numeric columns only. A text-length histogram merges and compares
    exactly as well, but "channel" means a measurement, and a mean and a
    sigma quoted over string lengths would read as one.
    """
    return {
        column: spec.display_edges()
        for column, spec in specs.items()
        if plan.get(column) == "numeric"
    }


def cached_numeric_edges(payloads: dict[str, dict[str, Any]]) -> dict[str, list[float]]:
    """The same edges, read back OUT of the cache instead of re-derived.

    Every histogram is stored with the spec that produced it — five numbers
    beside the counts — so the edges a completed profile binned against are
    already written down. Reading them back is what lets the grouping
    column be changed after a build without rebuilding anything: the bins
    do not depend on how the files are grouped.

    A column whose files disagree about their spec is skipped rather than
    compared across mismatched bins, which is the one thing PSI must never
    be asked to do.
    """
    edges: dict[str, list[float]] = {}
    seen: dict[str, Any] = {}
    numeric = {
        column
        for payload in payloads.values()
        for column, entry in (payload.get(SECTION_CHART) or {}).items()
        if isinstance(entry, dict) and entry.get("kind") == "numeric"
    }
    for payload in payloads.values():
        for column, entry in (payload.get(SECTION_HIST) or {}).items():
            if column not in numeric or not isinstance(entry, dict):
                continue
            spec = HistogramSpec.from_json(entry.get("spec"))
            if spec is None:
                continue
            if column in seen:
                if seen[column] != spec:
                    edges.pop(column, None)
                continue
            seen[column] = spec
            edges[column] = spec.display_edges()
    return edges


def fleet_from_cache(
    files: list[Path],
    cache: ProfileCache | MemoryProfileCache,
    *,
    identity: list[dict[str, Any]] | None,
    catalog: dict[str, Any] | None,
    grouping_column: str | None,
    labels: dict[str, Any] | None = None,
    timestamp_column: str | None = None,
    thresholds: fleet.FleetThresholds | None = None,
    max_channels: int = 12,
) -> dict[str, Any]:
    """The group comparison rebuilt for a DIFFERENT grouping column.

    Reads the stored statistics and nothing else — no column data, and no
    re-binning, because the bins do not depend on the grouping. This is
    what makes the grouping picker instant: choosing a different level
    rearranges numbers that are already written down.
    """
    payloads = {str(file): (cache.get(file) or {}) for file in files}
    return build_fleet_payload(
        payloads,
        identity=identity,
        edges=cached_numeric_edges(payloads),
        catalog=catalog,
        grouping_column=grouping_column,
        labels=labels,
        timestamp_column=timestamp_column,
        thresholds=thresholds,
        max_channels=max_channels,
    )


def _grouping_label(column: str, labels: dict[str, Any]) -> str:
    """The user's word for the level this column came from.

    The mapping's labels line up with its group columns, so the label is
    read off by POSITION rather than guessed from the column name — which
    is how "System" reaches the UI while the code only ever knew
    `sys_key`. A column that is not one of the group levels falls back to
    its own name, which is the most honest label available for it.
    """
    columns = [str(name) for name in (labels.get("group_columns") or [])]
    group_labels = [str(text) for text in (labels.get("group_labels") or [])]
    if column in columns:
        index = columns.index(column)
        if index < len(group_labels):
            return group_labels[index]
    return column


# What SHAPE the per-chunk artifact has. Bumped when a new statistic is
# added to it, so chunks profiled by older code are re-analysed rather
# than merged as silently absent.
#
# Deliberately not profile_cache.SCHEMA_VERSION, which invalidates every
# row of every section wholesale: the time-series artifact already has a
# per-section reuse test, and throwing away a folder's null counts,
# histograms and categorical shortlists because a channel gained a bucket
# array would be charging for a change that did not touch them.
TIMESERIES_ARTIFACT_VERSION = 2


def timeseries_signature(timestamp_column: str, thresholds: ts.Thresholds) -> list[Any]:
    """What a cached artifact was computed against.

    Only the inputs that change the STORED numbers: the artifact's own
    shape, the timestamp column, and the documented ranges (which decide
    the exact out-of-range count). The gap factor and the flatline length
    are applied at merge time against stored histograms and run lengths,
    so changing them re-reads nothing — which is the point of storing
    those rather than verdicts.
    """
    return [
        TIMESERIES_ARTIFACT_VERSION,
        timestamp_column,
        sorted((name, *bounds) for name, bounds in thresholds.ranges.items()),
    ]
