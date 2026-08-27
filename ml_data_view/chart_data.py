"""Headless chart-data computation for Phase 1 profiling charts.

Pure functions over an already-loaded polars DataFrame — no pywebview or
Plotly imports here. Each function returns a small JSON-safe dict; the
frontend (app.js) turns that into a Plotly.js figure. Keeping the heavy
lifting in Python keeps this testable without a browser and keeps the
JS side a thin renderer.
"""
from __future__ import annotations

from typing import Any

import polars as pl

from ml_data_view.embedding_data import detect_embedding_columns, embedding_projection_data

NUMERIC_HISTOGRAM_BINS = 30
TEXT_LENGTH_HISTOGRAM_BINS = 30
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


def classify_column(df: pl.DataFrame, column: str) -> str:
    """Classify a column for charting purposes.

    Returns "numeric", "categorical", "text", or "unsupported" (no chart
    type defined yet, e.g. Date/Datetime columns).
    """
    dtype_name = str(df.schema[column])
    if dtype_name in NUMERIC_DTYPE_NAMES:
        return "numeric"
    if dtype_name == "Boolean":
        return "categorical"
    if dtype_name in STRING_DTYPE_NAMES:
        return _classify_string_column(df, column)
    return "unsupported"


def _classify_string_column(df: pl.DataFrame, column: str) -> str:
    non_null = df[column].drop_nulls()
    if non_null.len() == 0:
        return "categorical"
    n_unique = non_null.n_unique()
    if (
        n_unique <= CATEGORICAL_MAX_DISTINCT
        or n_unique / non_null.len() <= CATEGORICAL_MAX_DISTINCT_RATIO
    ):
        return "categorical"
    return "text"


def _drop_missing_numeric(series: pl.Series) -> pl.Series:
    """Drop nulls, plus NaN for float series (see _is_missing's docstring)."""
    series = series.drop_nulls()
    if series.dtype in (pl.Float32, pl.Float64):
        series = series.filter(~series.is_nan())
    return series


def _equal_width_histogram(values: pl.Series, bins: int) -> tuple[list[int], list[float]]:
    """Bin an already-missing-filtered numeric series into `bins` equal-width
    buckets. Returns (counts, bin_edges).

    Uses polars' native Series.hist() (Rust-vectorized) rather than a Python
    loop over the values — for a multi-million-row column this is roughly
    50-60x faster (measured: ~0.6s -> ~0.01s per column on 5M rows), which
    directly matters for how long a user waits after opening a large file.
    """
    hist_df = values.hist(bin_count=bins)
    counts = [int(count) for count in hist_df["count"].to_list()]
    # hist() returns each bin's right edge; the first bin's left edge is the
    # series' own minimum, which hist() doesn't include.
    bin_edges = [float(values.min())] + [float(edge) for edge in hist_df["breakpoint"].to_list()]
    return counts, bin_edges


def numeric_histogram_data(
    df: pl.DataFrame, column: str, bins: int = NUMERIC_HISTOGRAM_BINS
) -> dict[str, Any]:
    values = _drop_missing_numeric(df[column])
    if values.len() == 0:
        return {"column": column, "bin_edges": [], "counts": []}

    counts, bin_edges = _equal_width_histogram(values.cast(pl.Float64), bins)
    return {"column": column, "bin_edges": bin_edges, "counts": counts}


def categorical_bar_data(
    df: pl.DataFrame, column: str, top_n: int = TOP_N_CATEGORIES
) -> dict[str, Any]:
    counts_df = df[column].drop_nulls().value_counts().sort("count", descending=True)
    categories = [str(v) for v in counts_df[column].to_list()]
    counts = counts_df["count"].to_list()

    if len(categories) > top_n:
        other_count = sum(counts[top_n:])
        categories = categories[:top_n] + ["Other"]
        counts = counts[:top_n] + [other_count]

    return {"column": column, "categories": categories, "counts": counts}


def text_length_histogram_data(
    df: pl.DataFrame, column: str, bins: int = TEXT_LENGTH_HISTOGRAM_BINS
) -> dict[str, Any]:
    lengths = df[column].drop_nulls().str.len_chars().cast(pl.Float64)
    if lengths.len() == 0:
        return {"column": column, "bin_edges": [], "counts": []}

    counts, bin_edges = _equal_width_histogram(lengths, bins)
    return {"column": column, "bin_edges": bin_edges, "counts": counts}


def null_heatmap_data(
    df: pl.DataFrame,
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
    """
    sample_df = df.head(max_rows) if df.height > max_rows else df
    columns = sample_df.columns
    column_values = {column: sample_df[column].to_list() for column in columns}
    sampled_rows = sample_df.height

    if sampled_rows == 0:
        return {
            "columns": columns,
            "sampled_rows": 0,
            "display_rows": 0,
            "total_rows": df.height,
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
        "total_rows": df.height,
        "z": z,
    }


def build_chart_payload(df: pl.DataFrame) -> dict[str, Any]:
    """Compute every Phase 1+2 chart's data for the given DataFrame."""
    numeric_histograms = []
    categorical_bars = []
    text_length_histograms = []
    categorical_columns = []

    for column in df.columns:
        chart_type = classify_column(df, column)
        if chart_type == "numeric":
            numeric_histograms.append(numeric_histogram_data(df, column))
        elif chart_type == "categorical":
            categorical_bars.append(categorical_bar_data(df, column))
            categorical_columns.append(column)
        elif chart_type == "text":
            text_length_histograms.append(text_length_histogram_data(df, column))
        # "unsupported" columns (e.g. Date/Datetime) get no per-column chart
        # yet, but are still included in the null heatmap below.

    # Phase 2: embedding columns (list-of-float, consistent length > 8 dims)
    # get a PCA-projected scatter plot, colorable by any categorical column
    # found above. Detection is cheap (schema + length check); the PCA
    # itself is bounded by embedding_projection_data's row sampling, so
    # computing it eagerly here for every detected column stays cheap too.
    embedding_columns = detect_embedding_columns(df)
    embedding_projections = [
        embedding_projection_data(df, embedding["column"], categorical_columns=categorical_columns)
        for embedding in embedding_columns
    ]

    return {
        "null_heatmap": null_heatmap_data(df),
        "numeric_histograms": numeric_histograms,
        "categorical_bars": categorical_bars,
        "text_length_histograms": text_length_histograms,
        "embedding_columns": embedding_columns,
        "embedding_projections": embedding_projections,
    }
