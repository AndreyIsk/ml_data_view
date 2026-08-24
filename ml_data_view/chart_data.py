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

NUMERIC_HISTOGRAM_BINS = 30
TEXT_LENGTH_HISTOGRAM_BINS = 30
TOP_N_CATEGORIES = 10
NULL_HEATMAP_MAX_ROWS = 500
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


def _equal_width_histogram(values: list[float], bins: int) -> tuple[list[int], list[float]]:
    """Bin `values` into `bins` equal-width buckets. Returns (counts, bin_edges)."""
    lo, hi = min(values), max(values)
    if lo == hi:
        return [len(values)], [lo, lo + 1.0]

    width = (hi - lo) / bins
    edges = [lo + i * width for i in range(bins + 1)]
    counts = [0] * bins
    for value in values:
        idx = int((value - lo) / width)
        if idx >= bins:  # the max value lands exactly on the last edge
            idx = bins - 1
        counts[idx] += 1
    return counts, edges


def numeric_histogram_data(
    df: pl.DataFrame, column: str, bins: int = NUMERIC_HISTOGRAM_BINS
) -> dict[str, Any]:
    values = [
        float(v) for v in df[column].drop_nulls().to_list() if not _is_missing(v)
    ]
    if not values:
        return {"column": column, "bin_edges": [], "counts": []}

    counts, bin_edges = _equal_width_histogram(values, bins)
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
    lengths = [
        float(length)
        for length in df[column].drop_nulls().str.len_chars().to_list()
    ]
    if not lengths:
        return {"column": column, "bin_edges": [], "counts": []}

    counts, bin_edges = _equal_width_histogram(lengths, bins)
    return {"column": column, "bin_edges": bin_edges, "counts": counts}


def null_heatmap_data(df: pl.DataFrame, max_rows: int = NULL_HEATMAP_MAX_ROWS) -> dict[str, Any]:
    sample_df = df.head(max_rows) if df.height > max_rows else df
    columns = sample_df.columns
    column_values = {column: sample_df[column].to_list() for column in columns}

    z = [
        [1 if _is_missing(column_values[column][row_idx]) else 0 for column in columns]
        for row_idx in range(sample_df.height)
    ]

    return {
        "columns": columns,
        "sampled_rows": sample_df.height,
        "total_rows": df.height,
        "z": z,
    }


def build_chart_payload(df: pl.DataFrame) -> dict[str, Any]:
    """Compute every Phase 1 chart's data for the given DataFrame."""
    numeric_histograms = []
    categorical_bars = []
    text_length_histograms = []

    for column in df.columns:
        chart_type = classify_column(df, column)
        if chart_type == "numeric":
            numeric_histograms.append(numeric_histogram_data(df, column))
        elif chart_type == "categorical":
            categorical_bars.append(categorical_bar_data(df, column))
        elif chart_type == "text":
            text_length_histograms.append(text_length_histogram_data(df, column))
        # "unsupported" columns (e.g. Date/Datetime) get no per-column chart
        # yet, but are still included in the null heatmap below.

    return {
        "null_heatmap": null_heatmap_data(df),
        "numeric_histograms": numeric_histograms,
        "categorical_bars": categorical_bars,
        "text_length_histograms": text_length_histograms,
    }
