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
