"""Headless embedding-column detection and PCA projection for Phase 2.

Pure functions over an already-loaded polars DataFrame — no pywebview or
Plotly imports here, mirroring chart_data.py's split between data
computation (this module) and rendering (frontend/app.js).
"""
from __future__ import annotations

from typing import Any

import numpy as np
import polars as pl
from sklearn.decomposition import PCA

EMBEDDING_MIN_DIMENSIONS = 8
PCA_SAMPLE_MAX_ROWS = 10_000
# Scatter is an "all-pairs" chart form (dataviz skill's palette.md): any two
# categories can land adjacent to each other anywhere in the cloud, unlike a
# bar chart where only neighboring bars are ever compared. Only the first 3
# categorical color slots validate colorblind-safety across every pair, so
# this caps lower than chart_data.TOP_N_CATEGORIES's top-10 (an
# adjacent-pairs form, e.g. bars/lines).
EMBEDDING_COLOR_TOP_N = 3

_FLOAT_DTYPES = (pl.Float32, pl.Float64)


def _list_column_dimensions(df: pl.DataFrame, column: str, dtype: pl.DataType) -> int | None:
    """Return the consistent non-null list length for a List(Float) column, or None.

    None means "not a candidate": no usable rows, or lengths that vary
    across rows (an embedding column should have one fixed dimensionality).
    """
    if dtype.inner not in _FLOAT_DTYPES:
        return None
    lengths = df[column].list.len().drop_nulls()
    if lengths.len() == 0 or lengths.n_unique() != 1:
        return None
    return int(lengths[0])


def _array_column_dimensions(dtype: pl.DataType) -> int | None:
    """Return the fixed width for an Array(Float) column, or None."""
    if dtype.inner not in _FLOAT_DTYPES:
        return None
    return int(dtype.size)


def detect_embedding_columns(df: pl.DataFrame) -> list[dict[str, object]]:
    """Find columns that look like embeddings: a list of floats with a
    consistent length across rows, greater than EMBEDDING_MIN_DIMENSIONS.

    Returns a list of {"column": name, "dimensions": n} dicts, in column
    order.
    """
    candidates = []
    for column, dtype in df.schema.items():
        if isinstance(dtype, pl.List):
            dimensions = _list_column_dimensions(df, column, dtype)
        elif isinstance(dtype, pl.Array):
            dimensions = _array_column_dimensions(dtype)
        else:
            dimensions = None

        if dimensions is not None and dimensions > EMBEDDING_MIN_DIMENSIONS:
            candidates.append({"column": column, "dimensions": dimensions})

    return candidates


def _bucket_top_n_labels(series: pl.Series, top_n: int) -> list[str]:
    """Map each value to itself if it's among the top-N most frequent
    non-null values, else "Other"; nulls become "Missing".

    Mirrors chart_data.categorical_bar_data's top-N-plus-Other convention,
    but per-row (aligned to embedding points) rather than aggregated counts.
    """
    non_null = series.drop_nulls()
    if non_null.len() == 0:
        top_values: set[Any] = set()
    else:
        counts_df = non_null.value_counts().sort("count", descending=True)
        top_values = set(counts_df[series.name].to_list()[:top_n])

    labels = []
    for value in series.to_list():
        if value is None:
            labels.append("Missing")
        elif value in top_values:
            labels.append(str(value))
        else:
            labels.append("Other")
    return labels


def embedding_projection_data(
    df: pl.DataFrame,
    column: str,
    categorical_columns: list[str],
    sample_max: int = PCA_SAMPLE_MAX_ROWS,
    top_n: int = EMBEDDING_COLOR_TOP_N,
) -> dict[str, Any]:
    """Project an embedding column to 2D via PCA, for scatter-plotting.

    Randomly samples down to `sample_max` rows first when the dataset is
    larger than that (see PROJECT_BRIEF.md's Phase 2 performance note), then
    drops rows where the embedding itself is null before fitting PCA.
    `categorical_columns` are returned as per-point labels (top-N + "Other"/
    "Missing", see _bucket_top_n_labels) so the frontend can color the
    scatter by any of them without a second round trip.
    """
    total_rows = df.height
    is_sample = total_rows > sample_max
    sampled = df.sample(n=sample_max) if is_sample else df
    sampled_rows = sampled.height

    usable = sampled.filter(pl.col(column).is_not_null())
    categorical_colors = {
        col: _bucket_top_n_labels(usable[col], top_n) for col in categorical_columns
    }

    if usable.height < 2:
        return {
            "column": column,
            "x": [],
            "y": [],
            "is_sample": is_sample,
            "sampled_rows": sampled_rows,
            "total_rows": total_rows,
            "categorical_colors": categorical_colors,
        }

    matrix = np.array(usable[column].to_list(), dtype=np.float64)
    coords = PCA(n_components=2).fit_transform(matrix)

    return {
        "column": column,
        "x": [float(v) for v in coords[:, 0]],
        "y": [float(v) for v in coords[:, 1]],
        "is_sample": is_sample,
        "sampled_rows": sampled_rows,
        "total_rows": total_rows,
        "categorical_colors": categorical_colors,
    }
