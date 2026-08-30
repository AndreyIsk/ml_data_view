"""Headless embedding-column detection and PCA projection for Phase 2.

Pure functions over a polars LazyFrame — no pywebview or Plotly imports
here, mirroring chart_data.py's split between data computation (this
module) and rendering (frontend/app.js). See data_loader.py's module
docstring for the lazy/streaming architecture note.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import polars as pl
# sklearn is NOT imported here: importing it costs ~554MB of resident
# memory (measured — polars alone is 32MB, +pyarrow 527MB, +sklearn
# 1082MB), and it is only needed to project a detected embedding column.
# A dataset with no embedding columns — the common case, and every Criteo
# column — would otherwise pay that on every load, leaving too little
# headroom for the chart computations themselves. Imported inside
# embedding_projection_data() instead, where it is actually used.

from ml_data_view.data_loader import row_count

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


def _list_column_dimensions(lf: pl.LazyFrame, column: str, dtype: pl.DataType) -> int | None:
    """Return the consistent non-null list length for a List(Float) column, or None.

    None means "not a candidate": no usable rows, or lengths that vary
    across rows (an embedding column should have one fixed dimensionality).
    A single small streaming aggregation (count/n_unique/first of the
    length expression) decides this — never a full column collect.
    """
    if dtype.inner not in _FLOAT_DTYPES:
        return None
    lengths = pl.col(column).list.len().drop_nulls()
    result = lf.select(
        lengths.count().alias("n"), lengths.n_unique().alias("nu"), lengths.first().alias("first")
    ).collect(engine="streaming")
    n, n_unique, first = result[0, "n"], result[0, "nu"], result[0, "first"]
    if n == 0 or n_unique != 1:
        return None
    return int(first)


def _array_column_dimensions(dtype: pl.DataType) -> int | None:
    """Return the fixed width for an Array(Float) column, or None. Purely
    a schema/dtype check — an Array's width is static, no data query
    needed."""
    if dtype.inner not in _FLOAT_DTYPES:
        return None
    return int(dtype.size)


def detect_embedding_columns(lf: pl.LazyFrame) -> list[dict[str, object]]:
    """Find columns that look like embeddings: a list of floats with a
    consistent length across rows, greater than EMBEDDING_MIN_DIMENSIONS.

    Returns a list of {"column": name, "dimensions": n} dicts, in column
    order. collect_schema() is metadata-only; each List-typed candidate
    gets one small bounded aggregation (see _list_column_dimensions).
    """
    candidates = []
    for column, dtype in lf.collect_schema().items():
        if isinstance(dtype, pl.List):
            dimensions = _list_column_dimensions(lf, column, dtype)
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
    Operates on an already-bounded in-memory Series (see
    embedding_projection_data — never more than sample_max rows), so this
    stays a plain eager Series op, same as before.
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
    lf: pl.LazyFrame,
    column: str,
    categorical_columns: list[str],
    total_rows: int | None = None,
    sample_max: int = PCA_SAMPLE_MAX_ROWS,
    top_n: int = EMBEDDING_COLOR_TOP_N,
) -> dict[str, Any]:
    """Project an embedding column to 2D via PCA, for scatter-plotting.

    Sampling is a bounded slice of the lazy frame — `.head(sample_max)` —
    rather than a true random sample across the whole file: polars'
    LazyFrame has no native random-sample operator that stays streaming-
    safe, and a bounded slice never pulls more than sample_max rows through
    the pipeline regardless of the source file's size, which is what
    matters for a dataset far larger than RAM. (This is a deliberate
    change from the old eager df.sample(): the sample is now the first
    sample_max rows rather than a uniformly random subset.) Rows where the
    embedding itself is null are dropped after that bounded collect.
    `categorical_columns` are returned as per-point labels (top-N +
    "Other"/"Missing", see _bucket_top_n_labels) so the frontend can color
    the scatter by any of them without a second round trip.
    """
    if total_rows is None:
        total_rows = row_count(lf)
    is_sample = total_rows > sample_max

    needed_columns = list(dict.fromkeys([column, *categorical_columns]))
    bounded_lf = lf.head(sample_max) if is_sample else lf
    sampled = bounded_lf.select(needed_columns).collect()
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

    # Deferred import — see the note where sklearn is deliberately not
    # imported at module level. Reached only when there is an embedding
    # column with enough usable rows to project.
    from sklearn.decomposition import PCA

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
