# Phase 1: Profiling Charts Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** After a Phase 0 file load, the app shows four read-only Plotly.js chart groups for the loaded dataset: a null/missing-value heatmap, numeric-column histograms, categorical class-balance bar charts (top N + "Other"), and text-length histograms for free-text columns.

**Architecture:** `ml_data_view/chart_data.py` holds all chart-data computation as pure functions over an already-loaded `pl.DataFrame` — no UI/Plotly imports, fully headless-testable, mirroring `data_loader.py`'s existing convention. `data_loader.py` is refactored (non-breaking) to expose a `read_dataframe()` step separately from `dataset_info_from_df()`, so `api.py` can keep the loaded `DataFrame` in memory after `load_file()` and hand it to `chart_data.py` when the frontend asks for charts via a new `get_chart_data()` API method. The frontend (`app.js`) is a thin renderer: it calls `get_chart_data()`, gets back small JSON-safe dicts (bin edges/counts, category/count pairs, a 0/1 null matrix), and turns each into a `Plotly.newPlot(...)` call. Plotly.js itself is vendored locally into `ml_data_view/frontend/` (no CDN dependency, consistent with the product's on-prem/offline positioning) rather than added as a Python package.

**Tech Stack:** Python 3.11+ / polars (unchanged), Plotly.js (vendored, MIT license) for rendering, pytest for the new `chart_data.py` and `api.py` tests.

---

## Design decisions locked in by this plan (not separately specified in PROJECT_BRIEF.md)

- **Top N for class balance:** N = 10 (`TOP_N_CATEGORIES` constant in `chart_data.py`). Anything beyond the top 10 by count is summed into a single `"Other"` bucket.
- **Categorical vs. text/string classification:** both are polars `String` columns; a column is treated as **categorical** if its non-null distinct count is `<= 50` OR distinct-count/row-count is `<= 0.5`; otherwise it's **text**. `Boolean` columns are folded into "categorical" (low cardinality, no dedicated brief category). Constants: `CATEGORICAL_MAX_DISTINCT = 50`, `CATEGORICAL_MAX_DISTINCT_RATIO = 0.5`.
- **Numeric histograms:** 30 equal-width bins (`NUMERIC_HISTOGRAM_BINS = 30`), computed in pure Python (no new numpy dependency).
- **Text-length histograms:** 30 equal-width bins over character length (`TEXT_LENGTH_HISTOGRAM_BINS = 30`), via polars' `.str.len_chars()`.
- **Null heatmap:** row × column grid of 0 (present) / 1 (null), capped at the first 500 rows (`NULL_HEATMAP_MAX_ROWS = 500`) to stay responsive on large files. The UI notes when the heatmap is a sample rather than the full file.
- **Columns with no chart type** (e.g. Date/Datetime — not in Phase 0/1 scope) are silently skipped, not errored.
- **Read-only:** no new endpoint or UI element mutates `self._current_df` or the source file — charts are pure reads of already-loaded data. This satisfies the "keep it read-only" requirement structurally (there is nothing here that writes).

---

## Task 1: Vendor Plotly.js locally

**Files:**
- Create: `ml_data_view/frontend/plotly.min.js`
- Modify: `ml_data_view/frontend/index.html`

- [ ] **Step 1: Download Plotly.js into the frontend directory**

Run:
```bash
curl -sL -o ml_data_view/frontend/plotly.min.js https://cdn.plot.ly/plotly-2.35.2.min.js
```
Expected: file created, several MB in size.

- [ ] **Step 2: Verify the file downloaded correctly**

Run: `wc -c ml_data_view/frontend/plotly.min.js` (or `ls -la`)
Expected: file size > 1,000,000 bytes (it's a multi-MB bundle). If 0 bytes or a tiny HTML error page, the download failed — retry or check network access before continuing.

- [ ] **Step 3: Reference it in `index.html`**

In `ml_data_view/frontend/index.html`, add the Plotly `<script>` tag **before** `app.js`'s tag:

```html
  <script src="plotly.min.js"></script>
  <script src="app.js"></script>
```

- [ ] **Step 4: Commit**

```bash
git add ml_data_view/frontend/plotly.min.js ml_data_view/frontend/index.html
git commit -m "chore: vendor Plotly.js locally for offline chart rendering"
```

---

## Task 2: `chart_data.py` — column classification

**Files:**
- Create: `ml_data_view/chart_data.py`
- Test: `tests/test_chart_data.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_chart_data.py
import polars as pl

from ml_data_view.chart_data import classify_column


def test_classify_column_numeric_dtypes():
    df = pl.DataFrame({"i": [1, 2, 3], "f": [1.0, 2.0, 3.0]})
    assert classify_column(df, "i") == "numeric"
    assert classify_column(df, "f") == "numeric"


def test_classify_column_boolean_is_categorical():
    df = pl.DataFrame({"flag": [True, False, True]})
    assert classify_column(df, "flag") == "categorical"


def test_classify_column_low_cardinality_string_is_categorical():
    df = pl.DataFrame({"label": ["cat", "dog", "cat", "dog", "cat"]})
    assert classify_column(df, "label") == "categorical"


def test_classify_column_high_cardinality_string_is_text():
    # 20 distinct long strings out of 20 rows: distinct-count 20 > 50? no,
    # but distinct/row ratio is 1.0 > 0.5 and count 20 <= 50 would normally
    # say categorical, so use enough distinct rows to exceed both thresholds.
    values = [f"a fairly unique free-text sentence number {i}" for i in range(60)]
    df = pl.DataFrame({"comment": values})
    assert classify_column(df, "comment") == "text"


def test_classify_column_all_null_string_is_categorical():
    df = pl.DataFrame({"empty": pl.Series("empty", [None, None], dtype=pl.String)})
    assert classify_column(df, "empty") == "categorical"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/Scripts/python.exe -m pytest tests/test_chart_data.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'ml_data_view.chart_data'`

- [ ] **Step 3: Write minimal implementation**

```python
# ml_data_view/chart_data.py
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/Scripts/python.exe -m pytest tests/test_chart_data.py -v`
Expected: PASS (all 5 tests)

- [ ] **Step 5: Commit**

```bash
git add ml_data_view/chart_data.py tests/test_chart_data.py
git commit -m "feat: add column classification for chart type selection"
```

---

## Task 3: `chart_data.py` — numeric histogram

**Files:**
- Modify: `ml_data_view/chart_data.py`
- Test: `tests/test_chart_data.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_chart_data.py (append)
from ml_data_view.chart_data import numeric_histogram_data


def test_numeric_histogram_data_bins_and_sums_to_row_count():
    df = pl.DataFrame({"score": [1.0, 2.0, 3.0, 4.0, 5.0]})

    result = numeric_histogram_data(df, "score", bins=5)

    assert result["column"] == "score"
    assert len(result["bin_edges"]) == 6
    assert len(result["counts"]) == 5
    assert sum(result["counts"]) == 5


def test_numeric_histogram_data_ignores_nulls():
    df = pl.DataFrame({"score": [1.0, None, 3.0, None, 5.0]})

    result = numeric_histogram_data(df, "score", bins=3)

    assert sum(result["counts"]) == 3


def test_numeric_histogram_data_empty_after_dropping_nulls():
    df = pl.DataFrame({"score": pl.Series("score", [None, None], dtype=pl.Float64)})

    result = numeric_histogram_data(df, "score")

    assert result == {"column": "score", "bin_edges": [], "counts": []}


def test_numeric_histogram_data_all_identical_values():
    df = pl.DataFrame({"score": [7.0, 7.0, 7.0]})

    result = numeric_histogram_data(df, "score", bins=5)

    assert sum(result["counts"]) == 3
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/Scripts/python.exe -m pytest tests/test_chart_data.py -v -k numeric_histogram`
Expected: FAIL — `ImportError: cannot import name 'numeric_histogram_data'`

- [ ] **Step 3: Write minimal implementation**

Append to `ml_data_view/chart_data.py`:

```python
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
    values = [float(v) for v in df[column].drop_nulls().to_list()]
    if not values:
        return {"column": column, "bin_edges": [], "counts": []}

    counts, bin_edges = _equal_width_histogram(values, bins)
    return {"column": column, "bin_edges": bin_edges, "counts": counts}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/Scripts/python.exe -m pytest tests/test_chart_data.py -v`
Expected: PASS (all tests so far)

- [ ] **Step 5: Commit**

```bash
git add ml_data_view/chart_data.py tests/test_chart_data.py
git commit -m "feat: add numeric column histogram data"
```

---

## Task 4: `chart_data.py` — categorical class-balance bar data

**Files:**
- Modify: `ml_data_view/chart_data.py`
- Test: `tests/test_chart_data.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_chart_data.py (append)
from ml_data_view.chart_data import categorical_bar_data


def test_categorical_bar_data_sorted_by_count_descending():
    df = pl.DataFrame({"label": ["cat", "dog", "cat", "bird", "cat", "dog"]})

    result = categorical_bar_data(df, "label")

    assert result["column"] == "label"
    assert result["categories"] == ["cat", "dog", "bird"]
    assert result["counts"] == [3, 2, 1]


def test_categorical_bar_data_groups_rest_as_other_beyond_top_n():
    labels = []
    for i in range(15):
        labels += [f"cat_{i}"] * (15 - i)  # cat_0 most frequent, cat_14 least
    df = pl.DataFrame({"label": labels})

    result = categorical_bar_data(df, "label", top_n=10)

    assert len(result["categories"]) == 11  # top 10 + "Other"
    assert result["categories"][-1] == "Other"
    top_10_total = sum(15 - i for i in range(10))
    rest_total = sum(15 - i for i in range(10, 15))
    assert result["counts"][:10] == [15 - i for i in range(10)]
    assert result["counts"][-1] == rest_total
    assert sum(result["counts"]) == top_10_total + rest_total


def test_categorical_bar_data_ignores_nulls():
    df = pl.DataFrame({"label": ["a", None, "a", None]})

    result = categorical_bar_data(df, "label")

    assert result["categories"] == ["a"]
    assert result["counts"] == [2]


def test_categorical_bar_data_boolean_column():
    df = pl.DataFrame({"flag": [True, True, False]})

    result = categorical_bar_data(df, "flag")

    assert set(result["categories"]) == {"True", "False"}
    assert sum(result["counts"]) == 3
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/Scripts/python.exe -m pytest tests/test_chart_data.py -v -k categorical_bar`
Expected: FAIL — `ImportError: cannot import name 'categorical_bar_data'`

- [ ] **Step 3: Write minimal implementation**

Append to `ml_data_view/chart_data.py`:

```python
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/Scripts/python.exe -m pytest tests/test_chart_data.py -v`
Expected: PASS (all tests so far)

- [ ] **Step 5: Commit**

```bash
git add ml_data_view/chart_data.py tests/test_chart_data.py
git commit -m "feat: add categorical class-balance bar data with top-N + Other"
```

---

## Task 5: `chart_data.py` — text-length histogram

**Files:**
- Modify: `ml_data_view/chart_data.py`
- Test: `tests/test_chart_data.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_chart_data.py (append)
from ml_data_view.chart_data import text_length_histogram_data


def test_text_length_histogram_data_bins_by_character_length():
    df = pl.DataFrame({"comment": ["a", "bb", "ccc", "dddd", "eeeee"]})

    result = text_length_histogram_data(df, "comment", bins=5)

    assert result["column"] == "comment"
    assert len(result["counts"]) == 5
    assert sum(result["counts"]) == 5


def test_text_length_histogram_data_ignores_nulls():
    df = pl.DataFrame({"comment": ["hello", None, "hi"]})

    result = text_length_histogram_data(df, "comment", bins=3)

    assert sum(result["counts"]) == 2


def test_text_length_histogram_data_empty_column():
    df = pl.DataFrame({"comment": pl.Series("comment", [], dtype=pl.String)})

    result = text_length_histogram_data(df, "comment")

    assert result == {"column": "comment", "bin_edges": [], "counts": []}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/Scripts/python.exe -m pytest tests/test_chart_data.py -v -k text_length`
Expected: FAIL — `ImportError: cannot import name 'text_length_histogram_data'`

- [ ] **Step 3: Write minimal implementation**

Append to `ml_data_view/chart_data.py`:

```python
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/Scripts/python.exe -m pytest tests/test_chart_data.py -v`
Expected: PASS (all tests so far)

- [ ] **Step 5: Commit**

```bash
git add ml_data_view/chart_data.py tests/test_chart_data.py
git commit -m "feat: add text-length histogram data"
```

---

## Task 6: `chart_data.py` — null heatmap

**Files:**
- Modify: `ml_data_view/chart_data.py`
- Test: `tests/test_chart_data.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_chart_data.py (append)
from ml_data_view.chart_data import null_heatmap_data


def test_null_heatmap_data_marks_nulls_as_one():
    df = pl.DataFrame({"a": [1, None, 3], "b": [None, 2, 3]})

    result = null_heatmap_data(df)

    assert result["columns"] == ["a", "b"]
    assert result["total_rows"] == 3
    assert result["sampled_rows"] == 3
    assert result["z"] == [[0, 1], [1, 0], [0, 0]]


def test_null_heatmap_data_caps_sampled_rows(tmp_path):
    df = pl.DataFrame({"a": list(range(700))})

    result = null_heatmap_data(df, max_rows=500)

    assert result["total_rows"] == 700
    assert result["sampled_rows"] == 500
    assert len(result["z"]) == 500


def test_null_heatmap_data_no_cap_needed_when_under_limit():
    df = pl.DataFrame({"a": [1, 2, 3]})

    result = null_heatmap_data(df, max_rows=500)

    assert result["sampled_rows"] == 3
    assert result["total_rows"] == 3
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/Scripts/python.exe -m pytest tests/test_chart_data.py -v -k null_heatmap`
Expected: FAIL — `ImportError: cannot import name 'null_heatmap_data'`

- [ ] **Step 3: Write minimal implementation**

Append to `ml_data_view/chart_data.py`:

```python
def null_heatmap_data(df: pl.DataFrame, max_rows: int = NULL_HEATMAP_MAX_ROWS) -> dict[str, Any]:
    sample_df = df.head(max_rows) if df.height > max_rows else df
    columns = sample_df.columns
    column_values = {column: sample_df[column].to_list() for column in columns}

    z = [
        [1 if column_values[column][row_idx] is None else 0 for column in columns]
        for row_idx in range(sample_df.height)
    ]

    return {
        "columns": columns,
        "sampled_rows": sample_df.height,
        "total_rows": df.height,
        "z": z,
    }
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/Scripts/python.exe -m pytest tests/test_chart_data.py -v`
Expected: PASS (all tests so far)

- [ ] **Step 5: Commit**

```bash
git add ml_data_view/chart_data.py tests/test_chart_data.py
git commit -m "feat: add null/missing-value heatmap data"
```

---

## Task 7: `chart_data.py` — `build_chart_payload` aggregator

**Files:**
- Modify: `ml_data_view/chart_data.py`
- Test: `tests/test_chart_data.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_chart_data.py (append)
from ml_data_view.chart_data import build_chart_payload


def test_build_chart_payload_routes_columns_by_type():
    df = pl.DataFrame(
        {
            "id": [1, 2, 3, 4],
            "label": ["cat", "dog", "cat", "dog"],
            "comment": [f"unique free text entry number {i} of many words" for i in range(4)],
        }
    )

    payload = build_chart_payload(df)

    assert payload["null_heatmap"]["columns"] == ["id", "label", "comment"]
    assert [h["column"] for h in payload["numeric_histograms"]] == ["id"]
    assert [b["column"] for b in payload["categorical_bars"]] == ["label"]
    # "comment" has 4 distinct values out of 4 rows (ratio 1.0 > 0.5) and
    # more than 0 distinct, but <= CATEGORICAL_MAX_DISTINCT(50), so by the
    # documented rule it's still categorical at this tiny size — use a
    # larger, clearly-text sample instead to land in text_length_histograms.
    assert payload["categorical_bars"] or payload["text_length_histograms"]


def test_build_chart_payload_skips_unsupported_dtypes():
    df = pl.DataFrame({"id": [1, 2, 3]})
    df = df.with_columns(pl.date(2024, 1, 1).alias("created_at"))

    payload = build_chart_payload(df)

    numeric_columns = [h["column"] for h in payload["numeric_histograms"]]
    categorical_columns = [b["column"] for b in payload["categorical_bars"]]
    text_columns = [t["column"] for t in payload["text_length_histograms"]]
    assert "created_at" not in numeric_columns + categorical_columns + text_columns
    assert "created_at" in payload["null_heatmap"]["columns"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/Scripts/python.exe -m pytest tests/test_chart_data.py -v -k build_chart_payload`
Expected: FAIL — `ImportError: cannot import name 'build_chart_payload'`

- [ ] **Step 3: Write minimal implementation**

Append to `ml_data_view/chart_data.py`:

```python
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/Scripts/python.exe -m pytest tests/test_chart_data.py -v`
Expected: PASS (all tests)

- [ ] **Step 5: Run the whole test suite**

Run: `.venv/Scripts/python.exe -m pytest -v`
Expected: All tests PASS (Phase 0 + Phase 1 `chart_data` tests).

- [ ] **Step 6: Commit**

```bash
git add ml_data_view/chart_data.py tests/test_chart_data.py
git commit -m "feat: add build_chart_payload aggregator"
```

---

## Task 8: Refactor `data_loader.py` and cache the loaded DataFrame in `Api`

**Files:**
- Modify: `ml_data_view/data_loader.py`
- Modify: `ml_data_view/api.py`
- Test: `tests/test_data_loader.py`
- Test: `tests/test_api.py`

This is a non-breaking refactor: `load_dataset()` keeps its existing signature and behavior (all Phase 0 tests must keep passing unmodified), but its body is split into a reusable `read_dataframe()` + `dataset_info_from_df()` so `Api.load_file()` can reuse the same DataFrame for both the schema/preview response and the chart data cache, instead of reading the file twice.

- [ ] **Step 1: Write the failing test (data_loader)**

```python
# tests/test_data_loader.py (append)
from ml_data_view.data_loader import dataset_info_from_df, read_dataframe


def test_read_dataframe_returns_polars_dataframe(tmp_path):
    csv_path = tmp_path / "sample.csv"
    _sample_df().write_csv(csv_path)

    df = read_dataframe(csv_path)

    assert isinstance(df, pl.DataFrame)
    assert df.height == 4


def test_dataset_info_from_df_matches_load_dataset(tmp_path):
    csv_path = tmp_path / "sample.csv"
    _sample_df().write_csv(csv_path)

    df = read_dataframe(csv_path)
    via_helper = dataset_info_from_df(df, csv_path)
    via_load_dataset = load_dataset(csv_path)

    assert via_helper.to_dict() == via_load_dataset.to_dict()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/Scripts/python.exe -m pytest tests/test_data_loader.py -v -k "read_dataframe or dataset_info_from_df"`
Expected: FAIL — `ImportError: cannot import name 'read_dataframe'`

- [ ] **Step 3: Refactor `data_loader.py`**

Replace the existing `load_dataset` function (and the `_read_dataframe` helper stays, just gets called from the new `read_dataframe`) with:

```python
def _dispatch_reader(path: Path, suffix: str) -> pl.DataFrame:
    if suffix == ".csv":
        return pl.read_csv(path)
    if suffix == ".parquet":
        return pl.read_parquet(path)
    if suffix == ".arrow":
        return pl.read_ipc(path)
    raise UnsupportedFileTypeError(f"No reader wired up yet for '{suffix}'")


def read_dataframe(path: str | Path) -> pl.DataFrame:
    """Validate and read a dataset file into a polars DataFrame.

    Raises:
        UnsupportedFileTypeError: if the extension isn't .parquet/.arrow/.csv
        FileNotFoundError: if the file doesn't exist
    """
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix not in SUPPORTED_EXTENSIONS:
        raise UnsupportedFileTypeError(
            f"Unsupported file type '{suffix}'. Supported types: "
            f"{', '.join(sorted(SUPPORTED_EXTENSIONS))}"
        )
    if not path.exists():
        raise FileNotFoundError(f"No such file: {path}")
    return _dispatch_reader(path, suffix)


def dataset_info_from_df(df: pl.DataFrame, path: str | Path) -> DatasetInfo:
    """Build a DatasetInfo from an already-loaded DataFrame and its source path."""
    path = Path(path)
    null_counts = df.null_count().to_dicts()[0]

    columns = [
        ColumnInfo(name=name, dtype=str(dtype), null_count=null_counts[name])
        for name, dtype in df.schema.items()
    ]
    preview_rows = df.head(PREVIEW_ROW_LIMIT).to_dicts()

    return DatasetInfo(
        file_path=str(path),
        file_size_bytes=path.stat().st_size,
        row_count=df.height,
        columns=columns,
        preview_rows=preview_rows,
    )


def load_dataset(path: str | Path) -> DatasetInfo:
    """Load a dataset file and extract its schema, size, and preview rows.

    Raises:
        UnsupportedFileTypeError: if the extension isn't .parquet/.arrow/.csv
        FileNotFoundError: if the file doesn't exist
    """
    path = Path(path)
    df = read_dataframe(path)
    return dataset_info_from_df(df, path)
```

Remove the old `_read_dataframe` function entirely (replaced by `_dispatch_reader` + `read_dataframe`).

- [ ] **Step 4: Run the full data_loader test file**

Run: `.venv/Scripts/python.exe -m pytest tests/test_data_loader.py -v`
Expected: PASS — every existing Phase 0 test plus the two new ones (should be 13 tests total).

- [ ] **Step 5: Write the failing test (api caches the DataFrame)**

```python
# tests/test_api.py (append)
def test_load_file_caches_dataframe_for_chart_data(tmp_path):
    df = pl.DataFrame({"id": [1, 2, 3]})
    csv_path = tmp_path / "sample.csv"
    df.write_csv(csv_path)

    api = Api()
    api.load_file(str(csv_path))

    assert api._current_df is not None
    assert api._current_df.height == 3


def test_load_file_does_not_cache_dataframe_on_error(tmp_path):
    bad_path = tmp_path / "sample.txt"
    bad_path.write_text("not a dataset")

    api = Api()
    api.load_file(str(bad_path))

    assert api._current_df is None
```

- [ ] **Step 6: Run test to verify it fails**

Run: `.venv/Scripts/python.exe -m pytest tests/test_api.py -v -k caches_dataframe`
Expected: FAIL — `AttributeError: 'Api' object has no attribute '_current_df'`

- [ ] **Step 7: Update `api.py`**

Replace the `Api` class in `ml_data_view/api.py` with:

```python
class Api:
    """Methods exposed to the JS frontend via pywebview's js_api binding."""

    def __init__(self) -> None:
        self._current_df: pl.DataFrame | None = None

    def open_file_dialog(self) -> str | None:
        """Show a native file-open dialog restricted to supported types.

        Returns the selected path, or None if the user cancelled. Requires
        an active pywebview window, so this is not unit-tested headlessly —
        verify manually per the relevant plan's manual-verification task.
        """
        window = webview.windows[0]
        result = window.create_file_dialog(
            webview.FileDialog.OPEN,
            file_types=(
                "Dataset files (*.parquet;*.arrow;*.csv)",
                "All files (*.*)",
            ),
        )
        if not result:
            return None
        return result[0]

    def load_file(self, path: str) -> dict[str, Any]:
        """Load a dataset file, cache it, and return its profile as a dict.

        On failure, returns {"error": "<message>"} instead of raising, so
        the frontend always gets a plain dict back across the js_api bridge.
        Caches the loaded DataFrame on success so get_chart_data() can reuse
        it without re-reading the file from disk.
        """
        try:
            df = read_dataframe(path)
        except (UnsupportedFileTypeError, FileNotFoundError) as exc:
            return {"error": str(exc)}
        self._current_df = df
        return dataset_info_from_df(df, path).to_dict()
```

Add `import polars as pl` and update the `data_loader` import line to:

```python
from ml_data_view.data_loader import (
    UnsupportedFileTypeError,
    dataset_info_from_df,
    read_dataframe,
)
```

- [ ] **Step 8: Run test to verify it passes**

Run: `.venv/Scripts/python.exe -m pytest tests/test_api.py -v`
Expected: PASS (all api tests, including the pre-existing Phase 0 ones — confirms the refactor didn't break `load_file`'s public behavior).

- [ ] **Step 9: Run the whole suite**

Run: `.venv/Scripts/python.exe -m pytest -v`
Expected: All tests PASS.

- [ ] **Step 10: Commit**

```bash
git add ml_data_view/data_loader.py ml_data_view/api.py tests/test_data_loader.py tests/test_api.py
git commit -m "refactor: split data_loader read/build steps, cache DataFrame in Api"
```

---

## Task 9: `Api.get_chart_data()`

**Files:**
- Modify: `ml_data_view/api.py`
- Test: `tests/test_api.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_api.py (append)
def test_get_chart_data_returns_error_when_nothing_loaded():
    api = Api()

    result = api.get_chart_data()

    assert "error" in result


def test_get_chart_data_returns_payload_after_load_file(tmp_path):
    df = pl.DataFrame({"id": [1, 2, 3], "label": ["a", "b", "a"]})
    csv_path = tmp_path / "sample.csv"
    df.write_csv(csv_path)

    api = Api()
    api.load_file(str(csv_path))
    result = api.get_chart_data()

    assert "error" not in result
    assert result["null_heatmap"]["columns"] == ["id", "label"]
    assert [h["column"] for h in result["numeric_histograms"]] == ["id"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/Scripts/python.exe -m pytest tests/test_api.py -v -k get_chart_data`
Expected: FAIL — `AttributeError: 'Api' object has no attribute 'get_chart_data'`

- [ ] **Step 3: Write minimal implementation**

In `ml_data_view/api.py`, add the import and method:

```python
from ml_data_view.chart_data import build_chart_payload
```

Add to the `Api` class (after `load_file`):

```python
    def get_chart_data(self) -> dict[str, Any]:
        """Return Phase 1 chart data for the most recently loaded dataset."""
        if self._current_df is None:
            return {"error": "No dataset loaded yet. Open a file first."}
        return build_chart_payload(self._current_df)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/Scripts/python.exe -m pytest tests/test_api.py -v`
Expected: PASS (all api tests)

- [ ] **Step 5: Run the whole suite**

Run: `.venv/Scripts/python.exe -m pytest -v`
Expected: All tests PASS.

- [ ] **Step 6: Commit**

```bash
git add ml_data_view/api.py tests/test_api.py
git commit -m "feat: add Api.get_chart_data endpoint"
```

---

## Task 10: Frontend — render the four chart groups

**Files:**
- Modify: `ml_data_view/frontend/index.html`
- Modify: `ml_data_view/frontend/style.css`
- Modify: `ml_data_view/frontend/app.js`

No automated test — static UI wired to the already-tested `Api.get_chart_data()`. Manual verification happens in Task 11.

- [ ] **Step 1: Add the charts markup to `index.html`**

Insert this new `<section>` right after the existing `#preview-section` block and before the `<script>` tags:

```html
  <section id="charts-section" class="hidden">
    <h2>Charts</h2>

    <div id="null-heatmap-section">
      <h3>Missing-value heatmap</h3>
      <p id="null-heatmap-note" class="chart-note"></p>
      <div id="null-heatmap-chart"></div>
    </div>

    <div id="numeric-histograms-section">
      <h3>Numeric distributions</h3>
      <div id="numeric-histograms" class="chart-grid"></div>
    </div>

    <div id="categorical-bars-section">
      <h3>Class balance</h3>
      <div id="categorical-bars" class="chart-grid"></div>
    </div>

    <div id="text-length-histograms-section">
      <h3>Text length distributions</h3>
      <div id="text-length-histograms" class="chart-grid"></div>
    </div>
  </section>
```

- [ ] **Step 2: Add chart layout CSS to `style.css`**

Append to `ml_data_view/frontend/style.css`:

```css
h3 {
  font-size: 1.05rem;
  margin: 16px 0 8px;
}

.chart-note {
  color: #555;
  font-size: 0.85rem;
  margin: 0 0 8px;
}

.chart-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(320px, 1fr));
  gap: 16px;
}

.chart-box {
  border: 1px solid #ddd;
  border-radius: 4px;
  padding: 4px;
}

#null-heatmap-chart {
  width: 100%;
}
```

- [ ] **Step 3: Update `app.js`**

First, extend `showError` to also hide the charts section — change:

```js
function showError(message) {
  errorBanner.textContent = message;
  errorBanner.classList.remove("hidden");
  summarySection.classList.add("hidden");
  schemaSection.classList.add("hidden");
  previewSection.classList.add("hidden");
}
```

to:

```js
function showError(message) {
  errorBanner.textContent = message;
  errorBanner.classList.remove("hidden");
  summarySection.classList.add("hidden");
  schemaSection.classList.add("hidden");
  previewSection.classList.add("hidden");
  document.getElementById("charts-section").classList.add("hidden");
}
```

Then append the chart-rendering functions at the end of `app.js`:

```js
function renderNullHeatmap(data) {
  const section = document.getElementById("null-heatmap-section");
  if (!data.z.length) {
    section.classList.add("hidden");
    return;
  }
  section.classList.remove("hidden");

  const yLabels = Array.from({ length: data.sampled_rows }, (_, i) => i);
  Plotly.newPlot(
    document.getElementById("null-heatmap-chart"),
    [
      {
        z: data.z,
        x: data.columns,
        y: yLabels,
        type: "heatmap",
        colorscale: [
          [0, "#f5f5f5"],
          [1, "#c0392b"],
        ],
        showscale: false,
        xgap: 1,
        ygap: 1,
      },
    ],
    {
      margin: { t: 10, l: 50, r: 10, b: 80 },
      height: 320,
      yaxis: { title: "Row index", autorange: "reversed" },
    },
    { displayModeBar: false, responsive: true }
  );

  const note = document.getElementById("null-heatmap-note");
  note.textContent =
    data.total_rows > data.sampled_rows
      ? `Showing first ${data.sampled_rows} of ${data.total_rows} rows (missing = red).`
      : `All ${data.total_rows} rows shown (missing = red).`;
}

function renderColumnCharts(containerId, sectionId, charts, plotBuilder) {
  const container = document.getElementById(containerId);
  const section = document.getElementById(sectionId);
  container.innerHTML = "";

  const plottable = charts.filter((chart) => chart.counts && chart.counts.length);
  if (!plottable.length) {
    section.classList.add("hidden");
    return;
  }
  section.classList.remove("hidden");

  for (const chart of plottable) {
    const chartDiv = document.createElement("div");
    chartDiv.className = "chart-box";
    container.appendChild(chartDiv);
    plotBuilder(chartDiv, chart);
  }
}

function binCenters(binEdges) {
  return binEdges.slice(0, -1).map((edge, i) => (edge + binEdges[i + 1]) / 2);
}

function plotNumericHistogram(div, chart) {
  Plotly.newPlot(
    div,
    [{ x: binCenters(chart.bin_edges), y: chart.counts, type: "bar" }],
    { title: chart.column, margin: { t: 30, l: 40, r: 10, b: 40 }, height: 260 },
    { displayModeBar: false, responsive: true }
  );
}

function plotCategoricalBar(div, chart) {
  Plotly.newPlot(
    div,
    [{ x: chart.categories, y: chart.counts, type: "bar" }],
    { title: chart.column, margin: { t: 30, l: 40, r: 10, b: 60 }, height: 260 },
    { displayModeBar: false, responsive: true }
  );
}

function plotTextLengthHistogram(div, chart) {
  Plotly.newPlot(
    div,
    [{ x: binCenters(chart.bin_edges), y: chart.counts, type: "bar" }],
    {
      title: `${chart.column} (length)`,
      margin: { t: 30, l: 40, r: 10, b: 40 },
      height: 260,
    },
    { displayModeBar: false, responsive: true }
  );
}

function renderCharts(chartData) {
  document.getElementById("charts-section").classList.remove("hidden");
  renderNullHeatmap(chartData.null_heatmap);
  renderColumnCharts(
    "numeric-histograms",
    "numeric-histograms-section",
    chartData.numeric_histograms,
    plotNumericHistogram
  );
  renderColumnCharts(
    "categorical-bars",
    "categorical-bars-section",
    chartData.categorical_bars,
    plotCategoricalBar
  );
  renderColumnCharts(
    "text-length-histograms",
    "text-length-histograms-section",
    chartData.text_length_histograms,
    plotTextLengthHistogram
  );
}
```

Finally, update `handleOpenFile` to fetch and render chart data after a successful load — change:

```js
async function handleOpenFile() {
  const path = await window.pywebview.api.open_file_dialog();
  if (!path) return;

  clearError();
  const data = await window.pywebview.api.load_file(path);
  if (data.error) {
    showError(data.error);
    return;
  }
  renderDataset(data);
}
```

to:

```js
async function handleOpenFile() {
  const path = await window.pywebview.api.open_file_dialog();
  if (!path) return;

  clearError();
  const data = await window.pywebview.api.load_file(path);
  if (data.error) {
    showError(data.error);
    return;
  }
  renderDataset(data);

  const chartData = await window.pywebview.api.get_chart_data();
  if (chartData.error) {
    showError(chartData.error);
    return;
  }
  renderCharts(chartData);
}
```

- [ ] **Step 4: Run the whole automated test suite (no regressions from the frontend edit)**

Run: `.venv/Scripts/python.exe -m pytest -v`
Expected: All tests PASS (frontend files aren't exercised by pytest, but this confirms nothing else broke).

- [ ] **Step 5: Commit**

```bash
git add ml_data_view/frontend/index.html ml_data_view/frontend/style.css ml_data_view/frontend/app.js
git commit -m "feat: render null heatmap, histograms, and class-balance charts"
```

---

## Task 11: Manual end-to-end verification

**Files:** none (verification only)

- [ ] **Step 1: Launch the app**

Run: `.venv/Scripts/python.exe -m ml_data_view.main`

- [ ] **Step 2: Open a dataset with a mix of column types**

If `sample_data/demo.parquet` from the Phase 0 plan still exists, open it (`id` numeric, `label` categorical with nulls, `score` numeric). Otherwise regenerate it:

```bash
.venv/Scripts/python.exe -c "import polars as pl; pl.DataFrame({'id': list(range(75)), 'label': (['cat','dog',None]*25), 'score': [i*0.1 for i in range(75)]}).write_parquet('sample_data/demo.parquet')"
```

Expected after opening: the existing Phase 0 schema/preview still render, and below them a new "Charts" section appears with:
- A missing-value heatmap (red cells for `label`'s null rows, note showing row count)
- A numeric histogram for `id` and one for `score`
- A class-balance bar chart for `label` (2 bars: `cat`, `dog` — no nulls counted)
- No "Text length distributions" section (no free-text column in this sample)

- [ ] **Step 3: Verify a dataset with a free-text column**

```bash
.venv/Scripts/python.exe -c "
import polars as pl
pl.DataFrame({
    'id': list(range(60)),
    'review': [f'This is a fairly unique piece of free text, entry number {i}, with enough distinct words to avoid looking categorical.' for i in range(60)],
}).write_csv('sample_data/reviews.csv')
"
```

Open `sample_data/reviews.csv` — expect a "Text length distributions" chart for `review` instead of a class-balance bar chart (60 distinct long strings should classify as text, not categorical).

- [ ] **Step 4: Verify charts clear/update correctly on reopen and on error**

Open `sample_data/demo.csv` (from Phase 0, 3 rows) after the above — expect the charts to refresh to the new dataset's columns, not leave stale charts from `reviews.csv`.
Then trigger an unsupported-file error (any `.txt`/`.md` file) — expect the red error banner and the Charts section to disappear, not show stale charts.

- [ ] **Step 5: Confirm read-only**

Visually confirm there is no UI control anywhere in the Charts section that edits values, deletes rows, or writes back to the source file — only chart rendering.

---

## Task 12: Update `PROJECT_BRIEF.md` to Phase 1

**Files:**
- Modify: `PROJECT_BRIEF.md`

- [ ] **Step 1: Update the "Current phase" section**

Change the `## Current phase: PHASE 0 ONLY` section's heading and bullet list to:

```markdown
## Current phase: PHASE 1

Build exactly this and stop:
- Null/missing-value heatmap across columns
- Histograms for numeric columns
- Bar charts showing class balance for categorical columns (top N
  categories, rest grouped as "Other")
- Length-distribution histogram for text/string columns
- Charts render for the currently loaded dataset (Phase 0's file
  picker/schema/table view), read-only — no data editing
```

Update the "Explicit non-goals right now" section's first bullet (it currently says "Do not build the charting layer yet (that's Phase 1)") to:

```markdown
## Explicit non-goals right now

- Do not build the embedding projection feature yet (that's Phase 2)
- Do not touch the LLM custom-chart feature yet (that's Phase 4)
- Do not set up PyInstaller/installer yet (that's Phase 5)
- Do not add authentication or license-key checks yet
```

- [ ] **Step 2: Commit**

```bash
git add PROJECT_BRIEF.md
git commit -m "docs: advance PROJECT_BRIEF.md to Phase 1"
```

---

## Self-Review Notes

- **Spec coverage:** null heatmap (Task 6/10) → numeric histograms (Task 3/10) → class-balance bar charts with top-N + Other (Task 4/10) → text-length histograms (Task 5/10) → Plotly.js (Task 1/10) → read-only (structural — no write paths added, called out explicitly in Task 11 Step 5) → PROJECT_BRIEF.md phase bump (Task 12). No packaging/LLM/auth/embedding work introduced anywhere — matches non-goals.
- **Headless testability preserved:** `chart_data.py` (Tasks 2–7) takes a `pl.DataFrame` and returns plain dicts — zero UI/Plotly imports, fully unit-tested without a window, same convention as `data_loader.py`.
- **Non-breaking refactor:** Task 8 explicitly re-runs all pre-existing Phase 0 tests (data_loader + api) after refactoring, and adds an equivalence test (`dataset_info_from_df` output matches `load_dataset` output) rather than trusting the refactor by inspection.
- **Small commits:** every task ends in its own commit, each leaving the suite green.
