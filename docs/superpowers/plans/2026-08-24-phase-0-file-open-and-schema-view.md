# Phase 0: Open File & Schema View — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A pywebview desktop app where the user picks a `.parquet`, `.arrow`, or `.csv` file and sees its column names/dtypes/null counts, row count, file size, and the first 50 rows in a scrollable table.

**Architecture:** `ml_data_view/data_loader.py` holds all data-reading logic (polars/pyarrow) behind a single `load_dataset()` function returning plain dataclasses — importable and testable with no UI running. `ml_data_view/api.py` is a thin pywebview-facing wrapper that turns `load_dataset()` results into JSON-safe dicts and adds the native file-open dialog. `ml_data_view/main.py` wires up the pywebview window. The frontend is plain HTML/CSS/JS (`ml_data_view/frontend/`) with no framework and no charting library yet.

**Tech Stack:** Python 3.11+ (verified working on 3.14), polars, pyarrow, pywebview, pytest.

---

## Confirmed environment

A `.venv` already exists at the repo root with polars 1.44.0, pyarrow 25.0.1, pywebview, and pytest 9.1.1 installed and importable. Every "Run" command below assumes the venv's Python: on Windows that's `.venv\Scripts\python.exe` (PowerShell/cmd) or `./.venv/Scripts/python.exe` (git-bash/PowerShell-with-forward-slashes — this repo's Bash tool uses Git Bash).

## Task 1: Project scaffolding

**Files:**
- Create: `requirements.txt`
- Create: `pytest.ini`
- Create: `.gitignore`
- Create: `ml_data_view/__init__.py`
- Create: `ml_data_view/frontend/.gitkeep`
- Create: `tests/__init__.py`

- [ ] **Step 1: Create `requirements.txt`**

```
polars>=1.0
pyarrow>=15.0
pywebview>=5.0
pytest>=8.0
```

- [ ] **Step 2: Create `pytest.ini`**

```ini
[pytest]
pythonpath = .
testpaths = tests
```

- [ ] **Step 3: Create `.gitignore`**

```
.venv/
__pycache__/
*.pyc
.pytest_cache/
*.egg-info/
```

- [ ] **Step 4: Create empty package markers**

Create `ml_data_view/__init__.py` (empty file).
Create `tests/__init__.py` (empty file).
Create `ml_data_view/frontend/.gitkeep` (empty file — placeholder until Task 10 adds real frontend files).

- [ ] **Step 5: Verify pytest collects with no errors**

Run: `.venv/Scripts/python.exe -m pytest --collect-only`
Expected: `no tests ran` (or `collected 0 items`), no import errors.

- [ ] **Step 6: Commit**

```bash
git add requirements.txt pytest.ini .gitignore ml_data_view/__init__.py ml_data_view/frontend/.gitkeep tests/__init__.py
git commit -m "chore: project scaffolding for Phase 0"
```

---

## Task 2: Data model — `ColumnInfo` / `DatasetInfo` dataclasses

**Files:**
- Create: `ml_data_view/data_loader.py`
- Test: `tests/test_data_loader.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_data_loader.py
from ml_data_view.data_loader import ColumnInfo, DatasetInfo


def test_dataset_info_to_dict_is_json_safe_shape():
    dataset_info = DatasetInfo(
        file_path="C:/data/sample.csv",
        file_size_bytes=1234,
        row_count=2,
        columns=[
            ColumnInfo(name="id", dtype="Int64", null_count=0),
            ColumnInfo(name="label", dtype="String", null_count=1),
        ],
        preview_rows=[{"id": 1, "label": "cat"}, {"id": 2, "label": None}],
    )

    result = dataset_info.to_dict()

    assert result == {
        "file_path": "C:/data/sample.csv",
        "file_size_bytes": 1234,
        "row_count": 2,
        "columns": [
            {"name": "id", "dtype": "Int64", "null_count": 0},
            {"name": "label", "dtype": "String", "null_count": 1},
        ],
        "preview_rows": [{"id": 1, "label": "cat"}, {"id": 2, "label": None}],
    }
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/Scripts/python.exe -m pytest tests/test_data_loader.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'ml_data_view.data_loader'` (or `ImportError`).

- [ ] **Step 3: Write minimal implementation**

```python
# ml_data_view/data_loader.py
"""Headless data-loading logic for ML Dataset Profiler.

Reads .parquet, .arrow, and .csv files and extracts the schema/size/
preview info the UI needs. No pywebview or UI imports in this module,
so it can be imported and unit-tested without launching the app.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import polars as pl

PREVIEW_ROW_LIMIT = 50
SUPPORTED_EXTENSIONS = {".parquet", ".arrow", ".csv"}


class UnsupportedFileTypeError(ValueError):
    """Raised when the given file extension is not one we support."""


@dataclass
class ColumnInfo:
    name: str
    dtype: str
    null_count: int

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "dtype": self.dtype, "null_count": self.null_count}


@dataclass
class DatasetInfo:
    file_path: str
    file_size_bytes: int
    row_count: int
    columns: list[ColumnInfo]
    preview_rows: list[dict[str, Any]]

    def to_dict(self) -> dict[str, Any]:
        return {
            "file_path": self.file_path,
            "file_size_bytes": self.file_size_bytes,
            "row_count": self.row_count,
            "columns": [column.to_dict() for column in self.columns],
            "preview_rows": self.preview_rows,
        }
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/Scripts/python.exe -m pytest tests/test_data_loader.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add ml_data_view/data_loader.py tests/test_data_loader.py
git commit -m "feat: add ColumnInfo/DatasetInfo data model"
```

---

## Task 3: `load_dataset()` — CSV support

**Files:**
- Modify: `ml_data_view/data_loader.py`
- Test: `tests/test_data_loader.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_data_loader.py (append)
import polars as pl

from ml_data_view.data_loader import load_dataset


def _sample_df() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "id": [1, 2, 3, 4],
            "label": ["cat", "dog", None, "cat"],
            "score": [0.9, None, 0.75, 0.5],
        }
    )


def test_load_csv_reads_schema_nulls_and_row_count(tmp_path):
    csv_path = tmp_path / "sample.csv"
    _sample_df().write_csv(csv_path)

    dataset_info = load_dataset(csv_path)

    assert dataset_info.row_count == 4
    assert dataset_info.file_size_bytes == csv_path.stat().st_size
    assert dataset_info.file_path == str(csv_path)

    columns_by_name = {c.name: c for c in dataset_info.columns}
    assert set(columns_by_name) == {"id", "label", "score"}
    assert columns_by_name["label"].null_count == 1
    assert columns_by_name["score"].null_count == 1
    assert columns_by_name["id"].null_count == 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/Scripts/python.exe -m pytest tests/test_data_loader.py -v`
Expected: FAIL — `ImportError: cannot import name 'load_dataset'`

- [ ] **Step 3: Write minimal implementation**

Append to `ml_data_view/data_loader.py`:

```python
def _read_dataframe(path: Path, suffix: str) -> pl.DataFrame:
    if suffix == ".csv":
        return pl.read_csv(path)
    raise UnsupportedFileTypeError(f"No reader wired up yet for '{suffix}'")


def load_dataset(path: str | Path) -> DatasetInfo:
    """Load a dataset file and extract its schema, size, and preview rows.

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

    df = _read_dataframe(path, suffix)
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/Scripts/python.exe -m pytest tests/test_data_loader.py -v`
Expected: PASS (all tests so far)

- [ ] **Step 5: Commit**

```bash
git add ml_data_view/data_loader.py tests/test_data_loader.py
git commit -m "feat: load_dataset reads CSV files"
```

---

## Task 4: `load_dataset()` — Parquet support

**Files:**
- Modify: `ml_data_view/data_loader.py`
- Test: `tests/test_data_loader.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_data_loader.py (append)
def test_load_parquet_reads_schema_nulls_and_row_count(tmp_path):
    parquet_path = tmp_path / "sample.parquet"
    _sample_df().write_parquet(parquet_path)

    dataset_info = load_dataset(parquet_path)

    assert dataset_info.row_count == 4
    columns_by_name = {c.name: c for c in dataset_info.columns}
    assert columns_by_name["label"].null_count == 1
    assert columns_by_name["id"].dtype == "Int64"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/Scripts/python.exe -m pytest tests/test_data_loader.py -v`
Expected: FAIL — `UnsupportedFileTypeError: No reader wired up yet for '.parquet'`

- [ ] **Step 3: Write minimal implementation**

In `ml_data_view/data_loader.py`, update `_read_dataframe`:

```python
def _read_dataframe(path: Path, suffix: str) -> pl.DataFrame:
    if suffix == ".csv":
        return pl.read_csv(path)
    if suffix == ".parquet":
        return pl.read_parquet(path)
    raise UnsupportedFileTypeError(f"No reader wired up yet for '{suffix}'")
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/Scripts/python.exe -m pytest tests/test_data_loader.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add ml_data_view/data_loader.py tests/test_data_loader.py
git commit -m "feat: load_dataset reads Parquet files"
```

---

## Task 5: `load_dataset()` — Arrow (IPC) support

**Files:**
- Modify: `ml_data_view/data_loader.py`
- Test: `tests/test_data_loader.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_data_loader.py (append)
def test_load_arrow_reads_schema_nulls_and_row_count(tmp_path):
    arrow_path = tmp_path / "sample.arrow"
    _sample_df().write_ipc(arrow_path)

    dataset_info = load_dataset(arrow_path)

    assert dataset_info.row_count == 4
    columns_by_name = {c.name: c for c in dataset_info.columns}
    assert columns_by_name["score"].null_count == 1
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/Scripts/python.exe -m pytest tests/test_data_loader.py -v`
Expected: FAIL — `UnsupportedFileTypeError: No reader wired up yet for '.arrow'`

- [ ] **Step 3: Write minimal implementation**

In `ml_data_view/data_loader.py`, update `_read_dataframe`:

```python
def _read_dataframe(path: Path, suffix: str) -> pl.DataFrame:
    if suffix == ".csv":
        return pl.read_csv(path)
    if suffix == ".parquet":
        return pl.read_parquet(path)
    if suffix == ".arrow":
        return pl.read_ipc(path)
    raise UnsupportedFileTypeError(f"No reader wired up yet for '{suffix}'")
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/Scripts/python.exe -m pytest tests/test_data_loader.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add ml_data_view/data_loader.py tests/test_data_loader.py
git commit -m "feat: load_dataset reads Arrow IPC files"
```

---

## Task 6: Error handling — unsupported extension and missing file

**Files:**
- Test: `tests/test_data_loader.py`

(No implementation changes expected — `load_dataset` already raises both errors from Task 3. This task locks the behavior in with tests.)

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_data_loader.py (append)
import pytest

from ml_data_view.data_loader import UnsupportedFileTypeError


def test_load_dataset_raises_on_unsupported_extension(tmp_path):
    bad_path = tmp_path / "sample.txt"
    bad_path.write_text("not a dataset")

    with pytest.raises(UnsupportedFileTypeError):
        load_dataset(bad_path)


def test_load_dataset_raises_on_missing_file(tmp_path):
    missing_path = tmp_path / "does_not_exist.csv"

    with pytest.raises(FileNotFoundError):
        load_dataset(missing_path)
```

- [ ] **Step 2: Run tests to verify current behavior**

Run: `.venv/Scripts/python.exe -m pytest tests/test_data_loader.py -v`
Expected: PASS immediately — these tests document existing behavior from Task 3's implementation. If either fails, fix `load_dataset` in `ml_data_view/data_loader.py` to match the docstring contract (extension check first, then existence check) before proceeding.

- [ ] **Step 3: Commit**

```bash
git add tests/test_data_loader.py
git commit -m "test: lock in unsupported-extension and missing-file errors"
```

---

## Task 7: Preview rows capped at 50

**Files:**
- Test: `tests/test_data_loader.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_data_loader.py (append)
def test_preview_rows_capped_at_50_for_larger_files(tmp_path):
    big_df = pl.DataFrame({"id": list(range(60)), "value": [x * 2 for x in range(60)]})
    csv_path = tmp_path / "big.csv"
    big_df.write_csv(csv_path)

    dataset_info = load_dataset(csv_path)

    assert dataset_info.row_count == 60
    assert len(dataset_info.preview_rows) == 50
    assert dataset_info.preview_rows[0]["id"] == 0
    assert dataset_info.preview_rows[-1]["id"] == 49


def test_preview_rows_returns_all_when_fewer_than_50(tmp_path):
    small_df = pl.DataFrame({"id": [1, 2, 3]})
    csv_path = tmp_path / "small.csv"
    small_df.write_csv(csv_path)

    dataset_info = load_dataset(csv_path)

    assert dataset_info.row_count == 3
    assert len(dataset_info.preview_rows) == 3
```

- [ ] **Step 2: Run tests to verify current behavior**

Run: `.venv/Scripts/python.exe -m pytest tests/test_data_loader.py -v`
Expected: PASS immediately — `df.head(PREVIEW_ROW_LIMIT)` from Task 3 already caps correctly. If either fails, check `PREVIEW_ROW_LIMIT` and the `df.head(...)` call in `load_dataset`.

- [ ] **Step 3: Run the full data_loader test file one more time**

Run: `.venv/Scripts/python.exe -m pytest tests/test_data_loader.py -v`
Expected: All tests PASS (should be 9 tests total by now).

- [ ] **Step 4: Commit**

```bash
git add tests/test_data_loader.py
git commit -m "test: lock in 50-row preview cap behavior"
```

---

## Task 8: pywebview API layer

**Files:**
- Create: `ml_data_view/api.py`
- Test: `tests/test_api.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_api.py
import polars as pl

from ml_data_view.api import Api


def test_load_file_returns_dataset_dict_for_valid_file(tmp_path):
    df = pl.DataFrame({"id": [1, 2], "label": ["a", None]})
    csv_path = tmp_path / "sample.csv"
    df.write_csv(csv_path)

    api = Api()
    result = api.load_file(str(csv_path))

    assert "error" not in result
    assert result["row_count"] == 2
    assert result["columns"][0]["name"] == "id"
    assert result["preview_rows"] == [{"id": 1, "label": "a"}, {"id": 2, "label": None}]


def test_load_file_returns_error_dict_for_unsupported_extension(tmp_path):
    bad_path = tmp_path / "sample.txt"
    bad_path.write_text("not a dataset")

    api = Api()
    result = api.load_file(str(bad_path))

    assert "error" in result


def test_load_file_returns_error_dict_for_missing_file(tmp_path):
    missing_path = tmp_path / "does_not_exist.csv"

    api = Api()
    result = api.load_file(str(missing_path))

    assert "error" in result
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/Scripts/python.exe -m pytest tests/test_api.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'ml_data_view.api'`

- [ ] **Step 3: Write minimal implementation**

```python
# ml_data_view/api.py
"""pywebview-facing API layer.

Thin wrapper around data_loader.py — exposes methods pywebview binds to
`window.pywebview.api` in the frontend. Keep this file free of business
logic so data_loader.py stays importable/testable without a window.

Known Phase 0 limitation: date/datetime columns are not yet stringified,
so a source file containing them may fail to serialize across the js_api
bridge. Not a concern for the CSV/Parquet/Arrow test fixtures used here;
revisit if/when a dataset with date columns is opened.
"""
from __future__ import annotations

from typing import Any

import webview

from ml_data_view.data_loader import UnsupportedFileTypeError, load_dataset


class Api:
    """Methods exposed to the JS frontend via pywebview's js_api binding."""

    def open_file_dialog(self) -> str | None:
        """Show a native file-open dialog restricted to supported types.

        Returns the selected path, or None if the user cancelled. Requires
        an active pywebview window, so this is not unit-tested headlessly —
        verify manually per Task 11's checklist.
        """
        window = webview.windows[0]
        result = window.create_file_dialog(
            webview.OPEN_DIALOG,
            file_types=(
                "Dataset files (*.parquet;*.arrow;*.csv)",
                "All files (*.*)",
            ),
        )
        if not result:
            return None
        return result[0]

    def load_file(self, path: str) -> dict[str, Any]:
        """Load a dataset file and return its profile as a JSON-safe dict.

        On failure, returns {"error": "<message>"} instead of raising, so
        the frontend always gets a plain dict back across the js_api bridge.
        """
        try:
            dataset_info = load_dataset(path)
        except (UnsupportedFileTypeError, FileNotFoundError) as exc:
            return {"error": str(exc)}
        return dataset_info.to_dict()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/Scripts/python.exe -m pytest tests/test_api.py -v`
Expected: PASS

- [ ] **Step 5: Run the whole test suite**

Run: `.venv/Scripts/python.exe -m pytest -v`
Expected: All tests PASS (data_loader + api).

- [ ] **Step 6: Commit**

```bash
git add ml_data_view/api.py tests/test_api.py
git commit -m "feat: add pywebview Api layer with load_file and open_file_dialog"
```

---

## Task 9: Frontend — HTML/CSS/JS

**Files:**
- Create: `ml_data_view/frontend/index.html`
- Create: `ml_data_view/frontend/style.css`
- Create: `ml_data_view/frontend/app.js`
- Delete: `ml_data_view/frontend/.gitkeep` (no longer needed once real files exist)

No automated test for this step — it's static UI wired to the `Api` methods tested in Task 8. Manual verification happens in Task 11 once `main.py` can launch it.

- [ ] **Step 1: Create `ml_data_view/frontend/index.html`**

```html
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <title>ML Dataset Profiler</title>
  <link rel="stylesheet" href="style.css" />
</head>
<body>
  <header>
    <h1>ML Dataset Profiler</h1>
    <button id="open-file-btn">Open File…</button>
    <span id="file-path"></span>
  </header>

  <section id="error-banner" class="hidden"></section>

  <section id="summary" class="hidden">
    <div><strong>Rows:</strong> <span id="row-count"></span></div>
    <div><strong>File size:</strong> <span id="file-size"></span></div>
  </section>

  <section id="schema-section" class="hidden">
    <h2>Schema</h2>
    <table id="schema-table">
      <thead>
        <tr><th>Column</th><th>Type</th><th>Null count</th></tr>
      </thead>
      <tbody></tbody>
    </table>
  </section>

  <section id="preview-section" class="hidden">
    <h2>Preview (first 50 rows)</h2>
    <div id="preview-wrapper">
      <table id="preview-table">
        <thead></thead>
        <tbody></tbody>
      </table>
    </div>
  </section>

  <script src="app.js"></script>
</body>
</html>
```

- [ ] **Step 2: Create `ml_data_view/frontend/style.css`**

```css
* { box-sizing: border-box; }

body {
  font-family: system-ui, -apple-system, "Segoe UI", sans-serif;
  margin: 0;
  padding: 16px 24px;
  color: #1a1a1a;
}

header {
  display: flex;
  align-items: center;
  gap: 12px;
  margin-bottom: 16px;
}

h1 { font-size: 1.25rem; margin: 0; }

#open-file-btn {
  padding: 8px 16px;
  font-size: 0.95rem;
  cursor: pointer;
}

#file-path { color: #555; font-size: 0.85rem; }

.hidden { display: none; }

#error-banner {
  background: #fdecea;
  color: #611a15;
  border: 1px solid #f5c6cb;
  padding: 10px 14px;
  border-radius: 4px;
  margin-bottom: 16px;
}

#summary {
  display: flex;
  gap: 24px;
  margin-bottom: 16px;
  font-size: 0.95rem;
}

section { margin-bottom: 24px; }

table {
  border-collapse: collapse;
  width: 100%;
}

th, td {
  border: 1px solid #ddd;
  padding: 6px 10px;
  text-align: left;
  font-size: 0.85rem;
  white-space: nowrap;
}

th { background: #f5f5f5; position: sticky; top: 0; }

#preview-wrapper {
  max-height: 500px;
  overflow: auto;
  border: 1px solid #ddd;
}
```

- [ ] **Step 3: Create `ml_data_view/frontend/app.js`**

```js
const openFileBtn = document.getElementById("open-file-btn");
const filePathSpan = document.getElementById("file-path");
const errorBanner = document.getElementById("error-banner");
const summarySection = document.getElementById("summary");
const rowCountSpan = document.getElementById("row-count");
const fileSizeSpan = document.getElementById("file-size");
const schemaSection = document.getElementById("schema-section");
const schemaTableBody = document.querySelector("#schema-table tbody");
const previewSection = document.getElementById("preview-section");
const previewTableHead = document.querySelector("#preview-table thead");
const previewTableBody = document.querySelector("#preview-table tbody");

function formatBytes(bytes) {
  if (bytes < 1024) return `${bytes} B`;
  const units = ["KB", "MB", "GB", "TB"];
  let value = bytes / 1024;
  let unitIndex = 0;
  while (value >= 1024 && unitIndex < units.length - 1) {
    value /= 1024;
    unitIndex += 1;
  }
  return `${value.toFixed(1)} ${units[unitIndex]}`;
}

function showError(message) {
  errorBanner.textContent = message;
  errorBanner.classList.remove("hidden");
  summarySection.classList.add("hidden");
  schemaSection.classList.add("hidden");
  previewSection.classList.add("hidden");
}

function clearError() {
  errorBanner.classList.add("hidden");
  errorBanner.textContent = "";
}

function renderSchema(columns) {
  schemaTableBody.innerHTML = "";
  for (const column of columns) {
    const row = document.createElement("tr");
    row.innerHTML = `<td>${column.name}</td><td>${column.dtype}</td><td>${column.null_count}</td>`;
    schemaTableBody.appendChild(row);
  }
  schemaSection.classList.remove("hidden");
}

function renderPreview(columns, rows) {
  const headRow = document.createElement("tr");
  headRow.innerHTML = columns.map((c) => `<th>${c.name}</th>`).join("");
  previewTableHead.innerHTML = "";
  previewTableHead.appendChild(headRow);

  previewTableBody.innerHTML = "";
  for (const row of rows) {
    const tr = document.createElement("tr");
    tr.innerHTML = columns
      .map((c) => {
        const value = row[c.name];
        return `<td>${value === null || value === undefined ? "" : value}</td>`;
      })
      .join("");
    previewTableBody.appendChild(tr);
  }
  previewSection.classList.remove("hidden");
}

function renderDataset(data) {
  filePathSpan.textContent = data.file_path;
  rowCountSpan.textContent = data.row_count.toLocaleString();
  fileSizeSpan.textContent = formatBytes(data.file_size_bytes);
  summarySection.classList.remove("hidden");
  renderSchema(data.columns);
  renderPreview(data.columns, data.preview_rows);
}

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

openFileBtn.addEventListener("click", handleOpenFile);
```

- [ ] **Step 4: Remove the placeholder and stage the frontend**

```bash
git rm ml_data_view/frontend/.gitkeep
git add ml_data_view/frontend/index.html ml_data_view/frontend/style.css ml_data_view/frontend/app.js
```

- [ ] **Step 5: Commit**

```bash
git commit -m "feat: add frontend UI for file open, schema, and preview table"
```

---

## Task 10: Entry point — `main.py`

**Files:**
- Create: `ml_data_view/main.py`

- [ ] **Step 1: Write `ml_data_view/main.py`**

```python
"""Entry point — launches the pywebview window."""
from __future__ import annotations

from pathlib import Path

import webview

from ml_data_view.api import Api

FRONTEND_DIR = Path(__file__).parent / "frontend"


def main() -> None:
    api = Api()
    webview.create_window(
        "ML Dataset Profiler",
        str(FRONTEND_DIR / "index.html"),
        js_api=api,
        width=1200,
        height=800,
    )
    webview.start()


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Verify the module imports cleanly (headless check)**

Run: `.venv/Scripts/python.exe -c "import ml_data_view.main"`
Expected: No output, exit code 0 (import must not error; `main()` is not called by this check since `webview.start()` would block/open a window).

- [ ] **Step 3: Run full automated test suite**

Run: `.venv/Scripts/python.exe -m pytest -v`
Expected: All tests still PASS.

- [ ] **Step 4: Commit**

```bash
git add ml_data_view/main.py
git commit -m "feat: wire up pywebview entry point"
```

---

## Task 11: Manual end-to-end verification

**Files:** none (verification only)

- [ ] **Step 1: Prepare a small test dataset**

```bash
.venv/Scripts/python.exe -c "import polars as pl; pl.DataFrame({'id': list(range(75)), 'label': (['cat','dog',None]*25), 'score': [i*0.1 for i in range(75)]}).write_parquet('sample_data/demo.parquet')"
```

(Create the `sample_data/` directory first if needed — this file is a manual-testing aid, not part of the app; add `sample_data/` to `.gitignore` rather than committing generated data.)

- [ ] **Step 2: Launch the app**

Run: `.venv/Scripts/python.exe -m ml_data_view.main`
Expected: A native window titled "ML Dataset Profiler" opens with an "Open File…" button.

- [ ] **Step 3: Open the demo file**

Click "Open File…", navigate to `sample_data/demo.parquet`, select it.
Expected:
- File path appears next to the button
- Rows shows `75`
- File size shows a human-readable size
- Schema table lists `id` (Int64/Int, 0 nulls), `label` (String, ~25 nulls), `score` (Float64, 0 nulls)
- Preview table shows exactly 50 scrollable rows starting at `id=0`

- [ ] **Step 4: Verify a CSV and an unsupported file**

Run: `.venv/Scripts/python.exe -c "import polars as pl; pl.DataFrame({'id':[1,2,3]}).write_csv('sample_data/demo.csv')"`
Open `sample_data/demo.csv` in the app — expect it to load correctly.
Then try opening any `.txt` or `.json` file (switch "Dataset files" to "All files" in the dialog) — expect the red error banner to show the unsupported-type message instead of a crash.

- [ ] **Step 5: Add `.gitignore` entry for manual test data**

Add `sample_data/` to `.gitignore`.

- [ ] **Step 6: Commit**

```bash
git add .gitignore
git commit -m "chore: ignore manual-testing sample_data directory"
```

---

## Task 12: README

**Files:**
- Create: `README.md`

- [ ] **Step 1: Write `README.md`**

```markdown
# ML Dataset Profiler (Phase 0)

A Windows-installable, no-code desktop app for profiling AI-training-dataset
files. Phase 0 scope: open a `.parquet`, `.arrow`, or `.csv` file and view
its schema (column names, dtypes, null counts), row count, file size, and
the first 50 rows.

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate      # Windows cmd/PowerShell
pip install -r requirements.txt
```

## Run

```bash
.venv\Scripts\python.exe -m ml_data_view.main
```

## Test

```bash
.venv\Scripts\python.exe -m pytest -v
```

## Project layout

- `ml_data_view/data_loader.py` — headless file-reading logic (polars/pyarrow).
  Importable and testable without launching the app.
- `ml_data_view/api.py` — pywebview-facing wrapper exposed to the frontend as
  `window.pywebview.api`.
- `ml_data_view/main.py` — app entry point; creates the pywebview window.
- `ml_data_view/frontend/` — plain HTML/CSS/JS UI (no framework yet).
- `tests/` — pytest suite for `data_loader.py` and `api.py`.

See `PROJECT_BRIEF.md` for full scope and roadmap.
```

- [ ] **Step 2: Commit**

```bash
git add README.md
git commit -m "docs: add README with setup/run/test instructions"
```

---

## Self-Review Notes

- **Spec coverage:** file picker (Task 8 `open_file_dialog` + Task 9 button) → read with polars/pyarrow (Tasks 3–5) → column names/dtypes/row count/file size/null counts (Task 2 model + Tasks 3–5 tests) → first 50 rows scrollable table (Task 7 cap + Task 9 `#preview-wrapper`). No charts, no LLM, no packaging, no auth anywhere in this plan — matches the brief's non-goals.
- **Headless testability:** `data_loader.py` and `api.py` (Tasks 2–8) have zero `webview` window dependency in their tested code paths, satisfying the brief's "testable headless" convention. Only `open_file_dialog` needs a live window, and that's explicitly called out as manual-only.
- **Small commits:** every task ends in its own commit, each leaving the suite green.
