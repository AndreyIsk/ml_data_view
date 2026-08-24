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
