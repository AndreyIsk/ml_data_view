# ML Dataset Profiler

A no-code desktop app for sanity-checking machine-learning training data.
Point it at a `.parquet`, `.arrow`/`.feather` or `.csv` file — or at a whole
folder tree of them — and get the schema, null patterns, class balance,
embedding projections, media previews and time-series health checks without
writing any pandas.

[![CI](https://github.com/AndreyIsk/ml-dataset-profiler/actions/workflows/ci.yml/badge.svg)](https://github.com/AndreyIsk/ml-dataset-profiler/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/python-3.11%2B-blue)](https://www.python.org/downloads/)
[![Platform](https://img.shields.io/badge/platform-Windows-lightgrey)](#requirements)
[![License](https://img.shields.io/badge/license-proprietary-red)](LICENSE)

Built for ML and data engineers who need to know whether a dataset is usable
*before* spending a training run finding out. Everything runs locally: no
upload step, no account, no network calls.

<!-- Screenshot goes here once there is one, e.g.:
     ![The Distributions tab](docs/images/distributions.png)
     A 1200x800 window capture of the Distributions or Health tab reads well. -->

## What it does

**Opens datasets larger than memory.** Loading is built on lazy, streaming
polars scans — bounded slices and streaming aggregations, never a full
`collect()`. A folder of roughly 200 GB profiles without ever being
materialised whole, and heavy work runs in worker subprocesses held to an
OS-enforced memory ceiling, so a runaway is killed by the OS instead of
taking the machine down.

**Profiles incrementally.** Per-file results are cached in a local SQLite
store keyed on path, size and mtime. Re-opening an unchanged folder reads no
column data at all; adding one file to 278 reads exactly that one file. A
folder can also be *watched*, so files arriving on disk are profiled as they
land.

| Area | What you get |
|---|---|
| **Schema & preview** | Column names, dtypes, null counts, row count, file size, first 50 rows |
| **Distributions** | Null heatmap, numeric histograms, class balance, text-length distributions |
| **Embeddings** | Embedding-column detection and a 2-D PCA projection coloured by label |
| **Media** | Inline thumbnail previews for image columns, plus audio and text |
| **Search** | Filter rows across the dataset; the charts and counts follow the filter |
| **Health** | Checks that declare a precondition and *skip* with a reason when it is absent — never a green tick for something unexamined |
| **Time series** | Coverage, sample interval, gaps, overlaps and stuck channels for any dataset with a time column |
| **Timeline** | A unit's channels over time, at whatever resolution the window needs, reading only the row groups it touches |
| **Catalogue & groups** | One row per unit of analysis, and each group compared against the typical group per channel — assembled from cache, not a new pass over the data |
| **Compare** | Two datasets or folders side by side, schema and distributions |
| **Reports** | PDF export of a dataset's profile, with an optional classification banner and audit log |

For how any of this works, see [`docs/architecture.md`](docs/architecture.md).

## Requirements

- **Python 3.11 or newer**
- **Windows** for the full feature set. The OS-enforced memory ceiling uses
  a Windows Job Object via `pywin32`; elsewhere it degrades to an
  unenforced ceiling and says so. `pip install` works on macOS and Linux,
  but the pywebview window needs a GTK or Qt backend you install yourself.
- No database, service or network access.

## Quick start

```bash
git clone https://github.com/AndreyIsk/ml-dataset-profiler.git
cd ml-dataset-profiler

python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # macOS / Linux

pip install -e ".[dev]"
```

Then run it:

```bash
ml-dataset-profiler
```

or, equivalently:

```bash
python -m ml_data_view.main
```

Use **Open File…** for a single dataset or **Open Folder…** to walk a tree,
group the files by schema and profile a whole group at once.

> `pip install -r requirements.txt` does the same thing — that file just
> points at `pyproject.toml` so the dependency list lives in one place.

## Opening it in an IDE

Both of the editors below read `pyproject.toml`, so the project resolves
with no extra configuration.

<details>
<summary><strong>PyCharm</strong></summary>

1. **File → Open…** and pick the cloned folder.
2. PyCharm detects `pyproject.toml` and offers to create a virtualenv —
   accept it, or point it at an existing `.venv` under
   **Settings → Project → Python Interpreter**.
3. Install into it from the terminal: `pip install -e ".[dev]"`.
4. To run the app: right-click `ml_data_view/main.py` → **Run 'main'**.
5. To run the tests: right-click the `tests/` folder → **Run 'pytest in
   tests'**. PyCharm picks up `[tool.pytest.ini_options]` automatically.

</details>

<details>
<summary><strong>VS Code</strong></summary>

1. **File → Open Folder…** and pick the cloned folder.
2. Accept the recommended extensions when prompted
   (`.vscode/extensions.json`).
3. Select the `.venv` interpreter — **Ctrl+Shift+P → Python: Select
   Interpreter**.
4. Press **F5** to launch the app; the committed `ML Dataset Profiler`
   configuration runs it as a module with subprocess debugging on, so
   breakpoints bind inside background jobs too.
5. The Testing panel is pre-wired to pytest.

</details>

## Tests

```bash
python -m pytest           # ~4 minutes: it builds real parquet fixtures
python -m pytest -q tests/test_api.py
```

The suite is 1,100+ tests and is the project's actual specification — the
architecture notes name the tests that guard each invariant (that discovery
never calls `collect()`, that bundled resource paths match their
destinations, and so on). CI runs it on Windows against Python 3.11 and
3.13.

## Building the Windows app

```bash
python scripts/build_release.py
```

Runs the whole pipeline: version metadata, PyInstaller (onedir), then Inno
Setup, producing `dist/MLDatasetProfiler/` and
`dist/MLDatasetProfiler-Setup-<version>.exe`. Requires Inno Setup 6
(`winget install JRSoftware.InnoSetup`) — the script fails with that exact
command if it is missing rather than leaving a half-built `dist/`.

Release builds install from `requirements-lock.txt`, which pins every
package with `==`; a test fails if any line loses its pin. The installer is
currently **unsigned**. See
[`docs/architecture.md`](docs/architecture.md) → "Executable build" and
"Installer".

## Project layout

```
ml_data_view/
  main.py              app entry point; creates the pywebview window
  api.py               the surface exposed to the frontend as window.pywebview.api
  data_loader.py       lazy/streaming scans -- returns LazyFrames, never DataFrames
  folder_loader.py     folder discovery and schema grouping, metadata-only
  hierarchy.py         directory-structure detection (group / series / chunk roles)
  profile_cache.py     per-file SQLite profile cache
  file_profile.py      computes one file's statistics, and merges a folder's
  folder_watcher.py    filesystem events -> "this file is ready to read"
  jobs.py              worker subprocesses under an OS-enforced memory ceiling
  timeseries.py        order-dependent statistics, computed per chunk
  timeline.py          one unit's channels over time
  catalog.py, fleet.py unit catalogue and group comparison, from cache alone
  findings.py          checks that declare a precondition and skip without one
  report.py            PDF export
  frontend/            plain HTML/CSS/JS + Plotly.js, no framework
tests/                 pytest suite
scripts/               release build, icon generation, diagnostics
installer/             Inno Setup script
docs/
  architecture.md      the engineering reference -- read before changing an area
  superpowers/         per-feature design specs and implementation plans
```

The data layer is deliberately importable and testable without launching a
window; keep it that way.

## License

Copyright (c) 2026 Andrey Isakov. All rights reserved.

The source is public to read and evaluate, but it is **not** open source —
no license to use, modify or redistribute is granted. See [LICENSE](LICENSE)
for the exact terms.
