\# Project: ML Dataset Profiler (working name)



\## One-liner

A Windows-installable, no-code desktop app that opens AI-training-dataset

files (parquet, arrow, csv) and generates ML-specific profiling visuals —

null patterns, class balance, embedding projections, image/audio previews —

plus an LLM-driven "ask for a custom chart" feature. One-time purchase,

on-prem by default.



\## Target user

ML/data engineers and data annotation teams who need to sanity-check a

training dataset without writing pandas code every time.



\## Tech stack (decided — do not change without discussion)

\- Language: Python 3.11+

\- Data layer: polars (primary), pyarrow for arrow/parquet specifics,

&#x20; duckdb available for ad-hoc SQL later. Data loading uses lazy/

&#x20; streaming polars scans, not eager reads — see "Architecture notes"

&#x20; below before touching ml\_data\_view/data\_loader.py or any function

&#x20; that takes a `pl.LazyFrame`.

\- UI shell: pywebview (native window wrapping a local HTML/JS frontend)

\- Frontend: plain HTML/CSS/JS + Plotly.js for charts (no framework yet —

&#x20; keep it simple until the app has real features)

\- Packaging: PyInstaller -> Windows .exe (do NOT set this up yet — Phase 5)

\- Reporting (Phase 7): reportlab for PDF layout (pure Python, no native

&#x20; binary deps) + Python-side plotly (plotly>=6.1.1) with kaleido>=1.3 to

&#x20; rasterize chart images for embedding. Note for Phase 5: kaleido's v1

&#x20; engine drives a real Chrome/Chromium via the bundled `choreographer`

&#x20; package — confirm it still works from inside a PyInstaller bundle (a

&#x20; Chrome-finding/downloading step may need to ship or run at install

&#x20; time) before assuming PDF export works offline out of the box.



\## Architecture notes

\- \*\*Data loading uses lazy/streaming polars scans, not eager reads.\*\*

&#x20; ml\_data\_view/data\_loader.py's scan\_dataframe() returns a

&#x20; `pl.LazyFrame` (via scan\_csv/scan\_parquet/scan\_ipc) — nothing calls

&#x20; pl.read\_csv/read\_parquet/read\_ipc, and nothing downstream collect()s

&#x20; the whole frame. Every function in data\_loader.py/chart\_data.py/

&#x20; embedding\_data.py/image\_data.py that reads real row data does so via

&#x20; either a bounded slice (`.head(n)`/`.slice(offset, n)`, e.g. the 50-row

&#x20; table preview, the 500-row null-heatmap sample, the 10,000-row PCA

&#x20; embedding sample) or a streaming aggregation (`.collect(engine=

&#x20; "streaming")`, e.g. row count, null counts, histograms, class-balance

&#x20; value\_counts). This is what lets a dataset far larger than RAM — up to

&#x20; roughly 200GB — get profiled without ever materializing it whole.

&#x20; `Api` (api.py) caches the `LazyFrame` plus a once-computed row count,

&#x20; not a `DataFrame`. tests/test\_lazy\_loading.py asserts this end-to-end

&#x20; against a large synthetic fixture — keep it passing. Any new backend

&#x20; function that touches the loaded dataset should take a `pl.LazyFrame`

&#x20; and follow the same bounded-slice-or-streaming-aggregation rule; don't

&#x20; reintroduce a full `.collect()`.

\- \*\*Folder loading (Stage 2) discovers and groups by schema without ever

&#x20; collect()ing.\*\* ml\_data\_view/folder\_loader.py's scan\_folder() walks a

&#x20; folder tree (any depth) and reads each file's schema via

&#x20; collect\_schema() only; per-group row counts come from format-native

&#x20; metadata read directly via pyarrow (ParquetFile.metadata.num\_rows;

&#x20; IPC record-batch headers) — never a polars `.collect()`, and never a

&#x20; row count at all for .csv (no metadata-only path exists for it, so

&#x20; it's reported as unknown rather than forcing a full scan just to

&#x20; discover files). tests/test\_folder\_loader.py's

&#x20; test\_scan\_folder\_never\_calls\_collect\_during\_discovery asserts zero

&#x20; `LazyFrame.collect()` calls happen during discovery — keep it passing.

&#x20; Once the user picks a schema group, folder\_loader.load\_group()

&#x20; combines its files into one `pl.LazyFrame` via `pl.concat(...,

&#x20; how="vertical")` (lazy — this doesn't read data either) and hands it,

&#x20; unmodified, to the exact same chart\_data.py/embedding\_data.py/

&#x20; image\_data.py functions a single-file load uses — they already just

&#x20; take a `pl.LazyFrame` and don't know or care whether it came from one

&#x20; file or many.



\## Current phase: PHASE 7 (Reports) — PDF export shipped, plus large-folder scaling and report refinements

Built:

\- Section availability (ml\_data\_view/report.py's available\_sections()): for the loaded dataset, Overview/Distributions/Embeddings/Media are each only offered as a Reports checkbox when that section would actually show something live too (e.g. no embedding column detected -> no Embeddings checkbox). Defaults to all-available-checked.

\- PDF generation (ml\_data\_view/report.py + report\_charts.py): reuses chart\_data.py/embedding\_data.py's existing computations and image\_data.py's decoded thumbnails — no chart logic reimplemented. Charts are rasterized via Python-side plotly + kaleido; layout via reportlab. Every page gets a filename + generation-timestamp header/footer, plus the Settings classification banner (top and bottom) reproduced when enabled. Overview embeds the schema table; Media embeds a static, non-interactive thumbnail grid (first ~12 items) per image column.

\- UI: Reports section shows checkboxes for the available sections, a "Generate PDF Report" button, a native save-file dialog, and the saved path once done.

\- Audit log: report generation is logged as a "report\_generated" action the same way "file\_open" is, when Settings' audit log is enabled.

Since Phase 7, on the lazy-loading architecture track (see "Architecture notes" above):

\- Stage 1: the data-loading layer (Phases 0-3's table preview, row count, null/histogram/class-balance/text-length stats, embedding sampling, media previews) was refactored from eager reads to lazy/streaming polars scans. Purely an internal engine swap: single-file behavior and output unchanged from the user's perspective.

\- Stage 2 (folder loading, done): "Open Folder…" now sits alongside "Open File…" (both remain available). Recursively discovers .parquet/.csv/.arrow/.feather files under a chosen folder, groups them by exact schema match (schema-only reads, format-native metadata for row counts — never a collect()), and — when more than one schema group is found — lets the user pick which to load (defaulting to the largest by file count) via a chooser that also lists every excluded file with a short reason. The chosen group's files are combined into one LazyFrame (`pl.concat(..., how="vertical")`, lazy) and handed to the existing chart/table/embedding/media code completely unchanged. Audit-logs a "folder\_open" action the same way file opens do.

\- Stage 3 (bounded folder loading + progress/cancel, done): loading a real 250-file, 195.8M-row, 12.2GB folder previously exhausted memory (42GB, machine freeze). Root cause: each file is a single row group of ~1.5M rows, and polars streams at row-group granularity — it cannot subdivide one — so any query over the multi-file concat grew ~28-42MB per file (measured: 10 files 976MB, 50 files 1956MB, 100 files over a 2GB cap on ONE column). Fixes, all following the same rule — never query the concat, process per file: null counts now decode via bounded pyarrow iter\_batches one column at a time (data\_loader.\_bounded\_parquet\_missing\_counts); categorical-vs-text classification combines per-file distinct/row RATIOS and uses approx\_n\_unique rather than an unbounded exact distinct set (chart\_data.\_classify\_string\_column\_per\_file). That folder now loads in ~113s at a 1972 MiB peak. Loading runs on a worker thread (folder\_load\_job.py) with a cost-weighted progress bar (78% null counts / 22% classification, from measured timings) and a Stop button that truncates to whole completed files — a stopped load is a SMALLER dataset, never a partly-counted one, and its counts are identical to opening those files directly.

\- Diagnostics: scripts/job\_cap.py runs any subprocess under a hard, OS-enforced Windows Job Object memory ceiling (default 2GB), so a runaway load is killed by the OS instead of taking the machine down. Used for every measurement above.

\## Reports refinements (done)

\- Log-scale histograms: numeric columns whose full range dwarfs their interquartile spread (chart\_data.\_is\_skewed) are binned in log1p space, with an offset applied first so zeros and negatives are safe. Bin edges are reported in original units; both the live chart and the PDF label the axis identically. Non-skewed columns are unchanged.

\- Correlated missingness (null\_patterns.py): columns whose nulls fall on the SAME rows are detected in two passes — group by near-equal null count (free, reusing the schema table's counts), then verify row-level co-occurrence against one anchor column per group via a streaming aggregation. Anchor-only rather than all-pairs, so cost is linear in group size, not quadratic in schema width. Surfaced in the Distributions tab and the PDF.

\- Denser PDF: per-column charts are laid out two-up in a grid (~8 per page, was 1-2). The null heatmap keeps full width, since comparing many columns side by side is its whole purpose.

\## Bounded chart generation + chart progress bar (done)

Chart generation could not complete on the real folder at all — categorical\_bar\_data called value\_counts() over the concat, materializing every distinct value (~2.2M across 26 columns on TWO files) to show ten bars, and exceeded a 2GB cap. Fixed by applying the same never-query-the-concat rule everywhere it was still violated:

\- categorical\_bar\_data: per-file top-K shortlist merged (approximate; a value below K in every file could be missed).

\- numeric\_histogram\_data: global min/max per file, then each file binned against FIXED bin edges and summed — exact, pinned by a test against the single-frame result.

\- correlated\_missingness: co-null counts summed per file — exact, since a co-null row belongs to exactly one file.

\- High-cardinality columns (>200 distinct) now get a value-frequency histogram ("detections per value": how many values occur once, 2-3 times, ...) instead of a top-N bar naming arbitrary IDs. Grouped by a 64-bit hash of the value rather than the value itself — only counts matter, and grouping ~500k strings cost ~400MB per column and accumulated.

\- detect\_image\_columns: its "bounded sample" put .head() INSIDE the select expression, so polars evaluated drop\_nulls() over the whole concat before truncating — reading every row of 26 columns. Moved to a frame-level head (slice pushdown) and narrowed to the first file. Worth 1.9GB: that stage went from 975MB@2 files / 2872MB@50 to a flat ~920MB.

\- Lazy sklearn import (embedding\_data.py): importing sklearn cost ~554MB of the ~1.1GB import baseline (polars alone is 32MB, +pyarrow 527MB, +sklearn 1082MB) and is only needed to project an embedding column, which most datasets do not have.

Result: build\_chart\_payload is now FLAT in file count — 2132MB@2 files, 1955@10, 1865@25, 1934@50 (was 2218/2838/3400/4990). Time is linear at ~0.54s/file.

\- Chart progress bar (chart\_build\_job.py): chart generation runs on a worker thread with a progress bar over the per-column loop plus the four named whole-dataset steps, polled by the UI. Deliberately NOT cancellable, unlike the folder load: a truncated file list is still an honest dataset, but a half-built chart payload is not — some columns charted, others silently absent, with nothing on screen to tell the difference.

All existing tests pass, plus new large-fixture/many-file tests proving no full/unbounded collect() ever happens in either stage.

Still outstanding for Phase 6 (not part of this stage, and not part of Phase 7 either): LLM config (provider/key setup for Phase 4) — see roadmap below.



\## Row search (done)

User-built row search over a loaded file or folder (search.py, search\_jobs.py, frontend/search.js). Design spec: docs/superpowers/specs/2026-08-29-row-search-design.md.

\- The user picks columns, operators (is / is not / > / >= / < / <= / between / is null / is not null / contains) and values; all conditions AND together. Uniqueness is whatever the user defines it to be — e.g. an ID column plus a RecKey column isolates one track — rather than a fixed property of the schema.

\- Per-file scanning with early exit: search\_rows stops opening files as soon as the row limit is reached. Measured on 50 real files / 39.8M rows: composite-key lookup 0.04s, numeric range 0.03s, two thresholds 0.03s, is-null 0.03s. Selecting a subset of columns is a real speed win, not presentation — 3 of 40 columns ran 3x faster (0.01s vs 0.03s) because parquet reads only those column chunks.

\- Exact match counts run as a separate cancellable background job. Rows appear immediately; the count follows. This split exists because a count cannot short-circuit, and Spark-written files carry no statistics for string columns, so a string filter's count is a full scan (a numeric one skips row groups via min/max statistics — 0.13s for an exact count over 39.8M rows).

\- Export writes all matches to CSV per file via streaming, so it is bounded however large the result. A cancelled export leaves a partial file and says so, rather than handing over a truncated CSV that looks complete.

\- Saved named searches persist in the settings file. A saved search naming a column absent from the current schema disables just that row with an explanation, since a folder can legitimately be reloaded with a different schema group.

\- Deliberately not built (see the spec): charts do not follow the filter, no result sorting (it would require collecting every match before showing the first row, destroying the early exit), and no OR/nested condition groups.


\## Dataset comparison (done)

Compare the loaded dataset against another file or folder and flag what changed (compare.py, compare\_jobs.py, frontend/compare.js). Design spec: docs/superpowers/specs/2026-08-30-dataset-comparison-design.md.

\- Schema differences (columns only on one side, type changes) come from footer metadata only, so they render immediately from start\_comparison()'s own return value, before any distribution work begins.

\- Distribution differences are scored with PSI (Population Stability Index), the standard ML drift metric, using its conventional thresholds (<0.1 stable, 0.1-0.25 moderate, >0.25 significant) rather than a percent-change rule invented here. Empty bins are floored at 1e-6 so a bin populated on one side and empty on the other — the interesting case — gives a large finite number rather than inf.

\- Shared bins are mandatory: two histograms only compare if both sides use identical edges, so a joint min/max is computed across BOTH datasets first. The consequence is that an already-loaded dataset's cached histograms cannot be reused (they were binned against their own range), which is why comparison computes its own lean profile instead of diffing two build\_chart\_payload results.

\- Sampling defaults to 20 files per side, chosen EVENLY SPACED across each folder rather than the first 20 — these folders are day-partitioned, so a head sample would compare early-in-the-day against early-in-the-day and report drift that is really time of day. "Compare all files" is one click away.

\- The results table fills in row by row: CompareJob carries completed column rows in its progress state, so the UI is never blank while work continues. Cancelling keeps the columns already compared, since half a comparison is half an answer (unlike a partial chart payload).

\- Measured on the two real Criteo day folders (20 files sampled per side, 16.8M vs 15.6M rows): 18.3s, 2140 MiB peak, finding 2 significant and 3 moderate drifted columns out of 40.

\- Row-level differences (which individual rows changed) are deliberately out of scope — that needs a full scan of both sides plus a keyed join, not summary statistics.

\- Refactor: BackgroundJob (the start/poll/cancel lifecycle shared by the polled jobs) moved from inside search\_jobs.py to its own module ml\_data\_view/background\_job.py when a fifth job class needed it. FolderLoadJob and ChartBuildJob predate it and still carry their own copies.


\## Explicit non-goals right now

\- Do not touch the LLM custom-chart feature yet (that's Phase 4, and includes the LLM-config Settings UI still outstanding from Phase 6)

\- Do not set up PyInstaller/installer yet (that's Phase 5)

\- Do not add authentication or license-key checks yet



\## Conventions

\- Type hints throughout

\- Keep data-loading logic separate from UI code, so it's testable headless

&#x20; (i.e. importable and callable without launching the pywebview window)

\- Small, incremental commits — commit as soon as something works, before

&#x20; starting the next small piece

\- Prefer polars over pandas for anything performance-sensitive



\## Roadmap (for context only — do not build ahead of current phase)

1\. Phase 0: open file, show schema + table (this brief's scope)

2\. Phase 1: null heatmap, numeric histograms, class balance, text-length

&#x20;  distributions

3\. Phase 2: embedding column detection, 2D projection (PCA then UMAP),

&#x20;  scatter plot colored by label — this is the core differentiator

4\. Phase 3: inline preview for image/audio/text columns

5\. Phase 4: natural-language custom chart via LLM (declarative chart-spec

&#x20;  first; sandboxed code-gen only as a fallback for non-standard requests)

6\. Phase 5: packaging, installer, licensing

7\. Phase 6: Settings — classification banner (done: top+bottom strips,

&#x20;  configurable label/color) + audit log (done: local JSON-lines log of

&#x20;  file-open actions) + LLM config (provider/key setup for Phase 4 — not

&#x20;  yet built)

8\. Phase 7: Reports — PDF export of a dataset's profiling results (done:

&#x20;  section-availability checkboxes, header/footer + reproduced

&#x20;  classification banner, embedded charts/schema table/media grid, audit

&#x20;  logging of report generation)

