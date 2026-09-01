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



\## Current phase: PHASE 5 (Packaging) — parts 1 and 2 done: the standalone .exe and the Windows installer. BUILDS AND INSTALLS; STILL NEEDS CLEAN-MACHINE VERIFICATION (see "Executable build" and "Installer" below). Licensing is a separate follow-up step, not started.

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


\## Charts follow the search filter (done)

Applying a search filter makes EVERY derived figure describe only matching rows — charts, row count, null counts and preview alike — so no two numbers on screen ever describe different row sets.

\- Explicit action ("Use this filter for charts"), not automatic: a search returns in milliseconds while a chart rebuild takes minutes on a large folder, so filtering must not start that work on every search.

\- Correctness trap handled: parquet footer null counts describe the WHOLE file and are invalid under a filter, so the filtered path decodes instead. Demonstrated on 10 real files — filtering integer\_feature\_1 >= 100 excludes nulls by definition, giving a filtered null count of 0 where the footer would have reported 1,561,097.

\- Every per-file scan in chart\_data.py routes through \_scan\_filtered(), so a filter cannot end up applied in some chart computations and forgotten in others.

\- A persistent, non-dismissible banner states the active filter until cleared — same reasoning as the partial-load banner. Opening a new dataset clears the filter on both sides.

\- Filtered builds are faster in practice (6.8s vs 11.6s on 10 real files) since a selective filter leaves far less data to aggregate, even though it forfeits the metadata shortcut.


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


\## Load performance stat line (done)

One quiet line under the top strip after any successful load: "Profiled 195,800,000 rows across 250 files in 1m 53s — peak memory 1.9 GiB (process)". Surfaces in the app the same figures the measurements above were quoting by hand from diagnostic runs.

\- Formatting lives in ml\_data\_view/load\_stats.py, not in JS: the backend hands the frontend a ready-made sentence. The omission rules are the point — any figure that is not known drops its whole clause rather than showing a zero or a guess, and when nothing is known there is no line at all. Duplicating that logic in the frontend is how the two would drift apart, so app.js only renders `load_stats.line`.

\- Peak memory is psutil's `peak_wset` — the process high-water mark, which Windows offers no way to reset. So it is labelled "(process)" rather than presented as this load's own peak: after a 250-file folder, a small file opened next would otherwise appear to have used the folder's memory. Off Windows there is no equivalent reading wired up, and current RSS is NOT substituted (it would silently understate the peak) — the clause is dropped instead.

\- Timing wraps the real load path at each of the three entry points: load\_file(), load\_folder\_group(), and FolderLoadJob.\_load() (so a folder load measures the worker thread's whole run, not the adoption that follows it). The audit-log write sits outside the clock.

\- A stopped folder load reports the files actually used, not the folder's full count, so the stat line agrees with the partial banner instead of contradicting it.

\- Styled as a measurement, not a banner: muted, small, no background or border. The banners below it carry warnings a user must act on, and giving a measurement equal visual weight would flatten that distinction.

\- No data-layer change: every figure was already computed by the time a load finishes (row count from a streaming aggregation or parquet metadata, file count from the schema group), so this adds no read to the lazy/streaming path.

\## Findings tab (done)

A tab that answers "is anything wrong with this dataset?" instead of leaving the user to read every chart and infer it (findings.py, deep\_checks.py, findings\_job.py, frontend/findings.js). Design spec: docs/superpowers/specs/2026-08-31-findings-tab-design.md.

\- Nine INSTANT checks are pure functions over two dicts the app has already computed — `Api.\_dataset\_info` and the chart payload. None takes a `pl.LazyFrame`, imports polars, or performs I/O. That is a design constraint, not an implementation detail: it makes the whole instant catalogue structurally incapable of violating the lazy/streaming rule, and lets every check be tested against a hand-built dict with no fixture file at all.

\- The cost of that is paid in TIMING: instant checks cannot run until the chart build finishes, which on a 250-file folder is minutes. Until then the tab says "Waiting for chart data…" rather than showing a clean state — "nothing found" before anything looked is a lie at exactly the moment a user is most likely to trust it.

\- Checks reuse decisions chart\_data.py already made rather than re-deriving them: a one-bin equal-edged histogram IS `numeric\_histogram\_data`'s "every value identical", and `x\_scale == "log"` IS `\_is\_skewed`'s verdict. Re-deriving either would create a second source of truth that could disagree with the chart on screen.

\- Check precedence stops one problem being reported three times. An all-null column suppresses `high\_missingness` and `constant\_column` for that column; a constant column suppresses `class\_imbalance` (a single-value column is trivially "100% one class"). Without this, one dead column produced three findings and counted three times in the severity summary.

\- Three DEEP checks read data and run only on an explicit button, as a cancellable job. Two are SAMPLED and say so in their own finding text: an exact duplicate count over the 195.8M-row folder means holding ~195M row hashes, the same unbounded-set pattern that exhausted a 2GB cap in categorical\_bar\_data. Duplicate detection also excludes Binary/List/Array columns from row identity — 100,000 rows of image blobs is not a bounded read — which changes what "duplicate" means, so the finding names the excluded columns rather than quietly redefining the term. `sentinel\_values` is EXACT, because counting equality against a fixed value set is a streaming aggregation that sums precisely per file.

\- `unresolvable\_images` exists because the instant image check cannot do its job alone: `image\_previews\_data` records one null thumbnail for three different causes (null value, missing file, undecodable bytes). The instant check reports them together in those words; the deep check re-reads the values and separates them.

\- The tab reads as a scan: `list\_checks()` is deliberately independent of a loaded dataset, so every check name is painted with a pending marker before any work happens. The deep checks' pending/running/done is REAL per-check status polled from DeepCheckJob. The instant checks' stagger is not — they take ~5ms total, so their reveal is an animation capped at 120ms per check, and the code says so rather than dressing 5ms up as a scan.

\- Cancelling deep checks keeps completed ones (half a findings list is half an answer, like a stopped folder load); a check that never ran is marked "not run", never "passed". A check that raises becomes an explicit error row — a check that could not run is not a check that passed.

\- Known limitation: `likely\_id\_column` reads `value\_frequency\_histograms`, which only exists for columns classified CATEGORICAL with over 200 distinct values. A mostly-unique short string (e.g. "t1729") classifies as `text` and gets a text-length histogram instead, so the check cannot see it. Widening it would need cardinality carried into the payload for text columns — a data-layer change, deliberately not made for a presentation feature.

\## Night mode (done)

A dark theme, chosen in Settings > Appearance as System / Light / Dark (`AppSettings.theme`, default "system").

\- The change that made it tractable was collapsing TWO color sources into one. Chart colors used to be literals in app.js while the UI palette lived in style.css; a second palette would have made that four lists to keep in sync. Now every color is a role in style.css and app.js reads the chart ones back out with `themeColor()` AT DRAW TIME — which is also what makes a live theme switch possible at all. Same no-two-sources-of-truth rule the Findings checks follow by reusing chart\_data's verdicts.

\- The dark column is SELECTED, not an inversion: each hue re-stepped for a dark ground from the dataviz skill's reference palette. The three scatter slots were run through that skill's validator with `--pairs all` against each mode's own surface — light worst CVD dE 9.2 / normal 24.0, dark 9.4 / 20.9, both clear of the >=8 and >=15 floors. tests/test\_theme.py pins those six hexes so a hue cannot be swapped without sending whoever did it back to the validator.

\- "system" is resolved in JS via matchMedia rather than by a CSS media query, so the dark values are written in exactly one place and an explicit Light/Dark choice wins in both directions without a specificity fight. The OS listener only acts while the preference is "system" — an explicit choice is not overridden when the machine flips at sunset.

\- A theme change re-renders charts from a cached `latestChartData`, never a rebuild: rebuilding costs ~0.54s per file (minutes on a big folder) while re-rendering already-fetched data is instant. Per-card "Color by" selections are captured and restored across that re-render.

\- tests/test\_theme.py parses both palettes straight out of style.css and gates every text pair at WCAG AA 4.5:1 and every chart mark at 3:1. Three marks are exempt WITH their reasons recorded in NON\_TEXT\_EXEMPT (light aqua at 2.82:1 is a documented property of the validated palette, relieved by the legend; the "Missing" bucket is recessive by design, like a gridline) — an exemption is a visible decision, not a gap in coverage.

\- That gate caught a PRE-EXISTING bug: white on `--accent` measures 4.37:1, under AA for .btn-primary's 0.9rem/500 text — every primary button in the app. `--accent` could not simply darken, since it is also `--chart-series` and validated categorical slot 1, so the roles split: `--accent-strong` (#1c5cab light) backs filled buttons at 6.8:1.

\- Deliberately NOT themed: the PDF report (printed and shared outside the app, so report\_charts.py keeps its own light palette) and the classification banner (its color is a deliberate choice for a security label, and app.js already picks contrasting text for it).

\## App icon (done)

An otter logo wired in as the app icon. Source art at assets/otter.png; scripts/make\_icons.py regenerates every derived file from it.

\- assets/otter.ico bundles 16/32/48/64/128/256px frames in one file. Windows pulls a DIFFERENT frame from the same .ico for the title bar (16), taskbar (32), Alt-Tab (48) and Explorer's large tiles (256), so a single-size .ico looks resampled in most of those places. Each frame is resampled from the full 2048px source rather than from the previous step, so the 16px frame is a clean reduction of the original rather than a reduction of a reduction.

\- ml\_data\_view/main.py passes it via `webview.start(icon=...)`. pywebview's own documentation says that parameter is "supported only on GTK and QT" — that is OUT OF DATE. The installed 6.2.1's winforms backend (platforms/winforms.py) reads the same `\_state['icon']` and assigns `self.Icon = Icon(path)`, falling back to extracting python.exe's icon when unset, which is why the window used to show a Python logo. Confirmed against a running window (WM\_GETICON returned the otter, and it renders in the title bar), not just by reading the source. .NET's System.Drawing.Icon only accepts .ico, so a PNG here would fail.

\- assets/icons/otter-{16,32,48,128,256,512}.png is the same art as standalone PNGs for docs, a web page, or a non-Windows packaging target.

\- NOTE ON THE SOURCE ART: assets/otter.png is actually a JPEG (RGB, no alpha) on a solid teal background, despite the .png extension. So every icon carries that teal square rather than a transparent cutout. Keying the teal out is not viable on this file — the fur edges and whiskers are antialiased against it and would fringe or vanish. A genuinely transparent source can be dropped in at the same path and `python scripts/make\_icons.py` re-run; nothing else needs to change.

\- The .exe/installer icon is NOT wired up: that is packaging, which the roadmap puts in Phase 5. The .ico is the artifact PyInstaller will want when that phase starts.

\## Executable build (Phase 5, part 1)

Status: **builds successfully; needs clean-machine verification.** Everything below was verified on the DEVELOPMENT machine, which has Python, the venv, and Google Chrome installed — none of which proves behaviour on a bare Windows box.

\### Build command

```
.venv/Scripts/python.exe -m pip install -r requirements-lock.txt
.venv/Scripts/python.exe -m PyInstaller --noconfirm --clean MLDatasetProfiler.spec
```

Output: `dist/MLDatasetProfiler.exe` (~193MB, onefile, windowed) and `dist/MLDatasetProfiler-selftest.exe` (same bundle, console). `build/` and `dist/` are gitignored; the .spec is committed so the build is reproducible.

\- requirements-lock.txt pins every package with `==` (requirements.txt uses `>=` and would resolve newer ones). tests/test\_resources.py fails if any line loses its pin.

\### How the frozen build is verified at all

scripts/frozen\_selftest.py is built as its own CONSOLE executable from the same Analysis. A windowed app cannot report on itself, and the failures that matter here are invisible from source — a lazily-imported package left out of the bundle, a data file at the wrong bundled path, kaleido unable to find a browser. Run it and read the output; exit code is 0 only if all 12 checks pass.

Result on the dev machine, frozen: **12/12 pass** (identical to the same script run from source). Covers: \_MEIPASS resolution; all 7 bundled frontend/asset files present; writable dir outside the bundle; polars+pyarrow lazy scan staying bounded (largest collect 50 rows of 50,000); sklearn's lazy import projecting an embedding; kaleido browser discovery; kaleido PNG rasterization; a full reportlab PDF; the Api layer (open, chart build, findings); both worker-thread job classes; settings round-trip; checkpoint logging.

\### Frozen-only fixes this required

\- **ml\_data\_view/resources.py (new).** Once frozen, `Path(\_\_file\_\_).parent` names nothing real — code runs from a PYZ archive and data files unpack to `sys.\_MEIPASS`. `resource_path()` answers "where is this bundled file", `writable_dir()` answers "where may I write". They are different questions with different answers, and nothing writes to the bundle.
\- **main.py** resolved FRONTEND\_DIR and ICON\_PATH through `resource_path()`; both were `Path(\_\_file\_\_)` anchors that would have found nothing in the .exe.
\- **checkpoint\_log.py** wrote its log to a path derived from `\_\_file\_\_`, i.e. inside `\_MEIPASS` — a temp directory deleted on exit, and potentially read-only. Since `checkpoint()` is called on every folder load and was unguarded, a failed write there would have crashed folder loading in the packaged app only. Now writes to the per-user app-data directory and swallows OSError.
\- **kaleido.mocker** calls argparse's `parse\_args()` AT IMPORT TIME, so PyInstaller's submodule scan exited with SystemExit and killed the build. Filtered out in the .spec.

\### THE KALEIDO CAVEAT — read before shipping

The Phase 5 note in "Tech stack" above was correct: **kaleido does not bundle a browser.** `kaleido/executable/bin/` contains only log files. Chart rasterization for PDF export drives a real Chromium through choreographer, which finds one by (a) a copy kaleido previously downloaded into `%LOCALAPPDATA%\plotly\choreographer\deps`, or (b) scanning for an installed Chrome/Chromium/Edge/Brave/Vivaldi.

So **PDF export is NOT self-contained.** On the dev machine it resolves `C:\Program Files (x86)\Google\Chrome\Application\chrome.exe`. On a clean machine it will most likely find Edge, which ships with Windows 10/11 and is in choreographer's list — but that is reasoning, not a measurement, and it is exactly what the clean-machine test must settle. Everything else in the app works without a browser; only Reports depends on this.

\### What was NOT verified, and what to test on a clean machine

Verified only on a machine that has Python, the venv and Chrome. A clean Windows VM with NO Python and NO Chrome must confirm:

1. **It launches at all.** Copy `dist/MLDatasetProfiler.exe` across and double-click. This is the one thing a dev-machine test cannot establish — a bundle can silently pick up a DLL from the dev machine's PATH.
2. **Run `MLDatasetProfiler-selftest.exe` first.** It prints 12 PASS/FAIL lines and is the fastest way to localise any failure.
3. **PDF export specifically**, on a VM with Edge but no Chrome, and again on one with neither. Expect "kaleido: browser discovery" to FAIL in the second case; decide then whether to ship a browser or require one.
4. **Open a real dataset** — a large folder, not the tiny synthetic frames the self-test uses. Peak memory and the bounded-collect behaviour were measured from source, never from the .exe against the 195.8M-row Criteo folder.
5. **The window icon and title bar** (verified here: WM\_GETICON returns the otter on the frozen window).
6. **Windows SmartScreen / Defender.** The .exe is unsigned; SmartScreen will warn on first run on a machine that has not seen it. Code signing is not part of this step.

\### Known properties

\- **Onefile costs ~5-6s of startup**, every launch: the 193MB archive is extracted to a temp directory each time (measured: 11.4/11.5/11.7s frozen vs 5.3/6.3s from source, for the same work). `onedir` removes that and is probably what the installer step wants — it is a one-line change in the .spec.
\- **scripts/job\_cap.py works against the frozen .exe** (it CreateProcess's an arbitrary command line, so freezing is irrelevant to it). Verified both directions: 3072MB cap → completes, peak 1529.7 MiB; 400MB cap → allocations refused by the OS, exit code 1 at peak 373.5 MiB. job\_cap itself is a dev/diagnostics script and is NOT bundled into the .exe.
\- The .exe is unsigned and ~193MB.

\## Installer (Phase 5, part 2)

Design spec: docs/superpowers/specs/2026-08-31-windows-installer-design.md.

\### Build command

```
python scripts/build_release.py
```

Runs the whole pipeline: generates build/version\_info.txt from `ml\_data\_view.\_\_version\_\_`, PyInstaller (onedir), then Inno Setup. Produces `dist/MLDatasetProfiler/` (571 MB) and `dist/MLDatasetProfiler-Setup-0.1.0.exe` (164 MB). Requires Inno Setup 6 (`winget install JRSoftware.InnoSetup`); the script fails with that exact command if it is missing rather than leaving a dist/ that looks finished.

\### onefile -> onedir was the biggest single change

onefile unpacked ~193MB to a temp directory on EVERY launch. Measured, same 12 self-test checks: **11.4/11.5/11.7s onefile vs 5.6/5.9/5.6s onedir**, against a 5.3-6.3s from-source baseline — so extraction overhead went from ~6s to essentially zero. The installed app puts a window on screen **1.0s** after clicking its Start Menu shortcut. No amount of installer polish survives a 6-second wait, so this came first.

`sys.\_MEIPASS` is still set under onedir (pointing at `\_internal/`), so resources.resource\_path() needed no change, and the frozen self-test still passes 12/12 from the onedir build.

\### Version metadata (new)

The project had no version string, so Properties > Details on the .exe was blank. `ml\_data\_view.\_\_version\_\_` is now the single source; the Win32 VERSIONINFO resource and the installer's AppVersion both derive from it, and tests/test\_version.py asserts they agree. Version drift between an executable and the installer shipping it is the classic packaging bug.

\### What the install actually does

\- Per-user to `%LOCALAPPDATA%\Programs\MLDatasetProfiler` — `PrivilegesRequired=lowest`, **no UAC prompt at any point**. Consequence, accepted: installed for the current user only.
\- Full classic seven-screen wizard: Welcome, Destination (Browse + disk space), Start Menu folder, Additional Tasks, Ready to Install summary, progress, Finish. An earlier draft specced a three-screen wizard with no directory page, reasoning from Claude Desktop; that was wrong for software people buy and install deliberately.
\- Destination page checks the chosen folder is writable and says so on that page, since Setup runs unelevated and browsing to Program Files would otherwise fail partway through the copy.
\- Start Menu entry + uninstaller shortcut; desktop icon opt-in.
\- Add/Remove Programs entry, verified present: DisplayName "ML Dataset Profiler", DisplayVersion 0.1.0, Publisher, InstallLocation, EstimatedSize, working UninstallString.
\- Stable `AppId` GUID so a newer Setup.exe upgrades in place. `CloseApplications=yes` (Restart Manager) so an upgrade over a running instance offers to close it — deliberately not AppMutex, which would mean adding a mutex to main.py for the installer's benefit.
\- **Uninstall leaves `%APPDATA%\MLDatasetProfiler` alone.** Settings, saved searches and the audit log are the user's data; an audit log exists precisely so it survives. There is no [UninstallDelete] section, and a test asserts there never is one.

\### UNSIGNED — the remaining gap

No code-signing certificate, so `SignTool=` sits commented in the .iss. Windows SmartScreen shows "Windows protected your PC — Unknown publisher" on first run and the user must click More info > Run anyway. **No installer setting closes this**; it needs a certificate. OV (~$200-400/yr, warning fades as install reputation accrues) or EV (~$400-600/yr, trusted immediately, hardware token or cloud HSM). Azure Trusted Signing is ~$10/mo but restricted to US/CA organisations 3+ years old.

\### Verified here vs still needing a clean machine

Verified on the dev machine: the wizard runs through all seven screens; files land in `%LOCALAPPDATA%\Programs\MLDatasetProfiler` (5,657 files, 577 MB); Start Menu and Add/Remove Programs entries are correct; the app launches from its Start Menu shortcut in 1.0s with the otter icon; the frozen self-test passes 12/12 from the installed onedir tree.

Still unverified, and unchanged from part 1: a machine with no Python and no Chromium browser. The clean-machine checklist under "Executable build" still applies in full, plus: does Setup.exe run on a box that has never seen it (SmartScreen), does the uninstaller remove cleanly, and does a second Setup.exe upgrade in place rather than installing beside.

\### Deliberately not built

Auto-update (an update feed, signed release artifacts and a background updater — and on-prem customers often forbid self-updating software), MSI/Group Policy deployment, and code signing itself.

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

