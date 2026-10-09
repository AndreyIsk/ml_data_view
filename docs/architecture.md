# Architecture

The engineering reference for ML Dataset Profiler: the decisions behind the
data layer, the module boundaries, and the invariants the test suite exists
to protect. Read the section for an area before changing it.

See the repository README for installing and running the app, and
`docs/superpowers/` for the per-feature design specs.

## Overview

A Windows-installable, no-code desktop app that opens AI-training-dataset files (parquet, arrow, csv) and generates ML-specific profiling visuals — null patterns, class balance, embedding projections, image/audio previews — plus an LLM-driven "ask for a custom chart" feature. On-prem by default.

## Target user

ML/data engineers and data annotation teams who need to sanity-check a training dataset without writing pandas code every time.

## Tech stack

- Language: Python 3.11+
- Data layer: polars (primary), pyarrow for arrow/parquet specifics, duckdb available for ad-hoc SQL later. Data loading uses lazy/streaming polars scans, not eager reads — see "Architecture notes" below before touching ml_data_view/data_loader.py or any function that takes a `pl.LazyFrame`.
- UI shell: pywebview (native window wrapping a local HTML/JS frontend)
- Filesystem events: watchdog (ml_data_view/folder_watcher.py). Picks its observer backend at runtime, so the PyInstaller spec bundles it explicitly — see "Folder watching".
- Frontend: plain HTML/CSS/JS + Plotly.js for charts (no framework yet — keep it simple until the app has real features)
- Packaging: PyInstaller -> Windows .exe (see "Executable build" below)
- Reporting: reportlab for PDF layout (pure Python, no native binary deps) + Python-side plotly (plotly>=6.1.1) with kaleido>=1.3 to rasterize chart images for embedding. Note: kaleido's v1 engine drives a real Chrome/Chromium via the bundled `choreographer` package — confirm it still works from inside a PyInstaller bundle (a Chrome-finding/downloading step may need to ship or run at install time) before assuming PDF export works offline out of the box.

## Architecture notes

- **Data loading uses lazy/streaming polars scans, not eager reads.** ml_data_view/data_loader.py's scan_dataframe() returns a `pl.LazyFrame` (via scan_csv/scan_parquet/scan_ipc) — nothing calls pl.read_csv/read_parquet/read_ipc, and nothing downstream collect()s the whole frame. Every function in data_loader.py/chart_data.py/embedding_data.py/image_data.py that reads real row data does so via either a bounded slice (`.head(n)`/`.slice(offset, n)`, e.g. the 50-row table preview, the 500-row null-heatmap sample, the 10,000-row PCA embedding sample) or a streaming aggregation (`.collect(engine="streaming")`, e.g. row count, null counts, histograms, class-balance value_counts). This is what lets a dataset far larger than RAM — up to roughly 200GB — get profiled without ever materializing it whole. `Api` (api.py) caches the `LazyFrame` plus a once-computed row count, not a `DataFrame`. tests/test_lazy_loading.py asserts this end-to-end against a large synthetic fixture — keep it passing. Any new backend function that touches the loaded dataset should take a `pl.LazyFrame` and follow the same bounded-slice-or-streaming-aggregation rule; don't reintroduce a full `.collect()`.
- **Folder loading (Stage 2) discovers and groups by schema without ever collect()ing.** ml_data_view/folder_loader.py's scan_folder() walks a folder tree (any depth) and reads each file's schema via collect_schema() only; per-group row counts come from format-native metadata read directly via pyarrow (ParquetFile.metadata.num_rows; IPC record-batch headers) — never a polars `.collect()`, and never a row count at all for .csv (no metadata-only path exists for it, so it's reported as unknown rather than forcing a full scan just to discover files). tests/test_folder_loader.py's test_scan_folder_never_calls_collect_during_discovery asserts zero `LazyFrame.collect()` calls happen during discovery — keep it passing. Once the user picks a schema group, folder_loader.load_group() combines its files into one `pl.LazyFrame` via `pl.concat(..., how="vertical")` (lazy — this doesn't read data either) and hands it, unmodified, to the exact same chart_data.py/embedding_data.py/image_data.py functions a single-file load uses — they already just take a `pl.LazyFrame` and don't know or care whether it came from one file or many.
- **Directory structure is DETECTED, CONFIRMED and OPTIONAL — never assumed.** ml_data_view/hierarchy.py describes the shape of whatever was opened; it does not require one. Each directory LEVEL gets one of three roles: GROUP (its names identify something the data is split by — a vehicle, a day, a site), SERIES (a directory here holds the ordered CHUNKS of one dataset), or IGNORE. Any number of group levels; at most one series level; **both optional**. A group is suggested from the hive `key=value` convention; a SERIES is suggested from EVIDENCE — the shallowest named level whose directories mostly hold same-schema files with non-overlapping time ranges, all of it read from parquet footers. The UNIT is what loads as one dataset: a series directory and its ordered chunks, or — with no series level — a single FILE, which is exactly how the app behaved before any of this existed. So a single file, a flat folder, hive-only partitions, an arbitrary nested tree and a fully mapped tree all come out of one model rather than needing separate paths, and the four shapes with no roles keep the schema-group behaviour untouched. **THE CHUNK IS THE CACHE UNIT** — a chunk is one file, which is what profile_cache.py keys on, so unit stats are a merge of its chunks', group stats a merge of its units', and the whole-dataset stats a merge of those: the same merge at every width, with nothing above the chunk ever cached separately because nothing above the chunk is a file. Everything in detection and ordering is METADATA-ONLY (one parquet-footer read per chunk gives schema, row count and time bounds together). **No vocabulary is hardcoded**: the code says group / series / chunk / unit, and the words a user sees are labels carried by their own mapping. See "Directory structure" below.
- **Time-ordered statistics are computed per CHUNK and merged per unit (ml_data_view/timeseries.py).** Everything about ORDER — is time going forwards, is it evenly spaced, did a channel stop moving — is the one class of question the rest of the app cannot answer, because every other statistic it computes is order-free. Memory is bounded by a BATCH, not by a chunk: each column is read through pyarrow's `iter_batches`, which splits within a row group, and run lengths, dt histograms and boundary state are carried incrementally across batches and then across chunks. Measured: six times the rows cost 1.7% more peak memory (722.8MB over 10 chunks, 735.2MB over 60). The per-chunk artifact is cached beside every other per-chunk statistic, and the recording-level answers that only exist AT a seam — a gap between two chunks, an overlap, a run of nulls spanning a boundary — are arithmetic on neighbouring chunks' cached first/last values, never a re-read. See "Time-series health" below.
- **A timeline fetch reads row groups, or nothing at all.** ml_data_view/timeline.py draws one unit's channels over time, and answers every request one of two ways: by folding the per-chunk buckets already in the profile cache, or — when the window is narrow enough that the parquet FOOTERS say the read is within `timeline_max_rows` — by decoding only the row groups whose own time range overlaps it. So the view opens instantly on a recording of any size and gets finer as you zoom, and raw rows never leave Python at any zoom: what crosses the bridge is a bounded number of buckets carrying min, max, mean and a count. Its precondition is a detected timestamp column and NOTHING else. See "Channel timeline" below.
- **The catalogue and the group comparison are a REARRANGEMENT of cached statistics, not a pass over data.** ml_data_view/catalog.py describes one row per unit of analysis; ml_data_view/fleet.py compares each group of files against the typical group, per channel, with compare.py's own PSI over the shared bin edges every file was already binned into. Neither reads a byte: a row count is the sum of its chunks' cached counts, a group's histogram is the elementwise sum of its files' cached bin counts, a merged mean is the cached sum over the cached count. Both are OPTIONAL in the same way everything else here is — the catalogue's unit is a series where one was mapped and otherwise a file, so a single CSV gets one row; the comparison needs a grouping (a directory role, a hive key, a column the user picks) and SKIPS with a stated reason when there is none. See "Unit catalogue and group comparison" below.
- **Every check declares a PRECONDITION and skips when it is absent.** findings.py's CheckSpec carries an `applies` predicate; a check whose precondition the loaded dataset does not satisfy reports status `skipped` with the reason, which is neither a pass nor a failure — a green tick there would claim the dataset had been examined for something it cannot have. The eleven time-series checks require a detected timestamp column and NOTHING else, so they run on a single CSV with a time column; the two chunk-boundary checks (missing chunk, overlap) additionally require a series level; group comparison requires a grouping column (a directory role, a hive key, or any column the user picks). The preconditions are computed once per load into `dataset_info["capabilities"]`, which keeps the instant catalogue's rule intact: every check stays a pure function over two dicts the app already has.
- **A watched folder re-profiles incrementally, and only when a file is genuinely finished.** ml_data_view/folder_watcher.py turns filesystem events into the much narrower claim "this file is ready to read", behind three gates: debounce (a burst for one path is one pending entry), size stability (default 10s, any change restarts the clock), and — for parquet — footer validation, since the footer is written last. A file that passes none of them stays visible as "still being written" rather than being dropped or silently retried. Everything the watcher triggers goes through the SAME capped background jobs and the SAME per-file cache as a manual open, so a file arriving while nobody is looking cannot do anything a user-initiated load could not, and costs only the new file. See "Folder watching" below.
- **Profiling results are cached per FILE and folder profiles are MERGED from them, not rescanned.** ml_data_view/profile_cache.py is a local SQLite store keyed on (absolute path, size, mtime_ns) plus a SCHEMA_VERSION constant; ml_data_view/file_profile.py computes one file's statistics and merges a folder's. Re-opening an unchanged folder reads no column data at all; adding one file to 278 reads exactly that file. This is the reason every per-file statistic in this codebase has to be MERGEABLE — a folder-level number is assembled from per-file ones, so anything that cannot be summed, min'd or max'd either carries a stated approximation or is recomputed on every load. The three categories, and which statistic is in which, are listed in file_profile.py's module docstring and in "Per-file profiling cache" below. Histograms are the case that shaped the design: per-file bin counts only add up if the bins are identical, so each file's counts are stored with the SPEC that produced them, and a new file that widens the global range invalidates that one COLUMN across the folder rather than the whole profile. Nothing here touches the network, and clearing the cache never loses data — only the time to recompute it.
- **Heavy work runs in capped background jobs via ml_data_view/jobs.py.** Building charts, running deep findings checks, loading a folder, comparing datasets and generating a PDF do NOT run in the window's process. Each runs in a worker subprocess held to an OS-enforced memory ceiling (a Windows Job Object), reports progress through a temp file the parent polls, returns its result through another (JSON, or parquet for tabular results), and can be cancelled. The UI process never allocates the data it is profiling, so a runaway is killed by the OS instead of taking the machine down — which is what happened before (42GB, machine freeze; see Stage 3 below). Instant/cheap work — schema reads, footer metadata, the 50-row preview, already-bounded aggregations — stays in-process: paying ~1s of process startup to read a parquet footer would be a worse experience, not a safer one. The worker runs the SAME in-process job classes as before (ChartBuildJob, DeepCheckJob, FolderLoadJob, CompareJob); jobs.py only wraps, polls and publishes them, so their existing tests still cover the real code path and the progress dicts the frontend polls kept their exact shape. See "Background job runner" below for what moved and what deliberately did not.

## Implementation history

Built:

- Section availability (ml_data_view/report.py's available_sections()): for the loaded dataset, Overview/Distributions/Embeddings/Media are each only offered as a Reports checkbox when that section would actually show something live too (e.g. no embedding column detected -> no Embeddings checkbox). Defaults to all-available-checked.
- PDF generation (ml_data_view/report.py + report_charts.py): reuses chart_data.py/embedding_data.py's existing computations and image_data.py's decoded thumbnails — no chart logic reimplemented. Charts are rasterized via Python-side plotly + kaleido; layout via reportlab. Every page gets a filename + generation-timestamp header/footer, plus the Settings classification banner (top and bottom) reproduced when enabled. Overview embeds the schema table; Media embeds a static, non-interactive thumbnail grid (first ~12 items) per image column.
- UI: Reports section shows checkboxes for the available sections, a "Generate PDF Report" button, a native save-file dialog, and the saved path once done.
- Audit log: report generation is logged as a "report_generated" action the same way "file_open" is, when Settings' audit log is enabled. Since Phase 7, on the lazy-loading architecture track (see "Architecture notes" above):
- Stage 1: the data-loading layer (Phases 0-3's table preview, row count, null/histogram/class-balance/text-length stats, embedding sampling, media previews) was refactored from eager reads to lazy/streaming polars scans. Purely an internal engine swap: single-file behavior and output unchanged from the user's perspective.
- Stage 2 (folder loading, done): "Open Folder…" now sits alongside "Open File…" (both remain available). Recursively discovers .parquet/.csv/.arrow/.feather files under a chosen folder, groups them by exact schema match (schema-only reads, format-native metadata for row counts — never a collect()), and — when more than one schema group is found — lets the user pick which to load (defaulting to the largest by file count) via a chooser that also lists every excluded file with a short reason. The chosen group's files are combined into one LazyFrame (`pl.concat(..., how="vertical")`, lazy) and handed to the existing chart/table/embedding/media code completely unchanged. Audit-logs a "folder_open" action the same way file opens do.
- Stage 3 (bounded folder loading + progress/cancel, done): loading a real 250-file, 195.8M-row, 12.2GB folder previously exhausted memory (42GB, machine freeze). Root cause: each file is a single row group of ~1.5M rows, and polars streams at row-group granularity — it cannot subdivide one — so any query over the multi-file concat grew ~28-42MB per file (measured: 10 files 976MB, 50 files 1956MB, 100 files over a 2GB cap on ONE column). Fixes, all following the same rule — never query the concat, process per file: null counts now decode via bounded pyarrow iter_batches one column at a time (data_loader._bounded_parquet_missing_counts); categorical-vs-text classification combines per-file distinct/row RATIOS and uses approx_n_unique rather than an unbounded exact distinct set (chart_data._classify_string_column_per_file). That folder now loads in ~113s at a 1972 MiB peak. Loading runs on a worker thread (folder_load_job.py) with a cost-weighted progress bar (78% null counts / 22% classification, from measured timings) and a Stop button that truncates to whole completed files — a stopped load is a SMALLER dataset, never a partly-counted one, and its counts are identical to opening those files directly.
- Diagnostics: scripts/job_cap.py ran any subprocess under a hard, OS-enforced Windows Job Object memory ceiling (default 2GB), so a runaway load was killed by the OS instead of taking the machine down. Used for every measurement above. That wrapper has since moved into the package as ml_data_view/jobs.py and become how the app itself runs every heavy operation; scripts/job_cap.py is now a thin re-export keeping the same CLI and `from scripts.job_cap import run_capped` surface these measurement notes quote.

## Reports refinements (done)

- Log-scale histograms: numeric columns whose full range dwarfs their interquartile spread (chart_data._is_skewed) are binned in log1p space, with an offset applied first so zeros and negatives are safe. Bin edges are reported in original units; both the live chart and the PDF label the axis identically. Non-skewed columns are unchanged.
- Correlated missingness (null_patterns.py): columns whose nulls fall on the SAME rows are detected in two passes — group by near-equal null count (free, reusing the schema table's counts), then verify row-level co-occurrence against one anchor column per group via a streaming aggregation. Anchor-only rather than all-pairs, so cost is linear in group size, not quadratic in schema width. Surfaced in the Distributions tab and the PDF.
- Denser PDF: per-column charts are laid out two-up in a grid (~8 per page, was 1-2). The null heatmap keeps full width, since comparing many columns side by side is its whole purpose.

## Bounded chart generation + chart progress bar (done)

Chart generation could not complete on the real folder at all — categorical_bar_data called value_counts() over the concat, materializing every distinct value (~2.2M across 26 columns on TWO files) to show ten bars, and exceeded a 2GB cap. Fixed by applying the same never-query-the-concat rule everywhere it was still violated:

- categorical_bar_data: per-file top-K shortlist merged (approximate; a value below K in every file could be missed).
- numeric_histogram_data: global min/max per file, then each file binned against FIXED bin edges and summed — exact, pinned by a test against the single-frame result.
- correlated_missingness: co-null counts summed per file — exact, since a co-null row belongs to exactly one file.
- High-cardinality columns (>200 distinct) now get a value-frequency histogram ("detections per value": how many values occur once, 2-3 times, ...) instead of a top-N bar naming arbitrary IDs. Grouped by a 64-bit hash of the value rather than the value itself — only counts matter, and grouping ~500k strings cost ~400MB per column and accumulated.
- detect_image_columns: its "bounded sample" put .head() INSIDE the select expression, so polars evaluated drop_nulls() over the whole concat before truncating — reading every row of 26 columns. Moved to a frame-level head (slice pushdown) and narrowed to the first file. Worth 1.9GB: that stage went from 975MB@2 files / 2872MB@50 to a flat ~920MB.
- Lazy sklearn import (embedding_data.py): importing sklearn cost ~554MB of the ~1.1GB import baseline (polars alone is 32MB, +pyarrow 527MB, +sklearn 1082MB) and is only needed to project an embedding column, which most datasets do not have. Result: build_chart_payload is now FLAT in file count — 2132MB@2 files, 1955@10, 1865@25, 1934@50 (was 2218/2838/3400/4990). Time is linear at ~0.54s/file.
- Chart progress bar (chart_build_job.py): chart generation runs on a worker thread with a progress bar over the per-column loop plus the four named whole-dataset steps, polled by the UI. Deliberately NOT cancellable, unlike the folder load: a truncated file list is still an honest dataset, but a half-built chart payload is not — some columns charted, others silently absent, with nothing on screen to tell the difference. All existing tests pass, plus new large-fixture/many-file tests proving no full/unbounded collect() ever happens in either stage. Still outstanding for Phase 6 (not part of this stage, and not part of Phase 7 either): LLM config (provider/key setup for Phase 4) — see roadmap below.

## Row search (done)

User-built row search over a loaded file or folder (search.py, search_jobs.py, frontend/search.js). Design spec: docs/superpowers/specs/2026-08-29-row-search-design.md.

- The user picks columns, operators (is / is not / > / >= / < / <= / between / is null / is not null / contains) and values; all conditions AND together. Uniqueness is whatever the user defines it to be — e.g. an ID column plus a RecKey column isolates one track — rather than a fixed property of the schema.
- Per-file scanning with early exit: search_rows stops opening files as soon as the row limit is reached. Measured on 50 real files / 39.8M rows: composite-key lookup 0.04s, numeric range 0.03s, two thresholds 0.03s, is-null 0.03s. Selecting a subset of columns is a real speed win, not presentation — 3 of 40 columns ran 3x faster (0.01s vs 0.03s) because parquet reads only those column chunks.
- Exact match counts run as a separate cancellable background job. Rows appear immediately; the count follows. This split exists because a count cannot short-circuit, and Spark-written files carry no statistics for string columns, so a string filter's count is a full scan (a numeric one skips row groups via min/max statistics — 0.13s for an exact count over 39.8M rows).
- Export writes all matches to CSV per file via streaming, so it is bounded however large the result. A cancelled export leaves a partial file and says so, rather than handing over a truncated CSV that looks complete.
- Saved named searches persist in the settings file. A saved search naming a column absent from the current schema disables just that row with an explanation, since a folder can legitimately be reloaded with a different schema group.
- Deliberately not built (see the spec): charts do not follow the filter, no result sorting (it would require collecting every match before showing the first row, destroying the early exit), and no OR/nested condition groups.

## Charts follow the search filter (done)

Applying a search filter makes EVERY derived figure describe only matching rows — charts, row count, null counts and preview alike — so no two numbers on screen ever describe different row sets.

- Explicit action ("Use this filter for charts"), not automatic: a search returns in milliseconds while a chart rebuild takes minutes on a large folder, so filtering must not start that work on every search.
- Correctness trap handled: parquet footer null counts describe the WHOLE file and are invalid under a filter, so the filtered path decodes instead. Demonstrated on 10 real files — filtering integer_feature_1 >= 100 excludes nulls by definition, giving a filtered null count of 0 where the footer would have reported 1,561,097.
- Every per-file scan in chart_data.py routes through _scan_filtered(), so a filter cannot end up applied in some chart computations and forgotten in others.
- A persistent, non-dismissible banner states the active filter until cleared — same reasoning as the partial-load banner. Opening a new dataset clears the filter on both sides.
- Filtered builds are faster in practice (6.8s vs 11.6s on 10 real files) since a selective filter leaves far less data to aggregate, even though it forfeits the metadata shortcut.

## Dataset comparison (done)

Compare the loaded dataset against another file or folder and flag what changed (compare.py, compare_jobs.py, frontend/compare.js). Design spec: docs/superpowers/specs/2026-08-30-dataset-comparison-design.md.

- Schema differences (columns only on one side, type changes) come from footer metadata only, so they render immediately from start_comparison()'s own return value, before any distribution work begins.
- Distribution differences are scored with PSI (Population Stability Index), the standard ML drift metric, using its conventional thresholds (<0.1 stable, 0.1-0.25 moderate, >0.25 significant) rather than a percent-change rule invented here. Empty bins are floored at 1e-6 so a bin populated on one side and empty on the other — the interesting case — gives a large finite number rather than inf.
- Shared bins are mandatory: two histograms only compare if both sides use identical edges, so a joint min/max is computed across BOTH datasets first. The consequence is that an already-loaded dataset's cached histograms cannot be reused (they were binned against their own range), which is why comparison computes its own lean profile instead of diffing two build_chart_payload results.
- Sampling defaults to 20 files per side, chosen EVENLY SPACED across each folder rather than the first 20 — these folders are day-partitioned, so a head sample would compare early-in-the-day against early-in-the-day and report drift that is really time of day. "Compare all files" is one click away.
- The results table fills in row by row: CompareJob carries completed column rows in its progress state, so the UI is never blank while work continues. Cancelling keeps the columns already compared, since half a comparison is half an answer (unlike a partial chart payload).
- Measured on the two real Criteo day folders (20 files sampled per side, 16.8M vs 15.6M rows): 18.3s, 2140 MiB peak, finding 2 significant and 3 moderate drifted columns out of 40.
- Row-level differences (which individual rows changed) are deliberately out of scope — that needs a full scan of both sides plus a keyed join, not summary statistics.
- Refactor: BackgroundJob (the start/poll/cancel lifecycle shared by the polled jobs) moved from inside search_jobs.py to its own module ml_data_view/background_job.py when a fifth job class needed it. FolderLoadJob and ChartBuildJob predate it and still carry their own copies.

## Load performance stat line (done)

One quiet line under the top strip after any successful load: "Profiled 195,800,000 rows across 250 files in 1m 53s — peak memory 1.9 GiB (process)". Surfaces in the app the same figures the measurements above were quoting by hand from diagnostic runs.

- Formatting lives in ml_data_view/load_stats.py, not in JS: the backend hands the frontend a ready-made sentence. The omission rules are the point — any figure that is not known drops its whole clause rather than showing a zero or a guess, and when nothing is known there is no line at all. Duplicating that logic in the frontend is how the two would drift apart, so app.js only renders `load_stats.line`.
- Peak memory is psutil's `peak_wset` — the process high-water mark, which Windows offers no way to reset. So it is labelled "(process)" rather than presented as this load's own peak: after a 250-file folder, a small file opened next would otherwise appear to have used the folder's memory. Off Windows there is no equivalent reading wired up, and current RSS is NOT substituted (it would silently understate the peak) — the clause is dropped instead.
- Timing wraps the real load path at each of the three entry points: load_file(), load_folder_group(), and FolderLoadJob._load() (so a folder load measures the worker thread's whole run, not the adoption that follows it). The audit-log write sits outside the clock.
- A stopped folder load reports the files actually used, not the folder's full count, so the stat line agrees with the partial banner instead of contradicting it.
- Styled as a measurement, not a banner: muted, small, no background or border. The banners below it carry warnings a user must act on, and giving a measurement equal visual weight would flatten that distinction.
- No data-layer change: every figure was already computed by the time a load finishes (row count from a streaming aggregation or parquet metadata, file count from the schema group), so this adds no read to the lazy/streaming path.

## Findings tab (done)

A tab that answers "is anything wrong with this dataset?" instead of leaving the user to read every chart and infer it (findings.py, deep_checks.py, findings_job.py, frontend/findings.js). Design spec: docs/superpowers/specs/2026-08-31-findings-tab-design.md.

- Nine INSTANT checks are pure functions over two dicts the app has already computed — `Api._dataset_info` and the chart payload. None takes a `pl.LazyFrame`, imports polars, or performs I/O. That is a design constraint, not an implementation detail: it makes the whole instant catalogue structurally incapable of violating the lazy/streaming rule, and lets every check be tested against a hand-built dict with no fixture file at all.
- The cost of that is paid in TIMING: instant checks cannot run until the chart build finishes, which on a 250-file folder is minutes. Until then the tab says "Waiting for chart data…" rather than showing a clean state — "nothing found" before anything looked is a lie at exactly the moment a user is most likely to trust it.
- Checks reuse decisions chart_data.py already made rather than re-deriving them: a one-bin equal-edged histogram IS `numeric_histogram_data`'s "every value identical", and `x_scale == "log"` IS `_is_skewed`'s verdict. Re-deriving either would create a second source of truth that could disagree with the chart on screen.
- Check precedence stops one problem being reported three times. An all-null column suppresses `high_missingness` and `constant_column` for that column; a constant column suppresses `class_imbalance` (a single-value column is trivially "100% one class"). Without this, one dead column produced three findings and counted three times in the severity summary.
- Three DEEP checks read data and run only on an explicit button, as a cancellable job. Two are SAMPLED and say so in their own finding text: an exact duplicate count over the 195.8M-row folder means holding ~195M row hashes, the same unbounded-set pattern that exhausted a 2GB cap in categorical_bar_data. Duplicate detection also excludes Binary/List/Array columns from row identity — 100,000 rows of image blobs is not a bounded read — which changes what "duplicate" means, so the finding names the excluded columns rather than quietly redefining the term. `sentinel_values` is EXACT, because counting equality against a fixed value set is a streaming aggregation that sums precisely per file.
- `unresolvable_images` exists because the instant image check cannot do its job alone: `image_previews_data` records one null thumbnail for three different causes (null value, missing file, undecodable bytes). The instant check reports them together in those words; the deep check re-reads the values and separates them.
- The tab reads as a scan: `list_checks()` is deliberately independent of a loaded dataset, so every check name is painted with a pending marker before any work happens. The deep checks' pending/running/done is REAL per-check status polled from DeepCheckJob. The instant checks' stagger is not — they take ~5ms total, so their reveal is an animation capped at 120ms per check, and the code says so rather than dressing 5ms up as a scan.
- Cancelling deep checks keeps completed ones (half a findings list is half an answer, like a stopped folder load); a check that never ran is marked "not run", never "passed". A check that raises becomes an explicit error row — a check that could not run is not a check that passed.
- Known limitation: `likely_id_column` reads `value_frequency_histograms`, which only exists for columns classified CATEGORICAL with over 200 distinct values. A mostly-unique short string (e.g. "t1729") classifies as `text` and gets a text-length histogram instead, so the check cannot see it. Widening it would need cardinality carried into the payload for text columns — a data-layer change, deliberately not made for a presentation feature.

## Night mode (done)

A dark theme, chosen in Settings > Appearance as System / Light / Dark (`AppSettings.theme`, default "system").

- The change that made it tractable was collapsing TWO color sources into one. Chart colors used to be literals in app.js while the UI palette lived in style.css; a second palette would have made that four lists to keep in sync. Now every color is a role in style.css and app.js reads the chart ones back out with `themeColor()` AT DRAW TIME — which is also what makes a live theme switch possible at all. Same no-two-sources-of-truth rule the Findings checks follow by reusing chart_data's verdicts.
- The dark column is SELECTED, not an inversion: each hue re-stepped for a dark ground from the dataviz skill's reference palette. The three scatter slots were run through that skill's validator with `--pairs all` against each mode's own surface — light worst CVD dE 9.2 / normal 24.0, dark 9.4 / 20.9, both clear of the >=8 and >=15 floors. tests/test_theme.py pins those six hexes so a hue cannot be swapped without sending whoever did it back to the validator.
- "system" is resolved in JS via matchMedia rather than by a CSS media query, so the dark values are written in exactly one place and an explicit Light/Dark choice wins in both directions without a specificity fight. The OS listener only acts while the preference is "system" — an explicit choice is not overridden when the machine flips at sunset.
- A theme change re-renders charts from a cached `latestChartData`, never a rebuild: rebuilding costs ~0.54s per file (minutes on a big folder) while re-rendering already-fetched data is instant. Per-card "Color by" selections are captured and restored across that re-render.
- tests/test_theme.py parses both palettes straight out of style.css and gates every text pair at WCAG AA 4.5:1 and every chart mark at 3:1. Three marks are exempt WITH their reasons recorded in NON_TEXT_EXEMPT (light aqua at 2.82:1 is a documented property of the validated palette, relieved by the legend; the "Missing" bucket is recessive by design, like a gridline) — an exemption is a visible decision, not a gap in coverage.
- That gate caught a PRE-EXISTING bug: white on `--accent` measures 4.37:1, under AA for .btn-primary's 0.9rem/500 text — every primary button in the app. `--accent` could not simply darken, since it is also `--chart-series` and validated categorical slot 1, so the roles split: `--accent-strong` (#1c5cab light) backs filled buttons at 6.8:1.
- Deliberately NOT themed: the PDF report (printed and shared outside the app, so report_charts.py keeps its own light palette) and the classification banner (its color is a deliberate choice for a security label, and app.js already picks contrasting text for it).

## App icon (done)

An otter logo wired in as the app icon. Source art at assets/otter.png; scripts/make_icons.py regenerates every derived file from it.

- assets/otter.ico bundles 16/32/48/64/128/256px frames in one file. Windows pulls a DIFFERENT frame from the same .ico for the title bar (16), taskbar (32), Alt-Tab (48) and Explorer's large tiles (256), so a single-size .ico looks resampled in most of those places. Each frame is resampled from the full 2048px source rather than from the previous step, so the 16px frame is a clean reduction of the original rather than a reduction of a reduction.
- ml_data_view/main.py passes it via `webview.start(icon=...)`. pywebview's own documentation says that parameter is "supported only on GTK and QT" — that is OUT OF DATE. The installed 6.2.1's winforms backend (platforms/winforms.py) reads the same `_state['icon']` and assigns `self.Icon = Icon(path)`, falling back to extracting python.exe's icon when unset, which is why the window used to show a Python logo. Confirmed against a running window (WM_GETICON returned the otter, and it renders in the title bar), not just by reading the source. .NET's System.Drawing.Icon only accepts .ico, so a PNG here would fail.
- assets/icons/otter-{16,32,48,128,256,512}.png is the same art as standalone PNGs for docs, a web page, or a non-Windows packaging target.
- NOTE ON THE SOURCE ART: assets/otter.png is actually a JPEG (RGB, no alpha) on a solid teal background, despite the .png extension. So every icon carries that teal square rather than a transparent cutout. Keying the teal out is not viable on this file — the fur edges and whiskers are antialiased against it and would fringe or vanish. A genuinely transparent source can be dropped in at the same path and `python scripts/make_icons.py` re-run; nothing else needs to change.
- The .exe/installer icon is NOT wired up: that is packaging, which the roadmap puts in Phase 5. The .ico is the artifact PyInstaller will want when that phase starts.

## Executable build (Phase 5, part 1)

Status: **builds successfully; needs clean-machine verification.** Everything below was verified on the DEVELOPMENT machine, which has Python, the venv, and Google Chrome installed — none of which proves behaviour on a bare Windows box.

### Build command

```
.venv/Scripts/python.exe -m pip install -r requirements-lock.txt
.venv/Scripts/python.exe -m PyInstaller --noconfirm --clean MLDatasetProfiler.spec
```

Output: `dist/MLDatasetProfiler.exe` (~193MB, onefile, windowed) and `dist/MLDatasetProfiler-selftest.exe` (same bundle, console). `build/` and `dist/` are gitignored; the .spec is committed so the build is reproducible.

- requirements-lock.txt pins every package with `==` (pyproject.toml uses `>=` and would resolve newer ones). tests/test_resources.py fails if any line loses its pin.

### How the frozen build is verified at all

scripts/frozen_selftest.py is built as its own CONSOLE executable from the same Analysis. A windowed app cannot report on itself, and the failures that matter here are invisible from source — a lazily-imported package left out of the bundle, a data file at the wrong bundled path, kaleido unable to find a browser. Run it and read the output; exit code is 0 only if all 12 checks pass. Result on the dev machine, frozen: **12/12 pass** (identical to the same script run from source). Covers: _MEIPASS resolution; all 7 bundled frontend/asset files present; writable dir outside the bundle; polars+pyarrow lazy scan staying bounded (largest collect 50 rows of 50,000); sklearn's lazy import projecting an embedding; kaleido browser discovery; kaleido PNG rasterization; a full reportlab PDF; the Api layer (open, chart build, findings); both worker-thread job classes; settings round-trip; checkpoint logging.

### Frozen-only fixes this required

- **ml_data_view/resources.py (new).** Once frozen, `Path(__file__).parent` names nothing real — code runs from a PYZ archive and data files unpack to `sys._MEIPASS`. `resource_path()` answers "where is this bundled file", `writable_dir()` answers "where may I write". They are different questions with different answers, and nothing writes to the bundle.
- **main.py** resolved FRONTEND_DIR and ICON_PATH through `resource_path()`; both were `Path(__file__)` anchors that would have found nothing in the .exe.
- **checkpoint_log.py** wrote its log to a path derived from `__file__`, i.e. inside `_MEIPASS` — a temp directory deleted on exit, and potentially read-only. Since `checkpoint()` is called on every folder load and was unguarded, a failed write there would have crashed folder loading in the packaged app only. Now writes to the per-user app-data directory and swallows OSError.
- **kaleido.mocker** calls argparse's `parse_args()` AT IMPORT TIME, so PyInstaller's submodule scan exited with SystemExit and killed the build. Filtered out in the .spec.

### THE KALEIDO CAVEAT — read before shipping

The Phase 5 note in "Tech stack" above was correct: **kaleido does not bundle a browser.** `kaleido/executable/bin/` contains only log files. Chart rasterization for PDF export drives a real Chromium through choreographer, which finds one by (a) a copy kaleido previously downloaded into `%LOCALAPPDATA%\plotly\choreographer\deps`, or (b) scanning for an installed Chrome/Chromium/Edge/Brave/Vivaldi. So **PDF export is NOT self-contained.** On the dev machine it resolves `C:\Program Files (x86)\Google\Chrome\Application\chrome.exe`. On a clean machine it will most likely find Edge, which ships with Windows 10/11 and is in choreographer's list — but that is reasoning, not a measurement, and it is exactly what the clean-machine test must settle. Everything else in the app works without a browser; only Reports depends on this.

### What was NOT verified, and what to test on a clean machine

Verified only on a machine that has Python, the venv and Chrome. A clean Windows VM with NO Python and NO Chrome must confirm:

1. **It launches at all.** Copy `dist/MLDatasetProfiler.exe` across and double-click. This is the one thing a dev-machine test cannot establish — a bundle can silently pick up a DLL from the dev machine's PATH.
2. **Run `MLDatasetProfiler-selftest.exe` first.** It prints 12 PASS/FAIL lines and is the fastest way to localise any failure.
3. **PDF export specifically**, on a VM with Edge but no Chrome, and again on one with neither. Expect "kaleido: browser discovery" to FAIL in the second case; decide then whether to ship a browser or require one.
4. **Open a real dataset** — a large folder, not the tiny synthetic frames the self-test uses. Peak memory and the bounded-collect behaviour were measured from source, never from the .exe against the 195.8M-row Criteo folder.
5. **The window icon and title bar** (verified here: WM_GETICON returns the otter on the frozen window).
6. **Windows SmartScreen / Defender.** The .exe is unsigned; SmartScreen will warn on first run on a machine that has not seen it. Code signing is not part of this step.

### Known properties

- **Onefile costs ~5-6s of startup**, every launch: the 193MB archive is extracted to a temp directory each time (measured: 11.4/11.5/11.7s frozen vs 5.3/6.3s from source, for the same work). `onedir` removes that and is probably what the installer step wants — it is a one-line change in the .spec.
- **scripts/job_cap.py works against the frozen .exe** (it CreateProcess's an arbitrary command line, so freezing is irrelevant to it). Verified both directions: 3072MB cap → completes, peak 1529.7 MiB; 400MB cap → allocations refused by the OS, exit code 1 at peak 373.5 MiB. The script itself is still dev-only and NOT bundled — but the code behind it now IS, as ml_data_view/jobs.py, and the frozen self-test has a check for it (see "Background job runner").
- The .exe is unsigned and ~193MB.

## Installer (Phase 5, part 2)

Design spec: docs/superpowers/specs/2026-08-31-windows-installer-design.md.

### Build command

```
python scripts/build_release.py
```

Runs the whole pipeline: generates build/version_info.txt from `ml_data_view.__version__`, PyInstaller (onedir), then Inno Setup. Produces `dist/MLDatasetProfiler/` (571 MB) and `dist/MLDatasetProfiler-Setup-0.1.0.exe` (164 MB). Requires Inno Setup 6 (`winget install JRSoftware.InnoSetup`); the script fails with that exact command if it is missing rather than leaving a dist/ that looks finished.

### onefile -> onedir was the biggest single change

onefile unpacked ~193MB to a temp directory on EVERY launch. Measured, same 12 self-test checks: **11.4/11.5/11.7s onefile vs 5.6/5.9/5.6s onedir**, against a 5.3-6.3s from-source baseline — so extraction overhead went from ~6s to essentially zero. The installed app puts a window on screen **1.0s** after clicking its Start Menu shortcut. No amount of installer polish survives a 6-second wait, so this came first. `sys._MEIPASS` is still set under onedir (pointing at `_internal/`), so resources.resource_path() needed no change, and the frozen self-test still passes 12/12 from the onedir build.

### Version metadata (new)

The project had no version string, so Properties > Details on the .exe was blank. `ml_data_view.__version__` is now the single source; the Win32 VERSIONINFO resource and the installer's AppVersion both derive from it, and tests/test_version.py asserts they agree. Version drift between an executable and the installer shipping it is the classic packaging bug.

### What the install actually does

- Per-user to `%LOCALAPPDATA%\Programs\MLDatasetProfiler` — `PrivilegesRequired=lowest`, **no UAC prompt at any point**. Consequence, accepted: installed for the current user only.
- Full classic seven-screen wizard: Welcome, Destination (Browse + disk space), Start Menu folder, Additional Tasks, Ready to Install summary, progress, Finish. An earlier draft specced a three-screen wizard with no directory page, reasoning from Claude Desktop; that was wrong for software people buy and install deliberately.
- Destination page checks the chosen folder is writable and says so on that page, since Setup runs unelevated and browsing to Program Files would otherwise fail partway through the copy.
- Start Menu entry + uninstaller shortcut; desktop icon opt-in.
- Add/Remove Programs entry, verified present: DisplayName "ML Dataset Profiler", DisplayVersion 0.1.0, Publisher, InstallLocation, EstimatedSize, working UninstallString.
- Stable `AppId` GUID so a newer Setup.exe upgrades in place. `CloseApplications=yes` (Restart Manager) so an upgrade over a running instance offers to close it — deliberately not AppMutex, which would mean adding a mutex to main.py for the installer's benefit.
- **Uninstall leaves `%APPDATA%\MLDatasetProfiler` alone.** Settings, saved searches and the audit log are the user's data; an audit log exists precisely so it survives. There is no [UninstallDelete] section, and a test asserts there never is one.

### UNSIGNED — the remaining gap

No code-signing certificate, so `SignTool=` sits commented in the .iss. Windows SmartScreen shows "Windows protected your PC — Unknown publisher" on first run and the user must click More info > Run anyway. **No installer setting closes this**; it needs a certificate. OV (~$200-400/yr, warning fades as install reputation accrues) or EV (~$400-600/yr, trusted immediately, hardware token or cloud HSM). Azure Trusted Signing is ~$10/mo but restricted to US/CA organisations 3+ years old.

### Verified here vs still needing a clean machine

Verified on the dev machine: the wizard runs through all seven screens; files land in `%LOCALAPPDATA%\Programs\MLDatasetProfiler` (5,657 files, 577 MB); Start Menu and Add/Remove Programs entries are correct; the app launches from its Start Menu shortcut in 1.0s with the otter icon; the frozen self-test passes 12/12 from the installed onedir tree. Still unverified, and unchanged from part 1: a machine with no Python and no Chromium browser. The clean-machine checklist under "Executable build" still applies in full, plus: does Setup.exe run on a box that has never seen it (SmartScreen), does the uninstaller remove cleanly, and does a second Setup.exe upgrade in place rather than installing beside.

### Deliberately not built

Auto-update (an update feed, signed release artifacts and a background updater — and on-prem customers often forbid self-updating software), MSI/Group Policy deployment, and code signing itself.

## Background job runner (ml_data_view/jobs.py)

The memory ceiling was a diagnostics script, used by hand to take the measurements quoted throughout this document. It is now the mechanism the app runs its own heavy work through, and the limit is a Settings field.

### What moved, and what deliberately did not

MOVED into a capped worker subprocess — everything measured at or near a gigabyte, or that allocates in proportion to the dataset:

- **Chart building** (`start_chart_build`): 2132MB on two real files, and the embedding projection inside it pulls in sklearn (+554MB of import baseline on its own). This is the operation the ceiling exists for.
- **Deep findings checks** (`start_deep_checks`): reads real data; duplicate detection holds row hashes, the same unbounded-set pattern that exhausted a 2GB cap in categorical_bar_data.
- **Folder loading** (`start_folder_load`): the 42GB incident. Bounded now (1972 MiB on 250 real files), but a ceiling protects against the regression, not just the known case.
- **Dataset comparison** (`start_comparison`): measured at 2140 MiB — already over a 2GB cap.
- **PDF generation** (`start_report_generation`): belongs here for a reason the others do not share. kaleido rasterizes every chart by driving a real Chromium, so this operation spawns a browser; inside a Job Object that browser is a job member, so cancelling or hitting the ceiling takes it down too rather than orphaning a headless Chrome. LEFT IN-PROCESS, on purpose:
- **Everything instant**: schema reads, parquet footer metadata, folder discovery, the 50-row preview, `get_chart_data` over an already-built payload. Paying ~1s of process startup to read a footer would be a worse experience, not a safer one.
- **Row search, its match count and its CSV export** (search.py, search_jobs.py). Search is per-file with early exit and returns in ~0.03s; the count is a streaming aggregation; the export writes per file via streaming and is bounded however large the result. None of the three holds anything proportional to the dataset, and a job per search would turn a 30ms operation into a second.
- **Instant findings checks**: pure functions over two dicts the app already computed, performing no I/O at all — a design constraint of that catalogue, not an accident.

### How it works

- **The channel is files, never a pipe.** The parent writes `spec.json` into a per-job temp directory; the child rewrites a small `progress.json` atomically while it works, and writes `result.json` (or `result.parquet` for tabular results) at the end. The parent polls the first and reads the second exactly once, then deletes the directory. Pickling a chart payload through a pipe would mean holding it twice — in the one process that is NOT capped.
- **Nothing large crosses the spec.** A job is described by file paths, row counts and condition dicts. `pl.scan_*`/`pl.concat` are lazy, so the child rebuilds the same LazyFrame for free and does its own reading inside the ceiling. A chart filter travels as its search conditions, not as a polars `Expr` — api.py already kept both side by side.
- **The worker wraps, it does not reimplement.** The child constructs the same in-process job class the app used before, polls its `progress()` every 100ms and republishes it. So every existing test of those classes still tests the real thing, and the state the frontend polls has the shape it always had.
- **Cancel is graceful first, forced second.** `cancel()` drops a sentinel file; the job sees it at its next item boundary and returns what it finished — which is what preserves the promises those jobs already made (a stopped folder load is a SMALLER dataset; cancelled deep checks keep the checks that completed). A child that has not stopped within 5s is terminated through the Job Object. Either way the temp directory goes, so a partial result can never survive to be mistaken for output.
- **Two layers of enforcement**, unchanged from the original script: the Job Object's commit-charge limit (the OS refusing the allocation, synchronous with it) plus a 50ms polling watchdog reading the job's own OS-maintained peak counters. `JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE` means a crash of the app cannot orphan an uncapped child.
- **A ceiling hit is an expected outcome, not a crash.** It arrives through the same progress dict every other outcome uses — `stage: "ceiling"`, and `error` set to "Stopped at the 8 GB ceiling. Raise the limit in Settings or narrow the selection." Nothing raises across the js_api bridge; the UI shows it in the error banner like any other message. Detecting it takes both signals: the watchdog usually terminates the job, but the commit-charge limit can get there first, in which case the child catches a MemoryError — which stringifies to nothing — and records it as an ordinary error. So a job that FAILED while its peak sat at >=95% of the cap is reported as a ceiling hit; one that SUCCEEDED at 99% simply fitted, and keeps its result.
- **ETA from per-file throughput** where the unit of work is a file (elapsed / files done, times files remaining — a real rate, not an extrapolation of a weighted progress bar), falling back to the fraction for jobs that count something else. Nothing is estimated in the first second or below 3%: one sample of a process still importing polars is noise presented as a measurement. An unknown ETA is omitted entirely, the same omission rule the load stat line follows.

### Frozen builds re-run the .exe as their own worker

`sys.executable` IS the app once packaged, so a worker can only be started by re-running it — and without a check, that would open a second window instead of doing the work. main.py handles `--jobs-worker <dir>` before it creates one; from source the worker is `python -m ml_data_view.jobs <dir>`. This is invisible from source, so frozen_selftest.py has a check that exercises the real path, plus a second that starts a deliberate runaway and confirms the OS stops it and the temp directory is gone. 13/13 now, was 12/12.

### Settings

"Memory ceiling", in GB. Stored as `memory_ceiling_mb`, defaulting to `None` = automatic: half of physical RAM via psutil, floored at 2 GB, with a fixed 4 GB fallback when RAM cannot be read (falling back to the minimum would make every large folder fail on a workstation for no reason). Stored as None rather than as the resolved number so the default follows the hardware — a settings file copied to a bigger machine should not pin an old ceiling forever. A value below the minimum is clamped, not rejected: the field is a number box, and someone who types 1 wants the smallest ceiling available. The box always shows the ceiling actually in force; the hint below says whether it was chosen or derived. **Off Windows there is no OS ceiling**, and Settings says so in a warning rather than letting the app imply a protection it does not have. Everything else about a job still holds — separate process, progress file, cancellation, cleanup. The ceiling tests skip where it cannot be enforced instead of pretending.

### Jobs indicator

One row per running job in the persistent top strip: title, progress bar, "12/250 files · 42,100,000 rows", an ETA when there is one, and its own Cancel. In the top strip rather than in the tab that started the job, because the work outlives the tab the user is looking at — and a job with no visible home is a job nobody can stop. Polled at 500ms (the in-flight load/chart bars poll at 200ms; this one is peripheral and runs the whole time the app is open). That poll is also what keeps "the main thread never blocks" true in practice rather than in principle: `list_jobs()` reads each job's last published snapshot and returns immediately, and a test asserts ten consecutive UI polls all return in well under a second while a job runs.

### Load stat line

When a load ran as a capped job the memory clause answers a different question, so it reads differently: "peak 1.4 GB of 8 GB ceiling" instead of "peak memory 1.9 GiB (process)". The first is that ONE job's peak, measured by the Job Object against the limit it ran under; the second is this process's all-time high-water mark, which Windows offers no way to reset. Carrying "(process)" over would make the more precise number read as the vaguer one. A single opened file still loads in-process and keeps the old clause.

## Per-file profiling cache (ml_data_view/profile_cache.py, file_profile.py)

Re-opening a folder used to redo every scan. Almost none of that work changes between opens: a parquet file on disk is immutable in practice, so its row count, null counts and per-column statistics are the same today as the last time they were computed. They are now stored per FILE — which is what makes an incremental open possible at all: add one file to a folder of 278 and only the new one is read.

Measured on a 4-file synthetic folder through the whole Api round trip: first open scans 4 files, second open scans 0 and produces a byte-identical chart payload.

### The store: SQLite, not one JSON document per file

Both were considered. SQLite wins on four specific things:

- A folder open touches EVERY file's entry. JSON-per-file means 279 opens, reads and parses to answer "is any of this still valid", against one indexed query.
- A cache key is an absolute path, and a path is not a filename — it exceeds Windows' filename length limit, contains separators, and hashing it into a name reintroduces collisions to handle. A TEXT primary key has none of those problems.
- Writes are transactional. Writing 279 documents is 279 chances to be interrupted and leave a half-written one that still parses as valid JSON.
- "Cache size on disk" and "Clear cache", both required by Settings, are one stat() and one DELETE rather than a tree walk. sqlite3 ships with Python, so this adds no dependency. The statistics themselves are a JSON document in one column: their shape evolves with the profiler, and a SCHEMA_VERSION bump invalidates every row wholesale, which suits data that is always safe to recompute far better than DDL migrations.

### The key, and what invalidates it

`(absolute path, size, mtime_ns, SCHEMA_VERSION)`. Not a content hash: hashing a 12GB folder to decide whether to read it defeats the purpose, since the read IS the expensive part. mtime in NANOSECONDS, because `st_mtime` is a float that loses precision at real timestamps and can make two different modification times compare equal. `SCHEMA_VERSION` is bumped whenever a cached number could be computed DIFFERENTLY by new code — a changed bin count, a changed definition of "missing", a new per-file statistic old rows do not carry — so a code change can never serve a stale statistic that silently disagrees with what the same code would produce now. It is deliberately NOT tied to `__version__`: most releases change nothing about how a statistic is computed, and invalidating a 279-file cache every release would make the feature pointless.

### EXACT MERGES — identical to a full recompute

- Row count (sum), per-column null counts (sum), NaN counts for floats (sum), missing counts = nulls + NaN (sum). Nulls and NaN are cached separately as well as together: they have different causes, and a NaN from a broken upstream computation is worth being able to see on its own.
- Numeric per-column count, min, max, sum (sum / min / max). `sum` is not drawn by any chart yet; it is cached because it is the one aggregate that makes a merged MEAN exact, and it costs nothing in a query already running.
- Value-frequency ("detections per value") buckets — sum. Exact for the statistic AS DEFINED: the folder version already counts a value once per file it appears in and says so (`per_file`).
- Numeric histograms AND text-length histograms — sum of per-file bin counts, PROVIDED every file was binned against the same edges. That proviso is the whole reason a spec is stored beside the counts (below).
- Correlated missingness' per-file co-null COUNTS (sum) — a co-null row belongs to exactly one file. See the "recomputed" list for the part of that feature which does not cache. tests/test_file_profile.py pins all of these against a direct recompute on a multi-file fixture — whole payload, then bin edges, counts and x_scale column by column, then nulls/NaN/min/max against a query over the concat.

### APPROXIMATE MERGES — deliberate, with the error stated

- **Categorical top-K.** Each file caches its top 1000 values with counts (raised from the 200 the bounded path used, because a merge is only as good as the shortlist it merges). A value that ranked below 1000 in EVERY file is missed. `truncated` is set only when some file actually filled its shortlist, and the UI marks that chart — "Counts merged from per-file shortlists — a rare category could be under-counted." "Other" still absorbs the difference against the exact non-null total, so the chart adds up to the real row count regardless.
- **p25/p75.** A row-weighted mean of per-file quantiles, which is not the global quantile and cannot be made into one from per-file summaries. They feed nothing but `_is_skewed`'s range-vs-spread test, whose threshold (10) sits an order of magnitude from the reference distributions — the same trade `_numeric_stats_per_file` has always made.
- **The categorical-vs-text verdict.** Combines per-file distinct/row RATIOS, since distinct values recur across files and cannot be added. Reproduces `_classify_string_column_per_file`'s existing folder verdict exactly, from cached scalars, with no scan — which is why the classification phase of a folder load is now arithmetic rather than a second pass over every file.

### RECOMPUTED ON LOAD — never cached, and why

- **Null heatmap**: a sample of actual ROWS. No merge of per-file summaries reconstructs specific rows.
- **Embedding projections**: a PCA basis is fitted to the data it projects; two files' bases are not combinable.
- **Image thumbnails**: decoded bytes, not statistics — and caching base64 blobs would bloat the store for something that is already a bounded read of the first 50 rows.
- **Correlated missingness' candidate GROUPS**: which pairs are worth checking depends on the MERGED null counts, so it cannot be known file by file. Only the group structure is recomputed; the per-file co-null counts it then needs are cached and summed exactly (above), so a re-opened folder pays nothing for the scan. All four are bounded reads by construction, so recomputing them costs a fraction of a cold profile rather than undoing the cache.

### Histograms carry their spec — the range-expansion case

Per-file bin counts only add up if the bins are the same bins, and the global min/max is not known until every file has been read. So the edges are derived from the MERGED min/max and each file's counts are stored with the spec that produced them — `[low, high, bins, scale, offset]`, five numbers rather than 31 floats per column per file, which across 250 files is megabytes of JSON to say what the five already say.

Re-opening an unchanged folder derives the same spec and every file hits. A new file that widens the range derives a different spec, so that COLUMN's counts miss for every file and are re-binned — and nothing else is: not the other columns, not the same column's min/max/quantiles/categories, not anything already merged. It is reported in the payload

(`cache_stats.range_expanded_columns`) and shown in the Distributions tab, because it is the one case where opening a cached folder does real work and a silent pause is worse than a named one.

### Incremental load, and what it says

`file_profile.load_folder_profile()` is the single entry point: hit the cache per file, scan the misses, merge. Removed files need no bookkeeping — the merge is over the files discovered NOW, so a file that is gone contributes nothing. Cancellation is still checked at a FILE boundary, so a stopped load is a smaller dataset rather than an incomplete one, exactly as before.

The stat line says what actually happened: "Profiled 195,800,000 rows in 4.1s — 12 of 279 files profiled, 267 from cache — peak 1.4 GB of 8 GB ceiling". The cache clause REPLACES the plain file count rather than joining it — two file counts in one sentence make a reader stop and check whether they are the same number — and is omitted entirely on a first open, where "279 of 279 files profiled, 0 from cache" is a sentence about an absence that "across 279 files" already says better.

### Settings

"Profiling cache" shows size on disk (including the WAL, which is part of the database and can be larger than the .sqlite after a big write) and how many files it holds profiles for — either number alone is misleading. **Clear cache** empties it and VACUUMs, so the size Settings reports actually goes down. **Force full re-profile** is deliberately a different action: it drops only the LOADED folder's entries and immediately reads it again, which is what someone does when they suspect the numbers on screen are wrong; other folders keep their profiles.

### Consequences elsewhere

- **A filtered chart build never uses the cache.** A filter changes which rows every file contributes, so a cached per-file statistic is not about the same rows — and keying entries on a filter the user edits constantly would cache nothing useful while making every entry look valid. `build_chart_payload`'s direct per-file path stays for that case, and is also the reference implementation the merge is tested against rather than dead code.
- **Category ranking is now deterministic.** Counts tie constantly on real categorical data; without a second sort key the same ten bars came back in different orders between runs, and the merged path (ranking a Python dict) and the direct path (ranking a polars table) disagreed outright. Both now break ties by value.
- **A constant-length text column gets one bin**, matching what numeric histograms already did. Asked for 30 bins over a zero-width range, polars widens it and returns edges whose first element sits ABOVE the second — a malformed axis for an answer that is really "one bin".
- **`load_folder_group()` now sets `_current_files`**, which it never did: a folder loaded through the synchronous path used to look like a single file whose "path" was a directory.
- The folder load's classification phase no longer scans. The distinct/row scalars come back with each file's profile in the same pass as its null counts, so phase 2 is arithmetic over numbers already in hand — fewer queries than before, even cold.

## Folder watching (ml_data_view/folder_watcher.py)

Point the app at a folder a pipeline is writing into and it re-profiles as files land — reading only what is new, because the per-file cache above makes that the natural thing to do rather than an optimisation.

### The module is a GATE, not a watcher

A filesystem event says "something happened to this path", which is not "this file is ready to read". The overwhelming majority of events a watched folder produces are the middle of a write, and profiling on the event gets a row count for a third of a file, cached under a key that looks perfectly valid. Three things stand between an event and a re-profile:

- **Debounce.** A burst of events for one path collapses into one pending entry. `note_event()` only ever updates state; nothing is emitted per event. A single write can produce dozens of events and causes one re-profile.
- **Size stability.** A file is finished only once its size has not changed for the configured interval (default 10s). ANY change restarts that clock, so a writer that appends every nine seconds never clears a ten-second gate — which is the correct answer.
- **Footer validation.** Size stability is still not proof: a writer can stall, and some tools pre-allocate a file at full size and fill it in afterwards. Parquet writes its footer LAST, so a file whose footer does not parse is not finished whatever its size has been doing. Reading it decodes no column data — the same metadata-only budget folder discovery works to. CSV has no equivalent (it is valid at every truncation point), so those rely on size stability alone, and the code says so. A file that never passes all three is not dropped and not retried into silence: it stays in `pending()` as "still being written" and the top strip says so. A stalled pipeline should look like a stalled pipeline.

### Testability is the shape of the module

The state machine is driven by two methods — `note_event()` and `tick()` — that take no filesystem events and do no sleeping. The watchdog observer is a thin adapter calling the first; a poller thread calls the second. So the gate is tested with an injected clock and a burst of events is a burst of calls, and exactly one test uses a real observer against a real temp directory, covering the one part the rest cannot: that watchdog's events reach `note_event` at all.

### One re-profile, batched, and never in the foreground

The watcher runs in the UI process — an observer thread plus a stat() per pending file, which is not work. It NEVER starts a job itself: `on_change` sets a flag, and the frontend's existing 500ms poll picks it up. That is what keeps new data from interrupting whatever someone is in the middle of; nothing is pushed from Python, and there is no dialog and no stolen focus. A flag rather than a queue, deliberately: twenty files arriving at once want ONE re-profile, because the merge covers the whole folder regardless of which file prompted it. The flag is cleared when the re-profile STARTS, not when it finishes, so a file arriving mid-re-profile sets it again and gets its own pass rather than being swallowed by the one in flight. The re-profile itself re-discovers the folder rather than appending the new file to the loaded list. Discovery is schema reads and footer metadata only (never a collect), it is how a REMOVED file drops out, and it is the only thing that can tell whether an arrival belongs to the loaded schema group or a different one. The group is then found again BY SCHEMA, not by index: groups are ordered by file count, so a second group growing past the first would otherwise swap the dataset under the user without a word. The load that follows is incremental through the profile cache — measured on a 3-file folder, an arrival reads 1 file and takes 2 from cache, and the stat line says so.

### What the user sees

- A **"Watching <folder>"** indicator in the persistent top strip, with a live dot, pause/resume, and a count of anything still being written. In the top strip because a watcher that reads files on its own initiative must be visible and must be stoppable from wherever the user is.
- **Pause suppresses PROCESSING, not collecting.** Events keep landing while paused and resuming catches up on all of them — a pause that dropped events would leave the profile describing fewer files than the folder holds, which is the exact failure this feature exists to prevent.
- A **corner toast** when a re-profile completes: "Re-profiled 1 new file — 120 rows total. 1 new finding: High null rate in column value." It fades on its own, takes no focus, and is `pointer-events: none` so it cannot swallow a click meant for what is underneath it.
- The findings half of that sentence comes from `findings.diff_findings()`, against the previous scan OF THE SAME DATASET. Identity is the check plus its columns, NOT the detail text — "41% missing" becoming "43% missing" is the same finding, and reporting it as new would train people to ignore the notification. A first scan reports `None` rather than calling every finding on a freshly opened dataset "new", and switching datasets resets the comparison instead of diffing one against the other.

### Settings

"Folder watching": enable/disable, the watched folder (with "Watch the loaded folder" — by far the common case — and a folder picker), and the stability interval in seconds, floored at 1. Below a second, ordinary filesystem latency alone can make a file that is still being written look stable between two stat() calls, so a user who types 0 gets the floor rather than a watcher that profiles half-written files.

One folder at a time, deliberately: two watched folders means two merged profiles to keep straight, and the app shows one dataset. A watch that was on when the app closed is on when it reopens (`sync_watch()` at startup); changing the folder or the interval REPLACES the watcher rather than reconfiguring it, since both are decided at construction and a half-reconfigured watcher is the kind of thing that quietly keeps watching the old folder.

### Audit log

`watch_started`, `watch_stopped` and `reprofile` join `file_open` / `folder_open` / `report_generated`, subject to the same enable switch. Their own actions rather than folded into `folder_open`: watching means the app reads files nobody explicitly opened, at times nobody chose, and an audit log exists precisely to record that. `watch_stopped` is the closing bracket, so a reader can tell a live watch from a finished one without inferring it from the absence of later entries.

### Packaging note

watchdog picks its observer backend at RUNTIME (ReadDirectoryChangesW on Windows), so nothing static reaches `watchdog.observers.winapi` and the module graph misses the whole backend. The .spec bundles it explicitly; the frozen self-test starts a real observer and asserts an event arrives, because the failure mode otherwise is an .exe that reports "Watching" forever and never fires. 15/15 now, was 14/14.

## Directory structure (ml_data_view/hierarchy.py)

Otter opens whatever it is pointed at. The structure model exists to DESCRIBE what that turned out to be, not to require a shape. Five inputs have to keep working, and all five are in the tests:

| shape | what it is | detected? | unit | scope |

| --- | --- | --- | --- | --- |

| single | one file | no | the file | none |

| flat | a folder of files | no | the file | none |

| hive | `day=.../part-*.parquet` | yes — groups, no series | the file | by partition | | nested | an arbitrary tree | no | the file | none (mappable by hand) | | full | `sys_key=…/run/…/chunks` | yes — group + series | the series | by group, then unit |

### Three roles, both of them optional

- **GROUP** — this level's directory names identify something the data is split by. Any number of group levels, nested.
- **SERIES** — a directory at this level holds the ordered chunks of one dataset. At most one.
- **IGNORE** — structure, not a role. Also the default. The UNIT is what loads as one dataset: a series directory and its chunks, or — with no series level — a single file. That is not a special case bolted on; it is the same model with a level absent, which is why a flat folder needs no separate code path and why "no roles at all" degrades to exactly the schema-group behaviour that predates all of this.

### Detection suggests from EVIDENCE; it never decides

Every HIVE (`key=value`) level is a group — `key=value` is the one convention that explicitly says "this level identifies something". Only a contiguous run of hive levels from the outermost counts, so a hive directory buried under three arbitrary levels is a partition INSIDE something rather than the thing itself.

A SERIES is not positional. `_suggest_series_depth` picks the SHALLOWEST named level whose directories mostly look like series, where "looks like a series" means, from the parquet footer alone:

- **same schema** across the directory's files — a directory holding two shapes of file is holding two things;
- **non-overlapping time ranges** — chunks of one dataset partition time, they do not cover the same moments twice. Overlap disqualifies; a GAP does not, because a recording can legitimately pause and treating a pause as evidence against would reject most real data;
- **at least two files** in at least one directory — one file per directory is just a file in a folder, and the tree is making no claim about ordering. A MAJORITY of the level's directories must qualify, not all of them: one directory holding a stray file should not cost the whole level its role, the same reasoning as the majority-schema rule for chunks within a unit. Hive levels are excluded outright — a group of well-behaved series would otherwise qualify as a series itself. Shallowest, because a series is the outermost directory holding one dataset's chunks; every level below it would qualify on the same evidence and is structure inside that dataset. The positional rule this replaced ("the named level below the deepest group") was rejected for two reasons: it cannot see a series that has no group above it, and it names a level a series on no evidence whenever one happens to sit there. Evidence also means the answer degrades honestly — a tree with no timestamp column gets no series suggestion at all, because the evidence is about time and no evidence is not evidence of a series. Detection now reads EVERY chunk's footer rather than the first one's, so hierarchy.py memoises footers per (path, size, mtime) for the session: confirming a mapping would otherwise pay for a second pass over the whole tree for answers already in hand. That memo is a convenience within one discovery — profile_cache.py is the one that persists. `detected` means "there is something to confirm", not "this is a fleet". And detection failing is not the same as there being nothing to map: an undetected tree still reports its levels, so somebody with one can assign roles the code refused to guess. The dialog is always offered on a first open, always overridable, and carries a **"Load as a flat folder"** button — the escape hatch that keeps the structure optional in practice as well as in principle. Declining is remembered as firmly as confirming, or the offer would be undone by the next open. **A folder with sub-directories that loads flat says so**, in a quiet note under the top strip with a "Set folder structure…" button beside it. Silently flattening someone's carefully arranged tree with no word about it is how they conclude the app cannot read their data. A folder of files gets no note — a flat folder is not a tree that failed to be recognised, and there is nothing to explain.

### The vocabulary is generic; the words are the user's

Nothing in the code knows what a "system" or a "recording" is. The mapping carries a LABEL per level plus a label for the root, and those are what the breadcrumb, the dialog and the derived column names use. Defaults are "Group", "Series" and "All" — and a hive level defaults to its own key, so `sys_key=123` produces a column called `sys_key` with no typing at all.

### Mapping persistence

Saved per root path in settings.json (beside saved searches: a mapping is a preference about a folder, not data about it, so clearing the profile cache must not make the app start asking again), against a FINGERPRINT of the tree's shape — which levels exist, how many distinct names each has, which are hive-style. Adding files never changes it; a new LEVEL does, and the dialog is offered again rather than leaving someone with a mapping that no longer describes their folder.

### Chunk ordering (a SERIES concern)

Natural sort of the chunk paths (`part-2` before `part-10`), then VERIFIED against each chunk's timestamp min/max from the parquet footer. Time wins on disagreement — a filename is a label, a timestamp is a measurement — and raises a `chunk_order_vs_time` finding rather than being silently corrected. If ANY chunk's bounds are unknown, filename order stands and no conflict is claimed: a partial ordering by time is not an ordering. A shape whose units hold one chunk each can raise none of this at all. Chunks of a unit must share a schema; one that deviates is excluded with a reason naming what differs and raises `deviant_chunk_schema`. The reference schema is the MAJORITY, ties broken by natural order — "the first chunk happened to be the broken one" is a real way to lose a whole unit to one bad file. Across units, schemas may differ; `schema_groups()` reports that as information and does not decide what loads.

### Timestamp detection works without a structure

A datetime dtype wins outright (the dtype IS the answer). A numeric column qualifies on its NAME (`ts`, `event_time`, `epoch`…), with a unit guessed from magnitude — seconds are ~1.7e9 and each finer unit adds three orders of magnitude — and both are confirmed in the dialog where there is one. `detect_timestamp_column()` takes a plain polars schema, so a single CSV reports its time column too. That is what lets a time-series check require a timestamp column **and nothing else**.

### Derived columns are named by the mapping

One per group level, then the unit column, then `chunk_index` and `chunk_file`. A level's name comes, in order of preference, from a LABEL the user typed ("System" gives `system_id`), else its HIVE KEY (`sys_key=123` gives `sys_key`, already the name the data uses), else a positional fallback (`group_1_id`). A label that is merely the hive key echoed back is not a choice — propose_mapping fills it in as the default — so the key wins there.

That is how the motivating tree reconciles with the no-hardcoded-vocabulary rule: labelling its levels "System" and "Recording" produces exactly `system_id, recording_id, chunk_index, chunk_file` — the columns its owner would have written by hand — with neither word appearing anywhere in the code. A different dataset labelled "Site"/"Trip" gets `site_id, trip_id`, and a test pins both.

A name that would collide with a real data column is suffixed rather than shadowing it. They are lazy

literals per chunk — constants known from the path — so attaching them reads nothing, and they reach:

- the preview and schema table (null counts stated as 0, not counted);
- search, its match count and its CSV export, attached PER FILE so filtering by a group keeps the early exit that makes a folder search return in milliseconds;
- the charts, as rows-per-group and rows-per-unit bars SYNTHESIZED from per-chunk row counts — exact, and no read at all. `chunk_index` and `chunk_file` get no chart: identifiers of position and of file, not measurements. Because the names come from the mapping, **no module downstream holds a fixed list of them**: `identity_columns()` reads them off the identity rows, and `chart_data.drop_derived()` takes them as an argument.

### Scope

A breadcrumb in the persistent top strip — root label, then one crumb per chosen group value, then the unit — in the MAPPING's words. Scope is a prefix of the group values, optionally ending at one unit, so it works at any depth including zero. **A scope change is a LOAD, not a filter**: narrowing means fewer chunks to read, and it is incremental through the profile cache (measured: narrowing a profiled tree scans 0 files, 5 from cache). Hidden entirely when there is no structure.

### Checks declare what they need

`CheckSpec.applies` is a predicate over `dataset_info["capabilities"]`, and a check whose precondition is absent reports `skipped` with a reason rather than passing. The conditional checks and what each needs:

- `structure` — a detected structure. Surfaces the findings hierarchy.py already computed from footers.
- `timestamp_integrity` — a timestamp column. Runs on a single CSV.
- `chunk_boundaries` — a series level. Reads the first/last timestamps cached WITH each chunk, so it re-reads nothing; that is exactly why those two numbers are cached there.
- `group_balance` — a grouping column: a directory role, a hive key, or one the user picks in the Findings tab. It will NOT default to an arbitrary categorical column the chart build happened to find — a check that fires on a guess is worse than one that does not fire.
- `fleet_channel_divergence` — a grouping that produced TWO OR MORE groups. One group is not a comparison, and "this group differs from the aggregate of itself" is always false rather than usefully false.
- `fleet_group_duration`, `unit_outlier` — that, and unit durations. A grouping with no time axis can still have its channels compared; how long its recordings ran is a question with no answer.

### Flat inputs are unchanged, and that is a test

tests/test_structure_api.py pins a flat folder's chart payload against `build_chart_payload`'s own per-file path — the implementation that predates every structure feature — key for key, and asserts no derived columns, no scope, no structure findings and no synthesized bars. Both test files are organised around the five shapes above, with the shape-independent rules parametrised across all of them: a rule that only holds for the shape it was written for is not a rule.

## Time-series health (ml_data_view/timeseries.py, ts_checks.py)

Eleven checks about ORDER, in the Findings registry beside every other check, plus the charts to see what they are talking about. The rest of the app computes order-free statistics: a null count does not care which rows the nulls were in. A flatline is nothing but which rows.

### Applicability, which is the whole design

These hang off a DETECTED TIMESTAMP COLUMN, not off a directory structure:

- **a timestamp column** — `timestamp_monotonic`, `duplicate_timestamps`, `sample_rate`, `time_gaps`, `short_unit`, `channel_nan_run`, `channel_flatline`, `channel_saturation`, `channel_out_of_range`. A single CSV with a time column gets all nine.
- **a timestamp column AND a series level** — `missing_chunk`, `chunk_overlap`. A unit that is one file has no boundary between chunks, so "is a chunk missing" cannot be asked of it, let alone answered with a green tick. The "unit" is whatever the structure mapping says — a series if there is one, else a single file, which is how a flat folder gets per-file checks and a lone CSV gets them too.

### Memory: bounded by a BATCH, not by a recording

The brief allowed per-chunk in-memory processing because a chunk is bounded. The implementation does better: every column is read through pyarrow's `iter_batches`, which splits WITHIN a row group, so peak memory is BATCH_ROWS x one column regardless of chunk size — the same technique and the same reason as data_loader._bounded_parquet_missing_counts. Run lengths, dt statistics and boundary state are carried incrementally across batches, then across chunks.

**Why not polars streaming**, since the brief asks: `diff`, `shift` and `rle` are all ACCEPTED under `collect(engine="streaming")` — they do not raise, which was verified — but each is a whole-column, order-dependent operation, so the engine materialises the column to answer it. That makes peak memory a function of chunk size rather than batch size, and it offers no way to carry a run across a chunk boundary, which is the entire difficulty. So the analysis is numpy over pyarrow batches. polars is still used where it genuinely streams.

Measured through the capped chart-build job, on 20,000-row chunks: **10 chunks (200k rows) peaked at 722.8MB; 60 chunks (1.2M rows) at 735.2MB** — six times the data for 1.7% more memory. A test asserts that ratio rather than an absolute number.

Non-parquet sources (a single CSV) have no seekable structure, so they are read in ONE streaming pass over just the two columns needed. Memory is then bounded by those columns rather than by a batch — a real difference, stated in the code, and only ever reached for a single file someone opened on purpose, since a series of chunks is parquet-only.

### What is stored per chunk, and how it merges

The per-chunk artifact is a fixed-size dt HISTOGRAM (log-spaced, ten bins per decade), the largest dts with their start times (capped at 200), up to 24 time buckets, and per channel: NaN and flatline run lengths with their locations, LEADING and TRAILING run lengths, first/last values, min/max with counts, and a bounded deterministic sample for the percentile rule. Three kinds of merge:

- **SUMS** — rows, NaN counts, duplicate counts. Exact.
- **HISTOGRAM-DERIVED** — median dt and its spread, read off the summed bins. Summing is exact; reading a quantile off ten-bins-per-decade is within about 12%, which is far finer than any threshold here (a gap is 5x the median, not 1.1x).
- **BOUNDARY-ONLY** — a gap falling exactly between two chunks, an overlap, a backwards step from one chunk to the next, a NaN run spanning a seam. None exists inside any chunk and none needs a re-read: each is arithmetic on neighbouring chunks' cached first/last values. That is exactly why those are cached. Runs join across seams using the leading/trailing lengths — a sensor that dies rarely does it inside one file, and reporting two shorter runs would understate precisely the case that matters. A flatline additionally requires the values either side of the seam to match: two chunks stuck at different values are two flatlines, not one. Saturation counts are EXACT rather than approximate: a chunk whose own minimum sits above the global one holds no value equal to the global one, so summing the counts from the chunks that do reach it is the whole answer.

### Out-of-range: two rules, labelled differently

With a DATA DICTIONARY configured (`timeseries_ranges` in Settings), this is an exact count against a documented range and is CRITICAL. Without one it is a HEURISTIC on the extremes — how far the largest value sits beyond the p0.1–p99.9 spread — which says "HEURISTIC" in its own finding text and points at Settings, because a number nobody can check must not be presented as one they can. The heuristic also stays silent below 1,000 rows: with a few hundred values p99.9 lands on the outlier itself and the rule cannot see what it is looking for.

### Severity

Missing chunk, overlap and non-monotonic time are CRITICAL — each means a figure on screen describes something that did not happen. Gaps, rate changes, flatlines, saturation, dropouts and a short recording are WARNINGS: the data is real, some of it is missing or stuck. Jitter is INFO. All thresholds are in Settings; changing one re-judges the numbers already cached, and only the documented ranges cause anything to be read again (they are the only input that changes what is STORED).

### Charts

A Health tab at unit scope: a COVERAGE STRIP (data / gap / missing-chunk), a DT-OVER-TIME plot, and a PER-CHANNEL strip (missing / flatline segments). Everything is downsampled before it leaves Python — bucketed aggregates and segments, never raw rows — so a 200-million-row recording sends the same few hundred numbers as a 200-row one. Which stretches count as gaps, and which gaps are missing chunks, is decided in Python beside the checks that report the same thing: a second copy of those rules in JS is how the chart and the finding next to it would end up telling different stories.

### PDF

A "Recording health" section, offered on the same precondition. The ROLLUP comes first — "3 of 19 recordings with missing chunks" is what a reader wants above system or fleet scope — then a row per unit. Every figure was computed during the chart build and cached per chunk; nothing is read to produce the section.

### Cost, stated

The timestamp pass is one read per chunk. Each CHANNEL is another, so a 40-column folder would multiply a first profile by forty — hence `timeseries_max_channels` (default 12), applied in schema order, with what was left out reported in the payload rather than silently absent.

## Unit catalogue and group comparison (ml_data_view/catalog.py, fleet.py)

A row per unit of analysis, and a per-channel comparison of every group against the others. Both are built during the chart build out of statistics already merged, and both are OPTIONAL in exactly the sense everything else in this app is: they describe whatever was opened rather than requiring a shape.

### Neither module knows what it is looking at

catalog.py is handed a list of units and describes them; fleet.py is handed a (file -> group value) map and compares those groups. Neither asks what the tree looked like, and neither contains the words "system", "recording" or "fleet" — the labels a user sees come from their own mapping, with generic defaults ("File", "Group", "All"). So the same table renders a fleet of 200 recordings and a folder of three parquet files, with different numbers in it. All five input shapes get a catalogue; the ones with nothing to group by SKIP the comparison rather than failing it.

| shape | catalogue rows | comparison |

| --- | --- | --- |

| single | one, the file | skipped — nothing to group by | | flat | one per file | skipped — nothing to group by |

| hive | one per file | by partition key |

| nested | one per file | skipped, unless a level is mapped by hand | | full | one per series | by group level |

### Every figure comes from something already computed

Chunk count, rows, size on disk and channel count come from the cached per-file load sections plus one stat() per file. Start, end, duration, rate and gaps come from the merged per-unit time-series artifact, and are ABSENT — the whole column dropped, not filled with zeroes — when no timestamp column was detected. Nothing is read to produce either section, on a first profile or on any after it, and a test asserts a second profile of the same tree scans zero files and still produces both.

### Health is attached after the checks, not during the build

A row's health is the worst severity among the findings that NAME that unit, with a count. Findings run after the chart build that produces everything else, so the payload carries "unknown" and api.py fills it in (`catalog.attach_health`) — a status computed before the checks ran would be a guess, and "unknown" is deliberately not green. `Finding.unit` is what makes this a lookup by equality rather than a parse of the detail text; it is also now part of `finding_key`, so the same gap appearing in a second recording is news rather than a duplicate. A finding that names no unit — a column is mostly null — belongs to the dataset and marks no row unhealthy: a catalogue that calls all nineteen recordings unhealthy because one column has a high null rate has stopped distinguishing them.

### The comparison reuses the Compare tab's measure

Divergence is PSI, scored with compare.py's own `psi` and `severity` and its conventional thresholds. Not a second drift statistic: someone who has read the Compare tab's numbers already knows what 0.31 means. The shared bins come for free. Every per-file histogram in the cache was binned against a spec derived from the MERGED min/max of the whole loaded scope, so every file in every group is already binned identically — a group's histogram is the elementwise sum of its files' cached counts, and the aggregate is the sum of all of them. Exact, and no read at all. Means are exact too (the cached per-column `sum` over the cached count, which is the reason `sum` was cached at all).

### The baseline is the TYPICAL GROUP, not the pooled aggregate

The bin-wise MEDIAN of the groups' proportions, renormalised. The pooled aggregate was the obvious choice and is wrong, for a reason that appears the moment the feature works: one system whose channel sits somewhere else puts a quarter of the aggregate's mass there, so every OTHER system now diverges from the aggregate too — and a screen that flags all four systems has told nobody which one to look at. A median profile cannot be moved by a minority of groups. `psi_aggregate` is reported beside it for anyone who wants the pooled number, and the pooled aggregate is still what the charts outline: it is a real distribution and a useful backdrop, just not a baseline. The same leave-one-out reasoning governs the duration comparison, which is against the median of the OTHER groups' medians.

### The grouping must be constant per file, and says so when it is not

A cached per-file histogram cannot be split by a value that varies WITHIN the file; recovering that means re-reading every file with a group_by, which is the cost this feature exists to avoid. A grouping that is a property of the path — a directory role, a hive key, any derived column — is supported from the cache; an ordinary data column reports the section unavailable WITH THAT REASON rather than quietly producing nothing. This is a statement about where the value lives, not about how deep the tree is: nothing about its shape is assumed either way. The other group check (`group_balance`) still accepts any column, because it reads a chart bar rather than a per-file histogram.

Changing the grouping column does NOT rebuild anything. The bins do not depend on how files are grouped, and every histogram is stored with the spec that produced it, so the edges are read back out of the cache and the whole comparison is rearranged from statistics already written down (`file_profile.fleet_from_cache`). The findings follow the same payload, so the tab and the findings beside it can never describe two different groupings of one folder.

### "vs the previous group" is offered on EVIDENCE

A time-based partitioning is just a grouping whose natural order is time, and for one of those "how does this compare with the PREVIOUS one" is the question people have. That is decided on evidence, not on a key called "day": the groups qualify when every one of them has known time bounds and, sorted by start, their spans do not overlap. A grouping by vehicle fails that immediately, which is the point. This subsumes what would otherwise have been a separate partition-drift feature.

### Robust everywhere, because the question is "which one is odd"

Group means are compared with a median and a MAD-scaled sigma, and a unit is compared against its OWN group's median — a group that records in five-minute bursts is not thereby full of anomalies. An outlier inflates a standard deviation, which is exactly how a real deviant hides. Two places where that needs a floor rather than a formula: a MAD of zero (most units agreeing to the last decimal) is raised to 1% of the median, because neither "infinitely many sigma" nor "no spread, so nothing to report" is right; and a spread of zero across group means quotes no sigma distance at all and lets PSI carry the claim.

### What is bounded, and what is not

Every channel is SCORED — a score is five numbers — so a finding about the fortieth column is never lost. Only the most divergent `fleet_max_channels` (default 12) carry their per-group bin counts, and the frontend's picker draws one of those at a time; a per-group histogram for every column of a wide dataset is a payload that grows with a column count nobody is looking at. Significant divergences are listed individually up to 20 and then counted; moderate ones are always a single INFO row, because forty rows of "slightly different" buries the two that are not.

### Comparing two of them is the same comparison

`start_scope_comparison(scope_a, scope_b)` hands two scopes' file lists to the SAME CompareJob the Compare tab already uses. "Compare two recordings" and "compare two systems" are one operation at two depths, because a scope is a prefix of the group values optionally ending at one unit — so the method needs no notion of either. Both sides are compared whole rather than sampled: a scope is already the narrowing, and sampling it as well would answer a question about part of the selection the user just made.

### Elsewhere

- Settings gains `fleet_duration_ratio` (0.25), `fleet_outlier_sigma` (3.0) and `fleet_max_channels` (12). The PSI thresholds are deliberately NOT settings: they are a standard scale, and a per-install version of one would make "significant" mean something different in each copy of the app.
- CSV export writes every row and raw numbers — a spreadsheet cannot sum "1m 23.4s" — with the group values as their own leading columns so it can be pivoted on them.
- The PDF gains a catalogue SUMMARY, offered above unit scope only: top-level numbers, a line per group, and the unhealthy units named. Not every row — a fleet of 200 recordings is a report nobody prints, and the CSV is one click away.
- `get_chart_data()` now passes the same time-series inputs `start_chart_build()` does. The two are meant to produce the same payload, and a headless caller should not silently get a dataset with no health section for taking the blocking route.
- tests/test_fleet_api.py builds a synthetic fleet whose channel values are an equidistributed sweep rather than a sawtooth or a ramp. A sawtooth revisits its extremes often enough to trip the saturation check; a ramp ties the range of values to the row count, so a short recording would look like a channel with a different distribution rather than like a short recording. The two properties under test have to be separable, or the test cannot say which one was detected.

## Channel timeline (ml_data_view/timeline.py)

A unit's channels drawn over time, one strip each on a shared axis, with the findings marked on it. Its precondition is a DETECTED TIMESTAMP COLUMN and nothing else — a single CSV with a time column gets one; a series just means the viewer has several chunks to choose between. Nothing in the module asks what the directory tree looked like.

### The one rule that makes it affordable

A fetch reads the ROW GROUPS whose own time range overlaps the requested window, and only when their combined row count is within

`timeline_max_rows` (default 2,000,000). Otherwise it answers from the per-chunk buckets already in the profile cache, at whatever resolution those give, and SAYS SO.

Both halves matter. Because the cached buckets are always an acceptable answer, the tab opens instantly on a recording of any size: a 200-million-row chunk hands back its twenty-four cached buckets rather than being read. Because a read is planned from parquet FOOTER statistics before a byte is decoded, zooming into that same chunk reads the handful of row groups the window touches — so detail appears as you zoom, the row count is known in advance, and no request ever reads more than the budget names. The payload reports which of the two happened, how many chunks and how many rows: "instant because it was cached" and "instant because there was nothing there" look identical from outside.

### Raw rows never leave Python, at any zoom

What crosses the bridge is a bounded number of buckets, each carrying t0, t1, a count, a min, a max and a mean. The bucket COUNT is tied to pixel width and capped at 2,000 — a bucket narrower than a pixel is a number nobody can see, and asking for one would be asking for the rows this design exists to avoid sending.

A MIN/MAX ENVELOPE, not a decimation. Picking every Nth row, or plotting the mean alone, makes a one-sample excursion vanish at exactly the zoom level where somebody is looking for it — and finding the moment a channel did something it should not have is the whole point of the view.

### The per-chunk buckets are new, and they were free

timeseries.py's channel pass was already reading each channel beside its timestamps, so it now also accumulates BUCKETS_PER_CHUNK row-equal buckets per channel — six parallel arrays rather than a list of dicts, because this lands in the cache as JSON where repeated key names would be most of the bytes. Folding two adjacent buckets is EXACT for all four figures (min of mins, max of maxes, counts and sums added), so the cached answer is not an approximation of the read: the only thing coarser about it is where the bucket edges fall.

They live in the CACHE and deliberately not in the merged payload. A merged unit already travels in chart_data to the Findings tab, the report and the frontend; four hundred buckets times six numbers times a channel would multiply that for a viewer most datasets never open. The timeline reads them back out of SQLite when it needs them, which is metadata, not data.

Adding them bumped `TIMESERIES_ARTIFACT_VERSION`, which is part of the time-series section's own signature — NOT profile_cache.SCHEMA_VERSION. That constant invalidates every row of every section wholesale, and throwing away a folder's null counts, histograms and categorical shortlists because a channel gained a bucket array would be charging for a change that did not touch them.

### The overlay is the findings, not a second opinion

Gaps, missing chunks, dropouts, flatlines and out-of-range extremes are turned into marks in Python, from the same merged artifact ts_checks.py reads. Two consequences worth stating:

- The out-of-range rule moved to `timeseries.out_of_range_extremes`, read by the check that REPORTS a spike and by the timeline that MARKS it. It had to leave ts_checks.py to be importable by a module findings.py does not know about — and one implementation is why a tooltip and the finding beside it cannot disagree about whether a spike counts. Both rules survive intact: exact against a documented range, HEURISTIC without one, in those words.
- A channel now records `min_at` and `max_at` — WHEN each extreme was reached. Two numbers, and the only thing that lets a finding about a value point at a place on a time axis; a count says a spike happened, not where to look. The marks are drawn as bands AND listed below the chart. A Plotly shape cannot be clicked, and "click to jump" has to land on something.

### A selected window is summarised, and then handed to Search

A horizontal box select is the lasso: a time window is one dimension, and offering a shape in the value axis would offer a selection the backend cannot answer. Its statistics are read off the buckets ON SCREEN rather than by scanning again, which is what makes a selection instant — and which decides what the numbers mean, so the payload says whether the selection landed on bucket edges (`exact`) and which stretch the figures actually cover.

"Search this window" does NOT reimplement export. A time window IS a search condition — `between` on the timestamp column — so the selection becomes a query in the feature that already counts matches and writes CSVs, with the query on screen and editable rather than as a second, invisible export path. Only Python can write that condition: the timeline works in seconds because that is what timeseries.py normalises to, and turning it back into what the COLUMN holds is the one step a frontend could get silently wrong. Which is also why `search._coerce` now parses an ISO timestamp for a temporal column — polars will not order a Datetime column against a string, so searching one was previously not possible at all.

### Consequences elsewhere

- `get_chart_data()` now passes the profile cache, like `start_chart_build()` already did. They are two forms of one computation, and a single opened file's per-chunk time-series artifact is now stored either way — which is what gives a lone CSV a timeline overview that costs nothing on re-open.
- Settings gains `timeline_max_channels` (6) and `timeline_max_rows` (2,000,000). Channels are capped because a strip per channel is a chart and forty is a wall, and because the read cost is per channel.
- The five shapes are in tests/test_timeline_api.py: a single CSV gets one timeline, a flat folder and a hive folder one per file, a nested tree one per file, and a full tree one per SERIES rather than one per chunk. A dataset with no time column reports `available: False` with the reason, which is a skip and not a failure.

## Not built yet

- The LLM custom-chart feature, and the LLM-config Settings UI it needs.

## Conventions

- Type hints throughout
- Keep data-loading logic separate from UI code, so it's testable headless (i.e. importable and callable without launching the pywebview window)
- Small, incremental commits — commit as soon as something works, before starting the next small piece
- Prefer polars over pandas for anything performance-sensitive

## Development phases

1. Phase 0: open file, show schema + table
2. Phase 1: null heatmap, numeric histograms, class balance, text-length distributions
3. Phase 2: embedding column detection, 2D projection (PCA then UMAP), scatter plot colored by label — this is the core differentiator
4. Phase 3: inline preview for image/audio/text columns
5. Phase 4: natural-language custom chart via LLM (declarative chart-spec first; sandboxed code-gen only as a fallback for non-standard requests)
6. Phase 5: packaging and installer
7. Phase 6: Settings — classification banner (done: top+bottom strips, configurable label/color) + audit log (done: local JSON-lines log of file-open actions) + LLM config (provider/key setup for Phase 4 — not yet built)
8. Phase 7: Reports — PDF export of a dataset's profiling results (done: section-availability checkboxes, header/footer + reproduced classification banner, embedded charts/schema table/media grid, audit logging of report generation)
