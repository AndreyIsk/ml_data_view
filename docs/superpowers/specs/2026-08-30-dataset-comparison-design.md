# Dataset comparison — design

**Date:** 2026-08-30
**Status:** approved, implementing

## Purpose

Compare two datasets — two files, or two folders — and flag what changed.
The motivating question is "did my new capture drift from the old one?",
which today can only be answered by loading each dataset in turn and
eyeballing charts side by side.

## Scope

Two kinds of difference, both chosen by the user:

1. **Schema differences** — columns only in A, only in B, and columns whose
   type changed. Read from footer metadata only, so this is effectively
   free and renders before any other work starts.
2. **Distribution differences** — row counts, null rates, and how each
   column's values are distributed, quantified with PSI.

**Explicitly out of scope: row-level differences.** Identifying which
individual rows were added, removed or changed requires a full scan of both
sides plus a keyed join, rather than summary statistics. Not built.

## Flagging: PSI

Population Stability Index is the standard drift metric in ML monitoring:

```
PSI = Σ (pA_i − pB_i) · ln(pA_i / pB_i)
```

over matching bins `i`, where `pA` and `pB` are proportions summing to 1.

Chosen over inventing a threshold on percent-change because it has
**conventional interpretation thresholds**, so severity is objective rather
than something this codebase made up:

| PSI | severity |
|---|---|
| < 0.10 | stable |
| 0.10 – 0.25 | moderate |
| > 0.25 | significant |

It also catches shape changes that summary statistics miss — a column whose
mean is unchanged while its spread doubled reads as identical under a
mean-delta rule and as significant under PSI.

**Empty bins.** `ln(0)` and division by zero are real cases: a bin populated
on one side and empty on the other is exactly the interesting case. Both
proportions are floored at `PSI_EPSILON = 1e-6` before the ratio. Flooring
rather than dropping the bin, because dropping it would hide the very
difference being measured. The distortion is negligible at this magnitude.

## Shared bins are mandatory

Two histograms are only comparable if both sides are binned with **identical
edges**. So a comparison computes a joint min/max across both datasets and
bins both against that.

The consequence, stated plainly: **the already-loaded side's cached
histograms cannot be reused**, because they were binned against its own
range. Null rates and cardinality can be reused; histograms cannot. This is
why comparison computes its own lean profile rather than diffing two
`build_chart_payload` results.

For categorical columns the equivalent of shared bins is a shared category
set: the union of each side's top `COMPARE_TOP_K = 20` categories, plus an
"Other" bucket absorbing the remainder on each side. Categories present in
one side and not the other are additionally listed outright, which is
usually the most actionable output.

## Sampling

Default: `COMPARE_SAMPLE_FILES = 20` files per side, or all of them if a
side has fewer.

Sampled files are **evenly spaced across the folder, never the first N**.
This is not a detail — the folders this is built for are day-partitioned, so
the first 20 files can all be from early in the day, and comparing head-of-A
against head-of-B would report drift that is really time-of-day. Even
spacing takes indices `round(i * (len-1) / (n-1))`.

The UI states which happened ("20 of 250 files per side") and offers
"Compare all files" for certainty. A single-file side is a one-element list
and is never sampled.

## Components

### `ml_data_view/compare.py` (pure logic, no threads)

- `compare_schemas(schema_a, schema_b) -> dict` — `only_in_a`, `only_in_b`,
  `type_changed`, `shared`. Metadata only.
- `evenly_spaced(files, n) -> list[Path]` — sampling described above.
- `joint_bin_edges(files_a, files_b, column, bins) -> list[float] | None`
  — min/max across both sides, then equal-width edges. Returns None when a
  column has no usable values on either side.
- `numeric_column_profile(files, column, edges) -> dict` — count, null
  count, histogram against the given edges, mean. Per-file, summed.
- `categorical_column_profile(files, column, top_k) -> dict` — count, null
  count, per-category counts for that side's top K.
- `psi(proportions_a, proportions_b) -> float`
- `severity(psi_value) -> "stable" | "moderate" | "significant"`
- `compare_column(...) -> dict` — one row of the results table: both sides'
  stats, PSI, severity, and for categoricals the new/vanished category
  lists.
- `compare_datasets(files_a, files_b, on_column=None, should_cancel=None)
  -> dict` — orchestrates the above over the shared columns.

### `ml_data_view/background_job.py` (targeted refactor)

`_BackgroundJob` currently lives inside `search_jobs.py`. This is the fifth
job class in the codebase, so it moves to its own module as `BackgroundJob`
and `search_jobs.py` imports it. `SearchCountJob` and `SearchExportJob`
already derive from it and follow the move for free; `FolderLoadJob` and
`ChartBuildJob` predate it and are left alone — converting them is unrelated
risk for no gain here.

### `ml_data_view/compare_jobs.py`

`CompareJob` — runs `compare_datasets` on a worker thread, reporting
progress per column and carrying **completed column rows in its progress
state**, so the results table fills in row by row rather than staying blank
until the end. Cancellable at column boundaries; completed rows survive
cancellation and the result is labelled partial.

### API

- `compare_picker_dialog(kind)` — native file or folder picker for the
  other side
- `start_comparison(other_path, sample_files=COMPARE_SAMPLE_FILES)` —
  schema diff returns immediately in the start result; distribution work
  runs in the job
- `get_comparison_progress()` / `cancel_comparison()` /
  `wait_for_comparison(timeout)`

The currently loaded dataset is side A; the picked path is side B.

### Frontend (`frontend/compare.js`, new "Compare" nav section)

- "Compare against…" with file and folder buttons
- Schema diff block, rendered as soon as the comparison starts
- Column table: name, A stats, B stats, PSI, severity badge — sorted by
  severity descending so the biggest changes are at the top
- Sampling note and a "Compare all files" button
- Progress bar with a Stop button, reusing the existing pattern

## Error handling

- A picked path that fails to scan returns an error without disturbing the
  loaded dataset.
- Comparing a dataset against itself is allowed and should report PSI 0
  everywhere — it is a useful sanity check, not an error.
- Columns present on only one side get no PSI; they appear in the schema
  diff only, so the table never shows a meaningless number.
- A column whose type changed between sides is reported in the schema diff
  and skipped for distribution comparison, since the distributions are not
  comparable.

## Testing

- `psi`: hand-computed value for a known pair; identical inputs give 0; a
  bin empty on one side yields a large finite number, never `inf` or `nan`.
- `severity`: boundary values at 0.10 and 0.25.
- `evenly_spaced`: spans the full list, includes first and last, returns all
  files when fewer than requested, and — pinned explicitly — does NOT
  return the first N.
- `joint_bin_edges`: spans both sides' ranges, including when one side's
  range sits entirely inside the other's.
- `compare_datasets`: identical folders give stable everywhere; a folder
  with a deliberately shifted numeric column crosses the significant
  threshold; a folder with a new category lists it; row-count and null-rate
  deltas are correct.
- Schema diff: added, removed, and type-changed columns.
- `CompareJob`: progress carries completed rows; cancellation keeps them and
  marks the result partial; an error is recorded rather than raised in the
  thread.
- API: start returns the schema diff immediately; comparison works when
  either side is a single file.

## Follow-ups deliberately left open

- Row-level diff (needs key columns and a keyed join).
- Charting a compared column's two distributions overlaid — the profiles
  computed here already contain everything such a chart would need.
