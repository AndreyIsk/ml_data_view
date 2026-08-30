# Row search — design

**Date:** 2026-08-29
**Status:** approved, not yet implemented

## Purpose

Let the user find specific rows in a loaded file or folder by building their
own query, rather than being limited to whatever the profiling charts happen
to show.

Two motivating cases, both from the user:

1. **Identify one entity by a composite key.** A "unique track" is defined by
   a unique combination of an ID column *and* a RecKey column. The user wants
   to name those columns themselves — uniqueness is not a fixed property of
   the schema, it is whatever the user says it is for this dataset.
2. **Filter by measurement.** Rows above some height, above some speed, or
   both.

The common requirement is that the *user* chooses which columns to search on
and how, at query time.

## Goals

- Arbitrary column choice, exact values and ranges, multiple conditions.
- Fast enough to feel interactive on a 250-file / 195.8M-row folder.
- Bounded memory, following the rule established across this codebase.
- Works identically for a single file and a folder.

## Non-goals (deliberate, revisit later)

- **Charts do not follow the filter.** Search produces a table of matching
  rows; the charts, heatmap and report continue to describe the whole
  dataset. The predicate is built as a reusable `pl.Expr` so that making
  charts follow it later is an addition, not a rewrite.
- **No sorting of results.** Sorting requires collecting every match before
  the first row can be shown, which destroys the early exit that makes this
  instant. Revisit only if the need proves real.
- **No OR / nested groups.** All conditions AND together. OR needs nesting in
  the UI and makes it markedly easier to build a query that does not mean
  what the user thought.

## Measured context

Measured on 50 real files (39.8M rows), under a 4GB cap, peak 1211 MiB
(essentially all import baseline):

| query | time |
|---|---|
| numeric `>` threshold, first 500 rows | 0.24s |
| numeric `>` threshold, exact total count | 0.02s |
| string `==`, first 500 rows | 0.05s |
| two-column AND (the unique-track shape) | 0.04s |
| per-file with early exit | 0.04s — 500 matches found in file 1 of 50 |

Two facts from this drive the design:

- **Parquet row-group statistics let polars skip data without decoding.** The
  0.02s exact count over 39.8M rows is min/max statistics eliminating almost
  every row group.
- **The Spark-written files in this dataset have no statistics on string
  columns.** (Same gap that made the null-count work necessary.) So numeric
  filters get row-group skipping; string filters do not. Fetching the first
  N matches is fast regardless, because early exit saves us. Computing an
  exact count for a rarely-matching string filter is a full scan of every
  file.

## Query model

`ml_data_view/search.py`:

```python
@dataclass
class Condition:
    column: str
    operator: str          # see below
    values: list[str]      # raw user input, coerced against the column dtype
```

Operators: `is`, `is not`, `>`, `>=`, `<`, `<=`, `between` (two values),
`is null`, `is not null` (no values), `contains` (string columns only).

`build_predicate(conditions, schema) -> pl.Expr` combines all conditions with
`&`. It coerces each raw string value against the column's dtype and raises
`SearchConditionError(condition_index, message)` on failure, so the UI can
mark the offending row rather than failing the whole search with a polars
traceback.

Coercion rules: integer columns accept only integer literals; float columns
accept int or float; string columns take the value as-is; boolean columns
accept `true`/`false` case-insensitively. `contains` on a non-string column
is rejected at build time.

## Execution

Three pure functions in the same module. All take an explicit `files` list
and scan **one file at a time** — never the concatenated frame. A single
concat query measured equally fast here, but concat queries have grown
unboundedly with file count in four separate places in this codebase, and
per-file additionally gives the count job natural progress reporting.

```python
search_rows(files, predicate, columns, limit) -> list[dict]
```
Walks files in order, applying `filter(predicate).head(remaining)`, and stops
as soon as `limit` rows are collected. `limit` defaults to
`SEARCH_ROW_LIMIT = 500`, matching the order of magnitude of the existing
`PREVIEW_ROW_LIMIT = 50` while giving enough rows to inspect a whole track.
`columns` restricts the projection —
this is a real speed win, not cosmetic: parquet is columnar, so selecting 5
of 40 columns reads 5 column chunks instead of 40.

```python
count_matches(files, predicate, on_progress=None, should_cancel=None) -> int
```
Per-file `filter(predicate).select(pl.len())`, summed. Reports progress per
file and checks cancellation at file boundaries — the same contract as
`data_loader._bounded_parquet_missing_counts`.

```python
export_matches(files, predicate, columns, out_path) -> Path
```
Streams matching rows to CSV via `sink_csv` per file (append after the
first), so memory stays bounded no matter how large the result set is.

## Jobs

`count_matches` and `export_matches` both run on worker threads, mirroring
`folder_load_job.py` and `chart_build_job.py`: `start()` returns at once, the
UI polls `progress()`, and the js_api bridge is never blocked.

Cancelling a count simply abandons it — the already-displayed rows are
unaffected, and the count line reverts to "not counted". This is safe in a
way that cancelling chart generation is not, because the rows and the count
are independent results rather than two halves of one payload.

`search_rows` is *not* a job. It is bounded by `limit` and measured in tens
of milliseconds; putting it on a thread would add latency and machinery for
no benefit.

## API surface

- `run_search(conditions, columns, limit)` → rows, or `{"error": ...}` with
  the offending condition index
- `start_match_count()` / `get_match_count_progress()` / `cancel_match_count()`
- `start_search_export(path)` / `get_search_export_progress()` /
  `cancel_search_export()` — cancelling leaves a partial CSV, and the
  progress result says so, so the user is never handed a truncated file
  that looks complete
- `save_search(name, conditions, columns)` / `list_saved_searches()` /
  `delete_saved_search(name)`

## Frontend

A new **Search** nav section, alongside Overview / Distributions / …

- Condition rows: column dropdown (from the loaded schema), operator
  dropdown (filtered to those valid for the column's dtype), value input(s).
  `+ Add condition`, and a remove button per row.
- Column multi-select for the results table, defaulting to all columns.
- Results table reusing the preview table's existing styling.
- A count line: `500 shown · counting…` → `500 shown of 12,481`.
- Export button, enabled once a search has run.
- Saved searches: a name field, a save button, and a list that re-runs a
  saved search on click.

## Persistence

Saved searches live in the existing settings file via `settings.py`, as a
list of `{name, conditions, columns}`. They are stored per name; saving an
existing name overwrites it.

## Error handling

- A value that cannot be coerced marks that condition row inline and blocks
  the search; other rows are untouched.
- A saved search naming a column absent from the current schema disables that
  row with an explanation rather than failing the whole search — a folder can
  legitimately be reloaded with a different schema group.
- Export to an unwritable path returns an error without disturbing the
  displayed results.

## Testing

- `build_predicate`: one test per operator; type coercion for int/float/
  string/bool columns; each failure mode's message and condition index;
  `contains` rejected on a non-string column.
- `search_rows`: per-file result matches a single-frame `filter().head()`
  exactly; `limit` respected; column projection returns only requested
  columns; early exit proven by asserting how many files were opened.
- `count_matches`: matches a single-frame count; progress fires once per
  file; cancellation stops early and the partial total covers exactly the
  files completed.
- `export_matches`: written CSV contains exactly the matching rows and the
  selected columns; works when the result set exceeds the display limit.
- Saved searches: round-trip through settings; unknown-column handling.
- API: `run_search` error shape carries the condition index.

## Follow-ups this design deliberately leaves open

- Making charts and reports follow the filter (the predicate is already
  reusable for this).
- Sorting results, if the early-exit cost proves acceptable.
- Populating value suggestions for low-cardinality columns from the top-K
  already computed for the categorical bar charts.
