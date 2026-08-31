# Findings tab — design

**Date:** 2026-08-31
**Status:** approved, implementing

## Purpose

Answer "is anything wrong with this dataset?" without making the user read
every chart and infer it. The app already computes the signals — a column
that is 98% null, a label column where one class holds 97% of the rows, an
image column whose files no longer resolve. Today those facts are visible
only to someone who opens Distributions and Media and knows what to look
for. The Findings tab states them.

## Scope

Two kinds of check, split by what they cost:

1. **Instant checks** — pure functions over data the app has already
   computed. Zero new reads.
2. **Deep checks** — three checks that require touching data, run on
   demand behind an explicit button, as a cancellable job.

**Explicitly out of scope: fixing anything.** Findings reports; it does not
offer to drop columns, fill nulls, or deduplicate. Editing a dataset is a
different product from profiling one, and a profiler that silently rewrites
its input is worse than one that only reads.

**Also out of scope: user-configurable thresholds.** Every threshold below
is a constant with its reasoning recorded next to it. Making them
configurable before anyone has asked is speculation, and a threshold the
user can move is a threshold no finding can be trusted against.

## Instant checks cost nothing because the work is already done

By the time the Findings tab can show anything, two dicts exist:

- `Api._dataset_info` — `row_count`, and per column `{name, dtype,
  null_count}`.
- `Api._chart_data` — `build_chart_payload()`'s output:
  `numeric_histograms` (with `bin_edges`, `counts`, `x_scale`),
  `categorical_bars` (`categories`, `counts`), `value_frequency_histograms`,
  `text_length_histograms`, `correlated_missingness`, `null_heatmap`,
  `embedding_columns`, `image_columns`, `image_previews`.

Every instant check reads only those. None takes a `pl.LazyFrame`, imports
polars, or performs I/O. That is the architectural point, not an
implementation detail: it makes the entire instant catalogue provably
incapable of violating PROJECT_BRIEF.md's lazy/streaming rule, and testable
against hand-built dicts with no fixture files at all.

**Consequence, accepted deliberately:** the tab cannot populate until the
chart build finishes, which on a 250-file folder is minutes. Until then it
shows "Waiting for chart data…" — not a clean state, which would be a lie
by omission at exactly the moment the user is most likely to trust it.

The alternative considered and rejected was a thin catalogue reading only
schema and null counts, which would populate the instant a load finishes.
Rejected because it drops class imbalance, constant-column detection and
broken image references — the checks that make the tab worth opening.

## The instant catalogue

| id | Check | Reads | Fires when | Severity |
|---|---|---|---|---|
| `empty_dataset` | Dataset has no rows | `row_count` | `row_count == 0` | critical |
| `all_null_column` | Column is entirely missing | `null_count`, `row_count` | `null_count == row_count`, `row_count > 0` | critical |
| `high_missingness` | Column mostly missing | `null_count`, `row_count` | `null_count / row_count > 0.5`, not all-null | warning |
| `constant_column` | Column has one value | `categorical_bars`, `numeric_histograms` | one category, or `bin_edges` min == max | warning |
| `class_imbalance` | One class dominates | `categorical_bars` | top category > 90% of counted rows | warning |
| `broken_image_refs` | Image values do not resolve | `image_previews` | any previewed row has `thumbnail == null` | warning |
| `correlated_missingness` | Columns missing together | `correlated_missingness` | any group reported | info |
| `likely_id_column` | Column looks like an identifier | `value_frequency_histograms` | the `"1"` bucket holds > 90% of distinct values | info |
| `skewed_numeric` | Column is heavily skewed | `numeric_histograms` | `x_scale == "log"` | info |

Thresholds and their reasoning:

- **`HIGH_MISSINGNESS = 0.5`.** Half is the point past which a column
  describes the minority of rows. Below it, the schema table's own null
  count already tells the story without a finding competing for attention.
- **`CLASS_IMBALANCE = 0.9`.** The conventional line in ML practice for
  when accuracy stops being a meaningful metric, because predicting the
  majority class alone beats it.
- Both are module constants, documented in place.

`constant_column` and `skewed_numeric` reuse decisions already made by
`chart_data.py` rather than recomputing them: a one-bin histogram is
exactly what `numeric_histogram_data()` emits when `maximum <= minimum`,
and `x_scale == "log"` is `_is_skewed()`'s verdict. Re-deriving either
here would create two sources of truth that could disagree on screen.

`class_imbalance` reads `categorical_bars`, whose per-file top-K merge is
documented as approximate — a value below K in every file can be missed.
That does not affect this check: it asks whether the TOP category
dominates, and the top category is the one the merge is most certain of.
The denominator is `sum(counts)`, which includes the synthetic `"Other"`
bucket when `categorical_bar_data` added one; `"Other"` itself is never
reported as the dominant class, since it names no actual value.

`likely_id_column` reads the `"1"` bucket of `value_frequency_histograms`
— values occurring exactly once. When nearly every distinct value is
unique, the column identifies rows rather than describing them, and its
top-10 bar chart would name arbitrary IDs. Caveat carried into the finding
text: `value_frequency_histogram_data` returns `per_file`, and when true
it counted each file separately, which shifts the distribution toward the
`"1"` bucket. The finding says so rather than overstating uniqueness on a
folder.

**Check precedence.** A column reported by `all_null_column` is skipped by
`high_missingness` and `constant_column` — an all-null column is trivially
"100% missing" and trivially "constant", and reporting one column three
times buries the finding that matters under two that add nothing.

## Deep checks

Run only when the user clicks "Run deep checks". Each follows the
never-query-the-concat rule — per file, released between files.

| id | Check | Method | Exactness |
|---|---|---|---|
| `duplicate_rows` | Identical rows | hash rows over a bounded sample | sample size stated |
| `sentinel_values` | Sentinels posing as data | streaming aggregation per file | **exact** |
| `unresolvable_images` | Image refs that fail to load | resolve a wider bounded sample | sample size stated |

**Duplicate rows are sampled, not exact, and the finding says so.** An
exact count over the real 195.8M-row folder means holding on the order of
195M row hashes — unbounded, and precisely the pattern that exhausted a 2GB
cap in `categorical_bar_data` (2.2M distinct values across 26 columns).
`DUPLICATE_SAMPLE_ROWS = 100_000` is collected as a bounded `.head()`,
hashed, and duplicates counted within it. The finding reads:

> 12 duplicate rows among the first 100,000 scanned. Not a dataset-wide
> count.

Two weaker properties follow and are documented rather than hidden: the
sample is the first N rows (`LazyFrame` has no streaming-safe random
sample — the same constraint `embedding_projection_data` records), and on a
folder those rows come from the first files only.

**Sentinel values are exact** because the question is bounded by
construction: count occurrences of `SENTINEL_VALUES = (-1, -999, -9999)` in
numeric columns via a streaming aggregation per file, summed. The counts
add exactly across files. These values are the common encodings for "no
data" that survive as real numbers into training and silently corrupt every
statistic computed from them — which is why the check is a warning rather
than info.

**What the instant image check can and cannot tell.**
`image_previews_data` records only `{row_index, thumbnail}`, and
`thumbnail` is `None` for three different reasons — the value was null,
the file was missing, or the bytes failed to decode. The instant check
cannot separate them, so it does not pretend to: it reports "N of the
first 50 previewed rows have no usable image (value missing, file
missing, or undecodable)". Distinguishing the causes is exactly what the
deep `unresolvable_images` check is for, since it re-reads the values and
knows which case each row is. No change to `image_data.py` is needed.

**Unresolvable images** re-runs the existing thumbnail resolution over
`IMAGE_CHECK_SAMPLE_ROWS = 500` rows rather than the 50 the Media preview
already covers, reusing `image_data.py`'s decode path unchanged. Bounded by
the sample; the finding states it.

## Severity

Three levels, findings sorted by them, with a summary count at the end.

| Severity | Meaning |
|---|---|
| `critical` | The dataset or column cannot be used as-is |
| `warning` | Likely to harm training if not addressed |
| `info` | Worth knowing; not necessarily a defect |

Reusing `compare.js`'s badge pattern, whose existing CSS comment states the
rule: severity is carried by an explicit text label, never by color alone.
The names are new — compare's `stable`/`moderate`/`significant` are
drift-specific and "stable" is meaningless for an all-null column.

## Components

### `ml_data_view/findings.py`

The catalogue and the instant checks. No polars import, no `LazyFrame`, no
I/O.

```python
@dataclass
class Finding:
    check_id: str
    severity: str          # "critical" | "warning" | "info"
    title: str
    detail: str
    columns: list[str]     # columns implicated, for the UI to link
    def to_dict(self) -> dict[str, Any]: ...

INSTANT_CHECKS: list[CheckSpec]        # ordered; (id, title, kind, fn)
def run_instant_checks(dataset_info, chart_data) -> list[dict[str, Any]]
def severity_summary(findings) -> dict[str, int]
```

Each check function is `(dataset_info, chart_data) -> list[Finding]` —
a list rather than a single finding, because one check legitimately reports
per column (six columns over 50% null is six findings, not one).

### `ml_data_view/deep_checks.py`

The three bounded checks, each `(lf, files, ...) -> list[Finding]`,
following the per-file pattern in `chart_data.py`.

### `ml_data_view/findings_job.py`

`DeepCheckJob(BackgroundJob)` — reuses `background_job.py`, as
`CompareJob`, `SearchCountJob` and `SearchExportJob` do. Per-check status
lives in the job's progress state so completed checks render while later
ones still run, the same way `CompareJob` carries finished column rows.
Cancelling keeps completed checks: half a findings list is half an answer,
unlike a half-built chart payload.

### API (`api.py`)

```python
list_checks()               -> {"checks": [{"id", "title", "kind"}]}
run_instant_checks()        -> {"checks": [...], "findings": [...],
                                "summary": {...}}  |  {"pending": "charts"}
start_deep_checks()         -> {"started": True}
get_deep_check_progress()   -> job snapshot with per-check status
cancel_deep_checks()        -> {"cancelled": True}
wait_for_deep_checks(t)     -> for headless callers and tests
```

`list_checks()` is separate so the tab can paint every check name as
pending before any work happens — what makes it read as a scan rather than
a list that pops into existence. `run_instant_checks()` returns
`{"pending": "charts"}` when `_chart_data` is absent.

### Frontend (`frontend/findings.js`, new "Findings" nav section)

Its own file, as `compare.js` and `search.js` are. On dataset load: render
every check from `list_checks()` as pending, call `run_instant_checks()`,
then reveal results one at a time.

**The stagger is a reveal animation, and the code says so.** Instant checks
take roughly 5ms in total because the payload is already computed; a
"running" state for them reports nothing real. `REVEAL_STAGGER_MS = 120`
(within the 150ms ceiling) applies to the reveal only — results are never
withheld beyond it, and the deep-check path uses genuine per-check status
from the job instead.

Empty state: "All N checks passed — no issues found", where N is the
number of checks actually run — 9 after the instant scan, 12 once deep
checks have also run. Hard-coding 9 would misreport the moment the deep
button is used. There is no existing
positive empty state in the app to match; this one is new, styled with the
existing note treatment rather than as a banner.

## Error handling

- No dataset loaded: the tab says so, as other sections do.
- Charts not built yet: "Waiting for chart data…", never a clean state.
- A single instant check raising must not take down the scan: each is
  wrapped, and a failed check renders as an explicit error row naming the
  check. A check that cannot run is not a check that passed.
- `DeepCheckJob` catches per `BackgroundJob`'s existing contract; a worker
  thread never dies silently.

## Testing

`tests/test_findings.py` — every instant check against hand-built
`dataset_info`/`chart_data` dicts: fires when it should, stays silent when
it should not, correct severity, correct implicated columns. Boundary cases
at each threshold (exactly 50% null, exactly 90% top class). Catalogue
ordering is stable. A check that raises produces an error row rather than
aborting the run.

`tests/test_deep_checks.py` — duplicate detection finds known planted
duplicates and reports the true scanned sample size; sentinel counts are
exact and sum correctly across multiple files; unresolvable images are
counted against a fixture with deliberately broken paths. Plus the
`collect()`-monitoring pattern from `tests/test_lazy_loading.py` asserting
no collect exceeds the bounded sample, run against a fixture large enough
for a full materialization to be unmistakable.

`tests/test_findings_job.py` — status sequence is pending → running → done
per check; completed checks are present in progress state before the job
finishes; cancelling keeps completed checks and marks the rest cancelled.

`tests/test_api.py` — `run_instant_checks()` before a chart build returns
the pending marker rather than a clean state.

Reveal timing is not tested.

## Follow-ups deliberately left open

- **Findings in the PDF report.** `report.py`'s `available_sections()`
  would need a Findings section. Deliberately not in this stage: the report
  pipeline has its own layout and section-availability rules, and bolting a
  tenth thing onto it while the check catalogue is still new invites
  reworking both.
- **Exact dataset-wide duplicate detection.** Would need an external-sort
  or disk-backed hash strategy rather than an in-memory set. Real, but a
  project of its own.
- **User-configurable thresholds.** See Scope.
