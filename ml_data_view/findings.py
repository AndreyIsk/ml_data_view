"""Instant dataset checks — the Findings tab's zero-cost half.

Every check in this module is a pure function over two dicts the app has
ALREADY computed: `dataset_info` (data_loader's schema + null counts + row
count) and `chart_data` (chart_data.build_chart_payload's output). Nothing
here takes a `pl.LazyFrame`, imports polars, or performs I/O.

That is a design constraint, not an accident of implementation. It makes
the whole instant catalogue structurally incapable of violating
PROJECT_BRIEF.md's lazy/streaming rule — there is no data access here to
get wrong — and it means every check can be tested against a hand-built
dict with no fixture file at all.

The cost is paid elsewhere: these checks cannot run until the chart build
finishes, which on a large folder is minutes. The Findings tab shows
"waiting" until then rather than a clean state, because a clean state
shown before the checks have run is a lie at exactly the moment a user is
most likely to trust it.

Deep checks — the ones that do have to read data — live in deep_checks.py.

Design spec: docs/superpowers/specs/2026-08-31-findings-tab-design.md
"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

# Share of rows missing, above which a column is worth a warning of its
# own. Half is the point past which a column describes the minority of
# rows; below it the schema table's null count already tells the story
# without a finding competing for attention. Strictly greater than.
HIGH_MISSINGNESS = 0.5
# Share of counted rows held by the top category, above which a
# categorical column is called imbalanced. 0.9 is the conventional line in
# ML practice for when accuracy stops meaning anything, since predicting
# the majority class alone beats it. Strictly greater than.
CLASS_IMBALANCE = 0.9
# Share of a column's DISTINCT values that occur exactly once, above which
# the column is reporting identity rather than describing rows.
ID_UNIQUE_SHARE = 0.9
# How many times bigger the largest group must be than the smallest before
# an imbalance is worth mentioning. Groups are rarely equal and a mild
# difference is ordinary; an order of magnitude is where a figure compared
# across them starts being a figure about the largest one.
GROUP_BALANCE_RATIO = 10.0

SEVERITIES = ("critical", "warning", "info")

# categorical_bar_data rolls everything past the top N into this synthetic
# bucket. It names no actual value, so it can never be "the dominant class".
_OTHER_BUCKET = "Other"


@dataclass
class Finding:
    """One thing worth telling the user about the loaded dataset.

    `columns` is the columns implicated — possibly several (correlated
    missingness), possibly none (the dataset itself is empty).

    `unit` is the UNIT it is about, when it is about one — the same key
    the time-series payload and the catalogue use ("123 / rec_a"), so a
    catalogue row can find its own findings by equality rather than by
    reading them out of the detail text. Empty for a finding about the
    dataset as a whole, which is not a defect of some particular
    recording and must not be counted against one.
    """

    check_id: str
    severity: str
    title: str
    detail: str
    columns: list[str] = field(default_factory=list)
    unit: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "check_id": self.check_id,
            "severity": self.severity,
            "title": self.title,
            "detail": self.detail,
            "columns": list(self.columns),
            "unit": self.unit,
        }


@dataclass
class CheckContext:
    """What every instant check gets. Bundled rather than passed as two
    loose dicts so a check's signature says what it may read — and so the
    derived lookups below are computed once for the whole scan instead of
    per check."""

    dataset_info: dict[str, Any]
    chart_data: dict[str, Any]

    @property
    def row_count(self) -> int:
        return int(self.dataset_info.get("row_count") or 0)

    @property
    def columns(self) -> list[dict[str, Any]]:
        return self.dataset_info.get("columns") or []

    @property
    def all_null_columns(self) -> set[str]:
        """Columns where every row is missing.

        Used for check PRECEDENCE: such a column is trivially "100%
        missing" and trivially "constant", and reporting one column under
        three findings buries the one that matters under two that add
        nothing. all_null_column reports it; the other two skip it.
        """
        total = self.row_count
        if total <= 0:
            return set()
        return {c["name"] for c in self.columns if int(c.get("null_count") or 0) >= total}

    def payload(self, key: str) -> list[dict[str, Any]]:
        """A chart-payload list, tolerant of a key that isn't there.

        get_chart_data()'s shape has grown over the phases; a check must
        degrade to "found nothing" on a payload predating a key it reads,
        not take the whole scan down.
        """
        value = self.chart_data.get(key)
        return value if isinstance(value, list) else []

    @property
    def capabilities(self) -> dict[str, Any]:
        """What the loaded dataset can support, computed once per load by
        api.py and carried here so a precondition stays a dict lookup
        rather than a reason to give these checks I/O."""
        value = self.dataset_info.get("capabilities")
        return value if isinstance(value, dict) else {}

    def capability(self, name: str) -> Any:
        return self.capabilities.get(name)


@dataclass
class CheckSpec:
    """One check, and what it needs before it means anything.

    `applies` is the whole point of this field existing. Otter opens a
    single CSV, a flat folder, a hive-partitioned tree or a fully mapped
    one, and a check that only makes sense for some of those must SKIP on
    the rest — not fail, and certainly not pass. A skipped check reports
    `requires` so the tab can say what was missing, because "this check did
    not run, and here is why" is information and a green tick is a lie.
    """

    id: str
    title: str
    kind: str  # "instant" here; deep_checks.py carries the "deep" ones
    fn: Callable[[CheckContext], list[Finding]]
    # None means "always applicable" — which is the right default: most
    # checks read null counts and row counts, and every input has those.
    applies: Callable[[CheckContext], bool] | None = None
    requires: str = ""


# --- the checks --------------------------------------------------------


def _check_empty_dataset(ctx: CheckContext) -> list[Finding]:
    if ctx.row_count > 0:
        return []
    return [
        Finding(
            check_id="empty_dataset",
            severity="critical",
            title="Dataset has no rows",
            detail="The loaded dataset contains 0 rows, so no profiling figure below describes anything.",
        )
    ]


def _check_all_null_column(ctx: CheckContext) -> list[Finding]:
    total = ctx.row_count
    if total <= 0:  # "0 nulls of 0 rows" is not an all-null column
        return []
    return [
        Finding(
            check_id="all_null_column",
            severity="critical",
            title=f"Column '{name}' is entirely missing",
            detail=(
                f"All {total:,} rows are missing a value for '{name}'. "
                "The column carries no information and cannot be trained on."
            ),
            columns=[name],
        )
        for name in sorted(ctx.all_null_columns, key=_column_order(ctx))
    ]


def _check_high_missingness(ctx: CheckContext) -> list[Finding]:
    total = ctx.row_count
    if total <= 0:
        return []
    skip = ctx.all_null_columns  # precedence — see CheckContext.all_null_columns
    findings = []
    for column in ctx.columns:
        name = column["name"]
        if name in skip:
            continue
        nulls = int(column.get("null_count") or 0)
        share = nulls / total
        if share <= HIGH_MISSINGNESS:
            continue
        findings.append(
            Finding(
                check_id="high_missingness",
                severity="warning",
                title=f"Column '{name}' is {share:.0%} missing",
                detail=(
                    f"{nulls:,} of {total:,} rows have no value for '{name}'. "
                    "Any statistic computed from it describes the minority of rows."
                ),
                columns=[name],
            )
        )
    return findings


def _constant_columns(ctx: CheckContext) -> set[str]:
    """Columns with exactly one distinct value, by either chart signal.

    Shared by _check_constant_column (which reports them) and
    _check_class_imbalance (which defers to it).
    """
    constant = {
        bar["column"]
        for bar in ctx.payload("categorical_bars")
        if len(bar.get("categories") or []) == 1
    }
    for histogram in ctx.payload("numeric_histograms"):
        edges = histogram.get("bin_edges") or []
        if len(edges) == 2 and edges[0] == edges[1]:
            constant.add(histogram["column"])
    return constant


def _check_constant_column(ctx: CheckContext) -> list[Finding]:
    """Zero-variance columns, read off charts already built.

    Both signals are decisions chart_data.py already made — a single
    category in categorical_bar_data, and the equal-edged single bin
    numeric_histogram_data emits when `maximum <= minimum`. Re-deriving
    either here would create a second source of truth that could disagree
    with the chart on screen.
    """
    skip = ctx.all_null_columns
    findings = []

    for bar in ctx.payload("categorical_bars"):
        name = bar.get("column")
        if name in skip or len(bar.get("categories") or []) != 1:
            continue
        findings.append(
            Finding(
                check_id="constant_column",
                severity="warning",
                title=f"Column '{name}' has a single value",
                detail=(
                    f"Every non-missing row of '{name}' holds '{bar['categories'][0]}'. "
                    "A column with no variance carries no signal for training."
                ),
                columns=[name],
            )
        )

    for histogram in ctx.payload("numeric_histograms"):
        name = histogram.get("column")
        edges = histogram.get("bin_edges") or []
        if name in skip or len(edges) != 2 or edges[0] != edges[1]:
            continue
        findings.append(
            Finding(
                check_id="constant_column",
                severity="warning",
                title=f"Column '{name}' has a single value",
                detail=(
                    f"Every non-missing row of '{name}' holds {edges[0]}. "
                    "A column with no variance carries no signal for training."
                ),
                columns=[name],
            )
        )

    return findings


def _check_class_imbalance(ctx: CheckContext) -> list[Finding]:
    """Categorical columns where the top class swamps the rest.

    Skips single-value and all-null columns: a constant column is trivially
    "100% one class", so reporting it here as well says the same thing
    twice and counts it twice in the severity summary. Same precedence
    reasoning as CheckContext.all_null_columns.
    """
    skip = ctx.all_null_columns | _constant_columns(ctx)
    findings = []
    for bar in ctx.payload("categorical_bars"):
        categories = bar.get("categories") or []
        counts = bar.get("counts") or []
        if bar.get("column") in skip or not categories or not counts:
            continue
        # The denominator includes "Other" — measuring the top category
        # against only the named ones would inflate its share.
        total = sum(counts)
        if total <= 0:
            continue
        top_index = max(range(len(counts)), key=lambda i: counts[i])
        if categories[top_index] == _OTHER_BUCKET:
            continue
        share = counts[top_index] / total
        if share <= CLASS_IMBALANCE:
            continue
        findings.append(
            Finding(
                check_id="class_imbalance",
                severity="warning",
                title=f"Column '{bar['column']}' is dominated by one class",
                detail=(
                    f"'{categories[top_index]}' holds {share:.0%} of the "
                    f"{total:,} counted rows. A model predicting it for every row "
                    "would score that accuracy while learning nothing."
                ),
                columns=[bar["column"]],
            )
        )
    return findings


def _check_broken_image_refs(ctx: CheckContext) -> list[Finding]:
    """Previewed image rows with no usable thumbnail.

    Deliberately vague about WHY, because the data cannot say:
    image_previews_data records only {row_index, thumbnail}, and a null
    thumbnail means the value was null, the file was missing, or the bytes
    failed to decode. Separating those is the deep unresolvable_images
    check's job — it re-reads the values and can tell.
    """
    findings = []
    for preview in ctx.payload("image_previews"):
        rows = preview.get("rows") or []
        if not rows:
            continue
        broken = sum(1 for row in rows if not row.get("thumbnail"))
        if not broken:
            continue
        findings.append(
            Finding(
                check_id="broken_image_refs",
                severity="warning",
                title=f"Column '{preview['column']}' has images that did not load",
                detail=(
                    f"{broken:,} of the first {len(rows):,} previewed rows have no usable "
                    "image — the value is missing, the file is missing, or the bytes are "
                    "undecodable. Run deep checks to tell these apart."
                ),
                columns=[preview["column"]],
            )
        )
    return findings


def _check_correlated_missingness(ctx: CheckContext) -> list[Finding]:
    return [
        Finding(
            check_id="correlated_missingness",
            severity="info",
            title=f"{len(group['columns'])} columns go missing together",
            detail=(
                f"{', '.join(group['columns'])} are missing on the same rows "
                f"({int(group.get('null_count') or 0):,} rows, "
                f"{float(group.get('overlap') or 0):.0%} overlap), which usually means "
                "one shared upstream cause rather than several independent ones."
            ),
            columns=list(group["columns"]),
        )
        for group in ctx.payload("correlated_missingness")
        if group.get("columns")
    ]


def _check_likely_id_column(ctx: CheckContext) -> list[Finding]:
    """Columns where almost every distinct value occurs exactly once.

    Such a column identifies rows rather than describing them — it is a
    track ID, not a feature — and a top-10 bar of it would name arbitrary
    values. The "1" bucket of value_frequency_histogram_data is exactly
    "values seen once".
    """
    findings = []
    for histogram in ctx.payload("value_frequency_histograms"):
        buckets = histogram.get("buckets") or []
        counts = histogram.get("counts") or []
        if "1" not in buckets or not counts:
            continue
        distinct = sum(counts)
        if distinct <= 0:
            continue
        once = counts[buckets.index("1")]
        if once / distinct <= ID_UNIQUE_SHARE:
            continue
        detail = (
            f"{once:,} of {distinct:,} distinct values in "
            f"'{histogram['column']}' occur exactly once, so the column identifies rows "
            "rather than grouping them."
        )
        if histogram.get("per_file"):
            # Per-file counting inflates the "1" bucket: a value present in
            # several files is counted once per file. Saying so beats
            # overstating uniqueness on a folder.
            detail += (
                " Counted per file, so a value appearing in several files is counted "
                "once for each — this overstates uniqueness on a folder."
            )
        findings.append(
            Finding(
                check_id="likely_id_column",
                severity="info",
                title=f"Column '{histogram['column']}' looks like an identifier",
                detail=detail,
                columns=[histogram["column"]],
            )
        )
    return findings


def _check_skewed_numeric(ctx: CheckContext) -> list[Finding]:
    """Columns chart_data._is_skewed already decided to bin in log space —
    reading its verdict rather than re-deriving one that could disagree."""
    return [
        Finding(
            check_id="skewed_numeric",
            severity="info",
            title=f"Column '{histogram['column']}' is heavily skewed",
            detail=(
                f"'{histogram['column']}' spans a range far wider than the bulk of its "
                "values, so it is charted on a log scale. Mean and standard deviation "
                "describe it poorly."
            ),
            columns=[histogram["column"]],
        )
        for histogram in ctx.payload("numeric_histograms")
        if histogram.get("x_scale") == "log"
    ]


def _column_order(ctx: CheckContext) -> Callable[[str], int]:
    """Sort key putting columns back in schema order — findings about a
    wide dataset should read down the table, not alphabetically."""
    order = {column["name"]: index for index, column in enumerate(ctx.columns)}
    return lambda name: order.get(name, len(order))


def _check_structure(context: CheckContext) -> list[Finding]:
    """Problems the directory structure itself reported (hierarchy.py).

    Carried in `dataset_info` rather than recomputed, which keeps this
    check inside the instant catalogue's rule: a pure function over two
    dicts the app already has, no I/O and no polars. The work was done at
    discovery time from parquet footers — a chunk whose schema deviates
    from its series, a series whose filenames do not run in timestamp
    order — and this is where the user is told about it.

    Skipped entirely when no structure was detected, which is the common
    case: a single file and a flat folder have no directory structure to
    have problems with.
    """
    return [
        Finding(
            check_id=finding.get("check_id", "structure"),
            severity=finding.get("severity", "warning"),
            title=finding.get("title", "Structure problem"),
            detail=finding.get("detail", ""),
            columns=list(finding.get("columns") or []),
        )
        for finding in (context.dataset_info.get("structure_findings") or [])
    ]


def _check_timestamp_integrity(context: CheckContext) -> list[Finding]:
    """Rows with no usable timestamp, in the column the dataset is ordered by.

    A TIME-SERIES check: its precondition is a detected timestamp column,
    and nothing else. It runs on a single CSV that happens to have one just
    as readily as on a mapped tree, because "does this time column have
    holes in it" is the same question either way — and a hole in a time
    column is a different problem from a hole in any other, since
    everything ordered by it is quietly wrong rather than merely missing.
    """
    column = context.capability("timestamp_column")
    entry = next((c for c in context.columns if c.get("name") == column), None)
    missing = int((entry or {}).get("null_count") or 0)
    if not missing:
        return []
    share = missing / context.row_count if context.row_count else 0
    return [
        Finding(
            check_id="timestamp_integrity",
            severity="critical" if share >= 0.01 else "warning",
            title="Timestamps are missing",
            detail=(
                f"{missing:,} of {context.row_count:,} rows ({share:.1%}) have no value in "
                f"{column!r}, the column this dataset is ordered by. Anything derived from "
                "time — ordering, gaps, rates — describes only the rows that have one."
            ),
            columns=[column],
        )
    ]


def _check_group_balance(context: CheckContext) -> list[Finding]:
    """Wildly uneven row counts across the grouping column's values.

    A GROUP-COMPARISON check: its precondition is a grouping column, which
    may come from a directory role, a hive key, or any column the user
    nominated. Without one there are no groups to compare and this is
    skipped — a dataset with nothing to group by is not an unbalanced one.
    """
    column = context.capability("grouping_column")
    bar = next(
        (b for b in context.payload("categorical_bars") if b.get("column") == column), None
    )
    if bar is None:
        return []
    counts = [
        count
        for value, count in zip(bar.get("categories") or [], bar.get("counts") or [])
        if value != "Other"
    ]
    if len(counts) < 2 or not sum(counts):
        return []
    largest, smallest = max(counts), min(counts)
    if smallest <= 0 or largest / smallest < GROUP_BALANCE_RATIO:
        return []
    return [
        Finding(
            check_id="group_balance",
            severity="info",
            title="Groups differ greatly in size",
            detail=(
                f"Grouped by {column!r}, the largest group has {largest:,} rows and the "
                f"smallest {smallest:,} — a {largest / smallest:.0f}x difference. Any figure "
                "compared across these groups is dominated by the largest."
            ),
            columns=[column],
        )
    ]


INSTANT_CHECKS: list[CheckSpec] = [
    CheckSpec(
        "structure",
        "Directory structure",
        "instant",
        _check_structure,
        applies=lambda ctx: bool(ctx.capability("has_structure")),
        requires="a detected directory structure",
    ),
    CheckSpec(
        "timestamp_integrity",
        "Timestamp integrity",
        "instant",
        _check_timestamp_integrity,
        applies=lambda ctx: bool(ctx.capability("timestamp_column")),
        requires="a timestamp column",
    ),
    CheckSpec(
        "group_balance",
        "Group sizes",
        "instant",
        _check_group_balance,
        applies=lambda ctx: bool(ctx.capability("grouping_column")),
        requires="a grouping column",
    ),
    CheckSpec("empty_dataset", "Dataset has rows", "instant", _check_empty_dataset),
    CheckSpec("all_null_column", "Entirely-missing columns", "instant", _check_all_null_column),
    CheckSpec("high_missingness", "Mostly-missing columns", "instant", _check_high_missingness),
    CheckSpec("constant_column", "Single-value columns", "instant", _check_constant_column),
    CheckSpec("class_imbalance", "Class balance", "instant", _check_class_imbalance),
    CheckSpec("broken_image_refs", "Image references", "instant", _check_broken_image_refs),
    CheckSpec(
        "correlated_missingness", "Correlated missingness", "instant", _check_correlated_missingness
    ),
    CheckSpec("likely_id_column", "Identifier-like columns", "instant", _check_likely_id_column),
    CheckSpec("skewed_numeric", "Numeric skew", "instant", _check_skewed_numeric),
]


def run_instant_checks(
    dataset_info: dict[str, Any], chart_data: dict[str, Any]
) -> list[dict[str, Any]]:
    """Run every instant check, in catalogue order.

    Returns one envelope per check — `{id, title, kind, status, findings,
    error}` — rather than a flat finding list, because the tab renders a
    row per CHECK: a check that found nothing must still be visibly
    accounted for, and one that failed must be visibly distinguishable
    from one that passed.

    A check that raises is caught and marked "error". A check that could
    not run is not a check that passed, and one malformed payload key must
    not take the whole scan down with it.
    """
    ctx = CheckContext(dataset_info=dataset_info or {}, chart_data=chart_data or {})
    results = []
    for spec in INSTANT_CHECKS:
        try:
            if spec.applies is not None and not spec.applies(ctx):
                # NOT a pass and NOT a failure. A check whose precondition
                # is absent had nothing to look at, and saying so is the
                # only honest answer — a green tick here would claim the
                # dataset had been examined for something it cannot have.
                results.append(
                    {
                        "id": spec.id,
                        "title": spec.title,
                        "kind": spec.kind,
                        "status": "skipped",
                        "findings": [],
                        "error": None,
                        "reason": f"needs {spec.requires}" if spec.requires else "not applicable",
                    }
                )
                continue
            findings = spec.fn(ctx)
            results.append(
                {
                    "id": spec.id,
                    "title": spec.title,
                    "kind": spec.kind,
                    "status": "done",
                    "findings": [finding.to_dict() for finding in findings],
                    "error": None,
                }
            )
        except Exception as exc:  # one bad check must not end the scan
            results.append(
                {
                    "id": spec.id,
                    "title": spec.title,
                    "kind": spec.kind,
                    "status": "error",
                    "findings": [],
                    "error": str(exc),
                }
            )
    return results


# Families of checks registered here rather than defined here: each has
# its own preconditions and its own module — time-series health in
# ts_checks.py, group comparison in fleet_checks.py — and this is the
# registry they join. Imported at the FOOT of this file because those
# modules need CheckSpec, CheckContext and Finding, which are defined
# above; the dependency runs one way.
def _register(specs: list[CheckSpec]) -> None:
    known = {spec.id for spec in INSTANT_CHECKS}
    for spec in specs:
        if spec.id not in known:
            INSTANT_CHECKS.append(spec)


def _register_external_checks() -> None:
    from ml_data_view.fleet_checks import FLEET_CHECKS
    from ml_data_view.ts_checks import TIMESERIES_CHECKS

    _register(TIMESERIES_CHECKS)
    _register(FLEET_CHECKS)


_register_external_checks()


def sorted_findings(results: list[dict[str, Any]]) -> list[Finding]:
    """Every finding across `results`, most severe first, catalogue order
    within a severity."""
    findings = [
        Finding(**finding) for result in results for finding in result.get("findings", [])
    ]
    return sorted(findings, key=lambda f: SEVERITIES.index(f.severity))


def severity_summary(findings: list[Finding]) -> dict[str, int]:
    """Count per severity, always with all three keys present — the UI
    renders a stable row of counts, including the zeroes."""
    summary = dict.fromkeys(SEVERITIES, 0)
    for finding in findings:
        if finding.severity in summary:
            summary[finding.severity] += 1
    return summary


# --- comparing one scan against the last -------------------------------
#
# A re-profile after new data arrives is only worth interrupting someone
# for if something CHANGED. "12 findings" after every file lands is noise;
# "1 new finding: high null rate in column X" is the thing they opened the
# app for.


def finding_key(finding: dict[str, Any]) -> tuple[str, str, tuple[str, ...]]:
    """What makes two findings the same finding.

    The check, the unit it is about, and the columns it implicates — not
    the detail text, which carries counts that move with every new file.
    "High null rate in column X" going from 41% to 43% is the same
    finding, and reporting it as a new one would train people to ignore
    the notification. The UNIT is part of the identity because the same
    gap appearing in a second recording IS news.
    """
    return (
        finding.get("check_id", ""),
        str(finding.get("unit") or ""),
        tuple(finding.get("columns") or []),
    )


def diff_findings(
    previous: list[dict[str, Any]] | None, current: list[dict[str, Any]]
) -> dict[str, Any] | None:
    """What appeared and what went away since the last scan.

    None when there IS no last scan — a first profile has nothing to
    compare against, and calling every finding on a freshly opened dataset
    "new" would make the word mean nothing.
    """
    if previous is None:
        return None
    before = {finding_key(f) for f in previous}
    after = {finding_key(f) for f in current}
    new = [f for f in current if finding_key(f) not in before]
    resolved = [f for f in previous if finding_key(f) not in after]
    return {
        "new": new,
        "resolved": resolved,
        "summary": describe_finding_changes(new, resolved),
    }


def describe_finding_changes(
    new: list[dict[str, Any]], resolved: list[dict[str, Any]]
) -> str:
    """One sentence for the re-profile notification.

    Built here rather than in JS for the same reason load_stats.py builds
    its line here: the omission rules are the content. Nothing new and
    nothing resolved is "No new findings." — which is worth saying, because
    it is the answer someone watching a pipeline most wants and is
    otherwise indistinguishable from the check not having run.
    """
    parts = []
    if new:
        lead = f"{len(new)} new finding{'s' if len(new) != 1 else ''}"
        parts.append(f"{lead}: {_name_finding(new[0])}" if len(new) == 1 else lead)
    if resolved:
        parts.append(f"{len(resolved)} resolved")
    if not parts:
        return "No new findings."
    return " · ".join(parts) + "."


def _name_finding(finding: dict[str, Any]) -> str:
    """"high null rate in column X" — the title, and the column it is
    about when there is exactly one. Several columns are named by count
    instead: a notification that lists eight column names is not a
    notification."""
    title = (finding.get("title") or "").rstrip(".")
    columns = finding.get("columns") or []
    if len(columns) == 1:
        return f"{title} in column {columns[0]}"
    if len(columns) > 1:
        return f"{title} across {len(columns)} columns"
    return title
