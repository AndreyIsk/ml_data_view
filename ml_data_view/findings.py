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

SEVERITIES = ("critical", "warning", "info")

# categorical_bar_data rolls everything past the top N into this synthetic
# bucket. It names no actual value, so it can never be "the dominant class".
_OTHER_BUCKET = "Other"


@dataclass
class Finding:
    """One thing worth telling the user about the loaded dataset.

    `columns` is the columns implicated — possibly several (correlated
    missingness), possibly none (the dataset itself is empty).
    """

    check_id: str
    severity: str
    title: str
    detail: str
    columns: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "check_id": self.check_id,
            "severity": self.severity,
            "title": self.title,
            "detail": self.detail,
            "columns": list(self.columns),
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


@dataclass
class CheckSpec:
    id: str
    title: str
    kind: str  # "instant" here; deep_checks.py carries the "deep" ones
    fn: Callable[[CheckContext], list[Finding]]


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


INSTANT_CHECKS: list[CheckSpec] = [
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
