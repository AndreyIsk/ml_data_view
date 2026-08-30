"""Compare two datasets — two files, or two folders — and flag what changed.

Answers "did my new capture drift from the old one?" without loading each
side in turn and eyeballing charts. Two kinds of difference:

  Schema — columns only on one side, and columns whose type changed. Read
  from footer metadata, so it is effectively free and can be shown before
  any other work starts.

  Distribution — row counts, null rates, and how values are distributed,
  quantified with PSI (see below).

Row-level differences (which individual rows were added or changed) are
deliberately NOT here: that needs a full scan of both sides plus a keyed
join, rather than summary statistics.

Everything below scans one file at a time, per the rule established across
this codebase (see PROJECT_BRIEF.md) — a query over a concatenated
multi-file LazyFrame grows with file count.
"""
from __future__ import annotations

import math
from collections.abc import Callable
from pathlib import Path
from typing import Any

import polars as pl

from ml_data_view.chart_data import NUMERIC_DTYPE_NAMES, STRING_DTYPE_NAMES
from ml_data_view.data_loader import _dispatch_scanner

COMPARE_BINS = 20
COMPARE_TOP_K = 20
COMPARE_SAMPLE_FILES = 20

# Conventional PSI interpretation thresholds from ML model monitoring.
# Used rather than a percent-change rule this codebase invented, so
# "significant" means something a reader can look up and argue with.
MODERATE_PSI = 0.10
SIGNIFICANT_PSI = 0.25
# Proportions are floored here before the log/ratio. A bin populated on one
# side and empty on the other is precisely the interesting case, so it must
# produce a large finite number rather than inf — and dropping such bins
# would hide the very difference being measured.
PSI_EPSILON = 1e-6


def _scan(path: Path) -> pl.LazyFrame:
    path = Path(path)
    return _dispatch_scanner(path, path.suffix.lower())


def psi(proportions_a: list[float], proportions_b: list[float]) -> float:
    """Population Stability Index over matching bins.

    PSI = sum (pA - pB) * ln(pA / pB). Symmetric, zero for identical
    inputs, and larger the further apart the two distributions are.
    """
    if len(proportions_a) != len(proportions_b):
        raise ValueError("PSI needs the same number of bins on both sides.")
    total = 0.0
    for a, b in zip(proportions_a, proportions_b):
        a = max(a, PSI_EPSILON)
        b = max(b, PSI_EPSILON)
        total += (a - b) * math.log(a / b)
    return total


def severity(value: float) -> str:
    if value >= SIGNIFICANT_PSI:
        return "significant"
    if value >= MODERATE_PSI:
        return "moderate"
    return "stable"


def evenly_spaced(items: list[Any], count: int) -> list[Any]:
    """`count` items spread across the whole list, NOT the first `count`.

    Not a detail: the folders this is built for are day-partitioned, so the
    first N files can all be from early in the day. Comparing head-of-A
    against head-of-B would report drift that is really time-of-day.
    """
    if count <= 0 or not items:
        return []
    if len(items) <= count:
        return list(items)
    if count == 1:
        return [items[0]]
    step = (len(items) - 1) / (count - 1)
    return [items[round(i * step)] for i in range(count)]


def compare_schemas(schema_a: pl.Schema, schema_b: pl.Schema) -> dict[str, Any]:
    """Column-level differences. Metadata only — no row data is read."""
    names_a, names_b = list(schema_a), list(schema_b)
    common = [name for name in names_a if name in schema_b]
    type_changed = [
        {"column": name, "dtype_a": str(schema_a[name]), "dtype_b": str(schema_b[name])}
        for name in common
        if str(schema_a[name]) != str(schema_b[name])
    ]
    changed_names = {entry["column"] for entry in type_changed}
    return {
        "only_in_a": [name for name in names_a if name not in schema_b],
        "only_in_b": [name for name in names_b if name not in schema_a],
        "type_changed": type_changed,
        # "shared" means comparable: same name AND same type. A retyped
        # column's distributions are not comparable, so it is excluded here
        # and reported in type_changed instead.
        "shared": [name for name in common if name not in changed_names],
    }


def _numeric_range(files: list[Path], column: str) -> tuple[float, float] | None:
    clean = pl.col(column).drop_nulls()
    minimum, maximum = math.inf, -math.inf
    for file in files:
        result = (
            _scan(file)
            .select(clean.min().alias("mn"), clean.max().alias("mx"), clean.count().alias("n"))
            .collect(engine="streaming")
        )
        if int(result[0, "n"]) == 0:
            continue
        minimum = min(minimum, float(result[0, "mn"]))
        maximum = max(maximum, float(result[0, "mx"]))
    if minimum == math.inf:
        return None
    return minimum, maximum


def joint_bin_edges(
    files_a: list[Path], files_b: list[Path], column: str, bins: int = COMPARE_BINS
) -> list[float] | None:
    """Equal-width edges spanning BOTH sides' ranges.

    Shared edges are what make two histograms comparable at all. This is
    also why an already-loaded dataset's cached histograms cannot be reused
    for a comparison — they were binned against their own range alone.
    """
    range_a = _numeric_range(files_a, column)
    range_b = _numeric_range(files_b, column)
    ranges = [r for r in (range_a, range_b) if r is not None]
    if not ranges:
        return None
    minimum = min(r[0] for r in ranges)
    maximum = max(r[1] for r in ranges)
    if maximum <= minimum:
        # Every value identical on both sides; one bin is the honest answer.
        return [minimum, minimum + 1.0]
    return [minimum + (maximum - minimum) * i / bins for i in range(bins + 1)]


def _numeric_profile(files: list[Path], column: str, edges: list[float]) -> dict[str, Any]:
    clean = pl.col(column).drop_nulls()
    counts = [0] * (len(edges) - 1)
    rows = 0
    nulls = 0
    weighted_sum = 0.0
    for file in files:
        result = (
            _scan(file)
            .select(
                pl.len().alias("rows"),
                pl.col(column).null_count().alias("nulls"),
                clean.sum().alias("total"),
                clean.hist(bins=edges, include_breakpoint=False).alias("hist"),
            )
            .collect(engine="streaming")
        )
        rows += int(result[0, "rows"])
        nulls += int(result[0, "nulls"])
        weighted_sum += float(result[0, "total"] or 0.0)
        for index, value in enumerate(result["hist"].to_list()[: len(counts)]):
            counts[index] += int(value)
    non_null = rows - nulls
    return {
        "rows": rows,
        "nulls": nulls,
        "counts": counts,
        "mean": (weighted_sum / non_null) if non_null else None,
    }


def _categorical_profile(files: list[Path], column: str, top_k: int) -> dict[str, Any]:
    """Per-category counts for this side's top K, plus totals.

    Grouped per file and merged, so no single query holds every distinct
    value of a high-cardinality column.
    """
    merged: dict[str, int] = {}
    rows = 0
    nulls = 0
    for file in files:
        scan = _scan(file)
        totals = (
            scan.select(pl.len().alias("rows"), pl.col(column).null_count().alias("nulls"))
            .collect(engine="streaming")
        )
        rows += int(totals[0, "rows"])
        nulls += int(totals[0, "nulls"])
        top = (
            scan.select(column)
            .drop_nulls()
            .group_by(column)
            .len()
            .sort("len", descending=True)
            .head(top_k)
            .collect(engine="streaming")
        )
        for value, count in zip(top[column].to_list(), top["len"].to_list()):
            key = str(value)
            merged[key] = merged.get(key, 0) + int(count)
    return {"rows": rows, "nulls": nulls, "counts": merged}


def _proportions(counts: list[int]) -> list[float]:
    total = sum(counts)
    return [c / total for c in counts] if total else [0.0] * len(counts)


def _compare_numeric(
    files_a: list[Path], files_b: list[Path], column: str, bins: int
) -> dict[str, Any] | None:
    edges = joint_bin_edges(files_a, files_b, column, bins)
    if edges is None:
        return None
    profile_a = _numeric_profile(files_a, column, edges)
    profile_b = _numeric_profile(files_b, column, edges)
    value = psi(_proportions(profile_a["counts"]), _proportions(profile_b["counts"]))
    return {
        "column": column,
        "kind": "numeric",
        "psi": round(value, 4),
        "severity": severity(value),
        "mean_a": profile_a["mean"],
        "mean_b": profile_b["mean"],
        "null_rate_a": profile_a["nulls"] / profile_a["rows"] if profile_a["rows"] else 0.0,
        "null_rate_b": profile_b["nulls"] / profile_b["rows"] if profile_b["rows"] else 0.0,
        "new_categories": [],
        "missing_categories": [],
    }


def _compare_categorical(
    files_a: list[Path], files_b: list[Path], column: str, top_k: int
) -> dict[str, Any]:
    profile_a = _categorical_profile(files_a, column, top_k)
    profile_b = _categorical_profile(files_b, column, top_k)

    # PSI needs the same buckets on both sides, so compare over the union of
    # each side's top K. "Other" absorbs whatever each side has outside that
    # union, so the proportions still describe the whole column.
    shared = sorted(set(profile_a["counts"]) | set(profile_b["counts"]))
    counts_a = [profile_a["counts"].get(name, 0) for name in shared]
    counts_b = [profile_b["counts"].get(name, 0) for name in shared]
    non_null_a = profile_a["rows"] - profile_a["nulls"]
    non_null_b = profile_b["rows"] - profile_b["nulls"]
    counts_a.append(max(non_null_a - sum(counts_a), 0))
    counts_b.append(max(non_null_b - sum(counts_b), 0))

    value = psi(_proportions(counts_a), _proportions(counts_b))
    return {
        "column": column,
        "kind": "categorical",
        "psi": round(value, 4),
        "severity": severity(value),
        "mean_a": None,
        "mean_b": None,
        "null_rate_a": profile_a["nulls"] / profile_a["rows"] if profile_a["rows"] else 0.0,
        "null_rate_b": profile_b["nulls"] / profile_b["rows"] if profile_b["rows"] else 0.0,
        # The most actionable output for a categorical column: what appeared
        # and what vanished, named outright rather than folded into a score.
        "new_categories": sorted(set(profile_b["counts"]) - set(profile_a["counts"])),
        "missing_categories": sorted(set(profile_a["counts"]) - set(profile_b["counts"])),
    }


def _row_count(files: list[Path]) -> int:
    return sum(
        int(_scan(f).select(pl.len()).collect(engine="streaming")[0, 0]) for f in files
    )


def compare_datasets(
    files_a: list[Path],
    files_b: list[Path],
    bins: int = COMPARE_BINS,
    top_k: int = COMPARE_TOP_K,
    on_column: Callable[[int, str, dict[str, Any]], None] | None = None,
    should_cancel: Callable[[], bool] | None = None,
) -> dict[str, Any]:
    """Full comparison of two datasets.

    `on_column(index, name, row)` fires as each column finishes, so a UI can
    fill its table in progressively rather than waiting for everything.
    Cancellation is checked at column boundaries and KEEPS the columns
    already completed — a partial comparison is still useful, unlike a
    partial chart payload.
    """
    schema_a = _scan(files_a[0]).collect_schema()
    schema_b = _scan(files_b[0]).collect_schema()
    schema_diff = compare_schemas(schema_a, schema_b)

    rows: list[dict[str, Any]] = []
    cancelled = False
    for index, column in enumerate(schema_diff["shared"]):
        if should_cancel is not None and should_cancel():
            cancelled = True
            break
        dtype_name = str(schema_a[column])
        if dtype_name in NUMERIC_DTYPE_NAMES:
            row = _compare_numeric(files_a, files_b, column, bins)
        elif dtype_name in STRING_DTYPE_NAMES or dtype_name == "Boolean":
            row = _compare_categorical(files_a, files_b, column, top_k)
        else:
            row = None  # no comparable distribution defined for this type yet
        if row is None:
            continue
        rows.append(row)
        if on_column is not None:
            on_column(index, column, row)

    # Biggest change first — the point of the feature is not to make someone
    # scan 40 rows of numbers looking for what moved.
    rows.sort(key=lambda r: r["psi"], reverse=True)
    return {
        "schema": schema_diff,
        "columns": rows,
        "row_count_a": _row_count(files_a),
        "row_count_b": _row_count(files_b),
        "files_a": len(files_a),
        "files_b": len(files_b),
        "cancelled": cancelled,
    }
