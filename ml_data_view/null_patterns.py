"""Detect columns whose missing values occur on the SAME rows.

Columns that go missing together usually share one upstream cause (a
join that didn't match, a feed that dropped out for a period), which is
worth surfacing on its own rather than leaving a reader to spot four
identical null counts in a schema table and guess.

Cost control — the reason this is two passes rather than one:

  Pass 1 groups columns by null COUNT alone, using counts already
  computed for the schema table. It costs nothing and is purely a filter:
  columns whose null counts differ cannot be null on the same rows, so
  they never reach pass 2.

  Pass 2 verifies actual row-level co-occurrence, but only WITHIN a
  candidate group, and only against one anchor column rather than every
  pair. Anchor-only is sound here because pass 1 has already established
  that every column in the group has (near-)equal null counts: if
  |A| ~ |B| and |A n B| ~ |A|, then A and B are near-identical sets, so
  agreeing with the anchor implies agreeing with each other. That turns
  what would be O(n^2) pairwise comparisons across the whole schema into
  O(k) streaming aggregations inside a small group.

Comparing every pair of columns row by row across a wide dataset — the
obvious implementation — is exactly what this avoids.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Protocol

import polars as pl


class CoNullCache(Protocol):
    """Persistent per-file co-null counts. file_profile.CachedCoNulls is
    the implementation; the protocol is here so null_patterns does not have
    to import the cache to accept one."""

    def get(self, file: Path, anchor: str, other: str) -> int | None: ...

    def put(self, file: Path, anchor: str, counts: dict[str, int]) -> None: ...

# Null counts this close (relative to the larger) are treated as possibly
# the same pattern. Real upstream causes rarely produce exactly equal
# counts, so an exact-match filter would miss most true positives.
NULL_COUNT_TOLERANCE = 0.02
# Share of the larger null set that must actually co-occur to report a
# group. Set high: the claim being made is "these go missing together",
# and a half-overlap is a different, weaker statement.
MIN_OVERLAP = 0.99
# Cap on aggregations per streaming query. Many aggregations over one scan
# is the pattern that repeatedly blew memory on wide multi-file data (see
# data_loader._bounded_parquet_missing_counts), so batch rather than
# building one query per group of unbounded width.
MAX_EXPRS_PER_QUERY = 8


def _candidate_groups(null_counts: dict[str, int], total_rows: int) -> list[list[str]]:
    """Pass 1: columns bucketed by near-equal null count. Columns with no
    nulls are excluded — "never missing" is not a shared pattern."""
    with_nulls = sorted(
        ((name, count) for name, count in null_counts.items() if count > 0),
        key=lambda item: item[1],
        reverse=True,
    )
    groups: list[list[str]] = []
    current: list[str] = []
    anchor_count = 0
    for name, count in with_nulls:
        if current and abs(anchor_count - count) <= NULL_COUNT_TOLERANCE * max(anchor_count, count):
            current.append(name)
            continue
        if len(current) > 1:
            groups.append(current)
        current = [name]
        anchor_count = count
    if len(current) > 1:
        groups.append(current)
    return groups


def _co_null_counts(
    lf: pl.LazyFrame,
    anchor: str,
    others: list[str],
    files: list[Path] | None,
    predicate: pl.Expr | None = None,
    co_null_cache: CoNullCache | None = None,
) -> dict[str, int]:
    """Rows where `anchor` and each of `others` are BOTH null.

    With `files`, each file is aggregated separately and the results are
    summed — which is exact, since a co-null row belongs to exactly one
    file. Querying the concatenated frame instead grows with file count
    (~28-42MB per file, measured), the pattern that repeatedly exhausted
    memory elsewhere in this codebase.

    `co_null_cache` makes that per-file half persistent. This is the one
    part of correlated missingness that caches: which PAIRS to check
    depends on merged null counts and so cannot be known file by file, but
    a chosen pair's count is a plain sum over files. Only the pairs a file
    has no cached answer for reach a query, so re-opening an unchanged
    folder reads nothing here. A filtered build never caches — the counts
    would describe a different set of rows.
    """
    anchor_null = pl.col(anchor).is_null()
    totals = dict.fromkeys(others, 0)
    usable_cache = co_null_cache if (files and predicate is None) else None
    sources: list[tuple[Path | None, pl.LazyFrame]] = (
        [(Path(file), pl.scan_parquet(file)) for file in files] if files else [(None, lf)]
    )
    if predicate is not None:
        sources = [(file, source.filter(predicate)) for file, source in sources]

    for file, source in sources:
        wanted = list(others)
        if usable_cache is not None and file is not None:
            remaining = []
            for name in wanted:
                cached = usable_cache.get(file, anchor, name)
                if cached is None:
                    remaining.append(name)
                else:
                    totals[name] += cached
            wanted = remaining
        computed: dict[str, int] = {}
        for start in range(0, len(wanted), MAX_EXPRS_PER_QUERY):
            batch = wanted[start : start + MAX_EXPRS_PER_QUERY]
            result = source.select(
                [(anchor_null & pl.col(name).is_null()).sum().alias(name) for name in batch]
            ).collect(engine="streaming")
            for name in batch:
                count = int(result[0, name])
                computed[name] = count
                totals[name] += count
        if usable_cache is not None and file is not None and computed:
            usable_cache.put(file, anchor, computed)
    return totals


def _verify_group(
    lf: pl.LazyFrame,
    group: list[str],
    null_counts: dict[str, int],
    files: list[Path] | None = None,
    predicate: pl.Expr | None = None,
    co_null_cache: CoNullCache | None = None,
) -> dict[str, Any] | None:
    """Pass 2: confirm the candidates really are null on the same rows.

    One streaming aggregation (batched) counts rows where the anchor AND
    each other column are both null. A column is kept only if that count
    covers MIN_OVERLAP of the LARGER of the two null sets — using the
    larger is what stops a column with few nulls from looking perfectly
    contained inside one with many.
    """
    anchor, others = group[0], group[1:]
    co_null = _co_null_counts(lf, anchor, others, files, predicate, co_null_cache)

    confirmed = [anchor]
    overlaps = []
    for name in others:
        larger = max(null_counts[anchor], null_counts[name])
        if larger == 0:
            continue
        overlap = co_null[name] / larger
        if overlap >= MIN_OVERLAP:
            confirmed.append(name)
            overlaps.append(overlap)

    if len(confirmed) < 2:
        return None
    return {
        "columns": confirmed,
        "null_count": null_counts[anchor],
        "overlap": round(min(overlaps), 4),
    }


def correlated_missingness(
    lf: pl.LazyFrame,
    null_counts: dict[str, int],
    total_rows: int,
    files: list[Path] | None = None,
    predicate: pl.Expr | None = None,
    co_null_cache: CoNullCache | None = None,
) -> list[dict[str, Any]]:
    """Groups of 2+ columns whose nulls co-occur near-perfectly.

    `null_counts` is reused from the schema table rather than recomputed —
    it is the whole of pass 1. Returns [] when nothing qualifies.

    `co_null_cache`, when given, persists pass 2's per-file counts so a
    re-opened folder does not pay for them again (see _co_null_counts).
    """
    if total_rows <= 0:
        return []
    groups = []
    for candidate in _candidate_groups(null_counts, total_rows):
        verified = _verify_group(
            lf,
            candidate,
            null_counts,
            files=files,
            predicate=predicate,
            co_null_cache=co_null_cache,
        )
        if verified is not None:
            groups.append(verified)
    return groups
