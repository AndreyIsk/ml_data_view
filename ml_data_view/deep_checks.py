"""Deep dataset checks — the Findings tab's half that has to read data.

findings.py's instant checks are free because the app already computed
what they read. These three cannot be: nothing in the chart payload
answers "are there duplicate rows". So they run only on an explicit "Run
deep checks" click, on a worker thread, and every one obeys
PROJECT_BRIEF.md's rule — a bounded slice or a streaming aggregation,
never a full materialization, and never a query over the multi-file
concat.

Two of the three are SAMPLED, and say so in their own finding text. That
is not a shortcut taken quietly: an exact duplicate count over the real
195.8M-row folder means holding ~195M row hashes, which is the same
unbounded-set pattern that exhausted a 2GB cap in categorical_bar_data.
A number the user might quote as dataset-wide must state its own scope.

Design spec: docs/superpowers/specs/2026-08-31-findings-tab-design.md
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import polars as pl

from ml_data_view.findings import CheckSpec, Finding
from ml_data_view.image_data import _encode_thumbnail, _resolve_path_bytes

# Rows compared for duplicate detection. Bounded on purpose — see the
# module docstring. Large enough that a genuinely duplicate-ridden dataset
# shows it immediately, small enough to stay cheap on a 200GB folder.
DUPLICATE_SAMPLE_ROWS = 100_000
# Rows whose image references are resolved. Wider than the Media tab's
# 50-row preview grid, since the point of this check is to find failures
# the preview was too small to reveal.
IMAGE_CHECK_SAMPLE_ROWS = 500
# Numeric values that commonly encode "no data" while surviving into
# training as real numbers, silently corrupting every statistic computed
# from them.
SENTINEL_VALUES = (-1, -999, -9999)
# Share of a column's non-null values a sentinel must reach before it is
# reported. Without this, -1 fires on nearly every dataset that has a
# legitimate -1 in it: a handful of them is ordinary data, whereas a code
# standing in for "missing" appears in a meaningful fraction of rows.
SENTINEL_MIN_SHARE = 0.01

_NUMERIC_DTYPES = (
    pl.Int8, pl.Int16, pl.Int32, pl.Int64,
    pl.UInt8, pl.UInt16, pl.UInt32, pl.UInt64,
    pl.Float32, pl.Float64,
)
# Excluded from row identity for duplicate detection: DUPLICATE_SAMPLE_ROWS
# rows of image blobs or 768-float embeddings is not a bounded read.
_UNCOMPARABLE_DTYPES = (pl.Binary, pl.List, pl.Array, pl.Struct, pl.Object)


@dataclass
class DeepContext:
    """What every deep check gets.

    `files` is the source file list for a folder load (None for a single
    file) — present so per-file checks never query the concatenated frame,
    the rule that keeps memory flat in file count everywhere else.
    """

    lf: pl.LazyFrame
    files: list[Path] | None = None
    dataset_dir: Path | None = None
    image_columns: list[dict[str, str]] = field(default_factory=list)


def _sources(ctx: DeepContext) -> list[pl.LazyFrame]:
    """One LazyFrame per source file, or the single frame when there is no
    file list. Per file, never the concat."""
    if ctx.files:
        return [pl.scan_parquet(file) for file in ctx.files]
    return [ctx.lf]


def duplicate_rows(
    ctx: DeepContext, sample_rows: int = DUPLICATE_SAMPLE_ROWS
) -> list[Finding]:
    """Identical rows within a bounded sample.

    Bounded twice: a frame-level `.head()` caps rows, and hashing happens
    inside polars so only `sample_rows` u64s ever come back rather than
    the rows themselves.

    Binary/List/Array/Struct columns are left OUT of row identity — see
    _UNCOMPARABLE_DTYPES. That genuinely changes what "duplicate" means
    (two rows alike on every scalar but holding different images count as
    duplicates here), so the finding names the excluded columns rather
    than quietly redefining the term.
    """
    schema = ctx.lf.collect_schema()
    comparable = [
        name for name, dtype in schema.items() if not isinstance(dtype, _UNCOMPARABLE_DTYPES)
    ]
    excluded = [name for name in schema if name not in comparable]
    if not comparable:
        # Nothing left to compare is not "no duplicates found" — it is no
        # answer at all, and reporting a clean result would be a lie.
        return []

    hashes = (
        ctx.lf.head(sample_rows)
        .select(pl.struct(comparable).hash().alias("__row_hash"))
        .collect()
    )
    scanned = hashes.height
    duplicates = scanned - hashes["__row_hash"].n_unique()
    if duplicates <= 0:
        return []

    detail = (
        f"{duplicates:,} duplicate rows among the first {scanned:,} scanned. "
        "Not a dataset-wide count — the check reads a bounded sample, and on a "
        "folder that sample comes from the first files only."
    )
    if excluded:
        detail += (
            f" Rows were compared on {len(comparable)} of {len(schema)} columns; "
            f"{', '.join(excluded)} excluded, since reading them for every sampled "
            "row would not be a bounded read."
        )
    return [
        Finding(
            check_id="duplicate_rows",
            severity="warning",
            title=f"{duplicates:,} duplicate rows in the sample",
            detail=detail,
            columns=list(comparable),
        )
    ]


def sentinel_values(ctx: DeepContext) -> list[Finding]:
    """Numeric columns where a missing-data code survives as a real value.

    EXACT, unlike the other two deep checks: counting equality against a
    fixed set of values is a streaming aggregation whose per-file results
    add up precisely, so there is nothing to sample and nothing to caveat.
    """
    schema = ctx.lf.collect_schema()
    numeric = [name for name, dtype in schema.items() if dtype in _NUMERIC_DTYPES]
    if not numeric:
        return []

    counts: dict[tuple[str, int], int] = {}
    totals: dict[str, int] = dict.fromkeys(numeric, 0)

    for source in _sources(ctx):
        exprs = [pl.col(name).drop_nulls().len().alias(f"__n__{name}") for name in numeric]
        exprs += [
            (pl.col(name) == sentinel).sum().alias(f"__s__{name}__{index}")
            for name in numeric
            for index, sentinel in enumerate(SENTINEL_VALUES)
        ]
        result = source.select(exprs).collect(engine="streaming")
        for name in numeric:
            totals[name] += int(result[0, f"__n__{name}"] or 0)
            for index, sentinel in enumerate(SENTINEL_VALUES):
                key = (name, sentinel)
                counts[key] = counts.get(key, 0) + int(result[0, f"__s__{name}__{index}"] or 0)

    findings = []
    for name in numeric:
        total = totals[name]
        if total <= 0:
            continue
        for sentinel in SENTINEL_VALUES:
            count = counts.get((name, sentinel), 0)
            share = count / total
            if count == 0 or share < SENTINEL_MIN_SHARE:
                continue
            findings.append(
                Finding(
                    check_id="sentinel_values",
                    severity="warning",
                    title=f"Column '{name}' contains {sentinel} as a value",
                    detail=(
                        f"{sentinel} appears {count:,} times in '{name}' "
                        f"({share:.1%} of its {total:,} non-missing values). Values like this "
                        "usually encode 'no data' upstream, but survive as real numbers into "
                        "every statistic computed from the column. This count is exact."
                    ),
                    columns=[name],
                )
            )
    return findings


def unresolvable_images(
    ctx: DeepContext, sample_rows: int = IMAGE_CHECK_SAMPLE_ROWS
) -> list[Finding]:
    """Image references that do not produce an image, and WHY.

    This is the check's reason to exist. findings._check_broken_image_refs
    can only say "no usable image", because image_previews_data records a
    single null thumbnail for three different causes. Here the values are
    re-read, so a missing file, undecodable bytes and a genuinely null
    value are counted separately.

    Bounded by a frame-level `.head()` on a single column.
    """
    findings = []
    for image_column in ctx.image_columns or []:
        column, source = image_column["column"], image_column["source"]
        values = (
            ctx.lf.select(pl.col(column)).head(sample_rows).collect()[column].to_list()
        )
        if not values:
            continue

        null_values = missing_files = undecodable = 0
        for value in values:
            if value is None:
                null_values += 1
                continue
            data = value if source == "bytes" else _resolve_path_bytes(value, ctx.dataset_dir)
            if data is None:
                missing_files += 1
            elif _encode_thumbnail(data, (32, 32)) is None:
                undecodable += 1

        if not (missing_files or undecodable):
            continue

        parts = []
        if missing_files:
            parts.append(f"{missing_files:,} with a missing file")
        if undecodable:
            parts.append(f"{undecodable:,} with undecodable bytes")
        if null_values:
            parts.append(f"{null_values:,} with no value at all")
        findings.append(
            Finding(
                check_id="unresolvable_images",
                severity="warning",
                title=f"Column '{column}' has {missing_files + undecodable:,} unusable images",
                detail=(
                    f"Of the first {len(values):,} rows scanned: {', '.join(parts)}. "
                    "A bounded sample, not a dataset-wide count."
                ),
                columns=[column],
            )
        )
    return findings


DEEP_CHECKS: list[CheckSpec] = [
    CheckSpec("duplicate_rows", "Duplicate rows", "deep", duplicate_rows),
    CheckSpec("sentinel_values", "Sentinel values", "deep", sentinel_values),
    CheckSpec("unresolvable_images", "Image references", "deep", unresolvable_images),
]
