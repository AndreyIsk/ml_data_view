"""Headless data-loading logic for ML Dataset Profiler.

Scans .parquet, .arrow, and .csv files lazily (polars LazyFrame) and
extracts the schema/size/preview info the UI needs. No pywebview or UI
imports in this module, so it can be imported and unit-tested without
launching the app.

Architecture note (see PROJECT_BRIEF.md): everything downstream of
scan_dataframe() works off a polars LazyFrame, not an eagerly-read
DataFrame, and only ever collect()s a bounded slice or a streaming
aggregation — never the whole frame. That's what lets a dataset far
larger than RAM (up to roughly 200GB) get profiled without materializing
it. Single-file only for now; multi-file/folder scanning is a later stage.
"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import polars as pl
import pyarrow.parquet as pq

PREVIEW_ROW_LIMIT = 50
SUPPORTED_EXTENSIONS = {".parquet", ".arrow", ".csv"}
# Rows per batch when decoding parquet columns for missing-value counts
# (see _bounded_parquet_missing_counts). This — not the row-group size —
# is what caps peak memory for that step.
BATCH_ROWS = 65_536
# Per-column aggregations per query on the filtered path. Same reasoning as
# null_patterns.MAX_EXPRS_PER_QUERY: many aggregations over one scan is what
# blew memory on wide multi-file data.
_FILTERED_EXPRS_PER_QUERY = 8


class UnsupportedFileTypeError(ValueError):
    """Raised when the given file extension is not one we support."""


@dataclass
class ColumnInfo:
    name: str
    dtype: str
    null_count: int

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "dtype": self.dtype, "null_count": self.null_count}


@dataclass
class DatasetInfo:
    file_path: str
    file_size_bytes: int
    row_count: int
    columns: list[ColumnInfo]
    preview_rows: list[dict[str, Any]]

    def to_dict(self) -> dict[str, Any]:
        return {
            "file_path": self.file_path,
            "file_size_bytes": self.file_size_bytes,
            "row_count": self.row_count,
            "columns": [column.to_dict() for column in self.columns],
            "preview_rows": self.preview_rows,
        }


def _dispatch_scanner(path: Path, suffix: str) -> pl.LazyFrame:
    if suffix == ".csv":
        return pl.scan_csv(path)
    if suffix == ".parquet":
        return pl.scan_parquet(path)
    # .feather is the same Arrow IPC format as .arrow, just a different
    # common extension for it — folder_loader.py's discovery accepts both;
    # single-file open (SUPPORTED_EXTENSIONS above) still doesn't, unchanged.
    if suffix in (".arrow", ".feather"):
        return pl.scan_ipc(path)
    raise UnsupportedFileTypeError(f"No scanner wired up yet for '{suffix}'")


def scan_dataframe(path: str | Path) -> pl.LazyFrame:
    """Validate and lazily scan a dataset file — nothing is read into
    memory yet beyond schema metadata.

    Raises:
        UnsupportedFileTypeError: if the extension isn't .parquet/.arrow/.csv
        FileNotFoundError: if the file doesn't exist
    """
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix not in SUPPORTED_EXTENSIONS:
        raise UnsupportedFileTypeError(
            f"Unsupported file type '{suffix}'. Supported types: "
            f"{', '.join(sorted(SUPPORTED_EXTENSIONS))}"
        )
    if not path.exists():
        raise FileNotFoundError(f"No such file: {path}")
    return _dispatch_scanner(path, suffix)


def row_count(lf: pl.LazyFrame) -> int:
    """Total row count via a single streaming aggregation — never a full
    materialization, regardless of source file size."""
    return int(lf.select(pl.len()).collect(engine="streaming")[0, 0])


def _null_count_expr(name: str, dtype: pl.DataType) -> pl.Expr:
    """Missing-value count for one column, expressed lazily. Counts NaN as
    missing too for float columns (see the docstring this replaced on the
    old eager `_missing_count`) — a NaN is just as "no usable value" as a
    null for profiling purposes, and this keeps it consistent with the
    Phase 1 null heatmap, which treats NaN as missing too.
    """
    column = pl.col(name)
    if dtype in (pl.Float32, pl.Float64):
        return (column.null_count() + column.is_nan().sum().fill_null(0)).alias(name)
    return column.null_count().alias(name)


def _parquet_null_counts_from_metadata(
    files: list[Path], schema: pl.Schema
) -> tuple[dict[str, int], set[str]]:
    """Sum each wanted column's null count directly from every parquet
    file's row-group column-chunk statistics — footer metadata only, the
    same technique already used for row counts, never decoding a column's
    actual data.

    Returns (counts, unavailable): `counts` has an entry for every column
    that had usable statistics in *every* row group of *every* file;
    `unavailable` lists everything else (Parquet's per-column statistics
    are optional, not guaranteed — a writer can omit them — and a
    non-scalar/nested column may not appear as a single matching leaf
    path at all). Callers fall back to decoding just the unavailable
    columns, not the whole query.
    """
    wanted = set(schema)
    totals: dict[str, int] = dict.fromkeys(wanted, 0)
    matched: set[str] = set()
    unavailable: set[str] = set()

    for file in files:
        metadata = pq.ParquetFile(file).metadata
        for rg_index in range(metadata.num_row_groups):
            row_group = metadata.row_group(rg_index)
            for col_index in range(row_group.num_columns):
                column_chunk = row_group.column(col_index)
                name = column_chunk.path_in_schema
                if name not in wanted or name in unavailable:
                    continue
                stats = column_chunk.statistics
                if stats is None or not stats.has_null_count:
                    unavailable.add(name)
                    continue
                totals[name] += stats.null_count
                matched.add(name)

    # Any wanted column that never matched a row-group column at all
    # (e.g. a nested/list type whose leaf path doesn't equal its own
    # name) is just as "unavailable" as one with an explicit missing flag.
    unavailable |= wanted - matched
    for name in unavailable:
        totals.pop(name, None)
    return totals, unavailable


def _bounded_parquet_missing_counts(
    files: list[Path],
    schema: pl.Schema,
    columns: set[str],
    on_file: Callable[[int, Path], None] | None = None,
    should_cancel: Callable[[], bool] | None = None,
) -> dict[str, int]:
    """Missing-value counts for `columns` by decoding parquet in bounded
    batches, for columns Parquet statistics couldn't answer.

    Why this exists instead of a polars streaming aggregation: polars
    streams at ROW GROUP granularity — a row group is the atomic unit it
    reads, and it cannot subdivide one. Spark-written files like the
    Criteo set are commonly a single row group covering the entire file
    (verified: 1 row group of 1,529,035 rows), so "streaming" a concat of
    250 such files means holding whole 1.5M-row chunks and parallelizing
    them across every core — which is what exhausted memory. pyarrow's
    iter_batches() splits *within* a row group, so peak memory here is
    bounded by BATCH_ROWS x len(columns) regardless of row-group size,
    file size, or how many files are in the folder.

    Files are processed one at a time and batches are released as they go,
    so cost does not grow with file count either.

    One column at a time, not all of them together: iter_batches bounds
    the batch it hands back, but decoding still buffers the underlying
    column chunks for the row group it is reading, so asking for N
    columns at once costs N chunks. Measured on one real Criteo file,
    all 26 string columns at once peaked at 1,154MB versus 623MB
    per-column (~552MB of both being fixed interpreter/library baseline).
    Parquet is columnar, so a per-column pass reads only that column's
    chunk rather than re-reading the file.

    Counts NaN as missing for float columns, matching _null_count_expr.

    `on_file(index, path)` fires after a file is FULLY counted, and
    `should_cancel()` is checked only at a file boundary. Both together
    give a clean truncation point: when cancelled after N callbacks, the
    returned totals are exact for files[:N] and contain nothing partial
    from the file that was next. Callers rely on that to present a
    stopped load as a smaller dataset rather than an incomplete one.
    """
    import pyarrow.compute as pc

    totals = dict.fromkeys(columns, 0)
    if not columns:
        # Still report file progress so a schema with nothing to decode
        # (every column answered from metadata) doesn't look frozen.
        for index, file in enumerate(files):
            if should_cancel is not None and should_cancel():
                break
            if on_file is not None:
                on_file(index, file)
        return totals
    float_columns = {name for name in columns if schema[name] in (pl.Float32, pl.Float64)}

    for index, file in enumerate(files):
        if should_cancel is not None and should_cancel():
            break
        parquet_file = pq.ParquetFile(file)
        available = set(parquet_file.schema_arrow.names)
        for name in sorted(columns):
            if name not in available:
                continue
            for batch in parquet_file.iter_batches(batch_size=BATCH_ROWS, columns=[name]):
                array = batch.column(0)
                count = array.null_count
                if name in float_columns:
                    nan_total = pc.sum(pc.is_nan(array)).as_py()
                    count += nan_total or 0
                totals[name] += count
        if on_file is not None:
            on_file(index, file)
    return totals


def _filtered_null_counts(
    files: list[Path], schema: pl.Schema, predicate: pl.Expr
) -> dict[str, int]:
    """Missing-value counts over only the rows matching `predicate`.

    A filter invalidates BOTH shortcuts the unfiltered path relies on:
    parquet footer null counts describe the whole file, and the pyarrow
    batch decode has no notion of a polars predicate. So this goes through
    polars per file, batching the per-column aggregations — many
    aggregations over one scan is the pattern that repeatedly exhausted
    memory on wide data, and a filter does not make that safe, only
    smaller.
    """
    totals = dict.fromkeys(schema, 0)
    names = list(schema)
    for file in files:
        scan = _dispatch_scanner(Path(file), Path(file).suffix.lower()).filter(predicate)
        for start in range(0, len(names), _FILTERED_EXPRS_PER_QUERY):
            batch = names[start : start + _FILTERED_EXPRS_PER_QUERY]
            result = scan.select(
                [_null_count_expr(name, schema[name]) for name in batch]
            ).collect(engine="streaming")
            for name in batch:
                totals[name] += int(result[0, name])
    return totals


def _null_counts(
    lf: pl.LazyFrame,
    schema: pl.Schema,
    files: list[Path] | None = None,
    on_file: Callable[[int, Path], None] | None = None,
    should_cancel: Callable[[], bool] | None = None,
    predicate: pl.Expr | None = None,
) -> dict[str, int]:
    """Every column's null count.

    When every file in `files` is a `.parquet`, this reads null counts
    straight from each file's footer statistics — metadata only, no
    column data decoded (see _parquet_null_counts_from_metadata). Float
    columns still need a value beyond what Parquet's own stats capture
    (NaN is not null), so those get one additional, much smaller
    streaming aggregation scoped to just is_nan().sum() on float columns
    — not the full null_count()-over-every-column query this replaces.

    A column with no usable metadata (missing stats, or non-parquet
    input) falls back to the original decode-based `_null_count_expr`
    for just that column, batched into the same small streaming query as
    the float NaN counts above, so there's still only one .collect().
    """
    if not schema:
        return {}

    if predicate is not None:
        if not files:
            exprs = [_null_count_expr(name, dtype) for name, dtype in schema.items()]
            decoded = lf.filter(predicate).select(exprs).collect(engine="streaming")
            return {name: int(decoded[0, name]) for name in schema}
        return _filtered_null_counts([Path(f) for f in files], schema, predicate)

    parquet_files = [Path(f) for f in files] if files else []
    use_metadata = bool(files) and all(f.suffix.lower() == ".parquet" for f in parquet_files)

    if not use_metadata:
        # Non-parquet input (csv/arrow): no footer statistics and no
        # iter_batches equivalent wired up here, so keep the original
        # decode-based streaming aggregation, unchanged.
        exprs = [_null_count_expr(name, dtype) for name, dtype in schema.items()]
        decoded = lf.select(exprs).collect(engine="streaming")
        return {name: int(decoded[0, name]) for name in schema}

    _, fallback_columns = _parquet_null_counts_from_metadata(parquet_files, schema)

    # Parquet's null_count statistic does not cover NaN, but the app counts
    # NaN as missing for floats — so every float column goes through the
    # bounded decode regardless of whether its statistics were present.
    # That pass returns nulls AND NaN together, so its result is used
    # whole rather than added to the metadata count (which would
    # double-count the nulls).
    float_dtypes = (pl.Float32, pl.Float64)
    float_columns = {name for name, dtype in schema.items() if dtype in float_dtypes}
    decode_columns = fallback_columns | float_columns

    processed = 0

    def _track(index: int, path: Path) -> None:
        nonlocal processed
        processed = index + 1
        if on_file is not None:
            on_file(index, path)

    decoded_counts = _bounded_parquet_missing_counts(
        parquet_files, schema, decode_columns, on_file=_track, should_cancel=should_cancel
    )

    # A cancelled pass covers only files[:processed]. The metadata-derived
    # counts must be recomputed over that same subset, or the result would
    # silently mix per-column totals taken from different numbers of files
    # — every metadata column reading high against every decoded one.
    counted_files = parquet_files if processed == len(parquet_files) else parquet_files[:processed]
    metadata_counts, _ = _parquet_null_counts_from_metadata(counted_files, schema)

    return {
        name: decoded_counts[name] if name in decode_columns else metadata_counts.get(name, 0)
        for name in schema
    }


def _json_safe_preview_value(value: Any) -> Any:
    """Replace non-JSON-safe values with a placeholder.

    Raw bytes (e.g. a Binary column holding image data, Phase 3) can't
    cross the pywebview js_api bridge, which serializes preview_rows to
    JSON — a bytes value there would crash load_file() for any dataset
    with such a column.
    """
    if isinstance(value, bytes):
        return f"<binary: {len(value)} bytes>"
    return value


def dataset_info_from_lazy(
    lf: pl.LazyFrame,
    path: str | Path,
    total_rows: int | None = None,
    file_size_bytes: int | None = None,
    files: list[Path] | None = None,
    null_counts: dict[str, int] | None = None,
    predicate: pl.Expr | None = None,
) -> DatasetInfo:
    """Build a DatasetInfo from a LazyFrame and its source path.

    `total_rows`, if the caller already computed it (e.g. api.py sharing
    one row_count() call across dataset info + chart data), is reused
    instead of counting again. collect_schema() reads schema metadata only
    (no row data); preview_rows is a bounded .head().collect(); null counts
    prefer parquet footer statistics (see _null_counts) over decoding data.

    `files`, if given, is the actual list of source files `lf` was built
    from — needed for the metadata-only null-count path, since `path`
    alone doesn't always identify them (folder_loader.py's `path` here is
    the containing folder, not its files). Defaults to `[path]` when
    `path` is itself an existing file, which covers the single-file
    load_dataset() case with no extra wiring.

    `file_size_bytes`, if given, is used as-is instead of `path.stat()`'s
    size — folder_loader.py passes the combined size of a schema group's
    files here, since `path` there is the folder, not a single file whose
    size would mean anything.
    """
    path = Path(path)
    schema = lf.collect_schema()
    # `predicate` makes every number below describe only matching rows —
    # row count, null counts and the preview alike. Applied to the frame
    # rather than to each call site so nothing can accidentally mix
    # filtered and unfiltered figures on one screen.
    if predicate is not None:
        lf = lf.filter(predicate)
    if total_rows is None:
        total_rows = row_count(lf)
    source_files = files if files is not None else ([path] if path.is_file() else None)
    # `null_counts`, like `total_rows`, is trusted when the caller already
    # has it — the cancellable folder loader computes it incrementally so
    # it can report progress, and must not have it recomputed here.
    if null_counts is None:
        null_counts = _null_counts(lf, schema, files=source_files, predicate=predicate)

    columns = [
        ColumnInfo(name=name, dtype=str(dtype), null_count=null_counts[name])
        for name, dtype in schema.items()
    ]
    preview_rows = [
        {key: _json_safe_preview_value(value) for key, value in row.items()}
        for row in lf.head(PREVIEW_ROW_LIMIT).collect().to_dicts()
    ]

    return DatasetInfo(
        file_path=str(path),
        file_size_bytes=file_size_bytes if file_size_bytes is not None else path.stat().st_size,
        row_count=total_rows,
        columns=columns,
        preview_rows=preview_rows,
    )


def load_dataset(path: str | Path) -> DatasetInfo:
    """Scan a dataset file and extract its schema, size, and preview rows.

    Raises:
        UnsupportedFileTypeError: if the extension isn't .parquet/.arrow/.csv
        FileNotFoundError: if the file doesn't exist
    """
    path = Path(path)
    lf = scan_dataframe(path)
    return dataset_info_from_lazy(lf, path)
