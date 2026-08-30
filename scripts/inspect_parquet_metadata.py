"""Read-only diagnostic — NOT part of the app, just a one-off investigation
tool for the folder-open OOM crash.

Inspects every .parquet file under a folder using footer metadata only:
row count, row-group count, per-row-group row counts and on-disk (compressed)
byte sizes, column count, and the shape of any list/embedding-like column.

Nothing here reads row data. Every field printed comes from one of two
metadata-only sources:
  - pq.ParquetFile(path).metadata     — the file's footer (row counts, row
                                         group byte sizes); parsing it does
                                         not touch any column's actual data.
  - pq.ParquetFile(path).schema_arrow — the Arrow schema pyarrow/polars
                                         embed in the footer's key/value
                                         metadata when they write the file;
                                         reading it round-trips exact types,
                                         including a fixed-size list's
                                         dimension, again without touching
                                         data.
A plain (variable-length) list column's per-row length is NOT retrievable
from metadata — Parquet only stores repetition/definition levels for those
inline with the data itself. This script reports such columns as
"variable-length" rather than guessing a dimension.

Usage:
    python scripts/inspect_parquet_metadata.py <folder> [--json]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq


def _describe_type(dtype: pa.DataType) -> dict[str, Any]:
    if pa.types.is_fixed_size_list(dtype):
        return {
            "kind": "fixed_size_list",
            "value_type": str(dtype.value_type),
            "dimension": dtype.list_size,
            "note": "dimension read directly from file metadata (Arrow FixedSizeListType)",
        }
    if pa.types.is_list(dtype) or pa.types.is_large_list(dtype):
        return {
            "kind": "variable_length_list",
            "value_type": str(dtype.value_type),
            "dimension": None,
            "note": "per-row length lives in the data itself, not in metadata — not read",
        }
    return {"kind": "scalar", "type": str(dtype)}


def _row_group_stats(metadata: pq.FileMetaData) -> dict[str, Any]:
    row_counts = [metadata.row_group(i).num_rows for i in range(metadata.num_row_groups)]
    # total_byte_size is the row group's UNCOMPRESSED size (pyarrow's own
    # term for it) — i.e. roughly what a reader materializes in memory if
    # it ever pulls that whole row group, not the on-disk (compressed) size.
    byte_sizes = [metadata.row_group(i).total_byte_size for i in range(metadata.num_row_groups)]
    return {
        "count": metadata.num_row_groups,
        "row_counts": row_counts,  # one entry per row group — no aggregation lost
        "total_uncompressed_bytes": sum(byte_sizes),
        "avg_uncompressed_bytes": (sum(byte_sizes) // len(byte_sizes)) if byte_sizes else 0,
        "max_uncompressed_bytes": max(byte_sizes, default=0),
    }


def inspect_file(path: Path) -> dict[str, Any]:
    """Metadata-only: opening ParquetFile and reading .metadata/.schema_arrow
    parses only the file's footer, never a row-group's actual column data."""
    parquet_file = pq.ParquetFile(path)
    metadata = parquet_file.metadata
    schema = parquet_file.schema_arrow
    return {
        "file": str(path),
        "file_size_bytes": path.stat().st_size,
        "row_count": metadata.num_rows,
        "column_count": metadata.num_columns,
        "row_groups": _row_group_stats(metadata),
        "columns": [
            {"name": field.name, **_describe_type(field.type)} for field in schema
        ],
    }


def _print_human(report: dict[str, Any]) -> None:
    print(f"\n{report['file']}")
    print(f"  file size on disk : {report['file_size_bytes']:,} bytes")
    print(f"  rows              : {report['row_count']:,}")
    print(f"  columns           : {report['column_count']}")
    rg = report["row_groups"]
    counts = rg["row_counts"]
    counts_repr = counts if len(counts) <= 8 else (
        f"min={min(counts)} max={max(counts)} avg={sum(counts)//len(counts)} (n={len(counts)})"
    )
    print(
        f"  row groups        : {rg['count']} - rows per group: {counts_repr}"
    )
    print(
        f"                      uncompressed bytes (per-row-group materialized size): "
        f"total={rg['total_uncompressed_bytes']:,} avg={rg['avg_uncompressed_bytes']:,} "
        f"max={rg['max_uncompressed_bytes']:,}"
    )
    list_columns = [c for c in report["columns"] if c["kind"] != "scalar"]
    if list_columns:
        print("  list/embedding-like columns:")
        for col in list_columns:
            if col["kind"] == "fixed_size_list":
                print(f"    - {col['name']}: fixed-size list<{col['value_type']}>, dimension={col['dimension']}")
            else:
                print(f"    - {col['name']}: variable-length list<{col['value_type']}> (dimension unknown, not read)")


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("folder", type=Path, help="Folder to scan recursively for .parquet files")
    parser.add_argument("--json", action="store_true", help="Emit one JSON object per file instead of the human summary")
    args = parser.parse_args(argv)

    files = sorted(p for p in args.folder.rglob("*.parquet") if p.is_file())
    if not files:
        print(f"No .parquet files found under {args.folder}")
        return 1

    if not args.json:
        print(f"Found {len(files)} parquet file(s) under {args.folder}")

    errors: list[tuple[Path, Exception]] = []
    for path in files:
        try:
            report = inspect_file(path)
        except Exception as exc:  # metadata-read failure — reported, not fatal
            errors.append((path, exc))
            if args.json:
                print(json.dumps({"file": str(path), "error": str(exc)}))
            else:
                print(f"\n{path}\n  COULD NOT READ METADATA: {exc}")
            continue
        if args.json:
            print(json.dumps(report))
        else:
            _print_human(report)

    if not args.json:
        print(f"\n--- done: {len(files)} file(s) inspected, {len(errors)} unreadable ---")
    return 0 if not errors else 2


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
