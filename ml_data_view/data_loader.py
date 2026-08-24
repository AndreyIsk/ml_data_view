"""Headless data-loading logic for ML Dataset Profiler.

Reads .parquet, .arrow, and .csv files and extracts the schema/size/
preview info the UI needs. No pywebview or UI imports in this module,
so it can be imported and unit-tested without launching the app.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import polars as pl

PREVIEW_ROW_LIMIT = 50
SUPPORTED_EXTENSIONS = {".parquet", ".arrow", ".csv"}


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


def _read_dataframe(path: Path, suffix: str) -> pl.DataFrame:
    if suffix == ".csv":
        return pl.read_csv(path)
    raise UnsupportedFileTypeError(f"No reader wired up yet for '{suffix}'")


def load_dataset(path: str | Path) -> DatasetInfo:
    """Load a dataset file and extract its schema, size, and preview rows.

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

    df = _read_dataframe(path, suffix)
    null_counts = df.null_count().to_dicts()[0]

    columns = [
        ColumnInfo(name=name, dtype=str(dtype), null_count=null_counts[name])
        for name, dtype in df.schema.items()
    ]

    preview_rows = df.head(PREVIEW_ROW_LIMIT).to_dicts()

    return DatasetInfo(
        file_path=str(path),
        file_size_bytes=path.stat().st_size,
        row_count=df.height,
        columns=columns,
        preview_rows=preview_rows,
    )
