"""User-built row search over a loaded file or folder.

The user picks the columns, the operators and the values — uniqueness is
not a fixed property of a schema, it is whatever the user says it is for
their data ("a track is unique per ID + RecKey"). So this module builds a
predicate from user input rather than exposing any fixed set of searches.

Every function here scans ONE FILE AT A TIME rather than a concatenated
LazyFrame. A single concat query measured just as fast for search
specifically, but concat queries have grown unboundedly with file count
in four separate places in this codebase (see PROJECT_BRIEF.md), and
per-file additionally gives the count a natural progress unit and a clean
cancellation point.

Performance notes, measured on 50 real files / 39.8M rows:
  - first 500 matching rows: 0.04-0.24s, because search_rows stops
    opening files as soon as the limit is reached (500 matches were found
    in file 1 of 50)
  - exact count on a NUMERIC filter: 0.02s — parquet row-group statistics
    let polars skip almost everything without decoding
  - exact count on a STRING filter: no such shortcut in Spark-written
    files, which carry no statistics for string columns, so it is a full
    scan. That is why counting is a separate, cancellable, progress-
    reporting operation rather than something search_rows blocks on.
"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import polars as pl

from ml_data_view.data_loader import _dispatch_scanner

# Enough rows to inspect a whole track without pulling an unbounded result
# set across the js_api bridge; same order of magnitude as
# data_loader.PREVIEW_ROW_LIMIT (50), which is for a much smaller purpose.
SEARCH_ROW_LIMIT = 500

_NO_VALUE_OPERATORS = {"is null", "is not null"}
_TWO_VALUE_OPERATORS = {"between"}
_STRING_ONLY_OPERATORS = {"contains"}
OPERATORS = (
    "is",
    "is not",
    ">",
    ">=",
    "<",
    "<=",
    "between",
    "is null",
    "is not null",
    "contains",
)

_INTEGER_DTYPES = {
    pl.Int8, pl.Int16, pl.Int32, pl.Int64, pl.Int128,
    pl.UInt8, pl.UInt16, pl.UInt32, pl.UInt64,
}
_FLOAT_DTYPES = {pl.Float32, pl.Float64}
_STRING_DTYPES = {pl.String, pl.Utf8}


class SearchConditionError(ValueError):
    """A condition the user cannot have meant.

    Carries `condition_index` so the UI can mark the offending row instead
    of failing the whole search with an opaque message.
    """

    def __init__(self, condition_index: int, message: str) -> None:
        self.condition_index = condition_index
        super().__init__(message)


@dataclass
class Condition:
    column: str
    operator: str
    values: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {"column": self.column, "operator": self.operator, "values": list(self.values)}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Condition:
        return cls(
            column=data["column"],
            operator=data["operator"],
            values=list(data.get("values") or []),
        )


def _scan(path: Path) -> pl.LazyFrame:
    """Lazily scan one source file. Routed through data_loader's dispatcher
    so .csv/.arrow work here exactly as .parquet does."""
    path = Path(path)
    return _dispatch_scanner(path, path.suffix.lower())


def _coerce(value: str, dtype: pl.DataType, index: int, column: str) -> Any:
    """Turn raw user text into a value comparable against `dtype`.

    Strict on purpose: '1.9' for an integer column is rejected rather than
    truncated to 1. Guessing which row the user meant is worse than saying
    the input does not fit the column.
    """
    if dtype in _INTEGER_DTYPES:
        try:
            return int(value)
        except (TypeError, ValueError):
            raise SearchConditionError(
                index, f"{column!r} holds whole numbers, so {value!r} is not a valid value."
            ) from None
    if dtype in _FLOAT_DTYPES:
        try:
            return float(value)
        except (TypeError, ValueError):
            raise SearchConditionError(
                index, f"{column!r} holds numbers, so {value!r} is not a valid value."
            ) from None
    if dtype == pl.Boolean:
        lowered = str(value).strip().lower()
        if lowered in ("true", "1", "yes"):
            return True
        if lowered in ("false", "0", "no"):
            return False
        raise SearchConditionError(
            index, f"{column!r} is true/false, so {value!r} is not a valid value."
        )
    return value


def _condition_expr(condition: Condition, dtype: pl.DataType, index: int) -> pl.Expr:
    operator = condition.operator
    column = pl.col(condition.column)

    if operator not in OPERATORS:
        raise SearchConditionError(index, f"Unknown operator {operator!r}.")

    if operator in _NO_VALUE_OPERATORS:
        return column.is_null() if operator == "is null" else column.is_not_null()

    expected = 2 if operator in _TWO_VALUE_OPERATORS else 1
    values = [v for v in condition.values if v is not None and str(v) != ""]
    if len(values) != expected:
        word = "two values" if expected == 2 else "a value"
        raise SearchConditionError(index, f"{operator!r} needs {word}.")

    if operator in _STRING_ONLY_OPERATORS:
        if dtype not in _STRING_DTYPES:
            raise SearchConditionError(
                index, f"'contains' only works on text columns, and {condition.column!r} is not one."
            )
        # literal=True: the user typed a value to look for, not a regex.
        return column.str.contains(values[0], literal=True)

    coerced = [_coerce(value, dtype, index, condition.column) for value in values]

    if operator == "is":
        return column == coerced[0]
    if operator == "is not":
        return column != coerced[0]
    if operator == ">":
        return column > coerced[0]
    if operator == ">=":
        return column >= coerced[0]
    if operator == "<":
        return column < coerced[0]
    if operator == "<=":
        return column <= coerced[0]
    low, high = coerced
    return (column >= low) & (column <= high)  # between is inclusive both ends


def build_predicate(conditions: list[Condition], schema: pl.Schema) -> pl.Expr:
    """Combine every condition with AND into one polars expression.

    Raises SearchConditionError (carrying the offending index) rather than
    letting a bad value surface as a polars traceback. With no conditions
    this matches everything, so an empty search is "show me rows" rather
    than an error.
    """
    predicate: pl.Expr | None = None
    for index, condition in enumerate(conditions):
        if condition.column not in schema:
            raise SearchConditionError(
                index, f"Column {condition.column!r} is not in this dataset."
            )
        expr = _condition_expr(condition, schema[condition.column], index)
        predicate = expr if predicate is None else (predicate & expr)
    return pl.lit(True) if predicate is None else predicate


def _projected(lf: pl.LazyFrame, columns: list[str] | None) -> pl.LazyFrame:
    """Restrict to `columns` if given. Parquet is columnar, so this is a
    real speed win rather than presentation: selecting 5 of 40 columns
    reads 5 column chunks instead of 40."""
    return lf.select(columns) if columns else lf


def search_rows(
    files: list[Path],
    predicate: pl.Expr,
    columns: list[str] | None = None,
    limit: int = SEARCH_ROW_LIMIT,
) -> list[dict[str, Any]]:
    """The first `limit` matching rows, in file order.

    Stops opening files the moment the limit is reached — the whole reason
    a folder search feels instant. Not run on a worker thread: it is
    bounded by `limit` and measured in tens of milliseconds, so threading
    it would add latency and machinery for no benefit.
    """
    rows: list[dict[str, Any]] = []
    for file in files:
        remaining = limit - len(rows)
        if remaining <= 0:
            break
        matched = (
            _projected(_scan(file), columns)
            .filter(predicate)
            .head(remaining)
            .collect(engine="streaming")
        )
        rows.extend(matched.to_dicts())
    return rows


def count_matches(
    files: list[Path],
    predicate: pl.Expr,
    on_progress: Callable[[int, Path], None] | None = None,
    should_cancel: Callable[[], bool] | None = None,
) -> int:
    """Total matching rows across every file.

    Separate from search_rows because this cannot short-circuit: on a
    string filter there are no parquet statistics to skip row groups with,
    so it is a full scan. Cancellation is checked at file boundaries, so a
    cancelled count covers exactly the files that completed — never a
    partial file.
    """
    total = 0
    for index, file in enumerate(files):
        if should_cancel is not None and should_cancel():
            break
        matched = _scan(file).filter(predicate).select(pl.len()).collect(engine="streaming")
        total += int(matched[0, 0])
        if on_progress is not None:
            on_progress(index, file)
    return total


def export_matches(
    files: list[Path],
    predicate: pl.Expr,
    columns: list[str] | None,
    out_path: str | Path,
    on_progress: Callable[[int, Path], None] | None = None,
    should_cancel: Callable[[], bool] | None = None,
) -> Path:
    """Write every matching row to CSV.

    Unlike search_rows this is not limited — the point of exporting is to
    get all of it. Rows are appended per file rather than accumulated, so
    memory stays flat however large the result is.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    wrote_header = False

    with open(out_path, "w", encoding="utf-8", newline="") as handle:
        for index, file in enumerate(files):
            if should_cancel is not None and should_cancel():
                break
            matched = (
                _projected(_scan(file), columns)
                .filter(predicate)
                .collect(engine="streaming")
            )
            # include_header only on the first write, so appending further
            # files does not repeat the header mid-file.
            handle.write(matched.write_csv(include_header=not wrote_header))
            wrote_header = True
            if on_progress is not None:
                on_progress(index, file)

    if not wrote_header:
        # No files at all — still produce a readable, empty CSV rather than
        # a zero-byte file that a reader would reject.
        out_path.write_text("", encoding="utf-8")
    return out_path
