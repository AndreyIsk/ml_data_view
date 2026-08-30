"""Worker-thread wrappers for the two search operations that can be slow.

Fetching matching rows is NOT here: it is bounded by SEARCH_ROW_LIMIT and
measured in tens of milliseconds, so threading it would add latency and
machinery for nothing.

Counting and exporting are different. Neither can short-circuit — a count
on a string filter has no parquet statistics to skip row groups with, so
it is a full scan of every file — and both are therefore cancellable and
report per-file progress.

Both derive from background_job.BackgroundJob, which carries the lifecycle
shared with the other polled jobs in this app.
"""
from __future__ import annotations

import threading
from pathlib import Path
from typing import Any

import polars as pl

from ml_data_view.background_job import BackgroundJob
from ml_data_view.search import count_matches, export_matches


class SearchCountJob(BackgroundJob):
    """Exact total number of matching rows.

    Cancelling abandons the count entirely — the displayed rows are a
    separate result and stay on screen untouched. The partial total is NOT
    reported as if it were the answer; `cancelled` is set so the UI shows
    "not counted" rather than a number that is quietly too low.
    """

    def __init__(self, files: list[Path], predicate: pl.Expr) -> None:
        super().__init__(files)
        self._predicate = predicate

    def _work(self) -> int:
        self._update(stage="counting", label="Counting matches…")
        return count_matches(
            self._files,
            self._predicate,
            on_progress=lambda i, p: self._note_file(i, p, "Counting matches"),
            should_cancel=self._cancel.is_set,
        )

    def _final_label(self, cancelled: bool) -> str:
        return "Count cancelled." if cancelled else "Count complete."


class SearchExportJob(BackgroundJob):
    """Write every matching row to CSV.

    A cancelled export leaves a valid but PARTIAL file. `cancelled` is
    surfaced so the UI can say so — handing someone a truncated CSV that
    looks complete is the failure mode worth designing against here.
    """

    def __init__(
        self,
        files: list[Path],
        predicate: pl.Expr,
        columns: list[str] | None,
        out_path: str | Path,
    ) -> None:
        super().__init__(files)
        self._predicate = predicate
        self._columns = columns
        self._out_path = Path(out_path)

    def _work(self) -> str:
        self._update(stage="exporting", label="Exporting matches…")
        path = export_matches(
            self._files,
            self._predicate,
            self._columns,
            self._out_path,
            on_progress=lambda i, p: self._note_file(i, p, "Exporting matches"),
            should_cancel=self._cancel.is_set,
        )
        return str(path)

    def _final_label(self, cancelled: bool) -> str:
        return "Export cancelled — the file is incomplete." if cancelled else "Export complete."
