"""Worker-thread wrapper for a dataset comparison.

Comparing two folders is minutes of work even when sampled, so it follows
the same pattern as every other long operation here: start() returns at
once, the UI polls progress(), cancel() takes effect at a column boundary.

One thing this job does that the others do not: it carries the column rows
completed so far in its progress state. That is what lets the results table
fill in row by row while the comparison runs, instead of staying blank and
then appearing all at once. A partial comparison is genuinely useful — half
the columns compared is half an answer — unlike a partial chart payload,
which is why cancelling here keeps what finished.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from ml_data_view.background_job import BackgroundJob
from ml_data_view.compare import COMPARE_BINS, COMPARE_TOP_K, compare_datasets


class CompareJob(BackgroundJob):
    def __init__(
        self,
        files_a: list[Path],
        files_b: list[Path],
        bins: int = COMPARE_BINS,
        top_k: int = COMPARE_TOP_K,
    ) -> None:
        super().__init__(files_a)
        self._files_a = list(files_a)
        self._files_b = list(files_b)
        self._bins = bins
        self._top_k = top_k
        # Rows completed so far, mirrored into progress state for the UI.
        self._rows: list[dict[str, Any]] = []
        self._update(rows=[], columns_done=0, columns_total=0)

    def _on_column(self, index: int, name: str, row: dict[str, Any]) -> None:
        self._rows.append(row)
        done = len(self._rows)
        total = max(self._state.get("columns_total") or 0, done)
        self._update(
            stage="comparing",
            columns_done=done,
            fraction=round(done / total, 4) if total else 0.0,
            label=f"Comparing columns — {done} of {total}",
            # A copy: the UI polls this while the worker keeps appending.
            rows=list(self._rows),
        )

    def _work(self) -> dict[str, Any]:
        self._update(stage="comparing", label="Reading schemas…")
        # Column count is only known once both schemas are read, so the
        # total is filled in here rather than in the constructor.
        from ml_data_view.compare import _scan, compare_schemas

        schema_a = _scan(self._files_a[0]).collect_schema()
        schema_b = _scan(self._files_b[0]).collect_schema()
        shared = compare_schemas(schema_a, schema_b)["shared"]
        self._update(columns_total=len(shared), files_total=len(shared))

        return compare_datasets(
            self._files_a,
            self._files_b,
            bins=self._bins,
            top_k=self._top_k,
            on_column=self._on_column,
            should_cancel=self._cancel.is_set,
        )

    def _final_label(self, cancelled: bool) -> str:
        if cancelled:
            return f"Stopped — {len(self._rows)} column(s) compared."
        return "Comparison complete."
