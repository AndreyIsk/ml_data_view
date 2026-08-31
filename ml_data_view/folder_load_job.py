"""Cancellable, progress-reporting folder load.

Runs the folder-open pipeline on a worker thread so the pywebview js_api
bridge is never blocked: `start()` returns at once, the UI polls
`progress()`, and `cancel()` asks the worker to stop. Nothing here
imports webview — the whole job is plain Python and is tested headless.

Why a worker thread and polling, rather than the API method doing the
work and pushing updates: a js_api call that runs for ~108 seconds gives
the frontend no way to ask anything else in the meantime (including
"stop"), and pushing progress out with window.evaluate_js would couple
this module to JS function names and need a real window to test.

Cancellation is only ever observed at an item boundary — between files
while counting nulls, between columns while classifying. That is what
lets a stopped load be presented as a SMALLER dataset rather than an
incomplete one: see _run's truncation, and the matching guarantee tested
in test_cancelled_counts_match_a_direct_load_of_the_same_files.
"""
from __future__ import annotations

import threading
from pathlib import Path
from time import perf_counter
from typing import Any

import polars as pl

from ml_data_view.chart_data import classify_column
from ml_data_view.data_loader import _null_counts, dataset_info_from_lazy
from ml_data_view.folder_loader import SchemaGroup
from ml_data_view.load_stats import build_load_stats

# Share of the progress bar given to each phase, from a measured 250-file
# load: 83.3s counting nulls, 24.0s classifying (discovery was 0.3s and is
# folded into the null-count phase rather than given a visible slice).
# Weighting matters — a naive "files done / files total" bar would sit at
# 100% for the entire classification phase.
NULL_COUNT_WEIGHT = 0.78
CLASSIFY_WEIGHT = 0.22


class FolderLoadJob:
    """One folder load, running on its own thread.

    Poll `progress()` for a JSON-safe snapshot; call `cancel()` to stop at
    the next item boundary. A job is single-use — construct a new one per
    load.
    """

    def __init__(self, group: SchemaGroup, folder_path: Path) -> None:
        self._group = group
        self._folder_path = Path(folder_path)
        self._lock = threading.Lock()
        self._cancel = threading.Event()
        self._thread: threading.Thread | None = None
        self._done = threading.Event()
        # Set once the load settles, so the caller can adopt the same
        # frame/file list the reported counts describe rather than
        # rebuilding it and risking a mismatch after a truncation.
        self.result_lazy: pl.LazyFrame | None = None
        self.used_files: list[Path] = []
        self._state: dict[str, Any] = {
            "stage": "pending",
            "label": "Waiting to start…",
            "fraction": 0.0,
            "files_done": 0,
            "files_total": group.file_count,
            "finished": False,
            "cancelled": False,
            "error": None,
            "result": None,
        }
        # Test seam: called with a snapshot after every state update, so a
        # test can cancel at a precise point without sleeping/racing.
        self._on_progress_for_test = None

    # --- public API -----------------------------------------------------

    def start(self) -> None:
        if self._thread is not None:
            raise RuntimeError("This FolderLoadJob has already been started.")
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def cancel(self) -> None:
        self._cancel.set()

    def wait(self, timeout: float | None = None) -> bool:
        return self._done.wait(timeout)

    def progress(self) -> dict[str, Any]:
        with self._lock:
            return dict(self._state)

    # --- internals ------------------------------------------------------

    def _update(self, **changes: Any) -> None:
        with self._lock:
            self._state.update(changes)
            snapshot = dict(self._state)
        if self._on_progress_for_test is not None:
            self._on_progress_for_test(snapshot)

    def _cancelled(self) -> bool:
        return self._cancel.is_set()

    def _run(self) -> None:
        try:
            self._load()
        except Exception as exc:  # a worker thread must never die silently
            self._update(stage="error", label=f"Load failed: {exc}", error=str(exc), result=None)
        finally:
            self._update(finished=True, cancelled=self._cancelled())
            self._done.set()

    def _load(self) -> None:
        started_at = perf_counter()
        files = self._group.files
        total = len(files)

        if self._cancelled():
            self._update(stage="cancelled", label="Cancelled before loading started.", result=None)
            return

        # --- phase 1: null counts, one file at a time ---
        self._update(stage="null_counts", label=f"Counting missing values — file 1 of {total}")
        schema = pl.scan_parquet(files[0]).collect_schema()

        def on_file(index: int, path: Path) -> None:
            done = index + 1
            self._update(
                files_done=done,
                fraction=round((done / total) * NULL_COUNT_WEIGHT, 4),
                label=f"Counting missing values — file {done} of {total}",
            )

        lf_all = pl.concat([pl.scan_parquet(f) for f in files], how="vertical")
        counts = _null_counts(
            lf_all, schema, files=files, on_file=on_file, should_cancel=self._cancelled
        )

        files_done = self.progress()["files_done"]
        if self._cancelled() and files_done == 0:
            self._update(stage="cancelled", label="Cancelled before any file finished.", result=None)
            return

        # Truncate to whole files only — a stopped load is a smaller
        # dataset, never a partly-counted one.
        used_files = files if not self._cancelled() else files[:files_done]
        partial = len(used_files) < total

        lf = pl.concat([pl.scan_parquet(f) for f in used_files], how="vertical")
        row_count = _row_count_from_metadata(used_files)
        self.result_lazy = lf
        self.used_files = list(used_files)

        info = dataset_info_from_lazy(
            lf,
            self._folder_path,
            total_rows=row_count,
            file_size_bytes=sum(f.stat().st_size for f in used_files),
            files=used_files,
            null_counts=counts,
        )
        payload = info.to_dict()

        # --- phase 2: classification, one column at a time ---
        columns = list(schema)
        text_columns: list[str] = []
        for index, column in enumerate(columns):
            if self._cancelled():
                break
            self._update(
                stage="classify",
                label=f"Classifying columns — {index + 1} of {len(columns)}",
                fraction=round(
                    NULL_COUNT_WEIGHT + ((index + 1) / len(columns)) * CLASSIFY_WEIGHT, 4
                ),
            )
            if classify_column(lf, column, files=used_files) == "text":
                text_columns.append(column)

        payload["text_columns"] = text_columns
        payload["source_type"] = "folder"
        payload["file_count"] = len(used_files)
        payload["files_used"] = len(used_files)
        payload["files_total"] = total
        payload["partial"] = partial or self._cancelled()
        # Counts describe what was actually loaded, so a stopped load's
        # stat line agrees with the partial banner instead of quoting the
        # folder's full size for a dataset that is smaller than that.
        payload["load_stats"] = build_load_stats(
            row_count=row_count,
            file_count=len(used_files),
            elapsed_seconds=perf_counter() - started_at,
        )

        # A cancelled load keeps the fraction it reached. Snapping the bar
        # to 100% on Stop reads as "finished successfully", which is the
        # opposite of what happened — the bar should show how far it got.
        cancelled = self._cancelled()
        self._update(
            stage="done",
            label="Stopped — showing partial results." if cancelled else "Loaded.",
            fraction=self.progress()["fraction"] if cancelled else 1.0,
            result=payload,
        )


def _row_count_from_metadata(files: list[Path]) -> int:
    """Row count for exactly `files`, from parquet footers — never a scan.

    Recomputed for the truncated set rather than reusing the group's
    pre-scan total, which covers every file and would overstate a
    stopped load.
    """
    import pyarrow.parquet as pq

    return sum(pq.ParquetFile(f).metadata.num_rows for f in files)
