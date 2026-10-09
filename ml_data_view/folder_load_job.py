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

from ml_data_view.data_loader import dataset_info_from_lazy
from ml_data_view.file_profile import load_folder_profile
from ml_data_view.folder_loader import SchemaGroup, load_files
from ml_data_view.load_stats import build_load_stats
from ml_data_view.profile_cache import MemoryProfileCache, ProfileCache

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

    def __init__(
        self,
        group: SchemaGroup,
        folder_path: Path,
        cache_path: str | Path | None = None,
        identity: list[dict] | None = None,
        timestamp_column: str | None = None,
    ) -> None:
        self._group = group
        self._folder_path = Path(folder_path)
        # Per-chunk derived-column values, when the folder has a detected
        # directory structure (hierarchy.py). Present, the loaded frame
        # carries that structure's derived columns — their names come from
        # the mapping, so nothing here can list them. Absent, this is an
        # ordinary flat folder load and nothing changes.
        self._identity = identity
        # Which column the structure mapping chose as "when". Passed down
        # so each chunk's cached profile carries its own first/last
        # timestamp — see file_profile._time_bounds.
        self._timestamp_column = timestamp_column
        # Where per-file profiles are remembered between opens. None means
        # "do not persist" — the load still works, it just reads every file
        # again, which is what a headless caller or a test gets unless it
        # asks for a cache.
        self._cache_path = cache_path
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
            "rows_done": None,
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

    def _frame(self, used_files: list[Path]):
        """The frame the app will hold.

        With a structure, each chunk's scan carries its derived columns —
        lazy literals, so they cost nothing and are known from the path
        alone. Without one, this is exactly the concat it always was.
        A truncated (stopped) load narrows the identity list to match, so
        the frame never references a chunk the counts do not cover.
        """
        if not self._identity:
            return load_files(used_files)
        from ml_data_view.hierarchy import scope_lazyframe

        wanted = {str(path) for path in used_files}
        narrowed = [row for row in self._identity if row["file"] in wanted]
        return scope_lazyframe(narrowed) if narrowed else load_files(used_files)

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

        # --- phase 1: profile each file, from cache where possible ---
        #
        # A file that has been profiled before and has not changed since
        # contributes its counts without being read at all (see
        # file_profile.load_folder_profile). A file that has not is scanned
        # exactly once, and what that scan learned is stored — so the chart
        # build that follows, and every later re-open, start from it.
        self._update(stage="null_counts", label=f"Counting missing values — file 1 of {total}")
        schema = pl.scan_parquet(files[0]).collect_schema()

        rows_seen = 0

        def on_file(index: int, path: Path) -> None:
            nonlocal rows_seen
            done = index + 1
            # Rows are counted from the parquet footer of each finished
            # file, not from the data — so "42.1M rows processed" in the
            # jobs indicator costs a metadata read per file and nothing
            # more. Unknown (and omitted) for formats with no such footer.
            rows_seen += _footer_rows(path)
            self._update(
                files_done=done,
                rows_done=rows_seen or None,
                fraction=round((done / total) * NULL_COUNT_WEIGHT, 4),
                label=f"Counting missing values — file {done} of {total}",
            )

        cache = (
            ProfileCache(self._cache_path)
            if self._cache_path is not None
            else MemoryProfileCache()
        )
        try:
            folder = load_folder_profile(
                files,
                schema,
                cache,
                on_file=on_file,
                should_cancel=self._cancelled,
                timestamp_column=self._timestamp_column,
            )
        finally:
            cache.close()

        if self._cancelled() and not folder.profiles:
            self._update(stage="cancelled", label="Cancelled before any file finished.", result=None)
            return

        # Truncate to whole files only — a stopped load is a smaller
        # dataset, never a partly-counted one. load_folder_profile checks
        # for cancellation at a file boundary, so `folder.profiles` is
        # already exactly the files that finished.
        used_files = [profile.path for profile in folder.profiles]
        partial = len(used_files) < total
        counts = folder.merged.missing_counts

        lf = self._frame(used_files)
        row_count = folder.merged.row_count
        self.result_lazy = lf
        self.used_files = list(used_files)

        # Derived columns are literals attached per chunk, so they are
        # never null — stated here rather than counted, because counting a
        # constant is a scan for an answer already known.
        for name in _derived_names(lf, counts):
            counts.setdefault(name, 0)

        info = dataset_info_from_lazy(
            lf,
            self._folder_path,
            total_rows=row_count,
            file_size_bytes=sum(f.stat().st_size for f in used_files),
            files=used_files,
            null_counts=counts,
        )
        payload = info.to_dict()

        # --- phase 2: classification ---
        #
        # No longer a per-file scan of its own: the distinct/row scalars it
        # needs came back with each file's profile in phase 1, so this is
        # arithmetic over numbers already in hand. The phase is still
        # reported per column — it is genuinely near-instant now, and a bar
        # that jumps because the work got cheaper is honest.
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
            if folder.merged.classification.get(column, ("unsupported", None))[0] == "text":
                text_columns.append(column)

        payload["text_columns"] = text_columns
        # NaN counts are kept apart from nulls here even though the schema
        # table shows them together as "missing": they have different
        # causes, and a NaN that arrived from a broken upstream computation
        # is worth being able to see on its own.
        payload["null_counts_only"] = dict(folder.merged.null_counts)
        payload["nan_counts"] = dict(folder.merged.nan_counts)
        payload["files_from_cache"] = folder.files_from_cache
        payload["files_scanned"] = folder.files_scanned
        payload["source_type"] = "folder"
        payload["file_count"] = len(used_files)
        payload["files_used"] = len(used_files)
        # The actual paths, not just the count: when this job runs in a
        # worker subprocess (jobs.py) the parent adopts the dataset from
        # this list, and after a stop it must be the files that finished
        # rather than the group's full set.
        payload["used_files"] = [str(f) for f in used_files]
        payload["files_total"] = total
        payload["partial"] = partial or self._cancelled()
        # Counts describe what was actually loaded, so a stopped load's
        # stat line agrees with the partial banner instead of quoting the
        # folder's full size for a dataset that is smaller than that.
        payload["load_stats"] = build_load_stats(
            row_count=row_count,
            file_count=len(used_files),
            elapsed_seconds=perf_counter() - started_at,
            files_from_cache=folder.files_from_cache,
            files_scanned=folder.files_scanned,
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


def _derived_names(lf, counts: dict[str, int]) -> list[str]:
    """Columns on the frame that no file contains — the derived ones."""
    return [name for name in lf.collect_schema() if name not in counts]


def _footer_rows(path: Path) -> int:
    """One file's row count from its parquet footer, or 0 when the format
    has no metadata-only count (.csv). Never a scan."""
    try:
        import pyarrow.parquet as pq

        return int(pq.ParquetFile(path).metadata.num_rows)
    except Exception:
        return 0


def _row_count_from_metadata(files: list[Path]) -> int:
    """Row count for exactly `files`, from parquet footers — never a scan.

    Recomputed for the truncated set rather than reusing the group's
    pre-scan total, which covers every file and would overstate a
    stopped load. The load itself now gets this by summing each file's
    cached per-file count (which is the same footer read, done once and
    remembered); this remains for callers that have only a file list.
    """
    import pyarrow.parquet as pq

    return sum(pq.ParquetFile(f).metadata.num_rows for f in files)
