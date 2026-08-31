"""pywebview-facing API layer.

Thin wrapper around data_loader.py — exposes methods pywebview binds to
`window.pywebview.api` in the frontend. Keep this file free of business
logic so data_loader.py stays importable/testable without a window.

Known Phase 0 limitation: date/datetime columns are not yet stringified,
so a source file containing them may fail to serialize across the js_api
bridge. Not a concern for the CSV/Parquet/Arrow test fixtures used here;
revisit if/when a dataset with date columns is opened.
"""
from __future__ import annotations

from pathlib import Path
from time import perf_counter
from typing import Any

import polars as pl
import webview

from ml_data_view.audit_log import (
    audit_log_file_path,
    record_file_open,
    record_folder_open,
    record_report_generated,
)
from ml_data_view.chart_build_job import ChartBuildJob
from ml_data_view.compare import COMPARE_SAMPLE_FILES, compare_schemas, evenly_spaced
from ml_data_view.compare_jobs import CompareJob
from ml_data_view.chart_data import build_chart_payload, classify_column
from ml_data_view.checkpoint_log import checkpoint
from ml_data_view.data_loader import (
    UnsupportedFileTypeError,
    _dispatch_scanner,
    _json_safe_preview_value,
    dataset_info_from_lazy,
    row_count,
    scan_dataframe,
)
from ml_data_view.deep_checks import DEEP_CHECKS, DeepContext
from ml_data_view.findings import INSTANT_CHECKS, run_instant_checks, severity_summary, sorted_findings
from ml_data_view.findings_job import DeepCheckJob
from ml_data_view.folder_load_job import FolderLoadJob
from ml_data_view.folder_loader import (
    FolderScanResult,
    discover_files,
    load_group,
    scan_folder,
)
from ml_data_view.image_data import resolve_image_full
from ml_data_view.load_stats import build_load_stats
from ml_data_view.report import available_sections, generate_report as write_report_pdf
from ml_data_view.search import (
    SEARCH_ROW_LIMIT,
    Condition,
    SearchConditionError,
    build_predicate,
    search_rows,
)
from ml_data_view.search_jobs import SearchCountJob, SearchExportJob
from ml_data_view.settings import (
    AppSettings,
    load_settings,
    remove_saved_search,
    save_settings,
    upsert_saved_search,
)


def _job_progress(job) -> dict[str, Any]:
    """Uniform snapshot for a background job the UI polls, including the
    idle case before one has been started."""
    if job is None:
        return {
            "stage": "idle",
            "label": "",
            "fraction": 0.0,
            "files_done": 0,
            "files_total": 0,
            "finished": False,
            "cancelled": False,
            "error": None,
            "result": None,
        }
    return job.progress()


class Api:
    """Methods exposed to the JS frontend via pywebview's js_api binding."""

    def __init__(self, settings_dir: Path | None = None) -> None:
        # A LazyFrame, not a DataFrame: nothing here is ever fully
        # materialized (see data_loader.py's module docstring and
        # PROJECT_BRIEF.md's architecture note) — every method below reads
        # through it via a bounded slice or a streaming aggregation, which
        # is what lets a dataset far larger than RAM get profiled at all.
        self._current_lazy: pl.LazyFrame | None = None
        self._current_path: str | None = None
        # Set explicitly by load_file() (the file's parent) and
        # load_folder_group() (the folder itself) — used to resolve
        # relative image paths (get_image_full/get_chart_data). Explicit
        # rather than derived from _current_path, since for a folder load
        # _current_path IS the directory already, not a file inside it.
        self._current_dataset_dir: Path | None = None
        # Cached by discover_folder() so load_folder_group() doesn't need
        # to re-walk/re-schema-scan the whole tree just to load the group
        # the user picked.
        self._folder_scan_result: FolderScanResult | None = None
        self._folder_path: str | None = None
        # The in-flight cancellable folder load, if any (start_folder_load).
        # `_folder_load_adopted` guards against re-adopting the same
        # finished result on every poll.
        self._folder_load_job: FolderLoadJob | None = None
        self._folder_load_adopted = False
        # Source files behind _current_lazy for a folder load; None for a
        # single-file open. Threaded into build_chart_payload so chart
        # computation stays per-file too.
        self._current_files: list[Path] | None = None
        # The in-flight chart build, if any (start_chart_build).
        self._chart_build_job: ChartBuildJob | None = None
        self._chart_build_adopted = False
        # The in-flight deep-check run, if any (start_deep_checks).
        self._deep_check_job: DeepCheckJob | None = None
        # Columns Phase 1 classified as "text" for the current dataset,
        # cached from the load rather than reclassified on demand.
        self._text_columns: list[str] = []
        # The predicate and column projection from the most recent
        # run_search(), reused by the count and export jobs so they can
        # never describe a different query than the rows on screen.
        self._search_predicate = None
        self._search_columns: list[str] | None = None
        self._search_conditions: list[dict[str, Any]] = []
        # When set, EVERY derived figure describes only matching rows —
        # charts, row count, null counts and preview alike. Kept separate
        # from _search_predicate so running a new search does not silently
        # change what the charts already on screen describe.
        self._chart_filter = None
        self._chart_filter_conditions: list[dict[str, Any]] = []
        self._match_count_job: SearchCountJob | None = None
        self._search_export_job: SearchExportJob | None = None
        # The in-flight dataset comparison, if any (start_comparison).
        self._compare_job: CompareJob | None = None
        # Computed once per load_file() (one streaming count) and reused by
        # everything below instead of each recounting — dataset_info,
        # get_chart_data, and (via _dataset_info) generate_report.
        self._current_row_count: int | None = None
        # Cached by load_file() so generate_report() doesn't need to
        # recompute it (another full streaming null-count scan) just to
        # rebuild the report's schema table.
        self._dataset_info: dict[str, Any] | None = None
        # Cached by get_chart_data() so Reports (get_report_sections(),
        # generate_report()) can reuse it instead of recomputing PCA/
        # thumbnails/etc — the frontend always calls get_chart_data() right
        # after a successful load_file(), before the user could reach
        # Reports, so by the time Reports needs it, it's already there.
        self._chart_data: dict[str, Any] | None = None
        # None means "use the real OS app-data directory" (settings.py's
        # default) — tests pass tmp_path here so they never touch it.
        self._settings_dir = settings_dir
        self._settings: AppSettings = load_settings(settings_dir)

    def open_file_dialog(self) -> str | None:
        """Show a native file-open dialog restricted to supported types.

        Returns the selected path, or None if the user cancelled. Requires
        an active pywebview window, so this is not unit-tested headlessly —
        verify manually per the Phase 0 plan's manual-verification task.
        """
        window = webview.windows[0]
        result = window.create_file_dialog(
            webview.FileDialog.OPEN,
            file_types=(
                "Dataset files (*.parquet;*.arrow;*.csv)",
                "All files (*.*)",
            ),
        )
        if not result:
            return None
        return result[0]

    def load_file(self, path: str) -> dict[str, Any]:
        """Scan a dataset file (lazily — see __init__), cache it, and
        return its profile as a dict.

        On failure, returns {"error": "<message>"} instead of raising, so
        the frontend always gets a plain dict back across the js_api bridge.
        Caches the LazyFrame (and its path, for get_chart_data()'s relative
        image-path resolution) on success so later calls can reuse it
        without re-scanning the file.

        Includes `text_columns` (Phase 1's classify_column, reused per
        PROJECT_BRIEF.md's Phase 3 detection note) alongside the schema/
        preview, computed here rather than as part of get_chart_data()'s
        payload, so the preview table can truncate long text values on its
        very first render instead of waiting for the chart-data round trip.
        """
        record_file_open(Path(path).name, self._settings, base_dir=self._settings_dir)
        # The clock starts at the real work, not at the audit write — the
        # stat line describes profiling the dataset, not bookkeeping.
        started_at = perf_counter()
        try:
            lf = scan_dataframe(path)
        except (UnsupportedFileTypeError, FileNotFoundError) as exc:
            return {"error": str(exc)}
        self._current_lazy = lf
        self._current_path = path
        self._current_dataset_dir = Path(path).parent
        self._chart_filter = None  # a new dataset never inherits a filter
        self._chart_filter_conditions = []
        self._search_predicate = None
        self._current_row_count = row_count(lf)  # one streaming count, shared below
        self._chart_data = None  # stale for the newly loaded dataset — get_chart_data() refills it

        dataset_info = dataset_info_from_lazy(lf, path, total_rows=self._current_row_count)
        self._dataset_info = dataset_info.to_dict()  # reused by generate_report(), no re-scan needed
        info = dataset_info.to_dict()  # a separate dict — text_columns below is UI-only, not part of it
        info["text_columns"] = [
            column for column in lf.collect_schema() if classify_column(lf, column) == "text"
        ]
        # Cached because the selection/detail surfaces need to know which
        # columns get the truncate-with-popup treatment, and classification
        # is a real query — not something to repeat per request.
        self._text_columns = list(info["text_columns"])
        info["source_type"] = "file"
        info["load_stats"] = build_load_stats(
            row_count=self._current_row_count,
            file_count=1,
            elapsed_seconds=perf_counter() - started_at,
        )
        return info

    def get_settings(self) -> dict[str, Any]:
        """Current settings plus the audit log's file path (Settings shows
        it so the user can find the file — it's not itself a setting)."""
        data = self._settings.to_dict()
        data["audit_log_path"] = str(audit_log_file_path(self._settings_dir))
        return data

    def save_settings(self, updates: dict[str, Any]) -> dict[str, Any]:
        """Merge `updates` onto the current settings, persist immediately,
        and return the same shape as get_settings() so the frontend can
        re-render from the confirmed, saved state."""
        merged = {**self._settings.to_dict(), **(updates or {})}
        self._settings = AppSettings.from_dict(merged)
        save_settings(self._settings, self._settings_dir)
        return self.get_settings()

    def _dataset_dir(self) -> Path | None:
        return self._current_dataset_dir

    def get_chart_data(self) -> dict[str, Any]:
        """Return Phase 1+2+3 chart data for the most recently loaded
        dataset, caching it for get_report_sections()/generate_report()."""
        if self._current_lazy is None:
            return {"error": "No dataset loaded yet. Open a file first."}
        checkpoint("chart_embedding_prep:start", f"row_count={self._current_row_count}")
        # Null counts from the already-built schema table feed the cheap
        # first pass of correlated-missingness detection (null_patterns.py)
        # instead of being recomputed.
        null_counts = {
            column["name"]: column["null_count"]
            for column in (self._dataset_info or {}).get("columns", [])
        }
        self._chart_data = build_chart_payload(
            self._current_lazy,
            total_rows=self._current_row_count,
            dataset_dir=self._dataset_dir(),
            null_counts=null_counts or None,
            # Per-file for folder loads: every chart builder that would
            # otherwise query the concatenated frame grows with file count.
            files=self._current_files or None,
        )
        checkpoint("chart_embedding_prep:done")
        return self._chart_data

    # --- chart building with progress -----------------------------------
    #
    # get_chart_data() above is the synchronous form, kept for headless use
    # and Reports. The frontend uses these two: chart generation runs at
    # ~0.54s per file, so on a large folder a single blocking js_api call
    # would leave the UI frozen for minutes. See chart_build_job.py.

    def start_chart_build(self) -> dict[str, Any]:
        """Begin building chart data on a worker thread; returns at once."""
        if self._current_lazy is None:
            return {"error": "No dataset loaded yet. Open a file or folder first."}
        null_counts = {
            column["name"]: column["null_count"]
            for column in (self._dataset_info or {}).get("columns", [])
        }
        self._chart_build_job = ChartBuildJob(
            self._current_lazy,
            total_rows=self._current_row_count,
            dataset_dir=self._dataset_dir(),
            null_counts=null_counts or None,
            files=self._current_files or None,
            predicate=self._chart_filter,
        )
        self._chart_build_adopted = False
        self._chart_build_job.start()
        return {"started": True}

    def get_chart_build_progress(self) -> dict[str, Any]:
        """Snapshot of the running chart build. On completion the payload
        is cached exactly as get_chart_data() would have cached it, so
        Reports keeps working unchanged."""
        job = self._chart_build_job
        if job is None:
            return {
                "stage": "idle",
                "label": "",
                "fraction": 0.0,
                "finished": False,
                "error": None,
                "result": None,
            }
        state = job.progress()
        if state["finished"] and state["result"] is not None and not self._chart_build_adopted:
            self._chart_data = state["result"]
            self._chart_build_adopted = True
        return state

    def wait_for_chart_build(self, timeout: float | None = None) -> dict[str, Any]:
        """Block until the build settles — for headless callers and tests."""
        if self._chart_build_job is None:
            return self.get_chart_build_progress()
        self._chart_build_job.wait(timeout)
        return self.get_chart_build_progress()

    def get_image_full(self, column: str, source: str, row_index: int) -> dict[str, Any]:
        """Resolve one row's image at a larger size, for the preview grid's
        "view larger" click (see image_data.resolve_image_full)."""
        if self._current_lazy is None:
            return {"error": "No dataset loaded yet. Open a file first."}
        return resolve_image_full(
            self._current_lazy, column, source, row_index, dataset_dir=self._dataset_dir()
        )

    # --- row search ------------------------------------------------------

    def _search_sources(self) -> list[Path]:
        """The files a search should scan. A folder load has its file list;
        a single-file open is a one-element list, so both go down the same
        per-file path."""
        if self._current_files:
            return list(self._current_files)
        return [Path(self._current_path)] if self._current_path else []

    def _build_search_predicate(self, conditions: list[dict[str, Any]]):
        parsed = [Condition.from_dict(c) for c in conditions or []]
        return build_predicate(parsed, self._current_lazy.collect_schema())

    def run_search(
        self,
        conditions: list[dict[str, Any]],
        columns: list[str] | None = None,
        limit: int = SEARCH_ROW_LIMIT,
    ) -> dict[str, Any]:
        """Matching rows, capped at `limit`. Fast enough to run inline —
        see search.py. A bad condition comes back with its index so the UI
        can mark that row rather than showing a generic failure."""
        if self._current_lazy is None:
            return {"error": "No dataset loaded yet. Open a file or folder first."}
        try:
            predicate = self._build_search_predicate(conditions)
        except SearchConditionError as exc:
            return {"error": str(exc), "condition_index": exc.condition_index}

        self._search_predicate = predicate
        self._search_conditions = list(conditions or [])
        self._search_columns = list(columns) if columns else None
        rows = search_rows(
            self._search_sources(), predicate, columns=self._search_columns, limit=limit
        )
        return {
            "rows": [
                {key: _json_safe_preview_value(value) for key, value in row.items()}
                for row in rows
            ],
            "columns": self._search_columns or list(self._current_lazy.collect_schema()),
            "limit": limit,
            "truncated": len(rows) >= limit,
        }

    def start_match_count(self) -> dict[str, Any]:
        """Count every match for the last run_search()'s conditions. Split
        out because it cannot short-circuit: a string filter has no parquet
        statistics to skip row groups with, so it scans every file."""
        if self._search_predicate is None:
            return {"error": "Run a search first."}
        self._match_count_job = SearchCountJob(self._search_sources(), self._search_predicate)
        self._match_count_job.start()
        return {"started": True}

    def get_match_count_progress(self) -> dict[str, Any]:
        return _job_progress(self._match_count_job)

    def wait_for_match_count(self, timeout: float | None = None) -> dict[str, Any]:
        """Block until the count settles — for headless callers and tests;
        the UI polls get_match_count_progress() instead."""
        if self._match_count_job is not None:
            self._match_count_job.wait(timeout)
        return self.get_match_count_progress()

    def cancel_match_count(self) -> dict[str, Any]:
        if self._match_count_job is None:
            return {"error": "No count in progress."}
        self._match_count_job.cancel()
        return {"cancelled": True}

    def search_export_dialog(self, default_filename: str = "search_results.csv") -> str | None:
        window = webview.windows[0]
        result = window.create_file_dialog(
            webview.FileDialog.SAVE, save_filename=default_filename, file_types=("CSV (*.csv)",)
        )
        if not result:
            return None
        return result if isinstance(result, str) else result[0]

    def start_search_export(self, path: str) -> dict[str, Any]:
        if self._search_predicate is None:
            return {"error": "Run a search first."}
        self._search_export_job = SearchExportJob(
            self._search_sources(), self._search_predicate, self._search_columns, path
        )
        self._search_export_job.start()
        return {"started": True}

    def get_search_export_progress(self) -> dict[str, Any]:
        return _job_progress(self._search_export_job)

    def wait_for_search_export(self, timeout: float | None = None) -> dict[str, Any]:
        """Block until the export settles — for headless callers and tests."""
        if self._search_export_job is not None:
            self._search_export_job.wait(timeout)
        return self.get_search_export_progress()

    def cancel_search_export(self) -> dict[str, Any]:
        if self._search_export_job is None:
            return {"error": "No export in progress."}
        self._search_export_job.cancel()
        return {"cancelled": True}

    def apply_chart_filter(self) -> dict[str, Any]:
        """Make every derived figure describe only the current search's rows.

        Deliberately an explicit action rather than something run_search()
        triggers: a search returns in milliseconds, while rebuilding charts
        takes minutes on a large folder, so filtering must not silently
        start that work on every keystroke.

        Recomputes dataset info (row count, null counts, preview) here and
        returns it; the frontend then restarts the chart build, which picks
        the same predicate up from _chart_filter.
        """
        if self._search_predicate is None:
            return {"error": "Run a search first."}
        self._chart_filter = self._search_predicate
        self._chart_filter_conditions = list(self._search_conditions)
        return self._rebuild_dataset_info()

    def clear_chart_filter(self) -> dict[str, Any]:
        """Drop the filter so everything describes the whole dataset again."""
        self._chart_filter = None
        self._chart_filter_conditions = []
        return self._rebuild_dataset_info()

    def _rebuild_dataset_info(self) -> dict[str, Any]:
        lf = self._current_lazy
        files = self._current_files or (
            [Path(self._current_path)] if self._current_path else None
        )
        info = dataset_info_from_lazy(
            lf,
            self._current_path,
            files=files,
            predicate=self._chart_filter,
            file_size_bytes=self._dataset_info.get("file_size_bytes")
            if self._dataset_info
            else None,
        )
        self._dataset_info = info.to_dict()
        self._current_row_count = info.row_count
        self._chart_data = None  # stale under a different filter
        payload = info.to_dict()
        payload["filter_active"] = self._chart_filter is not None
        payload["filter_conditions"] = self._chart_filter_conditions
        payload["source_type"] = "folder" if self._current_files else "file"
        return payload

    def get_chart_filter(self) -> dict[str, Any]:
        return {
            "filter_active": self._chart_filter is not None,
            "filter_conditions": self._chart_filter_conditions,
        }

    def list_saved_searches(self) -> list[dict[str, Any]]:
        return list(self._settings.saved_searches)

    def save_search(
        self, name: str, conditions: list[dict[str, Any]], columns: list[str] | None = None
    ) -> dict[str, Any]:
        try:
            self._settings = upsert_saved_search(self._settings, name, conditions, columns)
        except ValueError as exc:
            return {"error": str(exc)}
        save_settings(self._settings, self._settings_dir)
        return {"saved_searches": self.list_saved_searches()}

    def delete_saved_search(self, name: str) -> dict[str, Any]:
        self._settings = remove_saved_search(self._settings, name)
        save_settings(self._settings, self._settings_dir)
        return {"saved_searches": self.list_saved_searches()}

    # --- dataset comparison ---------------------------------------------

    def compare_picker_dialog(self, kind: str = "folder") -> str | None:
        """Native picker for the other side of a comparison."""
        window = webview.windows[0]
        if kind == "folder":
            result = window.create_file_dialog(webview.FileDialog.FOLDER)
        else:
            result = window.create_file_dialog(
                webview.FileDialog.OPEN,
                file_types=("Dataset files (*.parquet;*.arrow;*.csv)", "All files (*.*)"),
            )
        if not result:
            return None
        return result if isinstance(result, str) else result[0]

    def _comparison_files(self, path: str, sample_files: int | None) -> list[Path]:
        """The file list for one side, sampled if it is a large folder.

        Sampling is evenly spaced across the folder, never the first N —
        these folders are day-partitioned, so a head sample would compare
        early-in-the-day against early-in-the-day (see compare.evenly_spaced).
        """
        target = Path(path)
        files = discover_files(target) if target.is_dir() else [target]
        if sample_files and len(files) > sample_files:
            return evenly_spaced(files, sample_files)
        return files

    def start_comparison(
        self, other_path: str, sample_files: int | None = COMPARE_SAMPLE_FILES
    ) -> dict[str, Any]:
        """Compare the loaded dataset (side A) against `other_path` (side B).

        The schema diff is returned immediately in this result — it reads
        footer metadata only — so the UI can show it while the distribution
        comparison runs in the background.
        """
        if self._current_lazy is None:
            return {"error": "No dataset loaded yet. Open a file or folder first."}
        files_a = self._comparison_files(self._current_path, sample_files)
        try:
            files_b = self._comparison_files(other_path, sample_files)
            if not files_b:
                return {"error": "No dataset files found at that path."}
            schema_b = _dispatch_scanner(files_b[0], files_b[0].suffix.lower()).collect_schema()
        except Exception as exc:
            return {"error": f"Could not read that dataset: {exc}"}

        schema_diff = compare_schemas(self._current_lazy.collect_schema(), schema_b)
        self._compare_job = CompareJob(files_a, files_b)
        self._compare_job.start()
        return {
            "started": True,
            "schema": schema_diff,
            "files_a": len(files_a),
            "files_b": len(files_b),
            "sampled": bool(sample_files),
        }

    def get_comparison_progress(self) -> dict[str, Any]:
        return _job_progress(self._compare_job)

    def cancel_comparison(self) -> dict[str, Any]:
        if self._compare_job is None:
            return {"error": "No comparison in progress."}
        self._compare_job.cancel()
        return {"cancelled": True}

    def wait_for_comparison(self, timeout: float | None = None) -> dict[str, Any]:
        """Block until the comparison settles — for headless callers/tests."""
        if self._compare_job is not None:
            self._compare_job.wait(timeout)
        return self.get_comparison_progress()

    # --- Findings --------------------------------------------------------

    def list_checks(self) -> dict[str, Any]:
        """The full check catalogue, instant and deep.

        Deliberately independent of a loaded dataset: the tab paints every
        check name with a pending marker BEFORE running anything, which is
        what makes it read as a scan rather than a list that appears all at
        once.
        """
        return {
            "checks": [
                {"id": spec.id, "title": spec.title, "kind": spec.kind}
                for spec in (*INSTANT_CHECKS, *DEEP_CHECKS)
            ]
        }

    def run_instant_checks(self) -> dict[str, Any]:
        """Run every instant check over data already computed — no reads.

        Returns {"pending": "charts"} when the chart payload isn't built
        yet, rather than a clean result. The checks read that payload, so
        "nothing found" before it exists would mean "nothing looked",
        presented as reassurance.
        """
        if self._current_lazy is None:
            return {"error": "No dataset loaded yet. Open a file or folder first."}
        if not self._chart_data:
            return {"pending": "charts"}

        results = run_instant_checks(self._dataset_info or {}, self._chart_data)
        return {
            "checks": results,
            "findings": [finding.to_dict() for finding in sorted_findings(results)],
            "summary": severity_summary(sorted_findings(results)),
        }

    def _deep_context(self) -> DeepContext:
        return DeepContext(
            lf=self._current_lazy,
            files=list(self._current_files) if self._current_files else None,
            dataset_dir=self._dataset_dir(),
            image_columns=list((self._chart_data or {}).get("image_columns") or []),
        )

    def start_deep_checks(self) -> dict[str, Any]:
        """Begin the deep checks on a worker thread; returns at once."""
        if self._current_lazy is None:
            return {"error": "No dataset loaded yet. Open a file or folder first."}
        self._deep_check_job = DeepCheckJob(self._deep_context())
        self._deep_check_job.start()
        return {"started": True}

    def get_deep_check_progress(self) -> dict[str, Any]:
        """Snapshot of the running deep checks, including per-check status
        so completed checks render while later ones still run."""
        if self._deep_check_job is None:
            return {**dict(self._IDLE_PROGRESS), "checks": []}
        return self._deep_check_job.progress()

    def cancel_deep_checks(self) -> dict[str, Any]:
        """Stop at the next check boundary. Completed checks are kept."""
        if self._deep_check_job is None:
            return {"error": "No deep checks in progress."}
        self._deep_check_job.cancel()
        return {"cancelled": True}

    def wait_for_deep_checks(self, timeout: float | None = None) -> dict[str, Any]:
        """Block until the run settles — for headless callers and tests."""
        if self._deep_check_job is None:
            return self.get_deep_check_progress()
        self._deep_check_job.wait(timeout)
        return self.get_deep_check_progress()

    def _banner_settings(self) -> dict[str, Any]:
        return {
            "enabled": self._settings.classification_banner_enabled,
            "label": self._settings.classification_banner_label,
            "color": self._settings.classification_banner_color,
        }

    def get_report_sections(self) -> dict[str, Any]:
        """Which Reports checkboxes to offer for the currently loaded
        dataset (see report.available_sections)."""
        if self._current_lazy is None or self._chart_data is None:
            return {"error": "No dataset loaded yet. Open a file first."}
        return available_sections(self._chart_data)

    def save_report_dialog(self, default_filename: str = "report.pdf") -> str | None:
        """Show a native save-file dialog for choosing where to write the
        PDF report. Requires an active pywebview window, so — like
        open_file_dialog() — this isn't unit-tested headlessly."""
        window = webview.windows[0]
        result = window.create_file_dialog(
            webview.FileDialog.SAVE,
            save_filename=default_filename,
            file_types=("PDF files (*.pdf)", "All files (*.*)"),
        )
        if not result:
            return None
        return result[0]

    def generate_report(self, output_path: str, sections: list[str]) -> dict[str, Any]:
        """Write a PDF report for the currently loaded dataset to
        `output_path`, including whichever of `sections` are both
        requested and actually available. Audit-logs a "report_generated"
        action the same way load_file() logs "file_open"."""
        if (
            self._current_lazy is None
            or self._current_path is None
            or self._chart_data is None
            or self._dataset_info is None
        ):
            return {"error": "No dataset loaded yet. Open a file first."}
        if not output_path:
            return {"error": "No output path given."}

        try:
            result_path = write_report_pdf(
                output_path,
                dataset_info=self._dataset_info,
                chart_data=self._chart_data,
                sections=sections,
                banner=self._banner_settings(),
            )
        except OSError as exc:
            return {"error": f"Could not write report: {exc}"}

        record_report_generated(
            Path(self._current_path).name, self._settings, base_dir=self._settings_dir
        )
        return {"path": str(result_path)}

    def open_folder_dialog(self) -> str | None:
        """Show a native folder-picker dialog (Stage 2: folder loading).

        Returns the selected folder path, or None if the user cancelled.
        Requires an active pywebview window, so — like open_file_dialog()
        — this isn't unit-tested headlessly.
        """
        window = webview.windows[0]
        result = window.create_file_dialog(webview.FileDialog.FOLDER)
        if not result:
            return None
        return result[0]

    def discover_folder(self, path: str) -> dict[str, Any]:
        """Recursively discover dataset files under `path`, group them by
        exact schema match, and cache the result for load_folder_group().

        Returns {"error": ...} if nothing loadable was found, otherwise
        {"groups": [...], "unreadable": [...], "default_group_index": 0,
        "excluded": [...]} — `excluded` is everything not in the default
        group (other groups' files + unreadable files, each with a short
        reason), ready to show immediately; the frontend recomputes it
        client-side (same data, no round trip) if the user picks a
        different group than the default.
        """
        result = scan_folder(path)
        if not result.groups:
            return {"error": "No dataset files found in this folder."}
        self._folder_scan_result = result
        self._folder_path = path
        data = result.to_dict()
        data["excluded"] = result.excluded_from(result.default_group_index)
        return data

    # --- cancellable folder loading -------------------------------------
    #
    # load_folder_group() below is the synchronous form, kept for direct/
    # headless use. The frontend uses these three instead: a ~108-second
    # js_api call would leave the UI with no way to ask anything else in
    # the meantime, including "stop". See folder_load_job.py.

    _IDLE_PROGRESS = {
        "stage": "idle",
        "label": "",
        "fraction": 0.0,
        "files_done": 0,
        "files_total": 0,
        "finished": False,
        "cancelled": False,
        "error": None,
        "result": None,
    }

    def start_folder_load(self, group_index: int) -> dict[str, Any]:
        """Begin loading a schema group on a worker thread and return at
        once. Poll get_folder_load_progress(); stop with
        cancel_folder_load()."""
        if self._folder_scan_result is None:
            return {"error": "No folder scanned yet. Open a folder first."}
        groups = self._folder_scan_result.groups
        if group_index < 0 or group_index >= len(groups):
            return {"error": "Invalid group selection."}

        group = groups[group_index]
        folder_path = Path(self._folder_path)
        record_folder_open(folder_path.name, self._settings, base_dir=self._settings_dir)

        self._folder_load_job = FolderLoadJob(group, folder_path)
        self._folder_load_adopted = False
        self._folder_load_job.start()
        return {"started": True, "files_total": group.file_count}

    def get_folder_load_progress(self) -> dict[str, Any]:
        """A JSON-safe snapshot of the running load. Once it finishes with
        a result, that result is adopted as the current dataset (exactly
        what the synchronous load_folder_group() would have left behind),
        so get_chart_data() and Reports work unchanged."""
        job = self._folder_load_job
        if job is None:
            return dict(self._IDLE_PROGRESS)

        state = job.progress()
        if state["finished"] and state["result"] is not None and not self._folder_load_adopted:
            self._adopt_folder_load(job, state["result"])
            self._folder_load_adopted = True
        return state

    def cancel_folder_load(self) -> dict[str, Any]:
        """Ask the running load to stop at the next file/column boundary.
        Whatever finished completely is kept; see folder_load_job.py."""
        if self._folder_load_job is None:
            return {"error": "No folder load in progress."}
        self._folder_load_job.cancel()
        return {"cancelled": True}

    def wait_for_folder_load(self, timeout: float | None = None) -> dict[str, Any]:
        """Block until the load settles — for headless callers and tests;
        the UI polls get_folder_load_progress() instead."""
        if self._folder_load_job is None:
            return dict(self._IDLE_PROGRESS)
        self._folder_load_job.wait(timeout)
        return self.get_folder_load_progress()

    def _adopt_folder_load(self, job: FolderLoadJob, result: dict[str, Any]) -> None:
        self._current_lazy = job.result_lazy
        self._current_path = str(self._folder_path)
        self._current_dataset_dir = Path(self._folder_path)
        self._current_row_count = result["row_count"]
        self._current_files = list(job.used_files)
        self._chart_filter = None  # a new dataset never inherits a filter
        self._chart_filter_conditions = []
        self._search_predicate = None
        self._dataset_info = result
        self._text_columns = list(result.get("text_columns") or [])
        self._chart_data = None  # stale for the newly loaded dataset

    def load_folder_group(self, group_index: int) -> dict[str, Any]:
        """Load the schema group at `group_index` from the most recently
        discover_folder()'d folder — combining its files into one
        LazyFrame via folder_loader.load_group() (pl.concat(...,
        how="vertical"), lazy) and handing it to exactly the same caching/
        profiling path load_file() uses, unchanged. Audit-logs a
        "folder_open" action the same way load_file() logs "file_open".
        """
        checkpoint("group_selection:start", f"group_index={group_index}")
        started_at = perf_counter()
        if self._folder_scan_result is None:
            return {"error": "No folder scanned yet. Open a folder first."}
        groups = self._folder_scan_result.groups
        if group_index < 0 or group_index >= len(groups):
            return {"error": "Invalid group selection."}
        group = groups[group_index]
        folder_path = Path(self._folder_path)

        record_folder_open(folder_path.name, self._settings, base_dir=self._settings_dir)

        checkpoint("group_selection:concat_lazyframe", f"file_count={group.file_count}")
        lf = load_group(group)
        self._current_lazy = lf
        self._current_path = str(folder_path)
        self._current_dataset_dir = folder_path

        if group.row_count is not None:
            self._current_row_count = group.row_count
            checkpoint("group_selection:row_count_from_metadata", f"row_count={self._current_row_count}")
        else:
            checkpoint("group_selection:row_count_streaming_start")
            self._current_row_count = row_count(lf)
            checkpoint("group_selection:row_count_streaming_done", f"row_count={self._current_row_count}")
        self._chart_data = None

        total_size = sum(f.stat().st_size for f in group.files)
        checkpoint("preview_generation:start", f"row_count={self._current_row_count} total_bytes={total_size}")
        dataset_info = dataset_info_from_lazy(
            lf,
            folder_path,
            total_rows=self._current_row_count,
            file_size_bytes=total_size,
            files=group.files,
        )
        checkpoint("preview_generation:done")
        self._dataset_info = dataset_info.to_dict()
        info = dataset_info.to_dict()
        # files= keeps classification per-file: querying the concatenated
        # folder LazyFrame costs ~28-42MB per file and blew a 2GB cap at
        # 100 files on a single column (see _classify_string_column_per_file).
        info["text_columns"] = [
            column
            for column in lf.collect_schema()
            if classify_column(lf, column, files=group.files) == "text"
        ]
        self._text_columns = list(info["text_columns"])
        info["source_type"] = "folder"
        info["file_count"] = group.file_count
        info["load_stats"] = build_load_stats(
            row_count=self._current_row_count,
            file_count=group.file_count,
            elapsed_seconds=perf_counter() - started_at,
        )
        checkpoint("group_selection:done")
        return info
