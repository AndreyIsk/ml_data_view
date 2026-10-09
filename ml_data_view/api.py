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
    record_reprofile,
    record_report_generated,
    record_watch_started,
    record_watch_stopped,
)
from ml_data_view.compare import COMPARE_SAMPLE_FILES, compare_schemas, evenly_spaced
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
from ml_data_view.deep_checks import DEEP_CHECKS
from ml_data_view.findings import (
    INSTANT_CHECKS,
    diff_findings,
    run_instant_checks,
    severity_summary,
    sorted_findings,
)
from ml_data_view.hierarchy import (
    ROLE_GROUP,
    ROLE_SERIES,
    DatasetStructure,
    Scope,
    StructureMapping,
    build_structure,
    chunk_identity,
    detect_timestamp_column,
    propose_mapping,
    scope_files,
    scope_label,
    scope_lazyframe,
    scope_units,
)
from ml_data_view.folder_watcher import (
    DEFAULT_STABILITY_SECONDS,
    MIN_STABILITY_SECONDS,
    FolderWatcher,
)
from ml_data_view.folder_loader import (
    FolderScanResult,
    discover_files,
    load_files,
    load_group,
    scan_folder,
)
from ml_data_view.image_data import resolve_image_full
from ml_data_view.jobs import IDLE_STATE, Job, JobManager, ceiling_status, default_memory_ceiling_mb
from ml_data_view.load_stats import build_load_stats, format_peak_memory
from ml_data_view.profile_cache import SECTION_TIMESERIES, ProfileCache, cache_file_path
from ml_data_view.report import available_sections
from ml_data_view.search import (
    SEARCH_ROW_LIMIT,
    Condition,
    SearchConditionError,
    build_predicate,
    search_rows,
)
from ml_data_view.search_jobs import SearchCountJob, SearchExportJob
from ml_data_view.catalog import attach_health, catalog_csv
from ml_data_view.file_profile import timeseries_signature
from ml_data_view.timeline import DEFAULT_BUCKETS as DEFAULT_TIMELINE_BUCKETS
from ml_data_view.timeline import build_timeline
from ml_data_view.timeline import chunks_from_artifacts as timeline_chunks
from ml_data_view.timeline import overlay as timeline_overlay
from ml_data_view.timeline import window_stats
from ml_data_view.timeseries import Thresholds as _TimeseriesThresholds
from ml_data_view.file_profile import fleet_from_cache
from ml_data_view.fleet import FleetThresholds
from ml_data_view.fleet import unavailable as fleet_unavailable
from ml_data_view.settings import (
    MIN_MEMORY_CEILING_MB,
    AppSettings,
    fleet_thresholds,
    load_settings,
    remove_saved_search,
    resolve_memory_ceiling_mb,
    save_settings,
    structure_mapping_for,
    timeseries_thresholds,
    upsert_saved_search,
    upsert_structure_mapping,
)


def _job_progress(job) -> dict[str, Any]:
    """Uniform snapshot for a background job the UI polls, including the
    idle case before one has been started."""
    if job is None:
        return dict(IDLE_STATE)
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
        self._folder_load_job: Job | None = None
        self._folder_load_adopted = False
        # Source files behind _current_lazy for a folder load; None for a
        # single-file open. Threaded into build_chart_payload so chart
        # computation stays per-file too.
        self._current_files: list[Path] | None = None
        # The in-flight chart build, if any (start_chart_build).
        self._chart_build_job: Job | None = None
        self._chart_build_adopted = False
        # The in-flight deep-check run, if any (start_deep_checks).
        self._deep_check_job: Job | None = None
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
        self._compare_job: Job | None = None
        # The in-flight PDF report generation, if any.
        self._report_job: Job | None = None
        self._report_adopted = False
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
        # Every heavy operation below runs as a capped worker subprocess
        # (jobs.py). This is the registry the top strip's indicator reads
        # and cancels through; each method also keeps its own reference,
        # because only it knows how to adopt its own result.
        self._job_manager = JobManager()
        # Peak memory and ceiling of the job that produced the loaded
        # dataset, so the load stat line can say "peak 1.4 GB of 8 GB
        # ceiling" instead of quoting this process's all-time high-water
        # mark for work that did not happen in this process.
        self._load_job_memory: dict[str, Any] | None = None
        # The schema group the loaded folder came from, so "Force full
        # re-profile" can reload exactly what is on screen rather than
        # asking the user to find the folder again.
        self._loaded_group_index: int | None = None
        # The loaded group's schema, kept so a re-discovery after new files
        # arrive can find the SAME group again. The index cannot be reused:
        # groups are ordered by file count, so a growing second group can
        # take the first one's place and silently swap the dataset.
        self._loaded_group_signature: tuple | None = None
        # The directory structure, when one was detected and confirmed for
        # the open root (hierarchy.py). When it is set the UNIT is the
        # loading unit and schema grouping is information; when it is None
        # the app behaves exactly as it did before, on schema groups. Both
        # of the structure's roles are optional, so "set" covers a
        # hive-partitioned tree with no series as readily as a full one.
        self._structure: DatasetStructure | None = None
        self._structure_proposal = None
        # What the app is currently looking at: a prefix of the group
        # values, optionally ending at one unit. Every section reads the
        # loaded frame, so changing scope is a load, not a filter.
        self._scope: Scope = Scope()
        # A column the user nominated to compare groups by, when the
        # structure does not supply one. Checks that need a grouping column
        # declare it as a precondition and stay silent without one.
        self._grouping_column: str | None = None
        # The group comparison rebuilt for a grouping the chart build did
        # not use, memoised by column. Cleared wherever the chart payload
        # is, since it is derived from the same profile.
        self._fleet_regrouped: tuple[str | None, dict[str, Any]] | None = None
        # The timeline payload currently on screen, so a lassoed window is
        # summarised from the buckets the user is looking at rather than
        # by reading the same rows again.
        self._timeline: dict[str, Any] | None = None
        # Folder watching (folder_watcher.py). The watcher itself lives in
        # this process — it is an observer thread and a stat() per pending
        # file, not work — while the re-profiling it triggers goes through
        # the same capped jobs as everything else.
        self._watcher: FolderWatcher | None = None
        # Set when the watcher reports a file is ready or gone, cleared
        # when a re-profile actually starts. A flag rather than a queue:
        # twenty files arriving at once want ONE re-profile, and the
        # re-profile re-merges the whole folder regardless of what changed.
        self._watch_dirty = False
        self._watch_changes: list[dict[str, Any]] = []
        # The findings from the last instant scan, so the next one can say
        # what changed rather than restating everything. None means "no
        # scan to compare against" — a first profile, or a new dataset.
        self._last_findings: list[dict[str, Any]] | None = None
        # Which dataset those findings describe. Checked rather than
        # cleared at each of the four load paths: a diff against a
        # DIFFERENT dataset's findings would report every column of the new
        # one as "new", which is worse than reporting nothing.
        self._findings_dataset_key: tuple | None = None

    # --- background jobs -------------------------------------------------
    #
    # Heavy work does not run in this process. It runs in a worker
    # subprocess under an OS-enforced memory ceiling, reports progress
    # through a file, and can be cancelled or killed without taking the
    # window with it (jobs.py). Every method here returns immediately;
    # the frontend polls. Cheap work — schema, metadata, preview rows,
    # aggregations that are already bounded — stays in-process, because
    # paying ~1s of process startup to read a footer would be a worse
    # experience, not a safer one.

    def _ceiling_mb(self) -> int:
        return resolve_memory_ceiling_mb(self._settings)

    def _cache_path(self) -> str:
        """Where per-file profiles are remembered. Beside settings.json, so
        a test Api that redirects one redirects the other."""
        return str(cache_file_path(self._settings_dir))

    def _start_job(
        self, task: str, params: dict[str, Any], *, label: str, files_total: int = 0
    ) -> Job:
        job = Job(
            task,
            params,
            label=label,
            limit_mb=self._ceiling_mb(),
            files_total=files_total,
        )
        self._job_manager.register(job)
        job.start()
        return job

    def _dataset_params(self) -> dict[str, Any]:
        """How a worker rebuilds the loaded dataset: paths, never a frame.

        `pl.scan_*`/`pl.concat` are lazy, so the child reconstructs the
        same LazyFrame for free and then does its own reading inside the
        ceiling — which is the whole point. Serializing a frame would mean
        materializing it in the one process that isn't capped.
        """
        params: dict[str, Any] = {}
        if self._current_files:
            params["files"] = [str(f) for f in self._current_files]
        elif self._current_path:
            params["path"] = self._current_path
        dataset_dir = self._dataset_dir()
        if dataset_dir is not None:
            params["dataset_dir"] = str(dataset_dir)
        return params

    # --- structure and scope ----------------------------------------------
    #
    # A directory structure is DETECTED, CONFIRMED and OPTIONAL. Where one
    # exists, the UNIT replaces the dominant schema group as the thing that
    # loads, and a scope — any prefix of the group values, optionally one
    # unit — decides which units. Where one does not, every method here is
    # inert and the app keeps its schema-group behaviour untouched.
    #
    # Nothing in this file knows what the levels are CALLED. The labels
    # come from the mapping and are the user's own words for their own
    # data; the code says group, series, chunk and unit.

    def get_structure(self) -> dict[str, Any]:
        """The confirmed structure, or {"detected": False} when the open
        folder has none."""
        if self._structure is None:
            return {"detected": False}
        return {"detected": True, **self._structure.to_dict(), "scope": self.get_scope()}

    def get_structure_proposal(self) -> dict[str, Any]:
        """What the structure dialog shows. Independent of whether anything
        is loaded, so the dialog can be reopened to CHANGE an answer rather
        than only ever being offered once."""
        if self._structure_proposal is None:
            return {"detected": False}
        data = self._structure_proposal.to_dict()
        data["saved"] = structure_mapping_for(self._settings, data["root"])
        return data

    def confirm_structure(self, mapping: dict[str, Any]) -> dict[str, Any]:
        """Apply the user's answer and remember it for this root.

        Saved against the structure FINGERPRINT, not just the path, so
        "never asked again unless the structure changes" is true rather
        than "never asked again" — which would leave someone stuck with a
        mapping that no longer describes their folder.

        BOTH roles are optional. A mapping with only group levels is a
        hive-partitioned tree; one with only a series level is a single
        run; one with neither is not a structure at all, and is rejected as
        such rather than being applied to produce nothing.
        """
        if self._folder_path is None:
            return {"error": "Open a folder first."}
        proposal = self._structure_proposal
        payload = dict(mapping or {})
        payload.setdefault("root", self._folder_path)
        payload.setdefault("fingerprint", proposal.fingerprint if proposal else "")
        payload.pop("declined", None)  # confirming replaces a previous decline
        confirmed = StructureMapping.from_dict(payload)
        roles = set(confirmed.roles.values())
        if ROLE_GROUP not in roles and ROLE_SERIES not in roles:
            return {
                "error": "Mark at least one level as a group or a series, "
                "or open the folder without a structure."
            }

        self._settings = upsert_structure_mapping(self._settings, confirmed.to_dict())
        save_settings(self._settings, self._settings_dir)
        structure = self._apply_mapping(confirmed)
        if not structure.units:
            return {"error": "That mapping produced nothing to load. Try different levels."}
        return {"structure": structure.to_dict(), "scope": self.get_scope()}

    def clear_structure(self) -> dict[str, Any]:
        """Forget this root's mapping and fall back to schema groups.

        The escape hatch that keeps the structure optional in practice as
        well as in principle: someone who decides their tree is better
        treated as a flat folder must be able to say so.
        """
        if self._folder_path is None:
            return {"error": "Open a folder first."}
        proposal = self._structure_proposal
        self._settings = upsert_structure_mapping(
            self._settings,
            {
                "root": self._folder_path,
                "roles": {},
                "declined": True,
                "fingerprint": proposal.fingerprint if proposal else "",
            },
        )
        save_settings(self._settings, self._settings_dir)
        self._structure = None
        self._scope = Scope()
        return self.discover_folder(self._folder_path)

    def _apply_mapping(self, mapping: StructureMapping) -> DatasetStructure:
        structure = build_structure(self._folder_path, mapping)
        self._structure = structure
        self._scope = Scope()  # a newly mapped root always starts at the top
        self._folder_scan_result = None
        return structure

    def _rebuild_frame(self, used_files: list[Path]):
        """The frame a finished load leaves behind.

        With a structure, it carries the derived columns — rebuilt from the
        chunk identity, narrowed to the files the load actually used, so a
        stopped load's frame never references a chunk its counts do not
        cover. Without one, it is the plain concat it has always been:
        that branch is what keeps a flat folder byte-for-byte unchanged.
        """
        if not used_files:
            return None
        if self._structure is None:
            return load_files(used_files)
        wanted = {str(path) for path in used_files}
        narrowed = [row for row in self._scope_identity() if row["file"] in wanted]
        return scope_lazyframe(narrowed) if narrowed else load_files(used_files)

    def get_scope(self) -> dict[str, Any]:
        """The scope, its breadcrumb, and the choices one level down.

        The breadcrumb carries the MAPPING's labels; `options` is what the
        next step down offers — group values while there are group levels
        left, then units. Both in one payload so the breadcrumb needs no
        second round trip.
        """
        if self._structure is None:
            return {"active": False}
        structure = self._structure
        depth = len(self._scope.values)
        remaining_groups = len(structure.group_depths) - depth

        if self._scope.unit_id:
            options, option_label = [], ""
        elif remaining_groups > 0:
            options = structure.values_at(self._scope.values)
            option_label = structure.group_labels[depth]
        else:
            options = sorted({unit.unit_id for unit in scope_units(structure, self._scope)})
            option_label = structure.unit_label

        units = scope_units(structure, self._scope)
        return {
            "active": True,
            **self._scope.to_dict(),
            "crumbs": scope_label(structure, self._scope),
            "options": options,
            "option_label": option_label,
            "option_kind": "unit" if remaining_groups <= 0 else "group",
            "group_labels": structure.group_labels,
            "unit_label": structure.unit_label,
            "unit_count": len(units),
            "chunk_count": sum(len(unit.chunks) for unit in units),
            "row_count": sum(unit.row_count for unit in units),
        }

    def set_scope(
        self, values: list[str] | None = None, unit_id: str | None = None
    ) -> dict[str, Any]:
        """Change what the app is looking at, and start loading it.

        A scope change is a LOAD, not a filter: every section reads the
        loaded frame, and narrowing means fewer chunks to read rather than
        the same chunks with a predicate on them. Incremental through the
        profile cache, so moving around an already-profiled tree reads
        nothing.
        """
        if self._structure is None:
            return {"error": "This folder has no detected structure."}
        scope = Scope(values=[str(v) for v in (values or [])], unit_id=unit_id or None)
        if len(scope.values) > len(self._structure.group_depths):
            return {"error": "That is deeper than this folder's group levels go."}
        if not scope_units(self._structure, scope):
            return {"error": "Nothing in this folder matches that selection."}
        self._scope = scope
        return self.start_scope_load()

    def _structure_labels(self) -> dict[str, Any] | None:
        """The mapping's own words, and the columns they name.

        Everything downstream that has to SHOW a level — the catalogue's
        column heading, the fleet section's "System 456" — reads it from
        here rather than holding a word of its own. None when there is no
        structure, which is how the generic defaults ("File", "Group",
        "All") get used without anything having to ask for them.
        """
        if self._structure is None:
            return None
        derived = self._structure.derived_columns()
        group_count = len(self._structure.group_depths)
        return {
            "group_labels": self._structure.group_labels,
            "unit_label": self._structure.unit_label,
            "root_label": self._structure.mapping.root_label,
            "group_columns": derived[:group_count],
            "unit_column": derived[group_count],
        }

    def _scope_identity(self) -> list[dict[str, Any]]:
        if self._structure is None:
            return []
        return chunk_identity(self._structure, self._scope)

    def _identity_by_file(self) -> dict[str, dict[str, Any]] | None:
        """The same values keyed by path, for the per-file scanners.

        Search, its match count and its CSV export all scan one file at a
        time — that is what gives them their early exit — so they attach
        the derived columns per file rather than filtering a concatenated
        frame that already has them.
        """
        identity = self._scope_identity()
        if not identity:
            return None
        return {row["file"]: row for row in identity}

    def start_scope_load(self) -> dict[str, Any]:
        """Load the current scope's chunks. The structured counterpart of
        start_folder_load(), and the same job underneath — only the file
        list and the derived columns differ."""
        if self._structure is None:
            return {"error": "This folder has no detected structure."}
        files = scope_files(self._structure, self._scope)
        if not files:
            return {"error": "This scope has nothing to load."}
        folder_path = Path(self._folder_path)
        record_folder_open(folder_path.name, self._settings, base_dir=self._settings_dir)

        self._loaded_group_index = None
        self._loaded_group_signature = ("scope", tuple(self._scope.values), self._scope.unit_id)
        self._folder_load_job = self._start_job(
            "folder_load",
            {
                "files": [str(f) for f in files],
                "folder_path": str(folder_path),
                "row_count": None,
                "cache_path": self._cache_path(),
                # Paths and scalars, like every other spec — the worker
                # rebuilds the identical frame from them.
                "identity": self._scope_identity(),
                "timestamp_column": self._structure.timestamp_column,
            },
            label="Loading selection",
            files_total=len(files),
        )
        self._folder_load_adopted = False
        return {"started": True, "files_total": len(files), "scope": self.get_scope()}

    # --- what each check may assume ---------------------------------------
    #
    # Every check declares a precondition and simply does not run without
    # it. These are the facts those predicates read, computed once per load
    # and carried in dataset_info so the checks stay pure functions over
    # dicts the app already has.

    def get_capabilities(self) -> dict[str, Any]:
        """Which preconditions the loaded dataset satisfies.

        Read by the Findings tab so a skipped check can say WHY it was
        skipped rather than appearing to have passed.
        """
        return dict(self._capabilities())

    def _capabilities(self) -> dict[str, Any]:
        timestamp = self._detected_timestamp()
        return {
            # Structure checks. A single file and a flat folder have no
            # directory structure to have problems with.
            "has_structure": self._structure is not None,
            # Time-series checks. Available on ANY input with a time column
            # — a single CSV included — not only on a structured tree.
            "timestamp_column": timestamp,
            # Chunk-boundary checks. Only a series has boundaries between
            # chunks; a unit that is one file has none.
            "has_series": bool(self._structure and self._structure.has_series),
            # Group comparison. A directory role, a hive key, or any column
            # the user nominated.
            "grouping_column": self._resolved_grouping_column(),
            "grouping_candidates": self._grouping_candidates(),
        }

    def _timeseries_files(self) -> list[Path]:
        """The files the time-series pass reads.

        A folder's chunks, or — for a single opened file — that one file.
        Named separately from `_current_files` because the chart path uses
        that list to decide whether to scan per file, and a single file
        must not start taking the per-file parquet route just to get its
        time column analysed.
        """
        if self._current_files:
            return list(self._current_files)
        return [Path(self._current_path)] if self._current_path else []

    def _detected_timestamp(self) -> str | None:
        """The timestamp column for the loaded dataset, whatever its shape.

        A confirmed structure's choice wins — the user picked it — and
        otherwise it is detected from the schema, which is what lets a
        time-series check run on a single CSV that happens to have a time
        column.
        """
        if self._structure is not None and self._structure.timestamp_column:
            return self._structure.timestamp_column
        if self._current_lazy is None:
            return None
        found = detect_timestamp_column(self._current_lazy.collect_schema())
        return found["column"] if found else None

    def _grouping_candidates(self) -> list[str]:
        """Columns a group comparison could run over.

        The structure's own group columns first — they are the reason the
        structure was mapped — then any low-cardinality categorical column
        the chart build already classified. "Any column the user picks"
        means the list is an offer, not a restriction: set_grouping_column
        accepts anything in the schema.
        """
        candidates: list[str] = []
        if self._structure is not None:
            derived = self._structure.derived_columns()
            candidates.extend(derived[: len(self._structure.group_depths)])
        for bar in (self._chart_data or {}).get("categorical_bars") or []:
            name = bar.get("column")
            if name and name not in candidates:
                candidates.append(name)
        return candidates

    def _resolved_grouping_column(self) -> str | None:
        """The column a group comparison runs over, or None.

        The user's explicit choice, else a column the STRUCTURE supplies —
        a directory role or a hive key, which are grouping columns by
        definition. Never an arbitrary categorical column the chart build
        happened to find: picking one of those would be assuming a grouping
        nobody asked for, and a check that fires on a guess is worse than
        one that does not fire. Those columns are still offered as
        candidates; offering is not choosing.
        """
        if self._grouping_column:
            return self._grouping_column
        if self._structure is None:
            return None
        derived = self._structure.derived_columns()
        groups = derived[: len(self._structure.group_depths)]
        return groups[0] if groups else None

    def set_grouping_column(self, column: str | None) -> dict[str, Any]:
        """Nominate the column a group comparison runs over.

        Any column in the loaded schema, not just a structural one: a
        dataset can be grouped by something that was never a directory,
        and refusing that would make the feature depend on a folder layout
        it has no reason to depend on. None returns to the default.
        """
        if column and self._current_lazy is None:
            return {"error": "No dataset loaded yet."}
        if column and column not in self._current_lazy.collect_schema():
            return {"error": f"No column named {column!r} in this dataset."}
        self._grouping_column = column or None
        self._fleet_regrouped = None
        return self.get_capabilities()

    # --- folder watching --------------------------------------------------
    #
    # The watcher runs in THIS process: it is an observer thread plus a
    # stat() per pending file, which is not work. Everything it triggers —
    # profiling the new file, re-merging, rebuilding charts — goes through
    # the same capped background jobs as a manual open, so a file arriving
    # while nobody is looking cannot do anything a user-initiated load
    # could not.
    #
    # Nothing here pushes to the frontend. The UI polls get_watch_status()
    # on the same timer it polls the jobs indicator with and decides for
    # itself when to act, which is what keeps a re-profile from ever
    # interrupting whatever the user is doing.

    def _watch_settings(self) -> tuple[bool, str, float]:
        return (
            bool(self._settings.watch_enabled),
            self._settings.watch_folder or "",
            float(self._settings.watch_stability_seconds or DEFAULT_STABILITY_SECONDS),
        )

    def sync_watch(self) -> dict[str, Any]:
        """Make the running watcher match Settings. Idempotent.

        Called at startup and after any settings change, rather than having
        each setting know how to start and stop a watcher. A change to the
        folder or the stability interval replaces the watcher, because both
        are decided when it is constructed and a half-reconfigured watcher
        is the kind of thing that silently keeps watching the old folder.
        """
        enabled, folder, stability = self._watch_settings()
        wanted = enabled and bool(folder) and Path(folder).is_dir()

        current = self._watcher
        if current is not None:
            unchanged = (
                wanted
                and str(current.folder) == str(Path(folder))
                and current.stability_seconds == max(MIN_STABILITY_SECONDS, stability)
            )
            if unchanged:
                return self.get_watch_status()
            current.stop()
            self._watcher = None
            record_watch_stopped(
                current.folder.name, self._settings, base_dir=self._settings_dir
            )

        if not wanted:
            self._watch_dirty = False
            self._watch_changes = []
            return self.get_watch_status()

        self._watcher = FolderWatcher(
            folder, self._on_watch_change, stability_seconds=stability
        )
        self._watcher.start()
        record_watch_started(Path(folder).name, self._settings, base_dir=self._settings_dir)
        return self.get_watch_status()

    def _on_watch_change(self, events) -> None:
        """Called from the watcher's poller thread. Records; never acts.

        Starting a job from here would mean profiling triggered by a thread
        the UI cannot see, at a moment nobody chose. Instead this sets a
        flag the frontend picks up on its next poll — and a flag rather
        than a queue, because twenty files arriving at once want ONE
        re-profile: the merge covers the whole folder regardless of which
        file prompted it.
        """
        self._watch_dirty = True
        self._watch_changes = [
            {"file": Path(event.path).name, "kind": event.kind} for event in events
        ]

    def get_watch_status(self) -> dict[str, Any]:
        """What the top strip's Watching indicator shows, plus whether a
        re-profile is wanted. Cheap and non-blocking, like list_jobs()."""
        enabled, folder, stability = self._watch_settings()
        watcher = self._watcher
        status = (
            watcher.status()
            if watcher is not None
            else {
                "watching": False,
                "paused": False,
                "folder": folder,
                "stability_seconds": stability,
                "pending": [],
            }
        )
        status["enabled"] = enabled
        # Only offered when there is actually a loaded folder to re-merge
        # into. A watcher can legitimately be running over a folder the user
        # has not opened yet.
        status["reprofile_pending"] = bool(self._watch_dirty and self._can_reprofile())
        status["changes"] = list(self._watch_changes)
        return status

    def _can_reprofile(self) -> bool:
        watcher = self._watcher
        return (
            watcher is not None
            and self._folder_path is not None
            and self._loaded_group_signature is not None
            and Path(self._folder_path) == watcher.folder
        )

    def pause_watch(self) -> dict[str, Any]:
        """Stop PROCESSING events, not collecting them — resuming catches
        up on everything that arrived meanwhile (see FolderWatcher.pause)."""
        if self._watcher is None:
            return {"error": "Not watching any folder."}
        self._watcher.pause()
        return self.get_watch_status()

    def resume_watch(self) -> dict[str, Any]:
        if self._watcher is None:
            return {"error": "Not watching any folder."}
        self._watcher.resume()
        return self.get_watch_status()

    def watch_folder_dialog(self) -> str | None:
        """Native folder picker for the watched folder. Requires a window,
        so — like open_folder_dialog() — this is not unit-tested."""
        window = webview.windows[0]
        result = window.create_file_dialog(webview.FileDialog.FOLDER)
        if not result:
            return None
        return result[0]

    def watch_loaded_folder(self) -> dict[str, Any]:
        """Point the watcher at the folder currently on screen.

        The common case by a distance — someone opens a day's output and
        wants to see the rest of it arrive — and it saves them finding the
        same folder again in a picker.
        """
        if self._folder_path is None:
            return {"error": "Open a folder first."}
        return self.save_settings({"watch_enabled": True, "watch_folder": self._folder_path})

    def begin_watch_reprofile(self) -> dict[str, Any]:
        """Start an incremental re-profile of the watched folder.

        Re-discovers the folder rather than appending the new file to the
        loaded list: discovery is schema reads and footer metadata only
        (never a collect), it is how a REMOVED file drops out, and it is
        the only thing that can tell whether an arrival belongs to the
        loaded schema group or a different one. The load that follows is
        incremental through the profile cache, so the cost is the new files
        and nothing else.
        """
        if not self._can_reprofile():
            return {"error": "Nothing to re-profile — no watched folder is loaded."}

        result = scan_folder(self._folder_path)
        if not result.groups:
            return {"error": "No dataset files found in this folder."}
        index = self._matching_group_index(result)
        if index is None:
            return {
                "error": "The loaded schema group is no longer present in this folder."
            }
        self._folder_scan_result = result

        # Cleared BEFORE the load starts, not after: a file that arrives
        # while this re-profile runs must set the flag again and get its own
        # pass, rather than being swallowed by the one already in flight.
        self._watch_dirty = False
        record_reprofile(Path(self._folder_path).name, self._settings, base_dir=self._settings_dir)
        started = self.start_folder_load(index)
        if "error" in started:
            return started
        return {**started, "reprofile": True, "files_total": result.groups[index].file_count}

    def _matching_group_index(self, result: FolderScanResult) -> int | None:
        """Find the loaded group again after re-discovery, BY SCHEMA.

        Groups are ordered by file count, so the index is not stable: a
        second group that grows past the first takes its place, and reusing
        the index would swap the dataset under the user without a word.
        """
        for index, group in enumerate(result.groups):
            if group.signature == self._loaded_group_signature:
                return index
        return None

    # --- the profiling cache --------------------------------------------

    def get_cache_info(self) -> dict[str, Any]:
        """What Settings shows about the cache: how many files it has
        profiles for, how much disk that is, and where it lives."""
        with ProfileCache(self._cache_path()) as cache:
            size = cache.size_bytes()
            return {
                "entries": cache.entry_count(),
                "size_bytes": size,
                "size_label": format_peak_memory(size),
                "path": str(cache.path),
            }

    def clear_cache(self) -> dict[str, Any]:
        """Throw away every cached profile. The next open reads everything
        again, and the size Settings reports actually goes down (the store
        vacuums rather than just deleting rows)."""
        with ProfileCache(self._cache_path()) as cache:
            removed = cache.clear()
        info = self.get_cache_info()
        info["removed"] = removed
        return info

    def force_full_reprofile(self) -> dict[str, Any]:
        """Re-read the loaded folder from disk, ignoring what is cached.

        Distinct from "Clear cache", and not just a convenience wrapper on
        it: this drops the entries for THIS folder's files and immediately
        reloads them, which is the action someone takes when they suspect
        the numbers on screen are wrong. Other folders keep their profiles.
        """
        if self._folder_scan_result is None or self._loaded_group_index is None:
            return {"error": "Open a folder first — there is nothing to re-profile."}
        groups = self._folder_scan_result.groups
        index = self._loaded_group_index
        if index < 0 or index >= len(groups):
            return {"error": "The loaded folder is no longer available."}
        with ProfileCache(self._cache_path()) as cache:
            cache.forget(list(groups[index].files))
        return self.start_folder_load(index)

    def list_jobs(self) -> dict[str, Any]:
        """Running jobs, for the top strip's indicator: label, progress,
        ETA and an id to cancel by. Cheap and non-blocking — it reads each
        job's last published progress snapshot, never the job itself."""
        return {"jobs": self._job_manager.snapshot()}

    def cancel_job(self, job_id: str) -> dict[str, Any]:
        """Cancel one running job by id. Graceful at the job's next item
        boundary, forced shortly after if that is ignored."""
        if not self._job_manager.cancel(job_id):
            return {"error": "That job is no longer running."}
        return {"cancelled": True}

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
        self._current_files = None  # a single file is not a folder's file list
        self._loaded_group_signature = None  # a file belongs to no schema group
        self._chart_filter = None  # a new dataset never inherits a filter
        self._chart_filter_conditions = []
        self._search_predicate = None
        self._current_row_count = row_count(lf)  # one streaming count, shared below
        self._chart_data = None  # stale for the newly loaded dataset — get_chart_data() refills it
        self._fleet_regrouped = None

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
        """Current settings, plus the read-only facts Settings displays
        beside them: where the audit log is written, what the memory
        ceiling resolves to on this machine, and whether the OS can
        actually enforce it here.

        `memory_ceiling_status` is not a setting and not decoration: off
        Windows the ceiling is not enforced, and a limit the app implies it
        is honouring but isn't would be worse than showing none at all.
        """
        data = self._settings.to_dict()
        data["audit_log_path"] = str(audit_log_file_path(self._settings_dir))
        data["memory_ceiling_effective_mb"] = self._ceiling_mb()
        data["memory_ceiling_default_mb"] = default_memory_ceiling_mb()
        data["memory_ceiling_min_mb"] = MIN_MEMORY_CEILING_MB
        data["memory_ceiling_status"] = ceiling_status()
        data["cache"] = self.get_cache_info()
        data["watch_min_stability_seconds"] = int(MIN_STABILITY_SECONDS)
        data["watch"] = self.get_watch_status()
        return data

    def save_settings(self, updates: dict[str, Any]) -> dict[str, Any]:
        """Merge `updates` onto the current settings, persist immediately,
        and return the same shape as get_settings() so the frontend can
        re-render from the confirmed, saved state."""
        merged = {**self._settings.to_dict(), **(updates or {})}
        self._settings = AppSettings.from_dict(merged)
        save_settings(self._settings, self._settings_dir)
        # A watch setting that is saved but not applied is a checkbox that
        # lies. sync_watch() is idempotent, so this costs nothing when the
        # change was about something else.
        if any(key.startswith("watch_") for key in (updates or {})):
            self.sync_watch()
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
        self._fleet_regrouped = None
        self._chart_data = build_chart_payload(
            self._current_lazy,
            total_rows=self._current_row_count,
            dataset_dir=self._dataset_dir(),
            null_counts=null_counts or None,
            # Per-file for folder loads: every chart builder that would
            # otherwise query the concatenated frame grows with file count.
            files=self._current_files or None,
            # And the same cache. get_chart_data() is the synchronous form
            # of start_chart_build(), not a different computation: a
            # folder assembles from cached per-file statistics either way,
            # and a single file's per-chunk time-series artifact is stored
            # either way — which is what the channel timeline draws its
            # overview from.
            cache_path=self._cache_path(),
            # The same time-series inputs start_chart_build() passes. The
            # two paths are meant to produce the same payload — a headless
            # caller and a report should not silently get a dataset with
            # no health section just for taking the blocking route.
            identity=self._scope_identity() or None,
            timestamp_column=self._detected_timestamp(),
            timeseries_thresholds=timeseries_thresholds(self._settings),
            timeseries_max_channels=self._settings.timeseries_max_channels,
            timeseries_files=[str(f) for f in self._timeseries_files()],
            structure_labels=self._structure_labels(),
            grouping_column=self._resolved_grouping_column(),
            fleet_thresholds=fleet_thresholds(self._settings),
            fleet_max_channels=self._settings.fleet_max_channels,
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
        """Begin building chart data in a capped worker; returns at once.

        This is the operation the ceiling exists for: build_chart_payload
        peaked at 2132MB on two real files before it was made flat in file
        count, and the embedding projection inside it pulls in sklearn
        (~554MB of import baseline on its own). None of that now happens
        in the window's process.
        """
        if self._current_lazy is None:
            return {"error": "No dataset loaded yet. Open a file or folder first."}
        null_counts = {
            column["name"]: column["null_count"]
            for column in (self._dataset_info or {}).get("columns", [])
        }
        self._chart_build_job = self._start_job(
            "chart_build",
            {
                **self._dataset_params(),
                "total_rows": self._current_row_count,
                "null_counts": null_counts or None,
                # The conditions, not the polars expression: an Expr is not
                # JSON, and these are already kept beside the predicate.
                "conditions": list(self._chart_filter_conditions)
                if self._chart_filter is not None
                else None,
                "cache_path": self._cache_path(),
                # Lets the worker rebuild the same derived columns, and lets
                # the merge synthesize their statistics from row counts
                # rather than reading anything.
                "identity": self._scope_identity() or None,
                # Time-series health. The column is the applicability
                # predicate: without one, the whole pass is skipped and
                # every check that reads it skips too.
                "timestamp_column": self._detected_timestamp(),
                "timeseries_thresholds": timeseries_thresholds(self._settings),
                "timeseries_max_channels": self._settings.timeseries_max_channels,
                "timeseries_files": [str(f) for f in self._timeseries_files()],
                # The mapping's own words, and the column a group
                # comparison runs over. Labels and a name — no data — so
                # the catalogue and the fleet comparison can be assembled
                # in the worker from statistics it has already merged.
                "structure_labels": self._structure_labels(),
                "grouping_column": self._resolved_grouping_column(),
                "fleet_thresholds": fleet_thresholds(self._settings),
                "fleet_max_channels": self._settings.fleet_max_channels,
            },
            label="Building charts",
            files_total=len(self._current_files or []),
        )
        self._chart_build_adopted = False
        return {"started": True}

    def get_chart_build_progress(self) -> dict[str, Any]:
        """Snapshot of the running chart build. On completion the payload
        is cached exactly as get_chart_data() would have cached it, so
        Reports keeps working unchanged."""
        job = self._chart_build_job
        if job is None:
            return dict(IDLE_STATE)
        state = job.progress()
        if state["finished"] and state["result"] is not None and not self._chart_build_adopted:
            self._chart_data = state["result"]
            self._fleet_regrouped = None
            self._chart_build_adopted = True
        return state

    def cancel_chart_build(self) -> dict[str, Any]:
        """Stop a chart build.

        The Stop button the UI shows for a folder load is deliberately
        absent here — a half-built chart payload has no coherent meaning,
        so the build reports no partial result and cancelling discards it.
        The method exists because the jobs indicator offers Cancel on every
        running job, and a job the user can see running must be one they
        can stop.
        """
        if self._chart_build_job is None:
            return {"error": "No chart build in progress."}
        self._chart_build_job.cancel()
        return {"cancelled": True}

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
            self._search_sources(),
            predicate,
            columns=self._search_columns,
            limit=limit,
            identity=self._identity_by_file(),
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
        self._match_count_job = SearchCountJob(
            self._search_sources(), self._search_predicate, identity=self._identity_by_file()
        )
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
            self._search_sources(),
            self._search_predicate,
            self._search_columns,
            path,
            identity=self._identity_by_file(),
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
        self._fleet_regrouped = None
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
        self._compare_job = self._start_job(
            "compare",
            {
                "files_a": [str(f) for f in files_a],
                "files_b": [str(f) for f in files_b],
            },
            label="Comparing datasets",
        )
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

    # --- the channel timeline ---------------------------------------------
    #
    # One unit's channels over time. Its precondition is a detected
    # timestamp column and nothing else, so a single CSV with a time
    # column gets one; a series just means there are several chunks to
    # choose between. Nothing here is a job: a fetch either folds buckets
    # the profile cache already holds, or reads the row groups the window
    # touches within a stated row budget — see timeline.py.

    def _timeline_units(self) -> list[tuple[str, list[Path]]]:
        """(label, ordered chunk paths) per unit, keyed as everything else
        keys a unit — the group values then the unit id."""
        from ml_data_view.catalog import unit_label_key
        from ml_data_view.file_profile import unit_groups

        return [
            (unit_label_key(key), paths)
            for key, paths in unit_groups(self._timeseries_files(), self._scope_identity() or None)
        ]

    def _timeline_artifacts(self, paths: list[Path]) -> dict[str, dict[str, Any]]:
        """Each chunk's cached time-series artifact, or {} where there is
        none. Metadata reads out of SQLite; no column data."""
        signature = timeseries_signature(
            self._detected_timestamp() or "",
            _TimeseriesThresholds.from_dict(timeseries_thresholds(self._settings)),
        )
        artifacts: dict[str, dict[str, Any]] = {}
        with ProfileCache(self._cache_path()) as cache:
            for path in paths:
                section = (cache.get(path) or {}).get(SECTION_TIMESERIES)
                if isinstance(section, dict) and section.get("signature") == signature:
                    artifacts[str(path)] = section.get("result") or {}
        return artifacts

    def get_timeline(
        self,
        unit_id: str | None = None,
        channels: list[str] | None = None,
        start: float | None = None,
        end: float | None = None,
        buckets: int = DEFAULT_TIMELINE_BUCKETS,
    ) -> dict[str, Any]:
        """One unit's channels over a time window, downsampled.

        `buckets` is the frontend's pixel width, capped — the resolution
        asked for is the resolution that can be seen, and asking for more
        would be asking for rows nobody could look at.
        """
        column = self._detected_timestamp()
        if not column:
            return {"available": False, "reason": "no timestamp column was detected"}
        units = self._timeline_units()
        if not units:
            return {"available": False, "reason": "nothing is loaded to draw"}

        label, paths = next((u for u in units if u[0] == unit_id), units[0])
        artifacts = self._timeline_artifacts(paths)
        merged = ((self._chart_data or {}).get("timeseries") or {}).get("units", {}).get(label, {})
        available = list((merged.get("channels") or {}).keys()) or list(
            ((self._chart_data or {}).get("timeseries") or {}).get("channels_analysed") or []
        )
        chosen = [name for name in (channels or []) if name in available]
        if not chosen:
            chosen = available[: self._settings.timeline_max_channels]
        chosen = chosen[: self._settings.timeline_max_channels]

        window = (float(start), float(end)) if start is not None and end is not None else None
        payload = build_timeline(
            timeline_chunks(paths, artifacts),
            column,
            chosen,
            window=window,
            buckets=int(buckets or DEFAULT_TIMELINE_BUCKETS),
            max_rows=self._settings.timeline_max_rows,
        )
        payload.update(
            {
                "unit": label,
                "units": [name for name, _paths in units],
                "unit_label": (self._structure_labels() or {}).get("unit_label") or "File",
                "channels_available": available,
                "channels_selected": chosen,
                "max_channels": self._settings.timeline_max_channels,
                # The findings, as marks on the same axis — built in Python
                # beside the checks that report them, so a mark and the
                # finding it stands for cannot describe different moments.
                "marks": timeline_overlay(
                    merged,
                    chosen,
                    ranges=self._settings.timeseries_ranges or {},
                    range_spread_factor=self._settings.timeseries_range_spread_factor,
                )
                if merged
                else [],
            }
        )
        # Kept so a selection can be summarised from the buckets on screen
        # rather than by reading again — which is what makes a lasso
        # instant, and what decides which rows its figures describe.
        self._timeline = payload
        return payload

    def get_timeline_window(self, start: float, end: float) -> dict[str, Any]:
        """Summary statistics for a selected stretch of the timeline."""
        if not self._timeline:
            return {"error": "There is no timeline on screen to select from."}
        return window_stats(self._timeline, (float(start), float(end)))

    def timeline_window_conditions(self, start: float, end: float) -> dict[str, Any]:
        """A selected window, as a row-search condition.

        Export is not reimplemented for the timeline. A time window IS a
        search condition — `between` on the timestamp column — so the
        selection becomes a query in the feature that already knows how to
        count matches and write a CSV, and the user lands in it with the
        query filled in and editable. One export path, not two.

        Only Python can write this condition: the timeline works in
        seconds because that is what timeseries.py normalises to, and
        turning that back into what the COLUMN holds is the one step a
        frontend could get silently wrong.
        """
        column = self._detected_timestamp()
        if not column or self._current_lazy is None:
            return {"error": "No timestamp column to select a window from."}
        return {
            "conditions": [
                {
                    "column": column,
                    "operator": "between",
                    "values": self._timestamp_bounds(column, float(start), float(end)),
                }
            ]
        }

    def _timestamp_bounds(self, column: str, start: float, end: float) -> list[str]:
        """A window, written the way the timestamp COLUMN writes time.

        timeseries.py normalises a datetime column to epoch seconds so the
        analysis has one kind of number to work with; a numeric time
        column is left in whatever units it was already in. Going back the
        other way therefore has exactly two cases, and getting it wrong
        would silently select the wrong rows.
        """
        from datetime import datetime, timezone

        dtype = str(self._current_lazy.collect_schema().get(column, ""))
        if dtype.lower().startswith(("datetime", "date")):
            return [
                datetime.fromtimestamp(value, tz=timezone.utc)
                .replace(tzinfo=None)
                .isoformat(timespec="microseconds")
                for value in (start, end)
            ]
        return [repr(float(start)), repr(float(end))]

    # --- the catalogue and the group comparison ---------------------------
    #
    # Both are built during the chart build, out of statistics already
    # merged — cached row counts and schemas, the per-file histograms every
    # file was binned into against the same spec, and the per-unit
    # time-series merge. Neither reads anything, so both are simply part of
    # the payload rather than something to start and poll.

    def _instant_findings(self) -> list[dict[str, Any]]:
        """Every instant finding, without touching the diff state.

        run_instant_checks() below remembers its result so a re-profile can
        say what CHANGED; asking it for a health column would make merely
        looking at the catalogue count as a scan. The checks are pure
        functions over two dicts and take about 5ms in total, so running
        them again here is cheaper than the bookkeeping to avoid it.
        """
        if not self._chart_data:
            return []
        info = {**(self._dataset_info or {}), "capabilities": self._capabilities()}
        results = run_instant_checks(info, self._checks_chart_data())
        return [finding.to_dict() for finding in sorted_findings(results)]

    def _catalog_payload(self) -> dict[str, Any] | None:
        """The catalogue with its health column filled in.

        The rows come from the chart build; the health status is the worst
        severity among the findings that NAME each unit, which cannot be
        known until the checks have run. Kept apart for that reason rather
        than baked into the payload, where it would have to be a guess.
        """
        catalog = (self._chart_data or {}).get("catalog")
        if not catalog:
            return None
        return attach_health(catalog, self._instant_findings())

    def get_catalog(self) -> dict[str, Any]:
        """One row per unit, plus a summary per group.

        `{"available": False}` with a reason when there is nothing to
        catalogue yet — the same shape the group comparison uses, so the
        two tabs render an absence the same way.
        """
        if self._current_lazy is None:
            return {"available": False, "reason": "no dataset loaded", "rows": []}
        if not self._chart_data:
            return {"available": False, "reason": "waiting for the chart build", "rows": []}
        catalog = self._catalog_payload()
        if not catalog:
            return {
                "available": False,
                # A filtered build is the one case where a loaded dataset
                # has no catalogue, and saying which case it is beats a
                # blank tab.
                "reason": "a filtered build has no catalogue — clear the chart filter"
                if self._chart_filter is not None
                else "nothing to catalogue",
                "rows": [],
            }
        return catalog

    def _fleet_payload(self) -> dict[str, Any] | None:
        """The group comparison as it stands NOW.

        Built during the chart build. Rebuilt from the cache when the
        grouping column has changed since — nominating a different level
        rearranges statistics that are already written down and does not
        change a single bin, so making someone re-profile a folder to see
        their own tree grouped a second way would be charging them for
        arithmetic. Memoised per column, because the checks and the tab
        both ask.

        Reads the RAW catalogue, not the health-attached one: the
        comparison uses each row's group, duration, rate and gap share and
        none of its health, and going through the health-attached
        catalogue would mean running the checks to answer a question the
        checks are about to ask.
        """
        payload = (self._chart_data or {}).get("fleet")
        wanted = self._resolved_grouping_column()
        if payload is None or payload.get("group_column") == wanted:
            return payload
        if self._fleet_regrouped and self._fleet_regrouped[0] == wanted:
            return self._fleet_regrouped[1]
        if not self._current_files:
            return payload
        with ProfileCache(self._cache_path()) as cache:
            regrouped = fleet_from_cache(
                list(self._current_files),
                cache,
                identity=self._scope_identity() or None,
                catalog=(self._chart_data or {}).get("catalog"),
                grouping_column=wanted,
                labels=self._structure_labels(),
                timestamp_column=self._detected_timestamp(),
                thresholds=FleetThresholds.from_dict(fleet_thresholds(self._settings)),
                max_channels=self._settings.fleet_max_channels,
            )
        self._fleet_regrouped = (wanted, regrouped)
        return regrouped

    def _checks_chart_data(self) -> dict[str, Any]:
        """The chart payload the CHECKS see.

        Identical but for the group comparison, which follows the grouping
        column currently chosen. Without this the Fleet tab and the
        findings beside it would describe two different groupings of the
        same folder.
        """
        payload = self._fleet_payload()
        if payload is None:
            return dict(self._chart_data or {})
        return {**(self._chart_data or {}), "fleet": payload}

    def get_fleet(self) -> dict[str, Any]:
        """Per-channel divergence of each group from the typical group."""
        if self._current_lazy is None:
            return fleet_unavailable("no dataset loaded")
        if not self._chart_data:
            return fleet_unavailable("waiting for the chart build")
        payload = self._fleet_payload()
        if not payload:
            return fleet_unavailable(
                "a filtered build has no group comparison — clear the chart filter"
                if self._chart_filter is not None
                else "no grouping column"
            )
        return payload

    def catalog_export_dialog(self, default_filename: str = "catalog.csv") -> str | None:
        window = webview.windows[0]
        result = window.create_file_dialog(
            webview.FileDialog.SAVE,
            save_filename=default_filename,
            file_types=("CSV files (*.csv)", "All files (*.*)"),
        )
        if not result:
            return None
        return result if isinstance(result, str) else result[0]

    def export_catalog(self, path: str) -> dict[str, Any]:
        """Write the catalogue to CSV — every row, raw numbers.

        Not a job: the catalogue is already in memory and a few thousand
        rows of text is not work. Nothing is read to produce it.
        """
        if not path:
            return {"error": "No output path given."}
        catalog = self._catalog_payload()
        if not catalog:
            return {"error": "There is no catalogue to export yet."}
        try:
            Path(path).write_text(catalog_csv(catalog), encoding="utf-8", newline="")
        except OSError as exc:
            return {"error": f"Could not write the catalogue: {exc}"}
        return {"path": str(path), "rows": len(catalog.get("rows") or [])}

    def start_scope_comparison(
        self,
        scope_a: dict[str, Any],
        scope_b: dict[str, Any],
    ) -> dict[str, Any]:
        """Compare two scopes of the loaded structure against each other.

        "Compare two recordings" and "compare two systems" are the same
        operation at two depths, because a scope is a prefix of the group
        values optionally ending at one unit — so this needs no notion of
        either. It hands the file lists to the SAME CompareJob the Compare
        tab already uses; nothing about the comparison itself is new here.
        """
        if self._structure is None:
            return {"error": "This folder has no detected structure to compare within."}
        try:
            files_a, label_a = self._scope_side(scope_a)
            files_b, label_b = self._scope_side(scope_b)
        except ValueError as exc:
            return {"error": str(exc)}

        schema_a = _dispatch_scanner(files_a[0], files_a[0].suffix.lower()).collect_schema()
        schema_b = _dispatch_scanner(files_b[0], files_b[0].suffix.lower()).collect_schema()
        schema_diff = compare_schemas(schema_a, schema_b)
        self._compare_job = self._start_job(
            "compare",
            {
                "files_a": [str(f) for f in files_a],
                "files_b": [str(f) for f in files_b],
            },
            label=f"Comparing {label_a} with {label_b}",
        )
        return {
            "started": True,
            "schema": schema_diff,
            "files_a": len(files_a),
            "files_b": len(files_b),
            # Every chunk of both sides, not a sample. A scope is already
            # the narrowing — sampling it as well would answer a question
            # about part of the selection the user just made.
            "sampled": False,
            # Named sides, so the results table says "rec_a" and "rec_b"
            # rather than "A" and "B" — the whole point of launching a
            # comparison from a row of the catalogue.
            "label_a": label_a,
            "label_b": label_b,
        }

    def _scope_side(self, scope: dict[str, Any]) -> tuple[list[Path], str]:
        selection = Scope.from_dict(scope)
        if len(selection.values) > len(self._structure.group_depths):
            raise ValueError("That is deeper than this folder's group levels go.")
        files = scope_files(self._structure, selection)
        if not files:
            raise ValueError("Nothing in this folder matches that selection.")
        parts = list(selection.values)
        if selection.unit_id:
            parts.append(selection.unit_id)
        return files, " / ".join(parts) or self._structure.mapping.root_label

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

        # Capabilities travel WITH the dataset info so every check stays a
        # pure function over dicts the app already has — a precondition
        # must not be a reason to give the instant catalogue I/O.
        info = {
            **(self._dataset_info or {}),
            "capabilities": self._capabilities(),
        }
        results = run_instant_checks(info, self._checks_chart_data())
        findings = [finding.to_dict() for finding in sorted_findings(results)]
        # Against the PREVIOUS scan of this SAME dataset, which is what
        # makes a watch-triggered re-profile worth interrupting someone
        # for. None on a first scan, and None after switching datasets:
        # calling every finding on a freshly opened dataset "new" would
        # make the word mean nothing.
        key = (self._current_path, self._loaded_group_signature)
        previous = self._last_findings if key == self._findings_dataset_key else None
        changes = diff_findings(previous, findings)
        self._last_findings = findings
        self._findings_dataset_key = key
        return {
            "checks": results,
            "findings": findings,
            "summary": severity_summary(sorted_findings(results)),
            "changes": changes,
        }

    def start_deep_checks(self) -> dict[str, Any]:
        """Begin the deep checks in a capped worker; returns at once."""
        if self._current_lazy is None:
            return {"error": "No dataset loaded yet. Open a file or folder first."}
        self._deep_check_job = self._start_job(
            "deep_checks",
            {
                **self._dataset_params(),
                "image_columns": list((self._chart_data or {}).get("image_columns") or []),
            },
            label="Running deep checks",
        )
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
        return available_sections(self._report_chart_data())

    def _report_chart_data(self) -> dict[str, Any]:
        """The chart payload as the REPORT should see it.

        Identical but for the catalogue, whose health column is filled in
        here. report.py deliberately depends on nothing in the findings
        layer, so the statuses reach it as data rather than as an import.
        """
        catalog = self._catalog_payload()
        if catalog is None:
            return self._checks_chart_data()
        return {**self._checks_chart_data(), "catalog": catalog}

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

    def start_report_generation(self, output_path: str, sections: list[str]) -> dict[str, Any]:
        """Begin writing a PDF report in a capped worker; returns at once.

        PDF export belongs in a job for a reason the other tasks do not
        share: kaleido rasterizes every chart by driving a real Chromium
        through choreographer, so this operation spawns a browser. Inside
        a Job Object that browser is a member of the job, which means
        cancelling or hitting the ceiling takes it down too instead of
        leaving a headless Chrome running after the app has moved on.
        """
        if (
            self._current_lazy is None
            or self._current_path is None
            or self._chart_data is None
            or self._dataset_info is None
        ):
            return {"error": "No dataset loaded yet. Open a file first."}
        if not output_path:
            return {"error": "No output path given."}

        self._report_job = self._start_job(
            "report",
            {
                "output_path": output_path,
                # Already-computed dicts, so nothing is recomputed in the
                # worker — report.py reuses the chart payload exactly as it
                # does in-process.
                "dataset_info": self._dataset_info,
                "chart_data": self._report_chart_data(),
                "sections": list(sections or []),
                "banner": self._banner_settings(),
            },
            label="Generating PDF report",
        )
        self._report_adopted = False
        return {"started": True}

    def get_report_progress(self) -> dict[str, Any]:
        """Snapshot of the running report generation. The audit entry is
        written here, once, when it actually succeeds — a report that
        failed or was cancelled was not generated."""
        if self._report_job is None:
            return dict(IDLE_STATE)
        state = self._report_job.progress()
        if state["finished"] and state["result"] and not self._report_adopted:
            self._report_adopted = True
            record_report_generated(
                Path(self._current_path).name, self._settings, base_dir=self._settings_dir
            )
        return state

    def cancel_report_generation(self) -> dict[str, Any]:
        if self._report_job is None:
            return {"error": "No report generation in progress."}
        self._report_job.cancel()
        return {"cancelled": True}

    def generate_report(self, output_path: str, sections: list[str]) -> dict[str, Any]:
        """Blocking form of start_report_generation(), for headless callers
        and tests — the same relationship get_chart_data() has with
        start_chart_build(). The frontend uses the polled pair so the
        window stays responsive while Chromium runs."""
        started = self.start_report_generation(output_path, sections)
        if "error" in started:
            return started
        self._report_job.wait()
        state = self.get_report_progress()
        if state.get("error"):
            return {"error": f"Could not write report: {state['error']}"}
        result = state.get("result") or {}
        if not result.get("path"):
            return {"error": "Report generation produced no file."}
        return {"path": result["path"]}

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
        proposal = propose_mapping(path)
        self._structure_proposal = proposal
        self._folder_path = path

        saved = structure_mapping_for(self._settings, path)
        # A structure that was DECLINED is remembered as firmly as one that
        # was confirmed. Without this, "load as a flat folder" would be
        # undone by the next open, and the offer would not be a real one.
        declined = bool(
            saved and saved.get("declined") and saved.get("fingerprint") == proposal.fingerprint
        )

        if proposal.detected and not declined:
            self._folder_scan_result = None
            # Re-asked only when the SHAPE changed, never because files
            # were added: the fingerprint describes which levels exist, not
            # what is in them.
            if saved and saved.get("fingerprint") == proposal.fingerprint:
                structure = self._apply_mapping(StructureMapping.from_dict(saved))
                return {
                    "source": "structure",
                    "needs_confirmation": False,
                    "structure": structure.to_dict(),
                    "scope": self.get_scope(),
                }
            return {
                "source": "structure",
                "needs_confirmation": True,
                "structure_proposal": proposal.to_dict(),
            }

        # No hive-style level anywhere: this is an ordinary folder and the
        # existing schema-group behaviour applies, completely unchanged.
        self._structure = None
        self._scope = Scope()
        result = scan_folder(path)
        if not result.groups:
            return {"error": "No dataset files found in this folder."}
        self._folder_scan_result = result
        data = result.to_dict()
        data["excluded"] = result.excluded_from(result.default_group_index)
        data["source"] = "flat"
        data["needs_confirmation"] = False
        data["structure_proposal"] = proposal.to_dict()
        # A folder of files has nothing to explain. A folder with
        # SUBDIRECTORIES that were not recognised does: silently flattening
        # someone's carefully arranged tree, with no word about it, is how
        # they conclude the app cannot read their data. The levels are
        # still offered, so the answer the code declined to guess can be
        # given by hand.
        if declined:
            data["note"] = (
                "Loaded as a flat folder — you chose not to use this folder's structure."
            )
            data["mappable"] = True
        elif proposal.levels:
            data["note"] = (
                f"Loaded as a flat folder: no group or series level was recognised in this "
                f"folder's {len(proposal.levels)} directory level(s). You can set the roles "
                "yourself if it has a structure worth using."
            )
            data["mappable"] = True
        else:
            data["note"] = ""
            data["mappable"] = False
        return data

    # --- cancellable folder loading -------------------------------------
    #
    # load_folder_group() below is the synchronous form, kept for direct/
    # headless use. The frontend uses these three instead: a ~108-second
    # js_api call would leave the UI with no way to ask anything else in
    # the meantime, including "stop". See folder_load_job.py.

    _IDLE_PROGRESS = IDLE_STATE

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

        # The 42GB incident this app's memory work exists because of was a
        # folder load, so this is the operation with the strongest claim on
        # the ceiling — even though its per-file rewrite has since made it
        # bounded (1972 MiB measured on 250 real files). A ceiling protects
        # against the regression, not just the known case.
        self._loaded_group_index = group_index
        self._loaded_group_signature = group.signature
        self._folder_load_job = self._start_job(
            "folder_load",
            {
                "files": [str(f) for f in group.files],
                "folder_path": str(folder_path),
                "row_count": group.row_count,
                "cache_path": self._cache_path(),
            },
            label="Loading folder",
            files_total=group.file_count,
        )
        self._folder_load_adopted = False
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
            self._adopt_folder_load(state)
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

    def _adopt_folder_load(self, state: dict[str, Any]) -> None:
        """Take on the dataset the worker just profiled.

        The frame is rebuilt here from the file list the worker reports —
        `used_files`, which after a stop is the files that actually
        finished, not the group's full set. Rebuilding is free (pl.concat
        of lazy scans reads nothing) and is the only way the counts on
        screen and the frame behind them can be guaranteed to describe the
        same files, now that the worker holding the original is gone.
        """
        result = state["result"]
        used_files = [Path(p) for p in result.get("used_files") or []]
        self._current_lazy = self._rebuild_frame(used_files)
        # Structure problems (a deviant chunk, a series whose files are
        # not in time order) ride along with the dataset info so the
        # Findings tab can report them the same way it reports everything
        # else — they are things to tell the user, not load failures.
        if self._structure is not None:
            result["structure_findings"] = list(self._structure.findings)
            result["scope"] = self._scope.to_dict()
        self._current_path = str(self._folder_path)
        self._current_dataset_dir = Path(self._folder_path)
        self._current_row_count = result["row_count"]
        self._current_files = used_files
        self._chart_filter = None  # a new dataset never inherits a filter
        self._chart_filter_conditions = []
        self._search_predicate = None
        self._dataset_info = result
        self._text_columns = list(result.get("text_columns") or [])
        self._chart_data = None  # stale for the newly loaded dataset
        self._fleet_regrouped = None
        # The worker measured its own peak_wset; the Job Object measured
        # the same run against the ceiling it was actually held to. The
        # second is the honest number for this one load, so it replaces
        # the first in the stat line.
        result["load_stats"] = build_load_stats(
            row_count=result["row_count"],
            file_count=result.get("files_used"),
            elapsed_seconds=(result.get("load_stats") or {}).get("elapsed_seconds"),
            peak_bytes=state.get("peak_bytes"),
            ceiling_bytes=self._ceiling_mb() * 1024 * 1024 if state.get("capped") else None,
            # How much of this load was actually read, versus answered from
            # per-file profiles cached on a previous open.
            files_scanned=result.get("files_scanned"),
            files_from_cache=result.get("files_from_cache"),
        )

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
        # The file list, not just the frame. Every multi-file computation
        # in this app works per file rather than querying the concat, and
        # a worker subprocess rebuilds the dataset from exactly this list —
        # without it, a folder loaded through this synchronous path looks
        # like a single file whose "path" is a directory.
        self._current_files = list(group.files)
        self._loaded_group_index = group_index
        self._loaded_group_signature = group.signature

        if group.row_count is not None:
            self._current_row_count = group.row_count
            checkpoint("group_selection:row_count_from_metadata", f"row_count={self._current_row_count}")
        else:
            checkpoint("group_selection:row_count_streaming_start")
            self._current_row_count = row_count(lf)
            checkpoint("group_selection:row_count_streaming_done", f"row_count={self._current_row_count}")
        self._chart_data = None
        self._fleet_regrouped = None

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
