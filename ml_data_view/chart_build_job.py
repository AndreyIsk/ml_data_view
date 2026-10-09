"""Chart building on a worker thread, with progress.

Chart generation runs after a folder load and takes ~0.54s per file
(measured) — minutes on a large folder, during which the UI previously
showed only a static "Computing charts…" spinner. Same shape as
folder_load_job.py: start() returns at once, the UI polls progress(), and
the js_api bridge is never blocked by a long call.

No cancellation here, unlike the folder load. A half-built chart payload
has no coherent meaning — some columns charted, others silently absent,
with nothing on screen to distinguish "no chart for this column" from
"stopped before reaching it". The folder load can stop safely because a
truncated file list is still an honest, complete dataset; this cannot.
"""
from __future__ import annotations

import threading
from pathlib import Path
from typing import Any

import polars as pl

from ml_data_view.chart_data import build_chart_payload


class ChartBuildJob:
    """One chart-payload build, running on its own thread. Single-use."""

    def __init__(
        self,
        lf: pl.LazyFrame,
        total_rows: int | None = None,
        dataset_dir: Path | None = None,
        null_counts: dict[str, int] | None = None,
        files: list[Path] | None = None,
        predicate=None,
        cache_path: str | Path | None = None,
        identity: list[dict] | None = None,
        timestamp_column: str | None = None,
        timeseries_thresholds: dict | None = None,
        timeseries_max_channels: int = 12,
        timeseries_files: list[Path] | None = None,
        structure_labels: dict | None = None,
        grouping_column: str | None = None,
        fleet_thresholds: dict | None = None,
        fleet_max_channels: int = 12,
    ) -> None:
        self._lf = lf
        self._total_rows = total_rows
        self._dataset_dir = dataset_dir
        self._null_counts = null_counts
        self._files = files
        self._predicate = predicate
        # When set (and the build is unfiltered), build_chart_payload
        # assembles the payload by merging cached per-file statistics and
        # reads only the files whose entries are missing or stale.
        self._cache_path = cache_path
        # Per-chunk derived-column values when the folder has a detected
        # structure. Their statistics are SYNTHESIZED from row counts
        # rather than read (see file_profile), so this costs nothing.
        self._identity = identity
        # The detected timestamp column, and what the user set the
        # time-series thresholds to. Absent, the whole time-series pass is
        # skipped — which is the applicability rule, not an optimisation.
        self._timestamp_column = timestamp_column
        self._timeseries_thresholds = timeseries_thresholds
        self._timeseries_max_channels = timeseries_max_channels
        # The chunks the time-series pass reads. Equal to `files` for a
        # folder; for a single opened file there is no file list, and this
        # is how that one file still gets analysed.
        self._timeseries_files = timeseries_files
        # The mapping's own words for its levels, and which column a group
        # comparison runs over. Both are labels and a name — no data — and
        # they are what let the catalogue and the fleet comparison be
        # built in here from statistics already merged.
        self._structure_labels = structure_labels
        self._grouping_column = grouping_column
        self._fleet_thresholds = fleet_thresholds
        self._fleet_max_channels = fleet_max_channels
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self._done = threading.Event()
        self._state: dict[str, Any] = {
            "stage": "pending",
            "label": "Waiting to start…",
            "fraction": 0.0,
            "finished": False,
            "error": None,
            "result": None,
        }
        # Test seam, as in folder_load_job: called with a snapshot after
        # every update so a test can observe progress without sleeping.
        self._on_progress_for_test = None

    def start(self) -> None:
        if self._thread is not None:
            raise RuntimeError("This ChartBuildJob has already been started.")
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def wait(self, timeout: float | None = None) -> bool:
        return self._done.wait(timeout)

    def progress(self) -> dict[str, Any]:
        with self._lock:
            return dict(self._state)

    def _update(self, **changes: Any) -> None:
        with self._lock:
            self._state.update(changes)
            snapshot = dict(self._state)
        if self._on_progress_for_test is not None:
            self._on_progress_for_test(snapshot)

    def _run(self) -> None:
        try:
            payload = build_chart_payload(
                self._lf,
                total_rows=self._total_rows,
                dataset_dir=self._dataset_dir,
                null_counts=self._null_counts,
                files=self._files,
                predicate=self._predicate,
                cache_path=self._cache_path,
                identity=self._identity,
                timestamp_column=self._timestamp_column,
                timeseries_thresholds=self._timeseries_thresholds,
                timeseries_max_channels=self._timeseries_max_channels,
                timeseries_files=self._timeseries_files,
                structure_labels=self._structure_labels,
                grouping_column=self._grouping_column,
                fleet_thresholds=self._fleet_thresholds,
                fleet_max_channels=self._fleet_max_channels,
                on_progress=lambda fraction, label: self._update(
                    stage="building", fraction=round(fraction, 4), label=label
                ),
            )
            self._update(stage="done", fraction=1.0, label="Charts ready.", result=payload)
        except Exception as exc:  # a worker thread must never die silently
            self._update(
                stage="error", label=f"Chart generation failed: {exc}", error=str(exc), result=None
            )
        finally:
            self._update(finished=True)
            self._done.set()
