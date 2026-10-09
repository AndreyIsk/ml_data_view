"""What a capped worker subprocess actually runs.

One entry per heavy operation. Each is a thin adapter: rebuild the inputs
from the JSON spec the parent wrote, construct the SAME in-process job
class the app has always used, and republish its `progress()` through
jobs.py's temp-file channel.

Two rules shape every task here:

  * NOTHING LARGE CROSSES THE SPEC. A job is described by file paths,
    row counts and condition dicts — never by a serialized LazyFrame or a
    materialized frame. `pl.scan_parquet` / `pl.concat(..., how="vertical")`
    are lazy, so rebuilding the frame in the child costs no reads (see
    PROJECT_BRIEF.md's architecture notes) and the child reads the data
    itself, inside the ceiling, which is the entire point.

  * THE WORK IS NOT REIMPLEMENTED HERE. ChartBuildJob, DeepCheckJob,
    FolderLoadJob and CompareJob run unchanged, so their existing tests
    still cover the real code path and the progress dicts the frontend
    polls keep exactly the shape they had. A second implementation of
    "build the charts, but for a subprocess" is how the two would drift.

Cancellation is cooperative: the parent drops a sentinel file, the loop
below sees it and calls the job's own `cancel()`, which lands at the next
item boundary — so a stopped folder load is still a smaller-but-honest
dataset and cancelled deep checks still keep the checks that finished.
jobs.py force-terminates only when that is ignored.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

import polars as pl

from ml_data_view import jobs

# How often the child republishes its in-process job's progress. Well
# under the UI's 200ms poll, so the file the parent reads is never more
# than one tick stale, and cheap: it is a few hundred bytes.
PUBLISH_INTERVAL_S = 0.1


def _frame_from(params: dict[str, Any]) -> tuple[pl.LazyFrame, list[Path] | None]:
    """The dataset the job runs against, plus its file list.

    `files` is present for a folder load and absent for a single opened
    file; the distinction matters downstream, because every multi-file
    computation in this app works per file rather than querying the
    concatenation.
    """
    files = params.get("files")
    if files:
        identity = params.get("identity")
        if identity:
            # A structured folder: each chunk's scan carries its derived
            # columns, so the worker's frame is identical to the parent's.
            # Paths and strings in the spec, as always — never a frame.
            from ml_data_view.hierarchy import scope_lazyframe

            return scope_lazyframe(identity), [Path(f) for f in files]
        # The same concat the parent would have built — folder_loader is
        # the one place that knows how to turn a file list into a frame,
        # and it is lazy, so rebuilding it here reads nothing.
        from ml_data_view.folder_loader import load_files

        return load_files([Path(f) for f in files]), [Path(f) for f in files]
    from ml_data_view.data_loader import scan_dataframe

    return scan_dataframe(params["path"]), None


def _predicate_from(params: dict[str, Any], lf: pl.LazyFrame):
    """Rebuild the chart filter from its condition dicts.

    The conditions travel, not the polars expression: an Expr is not JSON,
    and api.py already keeps `_chart_filter_conditions` beside the
    predicate for exactly this kind of reuse.
    """
    conditions = params.get("conditions")
    if not conditions:
        return None
    from ml_data_view.search import Condition, build_predicate

    return build_predicate(
        [Condition.from_dict(c) for c in conditions], lf.collect_schema()
    )


class Task:
    """A job class plus the loop that publishes it.

    `build()` is called inside the child, so every import it triggers
    (polars, pyarrow, sklearn, reportlab) happens under the ceiling rather
    than in the UI process.
    """

    def __init__(self, build: Callable[[], Any]) -> None:
        self._build = build

    def run(self, job_dir: Path) -> None:
        job = self._build()
        # Checked BEFORE start, not only in the loop below: a subprocess
        # takes about a second to reach this line, which is long enough for
        # the user to have clicked Cancel already, and long enough for a
        # small job to finish before the loop's first tick. Either way the
        # answer must be "cancelled", not "done".
        if jobs.cancel_requested(job_dir):
            job.cancel()
        job.start()
        while not job.wait(PUBLISH_INTERVAL_S):
            if jobs.cancel_requested(job_dir):
                job.cancel()
            jobs.write_progress(job_dir, job.progress())
        state = job.progress()
        jobs.write_progress(job_dir, state)
        jobs.write_result(job_dir, state)


# --- chart building (distributions + embedding projection) -------------


def _chart_build(params: dict[str, Any]):
    from ml_data_view.chart_build_job import ChartBuildJob

    lf, files = _frame_from(params)
    dataset_dir = params.get("dataset_dir")
    cache_path = params.get("cache_path")
    return ChartBuildJob(
        lf,
        total_rows=params.get("total_rows"),
        dataset_dir=Path(dataset_dir) if dataset_dir else None,
        null_counts=params.get("null_counts") or None,
        files=files,
        predicate=_predicate_from(params, lf),
        # The cache lives in the user's app-data directory, which the
        # worker can reach on its own — but the path travels in the spec so
        # a test Api can point it somewhere else, exactly as it does for
        # settings.
        cache_path=cache_path,
        identity=params.get("identity") or None,
        timestamp_column=params.get("timestamp_column") or None,
        timeseries_thresholds=params.get("timeseries_thresholds") or None,
        timeseries_max_channels=int(params.get("timeseries_max_channels") or 12),
        timeseries_files=[Path(f) for f in (params.get("timeseries_files") or [])] or None,
        structure_labels=params.get("structure_labels") or None,
        grouping_column=params.get("grouping_column") or None,
        fleet_thresholds=params.get("fleet_thresholds") or None,
        fleet_max_channels=int(params.get("fleet_max_channels") or 12),
    )


# --- deep findings checks ----------------------------------------------


def _deep_checks(params: dict[str, Any]):
    from ml_data_view.deep_checks import DeepContext
    from ml_data_view.findings_job import DeepCheckJob

    lf, files = _frame_from(params)
    dataset_dir = params.get("dataset_dir")
    return DeepCheckJob(
        DeepContext(
            lf=lf,
            files=files,
            dataset_dir=Path(dataset_dir) if dataset_dir else None,
            image_columns=list(params.get("image_columns") or []),
        )
    )


# --- folder load --------------------------------------------------------


def _folder_load(params: dict[str, Any]):
    from ml_data_view.folder_load_job import FolderLoadJob
    from ml_data_view.folder_loader import SchemaGroup

    files = [Path(f) for f in params["files"]]
    # The signature is only read for its file list and count here, so it is
    # rebuilt empty rather than round-tripping polars dtypes through JSON —
    # the job re-reads the real schema from the first file anyway.
    group = SchemaGroup(signature=(), files=files, row_count=params.get("row_count"))
    return FolderLoadJob(
        group,
        Path(params["folder_path"]),
        cache_path=params.get("cache_path"),
        identity=params.get("identity") or None,
        timestamp_column=params.get("timestamp_column") or None,
    )


# --- dataset comparison -------------------------------------------------


def _compare(params: dict[str, Any]):
    from ml_data_view.compare import COMPARE_BINS, COMPARE_TOP_K
    from ml_data_view.compare_jobs import CompareJob

    return CompareJob(
        [Path(f) for f in params["files_a"]],
        [Path(f) for f in params["files_b"]],
        bins=params.get("bins") or COMPARE_BINS,
        top_k=params.get("top_k") or COMPARE_TOP_K,
    )


# --- PDF report generation ----------------------------------------------


class _CallableJob:
    """The polled-job shape around a single blocking call.

    PDF generation has no natural progress: reportlab lays the document out
    in one pass and kaleido rasterizes each chart through a real Chromium.
    Rather than invent a fake bar, this reports the one honest thing — that
    it is running — and is not cancellable at a boundary, because there
    isn't one. jobs.py's forced terminate still applies, and killing the
    Job Object takes kaleido's browser down with it rather than orphaning
    it, which is exactly why PDF export belongs in a job at all.
    """

    def __init__(self, label: str, fn: Callable[[], Any]) -> None:
        self._label = label
        self._fn = fn
        self._result: Any = None
        self._error: str | None = None
        self._stage = "pending"
        import threading

        self._done = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        import threading

        self._stage = "running"
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self) -> None:
        try:
            self._result = self._fn()
            self._stage = "done"
        except Exception as exc:
            # MemoryError stringifies to nothing, and a job that failed
            # must never report an empty reason — that is the case where
            # the ceiling most often fired.
            self._error = str(exc) or type(exc).__name__
            self._stage = "error"
        finally:
            self._done.set()

    def cancel(self) -> None:
        """No safe boundary exists mid-document — see the class docstring.
        The parent's forced terminate is what stops this."""

    def wait(self, timeout: float | None = None) -> bool:
        return self._done.wait(timeout)

    def progress(self) -> dict[str, Any]:
        finished = self._done.is_set()
        return {
            **jobs.IDLE_STATE,
            "stage": self._stage,
            "label": self._label if not finished else (self._error or "Done."),
            "fraction": 1.0 if self._stage == "done" else 0.0,
            "finished": finished,
            "error": self._error,
            "result": self._result,
        }


def _report(params: dict[str, Any]):
    from ml_data_view.report import generate_report as write_report_pdf

    def run() -> dict[str, Any]:
        path = write_report_pdf(
            params["output_path"],
            dataset_info=params["dataset_info"],
            chart_data=params["chart_data"],
            sections=list(params.get("sections") or []),
            banner=params.get("banner"),
        )
        return {"path": str(path)}

    return _CallableJob("Generating PDF report…", run)


# --- a deliberately memory-hungry task, for the tests --------------------


def _memory_hog(params: dict[str, Any]):
    """Allocates until something stops it. Exists so the ceiling can be
    tested against a job that really does exceed it, rather than against a
    mocked one — the failure mode being verified is the OS killing a
    process, which nothing short of a real process exercises."""

    def run() -> dict[str, Any]:
        chunk_mb = int(params.get("chunk_mb") or 64)
        limit_chunks = int(params.get("max_chunks") or 4096)
        held = []
        for _ in range(limit_chunks):
            held.append(bytearray(chunk_mb * 1024 * 1024))
            # Touch it: a bytearray is committed on allocation, but writing
            # makes that true of every page regardless of platform.
            held[-1][::4096] = b"\x01" * len(held[-1][::4096])
        return {"allocated_mb": chunk_mb * limit_chunks}

    return _CallableJob("Allocating…", run)


def _slow_loop(params: dict[str, Any]):
    """A cancellable job that does nothing but tick, for testing the
    cancel path end to end without a fixture."""
    from ml_data_view.background_job import BackgroundJob

    steps = int(params.get("steps") or 200)
    step_seconds = float(params.get("step_seconds") or 0.1)

    class SlowJob(BackgroundJob):
        def __init__(self) -> None:
            super().__init__([Path(f"step-{i}") for i in range(steps)])

        def _work(self) -> dict[str, Any]:
            import time

            self._update(stage="ticking")
            done = 0
            for index in range(steps):
                if self._cancel.is_set():
                    break
                time.sleep(step_seconds)
                done = index + 1
                self._note_file(index, Path(f"step-{index}"), "Ticking")
            return {"steps_done": done}

    return SlowJob()


def _ignores_cancel(params: dict[str, Any]):
    """Never stops when asked — proves the forced-terminate backstop is
    what bounds cancellation time, not the job's own cooperation."""

    def run() -> dict[str, Any]:
        import time

        time.sleep(float(params.get("seconds") or 300))
        return {"finished": True}

    return _CallableJob("Ignoring cancellation…", run)


def _echo_table(params: dict[str, Any]):
    """Returns tabular rows through the parquet side of the channel, which
    is the path an export/transform feature takes."""

    class TableJob(_CallableJob):
        def __init__(self) -> None:
            super().__init__("Building table…", lambda: None)

        def _run(self) -> None:
            self._result = pl.DataFrame(params["rows"])
            self._stage = "done"
            self._done.set()

    return TableJob()


class _TableTask(Task):
    """A task whose result is rows, written to parquet beside the JSON
    envelope rather than embedded in it (see jobs.write_table_result)."""

    def run(self, job_dir: Path) -> None:
        job = self._build()
        if jobs.cancel_requested(job_dir):
            job.cancel()
        job.start()
        while not job.wait(PUBLISH_INTERVAL_S):
            if jobs.cancel_requested(job_dir):
                job.cancel()
            jobs.write_progress(job_dir, job.progress())
        state = job.progress()
        table = state.pop("result", None)
        jobs.write_progress(job_dir, state)
        if table is None:
            jobs.write_result(job_dir, {**state, "result": None})
        else:
            jobs.write_table_result(job_dir, state, table)


TASKS: dict[str, Callable[[dict[str, Any]], Any]] = {
    "chart_build": _chart_build,
    "deep_checks": _deep_checks,
    "folder_load": _folder_load,
    "compare": _compare,
    "report": _report,
    # Test-only tasks. They live in the real registry rather than behind a
    # test hook because the behaviour under test IS the real runner: a
    # ceiling kill, a forced cancel, a parquet round trip.
    "_memory_hog": _memory_hog,
    "_slow_loop": _slow_loop,
    "_ignores_cancel": _ignores_cancel,
    "_echo_table": _echo_table,
}

_TABLE_TASKS = {"_echo_table"}


def build_task(name: str, params: dict[str, Any]) -> Task:
    factory = TASKS.get(name)
    if factory is None:
        raise KeyError(f"Unknown job task {name!r}.")
    cls = _TableTask if name in _TABLE_TASKS else Task
    return cls(lambda: factory(params))
