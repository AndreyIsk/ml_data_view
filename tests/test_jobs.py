"""The background job runner (ml_data_view/jobs.py).

These tests spawn real subprocesses. That is deliberate and not incidental
cost: the behaviours being verified are an operating system refusing a
process's memory, a process being terminated, and data crossing between
two processes through files. A mocked runner would assert that the mock
works.

The ceiling tests skip where the OS cannot enforce one (see
jobs.ceiling_status) rather than pretending. Windows is the target; the
rest of the module is expected to work everywhere and is tested
everywhere.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import pytest

from ml_data_view import jobs
from ml_data_view.jobs import Job, JobManager

needs_ceiling = pytest.mark.skipif(
    not jobs.memory_ceiling_supported(),
    reason="no OS-enforced memory ceiling on this platform (see jobs.ceiling_status)",
)


def _run(job: Job, timeout: float = 180) -> dict:
    job.start()
    assert job.wait(timeout), "job never settled"
    return job.progress()


# --- resolving the ceiling ---------------------------------------------


def test_default_ceiling_is_half_of_physical_ram(monkeypatch):
    monkeypatch.setattr(jobs, "total_physical_memory_bytes", lambda: 32 * 1024**3)

    assert jobs.default_memory_ceiling_mb() == 16 * 1024


def test_default_ceiling_never_drops_below_the_minimum(monkeypatch):
    """A 2GB machine would otherwise get a 1GB ceiling, and the import
    baseline alone (polars + pyarrow) does not fit in that."""
    monkeypatch.setattr(jobs, "total_physical_memory_bytes", lambda: 2 * 1024**3)

    assert jobs.default_memory_ceiling_mb() == jobs.MIN_MEMORY_CEILING_MB


def test_default_ceiling_without_psutil_is_not_the_minimum(monkeypatch):
    """Falling back to 2GB on a machine whose RAM simply could not be read
    would make every large folder fail for no reason."""
    monkeypatch.setattr(jobs, "total_physical_memory_bytes", lambda: None)

    assert jobs.default_memory_ceiling_mb() == jobs.DEFAULT_FALLBACK_CEILING_MB
    assert jobs.default_memory_ceiling_mb() > jobs.MIN_MEMORY_CEILING_MB


def test_ceiling_message_names_the_limit_and_what_to_do():
    assert jobs.ceiling_message(8192) == (
        "Stopped at the 8 GB ceiling. "
        "Raise the limit in Settings or narrow the selection."
    )


@pytest.mark.parametrize(
    "byte_count,expected",
    [
        (8 * 1024**3, "8 GB"),
        (int(1.4 * 1024**3), "1.4 GB"),
        (2048 * 1024 * 1024, "2 GB"),
    ],
)
def test_format_gb_keeps_a_round_limit_round(byte_count, expected):
    assert jobs.format_gb(byte_count) == expected


def test_ceiling_status_explains_itself_when_unsupported(monkeypatch):
    """Off Windows the app must SAY the ceiling is not enforced. Claiming a
    protection that is not there is worse than offering none."""
    monkeypatch.setattr(jobs, "_win32_job_modules", lambda: None)

    status = jobs.ceiling_status()

    assert status["supported"] is False
    assert status["reason"], "an unsupported ceiling must explain why"
    assert jobs.memory_ceiling_supported() is False


# --- the ETA -------------------------------------------------------------


def test_eta_comes_from_per_file_throughput():
    """10 files in 10s, 40 to go — 40s, not an extrapolation of a weighted
    progress bar."""
    assert jobs.estimate_eta(10.0, fraction=0.2, files_done=10, files_total=50) == 40.0


def test_eta_falls_back_to_the_fraction_when_work_is_not_files():
    """Deep checks count checks, not files."""
    assert jobs.estimate_eta(10.0, fraction=0.5, files_done=0, files_total=0) == 10.0


def test_no_eta_in_the_first_second():
    """One sample of a process that is still importing polars is noise
    presented as a measurement."""
    assert jobs.estimate_eta(0.4, fraction=0.5, files_done=5, files_total=10) is None


def test_no_eta_before_any_progress():
    assert jobs.estimate_eta(30.0, fraction=0.0, files_done=0, files_total=100) is None


def test_format_eta_omits_an_unknown_estimate():
    assert jobs.format_eta(None) == ""
    assert jobs.format_eta(113.0) == "about 1m 53s left"


# --- the temp-file channel ------------------------------------------------


def test_results_round_trip_as_json(tmp_path):
    state = {"stage": "done", "finished": True, "result": {"rows": [1, 2, 3], "name": "x"}}

    jobs.write_result(tmp_path, state)

    assert jobs.read_result(tmp_path) == state


def test_results_round_trip_as_parquet(tmp_path):
    """Tabular results take the parquet side of the channel: a million rows
    of JSON has to exist twice in memory to be produced, which is the exact
    thing the file channel exists to avoid."""
    import polars as pl

    rows = [{"a": index, "b": f"v{index}"} for index in range(500)]
    jobs.write_table_result(tmp_path, {"stage": "done", "finished": True}, pl.DataFrame(rows))

    state = jobs.read_result(tmp_path)

    assert state["stage"] == "done"
    assert state["result"] == rows


def test_reading_a_result_that_was_never_written_is_not_an_error(tmp_path):
    """This is what a ceiling kill looks like from the parent: the child
    died before it could write anything. It has to be a distinguishable
    outcome, not an exception."""
    assert jobs.read_result(tmp_path) is None


def test_progress_is_published_without_the_result(tmp_path):
    """Progress is rewritten many times a second; the result can be a
    multi-megabyte chart payload and travels exactly once."""
    jobs.write_progress(tmp_path, {"fraction": 0.5, "result": {"huge": "payload"}})

    published = json.loads((tmp_path / jobs.PROGRESS_FILENAME).read_text(encoding="utf-8"))

    assert published == {"fraction": 0.5}
    assert jobs.read_progress(tmp_path) == {"fraction": 0.5}


def test_reading_progress_before_the_child_writes_any_is_none(tmp_path):
    assert jobs.read_progress(tmp_path) is None


# --- running a job end to end ---------------------------------------------


def test_a_job_runs_to_completion_and_returns_its_result():
    state = _run(Job("_slow_loop", {"steps": 3, "step_seconds": 0.01}, label="Ticking", limit_mb=2048))

    assert state["finished"] is True
    assert state["error"] is None
    assert state["result"] == {"steps_done": 3}
    assert state["stage"] == "done"


def test_a_job_reports_progress_while_it_runs():
    job = Job(
        "_slow_loop",
        {"steps": 40, "step_seconds": 0.05},
        label="Ticking",
        limit_mb=2048,
        files_total=40,
    )
    job.start()

    seen = []
    while not job.wait(0.1):
        state = job.progress()
        seen.append((state["fraction"], state["files_done"]))
    job.wait(60)

    advanced = [entry for entry in seen if entry[1] > 0]
    assert advanced, f"no progress was ever published: {seen[:5]}"
    assert [f for f, _ in advanced] == sorted(f for f, _ in advanced), "progress went backwards"
    assert max(done for _, done in advanced) <= 40


def test_a_jobs_temp_directory_does_not_outlive_it():
    job = Job("_slow_loop", {"steps": 2, "step_seconds": 0.01}, label="Ticking", limit_mb=2048)
    _run(job)

    assert not job.directory.exists()


def test_an_unknown_task_is_an_error_not_a_hang():
    state = _run(Job("no_such_task", {}, label="Nothing", limit_mb=2048))

    assert state["finished"] is True
    assert "no_such_task" in state["error"]


# --- cancellation ---------------------------------------------------------


def test_cancel_keeps_what_a_cooperative_job_finished():
    """The semantics the in-process jobs already promised: a stopped folder
    load is a SMALLER dataset, cancelled deep checks keep the checks that
    finished. Killing the worker outright would throw that away."""
    job = Job(
        "_slow_loop",
        {"steps": 500, "step_seconds": 0.02},
        label="Ticking",
        limit_mb=2048,
        files_total=500,
    )
    job.start()
    time.sleep(3)

    job.cancel()
    assert job.wait(30), "a cooperative cancel must not need the forced kill"
    state = job.progress()

    assert state["cancelled"] is True
    assert 0 < state["result"]["steps_done"] < 500


def test_cancel_terminates_a_job_that_ignores_it_within_the_grace_period():
    """The guarantee is that Cancel cancels. A job with no safe boundary —
    PDF generation is the real one — is force-terminated instead."""
    job = Job(
        "_ignores_cancel",
        {"seconds": 300},
        label="Stubborn",
        limit_mb=2048,
        cancel_grace_seconds=2.0,
    )
    job.start()
    time.sleep(2)

    started_at = time.monotonic()
    job.cancel()
    assert job.wait(30), "a cancelled job must settle"
    elapsed = time.monotonic() - started_at

    assert elapsed < 15, f"cancellation took {elapsed:.1f}s"
    assert job.progress()["cancelled"] is True


def test_a_forcibly_cancelled_job_leaves_no_partial_outputs():
    job = Job(
        "_ignores_cancel",
        {"seconds": 300},
        label="Stubborn",
        limit_mb=2048,
        cancel_grace_seconds=1.0,
    )
    job.start()
    time.sleep(1.5)
    directory = job.directory
    job.cancel()
    job.wait(30)

    assert not directory.exists(), "a killed job's temp files must not survive to be mistaken for output"
    assert job.progress()["result"] is None


# --- the ceiling ----------------------------------------------------------


@needs_ceiling
def test_a_job_that_exceeds_the_ceiling_is_stopped_with_a_message_not_an_exception():
    """A ceiling hit is an expected outcome. The user gets a sentence that
    names the limit and what to do about it; nothing raises."""
    job = Job(
        "_memory_hog",
        {"chunk_mb": 16, "max_chunks": 4096},
        label="Allocating",
        limit_mb=jobs.MIN_MEMORY_CEILING_MB,
    )

    state = _run(job)

    assert state["finished"] is True
    assert state["ceiling_hit"] is True
    assert state["stage"] == "ceiling"
    assert state["error"] == jobs.ceiling_message(jobs.MIN_MEMORY_CEILING_MB)
    assert "Raise the limit in Settings" in state["label"]


@needs_ceiling
def test_a_ceiling_kill_cleans_up_its_temp_files():
    job = Job(
        "_memory_hog",
        {"chunk_mb": 16, "max_chunks": 4096},
        label="Allocating",
        limit_mb=jobs.MIN_MEMORY_CEILING_MB,
    )
    directory = job.directory

    _run(job)

    assert not directory.exists()


@needs_ceiling
def test_a_job_that_fits_under_the_ceiling_keeps_its_result():
    """The counterpart the ceiling tests need: the cap must not be reported
    for a job that simply ran."""
    job = Job(
        "_memory_hog",
        {"chunk_mb": 16, "max_chunks": 4},
        label="Allocating",
        limit_mb=jobs.MIN_MEMORY_CEILING_MB,
    )

    state = _run(job)

    assert state["ceiling_hit"] is False
    assert state["result"] == {"allocated_mb": 64}


@needs_ceiling
def test_the_ceiling_does_not_stop_the_parent_process():
    """The whole reason this is a subprocess: the runaway dies, the caller
    carries on. Asserting we got here at all after the kill is the test."""
    _run(
        Job(
            "_memory_hog",
            {"chunk_mb": 16, "max_chunks": 4096},
            label="Allocating",
            limit_mb=jobs.MIN_MEMORY_CEILING_MB,
        )
    )

    follow_up = _run(Job("_slow_loop", {"steps": 1, "step_seconds": 0.01}, label="After", limit_mb=2048))
    assert follow_up["result"] == {"steps_done": 1}


# --- the manager the UI reads --------------------------------------------


def test_the_manager_lists_a_running_job_with_progress_and_a_cancel_handle():
    manager = JobManager()
    job = manager.register(
        Job(
            "_slow_loop",
            {"steps": 200, "step_seconds": 0.05},
            label="Ticking",
            limit_mb=2048,
            files_total=200,
        )
    )
    job.start()
    try:
        time.sleep(2)
        listed = manager.snapshot()

        assert len(listed) == 1
        assert listed[0]["id"] == job.id
        assert listed[0]["title"] == "Ticking"
        assert listed[0]["files_total"] == 200
        assert manager.cancel(job.id) is True
    finally:
        job.cancel()
        job.wait(30)


def test_the_manager_drops_jobs_once_they_finish():
    """The indicator is a live view, not a history — a finished job must
    not leave a row with a Cancel button on it."""
    manager = JobManager()
    job = manager.register(
        Job("_slow_loop", {"steps": 1, "step_seconds": 0.01}, label="Ticking", limit_mb=2048)
    )
    job.start()
    job.wait(60)

    assert manager.snapshot() == []
    assert manager.snapshot() == []
    assert manager.cancel(job.id) is False


# --- how the worker is launched ------------------------------------------


def test_the_worker_runs_the_module_when_not_frozen(monkeypatch):
    from ml_data_view import resources

    monkeypatch.setattr(resources, "is_frozen", lambda: False)

    assert jobs.worker_command(Path("job-dir"))[1:] == [
        "-m",
        "ml_data_view.jobs",
        "job-dir",
    ]


def test_the_worker_is_the_app_itself_when_frozen(monkeypatch):
    """Frozen, sys.executable IS the app. Without the flag, starting a job
    would open a second window instead of doing the work — which is why
    main.py checks for it before creating one."""
    from ml_data_view import resources

    monkeypatch.setattr(resources, "is_frozen", lambda: True)

    command = jobs.worker_command(Path("job-dir"))

    assert command[1:] == [jobs.WORKER_FLAG, "job-dir"]
    assert "-m" not in command


def test_main_dispatches_the_worker_flag_before_opening_a_window():
    """Read as source rather than executed: running it either opens a
    window or runs a job, and neither belongs in a unit test."""
    source = (Path(__file__).resolve().parents[1] / "ml_data_view" / "main.py").read_text(
        encoding="utf-8"
    )
    worker_check = source.index("WORKER_FLAG")
    window_creation = source.index("webview.create_window")

    assert worker_check < window_creation


# --- the diagnostics script that used to own this code --------------------


def test_job_cap_still_exports_its_original_surface():
    """PROJECT_BRIEF.md's measurement notes quote `python scripts/job_cap.py
    --limit-mb ... -- ...` and `from scripts.job_cap import run_capped`."""
    from scripts import job_cap

    assert job_cap.run_capped is jobs.run_capped
    assert job_cap.JobCapResult is jobs.CappedResult
    assert callable(job_cap.main)
