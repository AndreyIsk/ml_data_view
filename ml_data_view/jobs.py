"""Background job runner: heavy work in a capped worker subprocess.

Every expensive operation in this app now runs here — outside the UI
process, under a hard, OS-enforced memory ceiling, cancellable, reporting
progress through a file the parent polls. The Job Object wrapper that used
to live in scripts/job_cap.py as a diagnostics tool is the enforcement
layer; scripts/job_cap.py is now a thin re-export of it.

WHY A SUBPROCESS AND NOT A THREAD
Threads share an address space, so a thread that allocates 42GB takes the
window down with it — that is exactly the incident (see PROJECT_BRIEF.md's
Stage 3 notes) this module exists to make impossible. A Job Object ceiling
can only be applied to a process, and killing a runaway is only survivable
if the thing killed is not also the UI.

THE CHANNEL
Nothing large crosses a pipe. The parent writes `spec.json` into a
per-job temp directory, the child writes `progress.json` (small, rewritten
atomically) while it works and `result.json` (or `result.parquet` for
tabular results) at the end. The parent polls the first and reads the
second exactly once, then deletes the directory. A pickled chart payload
through a pipe would mean holding the whole thing twice, in the one
process whose memory is NOT capped.

THE WORKER IS A WRAPPER, NOT A REWRITE
The child runs the same in-process job classes the app has always used
(ChartBuildJob, DeepCheckJob, FolderLoadJob, CompareJob — see
job_tasks.py), polls their `progress()`, and writes those dicts straight
out. So the state the frontend polls has exactly the shape it had before,
and every existing test of those classes still tests the real thing.

CANCEL IS GRACEFUL FIRST, THEN FORCED
`cancel()` writes a `cancel` sentinel; the in-process job sees it at its
next item boundary and returns what it has. That preserves the semantics
those jobs already promised — a stopped folder load is a SMALLER dataset,
cancelled deep checks keep the checks that finished. A child that does not
stop within CANCEL_GRACE_SECONDS is force-terminated through the Job
Object and its temp directory deleted, which is also what happens to a job
that never gets the chance to be graceful because the OS killed it first.

OFF WINDOWS
Job Objects are a Windows facility. Everywhere else the child still runs
as a subprocess with the same channel, cancel and progress behaviour —
just without the OS ceiling. `ceiling_status()` says so, Settings shows
it, and a warning is logged once. Windows is the target; the code must
not crash elsewhere.
"""
from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
import sys
import tempfile
import threading
import uuid
from dataclasses import dataclass
from pathlib import Path
from time import monotonic
from typing import Any, Callable

from ml_data_view.load_stats import format_duration

logger = logging.getLogger(__name__)

# --- the ceiling ------------------------------------------------------

# Floor for the user-configurable ceiling. Below roughly this, the import
# baseline alone (polars + pyarrow, ~1.1GB measured — see PROJECT_BRIEF.md)
# leaves no working room, so every job would die at startup and the
# setting would read as "break the app".
MIN_MEMORY_CEILING_MB = 2048
# Share of physical RAM the default ceiling takes. Half leaves the machine
# usable while the job runs, which is the whole point: the incident this
# guards against was not a slow app, it was a frozen desktop.
DEFAULT_CEILING_FRACTION = 0.5
DEFAULT_FALLBACK_CEILING_MB = 4096

DEFAULT_POLL_INTERVAL_S = 0.05
# How long a cancelled child gets to stop at its own boundary and write
# out what it finished, before it is terminated outright. Long enough for
# a per-file loop to reach the next file on a slow disk; short enough that
# "Cancel" never feels ignored.
CANCEL_GRACE_SECONDS = 5.0
# Share of the ceiling a failed job's peak must reach to be reported as a
# ceiling hit rather than an ordinary error. See Job._hit_ceiling.
CEILING_PEAK_THRESHOLD = 0.95

SPEC_FILENAME = "spec.json"
PROGRESS_FILENAME = "progress.json"
RESULT_FILENAME = "result.json"
TABLE_FILENAME = "result.parquet"
CANCEL_FILENAME = "cancel"

WORKER_FLAG = "--jobs-worker"


def format_gb(byte_count: float) -> str:
    """Bytes as a short "8 GB" / "1.4 GB" string.

    Deliberately coarser than load_stats.format_peak_memory's MiB/GiB: a
    memory ceiling is a round number the user typed, and quoting it back
    as "8.0 GiB" invites them to wonder whether it got rounded.
    """
    value = byte_count / (1024**3)
    return f"{value:.0f} GB" if abs(value - round(value)) < 0.05 else f"{value:.1f} GB"


def ceiling_message(limit_mb: int) -> str:
    """What the user sees when a job hits the ceiling.

    A ceiling hit is an expected outcome, not a crash: the whole point of
    setting one is that it fires. So this names the limit, and names the
    two things that actually change the result.
    """
    return (
        f"Stopped at the {format_gb(limit_mb * 1024 * 1024)} ceiling. "
        "Raise the limit in Settings or narrow the selection."
    )


def total_physical_memory_bytes() -> int | None:
    """Physical RAM, or None if psutil can't say."""
    try:
        import psutil
    except ImportError:
        return None
    try:
        return int(psutil.virtual_memory().total)
    except Exception:
        return None


def default_memory_ceiling_mb() -> int:
    """Half of physical RAM, never below MIN_MEMORY_CEILING_MB.

    Falls back to a fixed value when psutil can't read the machine's RAM,
    rather than to the minimum: a 2GB ceiling on a 64GB workstation would
    make every large folder fail for no reason.
    """
    total = total_physical_memory_bytes()
    if total is None:
        return DEFAULT_FALLBACK_CEILING_MB
    half_mb = int(total * DEFAULT_CEILING_FRACTION) // (1024 * 1024)
    return max(MIN_MEMORY_CEILING_MB, half_mb)


def _win32_job_modules():
    """The pywin32 pieces the ceiling needs, or None if unavailable.

    Imported through a function so this module stays importable on any
    platform and so a pywin32 that is installed but broken degrades the
    same way a missing one does.
    """
    if sys.platform != "win32":
        return None
    try:
        import pywintypes
        import win32event
        import win32job
        import win32process
    except ImportError:
        return None
    return pywintypes, win32event, win32job, win32process


_ceiling_warning_logged = False


def ceiling_status() -> dict[str, Any]:
    """Whether the OS ceiling is actually in force, and why not if it isn't.

    Settings renders this verbatim. A ceiling the app claims to enforce but
    doesn't would be worse than none at all, so the unsupported case is
    surfaced rather than silently ignored.
    """
    global _ceiling_warning_logged
    if _win32_job_modules() is not None:
        return {"supported": True, "reason": ""}
    if sys.platform != "win32":
        reason = (
            f"Memory ceilings are enforced by Windows Job Objects; on {sys.platform} "
            "background jobs run without an OS-enforced limit."
        )
    else:
        reason = (
            "pywin32 is not available, so the OS-enforced memory ceiling is off. "
            "Background jobs still run in a separate process."
        )
    if not _ceiling_warning_logged:
        logger.warning(reason)
        _ceiling_warning_logged = True
    return {"supported": False, "reason": reason}


def memory_ceiling_supported() -> bool:
    return bool(ceiling_status()["supported"])


# --- the capped child process -----------------------------------------


@dataclass
class CappedResult:
    pid: int
    exit_code: int | None
    killed_by_memory_limit: bool
    timed_out: bool
    peak_mb: float
    limit_mb: int
    # False when the OS ceiling could not be applied (see ceiling_status).
    # Callers must not present an uncapped run as a capped one.
    capped: bool = True


class CappedProcess:
    """A child process running under a hard memory ceiling.

    Two independent layers of enforcement on Windows, both scoped to this
    child only (never the parent or anything else on the machine):

      1. A Job Object commit-charge limit (JOB_OBJECT_LIMIT_PROCESS_MEMORY /
         JOB_OBJECT_LIMIT_JOB_MEMORY). The OS itself refuses the child's
         next allocation once the cap is hit — synchronous with the
         offending allocation, not a poll. On its own this is usually
         enough: a failed VirtualAlloc surfaces to Python as MemoryError
         and the process dies. It is not treated as sufficient, because a
         child that grows via many small allocations, or that catches
         MemoryError and limps on, would not necessarily die from this.

      2. A polling watchdog thread that reads the Job Object's own
         OS-maintained PeakProcessMemoryUsed/PeakJobMemoryUsed counters
         (not a live GetProcessMemoryInfo snapshot of the child — that was
         tried first and is racy: a child that dies in well under a second,
         as an allocation-failure crash typically does, is frequently
         already gone and reaped by the time a poll fires, reading back
         near-zero) and calls TerminateJobObject the moment either crosses
         the cap.

    JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE is also set, so if the parent dies
    (crash, Ctrl-C, kill) the child dies with it — nothing capped by this
    can be orphaned into an uncapped runaway.
    """

    def __init__(
        self,
        cmd: list[str],
        limit_mb: int,
        poll_interval_s: float = DEFAULT_POLL_INTERVAL_S,
        cwd: str | None = None,
        env: dict[str, str] | None = None,
    ) -> None:
        self._limit_mb = limit_mb
        self._limit_bytes = limit_mb * 1024 * 1024
        self._poll_interval_s = poll_interval_s
        self._killed_by_memory_limit = threading.Event()
        self._stop_watchdog = threading.Event()
        self._win32 = _win32_job_modules()
        self._popen: subprocess.Popen | None = None
        self._peak_bytes_fallback = 0
        self.pid = 0
        if self._win32 is not None:
            self._start_windows(cmd, cwd, env)
        else:
            self._start_portable(cmd, cwd, env)

    # --- Windows: Job Object + watchdog ---------------------------------

    def _start_windows(self, cmd: list[str], cwd: str | None, env: dict[str, str] | None) -> None:
        pywintypes, win32event, win32job, win32process = self._win32
        command_line = subprocess.list2cmdline(cmd)
        startup_info = win32process.STARTUPINFO()

        # CREATE_SUSPENDED so the child is assigned to the job BEFORE it
        # executes a single instruction — otherwise a fast allocation could
        # happen outside the cap.
        self._hProcess, self._hThread, self.pid, _tid = win32process.CreateProcess(
            None,
            command_line,
            None,
            None,
            True,
            win32process.CREATE_SUSPENDED,
            env,
            cwd,
            startup_info,
        )
        self._hJob = win32job.CreateJobObject(None, "")
        info = win32job.QueryInformationJobObject(
            self._hJob, win32job.JobObjectExtendedLimitInformation
        )
        info["BasicLimitInformation"]["LimitFlags"] = (
            win32job.JOB_OBJECT_LIMIT_PROCESS_MEMORY
            | win32job.JOB_OBJECT_LIMIT_JOB_MEMORY
            | win32job.JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
        )
        info["ProcessMemoryLimit"] = self._limit_bytes
        info["JobMemoryLimit"] = self._limit_bytes
        win32job.SetInformationJobObject(
            self._hJob, win32job.JobObjectExtendedLimitInformation, info
        )
        win32job.AssignProcessToJobObject(self._hJob, self._hProcess)

        self._watchdog_thread = threading.Thread(target=self._watchdog, daemon=True)
        self._watchdog_thread.start()
        win32process.ResumeThread(self._hThread)

    def _job_peak_bytes(self) -> int:
        """Peak commit charge the Job Object has recorded, across its
        per-process and job-wide counters. OS-maintained on the job itself,
        so — unlike GetProcessMemoryInfo on the child's handle — it is
        accurate even read after the child has already exited."""
        _pywintypes, _win32event, win32job, _win32process = self._win32
        info = win32job.QueryInformationJobObject(
            self._hJob, win32job.JobObjectExtendedLimitInformation
        )
        return max(info["PeakProcessMemoryUsed"], info["PeakJobMemoryUsed"])

    def _watchdog(self) -> None:
        pywintypes, win32event, win32job, _win32process = self._win32
        while not self._stop_watchdog.is_set():
            try:
                if self._job_peak_bytes() > self._limit_bytes:
                    self._killed_by_memory_limit.set()
                    win32job.TerminateJobObject(self._hJob, 1)
                    return
            except pywintypes.error:
                return
            if win32event.WaitForSingleObject(self._hProcess, 0) == win32event.WAIT_OBJECT_0:
                return
            self._stop_watchdog.wait(self._poll_interval_s)

    # --- everywhere else: an ordinary subprocess -------------------------

    def _start_portable(self, cmd: list[str], cwd: str | None, env: dict[str, str] | None) -> None:
        """No OS ceiling available — run the child anyway.

        Everything else about a job still holds off Windows: separate
        address space, progress file, cancellation, cleanup. Only the
        ceiling is missing, and ceiling_status() says so rather than
        letting the app imply a protection it does not have.
        """
        ceiling_status()  # logs the warning once
        self._popen = subprocess.Popen(cmd, cwd=cwd, env=env)
        self.pid = self._popen.pid
        self._sampler_thread = threading.Thread(target=self._sample_peak, daemon=True)
        self._sampler_thread.start()

    def _sample_peak(self) -> None:
        """Best-effort peak RSS for the uncapped path, so the stat line has
        something true to say. Not a high-water mark the OS maintains —
        a spike between samples is missed — which is why the capped path
        does not use this."""
        try:
            import psutil

            proc = psutil.Process(self.pid)
        except Exception:
            return
        while self._popen is not None and self._popen.poll() is None:
            try:
                self._peak_bytes_fallback = max(
                    self._peak_bytes_fallback, int(proc.memory_info().rss)
                )
            except Exception:
                return
            self._stop_watchdog.wait(self._poll_interval_s)

    # --- lifecycle -------------------------------------------------------

    def wait(self, timeout: float | None = None) -> bool:
        """True if the child has exited, False on timeout."""
        if self._win32 is not None:
            _pywintypes, win32event, _win32job, _win32process = self._win32
            wait_ms = win32event.INFINITE if timeout is None else int(timeout * 1000)
            return win32event.WaitForSingleObject(self._hProcess, wait_ms) == win32event.WAIT_OBJECT_0
        assert self._popen is not None
        try:
            self._popen.wait(timeout=timeout)
            return True
        except subprocess.TimeoutExpired:
            return False

    def terminate(self) -> None:
        """Kill the child now. On Windows this terminates the whole Job
        Object, so any grandchild it spawned (kaleido's Chromium, for one)
        dies with it rather than being orphaned."""
        if self._win32 is not None:
            _pywintypes, _win32event, win32job, _win32process = self._win32
            try:
                win32job.TerminateJobObject(self._hJob, 1)
            except Exception:
                pass
            return
        if self._popen is not None and self._popen.poll() is None:
            self._popen.terminate()
            try:
                self._popen.wait(timeout=2)
            except subprocess.TimeoutExpired:
                self._popen.kill()

    def result(self) -> CappedResult:
        """Final outcome. Call after wait() reports the child has exited."""
        self._stop_watchdog.set()
        if self._win32 is not None:
            _pywintypes, _win32event, _win32job, win32process = self._win32
            try:
                peak_bytes = self._job_peak_bytes()
            except Exception:
                peak_bytes = 0
            try:
                exit_code = win32process.GetExitCodeProcess(self._hProcess)
            except Exception:
                exit_code = None
            return CappedResult(
                pid=self.pid,
                exit_code=exit_code,
                killed_by_memory_limit=self._killed_by_memory_limit.is_set(),
                timed_out=False,
                # MiB, matching limit_mb's base (limit_bytes = limit_mb << 20).
                # Reporting this in decimal MB while the limit was binary made
                # a completed 1977 MiB run print as "2073.6MB" against a
                # "2048MB" cap, which reads as an overrun that never happened.
                peak_mb=peak_bytes / (1024 * 1024),
                limit_mb=self._limit_mb,
                capped=True,
            )
        assert self._popen is not None
        return CappedResult(
            pid=self.pid,
            exit_code=self._popen.returncode,
            killed_by_memory_limit=False,
            timed_out=False,
            peak_mb=self._peak_bytes_fallback / (1024 * 1024),
            limit_mb=self._limit_mb,
            capped=False,
        )

    def close(self) -> None:
        self._stop_watchdog.set()
        if self._win32 is not None:
            for handle in (
                getattr(self, "_hThread", None),
                getattr(self, "_hProcess", None),
                getattr(self, "_hJob", None),
            ):
                try:
                    if handle is not None:
                        handle.Close()
                except Exception:
                    pass


def run_capped(
    cmd: list[str],
    limit_mb: int = MIN_MEMORY_CEILING_MB,
    poll_interval_s: float = DEFAULT_POLL_INTERVAL_S,
    cwd: str | None = None,
    timeout: float | None = None,
    env: dict[str, str] | None = None,
) -> CappedResult:
    """Run `cmd` to completion under a memory ceiling, blocking.

    The original scripts/job_cap.py entry point, kept with the same
    signature and semantics — diagnostics scripts and the packaging notes
    in PROJECT_BRIEF.md call it directly. The app itself uses `Job`, which
    is this plus a progress/result channel and cancellation.
    """
    process = CappedProcess(cmd, limit_mb, poll_interval_s, cwd=cwd, env=env)
    try:
        exited = process.wait(timeout)
        if not exited:
            process.terminate()
            process.wait(2)
            result = process.result()
            return CappedResult(
                pid=result.pid,
                exit_code=None,
                killed_by_memory_limit=result.killed_by_memory_limit,
                timed_out=True,
                peak_mb=result.peak_mb,
                limit_mb=result.limit_mb,
                capped=result.capped,
            )
        return process.result()
    finally:
        process.close()


# --- the temp-file channel --------------------------------------------


def _atomic_write(path: Path, text: str) -> None:
    """Write via a sibling temp file + os.replace, so a reader polling this
    path never sees a half-written document."""
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def write_progress(job_dir: Path, state: dict[str, Any]) -> None:
    """Child side: publish a progress snapshot.

    `result` is stripped: progress is rewritten many times a second and the
    result can be a multi-megabyte chart payload. It travels once, through
    write_result.
    """
    payload = {key: value for key, value in state.items() if key != "result"}
    try:
        _atomic_write(Path(job_dir) / PROGRESS_FILENAME, json.dumps(payload, default=str))
    except OSError:
        # A job whose progress file cannot be written must still finish;
        # the parent falls back to its last snapshot.
        pass


def read_progress(job_dir: Path) -> dict[str, Any] | None:
    """Parent side: the child's latest snapshot, or None if there isn't a
    readable one yet (not started, or caught mid-replace on a platform
    where that isn't atomic)."""
    path = Path(job_dir) / PROGRESS_FILENAME
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def write_result(job_dir: Path, state: dict[str, Any]) -> None:
    """Child side: the final state, result included, as JSON."""
    _atomic_write(
        Path(job_dir) / RESULT_FILENAME,
        json.dumps({"format": "json", "state": state}, default=str),
    )


def write_table_result(job_dir: Path, state: dict[str, Any], table: Any) -> None:
    """Child side: a final state whose result is TABULAR.

    Rows go to parquet next to the JSON envelope rather than into it —
    which is the whole reason the channel is files and not a pipe. JSON of
    a million rows is both larger than the parquet and has to exist twice
    in memory to be produced. Used by export/transform tasks; the profiling
    tasks return nested dicts and take the JSON path above.
    """
    table_path = Path(job_dir) / TABLE_FILENAME
    table.write_parquet(table_path)
    envelope = {
        "format": "parquet",
        "table": TABLE_FILENAME,
        "state": {key: value for key, value in state.items() if key != "result"},
    }
    _atomic_write(Path(job_dir) / RESULT_FILENAME, json.dumps(envelope, default=str))


def read_result(job_dir: Path) -> dict[str, Any] | None:
    """Parent side: the final state, with a parquet result read back in.

    Returns None when the child produced no result file at all — which is
    what a ceiling kill looks like from here, and is why the absence has
    to be a distinguishable outcome rather than an exception.
    """
    path = Path(job_dir) / RESULT_FILENAME
    try:
        envelope = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    state = envelope.get("state") or {}
    if envelope.get("format") == "parquet":
        import polars as pl

        table_path = Path(job_dir) / envelope.get("table", TABLE_FILENAME)
        state["result"] = pl.read_parquet(table_path).to_dicts()
    return state


def request_cancel(job_dir: Path) -> None:
    """Parent side: ask the child to stop at its next item boundary."""
    try:
        (Path(job_dir) / CANCEL_FILENAME).write_text("1", encoding="utf-8")
    except OSError:
        pass


def cancel_requested(job_dir: Path) -> bool:
    """Child side: has a stop been asked for?"""
    return (Path(job_dir) / CANCEL_FILENAME).exists()


# --- ETA ---------------------------------------------------------------


def estimate_eta(
    elapsed: float,
    fraction: float | None,
    files_done: int | None,
    files_total: int | None,
) -> float | None:
    """Seconds remaining, or None when there is no honest way to say.

    Per-file throughput first — it is what these jobs actually scale with
    (~0.54s/file for chart building, measured) and it is a real rate rather
    than an extrapolation of a weighted progress bar. The fraction is the
    fallback for jobs whose unit of work is not a file (deep checks).

    Nothing is reported in the first MIN_ELAPSED seconds or below a few
    percent: an ETA computed off one sample of a process that is still
    importing polars is noise presented as a measurement.
    """
    MIN_ELAPSED = 1.0
    if elapsed < MIN_ELAPSED:
        return None
    if files_total and files_done and files_done > 0:
        per_file = elapsed / files_done
        remaining = max(0, files_total - files_done)
        return round(per_file * remaining, 1)
    if fraction and fraction >= 0.03:
        return round(elapsed * (1.0 - fraction) / fraction, 1)
    return None


def format_eta(seconds: float | None) -> str:
    """An ETA as "about 1m 53s left", or an empty string when unknown."""
    if seconds is None:
        return ""
    if seconds < 1:
        return "almost done"
    return f"about {format_duration(seconds)} left"


# --- the parent-side job ----------------------------------------------

IDLE_STATE: dict[str, Any] = {
    "stage": "idle",
    "label": "",
    "fraction": 0.0,
    "files_done": 0,
    "files_total": 0,
    "rows_done": None,
    "finished": False,
    "cancelled": False,
    "error": None,
    "result": None,
    "ceiling_hit": False,
}


def worker_command(job_dir: Path) -> list[str]:
    """How to launch the worker, from source and from the frozen .exe.

    Frozen, `sys.executable` IS the app — re-running it would open a second
    window — so the worker is reached through a flag main.py checks before
    it creates one. From source there is a real interpreter, so the module
    runs directly.
    """
    from ml_data_view.resources import is_frozen

    if is_frozen():
        return [sys.executable, WORKER_FLAG, str(job_dir)]
    return [sys.executable, "-m", "ml_data_view.jobs", str(job_dir)]


def _worker_env() -> dict[str, str]:
    """Child environment: the repository root on PYTHONPATH so `-m
    ml_data_view.jobs` resolves regardless of the parent's cwd."""
    from ml_data_view.resources import PROJECT_ROOT

    env = dict(os.environ)
    root = str(PROJECT_ROOT)
    existing = env.get("PYTHONPATH", "")
    if root not in existing.split(os.pathsep):
        env["PYTHONPATH"] = f"{root}{os.pathsep}{existing}" if existing else root
    return env


class Job:
    """One heavy operation, running in a capped worker subprocess.

    The lifecycle the UI sees is the same one every polled job in this app
    has always had — `start()` returns at once, `progress()` is a JSON-safe
    snapshot, `cancel()` takes effect at an item boundary — so a caller
    that swaps an in-process job for this one changes almost nothing.
    """

    def __init__(
        self,
        task: str,
        params: dict[str, Any],
        *,
        label: str,
        limit_mb: int,
        files_total: int = 0,
        poll_interval_s: float = 0.1,
        cancel_grace_seconds: float = CANCEL_GRACE_SECONDS,
    ) -> None:
        self.id = uuid.uuid4().hex[:12]
        self.task = task
        self.label = label
        self.limit_mb = limit_mb
        self._params = params
        self._poll_interval_s = poll_interval_s
        self._cancel_grace_seconds = cancel_grace_seconds
        self._dir = Path(tempfile.mkdtemp(prefix=f"mldv-job-{task}-"))
        self._lock = threading.Lock()
        self._done = threading.Event()
        self._cancel_requested = threading.Event()
        self._thread: threading.Thread | None = None
        self._process: CappedProcess | None = None
        self._started_at = monotonic()
        self._capped_result: CappedResult | None = None
        self._state: dict[str, Any] = {
            **IDLE_STATE,
            "stage": "pending",
            "label": "Waiting to start…",
            "files_total": files_total,
        }
        # Test seam, matching the in-process job classes': called with a
        # snapshot after every state update.
        self._on_progress_for_test: Callable[[dict[str, Any]], None] | None = None

    # --- public API -----------------------------------------------------

    def start(self) -> None:
        if self._thread is not None:
            raise RuntimeError("This job has already been started.")
        _atomic_write(
            self._dir / SPEC_FILENAME,
            json.dumps({"task": self.task, "params": self._params}, default=str),
        )
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def cancel(self) -> None:
        """Graceful first: the child stops at its next boundary and keeps
        what it finished. Forced only if it doesn't (see _run)."""
        self._cancel_requested.set()
        request_cancel(self._dir)

    def wait(self, timeout: float | None = None) -> bool:
        return self._done.wait(timeout)

    def progress(self) -> dict[str, Any]:
        with self._lock:
            state = dict(self._state)
        if not state["finished"]:
            published = read_progress(self._dir)
            if published:
                # "finished" is the parent's to set: it means "the result
                # has been read and adopted", which is strictly later than
                # the child's last progress write.
                state.update({k: v for k, v in published.items() if k != "finished"})
        elapsed = monotonic() - self._started_at
        eta = (
            None
            if state["finished"]
            else estimate_eta(
                elapsed, state.get("fraction"), state.get("files_done"), state.get("files_total")
            )
        )
        state["job_id"] = self.id
        state["task"] = self.task
        state["job_label"] = self.label
        state["elapsed_seconds"] = round(elapsed, 2)
        state["eta_seconds"] = eta
        state["eta_label"] = format_eta(eta)
        state["cancelling"] = self._cancel_requested.is_set() and not state["finished"]
        state["limit_mb"] = self.limit_mb
        if self._capped_result is not None:
            state["peak_bytes"] = int(self._capped_result.peak_mb * 1024 * 1024)
            state["capped"] = self._capped_result.capped
        return state

    @property
    def finished(self) -> bool:
        return self._done.is_set()

    @property
    def directory(self) -> Path:
        return self._dir

    # --- internals ------------------------------------------------------

    def _update(self, **changes: Any) -> None:
        with self._lock:
            self._state.update(changes)
        if self._on_progress_for_test is not None:
            self._on_progress_for_test(self.progress())

    def _run(self) -> None:
        try:
            self._process = CappedProcess(
                worker_command(self._dir),
                limit_mb=self.limit_mb,
                cwd=str(Path.cwd()),
                env=_worker_env(),
            )
            self._update(stage="running", label=self.label)
            self._supervise()
        except Exception as exc:  # a supervisor thread must never die silently
            self._update(
                stage="error",
                label=f"{self.label} failed: {exc}",
                error=str(exc),
                result=None,
                finished=True,
            )
            self._cleanup()
        finally:
            self._done.set()

    def _supervise(self) -> None:
        """Wait for the child, forcing the issue if a cancel is ignored."""
        assert self._process is not None
        cancel_deadline: float | None = None
        while True:
            if self._process.wait(self._poll_interval_s):
                break
            if self._cancel_requested.is_set():
                if cancel_deadline is None:
                    cancel_deadline = monotonic() + self._cancel_grace_seconds
                elif monotonic() > cancel_deadline:
                    # It had its chance to stop cleanly and keep its partial
                    # results; past the grace period the guarantee that
                    # Cancel actually cancels wins.
                    self._process.terminate()
                    self._process.wait(5)
                    break
        capped = self._process.result()
        self._capped_result = capped
        self._process.close()
        self._settle(capped)

    def _settle(self, capped: CappedResult) -> None:
        state = read_result(self._dir)
        forced = self._cancel_requested.is_set() and state is None
        # A job that FAILED while sitting at the ceiling was stopped by the
        # ceiling, however the failure surfaced — the OS refusing an
        # allocation, the watchdog terminating the job, or the child
        # catching the resulting MemoryError and recording it as an error
        # like any other. A job that SUCCEEDED at 99% of its ceiling simply
        # fitted, and keeps its result.
        failed = (
            state is None or bool(state.get("error")) or state.get("stage") == "error"
        )

        if failed and (capped.killed_by_memory_limit or self._hit_ceiling(capped)):
            message = ceiling_message(self.limit_mb)
            self._update(
                stage="ceiling",
                label=message,
                error=message,
                ceiling_hit=True,
                result=None,
                finished=True,
                cancelled=False,
            )
        elif forced:
            self._update(
                stage="cancelled",
                label="Cancelled.",
                error=None,
                result=None,
                finished=True,
                cancelled=True,
            )
        elif state is None:
            message = (
                f"{self.label} stopped unexpectedly (exit code {capped.exit_code}). "
                "Nothing was written."
            )
            self._update(
                stage="error", label=message, error=message, result=None, finished=True
            )
        else:
            state.setdefault("result", None)
            self._update(**{**state, "finished": True})
        self._cleanup()

    def _hit_ceiling(self, capped: CappedResult) -> bool:
        """A child that produced no result and whose peak reached the cap
        was killed by it.

        The watchdog usually catches this and sets killed_by_memory_limit,
        but the Job Object's own commit-charge limit can get there first —
        the allocation is refused, Python raises MemoryError, the process
        dies (or records it as an error) before the next 50ms poll. Same
        outcome, different reporter, so the peak counter is consulted too.
        The threshold is short of 100% because the refused allocation is
        the one that would have crossed the line, so the recorded peak
        sits just under the cap by whatever that allocation's size was.

        False off Windows: without the OS cap there is no ceiling to have
        hit, and reporting one would claim a protection that was not there.
        """
        if not capped.capped:
            return False
        return capped.peak_mb >= capped.limit_mb * CEILING_PEAK_THRESHOLD

    def _cleanup(self) -> None:
        """Delete the job's temp directory.

        Unconditional: a cancelled or ceiling-killed job leaves a partial
        result file and possibly a half-written parquet, and those are
        exactly the files that must not survive to be mistaken for output.
        The result has already been read into memory by the time this runs.
        """
        shutil.rmtree(self._dir, ignore_errors=True)


class JobManager:
    """The running jobs, for the top strip's indicator.

    Holds only what the UI needs to show and cancel; the owner of each job
    (Api) keeps its own reference for adopting the result. Finished jobs
    are dropped on the next snapshot rather than accumulating — the
    indicator is a live view, not a history.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._jobs: dict[str, Job] = {}

    def register(self, job: Job) -> Job:
        with self._lock:
            self._jobs[job.id] = job
        return job

    def snapshot(self) -> list[dict[str, Any]]:
        """One entry per still-running job: label, progress, ETA, id."""
        with self._lock:
            jobs = list(self._jobs.values())
        running = []
        for job in jobs:
            state = job.progress()
            if state["finished"]:
                continue
            running.append(
                {
                    "id": job.id,
                    "task": job.task,
                    "label": state.get("label") or job.label,
                    "title": job.label,
                    "fraction": state.get("fraction") or 0.0,
                    "files_done": state.get("files_done") or 0,
                    "files_total": state.get("files_total") or 0,
                    "rows_done": state.get("rows_done"),
                    "eta_label": state.get("eta_label") or "",
                    "eta_seconds": state.get("eta_seconds"),
                    "cancelling": bool(state.get("cancelling")),
                }
            )
        with self._lock:
            for job in jobs:
                if job.finished:
                    self._jobs.pop(job.id, None)
        return running

    def cancel(self, job_id: str) -> bool:
        with self._lock:
            job = self._jobs.get(job_id)
        if job is None:
            return False
        job.cancel()
        return True

    def cancel_all(self) -> None:
        with self._lock:
            jobs = list(self._jobs.values())
        for job in jobs:
            job.cancel()


# --- the worker (child) side ------------------------------------------


def run_worker(job_dir: str | Path) -> int:
    """Child entry point: run the task named in spec.json.

    Deliberately thin. The actual work is the same in-process job class the
    app used before this module existed (see job_tasks.py) — this loop
    republishes its progress and writes its final state out. Keeping the
    work in classes that run perfectly well in-process is what keeps them
    testable without spawning anything.
    """
    directory = Path(job_dir)
    spec = json.loads((directory / SPEC_FILENAME).read_text(encoding="utf-8"))

    # Imported here, not at module scope: the parent imports jobs.py in the
    # UI process, and job_tasks pulls in polars/pyarrow (~560MB of import
    # baseline, measured). Only the child needs them.
    from ml_data_view.job_tasks import build_task

    try:
        task = build_task(spec["task"], spec.get("params") or {})
    except Exception as exc:
        write_result(
            directory,
            {**IDLE_STATE, "stage": "error", "label": str(exc), "error": str(exc), "finished": True},
        )
        return 0

    task.run(directory)
    return 0


def main(argv: list[str]) -> int:
    """Both roles: `python -m ml_data_view.jobs <job_dir>` is the worker,
    `python -m ml_data_view.jobs --limit-mb N -- cmd…` is the original
    scripts/job_cap.py CLI, kept because PROJECT_BRIEF.md's measurement
    notes quote it."""
    import argparse

    if argv and not argv[0].startswith("-"):
        return run_worker(argv[0])

    parser = argparse.ArgumentParser(
        description="Run a command under a hard, OS-enforced memory ceiling (Windows Job Object)."
    )
    parser.add_argument("--limit-mb", type=int, default=MIN_MEMORY_CEILING_MB)
    parser.add_argument("--timeout", type=float, default=None, help="seconds; None = no timeout")
    parser.add_argument("command", nargs=argparse.REMAINDER, help="command to run, after --")
    args = parser.parse_args(argv)

    cmd = args.command
    if cmd and cmd[0] == "--":
        cmd = cmd[1:]
    if not cmd:
        parser.error("no command given (usage: --limit-mb 2048 -- python script.py)")

    result = run_capped(cmd, limit_mb=args.limit_mb, timeout=args.timeout)
    print(
        f"pid={result.pid} exit_code={result.exit_code} "
        f"killed_by_memory_limit={result.killed_by_memory_limit} "
        f"timed_out={result.timed_out} peak_mb={result.peak_mb:.1f} limit_mb={result.limit_mb}"
    )
    if result.killed_by_memory_limit or result.timed_out:
        return 137
    return result.exit_code if result.exit_code is not None else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
