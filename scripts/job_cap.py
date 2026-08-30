"""Reusable Windows Job Object wrapper: run a subprocess under a hard,
OS-enforced memory ceiling (default 2GB) so a runaway allocation gets
killed by Windows itself, rather than ballooning until the machine
freezes — see the "folder_open_diagnostics.log" OOM incident this was
built in response to.

Two independent layers of enforcement, both scoped to the child process
only (never this process or anything else on the machine):

  1. A Job Object commit-charge limit (JOB_OBJECT_LIMIT_PROCESS_MEMORY /
     JOB_OBJECT_LIMIT_JOB_MEMORY). This is the OS itself refusing the
     child's next allocation once the cap is hit — synchronous with the
     offending allocation, not a poll, so it is the more "instant" of
     the two. On its own this is usually enough: a failed VirtualAlloc/
     HeapAlloc surfaces to Python as MemoryError and the process dies.
     It is not treated as sufficient by itself here, because a child
     that grows via many small allocations, or that catches MemoryError
     and keeps limping along, would not necessarily die from this alone.

  2. A polling watchdog thread (default: every 50ms) that reads the Job
     Object's own OS-maintained PeakProcessMemoryUsed/PeakJobMemoryUsed
     counters (not a live snapshot of the child via GetProcessMemoryInfo
     — that was tried first and is racy: a child that dies in well
     under a second, as an allocation-failure crash typically does, is
     frequently already gone and reaped by the time a poll fires,
     reading back near-zero) and calls TerminateJobObject the moment
     either crosses the cap. This is the backstop that guarantees the
     process actually dies regardless of *how* it approached the limit
     or whether it tried to handle the resulting error itself.

JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE is also set, so if this wrapper
process itself dies (crash, Ctrl-C, kill), the child dies with it —
nothing capped by this can be orphaned into an uncapped runaway.

Usage as a library:
    from scripts.job_cap import run_capped
    result = run_capped([sys.executable, "script.py"], limit_mb=2048)
    print(result.exit_code, result.killed_by_memory_limit, result.peak_mb)

Usage as a CLI:
    python scripts/job_cap.py --limit-mb 2048 -- python script.py arg1
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import threading
from dataclasses import dataclass

import pywintypes
import win32event
import win32job
import win32process

DEFAULT_LIMIT_MB = 2048
DEFAULT_POLL_INTERVAL_S = 0.05


def _job_peak_bytes(hJob) -> int:
    """Peak commit charge the Job Object has ever recorded, across both
    its per-process and job-wide counters. OS-maintained on the job
    itself, so — unlike GetProcessMemoryInfo on the child's process
    handle — it's accurate even if read after the child has already
    exited (see module docstring)."""
    info = win32job.QueryInformationJobObject(hJob, win32job.JobObjectExtendedLimitInformation)
    return max(info["PeakProcessMemoryUsed"], info["PeakJobMemoryUsed"])


@dataclass
class JobCapResult:
    pid: int
    exit_code: int | None
    killed_by_memory_limit: bool
    timed_out: bool
    peak_mb: float
    limit_mb: int


def run_capped(
    cmd: list[str],
    limit_mb: int = DEFAULT_LIMIT_MB,
    poll_interval_s: float = DEFAULT_POLL_INTERVAL_S,
    cwd: str | None = None,
    timeout: float | None = None,
) -> JobCapResult:
    """Run `cmd` as a child process capped at `limit_mb` of memory,
    enforced by a Windows Job Object plus a polling watchdog backstop
    (see module docstring). The cap applies only to this child (and any
    further children it spawns, since Job Object membership is
    inherited) — nothing outside it is touched even if the cap fires.

    Returns a JobCapResult. `killed_by_memory_limit=True` means the
    watchdog observed the cap crossed and force-terminated the job;
    otherwise the process ran to its own completion (exit_code reflects
    whatever it exited with — a non-zero code there can still mean it
    hit the Job Object's own commit-charge limit and crashed with an
    unhandled MemoryError, just not via the watchdog).
    """
    limit_bytes = limit_mb * 1024 * 1024
    command_line = subprocess.list2cmdline(cmd)

    startup_info = win32process.STARTUPINFO()
    creation_flags = win32process.CREATE_SUSPENDED

    hProcess, hThread, pid, _tid = win32process.CreateProcess(
        None, command_line, None, None, True, creation_flags, None, cwd, startup_info
    )

    hJob = win32job.CreateJobObject(None, "")
    killed_by_memory_limit = threading.Event()
    stop_watchdog = threading.Event()

    try:
        info = win32job.QueryInformationJobObject(hJob, win32job.JobObjectExtendedLimitInformation)
        info["BasicLimitInformation"]["LimitFlags"] = (
            win32job.JOB_OBJECT_LIMIT_PROCESS_MEMORY
            | win32job.JOB_OBJECT_LIMIT_JOB_MEMORY
            | win32job.JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
        )
        info["ProcessMemoryLimit"] = limit_bytes
        info["JobMemoryLimit"] = limit_bytes
        win32job.SetInformationJobObject(hJob, win32job.JobObjectExtendedLimitInformation, info)
        win32job.AssignProcessToJobObject(hJob, hProcess)

        # Backstop watchdog: the commit-charge limit above is the primary,
        # synchronous enforcement (the OS refuses the very allocation that
        # would cross the cap). This thread exists for the case that
        # doesn't crash the child outright — e.g. it catches the resulting
        # error and keeps running pinned at the cap — by force-terminating
        # once the job's own peak-usage counters confirm the cap was hit.
        def watchdog() -> None:
            while not stop_watchdog.is_set():
                try:
                    if _job_peak_bytes(hJob) > limit_bytes:
                        killed_by_memory_limit.set()
                        win32job.TerminateJobObject(hJob, 1)
                        return
                except pywintypes.error:
                    return
                if win32event.WaitForSingleObject(hProcess, 0) == win32event.WAIT_OBJECT_0:
                    return
                stop_watchdog.wait(poll_interval_s)

        watchdog_thread = threading.Thread(target=watchdog, daemon=True)
        watchdog_thread.start()

        win32process.ResumeThread(hThread)

        wait_ms = int(timeout * 1000) if timeout is not None else win32event.INFINITE
        wait_result = win32event.WaitForSingleObject(hProcess, wait_ms)
        timed_out = wait_result == win32event.WAIT_TIMEOUT
        if timed_out:
            win32job.TerminateJobObject(hJob, 1)

        stop_watchdog.set()
        watchdog_thread.join(timeout=2)

        peak_bytes = _job_peak_bytes(hJob)
        exit_code = None if timed_out else win32process.GetExitCodeProcess(hProcess)
        return JobCapResult(
            pid=pid,
            exit_code=exit_code,
            killed_by_memory_limit=killed_by_memory_limit.is_set(),
            timed_out=timed_out,
            # MiB, matching limit_mb's base (limit_bytes = limit_mb << 20).
            # Reporting this in decimal MB while the limit was binary made
            # a completed 1977 MiB run print as "2073.6MB" against a
            # "2048MB" cap, which reads as an overrun that never happened.
            peak_mb=peak_bytes / (1024 * 1024),
            limit_mb=limit_mb,
        )
    finally:
        stop_watchdog.set()
        hThread.Close()
        hProcess.Close()
        hJob.Close()


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(
        description="Run a command under a hard, OS-enforced memory ceiling (Windows Job Object)."
    )
    parser.add_argument("--limit-mb", type=int, default=DEFAULT_LIMIT_MB)
    parser.add_argument("--timeout", type=float, default=None, help="seconds; None = no timeout")
    parser.add_argument("command", nargs=argparse.REMAINDER, help="command to run, after --")
    args = parser.parse_args(argv)

    cmd = args.command
    if cmd and cmd[0] == "--":
        cmd = cmd[1:]
    if not cmd:
        parser.error("no command given (usage: job_cap.py --limit-mb 2048 -- python script.py)")

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
