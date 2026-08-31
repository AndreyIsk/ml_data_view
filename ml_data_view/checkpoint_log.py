"""TEMPORARY debugging instrumentation for the folder-open OOM crash
investigation — not a permanent feature. Remove the import/checkpoint()
call sites (folder_loader.py, api.py) and this module once the crash is
understood and fixed.

Windows force-kills the process before Python gets a chance to print a
traceback, so ordinary logging (even to a file) is useless unless each
line is flushed and fsynced to disk the instant it's written — otherwise
the very last, most useful line is exactly the one still sitting in a
buffer when the kill happens. checkpoint() does both, and also records
current process memory so the log doubles as a coarse memory timeline:
if usage is climbing steadily across checkpoints, that's what's hunting
for the leak needs to see, not just "it died somewhere in stage X".

Log file: <project root>/folder_open_diagnostics.log (see LOG_PATH).
"""
from __future__ import annotations

import ctypes
import os
import sys
import time

from ml_data_view.resources import writable_dir

# Repository root from source; the per-user app-data directory once frozen.
# It must NOT be the bundle: sys._MEIPASS is a temp directory deleted on
# exit, so the log would vanish — and on a locked-down machine the open()
# below would fail outright, inside a function api.py calls on every
# folder load.
LOG_PATH = writable_dir() / "folder_open_diagnostics.log"

_start_time = time.monotonic()


class _ProcessMemoryCounters(ctypes.Structure):
    _fields_ = [
        ("cb", ctypes.c_ulong),
        ("PageFaultCount", ctypes.c_ulong),
        ("PeakWorkingSetSize", ctypes.c_size_t),
        ("WorkingSetSize", ctypes.c_size_t),
        ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
        ("QuotaPagedPoolUsage", ctypes.c_size_t),
        ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
        ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
        ("PagefileUsage", ctypes.c_size_t),
        ("PeakPagefileUsage", ctypes.c_size_t),
    ]


def _init_windows_memory_api():
    """Bind GetCurrentProcess/GetProcessMemoryInfo once, with explicit
    argtypes/restype. Without those, ctypes assumes plain (32-bit-ish)
    `int` for the HANDLE/BOOL values, which silently corrupts the call on
    64-bit Windows — GetProcessMemoryInfo() returns 0 (failure) instead of
    raising, so this would otherwise fail every time without a trace."""
    from ctypes import wintypes

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    psapi = ctypes.WinDLL("psapi", use_last_error=True)
    kernel32.GetCurrentProcess.restype = wintypes.HANDLE
    kernel32.GetCurrentProcess.argtypes = []
    psapi.GetProcessMemoryInfo.restype = wintypes.BOOL
    psapi.GetProcessMemoryInfo.argtypes = [
        wintypes.HANDLE,
        ctypes.POINTER(_ProcessMemoryCounters),
        wintypes.DWORD,
    ]
    return kernel32, psapi


_windows_memory_api = _init_windows_memory_api() if sys.platform == "win32" else None


def _current_memory_mb() -> float | None:
    """Best-effort current process memory in MB. No new dependency (no
    psutil) since this module is meant to be deleted again — uses the
    Windows API directly via ctypes, with a POSIX fallback for anyone
    running the diagnostic elsewhere."""
    if _windows_memory_api is not None:
        try:
            kernel32, psapi = _windows_memory_api
            counters = _ProcessMemoryCounters()
            counters.cb = ctypes.sizeof(_ProcessMemoryCounters)
            handle = kernel32.GetCurrentProcess()
            ok = psapi.GetProcessMemoryInfo(handle, ctypes.byref(counters), counters.cb)
            if ok:
                return counters.WorkingSetSize / 1e6
        except Exception:
            return None
        return None
    try:
        import resource

        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024  # KB -> MB on Linux
    except Exception:
        return None


def checkpoint(stage: str, detail: str = "") -> None:
    """Append one line to LOG_PATH and force it to disk before returning.

    Call this immediately BEFORE and AFTER every heavy step in the
    folder-open pipeline (discovery, schema grouping, group selection,
    preview generation, chart/embedding data prep) — a "start"/"done" pair
    around each — so a crash mid-step leaves the "start" line as the last
    thing in the file, pinpointing exactly which step (and which file, for
    the per-file schema-grouping lines) was running when it died.
    """
    elapsed = time.monotonic() - _start_time
    memory_mb = _current_memory_mb()
    memory_repr = f"{memory_mb:,.1f}MB" if memory_mb is not None else "unknown"
    line = f"{time.strftime('%Y-%m-%d %H:%M:%S')} | +{elapsed:8.3f}s | mem={memory_repr:>12} | {stage}"
    if detail:
        line += f" | {detail}"
    try:
        with open(LOG_PATH, "a", encoding="utf-8") as log_file:
            log_file.write(line + "\n")
            log_file.flush()
            try:
                os.fsync(log_file.fileno())
            except Exception:
                pass  # best-effort — flush() above already covers a plain process kill
    except OSError:
        # Diagnostics must never take the app down. This is temporary
        # instrumentation (see the module docstring); a read-only or
        # missing log directory in a packaged install is not a reason
        # for a folder load to fail.
        pass
