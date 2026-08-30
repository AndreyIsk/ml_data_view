"""Shared lifecycle for the background jobs the UI polls.

Every long operation in this app follows the same shape: start() returns at
once so the pywebview js_api bridge is never blocked, the UI polls
progress(), and cancel() takes effect at a safe boundary. Errors are
recorded into the state rather than raised into a dying daemon thread,
where nothing would ever see them.

Extracted here once a fifth job class needed it. FolderLoadJob and
ChartBuildJob predate this and still carry their own copies — converting
them is unrelated risk, so they were deliberately left alone.
"""
from __future__ import annotations

import threading
from pathlib import Path
from typing import Any


class BackgroundJob:
    """Common lifecycle: start() returns at once, poll progress(), cancel()
    stops at the next file boundary."""

    def __init__(self, files: list[Path]) -> None:
        self._files = list(files)
        self._lock = threading.Lock()
        self._cancel = threading.Event()
        self._done = threading.Event()
        self._thread: threading.Thread | None = None
        self._state: dict[str, Any] = {
            "stage": "pending",
            "label": "Waiting to start…",
            "fraction": 0.0,
            "files_done": 0,
            "files_total": len(self._files),
            "finished": False,
            "cancelled": False,
            "error": None,
            "result": None,
        }
        self._on_progress_for_test = None

    def start(self) -> None:
        if self._thread is not None:
            raise RuntimeError("This job has already been started.")
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def cancel(self) -> None:
        self._cancel.set()

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

    def _note_file(self, index: int, _path: Path, verb: str) -> None:
        done = index + 1
        total = len(self._files) or 1
        self._update(
            files_done=done,
            fraction=round(done / total, 4),
            label=f"{verb} — file {done} of {len(self._files)}",
        )

    def _work(self) -> Any:
        raise NotImplementedError

    def _run(self) -> None:
        try:
            result = self._work()
            cancelled = self._cancel.is_set()
            self._update(
                stage="cancelled" if cancelled else "done",
                label=self._final_label(cancelled),
                result=result,
            )
        except Exception as exc:  # a worker thread must never die silently
            self._update(stage="error", label=f"Failed: {exc}", error=str(exc), result=None)
        finally:
            self._update(finished=True, cancelled=self._cancel.is_set())
            self._done.set()

    def _final_label(self, cancelled: bool) -> str:
        return "Cancelled." if cancelled else "Done."
