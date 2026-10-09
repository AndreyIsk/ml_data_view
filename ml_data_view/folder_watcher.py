"""Watch a folder and say when a file is actually finished being written.

A profiling pipeline writes files over seconds or minutes. A filesystem
event says "something happened to this path", which is NOT the same as
"this file is ready to read" — the overwhelming majority of events a
watched folder produces are the middle of a write. Profiling on the event
itself gets you a row count for a third of a file, cached under a key that
looks perfectly valid.

So this module is mostly a gate, not a watcher. Three things stand between
an event and a re-profile:

  DEBOUNCE. A burst of events for one path collapses into one pending
  entry. Nothing is emitted per event; events only ever update state.

  SIZE STABILITY. A file is considered finished only once its size has
  not changed for `stability_seconds` (default 10). Any change restarts
  that clock. This is the check that actually catches a slow writer.

  FOOTER VALIDATION. For parquet, size stability is still not proof — a
  writer can stall, and a file copied by some tools is pre-allocated at
  full size and filled in afterwards. The footer is written last and
  carries the metadata, so a file whose footer does not parse is not
  finished, whatever its size has been doing.

A file that never passes all three is not dropped and not retried into
silence: it stays in `pending()` as "still being written", which is what
the UI shows. A stalled pipeline should look like a stalled pipeline.

TESTABILITY IS THE SHAPE OF THIS MODULE. The state machine above is
driven by two methods — `note_event()` and `tick()` — that take no
filesystem events and do no sleeping. The watchdog observer is a thin
adapter that calls the first, and a poller thread calls the second. Tests
drive the machine directly with an injected clock; one integration test
covers the adapter.

Nothing here reads column data. Deciding a file is ready costs a stat()
and, for parquet, a footer read — the same metadata-only budget folder
discovery works to.
"""
from __future__ import annotations

import threading
from dataclasses import dataclass
from pathlib import Path
from time import monotonic
from typing import Any, Callable

from ml_data_view.folder_loader import DISCOVERY_EXTENSIONS

# How long a file's size must hold still before it is considered written.
# Ten seconds is a compromise, and the reason it is configurable: too short
# and a pipeline that pauses mid-write gets profiled half-finished, too
# long and every arrival waits for nothing.
DEFAULT_STABILITY_SECONDS = 10.0
# Minimum the setting may be lowered to. Below a second, ordinary
# filesystem latency alone can make a file that is still being written look
# stable between two stat() calls.
MIN_STABILITY_SECONDS = 1.0

# How often the stability gate re-checks pending files. Independent of the
# stability interval: this is a stat() per pending file, and pending files
# are rare, so a short poll costs nothing and keeps the UI's "still being
# written" list current.
DEFAULT_POLL_INTERVAL = 1.0

# What `pending()` reports about a file that has not been cleared yet.
STATE_SETTLING = "settling"
STATE_UNREADABLE = "unreadable"


@dataclass
class WatchEvent:
    """One thing the watcher is confident about."""

    path: Path
    # "ready": finished being written, safe to profile.
    # "removed": gone, so it must drop out of the merged profile.
    kind: str


@dataclass
class _Pending:
    size_bytes: int
    changed_at: float
    first_seen_at: float
    state: str = STATE_SETTLING


def parquet_footer_is_valid(path: Path) -> bool:
    """Whether a parquet file's footer parses.

    The footer is written LAST and holds the schema and row counts, so it
    is the closest thing to a "the writer finished" marker the format has.
    Reading it decodes no column data — the same metadata-only read folder
    discovery already uses for row counts.

    Non-parquet files have no equivalent (a CSV is valid at every
    truncation point), so they rely on size stability alone and this
    returns True for them.
    """
    if path.suffix.lower() != ".parquet":
        return True
    try:
        import pyarrow.parquet as pq

        pq.ParquetFile(path).metadata.num_rows
        return True
    except Exception:
        return False


class FolderWatcher:
    """Watches one folder and reports files that are genuinely ready.

    One folder at a time, deliberately: two watched folders means two
    merged profiles to keep straight, and the app shows one dataset.

    `on_change` is called with a LIST of events, not one per file. A
    pipeline dropping twenty files at once should cause one re-profile,
    not twenty — and because a re-profile re-merges the whole folder
    anyway, a batch costs exactly what a single file costs.
    """

    def __init__(
        self,
        folder: str | Path,
        on_change: Callable[[list[WatchEvent]], None],
        *,
        stability_seconds: float = DEFAULT_STABILITY_SECONDS,
        poll_interval: float = DEFAULT_POLL_INTERVAL,
        extensions: set[str] | None = None,
        time_source: Callable[[], float] = monotonic,
    ) -> None:
        self.folder = Path(folder)
        self._on_change = on_change
        self.stability_seconds = max(MIN_STABILITY_SECONDS, float(stability_seconds))
        self._poll_interval = poll_interval
        self._extensions = extensions if extensions is not None else set(DISCOVERY_EXTENSIONS)
        self._now = time_source

        self._lock = threading.Lock()
        self._pending: dict[Path, _Pending] = {}
        self._removed: set[Path] = set()
        self._paused = False
        self._running = False
        self._observer: Any = None
        self._poller: threading.Thread | None = None
        self._stop = threading.Event()

    # --- lifecycle -------------------------------------------------------

    def start(self) -> None:
        """Begin watching. Idempotent — calling it on a running watcher is
        a no-op, since Settings can re-apply the same state."""
        if self._running:
            return
        self._stop.clear()
        self._running = True
        self._start_observer()
        self._poller = threading.Thread(target=self._poll_loop, daemon=True)
        self._poller.start()

    def stop(self) -> None:
        self._running = False
        self._stop.set()
        observer, self._observer = self._observer, None
        if observer is not None:
            try:
                observer.stop()
                observer.join(timeout=5)
            except Exception:
                pass
        poller, self._poller = self._poller, None
        if poller is not None:
            poller.join(timeout=5)

    def pause(self) -> None:
        """Stop PROCESSING events, not collecting them.

        Events keep landing in `_pending` while paused, so resuming catches
        up on everything that arrived — a pause that dropped events would
        silently leave the profile describing fewer files than the folder
        holds, which is the exact failure this feature exists to prevent.
        """
        with self._lock:
            self._paused = True

    def resume(self) -> None:
        with self._lock:
            self._paused = False

    @property
    def paused(self) -> bool:
        with self._lock:
            return self._paused

    @property
    def running(self) -> bool:
        return self._running

    # --- the state machine (no filesystem, no sleeping) ------------------

    def note_event(self, path: str | Path, removed: bool = False) -> None:
        """Record that something happened to `path`.

        Called for every filesystem event, so it does the minimum: update
        one dict entry. Nothing is emitted here — an event is not evidence
        that a file is finished, and treating it as one is the bug this
        whole module exists to avoid.
        """
        path = Path(path)
        if path.suffix.lower() not in self._extensions:
            return
        with self._lock:
            if removed:
                self._pending.pop(path, None)
                self._removed.add(path)
                return
            self._removed.discard(path)
            now = self._now()
            try:
                size = path.stat().st_size
            except OSError:
                # Gone between the event and here — a writer's temp file,
                # usually. Not an error, and not something to report.
                self._pending.pop(path, None)
                return
            existing = self._pending.get(path)
            if existing is None:
                self._pending[path] = _Pending(size, now, now)
            elif existing.size_bytes != size:
                # The clock restarts on every change. A slow writer that
                # appends every nine seconds never clears a ten-second gate,
                # which is the correct answer.
                existing.size_bytes = size
                existing.changed_at = now
                existing.state = STATE_SETTLING

    def tick(self) -> list[WatchEvent]:
        """One pass of the stability gate. Emits whatever is now certain.

        Returns the events (for tests and for callers driving this
        manually) as well as handing them to `on_change`.
        """
        if self.paused:
            return []

        events: list[WatchEvent] = []
        with self._lock:
            now = self._now()
            for path in list(self._pending):
                entry = self._pending[path]
                try:
                    size = path.stat().st_size
                except OSError:
                    del self._pending[path]
                    continue
                if size != entry.size_bytes:
                    entry.size_bytes = size
                    entry.changed_at = now
                    entry.state = STATE_SETTLING
                    continue
                if now - entry.changed_at < self.stability_seconds:
                    continue
                if not parquet_footer_is_valid(path):
                    # Size has held still but the footer is not there yet.
                    # Kept pending rather than retried into silence: a
                    # writer that stalled mid-file should look stalled.
                    entry.state = STATE_UNREADABLE
                    entry.changed_at = now
                    continue
                del self._pending[path]
                events.append(WatchEvent(path, "ready"))

            for path in sorted(self._removed):
                events.append(WatchEvent(path, "removed"))
            self._removed.clear()

        if events:
            self._on_change(events)
        return events

    # --- what the UI shows -----------------------------------------------

    def pending(self) -> list[dict[str, Any]]:
        """Files seen but not yet cleared — the "still being written" list.

        `seconds_waiting` is measured from when the file was FIRST seen,
        not from its last change: a user watching a stuck pipeline wants to
        know how long it has been stuck, not how long since the last byte.
        """
        with self._lock:
            now = self._now()
            return [
                {
                    "file": path.name,
                    "path": str(path),
                    "size_bytes": entry.size_bytes,
                    "seconds_waiting": round(now - entry.first_seen_at, 1),
                    "state": entry.state,
                }
                for path, entry in sorted(self._pending.items())
            ]

    def status(self) -> dict[str, Any]:
        return {
            "watching": self._running,
            "paused": self.paused,
            "folder": str(self.folder),
            "stability_seconds": self.stability_seconds,
            "pending": self.pending(),
        }

    # --- the watchdog adapter --------------------------------------------

    def _start_observer(self) -> None:
        """Attach a real filesystem observer, if watchdog is available.

        A missing or failing observer is not fatal: the watcher degrades to
        whatever `note_event` is told from elsewhere, which is exactly what
        the tests drive it with. That keeps one code path rather than
        making the library a hard requirement of the state machine.
        """
        try:
            from watchdog.events import FileSystemEventHandler
            from watchdog.observers import Observer
        except ImportError:  # pragma: no cover - watchdog is a declared dependency
            return

        watcher = self

        class _Handler(FileSystemEventHandler):
            def on_any_event(self, event) -> None:
                if getattr(event, "is_directory", False):
                    return
                kind = event.event_type
                if kind == "deleted":
                    watcher.note_event(event.src_path, removed=True)
                elif kind == "moved":
                    # A move is a deletion at the source and an arrival at
                    # the destination — and "write to a temp name, then
                    # rename into place" is the single most common way a
                    # pipeline publishes a finished file.
                    watcher.note_event(event.src_path, removed=True)
                    watcher.note_event(getattr(event, "dest_path", event.src_path))
                else:
                    watcher.note_event(event.src_path)

        observer = Observer()
        observer.schedule(_Handler(), str(self.folder), recursive=True)
        observer.start()
        self._observer = observer

    def _poll_loop(self) -> None:
        while not self._stop.is_set():
            try:
                self.tick()
            except Exception:
                # A watcher thread must never die silently, and it must
                # never take the window with it either. One bad pass is
                # dropped; the next one runs.
                pass
            self._stop.wait(self._poll_interval)
