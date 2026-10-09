"""Local, per-file cache of profiling results.

Re-opening a 279-file folder used to redo every scan. Almost none of that
work changes between opens: a parquet file on disk is immutable in
practice, so its row count, null counts and per-column statistics are the
same today as they were the last time they were computed. This caches them
per FILE (not per folder), which is what makes an incremental open
possible — add one file to a folder of 278 and only the new one is read.

WHY SQLITE, NOT ONE JSON DOCUMENT PER FILE
Both were considered; SQLite wins on four specific things, not on taste:

  1. A folder open touches every file's entry. JSON-per-file means 279
     opens, reads and parses — hundreds of syscalls to answer "is any of
     this still valid", against one indexed query here.
  2. A cache key is an absolute path, and a path is not a filename. Paths
     exceed Windows' filename length limit, contain separators and
     characters a filename cannot, and hashing them into names reintroduces
     collisions to handle. A TEXT primary key has none of those problems.
  3. Writes are transactional. Writing 279 documents is 279 chances to be
     interrupted and leave a half-written one that parses as valid JSON
     with a truncated array in it.
  4. "Cache size on disk" and "Clear cache" — both required by Settings —
     are one stat() and one DELETE, rather than walking a tree.

sqlite3 ships with Python, so this adds no dependency. The stats
themselves are stored as a JSON document in one column: their shape
evolves with the profiler, and SCHEMA_VERSION below invalidates old rows
wholesale, which is a far better fit than DDL migrations for data that is
always safe to recompute.

NOTHING HERE TOUCHES THE NETWORK. The database is a single file in the
same per-user app-data directory as settings.json.
"""
from __future__ import annotations

import json
import sqlite3
import threading
from dataclasses import dataclass
from pathlib import Path
from time import time
from typing import Any

CACHE_FILENAME = "profile_cache.sqlite"

# Bump this whenever a cached number could be computed DIFFERENTLY by new
# code — a changed bin count, a changed definition of "missing", a new
# per-file statistic that existing rows do not carry. Old rows then miss
# and are recomputed, so a code change can never serve a stale statistic
# that silently disagrees with what the same code would produce now.
#
# It is deliberately NOT tied to __version__: most releases change nothing
# about how a statistic is computed, and invalidating a 279-file cache on
# every release would make the feature pointless.
SCHEMA_VERSION = 1

# Section names inside a cached row's payload. Each is filled independently
# — a folder load writes "load", a chart build adds "chart" and "hist" to
# the same row — so a file profiled by one operation is not re-read by the
# next just because that one wants a different statistic.
SECTION_LOAD = "load"
SECTION_CHART = "chart"
SECTION_HIST = "hist"
SECTION_CO_NULL = "co_null"
# One chunk's time-ordered statistics (timeseries.py). Cached like every
# other per-chunk artifact, and keyed inside by the timestamp column it was
# computed against, so changing that column is a miss rather than a silently
# wrong answer.
SECTION_TIMESERIES = "timeseries"

_SCHEMA = """
CREATE TABLE IF NOT EXISTS file_profile (
    path           TEXT PRIMARY KEY,
    size_bytes     INTEGER NOT NULL,
    mtime_ns       INTEGER NOT NULL,
    schema_version INTEGER NOT NULL,
    updated_at     REAL    NOT NULL,
    payload        TEXT    NOT NULL
);
"""


@dataclass(frozen=True)
class FileKey:
    """What makes a cached profile still valid for a file on disk.

    Path plus size plus modification time. Not a content hash: hashing a
    12GB folder to decide whether to read it defeats the purpose — the
    read IS the expensive part. Size and mtime together catch every
    ordinary edit, and the failure mode of the pathological case (a file
    rewritten to exactly the same length within the same mtime tick) is a
    stale statistic the user can clear from Settings, not corruption.

    mtime is taken in nanoseconds (`st_mtime_ns`) rather than as a float:
    st_mtime is a float that loses precision on large timestamps, so two
    genuinely different mtimes can compare equal.
    """

    path: str
    size_bytes: int
    mtime_ns: int

    @classmethod
    def for_file(cls, file: str | Path) -> FileKey:
        resolved = Path(file).resolve()
        stat = resolved.stat()
        return cls(str(resolved), int(stat.st_size), int(stat.st_mtime_ns))


def cache_file_path(base_dir: Path | None = None) -> Path:
    """`base_dir` overrides the app-data directory — used by tests, and by
    api.py so a test Api never writes to the real user's cache."""
    from ml_data_view.settings import app_data_dir

    return (base_dir or app_data_dir()) / CACHE_FILENAME


class ProfileCache:
    """The store. One SQLite file, one connection, a lock around it.

    Opened by whichever process is doing the profiling — which is a capped
    worker subprocess (jobs.py), not the UI process. Two jobs can be in
    flight at once (a comparison while charts build), so the connection
    runs in WAL mode with a busy timeout rather than assuming a single
    writer.
    """

    def __init__(self, path: str | Path | None = None, base_dir: Path | None = None) -> None:
        self.path = Path(path) if path is not None else cache_file_path(base_dir)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._connection = sqlite3.connect(
            str(self.path), check_same_thread=False, timeout=30.0
        )
        with self._lock:
            # WAL so a reader is never blocked by the writer; both are
            # normal during a folder load, which reads every key up front
            # and writes back the misses as it goes.
            self._connection.execute("PRAGMA journal_mode=WAL")
            self._connection.execute("PRAGMA synchronous=NORMAL")
            self._connection.executescript(_SCHEMA)
            # Rows written by an older profiler can never be served (get()
            # checks the version too), so they are dropped here rather than
            # left to occupy disk that Settings then reports as "cache".
            self._connection.execute(
                "DELETE FROM file_profile WHERE schema_version != ?", (SCHEMA_VERSION,)
            )
            self._connection.commit()

    # --- lookup ---------------------------------------------------------

    def get(self, file: str | Path) -> dict[str, Any] | None:
        """The cached payload for `file`, or None on any kind of miss.

        A miss and a stale entry are the same answer on purpose: the caller
        rescans either way, and treating "the file changed" as anything
        other than "no cache" is how a half-updated profile happens.
        """
        try:
            key = FileKey.for_file(file)
        except OSError:
            return None  # vanished between discovery and here
        with self._lock:
            row = self._connection.execute(
                "SELECT size_bytes, mtime_ns, schema_version, payload"
                " FROM file_profile WHERE path = ?",
                (key.path,),
            ).fetchone()
        if row is None:
            return None
        size_bytes, mtime_ns, schema_version, payload = row
        if (
            size_bytes != key.size_bytes
            or mtime_ns != key.mtime_ns
            or schema_version != SCHEMA_VERSION
        ):
            return None
        try:
            return json.loads(payload)
        except json.JSONDecodeError:
            return None

    # --- writing --------------------------------------------------------

    def put(self, file: str | Path, payload: dict[str, Any]) -> None:
        """Replace `file`'s whole entry, stamped with its current key."""
        try:
            key = FileKey.for_file(file)
        except OSError:
            return
        with self._lock:
            self._connection.execute(
                "INSERT INTO file_profile"
                " (path, size_bytes, mtime_ns, schema_version, updated_at, payload)"
                " VALUES (?, ?, ?, ?, ?, ?)"
                " ON CONFLICT(path) DO UPDATE SET"
                "   size_bytes=excluded.size_bytes,"
                "   mtime_ns=excluded.mtime_ns,"
                "   schema_version=excluded.schema_version,"
                "   updated_at=excluded.updated_at,"
                "   payload=excluded.payload",
                (
                    key.path,
                    key.size_bytes,
                    key.mtime_ns,
                    SCHEMA_VERSION,
                    time(),
                    json.dumps(payload),
                ),
            )
            self._connection.commit()

    def merge_sections(self, file: str | Path, sections: dict[str, Any]) -> dict[str, Any]:
        """Add or replace named sections on `file`'s entry, keeping the rest.

        This is what makes profiling incremental across OPERATIONS as well
        as across files: a folder load writes "load", and the chart build
        that follows adds "chart" and "hist" to the same row instead of
        starting a new one and losing what the load already paid for.

        Re-read rather than merged from an in-memory copy, because the row
        may have been written by a different job process since.
        """
        payload = self.get(file) or {}
        payload.update(sections)
        self.put(file, payload)
        return payload

    # --- maintenance ----------------------------------------------------

    def forget(self, files: list[str | Path]) -> int:
        """Drop specific files' entries — what "Force full re-profile"
        does, so the next open reads them again."""
        paths = [str(Path(f).resolve()) for f in files]
        if not paths:
            return 0
        with self._lock:
            cursor = self._connection.executemany(
                "DELETE FROM file_profile WHERE path = ?", [(p,) for p in paths]
            )
            self._connection.commit()
            return cursor.rowcount if cursor.rowcount and cursor.rowcount > 0 else len(paths)

    def clear(self) -> int:
        """Empty the cache and hand the disk space back.

        VACUUM rather than DELETE alone: after "Clear cache" the size
        Settings reports must actually go down, and SQLite keeps freed
        pages in the file otherwise.
        """
        with self._lock:
            removed = self._connection.execute("SELECT COUNT(*) FROM file_profile").fetchone()[0]
            self._connection.execute("DELETE FROM file_profile")
            self._connection.commit()
            self._connection.execute("VACUUM")
            self._connection.commit()
        return int(removed)

    def entry_count(self) -> int:
        with self._lock:
            return int(self._connection.execute("SELECT COUNT(*) FROM file_profile").fetchone()[0])

    def size_bytes(self) -> int:
        """Bytes on disk, including the write-ahead log — which is part of
        the database and can be larger than the main file after a big
        write, so reporting only the .sqlite would understate it."""
        total = 0
        for suffix in ("", "-wal", "-shm"):
            candidate = Path(str(self.path) + suffix)
            try:
                total += candidate.stat().st_size
            except OSError:
                pass
        return total

    def close(self) -> None:
        with self._lock:
            self._connection.close()

    def __enter__(self) -> ProfileCache:
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()


class MemoryProfileCache:
    """The same interface, remembering nothing past this process.

    For callers that have no cache configured — a headless script, a test
    that is about the merge rather than the persistence. It keeps the
    profiling code down to ONE path: "profile every file and merge",
    with the only difference being whether yesterday's answers are
    available. A second, cache-free code path would be the one that
    silently stopped matching.
    """

    def __init__(self) -> None:
        self._entries: dict[str, dict[str, Any]] = {}

    @staticmethod
    def _key(file: str | Path) -> str:
        return str(Path(file).resolve())

    def get(self, file: str | Path) -> dict[str, Any] | None:
        return self._entries.get(self._key(file))

    def put(self, file: str | Path, payload: dict[str, Any]) -> None:
        self._entries[self._key(file)] = payload

    def merge_sections(self, file: str | Path, sections: dict[str, Any]) -> dict[str, Any]:
        payload = {**(self.get(file) or {}), **sections}
        self.put(file, payload)
        return payload

    def forget(self, files: list[str | Path]) -> int:
        removed = 0
        for file in files:
            removed += 1 if self._entries.pop(self._key(file), None) is not None else 0
        return removed

    def clear(self) -> int:
        removed = len(self._entries)
        self._entries.clear()
        return removed

    def entry_count(self) -> int:
        return len(self._entries)

    def size_bytes(self) -> int:
        return 0

    def close(self) -> None:
        pass

    def __enter__(self) -> MemoryProfileCache:
        return self

    def __exit__(self, *_exc: object) -> None:
        pass
