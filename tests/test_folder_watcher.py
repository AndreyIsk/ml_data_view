"""The folder watcher's stability gate (ml_data_view/folder_watcher.py).

The thing being tested is a DECISION — "is this file finished being
written" — and getting it wrong means profiling a third of a file and
caching the result under a key that looks perfectly valid.

So these drive the state machine directly, with an injected clock: no
sleeping, no waiting on a real observer, and a burst of events is a burst
of calls. One test at the bottom does use a real watchdog observer against
a real temp directory, because the adapter between the two is the one part
the rest of this file cannot cover.
"""
from __future__ import annotations

import time
from pathlib import Path

import polars as pl
import pytest

from ml_data_view.folder_watcher import (
    MIN_STABILITY_SECONDS,
    STATE_SETTLING,
    STATE_UNREADABLE,
    FolderWatcher,
    parquet_footer_is_valid,
)

STABILITY = 10.0


class Clock:
    """A hand-cranked clock. The gate is entirely about elapsed time, and a
    test that measures real elapsed time is a test that sleeps."""

    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


class Collector:
    def __init__(self) -> None:
        self.batches: list[list] = []

    def __call__(self, events) -> None:
        self.batches.append(list(events))

    @property
    def ready(self) -> list[Path]:
        return [e.path for batch in self.batches for e in batch if e.kind == "ready"]

    @property
    def removed(self) -> list[Path]:
        return [e.path for batch in self.batches for e in batch if e.kind == "removed"]


@pytest.fixture
def collector():
    return Collector()


@pytest.fixture
def clock():
    return Clock()


def _watcher(tmp_path, collector, clock, **kwargs):
    return FolderWatcher(
        tmp_path,
        collector,
        stability_seconds=kwargs.pop("stability_seconds", STABILITY),
        time_source=clock,
        **kwargs,
    )


def _parquet_bytes(rows: int = 20) -> bytes:
    """A real, valid parquet document — so a partial write is a genuinely
    invalid one rather than a file of zeroes."""
    import io

    buffer = io.BytesIO()
    pl.DataFrame({"v": list(range(rows))}).write_parquet(buffer)
    return buffer.getvalue()


def _chunks(payload: bytes, count: int) -> list[bytes]:
    """Split a document the way a pipeline appends it — in pieces, with the
    footer arriving last."""
    step = max(1, len(payload) // count)
    pieces = [payload[start : start + step] for start in range(0, len(payload), step)]
    return [piece for piece in pieces if piece]


# --- the stability gate --------------------------------------------------


def test_a_file_written_in_chunks_is_not_profiled_until_it_is_complete(
    tmp_path, collector, clock
):
    """The whole reason this module exists. A file that is still being
    written must never be handed to the profiler — and it is checked at
    every intermediate chunk, after waiting LONGER than the stability
    interval each time, so a gate that merely counted from first sight
    would have fired several times over.
    """
    path = tmp_path / "part.parquet"
    payload = _parquet_bytes()
    watcher = _watcher(tmp_path, collector, clock)
    chunks = list(_chunks(payload, count=4))

    for piece in chunks[:-1]:
        with path.open("ab") as handle:
            handle.write(piece)
        watcher.note_event(path)
        clock.advance(STABILITY + 1)
        watcher.tick()
        assert collector.ready == [], "a partly-written file was reported as ready"
        assert watcher.pending(), "a partly-written file stopped being tracked"

    with path.open("ab") as handle:
        handle.write(chunks[-1])
    watcher.note_event(path)
    clock.advance(STABILITY + 1)
    watcher.tick()

    assert collector.ready == [path]


def test_any_change_restarts_the_clock(tmp_path, collector, clock):
    path = tmp_path / "part.parquet"
    path.write_bytes(_parquet_bytes())
    watcher = _watcher(tmp_path, collector, clock)
    watcher.note_event(path)

    clock.advance(STABILITY - 1)
    watcher.tick()
    assert collector.ready == []

    path.write_bytes(_parquet_bytes(rows=40))  # a different size
    clock.advance(2)
    watcher.tick()  # observes the change, restarts the clock
    assert collector.ready == []

    clock.advance(STABILITY - 1)
    watcher.tick()
    assert collector.ready == [], "the clock did not restart on the change"

    clock.advance(2)
    watcher.tick()
    assert collector.ready == [path]


def test_a_burst_of_events_becomes_one_reprofile(tmp_path, collector, clock):
    """Debouncing. A single write can produce dozens of events; each one
    must update state and none of them must emit."""
    path = tmp_path / "part.parquet"
    path.write_bytes(_parquet_bytes())
    watcher = _watcher(tmp_path, collector, clock)

    for _ in range(50):
        watcher.note_event(path)
    clock.advance(STABILITY + 1)
    watcher.tick()
    clock.advance(STABILITY + 1)
    watcher.tick()

    assert collector.ready == [path]
    assert len(collector.batches) == 1


def test_several_files_arriving_together_are_one_batch(tmp_path, collector, clock):
    """Twenty files landing at once should mean one re-profile: the merge
    covers the whole folder regardless of which file prompted it."""
    paths = []
    for index in range(5):
        path = tmp_path / f"part{index}.parquet"
        path.write_bytes(_parquet_bytes())
        paths.append(path)

    watcher = _watcher(tmp_path, collector, clock)
    for path in paths:
        watcher.note_event(path)
    clock.advance(STABILITY + 1)
    watcher.tick()

    assert len(collector.batches) == 1
    assert sorted(collector.ready) == sorted(paths)


def test_nothing_is_emitted_before_the_interval_elapses(tmp_path, collector, clock):
    path = tmp_path / "part.parquet"
    path.write_bytes(_parquet_bytes())
    watcher = _watcher(tmp_path, collector, clock)
    watcher.note_event(path)

    clock.advance(STABILITY - 0.1)
    watcher.tick()

    assert collector.batches == []


def test_a_settling_file_shows_as_still_being_written(tmp_path, collector, clock):
    path = tmp_path / "part.parquet"
    path.write_bytes(_parquet_bytes())
    watcher = _watcher(tmp_path, collector, clock)
    watcher.note_event(path)
    clock.advance(3)

    pending = watcher.pending()

    assert len(pending) == 1
    assert pending[0]["file"] == "part.parquet"
    assert pending[0]["state"] == STATE_SETTLING
    # Measured from when it was FIRST seen: someone watching a stuck
    # pipeline wants to know how long it has been stuck.
    assert pending[0]["seconds_waiting"] == pytest.approx(3.0)


# --- the footer gate -----------------------------------------------------


def test_a_stable_but_truncated_parquet_is_not_profiled(tmp_path, collector, clock):
    """Size stability is not proof. A writer that stalls — or a tool that
    pre-allocates the file and fills it in afterwards — leaves a file whose
    size has held still for minutes and whose footer is not there."""
    path = tmp_path / "part.parquet"
    payload = _parquet_bytes()
    path.write_bytes(payload[: len(payload) // 2])
    watcher = _watcher(tmp_path, collector, clock)
    watcher.note_event(path)

    for _ in range(3):
        clock.advance(STABILITY + 1)
        watcher.tick()

    assert collector.ready == []
    assert watcher.pending()[0]["state"] == STATE_UNREADABLE


def test_a_file_that_never_stabilizes_stays_visible(tmp_path, collector, clock):
    """Not dropped, not retried into silence: a stalled pipeline should
    look like a stalled pipeline."""
    path = tmp_path / "part.parquet"
    payload = _parquet_bytes()
    path.write_bytes(payload[:40])
    watcher = _watcher(tmp_path, collector, clock)
    watcher.note_event(path)

    for _ in range(10):
        clock.advance(STABILITY + 1)
        watcher.tick()

    pending = watcher.pending()
    assert len(pending) == 1
    assert pending[0]["seconds_waiting"] > STABILITY * 5


def test_completing_a_truncated_file_lets_it_through(tmp_path, collector, clock):
    path = tmp_path / "part.parquet"
    payload = _parquet_bytes()
    path.write_bytes(payload[: len(payload) // 2])
    watcher = _watcher(tmp_path, collector, clock)
    watcher.note_event(path)
    clock.advance(STABILITY + 1)
    watcher.tick()
    assert collector.ready == []

    with path.open("ab") as handle:
        handle.write(payload[len(payload) // 2 :])
    watcher.note_event(path)
    clock.advance(STABILITY + 1)
    watcher.tick()

    assert collector.ready == [path]


def test_footer_validation_only_applies_to_parquet(tmp_path):
    """A CSV is valid at every truncation point, so there is no equivalent
    check — those rely on size stability alone, and this says so rather
    than silently never clearing them."""
    csv_path = tmp_path / "part.csv"
    csv_path.write_text("a,b\n1,2\n", encoding="utf-8")

    assert parquet_footer_is_valid(csv_path) is True


def test_a_valid_parquet_footer_is_recognised(tmp_path):
    path = tmp_path / "part.parquet"
    path.write_bytes(_parquet_bytes())

    assert parquet_footer_is_valid(path) is True


# --- pause and resume ----------------------------------------------------


def test_pause_suppresses_processing(tmp_path, collector, clock):
    path = tmp_path / "part.parquet"
    path.write_bytes(_parquet_bytes())
    watcher = _watcher(tmp_path, collector, clock)
    watcher.pause()

    watcher.note_event(path)
    clock.advance(STABILITY + 1)
    watcher.tick()

    assert collector.batches == []
    assert watcher.paused is True


def test_resume_catches_up_on_what_arrived_while_paused(tmp_path, collector, clock):
    """Pause stops PROCESSING, not collecting. A pause that dropped events
    would leave the profile describing fewer files than the folder holds —
    the exact failure this feature exists to prevent."""
    watcher = _watcher(tmp_path, collector, clock)
    watcher.pause()

    paths = []
    for index in range(3):
        path = tmp_path / f"part{index}.parquet"
        path.write_bytes(_parquet_bytes())
        watcher.note_event(path)
        paths.append(path)
    clock.advance(STABILITY + 1)
    watcher.tick()
    assert collector.ready == []

    watcher.resume()
    watcher.tick()

    assert sorted(collector.ready) == sorted(paths)


def test_a_removal_while_paused_is_also_caught_up(tmp_path, collector, clock):
    path = tmp_path / "part.parquet"
    path.write_bytes(_parquet_bytes())
    watcher = _watcher(tmp_path, collector, clock)
    watcher.pause()
    watcher.note_event(path, removed=True)
    watcher.tick()
    assert collector.batches == []

    watcher.resume()
    watcher.tick()

    assert collector.removed == [path]


# --- removal -------------------------------------------------------------


def test_a_removed_file_is_reported_immediately(tmp_path, collector, clock):
    """No stability interval: a file being gone is certain the moment it is
    observed, and the merged profile has to stop counting it."""
    path = tmp_path / "part.parquet"
    watcher = _watcher(tmp_path, collector, clock)

    watcher.note_event(path, removed=True)
    watcher.tick()

    assert collector.removed == [path]


def test_a_file_deleted_mid_write_is_forgotten_not_reported_ready(
    tmp_path, collector, clock
):
    path = tmp_path / "part.parquet"
    path.write_bytes(_parquet_bytes())
    watcher = _watcher(tmp_path, collector, clock)
    watcher.note_event(path)

    path.unlink()
    clock.advance(STABILITY + 1)
    watcher.tick()

    assert collector.ready == []
    assert watcher.pending() == []


def test_a_removal_cancels_a_pending_arrival(tmp_path, collector, clock):
    path = tmp_path / "part.parquet"
    path.write_bytes(_parquet_bytes())
    watcher = _watcher(tmp_path, collector, clock)
    watcher.note_event(path)

    watcher.note_event(path, removed=True)
    clock.advance(STABILITY + 1)
    watcher.tick()

    assert collector.ready == []
    assert collector.removed == [path]


# --- what counts as a dataset file ---------------------------------------


def test_unrelated_files_are_ignored(tmp_path, collector, clock):
    """A pipeline's temp files, checksums and logs land in the same folder
    and are not datasets."""
    watcher = _watcher(tmp_path, collector, clock)
    for name in ["notes.txt", "part.parquet.crc", "_SUCCESS", "part.parquet.tmp"]:
        path = tmp_path / name
        path.write_bytes(b"x")
        watcher.note_event(path)

    clock.advance(STABILITY + 1)
    watcher.tick()

    assert collector.batches == []
    assert watcher.pending() == []


def test_an_arrival_by_rename_is_seen(tmp_path, collector, clock):
    """"Write to a temp name, then rename into place" is the single most
    common way a pipeline publishes a finished file."""
    temp = tmp_path / "part.parquet.tmp"
    final = tmp_path / "part.parquet"
    temp.write_bytes(_parquet_bytes())
    watcher = _watcher(tmp_path, collector, clock)

    temp.rename(final)
    watcher.note_event(temp, removed=True)
    watcher.note_event(final)
    clock.advance(STABILITY + 1)
    watcher.tick()

    assert collector.ready == [final]


# --- configuration --------------------------------------------------------


def test_the_stability_interval_cannot_be_set_below_the_floor(tmp_path, collector, clock):
    """Below a second, ordinary filesystem latency alone can make a file
    that is still being written look stable between two stat() calls."""
    watcher = _watcher(tmp_path, collector, clock, stability_seconds=0)

    assert watcher.stability_seconds == MIN_STABILITY_SECONDS


def test_status_describes_the_watch(tmp_path, collector, clock):
    watcher = _watcher(tmp_path, collector, clock)

    status = watcher.status()

    assert status["watching"] is False
    assert status["paused"] is False
    assert status["folder"] == str(tmp_path)
    assert status["stability_seconds"] == STABILITY


# --- the watchdog adapter -------------------------------------------------


def test_a_real_observer_sees_a_file_appear(tmp_path):
    """The one part the hand-cranked tests above cannot cover: that
    watchdog's events actually reach note_event. Everything after that
    point is already proven, so this asserts only that the file became
    pending — no sleeping on a ten-second gate."""
    pytest.importorskip("watchdog")
    collector = Collector()
    watcher = FolderWatcher(tmp_path, collector, stability_seconds=STABILITY)
    watcher.start()
    try:
        (tmp_path / "arrived.parquet").write_bytes(_parquet_bytes())
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline and not watcher.pending():
            time.sleep(0.1)
        pending = watcher.pending()
    finally:
        watcher.stop()

    assert [entry["file"] for entry in pending] == ["arrived.parquet"]
    assert collector.ready == [], "a brand-new file cleared a 10s gate"
