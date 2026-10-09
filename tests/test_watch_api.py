"""Folder watching through the Api (api.py's watch methods).

folder_watcher.py's own tests cover the decision "is this file finished".
These cover what the app does with the answer: that a re-profile reads only
what is new, that a removed file drops out of the merged profile, that
pause suppresses it and resume catches up, and that the findings diff says
what changed rather than restating everything.

The watcher's clock is real here — these go through the Api, which builds
its own watcher — so the stability interval is set to its floor and the
tests drive `tick()` directly rather than waiting on the poller thread.
"""
from __future__ import annotations

import json
from pathlib import Path

import polars as pl
import pytest

from ml_data_view.api import Api
from ml_data_view.audit_log import audit_log_file_path

# The watcher's floor. Every test here drives tick() by hand, so the only
# thing this affects is how long a file must sit still — and these write
# files that are already complete.
STABILITY = 1


def _frame(rows: int, offset: int = 0) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "id": list(range(rows)),
            "label": [f"c{(value + offset) % 3}" for value in range(rows)],
            "score": [float(value + offset) for value in range(rows)],
        }
    )


@pytest.fixture
def watched(tmp_path):
    """An Api with a folder loaded and that folder being watched."""
    folder = tmp_path / "incoming"
    folder.mkdir()
    for index in range(2):
        _frame(40, index).write_parquet(folder / f"part{index}.parquet")

    api = Api(settings_dir=tmp_path / "settings")
    api.discover_folder(str(folder))
    api.start_folder_load(0)
    api.wait_for_folder_load(timeout=180)
    api.save_settings({"watch_stability_seconds": STABILITY})
    api.watch_loaded_folder()
    yield api, folder
    api.save_settings({"watch_enabled": False})


def _settle(api, seconds: float = STABILITY * 3) -> None:
    """Push the watcher past its stability interval without sleeping.

    The interval is real time, so the watcher's own clock is wound forward
    instead — the alternative is a test that sleeps once per assertion.
    """
    watcher = api._watcher
    base = watcher._now
    watcher._now = lambda: base() + seconds
    watcher.tick()
    watcher._now = base


def _reprofile(api) -> dict:
    started = api.begin_watch_reprofile()
    assert "error" not in started, started
    return api.wait_for_folder_load(timeout=180)


# --- a file still being written ------------------------------------------


def test_a_partly_written_file_does_not_trigger_a_reprofile(watched, tmp_path):
    """The gate that matters. A truncated parquet has no footer, so however
    long its size holds still it is not finished."""
    api, folder = watched
    complete = _frame(40, 9).write_parquet
    blob = tmp_path / "payload.parquet"
    complete(blob)
    payload = blob.read_bytes()

    partial = folder / "part9.parquet"
    partial.write_bytes(payload[: len(payload) // 2])
    api._watcher.note_event(partial)
    _settle(api)

    status = api.get_watch_status()
    assert status["reprofile_pending"] is False
    assert [entry["file"] for entry in status["pending"]] == ["part9.parquet"]
    assert status["pending"][0]["state"] == "unreadable"


def test_completing_the_file_triggers_exactly_one_reprofile(watched, tmp_path):
    api, folder = watched
    blob = tmp_path / "payload.parquet"
    _frame(40, 9).write_parquet(blob)
    payload = blob.read_bytes()
    arriving = folder / "part9.parquet"

    arriving.write_bytes(payload[: len(payload) // 2])
    api._watcher.note_event(arriving)
    _settle(api)
    assert api.get_watch_status()["reprofile_pending"] is False

    with arriving.open("ab") as handle:
        handle.write(payload[len(payload) // 2 :])
    api._watcher.note_event(arriving)
    _settle(api)

    status = api.get_watch_status()
    assert status["reprofile_pending"] is True
    assert status["changes"] == [{"file": "part9.parquet", "kind": "ready"}]

    state = _reprofile(api)
    assert state["result"]["row_count"] == 120
    # The flag is cleared by starting the re-profile, so the next poll does
    # not start a second one on top of it.
    assert api.get_watch_status()["reprofile_pending"] is False


def test_a_reprofile_reads_only_the_new_file(watched):
    """The whole point of doing this incrementally. The two files already
    profiled come back from the cache untouched."""
    api, folder = watched
    _frame(40, 9).write_parquet(folder / "part9.parquet")
    api._watcher.note_event(folder / "part9.parquet")
    _settle(api)

    stats = _reprofile(api)["result"]["load_stats"]

    assert stats["files_scanned"] == 1
    assert stats["files_from_cache"] == 2
    assert "1 of 3 files profiled, 2 from cache" in stats["line"]


def test_a_burst_of_arrivals_is_one_reprofile(watched):
    """Five files landing at once want one re-merge, not five."""
    api, folder = watched
    for index in range(5):
        path = folder / f"burst{index}.parquet"
        _frame(10, index).write_parquet(path)
        api._watcher.note_event(path)
    _settle(api)

    assert len(api.get_watch_status()["changes"]) == 5
    stats = _reprofile(api)["result"]["load_stats"]

    assert stats["files_scanned"] == 5
    assert stats["file_count"] == 7


# --- removal --------------------------------------------------------------


def test_removing_a_file_removes_it_from_the_merged_profile(watched):
    api, folder = watched
    _frame(40, 9).write_parquet(folder / "part9.parquet")
    api._watcher.note_event(folder / "part9.parquet")
    _settle(api)
    assert _reprofile(api)["result"]["row_count"] == 120

    (folder / "part9.parquet").unlink()
    api._watcher.note_event(folder / "part9.parquet", removed=True)
    _settle(api)
    assert api.get_watch_status()["reprofile_pending"] is True

    state = _reprofile(api)

    assert state["result"]["row_count"] == 80
    assert state["result"]["files_used"] == 2
    assert [Path(p).name for p in state["result"]["used_files"]] == [
        "part0.parquet",
        "part1.parquet",
    ]


# --- pause and resume -----------------------------------------------------


def test_pause_suppresses_processing_and_resume_catches_up(watched):
    api, folder = watched
    assert api.pause_watch()["paused"] is True

    _frame(40, 9).write_parquet(folder / "part9.parquet")
    api._watcher.note_event(folder / "part9.parquet")
    _settle(api)
    assert api.get_watch_status()["reprofile_pending"] is False

    assert api.resume_watch()["paused"] is False
    _settle(api)

    assert api.get_watch_status()["reprofile_pending"] is True
    assert _reprofile(api)["result"]["row_count"] == 120


def test_pausing_without_a_watcher_is_an_error_not_a_crash(tmp_path):
    api = Api(settings_dir=tmp_path)

    assert "error" in api.pause_watch()
    assert "error" in api.resume_watch()


# --- lifecycle ------------------------------------------------------------


def test_watching_starts_and_stops_with_the_setting(tmp_path):
    folder = tmp_path / "data"
    folder.mkdir()
    _frame(10).write_parquet(folder / "part0.parquet")
    api = Api(settings_dir=tmp_path / "settings")

    assert api.get_watch_status()["watching"] is False
    api.save_settings({"watch_enabled": True, "watch_folder": str(folder)})
    assert api.get_watch_status()["watching"] is True

    api.save_settings({"watch_enabled": False})
    assert api.get_watch_status()["watching"] is False


def test_sync_watch_is_idempotent(tmp_path):
    """Settings re-applies the whole record on every change, so this is
    called far more often than the state actually changes."""
    folder = tmp_path / "data"
    folder.mkdir()
    api = Api(settings_dir=tmp_path / "settings")
    api.save_settings({"watch_enabled": True, "watch_folder": str(folder)})

    first = api._watcher
    api.sync_watch()
    api.sync_watch()

    assert api._watcher is first, "an unchanged watch was torn down and rebuilt"
    api.save_settings({"watch_enabled": False})


def test_changing_the_folder_replaces_the_watcher(tmp_path):
    """Both the folder and the interval are decided when the watcher is
    constructed, so a change has to build a new one — a half-reconfigured
    watcher would quietly keep watching the old folder."""
    first_folder = tmp_path / "a"
    second_folder = tmp_path / "b"
    first_folder.mkdir()
    second_folder.mkdir()
    api = Api(settings_dir=tmp_path / "settings")
    api.save_settings({"watch_enabled": True, "watch_folder": str(first_folder)})
    first = api._watcher

    api.save_settings({"watch_folder": str(second_folder)})

    assert api._watcher is not first
    assert Path(api._watcher.folder) == second_folder
    assert first.running is False
    api.save_settings({"watch_enabled": False})


def test_watching_a_folder_that_does_not_exist_does_not_start(tmp_path):
    api = Api(settings_dir=tmp_path / "settings")

    status = api.save_settings(
        {"watch_enabled": True, "watch_folder": str(tmp_path / "nope")}
    )["watch"]

    assert status["watching"] is False
    assert status["enabled"] is True


def test_watch_the_loaded_folder_needs_a_loaded_folder(tmp_path):
    assert "error" in Api(settings_dir=tmp_path).watch_loaded_folder()


def test_a_reprofile_without_a_watched_folder_is_an_error(tmp_path):
    assert "error" in Api(settings_dir=tmp_path).begin_watch_reprofile()


def test_the_stability_interval_is_configurable_and_floored(tmp_path):
    folder = tmp_path / "data"
    folder.mkdir()
    api = Api(settings_dir=tmp_path / "settings")

    api.save_settings(
        {"watch_enabled": True, "watch_folder": str(folder), "watch_stability_seconds": 30}
    )
    assert api._watcher.stability_seconds == 30

    api.save_settings({"watch_stability_seconds": 0})
    assert api._watcher.stability_seconds == 1
    api.save_settings({"watch_enabled": False})


# --- the findings diff ----------------------------------------------------


def test_the_first_scan_of_a_dataset_reports_no_changes(watched):
    """Calling every finding on a freshly opened dataset "new" would make
    the word mean nothing."""
    api, _folder = watched
    api.start_chart_build()
    api.wait_for_chart_build(timeout=300)

    assert api.run_instant_checks()["changes"] is None


def test_a_reprofile_reports_what_changed(watched):
    api, folder = watched
    api.start_chart_build()
    api.wait_for_chart_build(timeout=300)
    api.run_instant_checks()

    _frame(40, 9).write_parquet(folder / "part9.parquet")
    api._watcher.note_event(folder / "part9.parquet")
    _settle(api)
    _reprofile(api)
    api.start_chart_build()
    api.wait_for_chart_build(timeout=300)

    changes = api.run_instant_checks()["changes"]

    assert changes is not None
    assert changes["summary"]
    assert set(changes) == {"new", "resolved", "summary"}


def test_a_new_finding_is_named_in_the_summary(tmp_path):
    """The notification the brief asks for: "1 new finding: ... in column
    X". Built by adding a file that makes one column almost entirely
    missing, which is a finding the instant checks report."""
    folder = tmp_path / "incoming"
    folder.mkdir()
    pl.DataFrame(
        {"id": list(range(60)), "value": [float(v) for v in range(60)]}
    ).write_parquet(folder / "part0.parquet")

    api = Api(settings_dir=tmp_path / "settings")
    api.discover_folder(str(folder))
    api.start_folder_load(0)
    api.wait_for_folder_load(timeout=180)
    api.start_chart_build()
    api.wait_for_chart_build(timeout=300)
    api.run_instant_checks()
    api.save_settings({"watch_stability_seconds": STABILITY})
    api.watch_loaded_folder()

    # Same schema, but "value" is null throughout — enough to push the
    # merged null rate over the high-missingness threshold.
    pl.DataFrame(
        {"id": list(range(600)), "value": [None] * 600},
        schema={"id": pl.Int64, "value": pl.Float64},
    ).write_parquet(folder / "part1.parquet")
    api._watcher.note_event(folder / "part1.parquet")
    _settle(api)
    _reprofile(api)
    api.start_chart_build()
    api.wait_for_chart_build(timeout=300)

    changes = api.run_instant_checks()["changes"]
    api.save_settings({"watch_enabled": False})

    assert changes["new"], "a column that went 90% missing produced no new finding"
    assert "value" in changes["summary"], changes["summary"]
    assert changes["summary"].startswith("1 new finding:") or "new finding" in changes["summary"]


# --- the audit log --------------------------------------------------------


def _actions(settings_dir: Path) -> list[str]:
    path = audit_log_file_path(settings_dir)
    if not path.exists():
        return []
    return [json.loads(line)["action"] for line in path.read_text(encoding="utf-8").splitlines()]


def test_watching_and_reprofiling_are_audit_logged(tmp_path):
    """Watching means the app reads files nobody explicitly opened, at
    times nobody chose. An audit log exists precisely to record that."""
    folder = tmp_path / "incoming"
    folder.mkdir()
    settings_dir = tmp_path / "settings"
    _frame(20).write_parquet(folder / "part0.parquet")

    api = Api(settings_dir=settings_dir)
    api.save_settings({"audit_log_enabled": True, "watch_stability_seconds": STABILITY})
    api.discover_folder(str(folder))
    api.start_folder_load(0)
    api.wait_for_folder_load(timeout=180)
    api.watch_loaded_folder()

    _frame(20, 5).write_parquet(folder / "part1.parquet")
    api._watcher.note_event(folder / "part1.parquet")
    _settle(api)
    _reprofile(api)
    api.save_settings({"watch_enabled": False})

    actions = _actions(settings_dir)
    assert "watch_started" in actions
    assert "reprofile" in actions
    assert "watch_stopped" in actions


def test_nothing_is_logged_when_the_audit_log_is_disabled(tmp_path):
    folder = tmp_path / "incoming"
    folder.mkdir()
    settings_dir = tmp_path / "settings"
    _frame(20).write_parquet(folder / "part0.parquet")

    api = Api(settings_dir=settings_dir)
    api.save_settings({"watch_enabled": True, "watch_folder": str(folder)})
    api.save_settings({"watch_enabled": False})

    assert _actions(settings_dir) == []
