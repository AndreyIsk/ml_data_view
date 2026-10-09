"""The channel timeline through the Api, across every input shape.

Its precondition is a detected timestamp column and NOTHING ELSE, so what
is pinned here is that a single CSV gets a timeline just as a mapped tree
does, that a shape with no time column is told why it does not, and that
the shapes with no structure behave exactly as they did before.
"""
from __future__ import annotations

import math
from pathlib import Path

import polars as pl
import pytest

from ml_data_view.api import Api
from tests.test_structure_api import (
    BASE_TS,
    build_flat,
    build_full,
    build_hive,
    build_nested,
    _confirm,
    _open,
)

HZ = 10.0


def _chunk(path: Path, start: float, rows: int, *, offset: float = 0.0) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pl.DataFrame(
        {
            "ts": [start + index / HZ for index in range(rows)],
            "temp": [offset + math.sin(index / 11.0) for index in range(rows)],
            "speed": [offset + (index % 17) * 0.25 for index in range(rows)],
        }
    ).write_parquet(path, row_group_size=250)


def build_recordings(root: Path) -> Path:
    """One system, two recordings of four chunks each."""
    for recording in range(2):
        for index in range(4):
            _chunk(
                root / "sys_key=1" / f"rec_{recording}" / f"part-{index}.parquet",
                float(BASE_TS) + recording * 10_000 + index * 100.0,
                1000,
            )
    return root


@pytest.fixture
def recordings_api(tmp_path):
    folder = build_recordings(tmp_path / "tree")
    api = Api(settings_dir=tmp_path / "settings")
    discovery = api.discover_folder(str(folder))
    _confirm(api, discovery, labels={0: "System", 1: "Recording"})
    api.start_scope_load()
    api.wait_for_folder_load(timeout=300)
    api.start_chart_build()
    api.wait_for_chart_build(timeout=600)
    return api, folder


# --- the viewer itself ---------------------------------------------------


def test_a_recording_has_a_timeline(recordings_api):
    api, _folder = recordings_api

    timeline = api.get_timeline()

    assert timeline["available"] is True
    assert timeline["unit"] == "sys_key=1 / rec_0"
    assert timeline["units"] == ["sys_key=1 / rec_0", "sys_key=1 / rec_1"]
    assert timeline["unit_label"] == "Recording"
    assert {c["column"] for c in timeline["channels"]} == {"temp", "speed"}


def test_the_channels_are_capped_and_the_cap_is_reported(recordings_api):
    api, _folder = recordings_api
    api.save_settings({"timeline_max_channels": 1})

    timeline = api.get_timeline()

    assert len(timeline["channels"]) == 1
    assert timeline["max_channels"] == 1
    assert timeline["channels_available"] == ["temp", "speed"]


def test_a_channel_can_be_picked(recordings_api):
    api, _folder = recordings_api

    timeline = api.get_timeline(channels=["speed"])

    assert [c["column"] for c in timeline["channels"]] == ["speed"]


def test_a_channel_that_is_not_there_falls_back_rather_than_erroring(recordings_api):
    api, _folder = recordings_api

    timeline = api.get_timeline(channels=["nonexistent"])

    assert timeline["available"] is True
    assert timeline["channels_selected"] == ["temp", "speed"]


def test_the_other_recording_can_be_asked_for(recordings_api):
    api, _folder = recordings_api

    timeline = api.get_timeline(unit_id="sys_key=1 / rec_1")

    assert timeline["unit"] == "sys_key=1 / rec_1"
    assert timeline["span"][0] == pytest.approx(BASE_TS + 10_000)


def test_no_raw_rows_ever_leave_python(recordings_api):
    """Whatever the zoom, the payload is buckets — a bounded number of
    them — and never a row."""
    api, _folder = recordings_api

    for window in (None, (BASE_TS + 100.0, BASE_TS + 110.0)):
        timeline = api.get_timeline(
            start=window[0] if window else None,
            end=window[1] if window else None,
            buckets=600,
        )
        for channel in timeline["channels"]:
            assert len(channel["t0"]) <= 600


def test_a_bucket_count_beyond_the_cap_is_clamped(recordings_api):
    api, _folder = recordings_api

    timeline = api.get_timeline(buckets=1_000_000)

    assert timeline["buckets"] <= 2000


# --- zoom reads only what it has to --------------------------------------


def test_a_zoomed_window_reads_only_the_chunks_it_overlaps(recordings_api, monkeypatch):
    from ml_data_view import timeline as tl

    api, _folder = recordings_api
    opened: list[str] = []
    original = tl._iter_window

    def watched(path, columns, row_groups):
        opened.append(Path(path).name)
        return original(path, columns, row_groups)

    monkeypatch.setattr(tl, "_iter_window", watched)

    result = api.get_timeline(start=BASE_TS + 210.0, end=BASE_TS + 230.0, buckets=400)

    assert result["source"] == "scan"
    assert set(opened) == {"part-2.parquet"}
    assert result["chunks_in_window"] == 1


def test_a_zoom_reads_only_the_row_groups_it_overlaps(recordings_api):
    api, _folder = recordings_api

    result = api.get_timeline(start=BASE_TS + 200.0, end=BASE_TS + 205.0, buckets=400)

    # 1000 rows in 250-row groups; five seconds at 10Hz is fifty rows.
    assert result["source"] == "scan"
    assert result["rows_read"] == 250


def test_the_whole_recording_is_answered_from_the_cache(recordings_api):
    """4000 rows over four chunks, twenty-four cached buckets each — the
    wide view costs nothing and says so."""
    api, _folder = recordings_api
    api.save_settings({"timeline_max_rows": 100})

    result = api.get_timeline(buckets=600)

    assert result["source"] == "cache"
    assert result["rows_read"] == 0
    assert result["chunks_read"] == 0
    assert "budget" in result["note"]


def test_zooming_in_brings_the_same_recording_back_under_budget(recordings_api):
    api, _folder = recordings_api
    api.save_settings({"timeline_max_rows": 300})

    wide = api.get_timeline(buckets=600)
    zoomed = api.get_timeline(start=BASE_TS + 200.0, end=BASE_TS + 205.0, buckets=600)

    assert wide["source"] == "cache"
    assert zoomed["source"] == "scan"


# --- the findings overlay ------------------------------------------------


def test_the_overlay_marks_a_gap_where_the_finding_reported_it(tmp_path):
    folder = tmp_path / "gappy"
    _chunk(folder / "sys_key=1" / "rec_0" / "part-0.parquet", float(BASE_TS), 1000)
    # A minute missing between the two chunks.
    _chunk(folder / "sys_key=1" / "rec_0" / "part-1.parquet", float(BASE_TS) + 160.0, 1000)
    _chunk(folder / "sys_key=1" / "rec_1" / "part-0.parquet", float(BASE_TS) + 5_000, 1000)
    _chunk(folder / "sys_key=1" / "rec_1" / "part-1.parquet", float(BASE_TS) + 5_100, 1000)

    api = Api(settings_dir=tmp_path / "settings")
    discovery = api.discover_folder(str(folder))
    _confirm(api, discovery, labels={0: "System", 1: "Recording"})
    api.start_scope_load()
    api.wait_for_folder_load(timeout=300)
    api.start_chart_build()
    api.wait_for_chart_build(timeout=600)

    timeline = api.get_timeline(unit_id="sys_key=1 / rec_0")
    unit = api._chart_data["timeseries"]["units"]["sys_key=1 / rec_0"]
    gap = unit["gaps"][0]

    mark = next(m for m in timeline["marks"] if m["kind"] in {"gap", "missing"})
    assert mark["t0"] == pytest.approx(gap["at"])
    assert mark["t1"] == pytest.approx(gap["at"] + gap["duration"])
    assert mark["t0"] >= timeline["span"][0]
    assert mark["t1"] <= timeline["span"][1] + gap["duration"]


def test_a_clean_recording_has_no_marks(recordings_api):
    api, _folder = recordings_api

    assert api.get_timeline()["marks"] == []


# --- a selected window ---------------------------------------------------


def test_a_selected_window_is_summarised_from_the_buckets_on_screen(recordings_api):
    api, _folder = recordings_api
    api.get_timeline(start=BASE_TS, end=BASE_TS + 100.0, buckets=200)

    stats = api.get_timeline_window(BASE_TS + 10.0, BASE_TS + 20.0)

    channel = next(c for c in stats["channels"] if c["column"] == "temp")
    assert channel["n"] > 0
    assert channel["min"] <= channel["mean"] <= channel["max"]
    assert stats["duration"] == pytest.approx(10.0)


def test_selecting_before_there_is_a_timeline_is_an_error(tmp_path):
    api = Api(settings_dir=tmp_path / "settings")

    assert "error" in api.get_timeline_window(0.0, 1.0)


def test_a_window_becomes_a_row_search_on_the_timestamp_column(recordings_api):
    """Export is not reimplemented: a window IS a search condition, and
    the CSV export the Search tab already has takes it from there."""
    api, _folder = recordings_api

    result = api.timeline_window_conditions(BASE_TS + 10.0, BASE_TS + 20.0)

    assert result["conditions"] == [
        {
            "column": "ts",
            "operator": "between",
            "values": [repr(float(BASE_TS + 10.0)), repr(float(BASE_TS + 20.0))],
        }
    ]
    # And they are conditions the search feature actually accepts.
    found = api.run_search(result["conditions"])
    stamps = [row["ts"] for row in found["rows"]]
    assert stamps
    assert min(stamps) >= BASE_TS + 10.0
    assert max(stamps) <= BASE_TS + 20.0


def test_a_datetime_time_column_gets_a_window_written_as_a_datetime(tmp_path):
    from datetime import datetime, timezone

    path = tmp_path / "stamped.parquet"
    pl.DataFrame(
        {
            "ts": pl.datetime_range(
                datetime(2026, 3, 1, 12, 0, 0),
                datetime(2026, 3, 1, 12, 0, 20),
                interval="1s",
                eager=True,
            ),
            "temp": [float(index) for index in range(21)],
        }
    ).write_parquet(path)

    api = Api(settings_dir=tmp_path / "settings")
    api.load_file(str(path))
    api.get_chart_data()

    start = datetime(2026, 3, 1, 12, 0, 5, tzinfo=timezone.utc).timestamp()
    end = datetime(2026, 3, 1, 12, 0, 10, tzinfo=timezone.utc).timestamp()
    result = api.timeline_window_conditions(start, end)

    assert result["conditions"][0]["values"] == [
        "2026-03-01T12:00:05.000000",
        "2026-03-01T12:00:10.000000",
    ]
    # A Datetime column cannot be ordered against a string, so this is
    # also the check that search now coerces one into an instant.
    assert len(api.run_search(result["conditions"])["rows"]) == 6


# --- the five shapes -----------------------------------------------------


def test_a_single_csv_with_a_time_column_gets_a_timeline(tmp_path):
    path = tmp_path / "one.csv"
    pl.DataFrame(
        {"ts": [index / HZ for index in range(500)], "temp": [float(index) for index in range(500)]}
    ).write_csv(path)
    api = Api(settings_dir=tmp_path / "settings")
    api.load_file(str(path))
    api.get_chart_data()

    timeline = api.get_timeline()

    assert timeline["available"] is True
    assert timeline["unit"] == "one.csv"
    assert timeline["unit_label"] == "File"
    assert [c["column"] for c in timeline["channels"]] == ["temp"]
    assert sum(timeline["channels"][0]["n"]) == 500


def test_a_flat_folder_gets_a_timeline_per_file(tmp_path):
    api = Api(settings_dir=tmp_path / "settings")
    _open(api, build_flat(tmp_path / "flat"))
    api.start_chart_build()
    api.wait_for_chart_build(timeout=300)

    timeline = api.get_timeline()

    assert timeline["available"] is True
    assert timeline["unit_label"] == "File"
    assert len(timeline["units"]) == 3
    assert timeline["units"][0].endswith(".parquet")


def test_a_hive_folder_gets_a_timeline_per_file(tmp_path):
    api = Api(settings_dir=tmp_path / "settings")
    _open(api, build_hive(tmp_path / "hive"))
    api.start_chart_build()
    api.wait_for_chart_build(timeout=300)

    timeline = api.get_timeline()

    assert timeline["available"] is True
    assert len(timeline["units"]) == 4
    assert timeline["units"][0].startswith("day=")


def test_a_nested_tree_gets_a_timeline_per_file(tmp_path):
    api = Api(settings_dir=tmp_path / "settings")
    _open(api, build_nested(tmp_path / "nested"))
    api.start_chart_build()
    api.wait_for_chart_build(timeout=300)

    timeline = api.get_timeline()

    assert timeline["available"] is True
    assert len(timeline["units"]) == 4


def test_a_full_tree_gets_a_timeline_per_series(tmp_path):
    api = Api(settings_dir=tmp_path / "settings")
    _open(api, build_full(tmp_path / "full"))
    api.start_chart_build()
    api.wait_for_chart_build(timeout=300)

    timeline = api.get_timeline()

    assert timeline["available"] is True
    assert len(timeline["units"]) == 3  # three series, not eight files


def test_a_dataset_with_no_time_column_is_told_why(tmp_path):
    folder = tmp_path / "plain"
    folder.mkdir()
    for index in range(2):
        pl.DataFrame({"speed": [1.0, 2.0], "label": ["a", "b"]}).write_parquet(
            folder / f"part{index}.parquet"
        )
    api = Api(settings_dir=tmp_path / "settings")
    _open(api, folder)
    api.start_chart_build()
    api.wait_for_chart_build(timeout=300)

    timeline = api.get_timeline()

    assert timeline["available"] is False
    assert "timestamp column" in timeline["reason"]


def test_nothing_loaded_is_told_why(tmp_path):
    api = Api(settings_dir=tmp_path / "settings")

    assert api.get_timeline()["available"] is False


# --- a flat input is untouched -------------------------------------------


@pytest.mark.parametrize("builder", [build_flat, build_nested], ids=["flat", "nested"])
def test_a_flat_inputs_charts_are_what_they_were(builder, tmp_path):
    """The timeline is additive. It moved no number that was already
    there, including the health payload it shares a cache section with."""
    from ml_data_view.chart_data import build_chart_payload
    from ml_data_view.folder_loader import discover_files, load_files

    api = Api(settings_dir=tmp_path / "settings")
    folder = builder(tmp_path / "shape")
    _open(api, folder)
    api.start_chart_build()
    produced = api.wait_for_chart_build(timeout=300)["result"]

    files = discover_files(folder)
    reference = build_chart_payload(load_files(files), files=files)

    for key in (
        "numeric_histograms",
        "categorical_bars",
        "text_length_histograms",
        "value_frequency_histograms",
        "correlated_missingness",
    ):
        assert produced[key] == reference[key], key


def test_the_chart_payload_carries_no_channel_buckets(recordings_api):
    """The per-chunk buckets live in the CACHE, not in the payload every
    other tab and the PDF carry around: 400 buckets x 6 numbers x a
    channel each would multiply the health section's size for a viewer
    most datasets never open."""
    api, _folder = recordings_api

    unit = api._chart_data["timeseries"]["units"]["sys_key=1 / rec_0"]

    assert "buckets" not in unit["channels"]["temp"]
