"""The channel timeline: bucket arithmetic, read planning, and the overlay.

Three claims are pinned here.

  The BUCKETS are right — a scanned bucket equals a direct computation on
  the same rows, and a cached bucket folded from per-chunk ones equals the
  same thing at a coarser resolution.

  Only the chunks (and row groups) a window TOUCHES are read, which is
  what makes zooming into a large recording affordable at all.

  The overlay's marks land where the findings said the problems were.
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from ml_data_view import timeline as tl
from ml_data_view import timeseries as ts

HZ = 10.0
ROWS_PER_GROUP = 500


def write_chunk(path: Path, start: float, rows: int, *, offset: float = 0.0) -> None:
    """One chunk with a time column and two channels, in small row groups
    so a window can select a few of them."""
    path.parent.mkdir(parents=True, exist_ok=True)
    frame = pl.DataFrame(
        {
            "ts": [start + index / HZ for index in range(rows)],
            "temp": [offset + math.sin(index / 7.0) for index in range(rows)],
            "speed": [offset + (index % 13) * 0.5 for index in range(rows)],
        }
    )
    frame.write_parquet(path, row_group_size=ROWS_PER_GROUP)


def analyse(path: Path, channels=("temp", "speed"), thresholds=None) -> dict:
    return ts.analyse_chunk(path, "ts", list(channels), thresholds or ts.Thresholds())


def chunks_for(paths: list[Path], **kwargs) -> list[tl.Chunk]:
    artifacts = {str(path): analyse(path, **kwargs) for path in paths}
    return tl.chunks_from_artifacts(paths, artifacts)


@pytest.fixture
def recording(tmp_path):
    """Four chunks, 1000 rows each, contiguous in time."""
    paths = []
    for index in range(4):
        path = tmp_path / f"part-{index}.parquet"
        write_chunk(path, index * 100.0, 1000)
        paths.append(path)
    return paths


# --- the per-chunk buckets the cache now carries --------------------------


def test_a_chunk_caches_per_channel_buckets(tmp_path):
    path = tmp_path / "one.parquet"
    write_chunk(path, 0.0, 1000)

    result = analyse(path)

    buckets = result["channels"]["temp"]["buckets"]
    assert set(buckets) == {"t0", "t1", "n", "min", "max", "sum"}
    assert len(buckets["t0"]) == ts.BUCKETS_PER_CHUNK
    assert sum(buckets["n"]) == 1000


def test_the_cached_buckets_are_exact_for_the_rows_they_cover(tmp_path):
    path = tmp_path / "one.parquet"
    write_chunk(path, 0.0, 1000)
    frame = pl.read_parquet(path)

    buckets = analyse(path)["channels"]["temp"]["buckets"]

    # Every value the file holds is inside exactly one bucket's range, and
    # the extremes add up to the column's own.
    assert min(v for v in buckets["min"] if v is not None) == pytest.approx(
        frame["temp"].min()
    )
    assert max(v for v in buckets["max"] if v is not None) == pytest.approx(
        frame["temp"].max()
    )
    assert sum(buckets["sum"]) == pytest.approx(frame["temp"].sum())


def test_a_chunk_records_when_each_extreme_happened(tmp_path):
    path = tmp_path / "one.parquet"
    write_chunk(path, 50.0, 1000)
    frame = pl.read_parquet(path)

    channel = analyse(path)["channels"]["temp"]

    expected = frame.filter(pl.col("temp") == frame["temp"].max())["ts"][0]
    assert channel["max_at"] == pytest.approx(expected)
    assert channel["min_at"] is not None


def test_folding_buckets_keeps_every_row_and_both_extremes():
    rows = [(float(i), float(i) + 1, 10, float(i), float(i) + 5, float(i) * 10) for i in range(20)]

    folded = ts.fold_channel_buckets(rows, 5)

    assert len(folded) <= 5
    assert sum(row[2] for row in folded) == sum(row[2] for row in rows)
    assert min(row[3] for row in folded) == min(row[3] for row in rows)
    assert max(row[4] for row in folded) == max(row[4] for row in rows)
    assert sum(row[5] for row in folded) == pytest.approx(sum(row[5] for row in rows))


# --- a scanned window equals a direct computation -------------------------


def _direct(paths: list[Path], column: str, window, buckets: int) -> dict:
    """The same aggregate, computed the obvious way over every row."""
    frame = pl.concat([pl.read_parquet(path) for path in paths])
    frame = frame.filter((pl.col("ts") >= window[0]) & (pl.col("ts") <= window[1]))
    width = (window[1] - window[0]) / buckets
    index = np.clip(((frame["ts"].to_numpy() - window[0]) / width).astype("int64"), 0, buckets - 1)
    values = frame[column].to_numpy()
    out = {"n": [], "min": [], "max": [], "mean": []}
    for bucket in range(buckets):
        chosen = values[index == bucket]
        out["n"].append(len(chosen))
        out["min"].append(float(chosen.min()) if len(chosen) else None)
        out["max"].append(float(chosen.max()) if len(chosen) else None)
        out["mean"].append(float(chosen.mean()) if len(chosen) else None)
    return out


def test_scanned_buckets_equal_a_direct_computation(recording):
    # Finer than the cached buckets can answer (4 chunks x 24), so this
    # takes the read — which is the path being checked.
    window = (0.0, 400.0)
    chunks = chunks_for(recording)

    result = tl.build_timeline(chunks, "ts", ["temp"], window=window, buckets=200)

    assert result["source"] == "scan"
    produced = result["channels"][0]
    expected = _direct(recording, "temp", window, 200)
    assert produced["n"] == expected["n"]
    for key in ("min", "max", "mean"):
        for got, want in zip(produced[key], expected[key]):
            if want is None:
                assert got is None
            else:
                assert got == pytest.approx(want)


def test_a_zoomed_window_equals_a_direct_computation_over_that_window(recording):
    window = (120.0, 155.0)
    chunks = chunks_for(recording)

    result = tl.build_timeline(chunks, "ts", ["speed"], window=window, buckets=25)

    expected = _direct(recording, "speed", window, 25)
    assert result["channels"][0]["n"] == expected["n"]
    assert sum(result["channels"][0]["n"]) == sum(expected["n"]) > 0


def test_an_empty_bucket_is_none_rather_than_zero(tmp_path):
    """A channel with no rows in a stretch and a channel reading zero
    there are different things."""
    first = tmp_path / "a.parquet"
    second = tmp_path / "b.parquet"
    write_chunk(first, 0.0, 100)
    write_chunk(second, 500.0, 100)  # a long gap between them

    result = tl.build_timeline(chunks_for([first, second]), "ts", ["temp"], buckets=50)

    channel = result["channels"][0]
    assert any(value is None for value in channel["min"])
    assert all(
        (count == 0) == (value is None) for count, value in zip(channel["n"], channel["min"])
    )


# --- only what the window touches is read --------------------------------


def test_only_the_chunks_overlapping_the_window_are_planned(recording):
    chunks = chunks_for(recording)

    plan = tl.plan_read(chunks, "ts", (210.0, 240.0))

    assert [Path(path).name for path, _groups in plan.groups] == ["part-2.parquet"]


def test_a_window_spanning_two_chunks_plans_both(recording):
    chunks = chunks_for(recording)

    plan = tl.plan_read(chunks, "ts", (195.0, 205.0))

    assert sorted(Path(path).name for path, _g in plan.groups) == [
        "part-1.parquet",
        "part-2.parquet",
    ]


def test_row_groups_outside_the_window_are_planned_away(recording):
    """Within a chunk, too: a zoom reads the row groups the window touches
    and not the rest, which is what makes one enormous chunk zoomable."""
    chunks = chunks_for(recording)

    plan = tl.plan_read(chunks, "ts", (200.0, 205.0))

    (_path, groups) = plan.groups[0]
    assert groups == [0]  # 1000 rows in 500-row groups, and 5s is the first
    assert plan.rows == ROWS_PER_GROUP


def test_only_the_planned_chunks_are_actually_opened(recording, monkeypatch):
    """Collect monitoring: nothing outside the window is opened at all."""
    opened: list[str] = []
    original = tl._iter_window

    def watched(path, columns, row_groups):
        opened.append(Path(path).name)
        return original(path, columns, row_groups)

    monkeypatch.setattr(tl, "_iter_window", watched)
    chunks = chunks_for(recording)

    tl.build_timeline(chunks, "ts", ["temp"], window=(210.0, 240.0), buckets=100)

    assert set(opened) == {"part-2.parquet"}


def test_a_chunk_with_unknown_bounds_is_always_looked_at(tmp_path):
    """Excluding it would silently drop rows from an answer presented as
    complete."""
    path = tmp_path / "mystery.parquet"
    write_chunk(path, 0.0, 100)
    chunks = [tl.Chunk(path=path, first_ts=None, last_ts=None, artifact={})]

    assert chunks[0].overlaps((1000.0, 2000.0)) is True


# --- the budget ----------------------------------------------------------


def test_a_window_over_budget_is_answered_from_the_cache_and_says_so(recording):
    chunks = chunks_for(recording)

    result = tl.build_timeline(chunks, "ts", ["temp"], buckets=600, max_rows=100)

    assert result["source"] == "cache"
    assert result["rows_read"] == 0
    assert "budget" in result["note"]
    assert sum(result["channels"][0]["n"]) == 4000


def test_zooming_brings_the_same_recording_back_under_budget(recording):
    chunks = chunks_for(recording)

    wide = tl.build_timeline(chunks, "ts", ["temp"], buckets=600, max_rows=600)
    zoomed = tl.build_timeline(
        chunks, "ts", ["temp"], window=(200.0, 205.0), buckets=600, max_rows=600
    )

    assert wide["source"] == "cache"
    assert zoomed["source"] == "scan"
    assert zoomed["rows_read"] == ROWS_PER_GROUP


def test_a_read_that_would_tell_us_nothing_new_is_not_taken(recording):
    """The cached buckets already answer a request this coarse."""
    chunks = chunks_for(recording)

    result = tl.build_timeline(chunks, "ts", ["temp"], buckets=8)

    assert result["source"] == "cache"
    assert result["note"] == ""


def test_the_payload_says_how_it_was_produced(recording):
    chunks = chunks_for(recording)

    result = tl.build_timeline(chunks, "ts", ["temp"], window=(0.0, 100.0), buckets=200)

    assert result["source"] == "scan"
    assert result["chunks_total"] == 4
    assert result["chunks_in_window"] == 2  # part-0 ends exactly where part-1 begins
    assert result["chunks_read"] >= 1
    assert result["rows_read"] > 0


# --- the cached path is exact, only coarser ------------------------------


def test_cached_buckets_carry_the_same_extremes_as_a_scan(recording):
    chunks = chunks_for(recording)

    scanned = tl.build_timeline(chunks, "ts", ["temp"], buckets=600)
    cached = tl.build_timeline(chunks, "ts", ["temp"], buckets=600, max_rows=1)

    assert scanned["source"] == "scan" and cached["source"] == "cache"
    for key, reduce in (("min", min), ("max", max)):
        assert reduce(v for v in cached["channels"][0][key] if v is not None) == pytest.approx(
            reduce(v for v in scanned["channels"][0][key] if v is not None)
        )
    assert sum(cached["channels"][0]["n"]) == sum(scanned["channels"][0]["n"])


def test_a_unit_with_no_time_bounds_reports_why(tmp_path):
    chunks = [tl.Chunk(path=tmp_path / "x.parquet", artifact={})]

    result = tl.build_timeline(chunks, "ts", ["temp"])

    assert result["available"] is False
    assert "time range" in result["reason"]


# --- the overlay ---------------------------------------------------------


def _merged(paths: list[Path], thresholds=None) -> dict:
    thresholds = thresholds or ts.Thresholds()
    artifacts = [analyse(path, thresholds=thresholds) for path in paths]
    return ts.merge_unit(artifacts, [path.name for path in paths], thresholds)


def test_a_gap_is_marked_where_the_finding_says_it_is(tmp_path):
    first = tmp_path / "a.parquet"
    second = tmp_path / "b.parquet"
    write_chunk(first, 0.0, 100)
    write_chunk(second, 60.0, 100)  # a 50s hole after the first ends at 9.9s
    unit = _merged([first, second])

    marks = tl.overlay(unit, ["temp"])

    gap = unit["gaps"][0]
    mark = next(m for m in marks if m["kind"] in {"gap", "missing"})
    assert mark["t0"] == pytest.approx(gap["at"])
    assert mark["t1"] == pytest.approx(gap["at"] + gap["duration"])


def test_a_missing_chunk_is_marked_as_one_and_is_critical(tmp_path):
    first = tmp_path / "a.parquet"
    second = tmp_path / "b.parquet"
    write_chunk(first, 0.0, 100)
    write_chunk(second, 60.0, 100)

    marks = tl.overlay(_merged([first, second]), ["temp"])

    mark = marks[0]
    assert mark["kind"] == "missing"
    assert mark["severity"] == "critical"
    assert mark["channel"] is None


def test_a_dropout_is_marked_on_its_own_channel(tmp_path):
    path = tmp_path / "holes.parquet"
    values = [1.0] * 2000
    values[500:900] = [None] * 400
    pl.DataFrame(
        {
            "ts": [index / HZ for index in range(2000)],
            "temp": values,
        }
    ).write_parquet(path)
    unit = _merged([path])

    marks = tl.overlay(unit, ["temp"])

    dropout = next(m for m in marks if m["kind"] == "nan")
    assert dropout["channel"] == "temp"
    segment = unit["channels"]["temp"]["segments"][0]
    assert dropout["t0"] == pytest.approx(segment["at"])
    assert dropout["t1"] == pytest.approx(
        segment["at"] + segment["length"] * unit["dt_median"]
    )


def test_a_documented_range_marks_the_extreme_that_broke_it(tmp_path):
    path = tmp_path / "spike.parquet"
    values = [1.0] * 2000
    values[1234] = 900.0
    pl.DataFrame(
        {"ts": [index / HZ for index in range(2000)], "temp": values}
    ).write_parquet(path)
    thresholds = ts.Thresholds(ranges={"temp": (0.0, 10.0)})
    unit = _merged([path], thresholds)

    marks = tl.overlay(unit, ["temp"], ranges={"temp": (0.0, 10.0)})

    spike = next(m for m in marks if m["kind"] == "out_of_range")
    assert spike["t0"] == pytest.approx(1234 / HZ)
    assert spike["severity"] == "critical"
    assert "documented" in spike["detail"]


def test_without_a_documented_range_the_mark_says_heuristic(tmp_path):
    path = tmp_path / "spike.parquet"
    values = [1.0 + (index % 3) * 0.01 for index in range(2000)]
    values[1500] = 900.0
    pl.DataFrame(
        {"ts": [index / HZ for index in range(2000)], "temp": values}
    ).write_parquet(path)

    marks = tl.overlay(_merged([path]), ["temp"])

    spike = next(m for m in marks if m["kind"] == "out_of_range")
    assert spike["severity"] == "warning"
    assert "HEURISTIC" in spike["detail"]
    assert spike["t0"] == pytest.approx(1500 / HZ)


def test_the_overlay_and_the_finding_agree_about_the_same_spike(tmp_path):
    """One implementation of the rule, read by both."""
    from ml_data_view.findings import CheckContext
    from ml_data_view.ts_checks import check_channel_out_of_range

    path = tmp_path / "spike.parquet"
    values = [1.0 + (index % 3) * 0.01 for index in range(2000)]
    values[1500] = 900.0
    pl.DataFrame(
        {"ts": [index / HZ for index in range(2000)], "temp": values}
    ).write_parquet(path)
    unit = _merged([path])
    payload = {"timeseries": {"units": {"u": unit}, "thresholds": ts.Thresholds().to_dict()}}

    findings = check_channel_out_of_range(CheckContext(dataset_info={}, chart_data=payload))
    marks = [m for m in tl.overlay(unit, ["temp"]) if m["kind"] == "out_of_range"]

    assert len(findings) == 1
    assert len(marks) == 1
    assert str(unit["channels"]["temp"]["max"]) in findings[0].detail
    assert marks[0]["t0"] == pytest.approx(unit["channels"]["temp"]["max_at"])


def test_a_clean_recording_has_nothing_to_overlay(recording):
    assert tl.overlay(_merged(recording), ["temp", "speed"]) == []


def test_the_overlay_is_bounded(tmp_path):
    unit = {
        "dt_median": 0.1,
        "gaps": [{"at": float(i), "duration": 1.0, "at_boundary": False} for i in range(500)],
        "channels": {},
    }

    assert len(tl.overlay(unit, [])) == tl.MAX_MARKS


# --- a selected window ---------------------------------------------------


def test_window_stats_summarise_the_buckets_on_screen(recording):
    chunks = chunks_for(recording)
    result = tl.build_timeline(chunks, "ts", ["temp"], window=(0.0, 100.0), buckets=100)

    stats = tl.window_stats(result, (0.0, 50.0))

    channel = stats["channels"][0]
    assert channel["column"] == "temp"
    assert channel["n"] > 0
    assert channel["min"] <= channel["mean"] <= channel["max"]
    assert stats["duration"] == 50.0


def test_window_stats_say_which_rows_they_actually_describe(recording):
    chunks = chunks_for(recording)
    result = tl.build_timeline(chunks, "ts", ["temp"], buckets=600, max_rows=1)

    stats = tl.window_stats(result, (137.3, 268.9))

    assert stats["source"] == "cache"
    assert stats["exact"] is False
    assert stats["covered"] is not None
    assert stats["covered"][0] >= 137.3


def test_window_stats_over_an_empty_stretch_report_nothing_not_zero(recording):
    chunks = chunks_for(recording)
    result = tl.build_timeline(chunks, "ts", ["temp"], window=(0.0, 400.0), buckets=100)

    stats = tl.window_stats(result, (399.99, 400.0))

    assert stats["channels"][0]["n"] == 0
    assert stats["channels"][0]["mean"] is None
