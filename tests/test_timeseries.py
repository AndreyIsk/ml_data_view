"""Time-series health, fault by fault (timeseries.py, ts_checks.py).

Every fault below is injected at a KNOWN place and asserted at that place.
A check that fires is only half the answer — one that fires on the wrong
recording, or names the wrong chunk, is worse than one that stays quiet,
because it sends someone to look at data that is fine.

The tree is built once by `tree`: two groups, several units each, and one
unit per fault so a finding can be traced to exactly one cause. A CLEAN
unit is in there too, and the last test in each section asserts it stayed
clean — a check that fires on everything has found nothing.

The five input shapes are covered at the bottom: these checks hang off a
detected TIMESTAMP COLUMN, so a single CSV with a time column gets them and
a flat folder without one does not.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl
import pytest

from ml_data_view.api import Api
from ml_data_view.timeseries import Thresholds, analyse_chunk, merge_unit

HZ = 100.0
DT = 1.0 / HZ
ROWS = 600
# Chunks are written back to back, so an ordinary seam has exactly one
# sample interval across it — which is what makes a seam GAP unambiguous.
SPAN = ROWS * DT


def write_chunk(path: Path, start: float, rows: int = ROWS, **faults) -> float:
    """One chunk, with whichever fault was asked for. Returns its last
    timestamp so the caller can lay the next one down contiguously."""
    path.parent.mkdir(parents=True, exist_ok=True)
    ts = start + np.arange(rows, dtype="float64") * DT
    if faults.get("out_of_order"):
        # A block of twenty samples reversed in the middle of the chunk.
        ts[200:220] = ts[200:220][::-1]
    if faults.get("duplicates"):
        ts[100:120] = ts[100]
    if faults.get("rate_change"):
        half = rows // 2
        ts[half:] = ts[half - 1] + np.arange(1, rows - half + 1) * DT * 10
    if faults.get("mid_gap"):
        ts[rows // 2 :] += 5.0

    signal = np.sin(ts)
    channel = np.cos(ts)
    if faults.get("nan_run"):
        signal[150:450] = np.nan
    if faults.get("flat"):
        signal[100:500] = 4.2
    if faults.get("saturated"):
        channel[:] = np.clip(channel * 5, -1.0, 1.0)
    if faults.get("spike"):
        channel[300] = 5_000.0
    pl.DataFrame({"ts": ts, "signal": signal, "channel": channel}).write_parquet(path)
    return float(ts[-1])


def _series(root: Path, group: str, unit: str, count: int = 2, **faults) -> None:
    """A unit of `count` contiguous chunks, the fault on the first."""
    end = 0.0
    for index in range(count):
        end = write_chunk(
            root / group / unit / f"part-{index}.parquet",
            end + DT if index else 0.0,
            **(faults if index == 0 else {}),
        )


@pytest.fixture
def tree(tmp_path) -> Path:
    root = tmp_path / "tree"
    _series(root, "sys=a", "clean", count=3)

    # A gap INSIDE a chunk: the rows are late, the file is there.
    _series(root, "sys=a", "midgap", mid_gap=True)

    # A gap exactly AT a seam: part-1 is written where part-2 would be, so
    # the middle chunk of the three is simply absent.
    write_chunk(root / "sys=a" / "missingchunk" / "part-0.parquet", 0.0)
    write_chunk(root / "sys=a" / "missingchunk" / "part-2.parquet", SPAN * 2)

    # Two chunks covering the same instants.
    write_chunk(root / "sys=a" / "overlap" / "part-0.parquet", 0.0)
    write_chunk(root / "sys=a" / "overlap" / "part-1.parquet", SPAN / 2)

    _series(root, "sys=b", "disorder", out_of_order=True)
    _series(root, "sys=b", "dupes", duplicates=True)
    _series(root, "sys=b", "ratechange", rate_change=True)
    _series(root, "sys=b", "nanrun", nan_run=True)
    _series(root, "sys=b", "flatline", flat=True)
    _series(root, "sys=b", "saturated", saturated=True)

    # One chunk, a tenth the length of everything else — the shape of a
    # recording that was cut off.
    write_chunk(root / "sys=b" / "truncated" / "part-0.parquet", 0.0, rows=60)
    return root


def _open(root: Path, settings_dir: Path, **settings) -> Api:
    api = Api(settings_dir=settings_dir)
    discovery = api.discover_folder(str(root))
    # A second open of the same root finds the mapping already saved and
    # asks nothing — which is the behaviour the cache test depends on.
    if discovery.get("needs_confirmation"):
        proposal = discovery["structure_proposal"]
        api.confirm_structure(
            {
                "root": proposal["root"],
                "roles": {level["depth"]: level["role"] for level in proposal["levels"]},
                "labels": {0: "System", 1: "Recording"},
                "fingerprint": proposal["fingerprint"],
                "timestamp_column": proposal["suggested_timestamp"],
            }
        )
    # A flatline threshold the fixture's 400-row runs actually cross; the
    # shipped default (500) is for real recordings at real rates.
    api.save_settings({"timeseries_flatline_rows": 100, **settings})
    api.start_scope_load()
    api.wait_for_folder_load(timeout=300)
    api.start_chart_build()
    state = api.wait_for_chart_build(timeout=600)
    assert state["error"] is None, state["error"]
    return api


@pytest.fixture
def opened(tree, tmp_path):
    api = _open(tree, tmp_path / "settings")
    return api, api.run_instant_checks()


def _findings(result: dict, check_id: str) -> list[dict]:
    return [f for f in result["findings"] if f["check_id"] == check_id]


def _units_with(result: dict, check_id: str) -> set[str]:
    """Which units a check fired on, read out of the detail text — which is
    what a reader sees, so it is what the test reads.

    A unit id is "group / unit", and every detail begins with it followed
    by a colon, so the label is everything up to the first ": ".
    """
    return {
        finding["detail"].split(": ", 1)[0].strip()
        for finding in _findings(result, check_id)
    }


def _fired_on(result: dict, check_id: str, unit: str) -> list[dict]:
    """The findings of `check_id` that name `unit`."""
    return [
        finding
        for finding in _findings(result, check_id)
        if finding["detail"].split(": ", 1)[0].strip().endswith(unit)
    ]


# --- timestamp faults ------------------------------------------------------


def test_an_out_of_order_block_is_found_in_its_own_unit(opened):
    _api, result = opened
    findings = _fired_on(result, "timestamp_monotonic", "disorder")

    assert len(findings) == 1
    assert findings[0]["severity"] == "critical"
    # Twenty samples reversed produce nineteen backwards steps.
    assert "19 step(s)" in findings[0]["detail"]
    # The clean units are not implicated.
    assert "sys=a / clean" not in _units_with(result, "timestamp_monotonic")


def test_overlapping_chunks_also_count_as_a_backwards_step(opened):
    """Not a duplicate finding: an overlap IS time going backwards at the
    seam, and reporting only one of the two would leave whichever the
    reader looked at first unexplained."""
    _api, result = opened

    assert _fired_on(result, "timestamp_monotonic", "overlap")
    assert _fired_on(result, "chunk_overlap", "overlap")


def test_duplicate_timestamps_are_found_with_their_worst_run(opened):
    _api, result = opened
    findings = _fired_on(result, "duplicate_timestamps", "dupes")

    assert len(findings) == 1
    assert "19 row(s)" in findings[0]["detail"]
    assert "longest unbroken run is 19" in findings[0]["detail"]


def test_a_gap_inside_a_chunk_is_a_gap_not_a_missing_chunk(opened):
    _api, result = opened

    assert _fired_on(result, "time_gaps", "midgap")
    # The five-second hole is inside part-0, so no file is implicated.
    assert not _fired_on(result, "missing_chunk", "midgap")


def test_a_gap_at_a_chunk_seam_is_a_missing_chunk(opened):
    _api, result = opened
    findings = _fired_on(result, "missing_chunk", "missingchunk")

    assert len(findings) == 1
    assert findings[0]["severity"] == "critical"
    assert "part-0.parquet" in findings[0]["detail"]
    assert "part-2.parquet" in findings[0]["detail"]


def test_a_seam_gap_is_not_also_reported_as_an_ordinary_gap(opened):
    """The two are different claims with different fixes, so a hole that
    lands on a seam is reported once, as the more specific one."""
    _api, result = opened

    assert not _fired_on(result, "time_gaps", "missingchunk")


def test_overlapping_chunks_are_found(opened):
    _api, result = opened
    findings = _fired_on(result, "chunk_overlap", "overlap")

    assert len(findings) == 1
    assert findings[0]["severity"] == "critical"
    assert "part-1.parquet" in findings[0]["detail"]


def test_a_rate_change_is_found_and_named_as_one(opened):
    _api, result = opened
    changes = [
        f
        for f in _findings(result, "sample_rate")
        if "changes mid-recording" in f["title"]
    ]

    assert [f for f in changes if f["detail"].startswith("sys=b / ratechange")]
    assert changes[0]["severity"] == "warning"


def test_a_short_recording_is_measured_against_its_siblings(opened):
    """Relative to the median of what is loaded — how long a recording
    ought to be is a property of this dataset and nothing else."""
    _api, result = opened

    findings = _fired_on(result, "short_unit", "truncated")
    assert len(findings) == 1
    assert "against a median of" in findings[0]["detail"]


# --- channel faults --------------------------------------------------------


def test_a_long_nan_run_is_found_with_its_location(opened):
    _api, result = opened
    findings = _fired_on(result, "channel_nan_run", "nanrun")

    assert len(findings) == 1
    assert findings[0]["columns"] == ["signal"]
    assert "300 missing values" in findings[0]["detail"]


def test_a_flatlined_channel_is_found(opened):
    _api, result = opened
    findings = _fired_on(result, "channel_flatline", "flatline")

    assert findings
    assert findings[0]["columns"] == ["signal"]
    assert "399 consecutive rows" in findings[0]["detail"]


def test_a_saturated_channel_is_found(opened):
    _api, result = opened

    findings = _fired_on(result, "channel_saturation", "saturated")
    assert findings
    assert findings[0]["columns"] == ["channel"]
    assert "pins at its limit" in findings[0]["title"]


def test_an_out_of_range_spike_is_exact_against_a_documented_range(tree, tmp_path):
    """With a data dictionary this is a count against a stated rule, and
    is critical. The heuristic path is a different, weaker statement and
    says so in its own text."""
    write_chunk(tree / "sys=b" / "spike" / "part-0.parquet", 0.0, spike=True)
    write_chunk(tree / "sys=b" / "spike" / "part-1.parquet", SPAN, spike=False)

    api = _open(
        tree,
        tmp_path / "settings",
        timeseries_ranges={"channel": [-2.0, 2.0]},
    )
    result = api.run_instant_checks()
    findings = _fired_on(result, "channel_out_of_range", "spike")

    assert len(findings) == 1
    assert findings[0]["severity"] == "critical"
    assert "1 value(s) outside its documented range" in findings[0]["detail"]


def test_the_heuristic_range_rule_labels_itself_as_one(tmp_path):
    """A number nobody can check must not be presented as one they can."""
    folder = tmp_path / "one"
    folder.mkdir()
    rows = 5_000
    ts = np.arange(rows, dtype="float64") * DT
    channel = np.cos(ts)
    channel[10] = 1e6
    pl.DataFrame({"ts": ts, "channel": channel}).write_parquet(folder / "a.parquet")

    api = Api(settings_dir=tmp_path / "settings")
    api.load_file(str(folder / "a.parquet"))
    api.start_chart_build()
    api.wait_for_chart_build(timeout=300)
    findings = _findings(api.run_instant_checks(), "channel_out_of_range")

    assert findings
    assert findings[0]["severity"] == "warning"
    assert "HEURISTIC" in findings[0]["detail"]
    assert "Configure a range in Settings" in findings[0]["detail"]


# --- the clean unit --------------------------------------------------------


def test_the_clean_recording_produces_no_findings(opened):
    """A check that fires on everything has found nothing."""
    _api, result = opened
    timeseries_checks = {
        "timestamp_monotonic",
        "duplicate_timestamps",
        "sample_rate",
        "time_gaps",
        "short_unit",
        "missing_chunk",
        "chunk_overlap",
        "channel_nan_run",
        "channel_flatline",
        "channel_saturation",
        "channel_out_of_range",
    }
    guilty = [
        (f["check_id"], f["detail"])
        for f in result["findings"]
        if f["check_id"] in timeseries_checks and f["detail"].startswith("sys=a / clean")
    ]

    assert guilty == []


def test_every_timeseries_check_ran(opened):
    """The whole tree has a series and a timestamp column, so nothing here
    has an excuse to skip."""
    _api, result = opened
    statuses = {check["id"]: check["status"] for check in result["checks"]}

    for check_id in ("timestamp_monotonic", "missing_chunk", "channel_flatline"):
        assert statuses[check_id] == "done", check_id


# --- applicability, across the input shapes --------------------------------


def test_a_single_csv_with_a_time_column_gets_the_checks(tmp_path):
    """The precondition is a timestamp column and NOTHING else."""
    path = tmp_path / "one.csv"
    rows = 2_000
    ts = np.arange(rows, dtype="float64") * DT
    ts[1000:] += 5.0  # one gap
    signal = np.sin(ts)
    signal[200:700] = np.nan
    pl.DataFrame({"ts": ts, "signal": signal}).write_csv(path)

    api = Api(settings_dir=tmp_path / "settings")
    api.load_file(str(path))
    api.save_settings({"timeseries_flatline_rows": 100})
    api.start_chart_build()
    api.wait_for_chart_build(timeout=300)
    result = api.run_instant_checks()
    statuses = {check["id"]: check["status"] for check in result["checks"]}

    assert statuses["time_gaps"] == "done"
    assert statuses["channel_nan_run"] == "done"
    assert _findings(result, "time_gaps")
    assert _findings(result, "channel_nan_run")
    # ...but a single file has no chunk seams, so those checks skip.
    assert statuses["missing_chunk"] == "skipped"
    assert statuses["chunk_overlap"] == "skipped"


def test_a_file_without_a_time_column_skips_every_timeseries_check(tmp_path):
    path = tmp_path / "one.csv"
    pl.DataFrame({"speed": [1.0, 2.0, 3.0], "label": ["a", "b", "c"]}).write_csv(path)

    api = Api(settings_dir=tmp_path / "settings")
    api.load_file(str(path))
    api.start_chart_build()
    api.wait_for_chart_build(timeout=300)
    statuses = {c["id"]: c["status"] for c in api.run_instant_checks()["checks"]}

    for check_id in ("timestamp_monotonic", "time_gaps", "channel_flatline", "missing_chunk"):
        assert statuses[check_id] == "skipped", check_id


def test_a_flat_folder_with_a_time_column_gets_the_per_file_checks(tmp_path):
    """A flat folder's unit is the FILE, so the timestamp and channel
    checks apply per file and the seam checks do not."""
    folder = tmp_path / "flat"
    folder.mkdir()
    for index in range(3):
        write_chunk(folder / f"part{index}.parquet", index * SPAN * 3, mid_gap=index == 1)

    api = Api(settings_dir=tmp_path / "settings")
    api.discover_folder(str(folder))
    api.start_folder_load(0)
    api.wait_for_folder_load(timeout=300)
    api.start_chart_build()
    api.wait_for_chart_build(timeout=300)
    result = api.run_instant_checks()
    statuses = {c["id"]: c["status"] for c in result["checks"]}

    assert statuses["time_gaps"] == "done"
    assert statuses["missing_chunk"] == "skipped"
    assert _units_with(result, "time_gaps") == {"part1.parquet"}



def test_a_hive_folder_has_no_chunk_seams(tmp_path):
    """Groups but no series: a unit is one file, so "is a chunk missing"
    cannot be asked of it."""
    folder = tmp_path / "hive"
    for day in ("day=1", "day=2"):
        for index in range(2):
            write_chunk(folder / day / f"part{index}.parquet", index * SPAN * 3)

    api = Api(settings_dir=tmp_path / "settings")
    discovery = api.discover_folder(str(folder))
    proposal = discovery["structure_proposal"]
    api.confirm_structure(
        {
            "root": proposal["root"],
            "roles": {level["depth"]: level["role"] for level in proposal["levels"]},
            "fingerprint": proposal["fingerprint"],
            "timestamp_column": "ts",
        }
    )
    api.start_scope_load()
    api.wait_for_folder_load(timeout=300)
    api.start_chart_build()
    api.wait_for_chart_build(timeout=300)
    statuses = {c["id"]: c["status"] for c in api.run_instant_checks()["checks"]}

    assert statuses["time_gaps"] == "done"
    assert statuses["missing_chunk"] == "skipped"
    assert statuses["chunk_overlap"] == "skipped"


# --- the charts ------------------------------------------------------------


def test_the_coverage_strip_marks_data_gaps_and_missing_chunks(opened):
    api, _result = opened
    charts = api._chart_data["timeseries"]["charts"]

    missing = next(key for key in charts if "missingchunk" in key)
    kinds = [segment["kind"] for segment in charts[missing]["coverage"]]
    assert kinds == ["data", "missing", "data"]

    midgap = next(key for key in charts if "midgap" in key)
    assert "gap" in [segment["kind"] for segment in charts[midgap]["coverage"]]


def test_the_charts_are_downsampled_not_raw(opened):
    """A recording of 200 million rows must send the same few hundred
    numbers as one of 200."""
    api, _result = opened
    charts = api._chart_data["timeseries"]["charts"]

    for unit_charts in charts.values():
        assert len(unit_charts["dt"]) <= 400
        assert len(unit_charts["coverage"]) <= 500
        assert len(unit_charts["channels"]) <= 8


def test_the_pdf_offers_a_health_section_only_with_a_time_axis(opened, tmp_path):
    from ml_data_view.report import available_sections

    api, _result = opened

    assert available_sections(api._chart_data)["health"] is True
    assert available_sections({"numeric_histograms": []})["health"] is False


# --- memory is bounded by a batch, not by a recording ----------------------


def test_analysis_memory_does_not_grow_with_chunk_count(tmp_path):
    """The claim the whole processing model rests on.

    Many small chunks analysed one after another must not accumulate: each
    per-chunk artifact is a fixed-size histogram and a handful of bounded
    lists, and the reader releases every batch. Measured as the size of
    what is HELD, which is what would grow if the design were wrong.
    """
    import json

    folder = tmp_path / "many"
    folder.mkdir()
    sizes = []
    end = 0.0
    for index in range(40):
        path = folder / f"part-{index:03d}.parquet"
        end = write_chunk(path, end + DT, rows=2_000)
        artifact = analyse_chunk(path, "ts", ["signal", "channel"])
        sizes.append(len(json.dumps(artifact)))

    # Every per-chunk artifact is the same size regardless of how many
    # came before it — there is no accumulating state.
    assert max(sizes) - min(sizes) < max(sizes) * 0.25
    assert max(sizes) < 20_000


def test_a_merged_unit_stays_bounded_however_many_chunks(tmp_path):
    """The merged record is capped too: gaps, buckets and segments all have
    limits, so a recording of a thousand chunks reports the same shape as
    one of two."""
    import json

    folder = tmp_path / "many"
    folder.mkdir()
    artifacts, names = [], []
    end = 0.0
    for index in range(60):
        path = folder / f"part-{index:03d}.parquet"
        end = write_chunk(path, end + DT, rows=1_000)
        artifacts.append(analyse_chunk(path, "ts", ["signal"]))
        names.append(path.name)

    merged = merge_unit(artifacts, names, Thresholds())

    assert merged["rows"] == 60_000
    assert len(merged["buckets"]) <= 400
    assert len(merged["gaps"]) <= 200
    assert len(json.dumps({k: v for k, v in merged.items() if k != "chunks"})) < 200_000


def test_per_chunk_results_are_cached_and_not_recomputed(tree, tmp_path):
    """The artifact is cached per chunk like every other per-chunk
    statistic, so a second open reads nothing."""
    from ml_data_view.file_profile import TIMESERIES_ARTIFACT_VERSION
    from ml_data_view.profile_cache import SECTION_TIMESERIES, ProfileCache

    api = _open(tree, tmp_path / "settings")
    first = api._chart_data["timeseries"]
    assert first["chunks_scanned"] > 0

    with ProfileCache(api._cache_path()) as cache:
        sample = next(tree.rglob("*.parquet"))
        payload = cache.get(sample)
    # [artifact version, timestamp column, documented ranges] — the three
    # inputs that decide whether a stored artifact still describes what
    # this code would compute.
    signature = payload[SECTION_TIMESERIES]["signature"]
    assert signature[0] == TIMESERIES_ARTIFACT_VERSION
    assert signature[1] == "ts"

    second = _open(tree, tmp_path / "settings")
    assert second._chart_data["timeseries"]["chunks_scanned"] == 0


def test_peak_memory_under_the_ceiling_does_not_grow_with_recording_length(tmp_path):
    """The claim the processing model rests on, measured for real.

    The chart build — which is where the time-series pass runs — happens in
    a capped worker subprocess, and the Job Object records that process's
    OWN peak. So this analyses a folder of ten chunks and then one of
    sixty, and asserts the peak barely moves: six times the rows must not
    cost six times the memory, because a chunk is read a batch at a time
    and only a fixed-size artifact is kept.

    Skipped where the OS cannot enforce a ceiling — without one there is no
    per-job peak to read, and guessing would prove nothing.
    """
    from ml_data_view.jobs import memory_ceiling_supported

    if not memory_ceiling_supported():
        pytest.skip("no OS-enforced memory ceiling on this platform")

    def build(name: str, chunks: int) -> int:
        folder = tmp_path / name
        folder.mkdir()
        end = 0.0
        for index in range(chunks):
            end = write_chunk(folder / f"part-{index:03d}.parquet", end + DT, rows=20_000)
        api = Api(settings_dir=tmp_path / f"settings-{name}")
        api.discover_folder(str(folder))
        api.start_folder_load(0)
        api.wait_for_folder_load(timeout=600)
        api.start_chart_build()
        state = api.wait_for_chart_build(timeout=900)
        assert state["error"] is None, state["error"]
        assert state["result"]["timeseries"]["chunks_total"] == chunks
        return int(state["peak_bytes"])

    small = build("small", 10)
    large = build("large", 60)

    # Six times the data. Anything proportional would show up as a
    # multiple here; what is allowed is the fixed cost of reading a wider
    # batch plus the interpreter's own noise.
    assert large < small * 1.5, (
        f"peak grew with recording length: {small / 1e6:.0f}MB for 10 chunks, "
        f"{large / 1e6:.0f}MB for 60"
    )
