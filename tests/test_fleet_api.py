"""The catalogue and the group comparison, end to end, across every shape.

Two things are being pinned here.

FIRST, that a SYNTHETIC FLEET produces the right answers: one system with
a shifted channel and one whose recordings are systematically short are
each flagged, on the right channel, and the well-behaved systems are not.

SECOND, and at least as importantly, that none of it is assumed. The five
input shapes Otter must keep working for — a single file, a flat folder,
hive-only partitions, an arbitrary nested tree, and the full group +
series tree — all get a catalogue; the ones with nothing to group by skip
the comparison rather than failing it; and a flat folder's payload is what
it was before any of this existed.
"""
from __future__ import annotations

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
# One cycle of the channel pattern. Chunk lengths are whole multiples of
# it, which makes every chunk's value distribution EXACTLY uniform over
# the same values however many rows it holds. That is what lets a system
# with short recordings be short without also having a different-looking
# channel — the two properties this fixture is testing have to be
# separable, or a test cannot say which one was detected.
PERIOD = 150


def _pattern(index: int, step: int) -> float:
    """A deterministic, equidistributed sweep of PERIOD distinct values.

    Not a sawtooth and not a ramp, for two specific reasons. A sawtooth
    revisits its extremes often enough to trip the saturation check, and a
    ramp ties the range of values to the number of rows — so a short
    recording would look like a channel with a different distribution
    rather than like a short recording.
    """
    return ((index * step) % PERIOD) / PERIOD


def _chunk(path: Path, start: float, rows: int, *, temp: float = 20.0) -> None:
    """One chunk of a recording: a time column and two channels."""
    path.parent.mkdir(parents=True, exist_ok=True)
    pl.DataFrame(
        {
            "ts": [start + index / HZ for index in range(rows)],
            "temp": [temp + _pattern(index, 37) for index in range(rows)],
            # Identical in every system, so a divergence reported here
            # would be a false positive rather than the fixture's doing.
            "speed": [5.0 + _pattern(index, 53) for index in range(rows)],
        }
    ).write_parquet(path)


def _recording(root: Path, start: float, *, temp: float, chunk_rows: int) -> None:
    for index, name in enumerate(("part-1.parquet", "part-2.parquet")):
        _chunk(
            root / name,
            start + index * chunk_rows / HZ,
            chunk_rows,
            temp=temp,
        )


def build_fleet_tree(root: Path) -> Path:
    """Four systems. Two ordinary; one with a shifted channel and one
    whose recordings are systematically half as long.

      sys_key=1, sys_key=2   three 600-row recordings each, temp ~20
      sys_key=3              three 600-row recordings, temp ~40  (SHIFTED)
      sys_key=4              three 300-row recordings, temp ~20  (SHORT)
    """
    plans = [
        ("sys_key=1", 20.0, PERIOD * 2),
        ("sys_key=2", 20.0, PERIOD * 2),
        ("sys_key=3", 40.0, PERIOD * 2),
        ("sys_key=4", 20.0, PERIOD),
    ]
    for system, temp, chunk_rows in plans:
        for recording in range(3):
            _recording(
                root / system / f"rec_{recording}",
                float(BASE_TS) + recording * 10_000,
                temp=temp,
                chunk_rows=chunk_rows,
            )
    return root


# The mean of one full cycle of the pattern, which is what every channel
# mean here works out to above its centre.
PATTERN_MEAN = (PERIOD - 1) / (2 * PERIOD)


@pytest.fixture
def fleet_api(tmp_path):
    folder = build_fleet_tree(tmp_path / "fleet")
    api = Api(settings_dir=tmp_path / "settings")
    discovery = api.discover_folder(str(folder))
    _confirm(api, discovery, labels={0: "System", 1: "Recording"})
    api.start_scope_load()
    api.wait_for_folder_load(timeout=300)
    api.start_chart_build()
    api.wait_for_chart_build(timeout=600)
    return api, folder


# --- the catalogue --------------------------------------------------------


def test_the_catalogue_has_one_row_per_recording(fleet_api):
    api, _folder = fleet_api

    catalog = api.get_catalog()

    assert catalog["available"] is True
    assert len(catalog["rows"]) == 12
    assert catalog["unit_label"] == "Recording"
    assert catalog["group_labels"] == ["System"]


def test_catalogue_rows_carry_the_right_numbers(fleet_api):
    api, _folder = fleet_api

    row = next(r for r in api.get_catalog()["rows"] if r["label"] == "sys_key=1 / rec_0")

    assert row["chunk_count"] == 2
    assert row["rows"] == 600
    assert row["duration"] == pytest.approx(599 / HZ, rel=1e-6)
    # Read off the log-spaced dt histogram — ten bins per decade, so
    # about 12% resolution. That is the documented merge, not a defect.
    assert row["hz"] == pytest.approx(HZ, rel=0.15)
    assert row["channel_count"] == 2  # temp and speed; ts is the time axis
    assert row["bytes"] > 0


def test_the_short_systems_rows_are_short(fleet_api):
    api, _folder = fleet_api

    rows = {r["label"]: r for r in api.get_catalog()["rows"]}

    assert rows["sys_key=4 / rec_0"]["rows"] == 300
    assert rows["sys_key=4 / rec_0"]["duration"] < rows["sys_key=1 / rec_0"]["duration"]


def test_the_system_summaries_roll_up(fleet_api):
    api, _folder = fleet_api

    groups = {g["value"]: g for g in api.get_catalog()["groups"]}

    assert groups["sys_key=1"]["unit_count"] == 3
    assert groups["sys_key=1"]["rows"] == 1800
    assert groups["sys_key=4"]["rows"] == 900
    assert set(groups["sys_key=1"]["health"]) >= {"critical", "warning", "info", "ok"}


def test_a_healthy_recording_reports_healthy(fleet_api):
    api, _folder = fleet_api

    rows = {r["label"]: r for r in api.get_catalog()["rows"]}

    assert rows["sys_key=1 / rec_0"]["health"]["status"] == "ok"


def test_the_catalogue_exports_as_csv(fleet_api, tmp_path):
    api, _folder = fleet_api
    out = tmp_path / "catalog.csv"

    result = api.export_catalog(str(out))

    assert result["rows"] == 12
    lines = out.read_text(encoding="utf-8").splitlines()
    assert lines[0].startswith("System,Recording,Chunks,Rows,")
    assert len(lines) == 13


def test_exporting_before_a_chart_build_is_an_error_not_an_empty_file(tmp_path):
    folder = build_fleet_tree(tmp_path / "fleet")
    api = Api(settings_dir=tmp_path / "settings")
    discovery = api.discover_folder(str(folder))
    _confirm(api, discovery)
    api.start_scope_load()
    api.wait_for_folder_load(timeout=300)

    result = api.export_catalog(str(tmp_path / "nope.csv"))

    assert "error" in result
    assert not (tmp_path / "nope.csv").exists()


# --- the group comparison -------------------------------------------------


def test_the_deviant_system_is_flagged_on_the_right_channel(fleet_api):
    api, _folder = fleet_api

    fleet = api.get_fleet()

    assert fleet["available"] is True
    assert fleet["group_label"] == "System"
    temp = next(c for c in fleet["channels"] if c["column"] == "temp")
    scores = {entry["value"]: entry for entry in temp["groups"]}
    assert scores["sys_key=3"]["severity"] == "significant"
    for system in ("sys_key=1", "sys_key=2", "sys_key=4"):
        assert scores[system]["severity"] == "stable", system


def test_the_channel_that_did_not_move_is_not_flagged(fleet_api):
    api, _folder = fleet_api

    speed = next(c for c in api.get_fleet()["channels"] if c["column"] == "speed")

    assert all(entry["severity"] == "stable" for entry in speed["groups"])


def test_the_timestamp_column_is_not_compared_as_a_channel(fleet_api):
    api, _folder = fleet_api

    assert "ts" not in {c["column"] for c in api.get_fleet()["channels"]}


def test_a_groups_mean_is_exact_from_the_cached_sums(fleet_api):
    api, _folder = fleet_api

    temp = next(c for c in api.get_fleet()["channels"] if c["column"] == "temp")
    scores = {entry["value"]: entry for entry in temp["groups"]}

    assert scores["sys_key=1"]["mean"] == pytest.approx(20.0 + PATTERN_MEAN, abs=0.01)
    assert scores["sys_key=3"]["mean"] == pytest.approx(40.0 + PATTERN_MEAN, abs=0.01)


def test_the_short_system_is_reported_as_systematically_short(fleet_api):
    api, _folder = fleet_api

    groups = {g["value"]: g for g in api.get_fleet()["units"]["groups"]}

    assert groups["sys_key=4"]["duration_ratio"] == pytest.approx(0.5, rel=0.05)
    assert groups["sys_key=1"]["duration_ratio"] == pytest.approx(1.0, rel=0.05)


def test_a_uniform_fleet_raises_no_recording_level_anomaly(fleet_api):
    """Every recording within a system is identical here, so nothing in
    one is unlike its siblings."""
    api, _folder = fleet_api

    assert api.get_fleet()["unit_anomalies"] == []


def test_systems_recorded_at_the_same_time_are_not_a_time_ordering(fleet_api):
    api, _folder = fleet_api

    assert api.get_fleet()["time_ordered"] is False


# --- the findings ---------------------------------------------------------


def _findings(api) -> list[dict]:
    return api.run_instant_checks()["findings"]


def test_the_divergence_reaches_the_findings_tab(fleet_api):
    api, _folder = fleet_api

    found = [f for f in _findings(api) if f["check_id"] == "fleet_channel_divergence"]

    assert any("sys_key=3" in f["title"] and "temp" in f["title"] for f in found)
    assert not any("sys_key=1" in f["title"] for f in found)


def test_the_divergence_finding_names_its_method(fleet_api):
    api, _folder = fleet_api

    found = next(f for f in _findings(api) if f["check_id"] == "fleet_channel_divergence")

    assert "PSI" in found["detail"]


def test_the_short_system_reaches_the_findings_tab(fleet_api):
    api, _folder = fleet_api

    found = [f for f in _findings(api) if f["check_id"] == "fleet_group_duration"]

    assert len(found) == 1
    assert "sys_key=4" in found[0]["title"]
    assert "shorter" in found[0]["title"]


def test_a_recording_unlike_its_siblings_is_flagged_and_names_its_unit(tmp_path):
    folder = build_fleet_tree(tmp_path / "fleet")
    # One recording of system 1 truncated: unlike its own siblings, which
    # is a different claim from system 4's recordings all being short.
    for part in ("part-1.parquet", "part-2.parquet"):
        (folder / "sys_key=1" / "rec_2" / part).unlink()
    _recording(
        folder / "sys_key=1" / "rec_2",
        float(BASE_TS + 20_000),
        temp=20.0,
        chunk_rows=PERIOD // 2,
    )

    api = Api(settings_dir=tmp_path / "settings")
    discovery = api.discover_folder(str(folder))
    _confirm(api, discovery, labels={0: "System", 1: "Recording"})
    api.start_scope_load()
    api.wait_for_folder_load(timeout=300)
    api.start_chart_build()
    api.wait_for_chart_build(timeout=600)

    outliers = [f for f in _findings(api) if f["check_id"] == "unit_outlier"]

    assert [f["unit"] for f in outliers] == ["sys_key=1 / rec_2"]
    rows = {r["label"]: r for r in api.get_catalog()["rows"]}
    assert rows["sys_key=1 / rec_2"]["health"]["status"] in {"warning", "critical"}
    assert rows["sys_key=1 / rec_0"]["health"]["status"] == "ok"


# --- nothing is re-read ---------------------------------------------------


def test_a_second_profile_reads_no_files_at_all(fleet_api):
    """The catalogue and the comparison are arithmetic over cached
    statistics. Re-profiling an unchanged fleet must scan zero files and
    still produce both."""
    api, _folder = fleet_api

    api.start_chart_build()
    result = api.wait_for_chart_build(timeout=600)["result"]

    assert result["cache_stats"]["files_scanned"] == 0
    assert result["cache_stats"]["files_from_cache"] == 24
    assert result["timeseries"]["chunks_scanned"] == 0
    assert result["catalog"]["available"] is True
    assert result["fleet"]["available"] is True


def test_the_second_profiles_catalogue_and_comparison_are_identical(fleet_api):
    api, _folder = fleet_api
    first_catalog = api.get_catalog()
    first_fleet = api.get_fleet()

    api.start_chart_build()
    api.wait_for_chart_build(timeout=600)

    assert api.get_catalog() == first_catalog
    assert api.get_fleet() == first_fleet


# --- launching a comparison from the catalogue ----------------------------


def test_two_recordings_can_be_compared_from_the_catalogue(fleet_api):
    api, _folder = fleet_api

    started = api.start_scope_comparison(
        {"values": ["sys_key=1"], "unit_id": "rec_0"},
        {"values": ["sys_key=3"], "unit_id": "rec_0"},
    )

    assert started["started"] is True
    assert started["label_a"] == "sys_key=1 / rec_0"
    assert started["label_b"] == "sys_key=3 / rec_0"
    assert started["files_a"] == 2
    assert started["schema"]["shared"] == ["ts", "temp", "speed"]

    result = api.wait_for_comparison(timeout=300)["result"]
    temp = next(row for row in result["columns"] if row["column"] == "temp")
    assert temp["severity"] == "significant"


def test_two_systems_can_be_compared_from_the_catalogue(fleet_api):
    api, _folder = fleet_api

    started = api.start_scope_comparison(
        {"values": ["sys_key=1"]}, {"values": ["sys_key=2"]}
    )

    assert started["label_a"] == "sys_key=1"
    assert started["files_a"] == 6
    result = api.wait_for_comparison(timeout=300)["result"]
    assert all(row["severity"] == "stable" for row in result["columns"])


def test_comparing_a_selection_that_matches_nothing_is_an_error(fleet_api):
    api, _folder = fleet_api

    result = api.start_scope_comparison({"values": ["sys_key=9"]}, {"values": ["sys_key=1"]})

    assert "error" in result


def test_a_folder_with_no_structure_cannot_compare_scopes(tmp_path):
    api = Api(settings_dir=tmp_path / "settings")
    _open(api, build_flat(tmp_path / "flat"))

    result = api.start_scope_comparison({"values": []}, {"values": []})

    assert "error" in result


# --- the five shapes ------------------------------------------------------


FLAT_SHAPES = {"flat": build_flat, "nested": build_nested}


@pytest.mark.parametrize("builder", list(FLAT_SHAPES), ids=list(FLAT_SHAPES))
def test_a_shape_with_no_roles_still_gets_a_catalogue_of_files(builder, tmp_path):
    """No structure means the unit is the FILE — exactly as it always
    was — so the catalogue is one row per file rather than absent."""
    api = Api(settings_dir=tmp_path / "settings")
    _open(api, FLAT_SHAPES[builder](tmp_path / builder))
    api.start_chart_build()
    api.wait_for_chart_build(timeout=300)

    catalog = api.get_catalog()

    assert catalog["available"] is True
    assert catalog["has_groups"] is False
    assert catalog["unit_label"] == "File"
    assert catalog["groups"] == []
    assert all(row["group_values"] == [] for row in catalog["rows"])


@pytest.mark.parametrize("builder", list(FLAT_SHAPES), ids=list(FLAT_SHAPES))
def test_a_shape_with_no_grouping_skips_the_comparison_rather_than_failing(
    builder, tmp_path
):
    api = Api(settings_dir=tmp_path / "settings")
    _open(api, FLAT_SHAPES[builder](tmp_path / builder))
    api.start_chart_build()
    api.wait_for_chart_build(timeout=300)

    assert api.get_fleet()["available"] is False
    statuses = {c["id"]: c["status"] for c in api.run_instant_checks()["checks"]}
    for check in ("fleet_channel_divergence", "fleet_group_duration", "unit_outlier"):
        assert statuses[check] == "skipped", check


def test_a_single_file_gets_a_one_row_catalogue_and_no_comparison(tmp_path):
    path = tmp_path / "one.csv"
    pl.DataFrame({"ts": [1.0, 2.0, 3.0], "temp": [1.0, 2.0, 3.0]}).write_csv(path)
    api = Api(settings_dir=tmp_path / "settings")
    api.load_file(str(path))
    api.get_chart_data()

    catalog = api.get_catalog()

    assert [row["unit_id"] for row in catalog["rows"]] == ["one.csv"]
    assert catalog["rows"][0]["rows"] == 3
    assert api.get_fleet()["available"] is False


def test_a_hive_folder_compares_its_partitions(tmp_path):
    """A hive key is a grouping column by definition, and a hive folder
    has no series level — so the units are files and the comparison still
    runs. Nothing about either half assumed the other."""
    folder = build_hive(tmp_path / "hive")
    api = Api(settings_dir=tmp_path / "settings")
    _open(api, folder)
    api.start_chart_build()
    api.wait_for_chart_build(timeout=300)

    catalog = api.get_catalog()
    fleet = api.get_fleet()

    assert catalog["group_labels"] == ["day"]
    assert catalog["unit_label"] == "File"
    assert len(catalog["rows"]) == 4
    assert fleet["available"] is True
    assert fleet["group_column"] == "day"
    assert {group["value"] for group in fleet["groups"]} == {
        "day=2026-03-01",
        "day=2026-03-02",
    }


def test_a_full_tree_gets_both(tmp_path):
    folder = build_full(tmp_path / "full")
    api = Api(settings_dir=tmp_path / "settings")
    _open(api, folder)
    api.start_chart_build()
    api.wait_for_chart_build(timeout=300)

    assert api.get_catalog()["available"] is True
    assert api.get_fleet()["available"] is True


def test_a_grouping_that_varies_within_a_file_says_why_it_cannot_be_used(fleet_api):
    """An ordinary data column is a legitimate grouping for the checks
    that read a chart bar, but a per-file histogram cannot be split by it
    without re-reading every file — so the section says that outright."""
    api, _folder = fleet_api
    api.set_grouping_column("temp")
    api.start_chart_build()
    api.wait_for_chart_build(timeout=600)

    fleet = api.get_fleet()

    assert fleet["available"] is False
    assert "constant within a file" in fleet["reason"]


# --- a flat input is untouched --------------------------------------------


@pytest.mark.parametrize("builder", list(FLAT_SHAPES), ids=list(FLAT_SHAPES))
def test_a_flat_inputs_charts_are_what_they_were(builder, tmp_path):
    """The reference payload, key for key. The catalogue is additive: it
    must not have moved a single existing number."""
    from ml_data_view.chart_data import build_chart_payload
    from ml_data_view.folder_loader import discover_files, load_files

    api = Api(settings_dir=tmp_path / "settings")
    folder = FLAT_SHAPES[builder](tmp_path / builder)
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
        "embedding_columns",
        "image_columns",
    ):
        assert produced[key] == reference[key], key


@pytest.mark.parametrize("builder", list(FLAT_SHAPES), ids=list(FLAT_SHAPES))
def test_the_merged_and_direct_catalogues_agree_on_a_flat_folder(builder, tmp_path):
    """The cached path assembles the catalogue from cached row counts; the
    direct path reads them from footers. Same folder, same answer."""
    from ml_data_view.chart_data import build_chart_payload
    from ml_data_view.folder_loader import discover_files, load_files

    api = Api(settings_dir=tmp_path / "settings")
    folder = FLAT_SHAPES[builder](tmp_path / builder)
    _open(api, folder)
    api.start_chart_build()
    produced = api.wait_for_chart_build(timeout=300)["result"]

    files = discover_files(folder)
    reference = build_chart_payload(
        load_files(files),
        files=files,
        # The same time column the app detected, so both sides run the
        # same time-series pass — otherwise one is comparing a catalogue
        # that has a duration column against one that does not.
        timestamp_column="ts",
        timeseries_files=files,
    )

    assert produced["catalog"]["rows"] == reference["catalog"]["rows"]
    assert produced["catalog"]["columns"] == reference["catalog"]["columns"]


def test_a_filtered_build_has_no_catalogue_and_says_why(tmp_path):
    """Every row count in a catalogue would describe the rows that
    survived the filter while every figure beside it describes the file."""
    api = Api(settings_dir=tmp_path / "settings")
    _open(api, build_flat(tmp_path / "flat"))
    api.run_search([{"column": "value", "operator": ">", "values": ["1"]}])
    api.apply_chart_filter()
    api.start_chart_build()
    api.wait_for_chart_build(timeout=300)

    catalog = api.get_catalog()

    assert catalog["available"] is False
    assert "filtered" in catalog["reason"]


# --- changing the grouping without rebuilding anything --------------------


def _nested_group_tree(root: Path) -> Path:
    """Two group levels — site, then rig — so there is a second grouping
    to switch to that is not the one the build used."""
    for site in ("site=north", "site=south"):
        for rig in ("rig=a", "rig=b"):
            _recording(root / site / rig / "rec_0", float(BASE_TS), temp=20.0, chunk_rows=PERIOD)
            _recording(
                root / site / rig / "rec_1", float(BASE_TS) + 5_000, temp=20.0, chunk_rows=PERIOD
            )
    return root


def test_a_different_grouping_level_is_instant_and_reads_nothing(tmp_path):
    folder = _nested_group_tree(tmp_path / "sites")
    api = Api(settings_dir=tmp_path / "settings")
    discovery = api.discover_folder(str(folder))
    _confirm(api, discovery)
    api.start_scope_load()
    api.wait_for_folder_load(timeout=300)
    api.start_chart_build()
    api.wait_for_chart_build(timeout=600)

    assert api.get_fleet()["group_column"] == "site"

    api.set_grouping_column("rig")
    regrouped = api.get_fleet()

    assert regrouped["available"] is True
    assert regrouped["group_column"] == "rig"
    assert {group["value"] for group in regrouped["groups"]} == {"rig=a", "rig=b"}
    # And the bins did not move: the edges are read back out of the cache,
    # where every histogram is stored with the spec that produced it.
    before = next(c for c in api._chart_data["fleet"]["channels"] if c["column"] == "temp")
    after = next(c for c in regrouped["channels"] if c["column"] == "temp")
    assert before["edges"] == after["edges"]


def test_the_findings_follow_the_grouping_too(tmp_path):
    """The Fleet tab and the findings beside it must never describe two
    different groupings of the same folder."""
    folder = _nested_group_tree(tmp_path / "sites")
    # One rig's channel shifted, which is invisible when grouped by site
    # (each site holds one of each rig) and obvious when grouped by rig.
    for site in ("site=north", "site=south"):
        for recording in ("rec_0", "rec_1"):
            for part in ("part-1.parquet", "part-2.parquet"):
                (folder / site / "rig=b" / recording / part).unlink()
            _recording(
                folder / site / "rig=b" / recording,
                float(BASE_TS) + (5_000 if recording == "rec_1" else 0),
                temp=40.0,
                chunk_rows=PERIOD,
            )

    api = Api(settings_dir=tmp_path / "settings")
    discovery = api.discover_folder(str(folder))
    _confirm(api, discovery)
    api.start_scope_load()
    api.wait_for_folder_load(timeout=300)
    api.start_chart_build()
    api.wait_for_chart_build(timeout=600)

    by_site = [f for f in _findings(api) if f["check_id"] == "fleet_channel_divergence"]
    api.set_grouping_column("rig")
    by_rig = [f for f in _findings(api) if f["check_id"] == "fleet_channel_divergence"]

    # Grouped by site the two rigs are mixed together and every site looks
    # the same; grouped by rig, one of them is the odd one out.
    assert by_site == []
    assert any("rig=b" in f["title"] and "temp" in f["title"] for f in by_rig)
    assert api.get_fleet()["group_column"] == "rig"
