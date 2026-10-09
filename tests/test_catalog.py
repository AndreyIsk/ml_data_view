"""The unit catalogue, over hand-built dicts.

Every function under test is pure (bar one injectable stat()), so these
describe folders that never existed on disk. The Api-level tests in
tests/test_fleet_api.py cover the five input shapes end to end.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from ml_data_view.catalog import (
    attach_health,
    build_catalog,
    catalog_csv,
    collect_units,
    group_summaries,
    unit_label_key,
)

NUMERIC = frozenset({"Float64", "Int64"})


def _paths(*names):
    return [Path(name) for name in names]


def _key(name: str) -> str:
    """Paths are compared as the platform writes them, not as this file
    types them — str(Path(...)) is what every caller keys the cache on."""
    return str(Path(name))


def _units():
    return collect_units(
        [
            (("123", "rec_a"), _paths("/f/a1.parquet", "/f/a2.parquet")),
            (("123", "rec_b"), _paths("/f/b1.parquet")),
            (("456", "rec_c"), _paths("/f/c1.parquet")),
        ],
        row_counts={
            _key("/f/a1.parquet"): 10,
            _key("/f/a2.parquet"): 15,
            _key("/f/b1.parquet"): 20,
            _key("/f/c1.parquet"): 5,
        },
        schemas={
            _key("/f/a1.parquet"): {"ts": "Int64", "temp": "Float64", "name": "String"},
            _key("/f/b1.parquet"): {"ts": "Int64", "temp": "Float64", "name": "String"},
            _key("/f/c1.parquet"): {"ts": "Int64", "temp": "Float64", "name": "String"},
        },
        numeric_dtypes=NUMERIC,
        size_of=lambda path: 1000,
    )


def _timeseries():
    return {
        "timestamp_column": "ts",
        "units": {
            "123 / rec_a": {
                "rows": 25,
                "first_ts": 100.0,
                "last_ts": 200.0,
                "duration": 100.0,
                "hz": 10.0,
                "gaps": [{"at": 150.0, "duration": 5.0, "at_boundary": True}],
                "gap_share": 0.05,
                "overlaps": [],
                "backwards": 0,
                "duplicates": 0,
                "channels": {"temp": {}},
            },
            "123 / rec_b": {
                "rows": 20,
                "first_ts": 300.0,
                "last_ts": 340.0,
                "duration": 40.0,
                "hz": 10.0,
                "gaps": [],
                "gap_share": None,
                "overlaps": [],
                "backwards": 0,
                "duplicates": 0,
                "channels": {"temp": {}},
            },
            "456 / rec_c": {
                "rows": 5,
                "first_ts": 500.0,
                "last_ts": 510.0,
                "duration": 10.0,
                "hz": 2.0,
                "gaps": [],
                "gap_share": None,
                "overlaps": [],
                "backwards": 0,
                "duplicates": 0,
                "channels": {"temp": {}},
            },
        },
    }


LABELS = {
    "group_labels": ["System"],
    "unit_label": "Recording",
    "group_columns": ["sys_key"],
    "unit_column": "recording_id",
    "root_label": "Fleet",
}


# --- collecting units -----------------------------------------------------


def test_a_units_rows_are_the_sum_of_its_chunks():
    units = _units()

    assert units[0]["rows"] == 25
    assert units[0]["chunk_count"] == 2


def test_a_unit_with_an_uncached_chunk_reports_unknown_rows_not_zero():
    units = collect_units(
        [(("u",), _paths("/f/known.parquet", "/f/unknown.parquet"))],
        row_counts={_key("/f/known.parquet"): 10},
        size_of=lambda path: 1,
    )

    assert units[0]["rows"] is None


def test_channel_count_counts_numeric_columns_only():
    assert _units()[0]["channel_count"] == 2  # ts and temp, not name


def test_a_file_that_cannot_be_stat_ed_makes_the_size_unknown():
    units = collect_units(
        [(("u",), _paths("/f/a.parquet", "/f/b.parquet"))],
        row_counts={_key("/f/a.parquet"): 1, _key("/f/b.parquet"): 1},
        size_of=lambda path: None if path.name == "b.parquet" else 10,
    )

    assert units[0]["bytes"] is None


def test_a_unit_with_no_group_levels_has_no_group_values():
    units = collect_units(
        [(("part0.parquet",), _paths("/f/part0.parquet"))],
        row_counts={_key("/f/part0.parquet"): 3},
        size_of=lambda path: 1,
    )

    assert units[0]["group_values"] == []
    assert units[0]["unit_id"] == "part0.parquet"


# --- the table ------------------------------------------------------------


def test_the_catalogue_has_one_row_per_unit():
    catalog = build_catalog(_units(), timeseries=_timeseries(), labels=LABELS)

    assert [row["label"] for row in catalog["rows"]] == [
        "123 / rec_a",
        "123 / rec_b",
        "456 / rec_c",
    ]


def test_time_figures_come_from_the_merged_time_series_artifact():
    catalog = build_catalog(_units(), timeseries=_timeseries(), labels=LABELS)

    row = catalog["rows"][0]
    assert row["start"] == 100.0
    assert row["end"] == 200.0
    assert row["duration"] == 100.0
    assert row["hz"] == 10.0
    assert row["gap_count"] == 1
    assert row["gap_share"] == 0.05


def test_a_gap_on_a_chunk_boundary_is_counted_as_a_missing_chunk_as_well():
    catalog = build_catalog(_units(), timeseries=_timeseries(), labels=LABELS)

    assert catalog["rows"][0]["issues"]["missing_chunks"] == 1
    assert catalog["rows"][0]["issues"]["gaps"] == 0


def test_without_a_timestamp_column_the_time_columns_are_absent_not_empty():
    catalog = build_catalog(_units(), timeseries=None, labels=LABELS)

    keys = [column["key"] for column in catalog["columns"]]
    assert "duration" not in keys
    assert "rows" in keys
    assert catalog["has_time"] is False
    assert catalog["rows"][0]["rows"] == 25


def test_the_unit_column_is_labelled_by_the_mapping():
    catalog = build_catalog(_units(), timeseries=_timeseries(), labels=LABELS)

    assert catalog["columns"][0]["label"] == "Recording"


def test_the_default_labels_are_generic():
    catalog = build_catalog(_units(), timeseries=None, labels=None)

    assert catalog["columns"][0]["label"] == "File"
    assert catalog["root_label"] == "All"
    assert catalog["group_labels"] == []


def test_totals_are_unknown_when_any_row_is_unknown():
    units = _units()
    units[1]["rows"] = None

    catalog = build_catalog(units, timeseries=_timeseries(), labels=LABELS)

    assert catalog["totals"]["rows"] is None
    assert catalog["totals"]["chunk_count"] == 4


# --- group summaries ------------------------------------------------------


def test_group_summaries_roll_up_by_the_outermost_group_value():
    catalog = build_catalog(_units(), timeseries=_timeseries(), labels=LABELS)

    groups = {group["value"]: group for group in catalog["groups"]}
    assert groups["123"]["unit_count"] == 2
    assert groups["123"]["rows"] == 45
    assert groups["123"]["duration"] == 140.0
    assert groups["456"]["unit_count"] == 1


def test_a_shape_with_no_group_levels_has_no_group_summaries():
    catalog = build_catalog(_units(), timeseries=_timeseries(), labels={"unit_label": "File"})

    assert catalog["groups"] == []
    assert catalog["has_groups"] is False


# --- health ---------------------------------------------------------------


def test_health_is_unknown_until_the_checks_have_run():
    catalog = build_catalog(_units(), timeseries=_timeseries(), labels=LABELS)

    assert catalog["rows"][0]["health"] == {"status": "unknown", "count": 0}


def test_a_unit_with_no_findings_is_healthy_once_the_checks_have_run():
    catalog = attach_health(
        build_catalog(_units(), timeseries=_timeseries(), labels=LABELS), []
    )

    assert catalog["rows"][0]["health"] == {"status": "ok", "count": 0}


def test_a_units_health_is_the_worst_severity_among_its_findings():
    findings = [
        {"unit": "123 / rec_a", "severity": "warning", "check_id": "time_gaps"},
        {"unit": "123 / rec_a", "severity": "critical", "check_id": "missing_chunk"},
        {"unit": "456 / rec_c", "severity": "info", "check_id": "sample_rate"},
    ]

    catalog = attach_health(
        build_catalog(_units(), timeseries=_timeseries(), labels=LABELS), findings
    )

    by_label = {row["label"]: row["health"] for row in catalog["rows"]}
    assert by_label["123 / rec_a"] == {"status": "critical", "count": 2}
    assert by_label["123 / rec_b"] == {"status": "ok", "count": 0}
    assert by_label["456 / rec_c"] == {"status": "info", "count": 1}


def test_a_finding_that_names_no_unit_marks_no_unit_unhealthy():
    """A high null rate is about the dataset, not about recording 7."""
    findings = [{"unit": "", "severity": "critical", "check_id": "all_null_column"}]

    catalog = attach_health(
        build_catalog(_units(), timeseries=_timeseries(), labels=LABELS), findings
    )

    assert {row["health"]["status"] for row in catalog["rows"]} == {"ok"}


def test_group_health_breakdown_counts_every_status():
    findings = [{"unit": "123 / rec_a", "severity": "critical", "check_id": "x"}]

    catalog = attach_health(
        build_catalog(_units(), timeseries=_timeseries(), labels=LABELS), findings
    )

    groups = {group["value"]: group["health"] for group in catalog["groups"]}
    assert groups["123"]["critical"] == 1
    assert groups["123"]["ok"] == 1
    assert groups["456"]["ok"] == 1


def test_health_can_be_left_unknown_explicitly():
    catalog = attach_health(
        build_catalog(_units(), timeseries=_timeseries(), labels=LABELS), [], scanned=False
    )

    assert catalog["rows"][0]["health"]["status"] == "unknown"


# --- CSV ------------------------------------------------------------------


def test_the_csv_header_puts_the_group_columns_first():
    catalog = build_catalog(_units(), timeseries=_timeseries(), labels=LABELS)

    header = catalog_csv(catalog).splitlines()[0]

    assert header.startswith("System,Recording,Chunks,Rows,Start,End,Duration")


def test_the_csv_exports_raw_numbers_a_spreadsheet_can_sum():
    catalog = attach_health(
        build_catalog(_units(), timeseries=_timeseries(), labels=LABELS), []
    )

    line = catalog_csv(catalog).splitlines()[1]

    assert line.startswith("123,rec_a,2,25,100.0,200.0,100.0,10.0,1,0.05,")
    assert line.endswith(",ok")


def test_the_csv_drops_the_time_columns_when_there_is_no_time():
    catalog = build_catalog(_units(), timeseries=None, labels=LABELS)

    header = catalog_csv(catalog).splitlines()[0]

    assert "Duration" not in header
    assert "Rows" in header


def test_an_unknown_figure_is_an_empty_cell_not_a_zero():
    units = _units()
    units[0]["rows"] = None
    catalog = build_catalog(units, timeseries=_timeseries(), labels=LABELS)

    assert catalog_csv(catalog).splitlines()[1].startswith("123,rec_a,2,,")


# --- the key every module joins on ----------------------------------------


@pytest.mark.parametrize(
    "key,expected",
    [
        (("123", "rec_a"), "123 / rec_a"),
        (("part0.parquet",), "part0.parquet"),
        (("a", "b", "c"), "a / b / c"),
    ],
)
def test_the_unit_key_is_the_group_values_then_the_unit_id(key, expected):
    assert unit_label_key(key) == expected


def test_group_summaries_of_nothing_are_nothing():
    assert group_summaries([], ["System"]) == []
