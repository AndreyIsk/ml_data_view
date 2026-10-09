"""What the capped worker actually runs (ml_data_view/job_tasks.py).

The rule these tests exist to hold is architectural, not cosmetic: a job is
described by PATHS, never by a serialized frame. If a chart payload or a
materialized DataFrame ever crossed the spec, it would have to exist in the
UI process first — the one process with no memory ceiling on it, and the
one this whole module exists to keep small.
"""
from __future__ import annotations

import json

import polars as pl
import pytest

from ml_data_view import job_tasks
from ml_data_view.api import Api


def _folder(tmp_path, file_count=3):
    folder = tmp_path / "parts"
    folder.mkdir()
    for index in range(file_count):
        pl.DataFrame(
            {"v": list(range(20)), "label": [f"c{n % 2}" for n in range(20)]}
        ).write_parquet(folder / f"part{index}.parquet")
    return folder


def test_every_registered_task_builds_something_runnable():
    """A task name in the registry that cannot be built is a job that
    starts a subprocess and immediately fails — with the failure only
    visible in a temp file."""
    for name in job_tasks.TASKS:
        assert callable(job_tasks.TASKS[name]), name


def test_an_unknown_task_name_is_rejected_by_name():
    with pytest.raises(KeyError, match="nope"):
        job_tasks.build_task("nope", {})


def test_a_single_file_dataset_travels_as_its_path(tmp_path):
    parquet_path = tmp_path / "data.parquet"
    pl.DataFrame({"v": [1, 2, 3]}).write_parquet(parquet_path)
    api = Api(settings_dir=tmp_path)
    api.load_file(str(parquet_path))

    params = api._dataset_params()

    assert params["path"] == str(parquet_path)
    assert "files" not in params


def test_a_folder_dataset_travels_as_its_file_list(tmp_path):
    folder = _folder(tmp_path)
    api = Api(settings_dir=tmp_path)
    api.discover_folder(str(folder))
    api.load_folder_group(0)

    params = api._dataset_params()

    assert len(params["files"]) == 3
    assert all(path.endswith(".parquet") for path in params["files"])
    assert "path" not in params


def test_a_job_spec_is_json_and_carries_no_frame(tmp_path):
    """The spec is written to disk as JSON, so a LazyFrame could not travel
    even by accident — but a caller could still stuff a materialized
    payload in, which is what this guards."""
    folder = _folder(tmp_path)
    api = Api(settings_dir=tmp_path)
    api.discover_folder(str(folder))
    api.load_folder_group(0)
    api.start_chart_build()
    job = api._chart_build_job

    spec = json.loads((job.directory / "spec.json").read_text(encoding="utf-8"))

    assert spec["task"] == "chart_build"
    assert set(spec["params"]) <= {
        "files",
        "path",
        "dataset_dir",
        "total_rows",
        "null_counts",
        "conditions",
        # Where per-file profiles are remembered — a path, like everything
        # else here, never the profiles themselves.
        "cache_path",
        # Per-chunk derived-column values for a structured folder
        # (hierarchy.py): strings and an integer per file, never a frame.
        "identity",
        # Time-series health: the column that makes those checks apply at
        # all, the thresholds they are judged against, and the chunks to
        # read. Scalars and paths, like everything else here.
        "timestamp_column",
        "timeseries_thresholds",
        "timeseries_max_channels",
        "timeseries_files",
        # The catalogue and the group comparison: the mapping's own labels,
        # the column to group by, and two thresholds. Strings and numbers,
        # and no more of a frame than the rest.
        "structure_labels",
        "grouping_column",
        "fleet_thresholds",
        "fleet_max_channels",
    }
    api.wait_for_chart_build(timeout=180)


def test_a_worker_rebuilds_a_folder_dataset_from_paths_alone(tmp_path):
    """Lazily — pl.concat of scans reads nothing — which is what makes
    rebuilding in the child free rather than a second load."""
    folder = _folder(tmp_path)
    files = sorted(str(path) for path in folder.glob("*.parquet"))

    lf, rebuilt = job_tasks._frame_from({"files": files})

    assert [str(path) for path in rebuilt] == files
    assert isinstance(lf, pl.LazyFrame)
    assert lf.select(pl.len()).collect(engine="streaming")[0, 0] == 60


def test_a_chart_filter_travels_as_conditions_not_as_an_expression(tmp_path):
    """A polars Expr is not JSON. api.py already keeps the conditions
    beside the predicate, so the worker rebuilds the same filter rather
    than receiving a second, hand-written one."""
    parquet_path = tmp_path / "data.parquet"
    pl.DataFrame({"v": list(range(10))}).write_parquet(parquet_path)
    api = Api(settings_dir=tmp_path)
    api.load_file(str(parquet_path))
    api.run_search([{"column": "v", "operator": ">=", "values": ["5"]}])
    api.apply_chart_filter()
    api.start_chart_build()

    spec = json.loads((api._chart_build_job.directory / "spec.json").read_text(encoding="utf-8"))
    state = api.wait_for_chart_build(timeout=180)

    assert spec["params"]["conditions"] == [
        {"column": "v", "operator": ">=", "values": ["5"]}
    ]
    # The rebuilt predicate has to actually be APPLIED, or the charts would
    # describe a different row set than the count beside them. Ten rows
    # went in; only five match.
    histogram = next(h for h in state["result"]["numeric_histograms"] if h["column"] == "v")
    assert sum(histogram["counts"]) == 5


def test_a_filterless_build_sends_no_conditions(tmp_path):
    parquet_path = tmp_path / "data.parquet"
    pl.DataFrame({"v": list(range(10))}).write_parquet(parquet_path)
    api = Api(settings_dir=tmp_path)
    api.load_file(str(parquet_path))
    api.start_chart_build()

    spec = json.loads((api._chart_build_job.directory / "spec.json").read_text(encoding="utf-8"))
    api.wait_for_chart_build(timeout=180)

    assert spec["params"]["conditions"] is None
