import math

import polars as pl
import pytest

from ml_data_view.compare import (
    MODERATE_PSI,
    SIGNIFICANT_PSI,
    compare_datasets,
    compare_schemas,
    evenly_spaced,
    joint_bin_edges,
    psi,
    severity,
)


# --- PSI ---------------------------------------------------------------


def test_identical_distributions_have_zero_psi():
    assert psi([0.25, 0.25, 0.5], [0.25, 0.25, 0.5]) == pytest.approx(0.0, abs=1e-9)


def test_psi_matches_a_hand_computed_value():
    a, b = [0.6, 0.4], [0.5, 0.5]
    expected = (0.6 - 0.5) * math.log(0.6 / 0.5) + (0.4 - 0.5) * math.log(0.4 / 0.5)
    assert psi(a, b) == pytest.approx(expected, rel=1e-9)


def test_psi_is_symmetric():
    assert psi([0.7, 0.3], [0.4, 0.6]) == pytest.approx(psi([0.4, 0.6], [0.7, 0.3]), rel=1e-9)


def test_a_bin_empty_on_one_side_is_large_but_finite():
    """The interesting case: a bin populated on one side and empty on the
    other must not produce inf or nan, or the whole row becomes unreadable."""
    value = psi([0.5, 0.5], [1.0, 0.0])
    assert math.isfinite(value)
    assert value > SIGNIFICANT_PSI


def test_psi_of_mismatched_lengths_is_rejected():
    with pytest.raises(ValueError):
        psi([0.5, 0.5], [1.0])


def test_severity_boundaries():
    assert severity(0.05) == "stable"
    assert severity(MODERATE_PSI) == "moderate"
    assert severity(0.2) == "moderate"
    assert severity(SIGNIFICANT_PSI + 0.001) == "significant"


# --- sampling ----------------------------------------------------------


def test_evenly_spaced_spans_the_whole_list_not_the_head():
    """Folders here are day-partitioned, so taking the first N files would
    compare early-in-the-day against early-in-the-day and report drift that
    is really time of day."""
    files = list(range(100))

    picked = evenly_spaced(files, 5)

    assert picked[0] == 0
    assert picked[-1] == 99
    assert picked != files[:5]
    assert len(picked) == 5


def test_evenly_spaced_returns_everything_when_fewer_than_requested():
    assert evenly_spaced([1, 2, 3], 10) == [1, 2, 3]


def test_evenly_spaced_handles_a_single_item_and_one_sample():
    assert evenly_spaced([7], 5) == [7]
    assert evenly_spaced([1, 2, 3], 1) == [1]


# --- schema diff -------------------------------------------------------


def test_schema_diff_reports_added_removed_and_retyped():
    a = pl.DataFrame({"keep": [1], "gone": ["x"], "retyped": [1]}).lazy().collect_schema()
    b = pl.DataFrame({"keep": [1], "added": [2.0], "retyped": ["now a string"]}).lazy().collect_schema()

    diff = compare_schemas(a, b)

    assert diff["only_in_a"] == ["gone"]
    assert diff["only_in_b"] == ["added"]
    assert [c["column"] for c in diff["type_changed"]] == ["retyped"]
    assert diff["shared"] == ["keep"]


def test_identical_schemas_report_no_differences():
    schema = pl.DataFrame({"a": [1], "b": ["x"]}).lazy().collect_schema()
    diff = compare_schemas(schema, schema)
    assert diff["only_in_a"] == [] and diff["only_in_b"] == [] and diff["type_changed"] == []


# --- joint bins --------------------------------------------------------


def _write(tmp_path, name, frame, count=2):
    files = []
    for i in range(count):
        path = tmp_path / f"{name}{i}.parquet"
        frame.write_parquet(path)
        files.append(path)
    return files


def test_joint_bin_edges_span_both_sides(tmp_path):
    a = _write(tmp_path, "a", pl.DataFrame({"v": [0.0, 10.0]}))
    b = _write(tmp_path, "b", pl.DataFrame({"v": [100.0, 200.0]}))

    edges = joint_bin_edges(a, b, "v", bins=4)

    assert edges[0] == 0.0
    assert edges[-1] == 200.0
    assert len(edges) == 5


def test_joint_bin_edges_when_one_range_sits_inside_the_other(tmp_path):
    a = _write(tmp_path, "a", pl.DataFrame({"v": [0.0, 100.0]}))
    b = _write(tmp_path, "b", pl.DataFrame({"v": [40.0, 60.0]}))

    edges = joint_bin_edges(a, b, "v", bins=4)

    assert edges[0] == 0.0 and edges[-1] == 100.0


def test_joint_bin_edges_none_when_no_usable_values(tmp_path):
    empty = pl.DataFrame({"v": pl.Series("v", [None, None], dtype=pl.Float64)})
    a = _write(tmp_path, "a", empty)
    b = _write(tmp_path, "b", empty)

    assert joint_bin_edges(a, b, "v", bins=4) is None


# --- compare_datasets --------------------------------------------------


def _numeric_frame(offset=0.0, rows=400):
    return pl.DataFrame(
        {
            "value": [float(n % 50) + offset for n in range(rows)],
            "kind": [f"k{n % 3}" for n in range(rows)],
            "maybe": [None if n % 10 == 0 else 1.0 for n in range(rows)],
        }
    )


def test_identical_datasets_are_stable_everywhere(tmp_path):
    a = _write(tmp_path, "a", _numeric_frame())
    b = _write(tmp_path, "b", _numeric_frame())

    result = compare_datasets(a, b)

    assert all(row["severity"] == "stable" for row in result["columns"]), result["columns"]
    assert result["row_count_a"] == result["row_count_b"]


def test_a_shifted_numeric_column_is_flagged_significant(tmp_path):
    a = _write(tmp_path, "a", _numeric_frame(offset=0.0))
    b = _write(tmp_path, "b", _numeric_frame(offset=500.0))

    result = compare_datasets(a, b)
    value_row = next(r for r in result["columns"] if r["column"] == "value")

    assert value_row["severity"] == "significant"
    assert value_row["psi"] > SIGNIFICANT_PSI


def test_a_new_category_is_listed(tmp_path):
    a = _write(tmp_path, "a", pl.DataFrame({"kind": ["x"] * 100 + ["y"] * 100}))
    b = _write(tmp_path, "b", pl.DataFrame({"kind": ["x"] * 100 + ["z"] * 100}))

    result = compare_datasets(a, b)
    row = next(r for r in result["columns"] if r["column"] == "kind")

    assert row["new_categories"] == ["z"]
    assert row["missing_categories"] == ["y"]


def test_null_rate_delta_is_reported(tmp_path):
    a = _write(tmp_path, "a", pl.DataFrame({"v": [1.0, None, 1.0, 1.0]}))
    b = _write(tmp_path, "b", pl.DataFrame({"v": [None, None, None, 1.0]}))

    result = compare_datasets(a, b)
    row = next(r for r in result["columns"] if r["column"] == "v")

    assert row["null_rate_a"] == pytest.approx(0.25)
    assert row["null_rate_b"] == pytest.approx(0.75)


def test_columns_on_only_one_side_are_not_given_a_psi(tmp_path):
    a = _write(tmp_path, "a", pl.DataFrame({"shared": [1.0], "only_a": [2.0]}))
    b = _write(tmp_path, "b", pl.DataFrame({"shared": [1.0], "only_b": [3.0]}))

    result = compare_datasets(a, b)

    assert [r["column"] for r in result["columns"]] == ["shared"]
    assert result["schema"]["only_in_a"] == ["only_a"]
    assert result["schema"]["only_in_b"] == ["only_b"]


def test_a_retyped_column_is_skipped_for_distribution_comparison(tmp_path):
    a = _write(tmp_path, "a", pl.DataFrame({"v": [1.0, 2.0]}))
    b = _write(tmp_path, "b", pl.DataFrame({"v": ["1", "2"]}))

    result = compare_datasets(a, b)

    assert result["columns"] == []
    assert [c["column"] for c in result["schema"]["type_changed"]] == ["v"]


def test_results_are_sorted_with_the_biggest_change_first(tmp_path):
    a = _write(tmp_path, "a", pl.DataFrame({"same": [1.0] * 200, "moved": [float(n % 20) for n in range(200)]}))
    b = _write(tmp_path, "b", pl.DataFrame({"same": [1.0] * 200, "moved": [float(n % 20) + 900 for n in range(200)]}))

    result = compare_datasets(a, b)

    assert result["columns"][0]["column"] == "moved"


def test_progress_and_cancellation_keep_completed_columns(tmp_path):
    a = _write(tmp_path, "a", _numeric_frame())
    b = _write(tmp_path, "b", _numeric_frame())
    seen = []

    result = compare_datasets(
        a,
        b,
        on_column=lambda i, name, row: seen.append(name),
        should_cancel=lambda: len(seen) >= 1,
    )

    assert len(seen) == 1
    assert len(result["columns"]) == 1  # the completed one is kept
    assert result["cancelled"] is True
