"""Instant dataset checks (findings.py).

Every check here is a pure function over two dicts the app has already
computed — `dataset_info` (schema + null counts) and `chart_data`
(build_chart_payload's output). So these tests build those dicts by hand:
no fixture files, no polars, no I/O. That is the point of the design, and
it means a check can be pinned to its exact trigger boundary cheaply.
"""
from __future__ import annotations

import pytest

from ml_data_view.findings import (
    CLASS_IMBALANCE,
    HIGH_MISSINGNESS,
    INSTANT_CHECKS,
    Finding,
    run_instant_checks,
    severity_summary,
    sorted_findings,
)


def _dataset_info(row_count: int, columns: list[tuple[str, str, int]]) -> dict:
    """(name, dtype, null_count) triples -> the dict api.py caches."""
    return {
        "row_count": row_count,
        "columns": [
            {"name": name, "dtype": dtype, "null_count": nulls}
            for name, dtype, nulls in columns
        ],
    }


def _chart_data(**overrides) -> dict:
    payload = {
        "correlated_missingness": [],
        "value_frequency_histograms": [],
        "null_heatmap": {},
        "numeric_histograms": [],
        "categorical_bars": [],
        "text_length_histograms": [],
        "embedding_columns": [],
        "embedding_projections": [],
        "image_columns": [],
        "image_previews": [],
    }
    payload.update(overrides)
    return payload


def _run(dataset_info, chart_data) -> list[Finding]:
    """Every finding produced, flattened — most checks are asserted on the
    findings themselves rather than the per-check envelope."""
    return sorted_findings(run_instant_checks(dataset_info, chart_data))


def _ids(findings: list[Finding]) -> list[str]:
    return [f.check_id for f in findings]


# --- empty_dataset -----------------------------------------------------


def test_empty_dataset_fires_on_zero_rows():
    findings = _run(_dataset_info(0, [("id", "Int64", 0)]), _chart_data())

    assert "empty_dataset" in _ids(findings)
    assert next(f for f in findings if f.check_id == "empty_dataset").severity == "critical"


def test_empty_dataset_silent_when_rows_exist():
    findings = _run(_dataset_info(10, [("id", "Int64", 0)]), _chart_data())

    assert "empty_dataset" not in _ids(findings)


# --- all_null_column ---------------------------------------------------


def test_all_null_column_fires_when_every_row_is_missing():
    findings = _run(_dataset_info(100, [("dead", "String", 100)]), _chart_data())

    finding = next(f for f in findings if f.check_id == "all_null_column")
    assert finding.severity == "critical"
    assert finding.columns == ["dead"]


def test_all_null_column_silent_on_an_empty_dataset():
    """0 nulls out of 0 rows is not an all-null column — the dataset being
    empty is the finding, and empty_dataset already makes it."""
    findings = _run(_dataset_info(0, [("id", "Int64", 0)]), _chart_data())

    assert "all_null_column" not in _ids(findings)


def test_all_null_column_reports_each_column_separately():
    findings = _run(
        _dataset_info(50, [("a", "String", 50), ("b", "String", 50)]), _chart_data()
    )

    all_null = [f for f in findings if f.check_id == "all_null_column"]
    assert [f.columns for f in all_null] == [["a"], ["b"]]


# --- high_missingness --------------------------------------------------


def test_high_missingness_fires_above_the_threshold():
    findings = _run(_dataset_info(100, [("sparse", "String", 51)]), _chart_data())

    finding = next(f for f in findings if f.check_id == "high_missingness")
    assert finding.severity == "warning"
    assert finding.columns == ["sparse"]


def test_high_missingness_silent_exactly_at_the_threshold():
    """The rule is strictly greater than HIGH_MISSINGNESS, so exactly half
    missing does not fire."""
    row_count = 100
    exactly_half = int(row_count * HIGH_MISSINGNESS)

    findings = _run(_dataset_info(row_count, [("half", "String", exactly_half)]), _chart_data())

    assert "high_missingness" not in _ids(findings)


def test_high_missingness_skips_a_column_already_reported_as_all_null():
    """Precedence: an all-null column is trivially 100% missing. Reporting
    it twice buries the finding that matters."""
    findings = _run(_dataset_info(100, [("dead", "String", 100)]), _chart_data())

    assert "all_null_column" in _ids(findings)
    assert "high_missingness" not in _ids(findings)


# --- constant_column ---------------------------------------------------


def test_constant_column_fires_for_a_single_category():
    findings = _run(
        _dataset_info(100, [("flag", "String", 0)]),
        _chart_data(categorical_bars=[{"column": "flag", "categories": ["yes"], "counts": [100]}]),
    )

    finding = next(f for f in findings if f.check_id == "constant_column")
    assert finding.severity == "warning"
    assert finding.columns == ["flag"]


def test_constant_column_fires_for_a_numeric_column_with_one_value():
    """numeric_histogram_data emits a single bin with equal edges when
    every value is identical — that is the signal, not a recomputation."""
    findings = _run(
        _dataset_info(100, [("const", "Int64", 0)]),
        _chart_data(
            numeric_histograms=[
                {"column": "const", "bin_edges": [7.0, 7.0], "counts": [100], "x_scale": "linear"}
            ]
        ),
    )

    assert "constant_column" in _ids(findings)


def test_constant_column_silent_with_two_categories():
    findings = _run(
        _dataset_info(100, [("flag", "String", 0)]),
        _chart_data(
            categorical_bars=[{"column": "flag", "categories": ["y", "n"], "counts": [90, 10]}]
        ),
    )

    assert "constant_column" not in _ids(findings)


def test_constant_column_skips_a_column_already_reported_as_all_null():
    findings = _run(
        _dataset_info(100, [("dead", "String", 100)]),
        _chart_data(categorical_bars=[{"column": "dead", "categories": [], "counts": []}]),
    )

    assert "constant_column" not in _ids(findings)


# --- class_imbalance ---------------------------------------------------


def test_class_imbalance_fires_when_one_class_dominates():
    findings = _run(
        _dataset_info(100, [("label", "String", 0)]),
        _chart_data(
            categorical_bars=[
                {"column": "label", "categories": ["a", "b", "c"], "counts": [95, 3, 2]}
            ]
        ),
    )

    finding = next(f for f in findings if f.check_id == "class_imbalance")
    assert finding.severity == "warning"
    assert finding.columns == ["label"]
    assert "95" in finding.detail  # the actual share is stated, not just "imbalanced"


def test_class_imbalance_silent_exactly_at_the_threshold():
    counts = [int(CLASS_IMBALANCE * 100), 100 - int(CLASS_IMBALANCE * 100)]

    findings = _run(
        _dataset_info(100, [("label", "String", 0)]),
        _chart_data(
            categorical_bars=[{"column": "label", "categories": ["a", "b"], "counts": counts}]
        ),
    )

    assert "class_imbalance" not in _ids(findings)


def test_class_imbalance_never_reports_the_synthetic_other_bucket():
    """"Other" is categorical_bar_data's roll-up of everything past the
    top N — it names no actual value, so it cannot be "the dominant class"."""
    findings = _run(
        _dataset_info(100, [("id", "String", 0)]),
        _chart_data(
            categorical_bars=[{"column": "id", "categories": ["x", "Other"], "counts": [3, 97]}]
        ),
    )

    assert "class_imbalance" not in _ids(findings)


def test_class_imbalance_counts_other_in_the_denominator():
    """The share is measured against every counted row including "Other",
    so a top category is not inflated by ignoring the roll-up."""
    findings = _run(
        _dataset_info(100, [("label", "String", 0)]),
        _chart_data(
            categorical_bars=[
                {"column": "label", "categories": ["a", "b", "Other"], "counts": [80, 5, 15]}
            ]
        ),
    )

    assert "class_imbalance" not in _ids(findings)  # 80/100, below the threshold


def test_class_imbalance_silent_when_no_rows_were_counted():
    findings = _run(
        _dataset_info(100, [("label", "String", 100)]),
        _chart_data(categorical_bars=[{"column": "label", "categories": [], "counts": []}]),
    )

    assert "class_imbalance" not in _ids(findings)


# --- broken_image_refs -------------------------------------------------


def test_broken_image_refs_fires_when_a_previewed_thumbnail_is_missing():
    findings = _run(
        _dataset_info(100, [("img", "Binary", 0)]),
        _chart_data(
            image_previews=[
                {
                    "column": "img",
                    "source": "path",
                    "rows": [
                        {"row_index": 0, "thumbnail": "data:image/png;base64,AAA"},
                        {"row_index": 1, "thumbnail": None},
                    ],
                    "total_rows": 100,
                }
            ]
        ),
    )

    finding = next(f for f in findings if f.check_id == "broken_image_refs")
    assert finding.severity == "warning"
    assert finding.columns == ["img"]


def test_broken_image_refs_states_it_cannot_separate_the_causes():
    """image_previews records only {row_index, thumbnail}: a null thumbnail
    means null value OR missing file OR undecodable bytes. The finding must
    not claim to know which."""
    findings = _run(
        _dataset_info(100, [("img", "Binary", 0)]),
        _chart_data(
            image_previews=[
                {
                    "column": "img",
                    "source": "path",
                    "rows": [{"row_index": 0, "thumbnail": None}],
                    "total_rows": 100,
                }
            ]
        ),
    )

    detail = next(f for f in findings if f.check_id == "broken_image_refs").detail
    assert "1 of the first 1" in detail
    assert "missing" in detail and "undecodable" in detail


def test_broken_image_refs_silent_when_every_thumbnail_resolved():
    findings = _run(
        _dataset_info(100, [("img", "Binary", 0)]),
        _chart_data(
            image_previews=[
                {
                    "column": "img",
                    "source": "bytes",
                    "rows": [{"row_index": 0, "thumbnail": "data:image/png;base64,AAA"}],
                    "total_rows": 100,
                }
            ]
        ),
    )

    assert "broken_image_refs" not in _ids(findings)


# --- correlated_missingness --------------------------------------------


def test_correlated_missingness_surfaces_a_reported_group():
    findings = _run(
        _dataset_info(100, [("a", "String", 40), ("b", "String", 40)]),
        _chart_data(
            correlated_missingness=[{"columns": ["a", "b"], "null_count": 40, "overlap": 0.99}]
        ),
    )

    finding = next(f for f in findings if f.check_id == "correlated_missingness")
    assert finding.severity == "info"
    assert finding.columns == ["a", "b"]


def test_correlated_missingness_silent_when_no_group_was_found():
    findings = _run(_dataset_info(100, [("a", "String", 40)]), _chart_data())

    assert "correlated_missingness" not in _ids(findings)


# --- likely_id_column --------------------------------------------------


def test_likely_id_column_fires_when_almost_every_value_occurs_once():
    findings = _run(
        _dataset_info(1000, [("track", "String", 0)]),
        _chart_data(
            value_frequency_histograms=[
                {
                    "column": "track",
                    "buckets": ["1", "2-3", "4-10"],
                    "counts": [950, 40, 10],
                    "per_file": False,
                }
            ]
        ),
    )

    finding = next(f for f in findings if f.check_id == "likely_id_column")
    assert finding.severity == "info"
    assert finding.columns == ["track"]


def test_likely_id_column_silent_when_values_repeat():
    findings = _run(
        _dataset_info(1000, [("cat", "String", 0)]),
        _chart_data(
            value_frequency_histograms=[
                {"column": "cat", "buckets": ["1", "2-3"], "counts": [10, 90], "per_file": False}
            ]
        ),
    )

    assert "likely_id_column" not in _ids(findings)


def test_likely_id_column_flags_the_per_file_caveat_in_its_detail():
    """On a folder the buckets are counted per file, which shifts the
    distribution toward "1". The finding must say so rather than
    overstating uniqueness."""
    findings = _run(
        _dataset_info(1000, [("track", "String", 0)]),
        _chart_data(
            value_frequency_histograms=[
                {"column": "track", "buckets": ["1"], "counts": [1000], "per_file": True}
            ]
        ),
    )

    detail = next(f for f in findings if f.check_id == "likely_id_column").detail
    assert "per file" in detail


# --- skewed_numeric ----------------------------------------------------


def test_skewed_numeric_fires_on_a_log_scaled_histogram():
    findings = _run(
        _dataset_info(100, [("amount", "Float64", 0)]),
        _chart_data(
            numeric_histograms=[
                {"column": "amount", "bin_edges": [0.0, 1.0, 500.0], "counts": [90, 10], "x_scale": "log"}
            ]
        ),
    )

    finding = next(f for f in findings if f.check_id == "skewed_numeric")
    assert finding.severity == "info"


def test_skewed_numeric_silent_on_a_linear_histogram():
    findings = _run(
        _dataset_info(100, [("amount", "Float64", 0)]),
        _chart_data(
            numeric_histograms=[
                {"column": "amount", "bin_edges": [0.0, 1.0], "counts": [100], "x_scale": "linear"}
            ]
        ),
    )

    assert "skewed_numeric" not in _ids(findings)


# --- the run envelope --------------------------------------------------


def test_run_instant_checks_reports_every_check_even_when_it_finds_nothing():
    """The tab renders one row per check, clean or not — a check that found
    nothing must still be visibly accounted for."""
    results = run_instant_checks(_dataset_info(10, [("id", "Int64", 0)]), _chart_data())

    assert [r["id"] for r in results] == [spec.id for spec in INSTANT_CHECKS]
    assert all(r["status"] == "done" for r in results)


def test_run_instant_checks_keeps_the_catalogue_order_stable():
    results = run_instant_checks(_dataset_info(10, [("id", "Int64", 0)]), _chart_data())

    assert [r["id"] for r in results] == [
        "empty_dataset",
        "all_null_column",
        "high_missingness",
        "constant_column",
        "class_imbalance",
        "broken_image_refs",
        "correlated_missingness",
        "likely_id_column",
        "skewed_numeric",
    ]


def test_a_check_that_raises_becomes_an_error_row_not_a_silent_pass(monkeypatch):
    """A check that could not run is not a check that passed."""
    def boom(ctx):
        raise ValueError("bad payload")

    monkeypatch.setattr(INSTANT_CHECKS[0], "fn", boom)

    results = run_instant_checks(_dataset_info(10, [("id", "Int64", 0)]), _chart_data())

    assert results[0]["status"] == "error"
    assert "bad payload" in results[0]["error"]
    # and the rest of the scan still ran
    assert all(r["status"] == "done" for r in results[1:])


def test_run_instant_checks_tolerates_a_chart_payload_missing_keys():
    """get_chart_data's shape has grown over time; a check must not explode
    on a payload that predates a key it reads."""
    results = run_instant_checks(_dataset_info(10, [("id", "Int64", 0)]), {})

    assert all(r["status"] == "done" for r in results)


# --- sorting and summary ------------------------------------------------


def test_sorted_findings_puts_critical_before_warning_before_info():
    findings = _run(
        _dataset_info(
            100, [("dead", "String", 100), ("sparse", "String", 80), ("a", "String", 40)]
        ),
        _chart_data(
            correlated_missingness=[{"columns": ["a"], "null_count": 40, "overlap": 0.99}]
        ),
    )

    severities = [f.severity for f in findings]
    assert severities == sorted(severities, key=["critical", "warning", "info"].index)


def test_severity_summary_counts_each_level():
    findings = [
        Finding("x", "critical", "t", "d", []),
        Finding("y", "warning", "t", "d", []),
        Finding("z", "warning", "t", "d", []),
    ]

    assert severity_summary(findings) == {"critical": 1, "warning": 2, "info": 0}


def test_severity_summary_is_all_zero_for_a_clean_dataset():
    assert severity_summary([]) == {"critical": 0, "warning": 0, "info": 0}


@pytest.mark.parametrize("spec", INSTANT_CHECKS, ids=lambda s: s.id)
def test_every_catalogued_check_has_a_title_and_is_marked_instant(spec):
    assert spec.title
    assert spec.kind == "instant"


def test_class_imbalance_skips_a_column_already_reported_as_constant():
    """Precedence, same reasoning as the all-null rule: a single-value
    column is trivially "100% one class". Reporting both says the same
    thing twice and doubles its weight in the severity counts."""
    findings = _run(
        _dataset_info(100, [("site", "String", 0)]),
        _chart_data(categorical_bars=[{"column": "site", "categories": ["only"], "counts": [100]}]),
    )

    assert "constant_column" in _ids(findings)
    assert "class_imbalance" not in _ids(findings)
