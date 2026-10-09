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
    diff_findings,
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
    # Every check is accounted for — but a check whose precondition this
    # bare dataset does not have is SKIPPED, which is neither a pass nor a
    # failure. The four conditional checks are exactly the ones with an
    # `applies` predicate.
    conditional = {spec.id for spec in INSTANT_CHECKS if spec.applies is not None}
    for result in results:
        expected = "skipped" if result["id"] in conditional else "done"
        assert result["status"] == expected, result
    assert all(r.get("reason") for r in results if r["status"] == "skipped")


def test_run_instant_checks_keeps_the_catalogue_order_stable():
    results = run_instant_checks(_dataset_info(10, [("id", "Int64", 0)]), _chart_data())

    assert [r["id"] for r in results] == [
        # The conditional checks come first: what data is here at all, and
        # whether it is ordered in time, frame everything below them. Each
        # declares a precondition and simply does not run without it, which
        # is why a flat CSV sees the four of them skipped.
        "structure",
        "timestamp_integrity",
        "group_balance",
        "empty_dataset",
        "all_null_column",
        "high_missingness",
        "constant_column",
        "class_imbalance",
        "broken_image_refs",
        "correlated_missingness",
        "likely_id_column",
        "skewed_numeric",
        # Time-series health joins the registry from its own module
        # (ts_checks.py), appended after the checks defined here.
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
        # Group comparison joins from its own module too (fleet_checks.py),
        # appended after the time-series family.
        "fleet_channel_divergence",
        "fleet_group_duration",
        "unit_outlier",
    ]


def test_a_check_that_raises_becomes_an_error_row_not_a_silent_pass(monkeypatch):
    """A check that could not run is not a check that passed."""
    def boom(ctx):
        raise ValueError("bad payload")

    # An UNCONDITIONAL check, so it genuinely runs and genuinely raises —
    # patching a skipped one would prove nothing.
    target = next(spec for spec in INSTANT_CHECKS if spec.applies is None)
    monkeypatch.setattr(target, "fn", boom)

    results = run_instant_checks(_dataset_info(10, [("id", "Int64", 0)]), _chart_data())

    failed = next(r for r in results if r["id"] == target.id)
    assert failed["status"] == "error"
    assert "bad payload" in failed["error"]
    # and the rest of the scan still ran
    assert all(
        r["status"] in ("done", "skipped") for r in results if r["id"] != target.id
    )


def test_run_instant_checks_tolerates_a_chart_payload_missing_keys():
    """get_chart_data's shape has grown over time; a check must not explode
    on a payload that predates a key it reads."""
    results = run_instant_checks(_dataset_info(10, [("id", "Int64", 0)]), {})

    assert all(r["status"] in ("done", "skipped") for r in results)
    assert not any(r["status"] == "error" for r in results)


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


# --- comparing one scan against the last ---------------------------------
#
# What a watch-triggered re-profile interrupts someone for. "12 findings"
# after every file lands is noise; "1 new finding: high null rate in column
# X" is the thing they opened the app for.


def _finding(check_id, columns, title="High null rate", detail="41% missing"):
    return {
        "check_id": check_id,
        "severity": "warning",
        "title": title,
        "detail": detail,
        "columns": list(columns),
    }


def test_no_previous_scan_means_no_diff():
    """A first profile has nothing to compare against, and calling every
    finding on a freshly opened dataset "new" would make the word mean
    nothing."""
    assert diff_findings(None, [_finding("a", ["x"])]) is None


def test_a_finding_that_appeared_is_new():
    changes = diff_findings([_finding("a", ["x"])], [_finding("a", ["x"]), _finding("a", ["y"])])

    assert [f["columns"] for f in changes["new"]] == [["y"]]
    assert changes["resolved"] == []
    assert changes["summary"] == "1 new finding: High null rate in column y."


def test_a_finding_that_went_away_is_resolved():
    changes = diff_findings([_finding("a", ["x"]), _finding("a", ["y"])], [_finding("a", ["x"])])

    assert [f["columns"] for f in changes["resolved"]] == [["y"]]
    assert changes["summary"] == "1 resolved."


def test_a_changed_detail_is_not_a_new_finding():
    """Identity is the check plus its columns, NOT the detail text — which
    carries counts that move with every file. "41% missing" becoming "43%
    missing" is the same finding, and reporting it as new would train
    people to ignore the notification."""
    before = [_finding("high_missingness", ["x"], detail="41% missing")]
    after = [_finding("high_missingness", ["x"], detail="43% missing")]

    changes = diff_findings(before, after)

    assert changes["new"] == []
    assert changes["summary"] == "No new findings."


def test_the_same_check_on_a_different_column_is_a_different_finding():
    changes = diff_findings([_finding("a", ["x"])], [_finding("a", ["z"])])

    assert len(changes["new"]) == 1
    assert len(changes["resolved"]) == 1


def test_nothing_changing_still_says_so():
    """Worth saying: it is the answer someone watching a pipeline most
    wants, and is otherwise indistinguishable from the check not running."""
    findings = [_finding("a", ["x"])]

    assert diff_findings(findings, findings)["summary"] == "No new findings."


def test_several_new_findings_are_counted_not_listed():
    """A notification that names eight columns is not a notification."""
    changes = diff_findings([], [_finding("a", ["x"]), _finding("b", ["y"])])

    assert changes["summary"] == "2 new findings."


def test_a_finding_across_several_columns_is_named_by_count():
    changes = diff_findings([], [_finding("corr", ["a", "b", "c"], title="Correlated missingness")])

    assert changes["summary"] == "1 new finding: Correlated missingness across 3 columns."


def test_a_finding_with_no_columns_is_named_by_title_alone():
    changes = diff_findings([], [_finding("empty", [], title="Dataset is empty")])

    assert changes["summary"] == "1 new finding: Dataset is empty."


def test_new_and_resolved_are_reported_together():
    changes = diff_findings([_finding("a", ["x"])], [_finding("b", ["y"])])

    assert "1 new finding" in changes["summary"]
    assert "1 resolved" in changes["summary"]

# --- fleet structure ------------------------------------------------------
#
# The structure check is the one instant check whose findings were computed
# somewhere else — hierarchy.py, from parquet footers at discovery time.
# It still obeys the catalogue's rule: a pure function over the two dicts
# the app already has, with no I/O of its own.


def test_structure_findings_are_surfaced_as_findings():
    info = _dataset_info(10, [("id", "Int64", 0)])
    # The check's precondition: a structure was detected at all.
    info["capabilities"] = {"has_structure": True}
    info["structure_findings"] = [
        {
            "check_id": "chunk_order_vs_time",
            "severity": "warning",
            "title": "Chunk order by name differs from time order",
            "detail": "sys_key=123 / rec_b: ordered by footer timestamps instead.",
            "columns": [],
        }
    ]

    findings = _run(info, _chart_data())

    assert "chunk_order_vs_time" in _ids(findings)
    assert "footer timestamps" in next(
        f for f in findings if f.check_id == "chunk_order_vs_time"
    ).detail


def test_a_flat_input_skips_the_structure_check_entirely():
    """An input with no directory structure has nothing to say here — and
    "skipped" is the honest answer, not a green tick claiming it was
    examined for something it cannot have."""
    results = run_instant_checks(_dataset_info(10, [("id", "Int64", 0)]), _chart_data())

    structure = next(r for r in results if r["id"] == "structure")
    assert structure["status"] == "skipped"
    assert structure["reason"] == "needs a detected directory structure"
    assert structure["findings"] == []
