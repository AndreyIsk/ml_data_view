"""Group-vs-all comparison, over hand-built cached statistics.

fleet.py has no I/O, so everything here is numbers in and numbers out —
the same rule findings.py follows and for the same reason. The synthetic
fleet that goes through the real cache lives in tests/test_fleet_api.py.
"""
from __future__ import annotations

import pytest

from ml_data_view.fleet import (
    DIVERGENCE_METHOD,
    FleetThresholds,
    build_fleet,
    diverging,
    group_time_order,
    median,
    robust_sigma,
    unit_groups,
)

EDGES = [0.0, 1.0, 2.0, 3.0, 4.0]


def _hist(counts, n=None, total=None):
    n = sum(counts) if n is None else n
    total = float(sum(i * c for i, c in enumerate(counts))) if total is None else total
    return {"hist": {"temp": list(counts)}, "numeric": {"temp": {"n": n, "sum": total}}}


def _fleet(per_file, file_groups, catalog=None, **kwargs):
    return build_fleet(
        group_column="sys_key",
        group_label="System",
        unit_label="Recording",
        file_groups=file_groups,
        per_file=per_file,
        edges={"temp": EDGES},
        catalog=catalog,
        **kwargs,
    )


# --- applicability --------------------------------------------------------


def test_no_grouping_column_means_no_comparison_and_says_so():
    result = build_fleet(
        group_column=None,
        group_label="",
        unit_label="File",
        file_groups=[],
        per_file={},
        edges={},
    )

    assert result["available"] is False
    assert result["reason"] == "no grouping column"


def test_one_group_is_not_a_comparison():
    result = _fleet({"a": _hist([1, 1, 1, 1])}, [("a", "only")])

    assert result["available"] is False
    assert "at least two" in result["reason"]


def test_a_channel_present_in_one_group_alone_is_not_a_divergence():
    """That is a schema difference, which the Compare tab reports."""
    result = _fleet(
        {"a": _hist([1, 1, 1, 1]), "b": {"hist": {}, "numeric": {}}},
        [("a", "one"), ("b", "two")],
    )

    assert result["channels"] == []


# --- the measure ----------------------------------------------------------


def test_identical_groups_diverge_by_zero():
    result = _fleet(
        {"a": _hist([4, 4, 4, 4]), "b": _hist([4, 4, 4, 4])},
        [("a", "one"), ("b", "two")],
    )

    scores = {entry["value"]: entry["psi"] for entry in result["channels"][0]["groups"]}
    assert scores == {"one": 0.0, "two": 0.0}
    assert all(
        entry["severity"] == "stable" for entry in result["channels"][0]["groups"]
    )


def test_a_shifted_group_is_the_one_that_diverges():
    result = _fleet(
        {
            "a": _hist([40, 40, 0, 0]),
            "b": _hist([40, 40, 0, 0]),
            "c": _hist([0, 0, 40, 40]),  # the deviant
        },
        [("a", "one"), ("b", "two"), ("c", "three")],
    )

    scores = {entry["value"]: entry for entry in result["channels"][0]["groups"]}
    assert scores["three"]["severity"] == "significant"
    assert scores["one"]["psi"] < scores["three"]["psi"]
    assert scores["two"]["psi"] < scores["three"]["psi"]


def test_the_method_is_named_in_the_payload():
    result = _fleet(
        {"a": _hist([1, 1, 1, 1]), "b": _hist([1, 1, 1, 1])},
        [("a", "one"), ("b", "two")],
    )

    assert result["method"] == DIVERGENCE_METHOD
    assert "shared bin edges" in result["method_detail"]


def test_a_groups_histogram_is_the_sum_of_its_files():
    result = _fleet(
        {
            "a1": _hist([1, 0, 0, 0]),
            "a2": _hist([2, 0, 0, 0]),
            "b1": _hist([0, 0, 0, 3]),
        },
        [("a1", "one"), ("a2", "one"), ("b1", "two")],
    )

    groups = {entry["value"]: entry for entry in result["channels"][0]["groups"]}
    assert groups["one"]["counts"] == [3, 0, 0, 0]
    assert result["channels"][0]["fleet_counts"] == [3, 0, 0, 3]


def test_a_merged_mean_is_exact_from_the_cached_sums():
    result = _fleet(
        {
            "a": {"hist": {"temp": [1, 1, 0, 0]}, "numeric": {"temp": {"n": 2, "sum": 10.0}}},
            "b": {"hist": {"temp": [0, 0, 1, 1]}, "numeric": {"temp": {"n": 2, "sum": 30.0}}},
        },
        [("a", "one"), ("b", "two")],
    )

    groups = {entry["value"]: entry for entry in result["channels"][0]["groups"]}
    assert groups["one"]["mean"] == 5.0
    assert groups["two"]["mean"] == 15.0
    assert result["channels"][0]["fleet_mean"] == 10.0


def test_the_sigma_distance_is_quoted_against_the_groups_own_medians():
    """Three groups near 1.0 and one far above: the far one is the one
    that gets a large σ distance, and the others do not."""
    per_file = {
        "a": {"hist": {"temp": [1, 0, 0, 0]}, "numeric": {"temp": {"n": 10, "sum": 10.0}}},
        "b": {"hist": {"temp": [1, 0, 0, 0]}, "numeric": {"temp": {"n": 10, "sum": 11.0}}},
        "c": {"hist": {"temp": [1, 0, 0, 0]}, "numeric": {"temp": {"n": 10, "sum": 9.0}}},
        "d": {"hist": {"temp": [0, 0, 0, 1]}, "numeric": {"temp": {"n": 10, "sum": 40.0}}},
    }
    result = _fleet(per_file, [("a", "1"), ("b", "2"), ("c", "3"), ("d", "4")])

    groups = {entry["value"]: entry for entry in result["channels"][0]["groups"]}
    assert abs(groups["4"]["mean_z"]) > 3
    assert all(abs(groups[v]["mean_z"]) <= 3 for v in ("1", "2", "3"))


def test_an_identical_spread_quotes_no_sigma_rather_than_infinity():
    per_file = {
        "a": {"hist": {"temp": [1, 0, 0, 0]}, "numeric": {"temp": {"n": 1, "sum": 1.0}}},
        "b": {"hist": {"temp": [1, 0, 0, 0]}, "numeric": {"temp": {"n": 1, "sum": 1.0}}},
    }
    result = _fleet(per_file, [("a", "one"), ("b", "two")])

    assert all(entry["mean_z"] is None for entry in result["channels"][0]["groups"])


# --- channel capping ------------------------------------------------------


def test_every_channel_is_scored_but_only_the_worst_carry_their_histograms():
    per_file = {
        "a": {
            "hist": {"x": [8, 0], "y": [4, 4], "z": [4, 4]},
            "numeric": {"x": {"n": 8, "sum": 1.0}, "y": {"n": 8, "sum": 1.0}, "z": {"n": 8, "sum": 1.0}},
        },
        "b": {
            "hist": {"x": [0, 8], "y": [4, 4], "z": [4, 4]},
            "numeric": {"x": {"n": 8, "sum": 1.0}, "y": {"n": 8, "sum": 1.0}, "z": {"n": 8, "sum": 1.0}},
        },
    }
    result = build_fleet(
        group_column="sys_key",
        group_label="System",
        unit_label="Recording",
        file_groups=[("a", "one"), ("b", "two")],
        per_file=per_file,
        edges={"x": [0.0, 1.0, 2.0], "y": [0.0, 1.0, 2.0], "z": [0.0, 1.0, 2.0]},
        max_channels=1,
    )

    assert result["channels_scored"] == ["x", "y", "z"]
    assert result["channels_charted"] == ["x"]
    charted = next(c for c in result["channels"] if c["column"] == "x")
    dropped = next(c for c in result["channels"] if c["column"] == "y")
    assert charted["groups"][0]["counts"] == [8, 0]
    assert "counts" not in dropped["groups"][0]
    # Scored all the same — a finding about the fortieth column is not
    # lost because its chart was not drawn.
    assert dropped["groups"][0]["psi"] == 0.0


# --- ordered groupings ----------------------------------------------------


def _catalog(rows, group_labels=("System",)):
    return {"rows": rows, "group_labels": list(group_labels), "groups": []}


def _resolved(catalog):
    """Every row keyed to its group, the way build_fleet resolves it."""
    pairs = [(file, row["group_values"][0]) for row in catalog["rows"] for file in row["files"]]
    return unit_groups(catalog, pairs)


def _row(label, group, start=None, end=None, duration=None, hz=None, gap_share=None):
    """A catalogue row. Its `files` are what a group is resolved through —
    the grouping column need not be the outermost level, so a row's own
    outermost group value is not the answer."""
    return {
        "label": label,
        "files": [group],
        "group_values": [group],
        "start": start,
        "end": end,
        "duration": duration,
        "hz": hz,
        "gap_share": gap_share,
    }


def test_groups_whose_spans_do_not_overlap_are_time_ordered():
    catalog = _catalog(
        [
            _row("d1 / r1", "day=1", start=0.0, end=10.0),
            _row("d2 / r1", "day=2", start=20.0, end=30.0),
        ]
    )

    assert group_time_order(catalog, _resolved(catalog)) == ["day=1", "day=2"]


def test_groups_that_overlap_in_time_are_not_a_time_ordering():
    catalog = _catalog(
        [
            _row("a / r1", "sys=1", start=0.0, end=100.0),
            _row("b / r1", "sys=2", start=50.0, end=150.0),
        ]
    )

    assert group_time_order(catalog, _resolved(catalog)) is None


def test_an_unknown_time_bound_is_not_a_time_ordering():
    catalog = _catalog(
        [_row("a / r1", "sys=1", start=0.0, end=1.0), _row("b / r1", "sys=2")]
    )

    assert group_time_order(catalog, _resolved(catalog)) is None


def test_a_time_ordered_grouping_also_compares_against_the_previous_group():
    catalog = _catalog(
        [
            _row("day=1 / f", "day=1", start=0.0, end=10.0),
            _row("day=2 / f", "day=2", start=20.0, end=30.0),
            _row("day=3 / f", "day=3", start=40.0, end=50.0),
        ]
    )
    per_file = {
        "day=1": _hist([8, 0, 0, 0]),
        "day=2": _hist([8, 0, 0, 0]),
        "day=3": _hist([0, 0, 0, 8]),
    }

    result = _fleet(
        per_file,
        [("day=1", "day=1"), ("day=2", "day=2"), ("day=3", "day=3")],
        catalog=catalog,
    )

    assert result["time_ordered"] is True
    entries = {entry["value"]: entry for entry in result["channels"][0]["groups"]}
    assert "psi_previous" not in entries["day=1"]  # nothing precedes the first
    assert entries["day=2"]["psi_previous"] == 0.0
    assert entries["day=3"]["previous"] == "day=2"
    assert entries["day=3"]["severity_previous"] == "significant"


def test_an_unordered_grouping_offers_no_previous_comparison():
    result = _fleet(
        {"a": _hist([1, 1, 1, 1]), "b": _hist([1, 1, 1, 1])},
        [("a", "one"), ("b", "two")],
    )

    assert result["time_ordered"] is False
    assert all("psi_previous" not in e for e in result["channels"][0]["groups"])


# --- unit rollups ---------------------------------------------------------


def test_a_group_whose_units_are_systematically_short_is_measured_as_a_ratio():
    rows = [_row(f"long / r{i}", "long", duration=100.0) for i in range(3)]
    rows += [_row(f"short / r{i}", "short", duration=60.0) for i in range(3)]
    catalog = _catalog(rows)

    result = _fleet(
        {"long": _hist([1, 1, 1, 1]), "short": _hist([1, 1, 1, 1])},
        [("long", "long"), ("short", "short")],
        catalog=catalog,
    )

    groups = {group["value"]: group for group in result["units"]["groups"]}
    assert groups["short"]["duration_ratio"] == 0.6
    assert groups["short"]["comparable"] is True


def test_a_group_with_two_units_is_computed_but_not_judged():
    rows = [_row(f"a / r{i}", "a", duration=10.0) for i in range(2)]
    rows += [_row(f"b / r{i}", "b", duration=10.0) for i in range(3)]

    result = _fleet(
        {"a": _hist([1, 1, 1, 1]), "b": _hist([1, 1, 1, 1])},
        [("a", "a"), ("b", "b")],
        catalog=_catalog(rows),
    )

    groups = {group["value"]: group for group in result["units"]["groups"]}
    assert groups["a"]["comparable"] is False
    assert groups["b"]["comparable"] is True


def test_no_durations_at_all_means_no_unit_summary():
    result = _fleet(
        {"one": _hist([1, 1, 1, 1]), "two": _hist([1, 1, 1, 1])},
        [("one", "one"), ("two", "two")],
        catalog=_catalog([_row("one / r", "one"), _row("two / r", "two")]),
    )

    assert result["units"]["available"] is False


# --- unit anomalies within a group ----------------------------------------


def test_a_unit_far_from_its_own_groups_median_is_an_anomaly():
    rows = [_row(f"a / r{i}", "a", duration=100.0) for i in range(5)]
    rows.append(_row("a / odd", "a", duration=5.0))
    rows += [_row(f"b / r{i}", "b", duration=100.0) for i in range(5)]

    result = _fleet(
        {"a": _hist([1, 1, 1, 1]), "b": _hist([1, 1, 1, 1])},
        [("a", "a"), ("b", "b")],
        catalog=_catalog(rows),
    )

    assert [item["unit"] for item in result["unit_anomalies"]] == ["a / odd"]
    assert result["unit_anomalies"][0]["metric"] == "duration"
    assert result["unit_anomalies"][0]["group_median"] == 100.0


def test_a_group_whose_units_all_differ_the_same_way_has_no_anomaly():
    """A group that records in five-minute bursts is not full of
    anomalies; it is a group with short recordings."""
    rows = [_row(f"a / r{i}", "a", duration=300.0) for i in range(5)]
    rows += [_row(f"b / r{i}", "b", duration=9000.0) for i in range(5)]

    result = _fleet(
        {"a": _hist([1, 1, 1, 1]), "b": _hist([1, 1, 1, 1])},
        [("a", "a"), ("b", "b")],
        catalog=_catalog(rows),
    )

    assert result["unit_anomalies"] == []


def test_a_group_with_two_units_raises_no_anomaly():
    rows = [_row("a / r1", "a", duration=100.0), _row("a / r2", "a", duration=1.0)]
    rows += [_row(f"b / r{i}", "b", duration=100.0) for i in range(3)]

    result = _fleet(
        {"a": _hist([1, 1, 1, 1]), "b": _hist([1, 1, 1, 1])},
        [("a", "a"), ("b", "b")],
        catalog=_catalog(rows),
    )

    assert [item["group"] for item in result["unit_anomalies"]] == []


def test_rate_and_gap_share_are_looked_at_as_well_as_duration():
    rows = [_row(f"a / r{i}", "a", duration=100.0, hz=10.0) for i in range(5)]
    rows.append(_row("a / slow", "a", duration=100.0, hz=1.0))

    result = _fleet(
        {"a": _hist([1, 1, 1, 1]), "b": _hist([1, 1, 1, 1])},
        [("a", "a"), ("b", "b")],
        catalog=_catalog(rows + [_row("b / r", "b", duration=100.0, hz=10.0)]),
    )

    assert result["unit_anomalies"][0]["metric"] == "hz"
    assert result["unit_anomalies"][0]["metric_label"] == "sample rate"


# --- thresholds -----------------------------------------------------------


def test_the_outlier_threshold_comes_from_settings():
    rows = [_row(f"a / r{i}", "a", duration=100.0) for i in range(5)]
    rows.append(_row("a / odd", "a", duration=95.0))
    rows += [_row(f"b / r{i}", "b", duration=100.0) for i in range(3)]
    catalog = _catalog(rows)
    files = {"a": _hist([1, 1, 1, 1]), "b": _hist([1, 1, 1, 1])}
    pairs = [("a", "a"), ("b", "b")]

    strict = _fleet(files, pairs, catalog=catalog, thresholds=FleetThresholds(outlier_sigma=0.5))
    loose = _fleet(files, pairs, catalog=catalog, thresholds=FleetThresholds(outlier_sigma=50.0))

    assert [item["unit"] for item in strict["unit_anomalies"]] == ["a / odd"]
    assert loose["unit_anomalies"] == []


def test_thresholds_round_trip_through_a_dict():
    assert FleetThresholds.from_dict({"duration_ratio": 0.4}).duration_ratio == 0.4
    assert FleetThresholds.from_dict({}).outlier_sigma == 3.0
    assert FleetThresholds.from_dict(None).to_dict() == FleetThresholds().to_dict()


# --- the reporting view ---------------------------------------------------


def test_diverging_lists_the_unstable_pairs_worst_first():
    result = _fleet(
        {
            "a": _hist([40, 40, 0, 0]),
            "b": _hist([40, 40, 0, 0]),
            "c": _hist([0, 0, 40, 40]),
        },
        [("a", "one"), ("b", "two"), ("c", "three")],
    )

    rows = diverging(result)

    assert rows[0]["group"] == "three"
    assert rows[0]["column"] == "temp"
    assert all(row["severity"] != "stable" for row in rows)


# --- the small statistics -------------------------------------------------


@pytest.mark.parametrize(
    "values,expected",
    [([], None), ([1.0], 1.0), ([1.0, 3.0], 2.0), ([5.0, 1.0, 3.0], 3.0)],
)
def test_median(values, expected):
    assert median(values) == expected


def test_robust_sigma_of_identical_values_is_zero():
    assert robust_sigma([2.0, 2.0, 2.0]) == 0.0


def test_robust_sigma_ignores_a_single_extreme():
    tight = robust_sigma([10.0, 10.0, 11.0, 10.0, 10.0])
    with_outlier = robust_sigma([10.0, 10.0, 11.0, 10.0, 10.0, 5000.0])

    assert with_outlier == pytest.approx(tight, rel=0.5)


# --- the baseline is not the pooled aggregate -----------------------------


def test_one_deviant_group_does_not_make_every_other_group_diverge():
    """The failure mode the typical-group baseline exists for: with a
    pooled aggregate, a quarter of the mass sitting somewhere else makes
    the three well-behaved groups diverge from it too, and a screen that
    flags all four has told nobody which one to look at."""
    per_file = {
        "a": _hist([40, 0, 0, 0]),
        "b": _hist([40, 0, 0, 0]),
        "c": _hist([40, 0, 0, 0]),
        "d": _hist([0, 0, 0, 40]),
    }
    result = _fleet(per_file, [("a", "1"), ("b", "2"), ("c", "3"), ("d", "4")])

    scores = {entry["value"]: entry for entry in result["channels"][0]["groups"]}
    assert scores["4"]["severity"] == "significant"
    assert [scores[v]["severity"] for v in ("1", "2", "3")] == ["stable"] * 3
    # And the pooled number is still reported, for anyone who wants it.
    assert scores["1"]["psi_aggregate"] > scores["1"]["psi"]


def test_the_typical_profile_is_a_distribution():
    from ml_data_view.fleet import typical_profile

    profile = typical_profile([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0]])

    assert profile == [1.0, 0.0]
    assert sum(profile) == pytest.approx(1.0)


def test_groups_sharing_no_bin_at_all_fall_back_to_a_mean_profile():
    from ml_data_view.fleet import typical_profile

    profile = typical_profile([[1.0, 0.0], [0.0, 1.0]])

    assert profile == [0.5, 0.5]


def test_a_grouping_below_the_outermost_level_rolls_units_up_by_IT():
    """The grouping column need not be the outermost level. Someone
    comparing rigs inside sites is asking about rigs, and rolling their
    units up by site would answer a different question with numbers that
    look right."""
    rows = [
        {
            "label": f"{site} / {rig} / rec",
            "files": [f"{site}-{rig}"],
            "group_values": [site, rig],
            "start": None,
            "end": None,
            "duration": 100.0 if rig == "rig=a" else 20.0,
            "hz": None,
            "gap_share": None,
        }
        for site in ("site=n", "site=s")
        for rig in ("rig=a", "rig=b")
    ]
    per_file = {row["files"][0]: _hist([1, 1, 1, 1]) for row in rows}
    pairs = [(row["files"][0], row["group_values"][1]) for row in rows]

    result = _fleet(per_file, pairs, catalog=_catalog(rows, ("Site", "Rig")))

    groups = {group["value"]: group for group in result["units"]["groups"]}
    assert set(groups) == {"rig=a", "rig=b"}
    assert groups["rig=b"]["median_duration"] == 20.0
    assert groups["rig=b"]["duration_ratio"] == 0.2
    totals = {group["value"]: group for group in result["groups"]}
    assert totals["rig=a"]["unit_count"] == 2
    assert totals["rig=a"]["duration"] == 200.0
