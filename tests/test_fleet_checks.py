"""Group-comparison checks, over hand-built payloads.

Same rule as tests/test_findings.py: these are pure functions over two
dicts, so nothing here touches a file. What is being pinned is the
JUDGEMENT — which numbers are worth a finding and which are not — not the
arithmetic, which tests/test_fleet.py covers.
"""
from __future__ import annotations

import pytest

from ml_data_view.findings import CheckContext, run_instant_checks
from ml_data_view.fleet_checks import (
    MAX_NAMED_DIVERGENCES,
    check_channel_divergence,
    check_group_duration,
    check_unit_outlier,
    has_fleet,
    has_unit_durations,
)


def _channel(column: str, entries: list[dict]) -> dict:
    return {
        "column": column,
        "fleet_mean": 10.0,
        "group_mean_median": 10.0,
        "groups": entries,
    }


def _entry(value: str, psi: float, severity: str, **extra) -> dict:
    return {"value": value, "psi": psi, "severity": severity, **extra}


def _context(fleet: dict | None) -> CheckContext:
    return CheckContext(dataset_info={}, chart_data={"fleet": fleet} if fleet else {})


def _fleet(**overrides) -> dict:
    payload = {
        "available": True,
        "group_label": "System",
        "unit_label": "Recording",
        "thresholds": {"duration_ratio": 0.25, "outlier_sigma": 3.0},
        "channels": [],
        "groups": [],
        "units": {"available": False, "groups": []},
        "unit_anomalies": [],
    }
    payload.update(overrides)
    return payload


# --- applicability --------------------------------------------------------


def test_no_fleet_payload_means_the_checks_do_not_apply():
    assert has_fleet(_context(None)) is False
    assert has_unit_durations(_context(None)) is False


def test_an_unavailable_comparison_does_not_apply():
    assert has_fleet(_context({"available": False, "reason": "no grouping column"})) is False


def test_channel_divergence_applies_without_durations():
    """A grouping with no time axis can still have its channels compared;
    how long its recordings ran is a question with no answer."""
    context = _context(_fleet())

    assert has_fleet(context) is True
    assert has_unit_durations(context) is False


def test_the_registry_skips_them_all_when_there_is_no_grouping():
    results = {r["id"]: r for r in run_instant_checks({}, {})}

    for check in ("fleet_channel_divergence", "fleet_group_duration", "unit_outlier"):
        assert results[check]["status"] == "skipped"
        assert "grouping" in results[check]["reason"]


# --- channel divergence ---------------------------------------------------


def test_a_stable_channel_produces_nothing():
    fleet = _fleet(channels=[_channel("temp", [_entry("a", 0.01, "stable")])])

    assert check_channel_divergence(_context(fleet)) == []


def test_a_significant_pair_is_named_with_its_group_and_channel():
    fleet = _fleet(
        channels=[
            _channel(
                "temp",
                [_entry("456", 0.41, "significant", mean=21.3, mean_z=3.2), _entry("1", 0.0, "stable")],
            )
        ]
    )

    findings = check_channel_divergence(_context(fleet))

    assert len(findings) == 1
    assert findings[0].severity == "warning"
    assert "System 456" in findings[0].title
    assert "temp" in findings[0].title
    assert findings[0].columns == ["temp"]
    assert "3.2σ" in findings[0].detail
    assert "PSI 0.41" in findings[0].detail


def test_the_sigma_clause_is_dropped_when_the_spread_is_zero():
    fleet = _fleet(
        channels=[
            _channel("temp", [_entry("a", 0.9, "significant", mean=1.0, mean_z=None), _entry("b", 0.0, "stable")])
        ]
    )

    detail = check_channel_divergence(_context(fleet))[0].detail

    assert "σ" not in detail
    assert "PSI 0.90" in detail


def test_moderate_pairs_are_counted_not_listed():
    fleet = _fleet(
        channels=[
            _channel(f"c{index}", [_entry("a", 0.15, "moderate"), _entry("b", 0.0, "stable")])
            for index in range(6)
        ]
    )

    findings = check_channel_divergence(_context(fleet))

    assert len(findings) == 1
    assert findings[0].severity == "info"
    assert "6 channel(s) moderately different" in findings[0].title


def test_past_the_cap_the_rest_become_one_row():
    fleet = _fleet(
        channels=[
            _channel(f"c{index}", [_entry("a", 0.9, "significant"), _entry("b", 0.0, "stable")])
            for index in range(MAX_NAMED_DIVERGENCES + 5)
        ]
    )

    findings = check_channel_divergence(_context(fleet))

    assert len(findings) == MAX_NAMED_DIVERGENCES + 1
    assert "5 further significant divergence(s)" in findings[-1].title


# --- group durations ------------------------------------------------------


def _units(groups: list[dict]) -> dict:
    return {"available": True, "duration_ratio_threshold": 0.25, "groups": groups}


def test_a_group_within_the_threshold_says_nothing():
    fleet = _fleet(
        units=_units([{"value": "a", "unit_count": 5, "duration_ratio": 1.1, "comparable": True}])
    )

    assert check_group_duration(_context(fleet)) == []


def test_a_group_whose_units_are_much_shorter_is_reported_as_a_percentage():
    fleet = _fleet(
        units=_units(
            [
                {
                    "value": "789",
                    "unit_count": 5,
                    "duration_ratio": 0.6,
                    "median_duration": 60.0,
                    "reference_duration": 100.0,
                    "comparable": True,
                }
            ]
        )
    )

    findings = check_group_duration(_context(fleet))

    assert len(findings) == 1
    assert "System 789" in findings[0].title
    assert "40% shorter" in findings[0].title
    assert "1m 40.0s" in findings[0].detail


def test_a_group_whose_units_are_much_longer_says_longer():
    fleet = _fleet(
        units=_units(
            [{"value": "a", "unit_count": 4, "duration_ratio": 2.0, "comparable": True}]
        )
    )

    assert "100% longer" in check_group_duration(_context(fleet))[0].title


def test_a_group_with_too_few_units_is_not_judged():
    fleet = _fleet(
        units=_units([{"value": "a", "unit_count": 2, "duration_ratio": 0.1, "comparable": False}])
    )

    assert check_group_duration(_context(fleet)) == []


def test_the_threshold_comes_from_the_payload():
    fleet = _fleet(
        thresholds={"duration_ratio": 0.6, "outlier_sigma": 3.0},
        units=_units([{"value": "a", "unit_count": 5, "duration_ratio": 0.6, "comparable": True}]),
    )

    assert check_group_duration(_context(fleet)) == []


# --- unit outliers --------------------------------------------------------


def test_a_unit_outlier_names_the_unit_it_is_about():
    fleet = _fleet(
        unit_anomalies=[
            {
                "unit": "123 / rec_7",
                "group": "123",
                "metric": "duration",
                "metric_label": "duration",
                "value": 5.0,
                "group_median": 600.0,
                "z": -95.0,
            }
        ]
    )

    findings = check_unit_outlier(_context(fleet))

    assert len(findings) == 1
    # The unit field is what a catalogue row joins on to colour its health
    # column — not a substring of the detail text.
    assert findings[0].unit == "123 / rec_7"
    assert "10m 00.0s" in findings[0].detail
    assert "95σ" in findings[0].detail


@pytest.mark.parametrize(
    "metric,value,expected",
    [("hz", 1.0, "1Hz"), ("gap_share", 0.42, "42.0%"), ("duration", 0.5, "500ms")],
)
def test_each_measure_is_shown_in_its_own_units(metric, value, expected):
    fleet = _fleet(
        unit_anomalies=[
            {
                "unit": "u",
                "group": "g",
                "metric": metric,
                "metric_label": metric,
                "value": value,
                "group_median": value,
                "z": 9.0,
            }
        ]
    )

    assert expected in check_unit_outlier(_context(fleet))[0].detail


def test_no_anomalies_is_no_findings():
    assert check_unit_outlier(_context(_fleet())) == []
