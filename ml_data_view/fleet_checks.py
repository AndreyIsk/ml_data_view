"""Group-comparison checks, as ordinary members of the Findings registry.

The same arrangement ts_checks.py has with timeseries.py: fleet.py works
out the numbers during the chart build, and this module decides which of
them are worth saying out loud. Every check here is a pure function over
the two dicts the app already has.

APPLICABILITY
  a GROUPING with two or more groups  -> `fleet_channel_divergence`
  that, and unit DURATIONS            -> `fleet_group_duration`,
                                         `unit_outlier`

None of those is "a folder that looks like a fleet". A grouping is a
directory role, a hive key, or a column the user nominated; a dataset
with one group, or none, skips rather than passing — there is nothing to
compare a group against.

SEVERITY
A SIGNIFICANT divergence (PSI > 0.25, the Compare tab's own threshold) is
a WARNING: this group's channel does not describe the same thing the rest
of the fleet's does, and a model trained across them has learnt two
distributions as one. MODERATE divergences are rolled up into a single
INFO finding rather than listed — on a wide dataset there are dozens, and
forty rows of "slightly different" buries the two that are not.
"""
from __future__ import annotations

from typing import Any

from ml_data_view.fleet import FleetThresholds, diverging
from ml_data_view.findings import CheckContext, CheckSpec, Finding

# Significant pairs named individually before the rest become a count.
# A findings list is read top to bottom; past this many rows of the same
# check nobody is reading them, they are scrolling past them.
MAX_NAMED_DIVERGENCES = 20


def _fleet(context: CheckContext) -> dict[str, Any]:
    value = context.chart_data.get("fleet")
    return value if isinstance(value, dict) else {}


def has_fleet(context: CheckContext) -> bool:
    """The precondition every check here shares: a grouping that produced
    two or more groups to compare."""
    return bool(_fleet(context).get("available"))


def has_unit_durations(context: CheckContext) -> bool:
    """The extra precondition for the unit-level comparisons. A grouping
    with no time axis can still have its channels compared; how long its
    recordings ran is a question that has no answer."""
    return has_fleet(context) and bool((_fleet(context).get("units") or {}).get("available"))


def _thresholds(context: CheckContext) -> FleetThresholds:
    return FleetThresholds.from_dict(_fleet(context).get("thresholds"))


def _label(context: CheckContext) -> str:
    return _fleet(context).get("group_label") or "Group"


def check_channel_divergence(context: CheckContext) -> list[Finding]:
    """A group whose channel does not look like the rest of the fleet's.

    The score is PSI against the TYPICAL GROUP PROFILE over shared bins —
    the same measure, the same thresholds and the same implementation the
    Compare tab uses (compare.psi), against a baseline a minority of
    groups cannot move (see fleet.py). The σ distance quoted beside it is
    a second, independent way of saying it: how far this group's mean sits
    from the median of the groups' means, in robust σ. Both are given
    because they fail differently — PSI sees a distribution that changed
    shape without moving, and the mean distance sees one that moved
    without changing shape.
    """
    fleet = _fleet(context)
    label = _label(context)
    rows = diverging(fleet)
    significant = [row for row in rows if row["severity"] == "significant"]
    moderate = [row for row in rows if row["severity"] == "moderate"]

    findings = [
        Finding(
            check_id="fleet_channel_divergence",
            severity="warning",
            title=f"{label} {row['group']}: channel '{row['column']}' diverges from the fleet",
            detail=(
                f"PSI {row['psi']:.2f} against the typical {label.lower()} "
                f"(>0.25 is significant)"
                + _mean_clause(row)
                + ". Scored from the cached per-file histograms, which every file was "
                "binned into against the same edges — nothing was re-read to say this."
            ),
            columns=[row["column"]],
        )
        for row in significant[:MAX_NAMED_DIVERGENCES]
    ]

    remainder = len(significant) - len(findings)
    if remainder > 0:
        findings.append(
            Finding(
                check_id="fleet_channel_divergence",
                severity="warning",
                title=f"{remainder} further significant divergence(s) not listed",
                detail=(
                    f"{remainder} more ({label.lower()}, channel) pair(s) score above 0.25. "
                    "The full list is in the fleet comparison table."
                ),
            )
        )
    if moderate:
        findings.append(
            Finding(
                check_id="fleet_channel_divergence",
                severity="info",
                title=f"{len(moderate)} channel(s) moderately different across {label.lower()}s",
                detail=(
                    f"{len(moderate)} ({label.lower()}, channel) pair(s) score between 0.10 "
                    "and 0.25 — moderate on the conventional PSI scale. Listed here as a "
                    "count rather than a row each, because past a handful they are the "
                    "background rather than the finding."
                ),
            )
        )
    return findings


def _mean_clause(row: dict[str, Any]) -> str:
    """"…, mean 21.3 against a fleet median of 18.0 — 3.2σ away".

    Dropped entirely when the spread across groups is zero, since a
    distance in σ then means nothing; PSI still carries the claim. Same
    omission rule as load_stats.py: a figure that is not known drops its
    clause rather than showing a zero.
    """
    mean, centre, distance = row.get("mean"), row.get("group_mean_median"), row.get("mean_z")
    if mean is None or centre is None or distance is None:
        return ""
    return f", mean {mean:.4g} against a median of {centre:.4g} across groups — {abs(distance):.1f}σ away"


def check_group_duration(context: CheckContext) -> list[Finding]:
    """A group whose units are systematically shorter — or longer — than
    the others'.

    Against the median of the OTHER groups' medians, so a group cannot
    dilute the baseline it is being compared with, and so one group with
    a hundred units does not become the definition of typical. A group
    with fewer than three units is computed and not judged: its own
    median is not a median.
    """
    fleet = _fleet(context)
    label = _label(context)
    threshold = _thresholds(context).duration_ratio
    findings = []
    for group in (fleet.get("units") or {}).get("groups") or []:
        ratio = group.get("duration_ratio")
        if not group.get("comparable") or ratio is None:
            continue
        if abs(ratio - 1.0) < threshold:
            continue
        direction = "shorter" if ratio < 1 else "longer"
        findings.append(
            Finding(
                check_id="fleet_group_duration",
                severity="warning",
                title=(
                    f"{label} {group['value']}: recordings are typically "
                    f"{abs(1 - ratio):.0%} {direction} than the rest"
                ),
                detail=(
                    f"Median across its {group['unit_count']} unit(s) is "
                    f"{_seconds(group.get('median_duration'))}, against "
                    f"{_seconds(group.get('reference_duration'))} for the other "
                    f"{label.lower()}s. A figure averaged across the whole dataset is "
                    f"weighted towards whichever {label.lower()} records longest."
                ),
            )
        )
    return findings


def check_unit_outlier(context: CheckContext) -> list[Finding]:
    """A unit unlike the others IN ITS OWN GROUP.

    Not unlike the fleet: a group that records in five-minute bursts is
    not thereby full of anomalies, and the useful question about one of
    its recordings is whether it is unlike its siblings. The distance is
    robust (median and MAD), so the outlier does not inflate the spread it
    is being measured against.
    """
    label = _label(context)
    return [
        Finding(
            check_id="unit_outlier",
            severity="warning",
            title=f"{item['unit']}: {item['metric_label']} is unlike the rest of {label.lower()} {item['group']}",
            detail=(
                f"{item['unit']}: {item['metric_label']} is "
                f"{_metric(item['metric'], item['value'])}, against a median of "
                f"{_metric(item['metric'], item['group_median'])} across that "
                f"{label.lower()}'s units — {abs(item['z']):.0f}σ away on a robust spread."
            ),
            unit=item["unit"],
        )
        for item in _fleet(context).get("unit_anomalies") or []
    ]


def _metric(metric: str, value: Any) -> str:
    if value is None:
        return "unknown"
    if metric == "duration":
        return _seconds(value)
    if metric == "hz":
        return f"{value:.4g}Hz"
    if metric == "gap_share":
        return f"{value:.1%}"
    return f"{value:.4g}"


def _seconds(value: Any) -> str:
    """Same shape as ts_checks._seconds — a duration as a person reads it."""
    if value is None:
        return "unknown"
    if value < 1:
        return f"{value * 1000:.0f}ms"
    if value < 90:
        return f"{value:.2f}s"
    minutes, seconds = divmod(value, 60)
    return f"{int(minutes)}m {seconds:04.1f}s"


FLEET_CHECKS: list[CheckSpec] = [
    CheckSpec(
        "fleet_channel_divergence",
        "Channel divergence across groups",
        "instant",
        check_channel_divergence,
        applies=has_fleet,
        requires="a grouping with two or more groups",
    ),
    CheckSpec(
        "fleet_group_duration",
        "Group recording lengths",
        "instant",
        check_group_duration,
        applies=has_unit_durations,
        requires="a grouping and unit durations",
    ),
    CheckSpec(
        "unit_outlier",
        "Units unlike their group",
        "instant",
        check_unit_outlier,
        applies=has_unit_durations,
        requires="a grouping and unit durations",
    ),
]
