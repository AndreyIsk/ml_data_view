"""Group-vs-all comparison over ANY grouping, from cached statistics.

WHAT A "GROUP" IS HERE IS NOT DECIDED HERE
A grouping is whatever the app already has one of: a group-level directory
role, a hive partition key, or a column the user nominated. This module is
handed a (file -> group value) map and compares those groups; it does not
know whether the values are vehicle serial numbers, days, or sites. With
no grouping there is nothing to compare and the whole section reports
itself unavailable rather than inventing one.

THE MEASURE IS THE COMPARE TAB'S MEASURE
Divergence is PSI — Population Stability Index — over the SHARED bin
edges, scored with compare.py's own `psi` and `severity`. Not a second
statistic invented here: a user who has read the Compare tab's numbers
already knows what 0.31 means, and two different drift measures in one app
is one too many.

WHAT A GROUP IS COMPARED AGAINST IS NOT THE POOLED AGGREGATE
It is the TYPICAL GROUP PROFILE: the bin-wise MEDIAN of the groups'
proportions, renormalised to sum to one. The pooled aggregate — every
row of every group in one histogram — was the obvious choice and is
wrong, for a reason that shows up the moment the feature does its job.
One system whose channel sits somewhere else puts a quarter of the
aggregate's mass there, so every OTHER system now diverges from the
aggregate too, and a screen that flags all four systems has told nobody
which one to look at. The median profile cannot be moved by a minority of
groups, so the deviant one is the only one that scores. `psi_aggregate`
is reported beside it for anyone who wants the pooled number, and the
pooled aggregate is still what the charts outline — it is a real
distribution and a useful backdrop, just not a baseline.

The shared edges come for free. Every per-file histogram in the profile
cache was binned against a spec derived from the MERGED min/max of the
whole loaded scope (file_profile.py, "histograms carry their spec"), so
every file in every group is already binned identically. A group's
histogram is therefore the elementwise SUM of its files' cached counts,
and the fleet aggregate is the sum of all of them — exact, and no read at
all. This is the reason the feature can be honest about costing nothing:
it is arithmetic over numbers the first profile already stored.

WHY THE GROUPING MUST BE CONSTANT PER FILE
A cached per-file histogram cannot be split by a value that varies WITHIN
the file — recovering that means re-reading every file with a group_by,
which is exactly the cost this feature exists to avoid. So a grouping that
is a property of the path (a directory role, a hive key, any derived
column) is supported from cache, and a grouping that is an ordinary data
column reports itself unavailable WITH THAT REASON rather than silently
producing nothing. Nothing about the shape of the tree is assumed either
way; the requirement is about where the value lives, not how deep it is.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ml_data_view.compare import psi, severity

# Groups a comparison needs before it means anything. One group is not a
# comparison, and "this group differs from the aggregate of itself" is
# always false rather than usefully false.
MIN_GROUPS = 2
# Units a group needs before its own median is worth comparing anything
# against. A median of two is not a median — the same floor
# ts_checks.MIN_UNITS_FOR_DURATION applies for the same reason.
MIN_UNITS_FOR_MEDIAN = 3
# 1 / Φ⁻¹(0.75). Scales a median absolute deviation onto the same footing
# as a standard deviation for normally distributed data, which is what
# lets a robust spread be quoted in σ without meaning something different
# from the σ everybody has in mind.
MAD_TO_SIGMA = 1.4826
# A floor under the robust spread, as a share of the group's own median.
# A median absolute deviation is exactly zero whenever most of a group's
# units agree to the last decimal, and dividing by that would either call
# every remaining difference infinite or (treating it as "no spread")
# call none of them anything. Neither is right, so the spread a distance
# is quoted against is never smaller than 1% of the median: a difference
# below that is not worth a finding however uniform its neighbours are.
MIN_RELATIVE_SIGMA = 0.01

DIVERGENCE_METHOD = "PSI (Population Stability Index)"
DIVERGENCE_METHOD_DETAIL = (
    "Population Stability Index over the shared bin edges every file was profiled "
    "against, scored with the same thresholds the Compare tab uses "
    "(<0.10 stable, 0.10–0.25 moderate, >0.25 significant). A group's histogram is "
    "the sum of its files' cached bin counts, so nothing is re-read. The baseline "
    "is the typical group profile — the bin-wise median across groups, renormalised "
    "— so one deviant group cannot become the thing every other group is measured "
    "against."
)

# The per-unit figures a within-group anomaly is looked for in, and the
# words used for them. Duration, rate and gap share — the three things a
# recording can be unlike its siblings in without any single check firing.
UNIT_METRICS = (
    ("duration", "duration"),
    ("hz", "sample rate"),
    ("gap_share", "gap share"),
)


@dataclass(frozen=True)
class FleetThresholds:
    """What counts as worth saying. In Settings, like every other
    threshold here, so changing one re-judges numbers already cached
    without re-reading anything."""

    # How far a group's median unit duration must sit from the fleet's
    # before it is reported. 0.25 = a quarter shorter or longer.
    duration_ratio: float = 0.25
    # Robust-σ distance at which a unit is unlike its own group's others.
    outlier_sigma: float = 3.0

    def to_dict(self) -> dict[str, Any]:
        return {"duration_ratio": self.duration_ratio, "outlier_sigma": self.outlier_sigma}

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> FleetThresholds:
        data = data or {}
        defaults = cls()
        return cls(
            duration_ratio=float(data.get("duration_ratio", defaults.duration_ratio)),
            outlier_sigma=float(data.get("outlier_sigma", defaults.outlier_sigma)),
        )


def unavailable(reason: str) -> dict[str, Any]:
    """The shape every caller gets when there is nothing to compare.

    A reason, always. "No fleet comparison" and "no fleet comparison
    BECAUSE this dataset has one group" are different answers, and only
    the second one tells someone what to do about it.
    """
    return {"available": False, "reason": reason, "groups": [], "channels": []}


def median(values: list[float]) -> float | None:
    ordered = sorted(values)
    if not ordered:
        return None
    middle = len(ordered) // 2
    if len(ordered) % 2:
        return float(ordered[middle])
    return (ordered[middle - 1] + ordered[middle]) / 2


def robust_sigma(values: list[float]) -> float | None:
    """A median absolute deviation, scaled to σ.

    Robust rather than a standard deviation because the question is "is
    THIS one unlike the others" — and an outlier inflates the very spread
    it is being measured against, which is how a real deviant hides.
    """
    centre = median(values)
    if centre is None:
        return None
    deviation = median([abs(value - centre) for value in values])
    return None if deviation is None else deviation * MAD_TO_SIGMA


def unit_sigma(values: list[float]) -> tuple[float | None, float | None]:
    """The centre and spread a unit-level anomaly is measured against.

    A robust sigma with MIN_RELATIVE_SIGMA as its floor — see that
    constant for why a zero spread cannot be taken at face value.
    """
    centre = median(values)
    spread = robust_sigma(values)
    if centre is None:
        return None, None
    floor = abs(centre) * MIN_RELATIVE_SIGMA
    return centre, max(spread or 0.0, floor)


def _z(value: float, centre: float | None, sigma: float | None) -> float | None:
    if centre is None or not sigma:
        # A spread of zero says every other value is identical. That a
        # differing value is "infinitely many σ away" is true and useless,
        # so the distance is not quoted and PSI carries the claim instead.
        return None
    return (value - centre) / sigma


def unit_groups(
    catalog: dict[str, Any] | None, file_groups: list[tuple[str, str]]
) -> dict[str, str]:
    """Which group each catalogue row belongs to, for THIS grouping.

    Resolved through the row's own files rather than by taking its
    outermost group value, because the grouping column need not be the
    outermost level: someone comparing `rig=` inside `site=` is asking
    about rigs, and rolling their units up by site would answer a
    different question with the right-looking numbers.
    """
    lookup = dict(file_groups)
    resolved: dict[str, str] = {}
    for row in (catalog or {}).get("rows") or []:
        for file in row.get("files") or []:
            if file in lookup:
                resolved[row["label"]] = lookup[file]
                break
    return resolved


def group_time_order(
    catalog: dict[str, Any] | None, unit_group: dict[str, str]
) -> list[str] | None:
    """The groups in time order, when they genuinely ARE in time order.

    A time-based partitioning (`day=2026-03-01/…`) is just a grouping
    whose natural order is time, and for one of those "how does this
    group compare with the PREVIOUS one" is the question people actually
    have. But that is decided on EVIDENCE, not on a key called "day": the
    groups qualify when every one of them has known time bounds and,
    sorted by their start, their spans do not overlap. A grouping by
    vehicle fails that immediately, which is the point.
    """
    if not catalog:
        return None
    spans: dict[str, list[float]] = {}
    for row in catalog.get("rows") or []:
        value = unit_group.get(row.get("label"))
        start, end = row.get("start"), row.get("end")
        if value is None or start is None or end is None:
            return None
        span = spans.setdefault(value, [float(start), float(end)])
        span[0] = min(span[0], float(start))
        span[1] = max(span[1], float(end))
    if len(spans) < MIN_GROUPS:
        return None
    ordered = sorted(spans.items(), key=lambda item: item[1][0])
    for (_a, first), (_b, second) in zip(ordered, ordered[1:]):
        if second[0] < first[1]:
            return None
    return [value for value, _span in ordered]


def _proportions(counts: list[int]) -> list[float]:
    total = sum(counts)
    return [count / total for count in counts] if total else [0.0] * len(counts)


def typical_profile(profiles: list[list[float]]) -> list[float]:
    """The bin-wise median of several distributions, renormalised.

    A distribution in its own right — every bin holds what a typical group
    puts there — and one that a minority of groups cannot move. See the
    module docstring for why the pooled aggregate is not used for this.

    Falls back to the elementwise mean when every median is zero, which
    happens only when no two groups share a single bin; there is no
    typical profile then, and the mean is at least a distribution.
    """
    if not profiles:
        return []
    width = len(profiles[0])
    medians = [median([profile[index] for profile in profiles]) or 0.0 for index in range(width)]
    total = sum(medians)
    if not total:
        means = [sum(profile[index] for profile in profiles) / len(profiles) for index in range(width)]
        total = sum(means)
        return [value / total for value in means] if total else [0.0] * width
    return [value / total for value in medians]


def _sum_counts(count_lists: list[list[int]], width: int) -> list[int]:
    totals = [0] * width
    for counts in count_lists:
        for index, count in enumerate(counts[:width]):
            totals[index] += int(count)
    return totals


def build_fleet(
    *,
    group_column: str | None,
    group_label: str,
    unit_label: str,
    file_groups: list[tuple[str, str]],
    per_file: dict[str, dict[str, Any]],
    edges: dict[str, list[float]],
    catalog: dict[str, Any] | None = None,
    exclude: set[str] | None = None,
    max_channels: int = 12,
    thresholds: FleetThresholds | None = None,
) -> dict[str, Any]:
    """Per-channel divergence from the typical group, plus the unit rollups.

    `per_file` is the profile cache's own per-file content, narrowed to
    what this needs: `{"hist": {column: [counts]}, "numeric": {column:
    {"n", "sum", "min", "max"}}}`. Passed in rather than read here so this
    module has no I/O at all and can be tested against hand-built numbers.

    `edges` are the merged display edges per column — the proof that every
    file was binned the same way, and what the charts draw against.
    """
    thresholds = thresholds or FleetThresholds()
    exclude = exclude or set()

    if not group_column:
        return unavailable("no grouping column")
    groups = _group_files(file_groups)
    if len(groups) < MIN_GROUPS:
        return unavailable(
            f"only {len(groups)} group in {group_column!r} — a comparison needs at least two"
        )

    unit_group = unit_groups(catalog, file_groups)
    order = group_time_order(catalog, unit_group)
    time_ordered = order is not None
    if order is None:
        order = sorted(groups, key=_natural_key)
    else:
        order = [value for value in order if value in groups]

    channels = [
        channel
        for column in sorted(edges)
        if column not in exclude
        and (channel := _channel(column, edges[column], groups, order, per_file, time_ordered))
        is not None
    ]
    # Biggest divergence first, and only the worst few carry their bin
    # counts: a score is five numbers, a histogram per group is hundreds,
    # and the payload should not grow with a column count nobody is
    # looking at. Every channel is still SCORED, so a finding about the
    # fortieth column is not lost because its chart was not drawn.
    channels.sort(key=lambda channel: -(channel["max_psi"] or 0.0))
    charted = {channel["column"] for channel in channels[:max_channels]}
    for channel in channels:
        if channel["column"] not in charted:
            for entry in channel["groups"]:
                entry.pop("counts", None)
            channel.pop("fleet_counts", None)
            channel.pop("typical_profile", None)
            channel.pop("edges", None)

    units = _unit_summary(catalog, unit_group, order, thresholds)
    return {
        "available": True,
        "group_column": group_column,
        "group_label": group_label,
        "unit_label": unit_label,
        "method": DIVERGENCE_METHOD,
        "method_detail": DIVERGENCE_METHOD_DETAIL,
        "thresholds": thresholds.to_dict(),
        # Time order is EVIDENCE, not a naming convention — see
        # group_time_order. When it holds, "vs the previous group" is
        # offered beside "vs every group".
        "time_ordered": time_ordered,
        "groups": [
            {
                "value": value,
                "file_count": len(groups[value]),
                **_group_totals(catalog, unit_group, value),
            }
            for value in order
        ],
        "channels": channels,
        "channels_charted": sorted(charted),
        "channels_scored": [channel["column"] for channel in channels],
        "units": units,
        "unit_anomalies": _unit_anomalies(catalog, unit_group, thresholds),
    }


def _natural_key(text: str):
    from ml_data_view.hierarchy import natural_key

    return natural_key(text)


def _group_files(file_groups: list[tuple[str, str]]) -> dict[str, list[str]]:
    groups: dict[str, list[str]] = {}
    for file, value in file_groups:
        groups.setdefault(str(value), []).append(str(file))
    return groups


def _channel(
    column: str,
    column_edges: list[float],
    groups: dict[str, list[str]],
    order: list[str],
    per_file: dict[str, dict[str, Any]],
    time_ordered: bool,
) -> dict[str, Any] | None:
    """One channel's per-group distributions and their divergence.

    None when fewer than two groups have any data for this column: a
    column that exists in one group's files and nowhere else is a schema
    difference, which the Compare tab reports, not a distribution that
    drifted.
    """
    width = max(len(column_edges) - 1, 0)
    if not width:
        return None

    per_group: dict[str, dict[str, Any]] = {}
    for value in order:
        counts = _sum_counts(
            [
                (per_file.get(file, {}).get("hist") or {}).get(column) or []
                for file in groups[value]
            ],
            width,
        )
        stats = [
            (per_file.get(file, {}).get("numeric") or {}).get(column) or {}
            for file in groups[value]
        ]
        n = sum(int(entry.get("n") or 0) for entry in stats)
        total = sum(float(entry.get("sum") or 0.0) for entry in stats)
        if not sum(counts) and not n:
            continue
        per_group[value] = {
            "value": value,
            "counts": counts,
            "n": n,
            # Exact: a row belongs to exactly one file, so a merged mean is
            # the merged sum over the merged count. `sum` is cached for
            # precisely this.
            "mean": (total / n) if n else None,
        }
    if len(per_group) < MIN_GROUPS:
        return None

    present = [value for value in order if value in per_group]
    fleet_counts = _sum_counts([per_group[value]["counts"] for value in present], width)
    fleet_n = sum(per_group[value]["n"] for value in present)
    fleet_sum = sum(
        (per_group[value]["mean"] or 0.0) * per_group[value]["n"] for value in present
    )
    # Two baselines, and they are not interchangeable. `typical` is what
    # the score is against — see the module docstring. `aggregate` is the
    # pooled distribution, kept for the chart's outline and reported as a
    # second number for anyone who wants it.
    aggregate = _proportions(fleet_counts)
    typical = typical_profile([_proportions(per_group[v]["counts"]) for v in present])

    means = [per_group[v]["mean"] for v in present if per_group[v]["mean"] is not None]
    centre = median(means)
    sigma = robust_sigma(means)

    entries = []
    previous: str | None = None
    for value in present:
        entry = per_group[value]
        proportions = _proportions(entry["counts"])
        score = psi(proportions, typical)
        entry["psi"] = round(score, 4)
        entry["severity"] = severity(score)
        entry["psi_aggregate"] = round(psi(proportions, aggregate), 4)
        entry["mean_z"] = (
            None
            if entry["mean"] is None
            else _round(_z(entry["mean"], centre, sigma))
        )
        if time_ordered and previous is not None:
            step = psi(
                _proportions(entry["counts"]), _proportions(per_group[previous]["counts"])
            )
            entry["psi_previous"] = round(step, 4)
            entry["previous"] = previous
            entry["severity_previous"] = severity(step)
        previous = value
        entries.append(entry)

    return {
        "column": column,
        "edges": list(column_edges),
        "fleet_counts": fleet_counts,
        "typical_profile": typical,
        "fleet_n": fleet_n,
        "fleet_mean": (fleet_sum / fleet_n) if fleet_n else None,
        # The centre and spread the σ distances are quoted against, so a
        # reader can check the arithmetic rather than take "3.2σ" on trust.
        "group_mean_median": centre,
        "group_mean_sigma": sigma,
        "groups": entries,
        "max_psi": max((entry["psi"] for entry in entries), default=0.0),
    }


def _round(value: float | None) -> float | None:
    return None if value is None else round(value, 2)


def _group_totals(
    catalog: dict[str, Any] | None, unit_group: dict[str, str], value: str
) -> dict[str, Any]:
    """Units, rows and total duration for one group.

    Summed from the catalogue's ROWS rather than read off its group
    summaries: those roll up by the outermost level, which is not
    necessarily the level being compared.
    """
    members = [
        row
        for row in (catalog or {}).get("rows") or []
        if unit_group.get(row.get("label")) == value
    ]
    if not members:
        return {"unit_count": None, "rows": None, "duration": None}
    return {
        "unit_count": len(members),
        "rows": _total(row.get("rows") for row in members),
        "duration": _total(row.get("duration") for row in members),
    }


def _total(values) -> Any:
    """Sum, unless anything is unknown — catalog._sum_or_none's rule, and
    for the same reason: a total assembled from some of its parts is not a
    smaller total, it is a wrong one."""
    collected = list(values)
    if not collected or any(value is None for value in collected):
        return None
    return sum(collected)


def _unit_values(
    catalog: dict[str, Any] | None, unit_group: dict[str, str], key: str
) -> dict[str, list[float]]:
    """One metric's values per group, skipping units that do not have it."""
    values: dict[str, list[float]] = {}
    for row in (catalog or {}).get("rows") or []:
        group = unit_group.get(row.get("label"))
        value = row.get(key)
        if group is None or value is None:
            continue
        values.setdefault(group, []).append(float(value))
    return values


def _unit_summary(
    catalog: dict[str, Any] | None,
    unit_group: dict[str, str],
    order: list[str],
    thresholds: FleetThresholds,
) -> dict[str, Any]:
    """How each group's typical unit compares with the REST of the fleet.

    "Typical" is the MEDIAN, not the mean: one recording that ran all
    weekend should not make a group's recordings look long. The comparison
    is a ratio rather than a difference because durations span orders of
    magnitude between datasets, and "40% shorter" travels where "1,204
    seconds shorter" does not.

    The reference is the median of the OTHER groups' medians — leave one
    out, and groups weighted equally. Both halves of that matter: a group
    included in its own baseline dilutes the very difference being
    measured (with two groups it halves it), and a fleet median taken over
    every unit is really the median of whichever group has the most of
    them.
    """
    durations = _unit_values(catalog, unit_group, "duration")
    if not durations:
        return {"available": False, "reason": "no unit durations", "groups": []}

    medians = {value: median(values) for value, values in durations.items()}
    known = {value: centre for value, centre in medians.items() if centre is not None}
    fleet = median(list(known.values()))

    groups = []
    for value in order:
        values = durations.get(value) or []
        centre = medians.get(value)
        others = [other for name, other in known.items() if name != value]
        reference = median(others)
        groups.append(
            {
                "value": value,
                "unit_count": len(values),
                "median_duration": centre,
                "reference_duration": reference,
                "duration_ratio": (centre / reference)
                if (centre is not None and reference)
                else None,
                # Below this many units the group's own median is not a
                # median, so the comparison is computed and NOT judged.
                "comparable": len(values) >= MIN_UNITS_FOR_MEDIAN,
            }
        )
    return {
        "available": True,
        "fleet_median_duration": fleet,
        "duration_ratio_threshold": thresholds.duration_ratio,
        "groups": groups,
    }


def _unit_anomalies(
    catalog: dict[str, Any] | None,
    unit_group: dict[str, str],
    thresholds: FleetThresholds,
) -> list[dict[str, Any]]:
    """Units unlike the OTHERS IN THEIR OWN GROUP.

    Against the group's own median, not the fleet's: a group that records
    in five-minute bursts is not thereby full of anomalies, and the
    question worth asking of one of its recordings is whether it is unlike
    its siblings.
    """
    anomalies = []
    for key, label in UNIT_METRICS:
        per_group = _unit_values(catalog, unit_group, key)
        for group_value, values in sorted(per_group.items()):
            if len(values) < MIN_UNITS_FOR_MEDIAN:
                continue
            centre, sigma = unit_sigma(values)
            for row in (catalog or {}).get("rows") or []:
                value = row.get(key)
                if unit_group.get(row.get("label")) != group_value or value is None:
                    continue
                distance = _z(float(value), centre, sigma)
                if distance is None or abs(distance) < thresholds.outlier_sigma:
                    continue
                anomalies.append(
                    {
                        "unit": row["label"],
                        "group": group_value,
                        "metric": key,
                        "metric_label": label,
                        "value": float(value),
                        "group_median": centre,
                        "sigma": sigma,
                        "z": _round(distance),
                    }
                )
    anomalies.sort(key=lambda item: -abs(item["z"] or 0.0))
    return anomalies


def diverging(fleet: dict[str, Any]) -> list[dict[str, Any]]:
    """Every (group, channel) pair whose divergence is worth reporting,
    worst first. Read by fleet_checks.py and by the PDF, so the two cannot
    disagree about which pairs count."""
    rows = []
    for channel in fleet.get("channels") or []:
        for entry in channel.get("groups") or []:
            if entry.get("severity") == "stable":
                continue
            rows.append(
                {
                    "group": entry["value"],
                    "column": channel["column"],
                    "psi": entry["psi"],
                    "psi_aggregate": entry.get("psi_aggregate"),
                    "severity": entry["severity"],
                    "mean": entry.get("mean"),
                    "mean_z": entry.get("mean_z"),
                    "fleet_mean": channel.get("fleet_mean"),
                    "group_mean_median": channel.get("group_mean_median"),
                }
            )
    rows.sort(key=lambda row: -row["psi"])
    return rows
