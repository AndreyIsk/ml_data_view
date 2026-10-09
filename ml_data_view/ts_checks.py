"""Time-series health checks, as ordinary members of the Findings registry.

Every check here is a pure function over the two dicts the app already has
— the same rule the rest of the instant catalogue follows. The reading was
done during the chart build (timeseries.py, cached per chunk); this module
only reads the merged result and decides what is worth saying.

APPLICABILITY, WHICH IS THE POINT
Three different preconditions live here, and none of them is "a folder
that looks like a fleet":

  a detected TIMESTAMP COLUMN   -> the timestamp and channel checks. A
                                   single CSV with a time column gets
                                   every one of them.
  a SERIES level as well        -> the chunk-boundary checks. A unit that
                                   is one file has no boundary between
                                   chunks, so "is a chunk missing" is not
                                   a question that can be asked of it, let
                                   alone answered with a green tick.

A check whose precondition is absent reports `skipped`, never `done` —
see findings.run_instant_checks.

SEVERITY
A missing chunk, an overlap and time running backwards are CRITICAL:
each one means a figure computed from this data describes something that
did not happen. Gaps, flatlines and saturation are WARNINGS: the data is
real, some of it is missing or stuck. Jitter is INFO: sample intervals
vary in every real recording and the number is worth knowing, not acting
on.
"""
from __future__ import annotations

from typing import Any

from ml_data_view.findings import CheckContext, CheckSpec, Finding

# A unit contributes to the "typical duration" comparison only if there are
# at least this many to compare against. A median of two is not a median.
MIN_UNITS_FOR_DURATION = 3
# The out-of-range rule and the row floor it needs live in timeseries.py,
# beside the numbers they read, because the TIMELINE reads them too — and
# a check that imports findings.py cannot also be imported by a module
# findings.py does not know about. Re-exported here because this is where
# the check that reports them lives.
from ml_data_view.timeseries import (  # noqa: E402  (kept beside its users)
    MIN_ROWS_FOR_RANGE_HEURISTIC,
    out_of_range_extremes,
)


def _payload(context: CheckContext) -> dict[str, Any] | None:
    value = context.chart_data.get("timeseries")
    return value if isinstance(value, dict) else None


def _units(context: CheckContext) -> list[tuple[str, dict[str, Any]]]:
    payload = _payload(context)
    if not payload:
        return []
    return sorted((payload.get("units") or {}).items())


def has_timeseries(context: CheckContext) -> bool:
    """The precondition every check in this module shares: a timestamp
    column was detected and the analysis actually ran."""
    payload = _payload(context)
    return bool(payload and payload.get("units"))


def has_series(context: CheckContext) -> bool:
    """The extra precondition for the chunk-boundary checks."""
    payload = _payload(context)
    return bool(payload and payload.get("has_series"))


def _seconds(value: float | None) -> str:
    """A duration in the units a person reads. The timestamps are already
    in seconds by the time they reach here (timeseries normalises), so this
    is presentation only."""
    if value is None:
        return "unknown"
    if value < 1:
        return f"{value * 1000:.0f}ms"
    if value < 90:
        return f"{value:.2f}s"
    minutes, seconds = divmod(value, 60)
    return f"{int(minutes)}m {seconds:04.1f}s"


def _at(value: float | None) -> str:
    return "unknown" if value is None else f"t={value:.3f}"


# --- timestamp checks -----------------------------------------------------


def check_monotonic(context: CheckContext) -> list[Finding]:
    """Time running backwards.

    CRITICAL without qualification: every dt, every rate, every gap and
    every ordering downstream of this is computed from a sequence that is
    not a sequence. Nothing else in the tab is safe to read until it is
    explained.
    """
    findings = []
    for unit_id, unit in _units(context):
        backwards = unit.get("backwards") or 0
        if not backwards:
            continue
        findings.append(
            Finding(
                check_id="timestamp_monotonic",
                unit=unit_id,
                severity="critical",
                title="Timestamps go backwards",
                detail=(
                    f"{unit_id}: {backwards:,} step(s) where the next timestamp is EARLIER "
                    f"than the one before it, out of {unit.get('rows', 0):,} rows. Sample "
                    "rate, gaps and chunk order are all computed from this sequence."
                ),
                columns=[context.capability("timestamp_column") or ""],
            )
        )
    return findings


def check_duplicate_timestamps(context: CheckContext) -> list[Finding]:
    """Two rows claiming the same instant.

    A warning rather than critical: duplicates are ordinary in data joined
    from several sources, and they do not invalidate an ordering the way a
    backwards step does. The WORST RUN is reported because one duplicated
    instant is noise and four hundred consecutive ones is a stuck clock.
    """
    findings = []
    for unit_id, unit in _units(context):
        duplicates = unit.get("duplicates") or 0
        if not duplicates:
            continue
        worst = unit.get("worst_duplicate_run") or 0
        findings.append(
            Finding(
                check_id="duplicate_timestamps",
                unit=unit_id,
                severity="warning",
                title="Duplicate timestamps",
                detail=(
                    f"{unit_id}: {duplicates:,} row(s) share a timestamp with the row before "
                    f"them; the longest unbroken run is {worst:,}. A run this long usually "
                    "means a clock stopped rather than two sources arriving together."
                ),
                columns=[context.capability("timestamp_column") or ""],
            )
        )
    return findings


def check_sample_rate(context: CheckContext) -> list[Finding]:
    """Jitter, and rate changes within one unit.

    Two different statements. JITTER is the spread of dt as a share of its
    median — every real recording has some, so this is info. A RATE CHANGE
    is a stretch of the recording whose median dt is simply different,
    which is a warning: something reconfigured mid-recording, and any
    figure averaged across the whole thing is an average of two regimes.
    """
    payload = _payload(context) or {}
    thresholds = payload.get("thresholds") or {}
    jitter_limit = float(thresholds.get("jitter_ratio") or 0.25)
    findings = []
    for unit_id, unit in _units(context):
        median = unit.get("dt_median")
        jitter = unit.get("jitter")
        if median and jitter is not None and jitter > jitter_limit:
            findings.append(
                Finding(
                    check_id="sample_rate",
                    unit=unit_id,
                    severity="info",
                    title="Irregular sample interval",
                    detail=(
                        f"{unit_id}: median interval {_seconds(median)} "
                        f"({unit.get('hz', 0):.1f}Hz), spread {jitter:.0%} of the median. "
                        "Above the configured jitter threshold, so intervals vary more than "
                        "a steady rate would explain."
                    ),
                    columns=[context.capability("timestamp_column") or ""],
                )
            )
        segments = unit.get("rate_segments") or []
        if segments:
            first = segments[0]
            findings.append(
                Finding(
                    check_id="sample_rate",
                    unit=unit_id,
                    severity="warning",
                    title="Sample rate changes mid-recording",
                    detail=(
                        f"{unit_id}: {len(segments)} stretch(es) run at a different rate from "
                        f"the unit's {_seconds(median)}. The first starts at "
                        f"{_at(first.get('t0'))} at {_seconds(first.get('dt_median'))} per "
                        "sample. Anything averaged across the whole unit averages two regimes."
                    ),
                    columns=[context.capability("timestamp_column") or ""],
                )
            )
    return findings


def check_time_gaps(context: CheckContext) -> list[Finding]:
    """Stretches with no data at all.

    Only gaps INSIDE a chunk are reported here — a gap that falls exactly
    on a chunk boundary is a different claim and has its own check, because
    the rows are probably not late, a whole file is probably absent.
    """
    payload = _payload(context) or {}
    factor = float((payload.get("thresholds") or {}).get("gap_factor") or 5.0)
    findings = []
    for unit_id, unit in _units(context):
        gaps = [gap for gap in (unit.get("gaps") or []) if not gap.get("at_boundary")]
        if not gaps:
            continue
        worst = gaps[0]
        share = unit.get("gap_share")
        findings.append(
            Finding(
                check_id="time_gaps",
                unit=unit_id,
                severity="warning",
                title="Gaps in the recording",
                detail=(
                    f"{unit_id}: {len(gaps)} gap(s) longer than {factor:g}x the median "
                    f"interval, totalling {_seconds(unit.get('gap_total'))}"
                    + (f" ({share:.1%} of the recording)" if share is not None else "")
                    + f". The longest is {_seconds(worst['duration'])} at {_at(worst['at'])}."
                ),
                columns=[context.capability("timestamp_column") or ""],
            )
        )
    return findings


def check_short_unit(context: CheckContext) -> list[Finding]:
    """A unit much shorter than its siblings.

    Relative to the MEDIAN of the units actually loaded, not to an absolute
    number: how long a recording ought to be is a property of this dataset
    and nothing else. Silent with fewer than three units, because a median
    of two is not a median.
    """
    payload = _payload(context) or {}
    ratio = float((payload.get("thresholds") or {}).get("short_unit_ratio") or 0.5)
    units = _units(context)
    durations = [
        (unit_id, unit.get("duration"))
        for unit_id, unit in units
        if unit.get("duration") is not None
    ]
    if len(durations) < MIN_UNITS_FOR_DURATION:
        return []
    ordered = sorted(value for _id, value in durations)
    median = ordered[len(ordered) // 2]
    if not median:
        return []
    findings = []
    for unit_id, duration in durations:
        if duration >= median * ratio:
            continue
        findings.append(
            Finding(
                check_id="short_unit",
                unit=unit_id,
                severity="warning",
                title="Recording is much shorter than its siblings",
                # Every finding in this module opens with "<unit>: " so a
                # reader — and the Health tab — can tell at a glance which
                # recording is being talked about.
                detail=(
                    f"{unit_id}: covers {_seconds(duration)}, against a median of "
                    f"{_seconds(median)} across the {len(durations)} loaded. "
                    "A truncated recording looks complete in every other figure here."
                ),
                columns=[],
            )
        )
    return findings


# --- chunk-boundary checks (need a series) --------------------------------


def check_missing_chunk(context: CheckContext) -> list[Finding]:
    """A gap that lands exactly on a chunk boundary.

    CRITICAL, and separate from an ordinary gap, because the two have
    different causes and different fixes: rows missing inside a chunk are
    a sensor or a writer hiccup, while a hole exactly between two files is
    almost always a file that never arrived.
    """
    findings = []
    for unit_id, unit in _units(context):
        seams = [gap for gap in (unit.get("gaps") or []) if gap.get("at_boundary")]
        if not seams:
            continue
        worst = max(seams, key=lambda gap: gap["duration"])
        findings.append(
            Finding(
                check_id="missing_chunk",
                unit=unit_id,
                severity="critical",
                title="Possible missing chunk",
                detail=(
                    f"{unit_id}: {len(seams)} gap(s) fall exactly on a chunk boundary. The "
                    f"largest is {_seconds(worst['duration'])} between "
                    f"{worst.get('after_chunk')} and {worst.get('next_chunk')} — the shape of "
                    "a file that never arrived rather than of rows that were late."
                ),
                columns=[],
            )
        )
    return findings


def check_chunk_overlap(context: CheckContext) -> list[Finding]:
    """Two chunks covering the same instants.

    CRITICAL: every row in the overlap is counted once per chunk that holds
    it, so row counts, distributions and rates are all inflated by an
    amount nothing on screen reveals.
    """
    findings = []
    for unit_id, unit in _units(context):
        overlaps = unit.get("overlaps") or []
        if not overlaps:
            continue
        worst = max(overlaps, key=lambda item: item.get("overlap") or 0)
        findings.append(
            Finding(
                check_id="chunk_overlap",
                unit=unit_id,
                severity="critical",
                title="Chunks overlap in time",
                detail=(
                    f"{unit_id}: {len(overlaps)} boundary/boundaries overlap. "
                    f"{worst.get('next_chunk')} starts {_seconds(worst.get('overlap'))} before "
                    f"{worst.get('after_chunk')} ends. Rows in the overlap are counted once "
                    "per chunk that holds them."
                ),
                columns=[],
            )
        )
    return findings


# --- channel checks -------------------------------------------------------


def check_channel_nan_runs(context: CheckContext) -> list[Finding]:
    """A channel that stopped reporting for a stretch.

    Distinct from the existing missingness checks, which count nulls
    without caring where they were: a channel that is 5% missing scattered
    is a noisy sensor, and one that is 5% missing in a single block is a
    sensor that died and came back.
    """
    payload = _payload(context) or {}
    limit = int((payload.get("thresholds") or {}).get("flatline_rows") or 500)
    findings = []
    for unit_id, unit in _units(context):
        for name, channel in sorted((unit.get("channels") or {}).items()):
            longest = channel.get("longest_nan") or 0
            if longest < limit:
                continue
            findings.append(
                Finding(
                    check_id="channel_nan_run",
                    unit=unit_id,
                    severity="warning",
                    title="Channel stops reporting for a stretch",
                    detail=(
                        f"{unit_id}: {name} has an unbroken run of {longest:,} missing values "
                        f"starting at {_at(channel.get('longest_nan_at'))} "
                        f"({channel.get('nan_share', 0):.1%} of the channel is missing in "
                        "total). A block like this is a sensor that dropped out, not noise."
                    ),
                    columns=[name],
                )
            )
    return findings


def check_channel_flatline(context: CheckContext) -> list[Finding]:
    """A channel that stopped changing — a stuck sensor.

    A fully constant channel is reported as such rather than as a long
    flatline: "this never changed at all" is a different sentence from
    "this stopped changing for a while", and the reader acts on them
    differently.
    """
    payload = _payload(context) or {}
    limit = int((payload.get("thresholds") or {}).get("flatline_rows") or 500)
    findings = []
    for unit_id, unit in _units(context):
        for name, channel in sorted((unit.get("channels") or {}).items()):
            longest = channel.get("longest_flat") or 0
            rows = channel.get("rows") or 0
            if longest < limit:
                continue
            constant = rows and longest >= rows - 1
            findings.append(
                Finding(
                    check_id="channel_flatline",
                    unit=unit_id,
                    severity="warning",
                    title="Channel never changes" if constant else "Channel flatlines",
                    detail=(
                        f"{unit_id}: {name} holds the same value for "
                        + (
                            f"the whole unit ({rows:,} rows)."
                            if constant
                            else f"{longest:,} consecutive rows from "
                            f"{_at(channel.get('longest_flat_at'))}, out of {rows:,}."
                        )
                        + " A signal that does not move is usually a sensor that stopped."
                    ),
                    columns=[name],
                )
            )
    return findings


def check_channel_saturation(context: CheckContext) -> list[Finding]:
    """Repeated pinning at a channel's own extreme.

    The share is EXACT, not sampled: a chunk whose own minimum sits above
    the global one holds no value equal to the global one, so summing the
    counts from the chunks that do reach it is the whole answer (see
    timeseries._merge_channels).
    """
    findings = []
    for unit_id, unit in _units(context):
        for name, channel in sorted((unit.get("channels") or {}).items()):
            for edge, share_key, count_key, value_key in (
                ("maximum", "max_share", "max_count", "max"),
                ("minimum", "min_share", "min_count", "min"),
            ):
                share = channel.get(share_key) or 0.0
                if share < 0.01 or (channel.get(count_key) or 0) < 2:
                    continue
                if channel.get("longest_flat", 0) >= (channel.get("rows") or 0) - 1:
                    continue  # a constant channel is already reported as one
                findings.append(
                    Finding(
                        check_id="channel_saturation",
                        unit=unit_id,
                        severity="warning",
                        title="Channel pins at its limit",
                        detail=(
                            f"{unit_id}: {name} sits exactly at its {edge} "
                            f"({channel.get(value_key)}) for {channel.get(count_key):,} rows "
                            f"({share:.1%}). Repeated pinning at one value is what a sensor "
                            "does when the real value is off the end of its range."
                        ),
                        columns=[name],
                    )
                )
    return findings


def check_channel_out_of_range(context: CheckContext) -> list[Finding]:
    """Values outside what the channel is supposed to hold.

    Two rules, and they are labelled differently because they are not
    equally trustworthy. With a DATA DICTIONARY configured, this is an
    exact count against a documented range and is critical. Without one it
    is a HEURISTIC on the extremes — how far the largest value sits beyond
    the inter-percentile spread — which says so in its own text, because a
    number nobody can check should not be presented as one they can.
    """
    payload = _payload(context) or {}
    thresholds = payload.get("thresholds") or {}
    ranges = thresholds.get("ranges") or {}
    factor = float(thresholds.get("range_spread_factor") or 10.0)
    findings = []
    for unit_id, unit in _units(context):
        for name, channel in sorted((unit.get("channels") or {}).items()):
            documented = ranges.get(name)
            if documented:
                count = channel.get("out_of_range") or 0
                if not count:
                    continue
                findings.append(
                    Finding(
                        check_id="channel_out_of_range",
                        unit=unit_id,
                        severity="critical",
                        title="Values outside the documented range",
                        detail=(
                            f"{unit_id}: {name} has {count:,} value(s) outside its documented "
                            f"range [{documented[0]}, {documented[1]}]. Observed "
                            f"{channel.get('min')} to {channel.get('max')}."
                        ),
                        columns=[name],
                    )
                )
                continue

            # The FIRST extreme only: a channel that reaches both ends is
            # one channel with a range problem, and two findings saying so
            # would double-count it in the severity summary. The timeline
            # marks both, because a mark is a place rather than a claim.
            extremes = out_of_range_extremes(channel, None, factor)
            if not extremes:
                continue
            worst = extremes[0]
            extreme, distance = worst["value"], worst["distance"]
            low, high = channel.get("p001"), channel.get("p999")
            findings.append(
                Finding(
                    check_id="channel_out_of_range",
                    unit=unit_id,
                    severity="warning",
                    title="Extreme value far outside the usual range",
                    detail=(
                        f"{unit_id}: {name} reaches {extreme}, which is {distance:.0f}x the "
                        f"p0.1-p99.9 spread beyond it (p0.1 {low:.4g}, p99.9 {high:.4g}). "
                        "HEURISTIC — there is no documented range for this channel, so this "
                        "is a shape-of-the-data guess, not a rule it broke. Configure a range "
                        "in Settings to make it an exact check."
                    ),
                    columns=[name],
                )
            )
    return findings


TIMESERIES_CHECKS: list[CheckSpec] = [
    CheckSpec(
        "timestamp_monotonic",
        "Timestamp monotonicity",
        "instant",
        check_monotonic,
        applies=has_timeseries,
        requires="a timestamp column",
    ),
    CheckSpec(
        "duplicate_timestamps",
        "Duplicate timestamps",
        "instant",
        check_duplicate_timestamps,
        applies=has_timeseries,
        requires="a timestamp column",
    ),
    CheckSpec(
        "sample_rate",
        "Sample rate",
        "instant",
        check_sample_rate,
        applies=has_timeseries,
        requires="a timestamp column",
    ),
    CheckSpec(
        "time_gaps",
        "Gaps in time",
        "instant",
        check_time_gaps,
        applies=has_timeseries,
        requires="a timestamp column",
    ),
    CheckSpec(
        "short_unit",
        "Recording duration",
        "instant",
        check_short_unit,
        applies=has_timeseries,
        requires="a timestamp column",
    ),
    CheckSpec(
        "missing_chunk",
        "Missing chunks",
        "instant",
        check_missing_chunk,
        applies=has_series,
        requires="a series of chunks",
    ),
    CheckSpec(
        "chunk_overlap",
        "Chunk overlap",
        "instant",
        check_chunk_overlap,
        applies=has_series,
        requires="a series of chunks",
    ),
    CheckSpec(
        "channel_nan_run",
        "Channel dropouts",
        "instant",
        check_channel_nan_runs,
        applies=has_timeseries,
        requires="a timestamp column",
    ),
    CheckSpec(
        "channel_flatline",
        "Channel flatlines",
        "instant",
        check_channel_flatline,
        applies=has_timeseries,
        requires="a timestamp column",
    ),
    CheckSpec(
        "channel_saturation",
        "Channel saturation",
        "instant",
        check_channel_saturation,
        applies=has_timeseries,
        requires="a timestamp column",
    ),
    CheckSpec(
        "channel_out_of_range",
        "Channel range",
        "instant",
        check_channel_out_of_range,
        applies=has_timeseries,
        requires="a timestamp column",
    ),
]
