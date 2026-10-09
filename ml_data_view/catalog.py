"""One row per UNIT of analysis — the catalogue, and its group rollups.

WHAT A UNIT IS HERE IS NOT DECIDED HERE
It is whatever hierarchy.py's mapping already said: a series and its
ordered chunks where a series level was marked, and otherwise a single
file — which is how a flat folder gets a row per file and a lone CSV gets
exactly one row. This module never asks what shape the input had; it is
handed a list of units and describes them. That is the whole reason a
"fleet of recordings" and "a folder of three parquet files" produce the
same table with different numbers in it.

WHAT IS IN A ROW, AND WHERE EACH FIGURE COMES FROM
  chunk count, rows, size on disk, channel count — metadata and cached
  per-file statistics. A row count is the sum of its chunks' cached row
  counts; a size is a stat(); a channel count is read off the cached
  schema. None of it reads column data.

  start, end, duration, rate, gaps — the merged per-unit time-series
  artifact (timeseries.py), which was itself computed per chunk and
  cached. ABSENT, not zero, when no timestamp column was detected: a
  catalogue of a folder with no time axis is still a catalogue, and a
  duration column full of dashes says so more honestly than one full of
  zeroes.

  health — the worst severity among the findings that name this unit,
  with a count. Attached separately (`attach_health`) because findings run
  after the chart build that produces everything else, and a status
  computed before the checks ran would be a guess.

Everything here is a pure function over dicts the app already has, with
one exception stated in `collect_units`: the size column costs one stat()
per file, which is metadata, not a read.
"""
from __future__ import annotations

import csv
import io
from pathlib import Path
from typing import Any, Callable

# Worst-first, which is also the order a rollup reads in.
SEVERITY_ORDER = ("critical", "warning", "info", "ok")

# The columns a catalogue row carries, in table order. Declared as data
# rather than spelled out in the frontend and again in the CSV writer and
# again in the PDF: three lists of column headings is three chances for
# them to disagree about what the fourth column is.
#
# `kind` is how a value is FORMATTED, not what it means — the frontend and
# report.py both read it, so a duration renders as a duration in each.
CATALOG_COLUMNS: list[dict[str, str]] = [
    {"key": "unit_id", "label": "", "kind": "text"},  # label filled from the mapping
    {"key": "chunk_count", "label": "Chunks", "kind": "int"},
    {"key": "rows", "label": "Rows", "kind": "int"},
    {"key": "start", "label": "Start", "kind": "time"},
    {"key": "end", "label": "End", "kind": "time"},
    {"key": "duration", "label": "Duration", "kind": "duration"},
    {"key": "hz", "label": "Rate", "kind": "rate"},
    {"key": "gap_count", "label": "Gaps", "kind": "int"},
    {"key": "gap_share", "label": "Gap %", "kind": "percent"},
    {"key": "channel_count", "label": "Channels", "kind": "int"},
    {"key": "bytes", "label": "Size", "kind": "bytes"},
    {"key": "health", "label": "Health", "kind": "health"},
]

# Columns that only mean anything with a time axis. Dropped from the table
# when no timestamp column was detected, rather than shown empty.
TIME_COLUMNS = ("start", "end", "duration", "hz", "gap_count", "gap_share")


def unit_label_key(key: tuple[str, ...] | list[str]) -> str:
    """The one identifier a unit is known by across every module.

    Group values then the unit id, joined — the same string
    file_profile.build_timeseries_payload keys its units dict on and the
    same one a finding names. Built in one place so the catalogue can join
    a finding to a row by equality rather than by parsing.
    """
    return " / ".join(str(part) for part in key)


def collect_units(
    unit_groups: list[tuple[tuple[str, ...], list[Path]]],
    *,
    row_counts: dict[str, int] | None = None,
    schemas: dict[str, dict[str, str]] | None = None,
    numeric_dtypes: frozenset[str] | set[str] = frozenset(),
    size_of: Callable[[Path], int | None] | None = None,
) -> list[dict[str, Any]]:
    """Turn (key, chunk paths) pairs into catalogue unit descriptors.

    `row_counts` and `schemas` are the cached per-file load sections keyed
    by path — already in hand wherever this is called from, so a row count
    is a sum of numbers rather than a scan. A file the cache has nothing
    for contributes no rows and is COUNTED as unknown rather than as zero:
    a catalogue that quietly reports a missing figure as 0 is worse than
    one that reports it as unknown.

    `size_of` is the one filesystem touch in this module — a stat() per
    file. Injectable so a test can describe a folder that is not on disk.
    """
    row_counts = row_counts or {}
    schemas = schemas or {}
    if size_of is None:
        size_of = _file_size

    units: list[dict[str, Any]] = []
    for key, paths in unit_groups:
        rows: int | None = 0
        for path in paths:
            count = row_counts.get(str(path))
            if count is None:
                rows = None
                break
            rows += int(count)
        sizes = [size_of(path) for path in paths]
        known = [size for size in sizes if size is not None]
        schema = next(
            (schemas[str(path)] for path in paths if str(path) in schemas), {}
        )
        units.append(
            {
                "key": list(key),
                # Group values then the unit id: the same split every other
                # module makes, so a caller never has to know how deep the
                # tree was.
                "group_values": list(key[:-1]),
                "unit_id": str(key[-1]) if key else "",
                "files": [str(path) for path in paths],
                "chunk_count": len(paths),
                "rows": rows,
                "bytes": sum(known) if len(known) == len(sizes) else None,
                "channel_count": sum(
                    1 for dtype in schema.values() if dtype in numeric_dtypes
                )
                or None,
            }
        )
    return units


def _file_size(path: Path) -> int | None:
    try:
        return Path(path).stat().st_size
    except OSError:
        # A file that vanished between discovery and here has no size, and
        # inventing 0 would make a group's total silently wrong.
        return None


def build_catalog(
    units: list[dict[str, Any]],
    *,
    timeseries: dict[str, Any] | None = None,
    labels: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """One row per unit, plus a summary row per group.

    `labels` carries the MAPPING's words — `group_labels`, `unit_label`,
    `group_columns`, `unit_column`, `root_label`. Defaults are generic
    ("Group", "File", "All"): nothing in this module knows what a system
    or a recording is, and the words a user sees are their own.
    """
    labels = dict(labels or {})
    group_labels = [str(text) for text in (labels.get("group_labels") or [])]
    unit_label = str(labels.get("unit_label") or "File")
    ts_units = (timeseries or {}).get("units") or {}
    has_time = bool(ts_units)

    rows = [_row(unit, ts_units.get(unit_label_key(unit["key"])) or {}) for unit in units]
    columns = _columns(unit_label, has_time)

    return {
        "available": bool(rows),
        "unit_label": unit_label,
        "group_labels": group_labels,
        "group_columns": [str(name) for name in (labels.get("group_columns") or [])],
        "unit_column": str(labels.get("unit_column") or "unit_id"),
        "root_label": str(labels.get("root_label") or "All"),
        "has_time": has_time,
        "has_groups": bool(group_labels),
        "timestamp_column": (timeseries or {}).get("timestamp_column"),
        "columns": columns,
        "rows": rows,
        "groups": group_summaries(rows, group_labels),
        "totals": _totals(rows),
    }


def _columns(unit_label: str, has_time: bool) -> list[dict[str, str]]:
    columns = []
    for column in CATALOG_COLUMNS:
        if not has_time and column["key"] in TIME_COLUMNS:
            continue
        entry = dict(column)
        if entry["key"] == "unit_id":
            entry["label"] = unit_label
        columns.append(entry)
    return columns


def _row(unit: dict[str, Any], ts_unit: dict[str, Any]) -> dict[str, Any]:
    """One catalogue row: the unit's own metadata, plus its time-series
    merge where there is one.

    `gap_count` counts every gap the merge reported, INCLUDING the ones
    that land on a chunk boundary — those are also counted separately as
    `missing_chunks`, because a gap where a file should have been is a
    different claim from a gap inside one, and a reader comparing two rows
    wants both numbers rather than a total that hides the distinction.
    """
    gaps = ts_unit.get("gaps") or []
    missing = sum(1 for gap in gaps if gap.get("at_boundary"))
    return {
        **{k: unit[k] for k in ("key", "group_values", "unit_id", "files", "chunk_count")},
        "label": unit_label_key(unit["key"]),
        # The time-series merge counted the rows it actually read; the
        # cached load section counted every row in the file. They agree,
        # and the cached one is present even with no time column, so it
        # wins and the other is not consulted.
        "rows": unit.get("rows"),
        "bytes": unit.get("bytes"),
        "channel_count": len(ts_unit.get("channels") or {}) or unit.get("channel_count"),
        "start": ts_unit.get("first_ts"),
        "end": ts_unit.get("last_ts"),
        "duration": ts_unit.get("duration"),
        "hz": ts_unit.get("hz"),
        "gap_count": len(gaps) if ts_unit else None,
        "gap_share": ts_unit.get("gap_share"),
        "issues": {
            "missing_chunks": missing,
            "gaps": len(gaps) - missing,
            "overlaps": len(ts_unit.get("overlaps") or []),
            "backwards": int(ts_unit.get("backwards") or 0),
            "duplicates": int(ts_unit.get("duplicates") or 0),
        }
        if ts_unit
        else {},
        # Filled in by attach_health once the checks have run. "unknown"
        # rather than "ok": nothing has looked yet, and a green tick before
        # anything looked is the one answer that would be a lie.
        "health": {"status": "unknown", "count": 0},
    }


def _totals(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "unit_count": len(rows),
        "chunk_count": sum(int(row["chunk_count"]) for row in rows),
        "rows": _sum_or_none(row.get("rows") for row in rows),
        "bytes": _sum_or_none(row.get("bytes") for row in rows),
        "duration": _sum_or_none(row.get("duration") for row in rows),
    }


def _sum_or_none(values) -> Any:
    """Sum, unless anything is unknown — in which case so is the total.

    A total assembled from some of its parts is not a smaller total, it is
    a wrong one, and a catalogue whose figures cannot be trusted to add up
    is worse than one that admits a gap.
    """
    collected = list(values)
    if not collected or any(value is None for value in collected):
        return None
    return sum(collected)


def group_summaries(
    rows: list[dict[str, Any]], group_labels: list[str]
) -> list[dict[str, Any]]:
    """One summary per group: how many units, how much data, how healthy.

    Grouped by the OUTERMOST group value only. Deeper levels are still in
    every row's `group_values` and the table can group by them; a summary
    row per combination of four levels is a second table, not a summary.
    """
    if not group_labels:
        return []
    summaries: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        values = row.get("group_values") or []
        if not values:
            continue
        summaries.setdefault(str(values[0]), []).append(row)
    return [
        {
            "value": value,
            "label": group_labels[0],
            "unit_count": len(members),
            "chunk_count": sum(int(row["chunk_count"]) for row in members),
            "rows": _sum_or_none(row.get("rows") for row in members),
            "bytes": _sum_or_none(row.get("bytes") for row in members),
            "duration": _sum_or_none(row.get("duration") for row in members),
            "health": health_breakdown(members),
        }
        for value, members in sorted(summaries.items())
    ]


def health_breakdown(rows: list[dict[str, Any]]) -> dict[str, int]:
    """Count of units per health status, all four keys always present —
    the UI draws a stable row of counts, zeroes included."""
    breakdown = dict.fromkeys(SEVERITY_ORDER, 0)
    breakdown["unknown"] = 0
    for row in rows:
        status = (row.get("health") or {}).get("status") or "unknown"
        breakdown[status] = breakdown.get(status, 0) + 1
    return breakdown


def attach_health(
    catalog: dict[str, Any], findings: list[dict[str, Any]], *, scanned: bool = True
) -> dict[str, Any]:
    """Fold the findings that NAME a unit into that unit's row.

    A finding carries the unit it is about (`Finding.unit`), so this is a
    lookup by equality rather than a parse of the detail text. Findings
    that name no unit — a column is mostly null, the dataset is empty —
    belong to the dataset, not to a row, and are deliberately not counted
    here: a catalogue that marks all nineteen recordings unhealthy because
    one column has a high null rate has stopped distinguishing them.

    `scanned=False` leaves every status "unknown", which is what the tab
    shows before the checks have run.
    """
    by_unit: dict[str, list[dict[str, Any]]] = {}
    for finding in findings or []:
        unit = str(finding.get("unit") or "")
        if unit:
            by_unit.setdefault(unit, []).append(finding)

    rows = []
    for row in catalog.get("rows") or []:
        found = by_unit.get(row["label"], [])
        rows.append({**row, "health": _health(found, scanned)})
    updated = {**catalog, "rows": rows}
    updated["groups"] = group_summaries(rows, catalog.get("group_labels") or [])
    updated["health"] = health_breakdown(rows)
    return updated


def _health(findings: list[dict[str, Any]], scanned: bool) -> dict[str, Any]:
    if not scanned:
        return {"status": "unknown", "count": 0}
    if not findings:
        return {"status": "ok", "count": 0}
    severities = {str(f.get("severity") or "info") for f in findings}
    worst = next((name for name in SEVERITY_ORDER if name in severities), "info")
    return {"status": worst, "count": len(findings)}


def catalog_csv(catalog: dict[str, Any]) -> str:
    """The catalogue as CSV — every row, every column that is on screen.

    The group values become their own leading columns rather than being
    folded into one label, because a spreadsheet's whole point is being
    able to pivot on them.
    """
    output = io.StringIO(newline="")
    writer = csv.writer(output, lineterminator="\n")
    group_labels = catalog.get("group_labels") or []
    columns = catalog.get("columns") or []
    writer.writerow(
        list(group_labels) + [column["label"] for column in columns]
    )
    for row in catalog.get("rows") or []:
        values = list(row.get("group_values") or [])
        values += [""] * (len(group_labels) - len(values))
        for column in columns:
            values.append(_csv_value(row, column["key"]))
        writer.writerow(values)
    return output.getvalue()


def _csv_value(row: dict[str, Any], key: str) -> Any:
    """Raw numbers, not the table's formatting.

    A CSV is read by a spreadsheet, and "1m 23.4s" in a duration column is
    a string it cannot sum. The one exception is health, which is a word
    on screen and a word here.
    """
    if key == "health":
        health = row.get("health") or {}
        count = int(health.get("count") or 0)
        status = health.get("status") or "unknown"
        return f"{status} ({count})" if count else status
    value = row.get(key)
    return "" if value is None else value
