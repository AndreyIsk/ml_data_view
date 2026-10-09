"""Directory structure as a DETECTED, CONFIRMED, OPTIONAL description.

Otter opens whatever it is pointed at. A single file, a flat folder, a
hive-partitioned tree, an arbitrary nest of directories nobody can name,
or a recorded fleet. This module describes the shape of what was opened —
it does not require one.

THE VOCABULARY IS GENERIC AND THE LABELS ARE THE USER'S
Nothing here knows what a "system" or a "recording" is. A directory level
gets one of three roles:

  GROUP   this level's directory names identify something the data is
          partitioned by — a vehicle, a day, a site. Any number of group
          levels, nested; zero is fine.
  SERIES  a directory at this level holds the ordered CHUNKS of one
          dataset. At most one series level; zero is fine.
  IGNORE  structure, not a role. Also the default.

Whatever the user calls those levels travels in the mapping as a LABEL and
is what the UI shows. "System" and "Recording" are one dataset's labels,
not concepts in the code; the defaults are "Group" and "Series".

THE UNIT IS WHAT LOADS AS ONE DATASET
With a series level, a unit is one series directory and its ordered
chunks. WITHOUT one, a unit is one file — which is precisely how the app
behaved before any of this existed, so a flat folder and a single file
fall out of the same model rather than needing a separate path. A chunk is
always one file, so a chunk is always what profile_cache.py keys on.

BOTH ROLES ARE OPTIONAL, AND SO IS THE WHOLE THING
  no roles at all      -> `detected` is False; the caller keeps its
                          schema-group behaviour, untouched.
  groups, no series    -> hive partitions: units are files, grouped.
  series, no groups    -> one flat run of chunks.
  groups and series    -> the full tree.

EVERYTHING HERE IS METADATA-ONLY
Roles come from directory names. Schemas, row counts and per-chunk time
bounds all come from the parquet footer through one `ParquetFile` open per
chunk — the same budget folder discovery already works to (see
PROJECT_BRIEF.md's architecture notes). Laying out a tree of thousands of
chunks reads no rows at all.
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# Only parquet files can be chunks of a series: ordering them rests on
# footer metadata — schema, row count and time bounds — and no other format
# here offers all three without reading data. Other formats still load;
# they just cannot form a series, which means their unit is the file.
CHUNK_SUFFIX = ".parquet"

ROLE_GROUP = "group"
ROLE_SERIES = "series"
ROLE_IGNORE = "ignore"
ROLES = (ROLE_GROUP, ROLE_SERIES, ROLE_IGNORE)

DEFAULT_GROUP_LABEL = "Group"
DEFAULT_SERIES_LABEL = "Series"
DEFAULT_ROOT_LABEL = "All"

# Hive-style partitioning: one `=` separating a non-empty key from a value.
# The only structural convention this module recognises without being told,
# because it is the only one that explicitly says "this level identifies
# something" rather than leaving it to be guessed from a name.
_HIVE = re.compile(r"^([^=/\\]+)=([^=/\\]*)$")

# Column names that mean "when", matched against whole words so `event_time`
# and `ts` hit but `optimistic` does not.
_TIME_NAME = re.compile(
    r"(^|[_\-.])(time|times|timestamp|timestamps|ts|datetime|date_time|epoch|utc)($|[_\-.])",
    re.IGNORECASE,
)

# Epoch magnitudes as log10. Seconds since 1970 are ~1.7e9 and each finer
# unit adds three orders of magnitude, so the gaps dwarf any ambiguity.
_UNIT_BY_MAGNITUDE = (
    (11.5, "s"),
    (14.5, "ms"),
    (17.5, "us"),
    (float("inf"), "ns"),
)
UNITS = ("s", "ms", "us", "ns")

# Derived columns that exist whenever a structure does, whatever its shape.
# The GROUP columns are named after their level (a hive key, or the level's
# label) and so cannot be listed here — see DatasetStructure.derived_columns.
UNIT_COLUMN = "unit_id"
CHUNK_INDEX_COLUMN = "chunk_index"
CHUNK_FILE_COLUMN = "chunk_file"
FIXED_DERIVED_COLUMNS = (UNIT_COLUMN, CHUNK_INDEX_COLUMN, CHUNK_FILE_COLUMN)


# --- natural ordering ----------------------------------------------------


def natural_key(text: str) -> tuple:
    """Sort key that reads runs of digits as numbers.

    `part-2` before `part-10`, which plain lexicographic ordering gets
    backwards — and chunk names are overwhelmingly of that shape.
    """
    return tuple(
        int(piece) if piece.isdigit() else piece.lower()
        for piece in re.split(r"(\d+)", text)
        if piece != ""
    )


def _slug(text: str) -> str:
    """A label turned into something usable as a column name."""
    cleaned = re.sub(r"[^0-9a-zA-Z_]+", "_", text).strip("_").lower()
    return cleaned or "group"


def _id_column(label: str) -> str:
    """A level's LABEL turned into an identifier column name.

    "System" becomes `system_id`, "Recording" becomes `recording_id` — so
    labelling the levels of the motivating tree produces exactly the column
    names that tree's owner would have written by hand, without any of
    those words existing in the code. A label that already ends in `_id`
    is left alone rather than becoming `thing_id_id`.
    """
    slug = _slug(label)
    return slug if slug.endswith("_id") else f"{slug}_id"


# --- the model -----------------------------------------------------------


@dataclass
class Chunk:
    """One file, and everything its footer said about it."""

    path: Path
    row_count: int
    schema_signature: tuple[tuple[str, str], ...]
    # From the timestamp column's footer statistics, when there is one.
    # None is a real state: a file written without column statistics cannot
    # be time-ordered and says so rather than being guessed at.
    first_timestamp: float | None = None
    last_timestamp: float | None = None
    index: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "file": str(self.path),
            "name": self.path.name,
            "row_count": self.row_count,
            "index": self.index,
            "first_timestamp": self.first_timestamp,
            "last_timestamp": self.last_timestamp,
        }


@dataclass
class ExcludedChunk:
    path: Path
    reason: str

    def to_dict(self) -> dict[str, Any]:
        return {"file": str(self.path), "name": self.path.name, "reason": self.reason}


@dataclass
class Unit:
    """What loads as one dataset.

    A series directory and its ordered chunks, or — when no series level
    was marked — a single file, which is a one-chunk unit. That is not a
    special case bolted on: it is the same structure with the series level
    absent, which is why a flat folder needs no separate code path.
    """

    key: tuple[str, ...]  # one value per group level, outermost first
    unit_id: str
    chunks: list[Chunk]
    excluded: list[ExcludedChunk] = field(default_factory=list)
    schema_signature: tuple[tuple[str, str], ...] = ()
    # "name" when natural filename order was used as-is, "timestamp" when
    # footer time bounds disagreed with it and won.
    order_source: str = "name"
    order_conflict: bool = False

    @property
    def row_count(self) -> int:
        return sum(chunk.row_count for chunk in self.chunks)

    @property
    def files(self) -> list[Path]:
        return [chunk.path for chunk in self.chunks]

    @property
    def first_timestamp(self) -> float | None:
        stamps = [c.first_timestamp for c in self.chunks if c.first_timestamp is not None]
        return min(stamps) if stamps else None

    @property
    def last_timestamp(self) -> float | None:
        stamps = [c.last_timestamp for c in self.chunks if c.last_timestamp is not None]
        return max(stamps) if stamps else None

    def to_dict(self) -> dict[str, Any]:
        return {
            "key": list(self.key),
            "unit_id": self.unit_id,
            "chunks": [chunk.to_dict() for chunk in self.chunks],
            "excluded": [item.to_dict() for item in self.excluded],
            "chunk_count": len(self.chunks),
            "row_count": self.row_count,
            "order_source": self.order_source,
            "order_conflict": self.order_conflict,
            "first_timestamp": self.first_timestamp,
            "last_timestamp": self.last_timestamp,
        }


@dataclass
class StructureMapping:
    """The user's confirmed answer for one root, and what it was about.

    `labels` is what the UI shows for each level — the user's own words for
    their own data. `fingerprint` describes the SHAPE of the tree, so
    "never asked again unless the structure changes" can be true without
    "never asked again" being true.
    """

    root: str
    roles: dict[int, str]
    fingerprint: str
    labels: dict[int, str] = field(default_factory=dict)
    timestamp_column: str | None = None
    timestamp_unit: str | None = None
    root_label: str = DEFAULT_ROOT_LABEL

    def label_for(self, depth: int) -> str:
        explicit = (self.labels or {}).get(depth)
        if explicit:
            return explicit
        role = self.roles.get(depth)
        return DEFAULT_SERIES_LABEL if role == ROLE_SERIES else DEFAULT_GROUP_LABEL

    def group_depths(self) -> list[int]:
        return sorted(d for d, role in self.roles.items() if role == ROLE_GROUP)

    def series_depth(self) -> int | None:
        depths = sorted(d for d, role in self.roles.items() if role == ROLE_SERIES)
        # At most one series level. If a hand-edited mapping names several,
        # the OUTERMOST wins — a series inside a series is not a series, it
        # is a group whose members happen to be series.
        return depths[0] if depths else None

    def to_dict(self) -> dict[str, Any]:
        return {
            "root": self.root,
            # JSON object keys are strings; depths come back as ints.
            "roles": {str(depth): role for depth, role in sorted(self.roles.items())},
            "labels": {str(depth): text for depth, text in sorted((self.labels or {}).items())},
            "fingerprint": self.fingerprint,
            "timestamp_column": self.timestamp_column,
            "timestamp_unit": self.timestamp_unit,
            "root_label": self.root_label,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> StructureMapping:
        roles = {
            int(depth): role
            for depth, role in (data.get("roles") or {}).items()
            if role in ROLES
        }
        labels = {
            int(depth): str(text)
            for depth, text in (data.get("labels") or {}).items()
            if str(text).strip()
        }
        return cls(
            root=str(data.get("root") or ""),
            roles=roles,
            fingerprint=str(data.get("fingerprint") or ""),
            labels=labels,
            timestamp_column=data.get("timestamp_column") or None,
            timestamp_unit=data.get("timestamp_unit") or None,
            root_label=str(data.get("root_label") or DEFAULT_ROOT_LABEL),
        )


@dataclass
class LevelProposal:
    """One directory level, and what this module suggests it is."""

    depth: int
    role: str
    label: str
    examples: list[str]
    directory_count: int
    hive_key: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "depth": self.depth,
            "role": self.role,
            "label": self.label,
            "examples": self.examples[:5],
            "directory_count": self.directory_count,
            "hive_key": self.hive_key,
        }


@dataclass
class StructureProposal:
    """What the structure dialog shows. Never applied on its own."""

    root: str
    levels: list[LevelProposal]
    fingerprint: str
    detected: bool
    timestamp_candidates: list[dict[str, Any]] = field(default_factory=list)
    suggested_timestamp: str | None = None
    suggested_unit: str | None = None

    def roles(self) -> dict[int, str]:
        return {level.depth: level.role for level in self.levels}

    def labels(self) -> dict[int, str]:
        return {level.depth: level.label for level in self.levels}

    def to_dict(self) -> dict[str, Any]:
        return {
            "root": self.root,
            "detected": self.detected,
            "fingerprint": self.fingerprint,
            "levels": [level.to_dict() for level in self.levels],
            "timestamp_candidates": list(self.timestamp_candidates),
            "suggested_timestamp": self.suggested_timestamp,
            "suggested_unit": self.suggested_unit,
        }


@dataclass
class Scope:
    """What the app is currently looking at.

    A prefix of the group values, optionally ending at one unit. Depth is
    whatever the mapping has — the same query at every width, which is what
    lets every section take a scope without knowing the shape of the tree.
    """

    values: list[str] = field(default_factory=list)
    unit_id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {"values": list(self.values), "unit_id": self.unit_id}

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> Scope:
        data = data or {}
        values = [str(value) for value in (data.get("values") or [])]
        return cls(values=values, unit_id=data.get("unit_id") or None)


@dataclass
class DatasetStructure:
    """A confirmed mapping, applied to a tree."""

    root: Path
    units: list[Unit]
    mapping: StructureMapping
    timestamp_column: str | None = None
    timestamp_unit: str | None = None
    findings: list[dict[str, Any]] = field(default_factory=list)

    # --- shape questions the rest of the app asks -----------------------

    @property
    def group_depths(self) -> list[int]:
        return self.mapping.group_depths()

    @property
    def has_series(self) -> bool:
        """True when a level was marked series — the precondition for every
        chunk-boundary check. Without one, a unit is a single file and
        "the boundary between chunks" is not a thing that exists."""
        return self.mapping.series_depth() is not None

    @property
    def group_labels(self) -> list[str]:
        return [self.mapping.label_for(depth) for depth in self.group_depths]

    @property
    def unit_label(self) -> str:
        series = self.mapping.series_depth()
        return self.mapping.label_for(series) if series is not None else "File"

    def derived_columns(self) -> list[str]:
        """The virtual column names this structure contributes, in order.

        One per group level, then the unit column, then `chunk_index` and
        `chunk_file`. A level's name comes, in order of preference, from:

          1. a LABEL the user typed — "System" gives `system_id`, so
             labelling the motivating tree's levels produces exactly the
             column names its owner would have written by hand, with none
             of those words appearing in this code;
          2. its HIVE KEY — `sys_key=123` gives `sys_key`, which is
             already the name the data itself uses;
          3. a positional fallback — `group_1_id`.

        Names that would collide with a real data column are suffixed
        rather than shadowing it: a derived column must never quietly
        replace one the file actually contains.
        """
        existing = {name for name, _ in (self.units[0].schema_signature if self.units else ())}
        names: list[str] = []
        for position, depth in enumerate(self.group_depths, start=1):
            names.append(_unique(self._level_column(depth, position), existing, names))
        names.append(_unique(self._unit_column(), existing, names))
        for base in (CHUNK_INDEX_COLUMN, CHUNK_FILE_COLUMN):
            names.append(_unique(base, existing, names))
        return names

    def _level_column(self, depth: int, position: int) -> str:
        hive_key = _hive_key_for_depth(self, depth)
        label = (self.mapping.labels or {}).get(depth) or ""
        # A label that is just the hive key echoed back is not a choice —
        # propose_mapping fills it in as the default — so the key wins
        # there and only a label the user actually changed overrides it.
        if label and label != hive_key:
            return _id_column(label)
        if hive_key:
            return hive_key
        return f"group_{position}_id"

    def _unit_column(self) -> str:
        series = self.mapping.series_depth()
        label = (self.mapping.labels or {}).get(series) if series is not None else None
        if label and label != DEFAULT_SERIES_LABEL:
            return _id_column(label)
        return UNIT_COLUMN

    def unit(self, key: tuple[str, ...], unit_id: str) -> Unit | None:
        return next(
            (u for u in self.units if u.unit_id == unit_id and tuple(u.key) == tuple(key)), None
        )

    def values_at(self, prefix: list[str]) -> list[str]:
        """Distinct values of the next group level below `prefix`."""
        depth = len(prefix)
        if depth >= len(self.group_depths):
            return []
        seen = []
        for unit in self.units:
            if list(unit.key[:depth]) != list(prefix):
                continue
            value = unit.key[depth]
            if value not in seen:
                seen.append(value)
        return sorted(seen, key=natural_key)

    def schema_groups(self) -> list[dict[str, Any]]:
        """Units grouped by schema — INFORMATION, not the loading unit.

        Across units, schemas legitimately differ (a sensor was added
        between runs). Worth showing; it must not decide what loads, which
        is the whole reason the unit replaced the schema group.
        """
        groups: dict[tuple, list[Unit]] = {}
        for unit in self.units:
            groups.setdefault(unit.schema_signature, []).append(unit)
        ordered = sorted(groups.items(), key=lambda item: len(item[1]), reverse=True)
        return [
            {
                "columns": [{"name": name, "dtype": dtype} for name, dtype in signature],
                "units": [{"key": list(u.key), "unit_id": u.unit_id} for u in units],
                "unit_count": len(units),
            }
            for signature, units in ordered
        ]

    def to_dict(self) -> dict[str, Any]:
        return {
            "root": str(self.root),
            "units": [unit.to_dict() for unit in self.units],
            "unit_count": len(self.units),
            "chunk_count": sum(len(u.chunks) for u in self.units),
            "row_count": sum(u.row_count for u in self.units),
            "group_labels": self.group_labels,
            "unit_label": self.unit_label,
            "root_label": self.mapping.root_label,
            "has_series": self.has_series,
            "has_groups": bool(self.group_depths),
            "derived_columns": self.derived_columns(),
            "timestamp_column": self.timestamp_column,
            "timestamp_unit": self.timestamp_unit,
            "mapping": self.mapping.to_dict(),
            "schema_groups": self.schema_groups(),
            "findings": list(self.findings),
        }


def _unique(base: str, existing: set[str], taken: list[str]) -> str:
    name = base
    while name in existing or name in taken:
        name = f"{name}_"
    return name


def _hive_key_for_depth(structure: DatasetStructure, depth: int) -> str | None:
    """The hive key at a group level, when every value there has one."""
    keys = set()
    for unit in structure.units:
        if depth >= len(unit.key):
            return None
        key = _hive_key(unit.key[depth])
        if key is None:
            return None
        keys.add(key)
    return keys.pop() if len(keys) == 1 else None


# --- reading a chunk's footer --------------------------------------------


@dataclass
class _Footer:
    row_count: int
    signature: tuple[tuple[str, str], ...]
    stats: dict[str, tuple[Any, Any]]


# Footers already read this session, keyed by the same (path, size, mtime)
# identity the profile cache uses. Detection now reads every chunk's footer
# to decide whether a level is a series, and build_structure reads them all
# again a moment later — without this, confirming a mapping would cost a
# second pass over a tree of thousands of files for answers already in
# hand. Bounded, because this is a convenience within one discovery, not a
# second cache: profile_cache.py is the one that persists.
_FOOTER_MEMO: dict[tuple[str, int, int], "_Footer | None"] = {}
_FOOTER_MEMO_LIMIT = 20_000


def read_footer(path: Path) -> _Footer | None:
    """One ParquetFile open: schema, row count and per-column min/max.

    All three come from the footer, so this decodes no column data. Opened
    ONCE per chunk rather than once per question, because the footer read
    is the cost and asking three times pays it three times.

    None when the file cannot be read at all, which callers report as an
    exclusion rather than letting one bad file take the tree down.
    """
    try:
        stat = Path(path).stat()
        key = (str(path), int(stat.st_size), int(stat.st_mtime_ns))
    except OSError:
        return None
    if key in _FOOTER_MEMO:
        return _FOOTER_MEMO[key]
    footer = _read_footer_uncached(path)
    if len(_FOOTER_MEMO) >= _FOOTER_MEMO_LIMIT:
        _FOOTER_MEMO.clear()
    _FOOTER_MEMO[key] = footer
    return footer


def _read_footer_uncached(path: Path) -> _Footer | None:
    try:
        import pyarrow.parquet as pq

        parquet = pq.ParquetFile(path)
        metadata = parquet.metadata
        signature = tuple((field.name, str(field.type)) for field in parquet.schema_arrow)
        column_index = {name: index for index, (name, _type) in enumerate(signature)}
        stats: dict[str, tuple[Any, Any]] = {}
        for group_index in range(metadata.num_row_groups):
            row_group = metadata.row_group(group_index)
            for name, index in column_index.items():
                if index >= row_group.num_columns:
                    continue
                column = row_group.column(index)
                if column.path_in_schema != name:
                    continue
                statistics = column.statistics
                if statistics is None or not statistics.has_min_max:
                    continue
                low, high = statistics.min, statistics.max
                if name in stats:
                    previous_low, previous_high = stats[name]
                    low = min(previous_low, low)
                    high = max(previous_high, high)
                stats[name] = (low, high)
        return _Footer(int(metadata.num_rows), signature, stats)
    except Exception:
        return None


# --- timestamps -----------------------------------------------------------


def _is_numeric_arrow_type(type_name: str) -> bool:
    return type_name.startswith(("int", "uint", "float", "double"))


def _is_temporal_arrow_type(type_name: str) -> bool:
    return type_name.startswith(("timestamp", "date", "time"))


def timestamp_candidates(
    signature: tuple[tuple[str, str], ...], stats: dict[str, tuple[Any, Any]] | None = None
) -> list[dict[str, Any]]:
    """Columns that could be "when", best first.

    A real datetime column wins outright — the dtype IS the answer, and
    nothing about it has to be inferred. A numeric column qualifies on its
    NAME, and carries a unit guess when footer statistics are available to
    guess from.

    `stats` is optional so this works on a schema alone: a single CSV has no
    footer, and a time-series check has to be able to run on one.
    """
    stats = stats or {}
    temporal: list[dict[str, Any]] = []
    numeric: list[dict[str, Any]] = []
    for name, type_name in signature:
        low_high = stats.get(name)
        if _is_temporal_arrow_type(type_name):
            temporal.append({"column": name, "dtype": type_name, "kind": "temporal", "unit": None})
        elif _is_numeric_arrow_type(type_name) and _TIME_NAME.search(name):
            numeric.append(
                {
                    "column": name,
                    "dtype": type_name,
                    "kind": "numeric",
                    "unit": detect_unit(low_high[1]) if low_high else None,
                }
            )
    return temporal + numeric


def detect_timestamp_column(schema: Any) -> dict[str, Any] | None:
    """The best timestamp candidate in a polars schema, or None.

    Takes a polars schema rather than an arrow signature so every load path
    — single file, flat folder, structured tree — can call it with what it
    already has. This is the precondition every time-series check declares,
    which is what lets those checks run on a single CSV with a time column
    and stay silent on one without.
    """
    signature = tuple((name, _arrow_name(dtype)) for name, dtype in schema.items())
    candidates = timestamp_candidates(signature)
    return candidates[0] if candidates else None


def _arrow_name(dtype: Any) -> str:
    """A polars dtype rendered in the arrow-ish vocabulary the matchers
    above use, so one set of rules covers both sources."""
    name = str(dtype)
    lowered = name.lower()
    if lowered.startswith("datetime") or lowered.startswith("date") or lowered.startswith("time"):
        return "timestamp" if lowered.startswith("datetime") else lowered
    if lowered.startswith(("int", "uint", "float")):
        return lowered
    return lowered


def detect_unit(value: Any) -> str | None:
    """Epoch unit of a numeric timestamp, by magnitude.

    A suggestion the user confirms: getting it wrong mislabels an axis, it
    does not change a count.
    """
    try:
        magnitude = abs(float(value))
    except (TypeError, ValueError):
        return None
    if magnitude <= 0:
        return None
    order = math.log10(magnitude)
    for threshold, unit in _UNIT_BY_MAGNITUDE:
        if order < threshold:
            return unit
    return "ns"


def _timestamp_value(raw: Any) -> float | None:
    """A footer statistic as a sortable number. Only the ORDER matters, so
    both datetimes and plain numbers collapse to a float and no unit
    conversion is needed or done."""
    if raw is None:
        return None
    if isinstance(raw, (int, float)):
        return float(raw)
    method = getattr(raw, "timestamp", None)
    if callable(method):
        try:
            return float(method())
        except Exception:
            return None
    return None


# --- detection ------------------------------------------------------------


def _chunk_files(root: Path) -> list[Path]:
    return sorted(path for path in root.rglob(f"*{CHUNK_SUFFIX}") if path.is_file())


def _levels(root: Path, files: list[Path]) -> list[list[str]]:
    """Directory names at each depth below `root`, across every file."""
    depths: list[list[str]] = []
    for file in files:
        parts = file.relative_to(root).parts[:-1]  # drop the filename
        for depth, name in enumerate(parts):
            while len(depths) <= depth:
                depths.append([])
            depths[depth].append(name)
    return depths


def _hive_key(name: str) -> str | None:
    match = _HIVE.match(name)
    return match.group(1) if match else None


def structure_fingerprint(root: Path, files: list[Path]) -> str:
    """A description of the tree's SHAPE, not its contents.

    Which levels exist, how many distinct names each has, whether each is
    hive-style. Adding a thousand files to an existing directory leaves it
    unchanged — which is the point, since the mapping is saved against it.
    """
    parts = []
    for depth, names in enumerate(_levels(root, files)):
        distinct = sorted(set(names))
        hive = sorted({key for key in (_hive_key(name) for name in distinct) if key})
        parts.append(f"{depth}:{len(distinct)}:{'|'.join(hive)}")
    return ";".join(parts)


# Share of a level's directories that must look like a series before the
# level is suggested as one. A majority rather than all of them, for the
# same reason a unit's reference schema is the majority: one directory that
# happens to hold a stray file should not cost the whole level its role.
SERIES_MAJORITY = 0.5
# A directory holding a single file is not evidence of anything — that is
# just a file in a folder. At least one directory at the level has to hold
# several chunks for "these are the ordered chunks of one dataset" to be a
# claim about the tree rather than a guess.
SERIES_MIN_CHUNKS = 2


def _looks_like_a_series(entries: list[tuple[Path, "_Footer"]], column: str | None) -> bool:
    """Whether one directory's files look like the chunks of ONE dataset.

    Two pieces of evidence, both from the footer:

      SAME SCHEMA. Chunks of one dataset have one schema. A directory
      holding two shapes of file is holding two things.

      NON-OVERLAPPING TIME RANGES. Chunks of one dataset partition time;
      they do not cover the same moments twice. Overlap is the
      disqualifier — a GAP is not, because a recording can legitimately
      pause, and treating a pause as evidence against would reject most
      real data.

    Unknown time bounds mean no evidence either way, and no evidence is not
    evidence of a series: this returns False rather than guessing.
    """
    if len(entries) < SERIES_MIN_CHUNKS:
        return False
    signatures = {footer.signature for _path, footer in entries}
    if len(signatures) != 1:
        return False
    if not column:
        return False

    ranges = []
    for _path, footer in entries:
        bounds = footer.stats.get(column)
        if bounds is None:
            return False
        low, high = _timestamp_value(bounds[0]), _timestamp_value(bounds[1])
        if low is None or high is None:
            return False
        ranges.append((low, high))

    ranges.sort()
    return all(later[0] >= earlier[1] for earlier, later in zip(ranges, ranges[1:]))


def _suggest_series_depth(
    root: Path, files: list[Path], level_count: int, group_depths: list[int], column: str | None
) -> int | None:
    """The shallowest named level whose directories look like series.

    Evidence, not position. A level qualifies when a majority of its
    directories hold same-schema files with non-overlapping time ranges —
    which is what "the ordered chunks of one dataset" actually looks like
    from the outside, and is checkable from parquet footers alone.

    SHALLOWEST, because a series is the outermost directory that holds one
    dataset's chunks; every level below it is structure inside that
    dataset, and would qualify on the same evidence. Hive levels are
    excluded outright: `key=value` says "this identifies something", which
    is a group, and a group of well-behaved series would otherwise qualify
    as a series itself.

    Positional rules were tried first and rejected: "the level below the
    deepest group" cannot see a series that has no group above it, and
    names a level a series on no evidence whenever one happens to sit
    there.
    """
    for depth in range(level_count):
        if depth in group_depths:
            continue
        buckets: dict[tuple[str, ...], list[tuple[Path, _Footer]]] = {}
        for file in files:
            parts = file.relative_to(root).parts[:-1]
            if depth >= len(parts):
                continue
            footer = read_footer(file)
            if footer is None:
                continue
            buckets.setdefault(tuple(parts[: depth + 1]), []).append((file, footer))
        if not buckets:
            continue
        qualifying = sum(1 for entries in buckets.values() if _looks_like_a_series(entries, column))
        if qualifying and qualifying > len(buckets) * SERIES_MAJORITY:
            return depth
    return None


def propose_mapping(root: str | Path) -> StructureProposal:
    """Read a tree and suggest a role and label per directory level.

    Every HIVE level is a group: `key=value` is the one convention that
    explicitly says "this level identifies something", and only a
    contiguous run of them from the outermost counts — a hive directory
    buried under three arbitrary levels is a partition INSIDE something
    rather than the thing itself.

    A series is suggested on EVIDENCE rather than position: the shallowest
    named level whose directories mostly hold same-schema files with
    non-overlapping time ranges (see _suggest_series_depth). That lets a
    series be found with no group above it, and — more importantly —
    declines to name one when the tree gives no reason to.

    Everything else is "ignore", which is a role rather than an absence of
    one. `detected` means "there is something to confirm", not "this is a
    fleet": a hive-only tree is detected with groups and no series, and a
    tree with nothing recognisable is not detected at all, in which case
    the caller keeps its schema-group behaviour untouched.
    """
    root = Path(root)
    files = _chunk_files(root)
    fingerprint = structure_fingerprint(root, files)
    levels = _levels(root, files)

    hive_depths = [
        depth
        for depth, names in enumerate(levels)
        if any(_hive_key(name) for name in set(names))
    ]
    group_depths: list[int] = []
    for depth in hive_depths:
        if depth == len(group_depths):
            group_depths.append(depth)
        else:
            break

    # The timestamp column has to be known BEFORE the series test, because
    # the test is about time. Read from one footer; a tree whose files
    # disagree about their own schema is not one this is going to rescue.
    first_footer = read_footer(files[0]) if files else None
    candidates: list[dict[str, Any]] = (
        timestamp_candidates(first_footer.signature, first_footer.stats)
        if first_footer is not None
        else []
    )
    time_column = candidates[0]["column"] if candidates else None

    series_depth = _suggest_series_depth(root, files, len(levels), group_depths, time_column)

    proposals: list[LevelProposal] = []
    for depth, names in enumerate(levels):
        distinct = sorted(set(names), key=natural_key)
        keys = sorted({key for key in (_hive_key(name) for name in distinct) if key})
        hive_key = keys[0] if len(keys) == 1 else None
        if depth in group_depths:
            role, label = ROLE_GROUP, hive_key or DEFAULT_GROUP_LABEL
        elif depth == series_depth:
            role, label = ROLE_SERIES, DEFAULT_SERIES_LABEL
        else:
            role, label = ROLE_IGNORE, ""
        proposals.append(
            LevelProposal(
                depth=depth,
                role=role,
                label=label,
                examples=distinct,
                directory_count=len(distinct),
                hive_key=hive_key,
            )
        )

    detected = bool(group_depths) or series_depth is not None

    return StructureProposal(
        root=str(root),
        levels=proposals,
        fingerprint=fingerprint,
        detected=detected,
        timestamp_candidates=candidates,
        suggested_timestamp=candidates[0]["column"] if candidates else None,
        suggested_unit=candidates[0]["unit"] if candidates else None,
    )


# --- applying a mapping ---------------------------------------------------


def build_structure(
    root: str | Path, mapping: StructureMapping, timestamp_column: str | None = None
) -> DatasetStructure:
    """Group a tree into units under a confirmed mapping.

    One footer read per file gives schema, row count and time bounds
    together. Nothing here reads a row.

    With no series level a unit is one FILE, which is how a flat folder and
    a hive-only tree come out of the same code as a fully structured one.
    """
    root = Path(root)
    files = _chunk_files(root)
    group_depths = mapping.group_depths()
    series_depth = mapping.series_depth()
    column = timestamp_column if timestamp_column is not None else mapping.timestamp_column

    grouped: dict[tuple[tuple[str, ...], str], list[tuple[Path, _Footer]]] = {}
    unreadable: list[ExcludedChunk] = []

    for file in files:
        parts = file.relative_to(root).parts[:-1]
        if any(depth >= len(parts) for depth in group_depths):
            continue  # this file does not sit under the mapped levels
        if series_depth is not None and series_depth >= len(parts):
            continue
        footer = read_footer(file)
        if footer is None:
            unreadable.append(ExcludedChunk(file, "Could not read this file's parquet footer."))
            continue
        key = tuple(parts[depth] for depth in group_depths)
        # No series level means the unit is the file itself, so its id is
        # the filename and it is the only chunk in it.
        unit_id = parts[series_depth] if series_depth is not None else file.name
        grouped.setdefault((key, unit_id), []).append((file, footer))

    if column is None and grouped:
        first_footer = next(iter(grouped.values()))[0][1]
        candidates = timestamp_candidates(first_footer.signature, first_footer.stats)
        column = candidates[0]["column"] if candidates else None

    findings: list[dict[str, Any]] = []
    units: list[Unit] = []
    for (key, unit_id), entries in sorted(
        grouped.items(), key=lambda item: (tuple(natural_key(v) for v in item[0][0]), natural_key(item[0][1]))
    ):
        unit = _build_unit(key, unit_id, entries, column)
        units.append(unit)
        findings.extend(_unit_findings(unit, mapping))

    for item in unreadable:
        findings.append(
            {
                "check_id": "unreadable_chunk",
                "severity": "warning",
                "title": "File could not be read",
                "detail": f"{item.path.name}: {item.reason}",
                "columns": [],
            }
        )

    return DatasetStructure(
        root=root,
        units=units,
        mapping=mapping,
        timestamp_column=column,
        timestamp_unit=mapping.timestamp_unit,
        findings=findings,
    )


def _build_unit(
    key: tuple[str, ...],
    unit_id: str,
    entries: list[tuple[Path, _Footer]],
    timestamp_column: str | None,
) -> Unit:
    """One unit: its chunks, in order, minus any that do not belong.

    The schema check comes FIRST, because a chunk with a different schema
    is not part of this unit at all and ordering it would be ordering
    something else.
    """
    entries = sorted(entries, key=lambda item: natural_key(str(item[0])))
    signature = _reference_signature([footer.signature for _path, footer in entries])

    kept: list[Chunk] = []
    excluded: list[ExcludedChunk] = []
    for path, footer in entries:
        if footer.signature != signature:
            excluded.append(ExcludedChunk(path, _schema_difference(signature, footer.signature)))
            continue
        low, high = (None, None)
        if timestamp_column and timestamp_column in footer.stats:
            raw_low, raw_high = footer.stats[timestamp_column]
            low, high = _timestamp_value(raw_low), _timestamp_value(raw_high)
        kept.append(Chunk(path, footer.row_count, footer.signature, low, high))

    ordered, order_source, conflict = order_chunks(kept)
    for index, chunk in enumerate(ordered):
        chunk.index = index

    return Unit(
        key=key,
        unit_id=unit_id,
        chunks=ordered,
        excluded=excluded,
        schema_signature=signature,
        order_source=order_source,
        order_conflict=conflict,
    )


def _reference_signature(signatures: list[tuple]) -> tuple:
    """The schema a unit's chunks are expected to have.

    The most common one, ties broken by natural file order. Majority rather
    than first-file, because "the first chunk happened to be the broken
    one" is a real way to lose a whole unit to a single bad file.
    """
    if not signatures:
        return ()
    counts: dict[tuple, int] = {}
    for signature in signatures:
        counts[signature] = counts.get(signature, 0) + 1
    best = max(counts.values())
    for signature in signatures:  # already in natural order
        if counts[signature] == best:
            return signature
    return signatures[0]


def _schema_difference(reference: tuple, other: tuple) -> str:
    """Why a chunk was excluded, in terms the user can act on."""
    reference_names = {name for name, _ in reference}
    other_names = {name for name, _ in other}
    missing = sorted(reference_names - other_names)
    extra = sorted(other_names - reference_names)
    reference_types = dict(reference)
    changed = sorted(
        name for name, dtype in other if name in reference_types and reference_types[name] != dtype
    )
    parts = []
    if missing:
        parts.append(f"missing {', '.join(missing)}")
    if extra:
        parts.append(f"unexpected {', '.join(extra)}")
    if changed:
        parts.append(f"different type for {', '.join(changed)}")
    detail = "; ".join(parts) if parts else "a different column order"
    return f"Schema differs from the rest of this series ({detail})."


def order_chunks(chunks: list[Chunk]) -> tuple[list[Chunk], str, bool]:
    """Chunks in the order they were recorded.

    Filename order first — natural sort, so `part-2` precedes `part-10` —
    then VERIFIED against the timestamp bounds the footer already gave us.
    Time wins on disagreement because a filename is a label and a timestamp
    is a measurement, but the disagreement is reported rather than quietly
    corrected: files out of time order usually mean something upstream
    renumbered, retried or back-filled.

    Falls back to filename order — without claiming a conflict — when any
    chunk's bounds are unknown, because a partial ordering by time is not
    an ordering.
    """
    by_name = sorted(chunks, key=lambda chunk: natural_key(str(chunk.path)))
    stamps = [chunk.first_timestamp for chunk in by_name]
    if not by_name or any(stamp is None for stamp in stamps):
        return by_name, "name", False

    by_time = sorted(
        by_name, key=lambda chunk: (chunk.first_timestamp, natural_key(str(chunk.path)))
    )
    if [chunk.path for chunk in by_time] == [chunk.path for chunk in by_name]:
        return by_name, "name", False
    return by_time, "timestamp", True


def _unit_findings(unit: Unit, mapping: StructureMapping) -> list[dict[str, Any]]:
    """Structure problems worth telling the user about.

    Findings rather than errors: neither a deviant chunk nor an
    out-of-order unit makes the rest of the data unusable, and refusing to
    load because one file is odd would be the wrong trade every time.

    A single-chunk unit can have neither problem, so a flat folder produces
    none of these — which is what "unchanged on flat inputs" means here.
    """
    findings: list[dict[str, Any]] = []
    label = " / ".join([*unit.key, unit.unit_id]) or unit.unit_id
    if unit.order_conflict:
        findings.append(
            {
                "check_id": "chunk_order_vs_time",
                "severity": "warning",
                "title": "Chunk order by name differs from time order",
                "detail": (
                    f"{label}: the chunk filenames do not run in timestamp order, so this "
                    "series is ordered by its footer timestamps instead. That usually means "
                    "chunks were renumbered, retried or back-filled upstream."
                ),
                "columns": [],
            }
        )
    for item in unit.excluded:
        findings.append(
            {
                "check_id": "deviant_chunk_schema",
                "severity": "warning",
                "title": "Chunk excluded from its series",
                "detail": f"{label}: {item.path.name} was excluded — {item.reason}",
                "columns": [],
            }
        )
    return findings


# --- scope ----------------------------------------------------------------


def scope_units(structure: DatasetStructure, scope: Scope) -> list[Unit]:
    """The units a scope covers.

    A prefix of the group values narrows; a unit id narrows to one. An
    empty scope is everything, which is what "the whole dataset" means at
    any depth, including zero.
    """
    prefix = list(scope.values)
    matched = [unit for unit in structure.units if list(unit.key[: len(prefix)]) == prefix]
    if scope.unit_id:
        matched = [unit for unit in matched if unit.unit_id == scope.unit_id]
    return matched


def scope_chunks(structure: DatasetStructure, scope: Scope) -> list[Chunk]:
    """Every chunk in scope — units in order, chunks in order within each,
    so a whole-dataset frame is still a sequence of whole units rather than
    an interleaving."""
    return [chunk for unit in scope_units(structure, scope) for chunk in unit.chunks]


def scope_files(structure: DatasetStructure, scope: Scope) -> list[Path]:
    return [chunk.path for chunk in scope_chunks(structure, scope)]


def scope_label(structure: DatasetStructure, scope: Scope) -> list[dict[str, Any]]:
    """The breadcrumb, in the MAPPING's words.

    The first crumb is the root's label ("All" by default); then one per
    chosen group value; then the unit. Nothing here decides what those
    levels are called — that came from the user.
    """
    crumbs: list[dict[str, Any]] = [
        {
            "label": structure.mapping.root_label or DEFAULT_ROOT_LABEL,
            "level_label": "",
            "values": [],
            "unit_id": "",
        }
    ]
    for index, value in enumerate(scope.values):
        crumbs.append(
            {
                "label": value,
                "level_label": structure.group_labels[index]
                if index < len(structure.group_labels)
                else DEFAULT_GROUP_LABEL,
                "values": list(scope.values[: index + 1]),
                "unit_id": "",
            }
        )
    if scope.unit_id:
        crumbs.append(
            {
                "label": scope.unit_id,
                "level_label": structure.unit_label,
                "values": list(scope.values),
                "unit_id": scope.unit_id,
            }
        )
    return crumbs


# --- the frame a scope hands downstream -----------------------------------


def chunk_identity(structure: DatasetStructure, scope: Scope) -> list[dict[str, Any]]:
    """Per-chunk derived-column values, in scope order.

    Travels in a job spec alongside the file list so a worker subprocess
    builds the identical frame — paths and scalars, the same rule every
    other spec follows. The COLUMN NAMES come from the structure, so a
    consumer reads them off the rows rather than from a hardcoded list.
    """
    names = structure.derived_columns()
    group_count = len(structure.group_depths)
    rows = []
    for unit in scope_units(structure, scope):
        for chunk in unit.chunks:
            row: dict[str, Any] = {"file": str(chunk.path)}
            for index, name in enumerate(names[:group_count]):
                row[name] = unit.key[index] if index < len(unit.key) else ""
            row[names[group_count]] = unit.unit_id
            row[names[group_count + 1]] = chunk.index
            row[names[group_count + 2]] = chunk.path.name
            rows.append(row)
    return rows


def identity_columns(identity: list[dict[str, Any]] | None) -> list[str]:
    """The derived column names carried by an identity list.

    Read off the rows rather than imported as a constant: the names depend
    on the mapping (a hive key becomes a column called `sys_key`, a labelled
    level becomes one called after its label), so no module downstream can
    hold a fixed list of them.
    """
    if not identity:
        return []
    return [name for name in identity[0] if name != "file"]


def scope_lazyframe(identity: list[dict[str, Any]]):
    """One LazyFrame over the scope's chunks, carrying the derived columns.

    Lazy throughout: `scan_parquet` reads nothing, `with_columns` of
    literals reads nothing, `concat` reads nothing. So a tree of thousands
    of chunks becomes a frame instantly and the derived columns cost
    nothing — they are constants per chunk, known from the path alone.
    """
    import polars as pl

    if not identity:
        raise ValueError("Cannot build a frame for an empty scope.")
    names = identity_columns(identity)
    scans = []
    for row in identity:
        literals = [
            pl.lit(row[name], dtype=pl.Int32 if name.endswith(CHUNK_INDEX_COLUMN) else pl.String).alias(name)
            for name in names
        ]
        scans.append(pl.scan_parquet(row["file"]).with_columns(*literals))
    if len(scans) == 1:
        return scans[0]
    return pl.concat(scans, how="vertical")
