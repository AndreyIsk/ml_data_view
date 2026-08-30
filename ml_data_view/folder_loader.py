"""Headless folder discovery, schema grouping, and lazy multi-file loading
for Stage 2 of the lazy-loading architecture (see PROJECT_BRIEF.md's
"Architecture notes" — Stage 1 was the single-file eager-to-lazy refactor).

Discovery only ever reads schema metadata (collect_schema() — no row data)
plus a cheap row count: parquet's format-native metadata (read directly via
pyarrow, never touching row data) or, for arrow/feather, a bounded
streaming length aggregation (.select(pl.len()).collect(engine=
"streaming"), which counts rows without materializing them — see
_cheap_row_count's docstring for why a raw per-batch pyarrow read isn't
good enough here even though it looks metadata-only). That's what lets
this stay fast across many files in a large folder tree. Loading a chosen
group hands a single concatenated LazyFrame to the existing chart/table/
embedding/media code unchanged — see chart_data.py et al., which already
operate on LazyFrames after Stage 1.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import polars as pl
import pyarrow.parquet as pq

from ml_data_view.checkpoint_log import checkpoint
from ml_data_view.data_loader import _dispatch_scanner

# .feather is the same Arrow IPC format as .arrow, just a different common
# extension for it. Discovery accepts both; single-file open (data_loader's
# SUPPORTED_EXTENSIONS) still only lists .parquet/.arrow/.csv — unchanged.
DISCOVERY_EXTENSIONS = {".parquet", ".csv", ".arrow", ".feather"}


def discover_files(root: str | Path) -> list[Path]:
    """Recursively walk `root` (any depth of nested subfolders) for
    supported dataset files, in a stable (sorted) order.

    Directory traversal and filename/extension checks only — no file is
    opened here.
    """
    root = Path(root)
    return sorted(
        path
        for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() in DISCOVERY_EXTENSIONS
    )


def _schema_signature(schema: pl.Schema) -> tuple[tuple[str, pl.DataType], ...]:
    """An exact, order-sensitive schema fingerprint — pl.concat(...,
    how="vertical") requires matching column order as well as names/types,
    so grouping must too (verified: concat raises otherwise, it does not
    align columns by name)."""
    return tuple(schema.items())


def _cheap_row_count(path: Path, suffix: str) -> int | None:
    """A row count that never reads actual row data.

    Parquet stores its row count directly in the file footer, so
    pq.ParquetFile(path).metadata.num_rows is metadata-only.

    Arrow IPC (.arrow/.feather) has no equivalent footer-level total, so
    this instead runs a bounded streaming length aggregation via polars
    (.select(pl.len()).collect(engine="streaming")) — a single scalar out,
    not a materialized frame. This is NOT the same as opening the file with
    pyarrow.ipc.open_file() and summing get_batch(i).num_rows per batch:
    that reads every batch's actual buffers off disk one at a time (real
    row data, just discarded right after counting) to get a number the
    streaming aggregation gets from the query engine without ever pulling
    row data into Python. Returns None when no such cheap path exists
    (.csv has no stored row count; getting one would mean a full scan,
    which isn't "cheap" for a large file during just the discovery step).
    """
    try:
        if suffix == ".parquet":
            return pq.ParquetFile(path).metadata.num_rows
        if suffix in (".arrow", ".feather"):
            return int(pl.scan_ipc(path).select(pl.len()).collect(engine="streaming")[0, 0])
    except Exception:
        return None
    return None


def _describe_columns(signature: tuple[tuple[str, pl.DataType], ...], limit: int = 3) -> str:
    names = [name for name, _ in signature]
    shown = ", ".join(names[:limit])
    if len(names) > limit:
        shown += ", …"
    return f"{len(names)} column{'s' if len(names) != 1 else ''} ({shown})" if names else "no columns"


@dataclass
class UnreadableFile:
    path: Path
    reason: str

    def to_dict(self) -> dict[str, Any]:
        return {"file": str(self.path), "reason": self.reason}


@dataclass
class SchemaGroup:
    signature: tuple[tuple[str, pl.DataType], ...]
    files: list[Path]
    row_count: int | None  # None unless every file in the group had a cheap count available

    @property
    def file_count(self) -> int:
        return len(self.files)

    def to_dict(self) -> dict[str, Any]:
        return {
            "columns": [{"name": name, "dtype": str(dtype)} for name, dtype in self.signature],
            "files": [str(f) for f in self.files],
            "file_count": self.file_count,
            "row_count": self.row_count,
        }


@dataclass
class FolderScanResult:
    groups: list[SchemaGroup]
    unreadable: list[UnreadableFile]

    @property
    def default_group_index(self) -> int:
        """The largest group by file count — groups are sorted
        largest-first, so this is always 0 when there's at least one
        group. Still a named property so callers don't have to know that."""
        return 0

    def excluded_from(self, chosen_index: int) -> list[dict[str, Any]]:
        """Every file NOT part of the group at `chosen_index`, each with a
        short reason: a different-schema group's files ("Different
        schema..."), or a file whose schema couldn't be read at all (the
        error message from discovery). Presented together — from the
        perspective of "what got loaded", both are simply excluded — so
        the user can investigate them separately rather than wondering
        where they went.
        """
        excluded: list[dict[str, Any]] = []
        for index, group in enumerate(self.groups):
            if index == chosen_index:
                continue
            reason = f"Different schema than the selected group: {_describe_columns(group.signature)}"
            excluded.extend({"file": str(f), "reason": reason} for f in group.files)
        excluded.extend(unreadable.to_dict() for unreadable in self.unreadable)
        return excluded

    def to_dict(self) -> dict[str, Any]:
        return {
            "groups": [g.to_dict() for g in self.groups],
            "unreadable": [u.to_dict() for u in self.unreadable],
            "default_group_index": self.default_group_index,
        }


def scan_folder(root: str | Path) -> FolderScanResult:
    """Discover files under `root` and group them by exact schema match.

    Each file's schema is read via collect_schema() (metadata only, no
    collect()); files that can't even be schema-read are reported in
    `unreadable` with a short reason rather than silently skipped. Groups
    are sorted largest-file-count-first (see default_group_index).
    """
    checkpoint("discovery:start", f"root={root}")
    files = discover_files(root)
    checkpoint("discovery:done", f"file_count={len(files)}")

    groups_by_signature: dict[tuple, list[Path]] = {}
    unreadable: list[UnreadableFile] = []

    checkpoint("schema_grouping:start", f"file_count={len(files)}")
    for index, path in enumerate(files):
        # One line per file, not just one for the whole stage: if this is
        # where the process dies, the last line here is the exact file
        # that was being schema-read or row-counted when it happened.
        checkpoint("schema_grouping:file", f"{index + 1}/{len(files)} {path}")
        suffix = path.suffix.lower()
        try:
            schema = _dispatch_scanner(path, suffix).collect_schema()
        except Exception as exc:
            unreadable.append(UnreadableFile(path=path, reason=f"Could not read schema: {exc}"))
            continue
        groups_by_signature.setdefault(_schema_signature(schema), []).append(path)
    checkpoint("schema_grouping:schemas_done", f"group_count={len(groups_by_signature)} unreadable={len(unreadable)}")

    groups = []
    for signature, group_files in groups_by_signature.items():
        row_counts = [_cheap_row_count(f, f.suffix.lower()) for f in group_files]
        total_row_count = None if any(rc is None for rc in row_counts) else sum(row_counts)
        groups.append(SchemaGroup(signature=signature, files=group_files, row_count=total_row_count))

    groups.sort(key=lambda g: g.file_count, reverse=True)
    checkpoint("schema_grouping:done", f"group_count={len(groups)}")
    return FolderScanResult(groups=groups, unreadable=unreadable)


def load_group(group: SchemaGroup) -> pl.LazyFrame:
    """Combine a schema group's files into a single LazyFrame via
    pl.concat(..., how="vertical") — lazy, so this doesn't read any row
    data itself. Handed unchanged to the same chart/table/embedding/media
    code single-file loads already use (see chart_data.py et al.)."""
    if not group.files:
        raise ValueError("Cannot load an empty schema group (no files).")
    scans = [_dispatch_scanner(f, f.suffix.lower()) for f in group.files]
    return pl.concat(scans, how="vertical")
