"""Local app settings: persisted as one JSON file in the OS-appropriate
per-user app-data directory. No cloud, no network — every read/write here
is a plain local file operation.

Kept separate from api.py (which is the pywebview-facing wrapper) so it
stays importable/testable headlessly, per this project's usual convention
(see chart_data.py, data_loader.py).
"""
from __future__ import annotations

import json
import os
import sys
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any

APP_NAME = "MLDatasetProfiler"
SETTINGS_FILENAME = "settings.json"

# Neutral gray (matches the frontend's --text-muted) — a banner that hasn't
# been configured yet shouldn't default to an alarming color.
DEFAULT_BANNER_COLOR = "#83827c"

# "system" follows the OS via prefers-color-scheme; the other two override
# it. Default is "system" so a first launch matches the machine rather than
# imposing a choice nobody made.
THEMES = ("system", "light", "dark")
DEFAULT_THEME = "system"

# `memory_ceiling_mb = None` means "work it out from this machine" — half
# of physical RAM, floored at 2 GB (jobs.default_memory_ceiling_mb). Stored
# as None rather than as the resolved number so the default follows the
# hardware: a settings file copied to a bigger machine, or one written
# before more RAM was installed, should not pin an old ceiling forever.
MIN_MEMORY_CEILING_MB = 2048

# Folder watching. One folder at a time (see folder_watcher.py): two
# watched folders would mean two merged profiles to keep straight, and the
# app shows one dataset. `watch_folder = ""` means "nothing chosen yet",
# which is distinct from watching being switched off.
DEFAULT_WATCH_STABILITY_SECONDS = 10


@dataclass(frozen=True)
class AppSettings:
    """One flat record of every user-configurable setting. Add new fields
    with a default here; `AppSettings.from_dict` ignores unknown keys and
    fills in missing ones, so old settings files stay loadable."""

    classification_banner_enabled: bool = False
    classification_banner_label: str = ""
    classification_banner_color: str = DEFAULT_BANNER_COLOR
    audit_log_enabled: bool = False
    theme: str = DEFAULT_THEME
    # None = automatic (see MIN_MEMORY_CEILING_MB above).
    memory_ceiling_mb: int | None = None
    watch_enabled: bool = False
    watch_folder: str = ""
    # How long a file's size must hold still before it is profiled. See
    # folder_watcher.py — this is the setting that decides whether a
    # pipeline's half-written file gets read.
    watch_stability_seconds: int = DEFAULT_WATCH_STABILITY_SECONDS
    # Time-series health thresholds (timeseries.py). One flat record rather
    # than a nested document so a single changed number is one obvious
    # line in settings.json; `timeseries_ranges` is the optional data
    # dictionary, {column: [low, high]}.
    timeseries_gap_factor: float = 5.0
    timeseries_flatline_rows: int = 500
    timeseries_short_unit_ratio: float = 0.5
    timeseries_jitter_ratio: float = 0.25
    timeseries_rate_change_ratio: float = 0.5
    timeseries_range_spread_factor: float = 10.0
    # How many channels the per-channel passes cover. The timestamp pass is
    # one read per chunk; each channel is another, so a 40-column folder
    # would multiply a first profile by forty. What is left out is reported
    # rather than silently absent.
    timeseries_max_channels: int = 12
    timeseries_ranges: dict = field(default_factory=dict)
    # The channel timeline (timeline.py). Channels are capped because a
    # strip per channel is a chart and forty is a wall — and because the
    # read cost is per channel. Rows are capped because that is the whole
    # bargain: a fetch that would decode more than this answers from the
    # per-chunk buckets already cached instead, so the viewer opens
    # instantly on a recording of any size and detail arrives as you zoom.
    timeline_max_channels: int = 6
    timeline_max_rows: int = 2_000_000
    # Group-comparison thresholds (fleet.py). The DIVERGENCE score is not
    # here: it is PSI, judged against the conventional thresholds
    # compare.py already documents, and a per-install version of a
    # standard scale would make "significant" mean something different in
    # each copy of the app. These two have no convention behind them.
    fleet_duration_ratio: float = 0.25
    fleet_outlier_sigma: float = 3.0
    # How many channels the fleet section carries HISTOGRAMS for. Every
    # channel is still scored — a score is five numbers — but a per-group
    # histogram for each of forty columns is a payload that grows with a
    # column count nobody is looking at.
    fleet_max_channels: int = 12
    # Named row searches, each {"name", "conditions", "columns"} — see
    # search.py. Stored here rather than in their own file so a user's
    # whole local state stays one JSON document.
    saved_searches: list[dict] = field(default_factory=list)
    # Confirmed fleet/system/recording mappings, one per root path (see
    # hierarchy.py). Kept here for the same reason saved searches are: a
    # user's local state is one document, and a mapping is a preference
    # about a folder, not data about it — clearing the profile cache must
    # not make the app start asking again.
    structure_mappings: list[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> AppSettings:
        defaults = cls()
        field_names = defaults.to_dict().keys()
        known = {k: v for k, v in data.items() if k in field_names}
        merged = {**defaults.to_dict(), **known}
        # An unrecognised theme (hand-edited file, or one written by a
        # future version) must not leave the frontend with a value it
        # cannot resolve to a stylesheet.
        if merged.get("theme") not in THEMES:
            merged["theme"] = DEFAULT_THEME
        merged["memory_ceiling_mb"] = _clean_memory_ceiling(merged.get("memory_ceiling_mb"))
        merged["watch_stability_seconds"] = _clean_stability_seconds(
            merged.get("watch_stability_seconds")
        )
        return cls(**merged)


def _clean_memory_ceiling(value: Any) -> int | None:
    """Normalise the ceiling to None (automatic) or at least the minimum.

    Clamped rather than rejected: the field is a number box, and a user who
    types 1 wants the smallest ceiling available, not a validation error.
    Anything that isn't a number at all falls back to automatic — a
    hand-edited or future-version settings file must not leave the runner
    with a ceiling it cannot use.
    """
    if value is None or isinstance(value, bool):
        return None
    try:
        ceiling = int(value)
    except (TypeError, ValueError):
        return None
    if ceiling <= 0:
        return None
    return max(MIN_MEMORY_CEILING_MB, ceiling)


def _clean_stability_seconds(value: Any) -> int:
    """Clamp the stability interval to something that can actually work.

    Below a second, ordinary filesystem latency alone can make a file that
    is still being written look stable between two stat() calls — so a user
    who types 0 gets the floor rather than a watcher that profiles
    half-written files. Anything non-numeric falls back to the default.
    """
    from ml_data_view.folder_watcher import MIN_STABILITY_SECONDS

    try:
        seconds = int(value)
    except (TypeError, ValueError):
        return DEFAULT_WATCH_STABILITY_SECONDS
    return max(int(MIN_STABILITY_SECONDS), seconds)


def timeseries_thresholds(settings: AppSettings) -> dict:
    """The time-series thresholds, in the shape timeseries.Thresholds takes.

    Assembled here rather than in timeseries.py so that module stays
    importable and testable with no app around it — the same rule
    chart_data.py and findings.py follow.
    """
    return {
        "gap_factor": settings.timeseries_gap_factor,
        "flatline_rows": settings.timeseries_flatline_rows,
        "short_unit_ratio": settings.timeseries_short_unit_ratio,
        "jitter_ratio": settings.timeseries_jitter_ratio,
        "rate_change_ratio": settings.timeseries_rate_change_ratio,
        "range_spread_factor": settings.timeseries_range_spread_factor,
        "ranges": dict(settings.timeseries_ranges or {}),
    }


def fleet_thresholds(settings: AppSettings) -> dict:
    """The group-comparison thresholds, in the shape fleet.FleetThresholds
    takes. Assembled here for the same reason timeseries_thresholds is:
    fleet.py stays importable with no app around it."""
    return {
        "duration_ratio": settings.fleet_duration_ratio,
        "outlier_sigma": settings.fleet_outlier_sigma,
    }


def resolve_memory_ceiling_mb(settings: AppSettings) -> int:
    """The ceiling to actually run jobs under: the user's, or this
    machine's default."""
    from ml_data_view.jobs import default_memory_ceiling_mb

    return settings.memory_ceiling_mb or default_memory_ceiling_mb()


def app_data_dir() -> Path:
    """OS-appropriate per-user app-data directory for this app.

    Windows: %APPDATA%/MLDatasetProfiler
    macOS:   ~/Library/Application Support/MLDatasetProfiler
    Linux/other: $XDG_CONFIG_HOME/MLDatasetProfiler, falling back to
      ~/.config/MLDatasetProfiler
    """
    if sys.platform == "win32":
        base = os.environ.get("APPDATA") or str(Path.home() / "AppData" / "Roaming")
        return Path(base) / APP_NAME
    if sys.platform == "darwin":
        return Path.home() / "Library" / "Application Support" / APP_NAME
    base = os.environ.get("XDG_CONFIG_HOME") or str(Path.home() / ".config")
    return Path(base) / APP_NAME


def settings_file_path(base_dir: Path | None = None) -> Path:
    """`base_dir` overrides the app-data directory — used by tests so they
    never touch the real user's app-data directory."""
    return (base_dir or app_data_dir()) / SETTINGS_FILENAME


def load_settings(base_dir: Path | None = None) -> AppSettings:
    path = settings_file_path(base_dir)
    if not path.exists():
        return AppSettings()
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return AppSettings()
    if not isinstance(data, dict):
        return AppSettings()
    return AppSettings.from_dict(data)


def save_settings(settings: AppSettings, base_dir: Path | None = None) -> None:
    path = settings_file_path(base_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(settings.to_dict(), indent=2), encoding="utf-8")


def upsert_saved_search(
    settings: AppSettings, name: str, conditions: list[dict], columns: list[str] | None
) -> AppSettings:
    """Add or replace a named search. Saving an existing name overwrites it
    rather than creating a duplicate the user cannot tell apart."""
    name = name.strip()
    if not name:
        raise ValueError("A saved search needs a name.")
    entry = {"name": name, "conditions": list(conditions), "columns": list(columns or [])}
    remaining = [s for s in settings.saved_searches if s.get("name") != name]
    return replace(settings, saved_searches=[*remaining, entry])


def structure_mapping_for(settings: AppSettings, root: str) -> dict | None:
    """The confirmed mapping for `root`, if there is one.

    Matched on the root path alone; whether it is still VALID for the tree
    as it stands now is a separate question, answered by comparing
    fingerprints (see hierarchy.structure_fingerprint) — the two are kept
    apart so "we have an answer" and "the answer still applies" never get
    confused for each other.
    """
    target = str(Path(root))
    for entry in settings.structure_mappings:
        if str(Path(entry.get("root") or "")) == target:
            return entry
    return None


def upsert_structure_mapping(settings: AppSettings, mapping: dict) -> AppSettings:
    """Save (or replace) one root's mapping. Replacing rather than
    appending: a root has exactly one answer, and two would mean the app
    could silently pick the stale one."""
    target = str(Path(mapping.get("root") or ""))
    remaining = [
        entry
        for entry in settings.structure_mappings
        if str(Path(entry.get("root") or "")) != target
    ]
    return replace(settings, structure_mappings=[*remaining, dict(mapping)])


def remove_structure_mapping(settings: AppSettings, root: str) -> AppSettings:
    """Forget one root's mapping, so the dialog is offered again."""
    target = str(Path(root))
    return replace(
        settings,
        structure_mappings=[
            entry
            for entry in settings.structure_mappings
            if str(Path(entry.get("root") or "")) != target
        ],
    )


def remove_saved_search(settings: AppSettings, name: str) -> AppSettings:
    """Drop a named search. Removing one that isn't there is a no-op, not
    an error — the UI may be acting on a stale list."""
    return replace(
        settings,
        saved_searches=[s for s in settings.saved_searches if s.get("name") != name],
    )
