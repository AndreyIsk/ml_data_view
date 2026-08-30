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

APP_NAME = "MLDatasetProfiler"
SETTINGS_FILENAME = "settings.json"

# Neutral gray (matches the frontend's --text-muted) — a banner that hasn't
# been configured yet shouldn't default to an alarming color.
DEFAULT_BANNER_COLOR = "#83827c"


@dataclass(frozen=True)
class AppSettings:
    """One flat record of every user-configurable setting. Add new fields
    with a default here; `AppSettings.from_dict` ignores unknown keys and
    fills in missing ones, so old settings files stay loadable."""

    classification_banner_enabled: bool = False
    classification_banner_label: str = ""
    classification_banner_color: str = DEFAULT_BANNER_COLOR
    audit_log_enabled: bool = False
    # Named row searches, each {"name", "conditions", "columns"} — see
    # search.py. Stored here rather than in their own file so a user's
    # whole local state stays one JSON document.
    saved_searches: list[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> AppSettings:
        defaults = cls()
        field_names = defaults.to_dict().keys()
        known = {k: v for k, v in data.items() if k in field_names}
        return cls(**{**defaults.to_dict(), **known})


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


def remove_saved_search(settings: AppSettings, name: str) -> AppSettings:
    """Drop a named search. Removing one that isn't there is a no-op, not
    an error — the UI may be acting on a stale list."""
    return replace(
        settings,
        saved_searches=[s for s in settings.saved_searches if s.get("name") != name],
    )
