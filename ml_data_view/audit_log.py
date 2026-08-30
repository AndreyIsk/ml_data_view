"""Local, plain-text audit log for file-open actions.

One JSON object per line (JSON Lines), written to a file in the same
OS-appropriate app-data directory as settings.json. No network calls, ever —
this module only ever touches the local filesystem.
"""
from __future__ import annotations

import getpass
import json
from datetime import datetime, timezone
from pathlib import Path

from ml_data_view.settings import AppSettings, app_data_dir

LOG_FILENAME = "audit_log.jsonl"


def audit_log_file_path(base_dir: Path | None = None) -> Path:
    """`base_dir` overrides the app-data directory — used by tests so they
    never touch the real user's app-data directory."""
    return (base_dir or app_data_dir()) / LOG_FILENAME


def write_audit_entry(
    action: str,
    filename: str,
    *,
    base_dir: Path | None = None,
    username: str | None = None,
    timestamp: str | None = None,
) -> None:
    """Append one JSON-line entry: timestamp, OS username, action, filename."""
    path = audit_log_file_path(base_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    entry = {
        "timestamp": timestamp or datetime.now(timezone.utc).isoformat(),
        "user": username or getpass.getuser(),
        "action": action,
        "filename": filename,
    }
    with path.open("a", encoding="utf-8") as log_file:
        log_file.write(json.dumps(entry) + "\n")


def _record_if_enabled(
    action: str,
    filename: str,
    settings: AppSettings,
    *,
    base_dir: Path | None = None,
    username: str | None = None,
    timestamp: str | None = None,
) -> None:
    if not settings.audit_log_enabled:
        return
    write_audit_entry(action, filename, base_dir=base_dir, username=username, timestamp=timestamp)


def record_file_open(
    filename: str,
    settings: AppSettings,
    *,
    base_dir: Path | None = None,
    username: str | None = None,
    timestamp: str | None = None,
) -> None:
    """Audit-log a file-open action, but only when settings.audit_log_enabled
    — a no-op (no file even created) otherwise."""
    _record_if_enabled(
        "file_open", filename, settings, base_dir=base_dir, username=username, timestamp=timestamp
    )


def record_report_generated(
    filename: str,
    settings: AppSettings,
    *,
    base_dir: Path | None = None,
    username: str | None = None,
    timestamp: str | None = None,
) -> None:
    """Audit-log a report-generation action, but only when
    settings.audit_log_enabled — a no-op otherwise. `filename` is the
    source dataset's name (the same field meaning as record_file_open's),
    not the generated PDF's path."""
    _record_if_enabled(
        "report_generated", filename, settings, base_dir=base_dir, username=username, timestamp=timestamp
    )


def record_folder_open(
    folder_name: str,
    settings: AppSettings,
    *,
    base_dir: Path | None = None,
    username: str | None = None,
    timestamp: str | None = None,
) -> None:
    """Audit-log a folder-open action (Stage 2 folder loading), but only
    when settings.audit_log_enabled — a no-op otherwise. Mirrors
    record_file_open; `folder_name` fills the same "filename" field."""
    _record_if_enabled(
        "folder_open", folder_name, settings, base_dir=base_dir, username=username, timestamp=timestamp
    )
