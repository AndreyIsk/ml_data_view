"""Where files live, in the source tree and inside the frozen .exe.

PyInstaller changes two things that silently break path handling, and
neither shows up when running from source:

  1. Code runs out of a PYZ archive, so `Path(__file__).parent` no longer
     names a real directory on disk. Bundled data files are unpacked to a
     temp directory whose path is in `sys._MEIPASS`.
  2. That temp directory is deleted when the process exits, and on a
     locked-down machine it may not be writable at all. Anything the app
     WRITES — logs, settings — must go somewhere else.

So there are two questions, and they have different answers: "where is
this bundled file?" (resource_path) and "where may I write?"
(writable_dir). Every path that reaches a shipped asset goes through the
first; nothing writes to the bundle.

See PROJECT_BRIEF.md's Phase 5 notes for the build itself.
"""
from __future__ import annotations

import sys
from pathlib import Path

# ml_data_view/resources.py -> ml_data_view/ -> repository root. Used only
# when NOT frozen; the frozen branch never touches it.
PROJECT_ROOT = Path(__file__).resolve().parent.parent


def is_frozen() -> bool:
    """True when running from a PyInstaller bundle.

    `sys._MEIPASS` is the reliable signal — PyInstaller sets it in both
    onefile and onedir mode. `sys.frozen` alone is set by other freezers
    too and does not tell us where the data went.
    """
    return hasattr(sys, "_MEIPASS")


def bundle_root() -> Path:
    """Directory that bundled data files are read from."""
    meipass = getattr(sys, "_MEIPASS", None)
    return Path(meipass) if meipass is not None else PROJECT_ROOT


def resource_path(*segments: str) -> Path:
    """Absolute path to a bundled data file, frozen or not.

    Segments are given relative to the repository root, so a call reads
    the same as the path in the source tree:

        resource_path("assets", "otter.ico")
        resource_path("ml_data_view", "frontend", "index.html")

    The .spec file's `datas` must place each of these at the SAME relative
    path inside the bundle, or this returns a path to nothing.
    """
    return bundle_root().joinpath(*segments)


def writable_dir() -> Path:
    """A directory the app may write to.

    Not the bundle: `sys._MEIPASS` is a temp extraction directory that is
    deleted on exit, so anything written there is lost, and on a
    locked-down machine the write can fail outright and take down whatever
    was logging. Frozen builds write to the same per-user app-data
    directory settings.py already uses, which is guaranteed writable and
    survives a restart.

    From source this stays the repository root, so a developer's
    diagnostic output keeps landing where it always has.
    """
    if not is_frozen():
        return PROJECT_ROOT
    # Imported here rather than at module scope: settings.py is a heavier
    # import than this module wants at startup, and the frozen branch is
    # the only one that needs it.
    from ml_data_view.settings import app_data_dir

    return app_data_dir()
