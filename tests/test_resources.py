"""Frozen-vs-source path resolution (resources.py).

Once PyInstaller freezes the app, `Path(__file__).parent` no longer points
anywhere useful: the code runs out of a PYZ archive and the data files are
unpacked to a temp directory named by `sys._MEIPASS`. Every path that
reaches a bundled file has to go through here, or it works in the source
tree and breaks in the .exe — the exact failure mode that cannot be caught
by running from source.

These tests fake `sys._MEIPASS` rather than requiring a real frozen build,
so the branch that only ever executes inside the .exe still gets exercised.
"""
from __future__ import annotations

import sys
from pathlib import Path

from ml_data_view import resources


def test_not_frozen_when_meipass_is_absent(monkeypatch):
    monkeypatch.delattr(sys, "_MEIPASS", raising=False)
    monkeypatch.setattr(sys, "frozen", False, raising=False)

    assert resources.is_frozen() is False


def test_frozen_when_meipass_is_present(monkeypatch):
    monkeypatch.setattr(sys, "_MEIPASS", r"C:\Temp\_MEI123", raising=False)

    assert resources.is_frozen() is True


def test_resource_path_uses_the_source_tree_when_not_frozen(monkeypatch):
    monkeypatch.delattr(sys, "_MEIPASS", raising=False)

    result = resources.resource_path("assets", "otter.ico")

    assert result == resources.PROJECT_ROOT / "assets" / "otter.ico"


def test_resource_path_uses_meipass_when_frozen(monkeypatch, tmp_path):
    monkeypatch.setattr(sys, "_MEIPASS", str(tmp_path), raising=False)

    result = resources.resource_path("assets", "otter.ico")

    assert result == tmp_path / "assets" / "otter.ico"


def test_resource_path_joins_several_segments(monkeypatch, tmp_path):
    monkeypatch.setattr(sys, "_MEIPASS", str(tmp_path), raising=False)

    assert resources.resource_path("a", "b", "c.txt") == tmp_path / "a" / "b" / "c.txt"


def test_resource_path_with_no_segments_is_the_bundle_root(monkeypatch, tmp_path):
    monkeypatch.setattr(sys, "_MEIPASS", str(tmp_path), raising=False)

    assert resources.resource_path() == tmp_path


# --- writable paths ----------------------------------------------------


def test_writable_dir_is_the_project_root_when_not_frozen(monkeypatch):
    monkeypatch.delattr(sys, "_MEIPASS", raising=False)

    assert resources.writable_dir() == resources.PROJECT_ROOT


def test_writable_dir_is_not_inside_the_bundle_when_frozen(monkeypatch, tmp_path):
    """_MEIPASS is a temp extraction directory that is deleted on exit —
    anything written there is lost, and on a locked-down machine the write
    can fail outright. A frozen app must not put its logs there."""
    monkeypatch.setattr(sys, "_MEIPASS", str(tmp_path), raising=False)

    target = resources.writable_dir()

    assert tmp_path not in target.parents
    assert target != tmp_path


def test_writable_dir_when_frozen_is_the_per_user_app_data_dir(monkeypatch, tmp_path):
    monkeypatch.setattr(sys, "_MEIPASS", str(tmp_path), raising=False)

    from ml_data_view.settings import app_data_dir

    assert resources.writable_dir() == app_data_dir()


# --- the real assets the app depends on --------------------------------


def test_the_frontend_entry_point_resolves_from_source():
    assert (resources.resource_path("ml_data_view", "frontend", "index.html")).is_file()


def test_the_icon_resolves_from_source():
    assert (resources.resource_path("assets", "otter.ico")).is_file()


def test_project_root_is_the_repository_root():
    """PROJECT_ROOT anchors every non-frozen lookup; if it drifts up or
    down a level, every bundled path silently goes with it."""
    assert (resources.PROJECT_ROOT / "requirements.txt").is_file()
    assert (resources.PROJECT_ROOT / "ml_data_view").is_dir()


def test_main_resolves_its_paths_through_resources():
    """Regression guard: main.py used Path(__file__) anchors, which point
    into the PYZ archive once frozen."""
    source = Path(resources.PROJECT_ROOT / "ml_data_view" / "main.py").read_text(encoding="utf-8")
    # Comments stripped: main.py explains in prose why a Path(__file__)
    # anchor would be wrong, and that sentence is not a violation.
    code = "\n".join(
        line for line in source.splitlines() if not line.lstrip().startswith("#")
    )

    assert "resource_path" in code
    assert "Path(__file__)" not in code


# --- the spec file's bundled paths must match resource_path() ----------

SPEC = resources.PROJECT_ROOT / "MLDatasetProfiler.spec"


def _spec_datas() -> list[tuple[str, str]]:
    """The app's own (source, destination) data entries from the .spec."""
    import re

    text = SPEC.read_text(encoding="utf-8")
    block = text.split("datas += [", 1)[1].split("]", 1)[0]
    return re.findall(r'\(\s*"([^"]+)"\s*,\s*"([^"]+)"\s*\)', block)


def test_the_spec_bundles_the_frontend_and_assets():
    destinations = {dest for _src, dest in _spec_datas()}

    assert "ml_data_view/frontend" in destinations
    assert "assets" in destinations


def test_every_bundled_destination_matches_its_source_path():
    """resource_path() joins the SOURCE-tree path onto sys._MEIPASS, so a
    .spec entry whose destination differs from its source silently breaks
    the frozen build and only the frozen build."""
    for source, destination in _spec_datas():
        assert source == destination, (
            f"{source!r} is bundled at {destination!r}; resource_path() would look "
            f"for it at {source!r} inside the bundle"
        )


def test_the_spec_sets_the_otter_icon():
    assert 'icon="assets/otter.ico"' in SPEC.read_text(encoding="utf-8")


def test_the_lockfile_pins_exact_versions():
    """A build that resolves newer versions is not the build that was
    verified, so every line must be an == pin."""
    lock = (resources.PROJECT_ROOT / "requirements-lock.txt").read_text(encoding="utf-8")
    lines = [l.strip() for l in lock.splitlines() if l.strip() and not l.startswith("#")]

    assert lines, "lockfile is empty"
    unpinned = [l for l in lines if "==" not in l]
    assert not unpinned, f"not pinned to an exact version: {unpinned}"


def test_the_lockfile_covers_the_packages_the_build_depends_on():
    lock = (resources.PROJECT_ROOT / "requirements-lock.txt").read_text(encoding="utf-8").lower()

    for package in ["polars", "pyarrow", "scikit-learn", "plotly", "kaleido", "choreographer", "pywebview"]:
        assert f"{package}==" in lock, f"{package} missing from the lockfile"
