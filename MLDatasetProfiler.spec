# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller build for ML Dataset Profiler (Phase 5, part 1: the .exe).

Build with:

    .venv/Scripts/python.exe -m PyInstaller --noconfirm MLDatasetProfiler.spec

Committed so the executable can be rebuilt identically rather than from a
remembered command line. Install from requirements-lock.txt first — that
file pins the exact versions the app was tested against, and a build that
resolves newer ones is not the build that was verified.

Why each block below exists is written next to it; several of these
packages produce an .exe that builds cleanly and then fails at runtime if
the entry is missing, which is the failure mode this file exists to
prevent.
"""
import os

from PyInstaller.utils.hooks import collect_all, collect_submodules

datas = []
binaries = []
hiddenimports = []


def _bundle(package, **kwargs):
    """collect_all() for one package, appended to the three lists above."""
    package_datas, package_binaries, package_hidden = collect_all(package, **kwargs)
    datas.extend(package_datas)
    binaries.extend(package_binaries)
    hiddenimports.extend(package_hidden)


# --- data-layer packages ----------------------------------------------
#
# polars ships a compiled extension plus a separate runtime distribution
# (polars-runtime-32) that PyInstaller does not follow from the `polars`
# import alone. pyarrow carries a large set of .dll/.pyd files that the
# parquet/IPC readers load at runtime, not import time, so the module
# graph misses them.
_bundle("polars")
_bundle("pyarrow")

# --- sklearn ----------------------------------------------------------
#
# Only reached to project an embedding column (embedding_data.py imports
# it lazily, deliberately — see PROJECT_BRIEF.md). A lazy import is
# invisible to PyInstaller's static analysis, so without this the .exe
# builds fine and then fails the moment a dataset with an embedding
# column is opened.
_bundle("sklearn")
_bundle("scipy")
hiddenimports += collect_submodules("sklearn.utils")

# --- reporting: plotly + kaleido --------------------------------------
#
# plotly's figure JSON schema and templates are data files, not code.
# kaleido drives a real Chromium via choreographer; see the note in
# PROJECT_BRIEF.md — the browser itself is NOT bundled here (kaleido does
# not ship one), so PDF export needs a Chromium-based browser present on
# the target machine. What IS bundled is kaleido's and choreographer's own
# package data, including choreographer/resources/last_known_good_chrome.json,
# without which browser discovery cannot even start.
_bundle("plotly")
# kaleido.mocker calls argparse's parse_args() AT IMPORT TIME, so
# PyInstaller's submodule scan (which imports every submodule to walk it)
# exits with SystemExit: 2 and takes the whole build down. It is a test
# double for kaleido's own test suite and nothing here imports it.
_bundle("kaleido", filter_submodules=lambda name: "kaleido.mocker" not in name)
_bundle("choreographer")
_bundle("logistro")
hiddenimports += ["orjson", "narwhals", "tenacity"]

# --- folder watching --------------------------------------------------
#
# watchdog picks its observer at runtime from what the platform offers
# (ReadDirectoryChangesW here, inotify/FSEvents elsewhere), so no static
# import reaches watchdog.observers.winapi and the module graph misses the
# whole backend. Without this the .exe builds and then silently never sees
# a file arrive — the failure mode is a watcher that reports "Watching"
# and never fires.
_bundle("watchdog")

# --- UI shell ---------------------------------------------------------
#
# pywebview picks its backend at runtime, so no static import reaches
# webview.platforms.winforms. That backend then goes through pythonnet
# (`clr`) into System.Windows.Forms, and pywebview's local HTTP server is
# bottle. None of these are reachable from the module graph.
_bundle("webview")
hiddenimports += [
    "webview.platforms.winforms",
    "webview.platforms.win32",
    "clr",
    "clr_loader",
    "bottle",
    "proxy_tools",
]

# --- reportlab --------------------------------------------------------
#
# Fonts and the .pfb/.afm metrics it lays PDF text out with are data files.
_bundle("reportlab")

# --- the app's own data files -----------------------------------------
#
# Placed at the SAME relative paths the source tree uses, because
# resources.resource_path() joins exactly those segments onto
# sys._MEIPASS. Changing a destination here without changing that call
# silently breaks the frozen build only.
datas += [
    ("ml_data_view/frontend", "ml_data_view/frontend"),
    ("assets", "assets"),
]

# psutil (load_stats.py's peak_wset reading, and jobs.py's default memory
# ceiling) and pywin32 are imported normally, but psutil's compiled
# extension and pywin32's DLLs are loaded late enough that they are worth
# naming. pywin32 is no longer only a diagnostics dependency: win32job IS
# the memory ceiling every background job runs under (ml_data_view/jobs.py).
hiddenimports += ["psutil", "win32api", "win32job", "win32event", "win32process", "pywintypes"]

# The worker subprocess's task registry. jobs.run_worker imports it inside
# a function, and it is the only import of it anywhere — so if this ever
# stopped being reachable from the module graph, every heavy operation
# would fail in the frozen build alone, with the error visible only in a
# temp file the parent then deletes.
hiddenimports += ["ml_data_view.job_tasks"]

# Excluded: pytest and friends are test-only, and pulling them in adds
# tens of megabytes of nothing to the shipped .exe.
excludes = ["pytest", "_pytest", "pluggy", "iniconfig", "pypdf", "pymupdf", "PyInstaller"]


def _analysis(entry_script):
    return Analysis(
        [entry_script],
        pathex=["."],
        binaries=binaries,
        datas=datas,
        hiddenimports=hiddenimports,
        hookspath=[],
        hooksconfig={},
        runtime_hooks=[],
        excludes=excludes,
        noarchive=False,
        optimize=0,
    )


# Win32 VERSIONINFO resource, generated by scripts/build_release.py from
# ml_data_view.__version__. Without it, Properties > Details on the .exe is
# blank — no product name, no publisher, no version. Absent when building
# the .spec directly rather than through build_release.py, which is fine:
# PyInstaller just omits the resource.
_version_info = os.path.join("build", "version_info.txt")
version_resource = _version_info if os.path.isfile(_version_info) else None


def _exe(analysis, name, *, console):
    """The EXE half of a onedir build — the payload goes in COLLECT below."""
    return EXE(
        PYZ(analysis.pure),
        analysis.scripts,
        [],
        exclude_binaries=True,  # onedir: binaries/datas are collected, not embedded
        name=name,
        debug=False,
        bootloader_ignore_signals=False,
        strip=False,
        upx=False,  # UPX mangles some compiled extensions; not worth the risk here
        console=console,
        disable_windowed_traceback=False,
        argv_emulation=False,
        target_arch=None,
        codesign_identity=None,
        entitlements_file=None,
        icon="assets/otter.ico",
        version=version_resource,
    )


# ONEDIR, not onefile. onefile is a self-extracting archive that unpacks
# ~193MB to a temp directory on EVERY launch — measured at 5-6 seconds of
# pure extraction before a line of app code runs, which is the loudest
# "this is a wrapped-up script" signal the app had. onedir extracts
# nothing, so startup is normal application launch time, and a directory
# tree is exactly what the installer wants to lay down anyway.
#
# sys._MEIPASS is still set under onedir (it points at _internal/ rather
# than a temp dir), so resources.resource_path() is unaffected.
app_analysis = _analysis("ml_data_view/main.py")
# The product. Windowed: set console=True temporarily when diagnosing a
# frozen-only crash, since a windowed build discards the traceback that
# would tell you what went wrong.
app_exe = _exe(app_analysis, "MLDatasetProfiler", console=False)

# The runtime self-test, from the SAME Analysis configuration, as a console
# executable so its report is readable. This is how the frozen build gets
# verified at all: the windowed app cannot report on itself, and the
# failures that matter (a lazily-imported package left out of the bundle,
# a data file at the wrong bundled path, kaleido unable to find a browser)
# are invisible from source. It is collected into the same directory, so
# it exercises the very bundle the product ships.
selftest_analysis = _analysis("scripts/frozen_selftest.py")
selftest_exe = _exe(selftest_analysis, "MLDatasetProfiler-selftest", console=True)

# One directory holding both executables and a single shared payload.
coll = COLLECT(
    app_exe,
    app_analysis.binaries,
    app_analysis.datas,
    selftest_exe,
    selftest_analysis.binaries,
    selftest_analysis.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name="MLDatasetProfiler",
)
