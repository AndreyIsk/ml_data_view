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
from PyInstaller.utils.hooks import collect_all, collect_data_files, collect_submodules

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

# psutil (load_stats.py's peak_wset reading) and pywin32 are imported
# normally, but psutil's compiled extension and pywin32's DLLs are loaded
# late enough that they are worth naming.
hiddenimports += ["psutil", "win32api", "win32job", "win32event", "win32process", "pywintypes"]

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


def _exe(analysis, name, *, console):
    return EXE(
        PYZ(analysis.pure),
        analysis.scripts,
        analysis.binaries,
        analysis.datas,
        [],
        name=name,
        debug=False,
        bootloader_ignore_signals=False,
        strip=False,
        upx=False,  # UPX mangles some compiled extensions; not worth the risk here
        upx_exclude=[],
        runtime_tmpdir=None,
        console=console,
        disable_windowed_traceback=False,
        argv_emulation=False,
        target_arch=None,
        codesign_identity=None,
        entitlements_file=None,
        icon="assets/otter.ico",
    )


# The product. Windowed: set console=True temporarily when diagnosing a
# frozen-only crash, since a windowed build discards the traceback that
# would tell you what went wrong.
app_exe = _exe(_analysis("ml_data_view/main.py"), "MLDatasetProfiler", console=False)

# The runtime self-test, from the SAME bundle configuration, as a console
# executable so its report is readable. This is how the frozen build gets
# verified at all: the windowed app cannot report on itself, and the
# failures that matter (a lazily-imported package left out of the bundle,
# a data file at the wrong bundled path, kaleido unable to find a browser)
# are invisible from source. Ship it alongside the app to whoever runs the
# clean-machine test; it is not needed by end users.
selftest_exe = _exe(
    _analysis("scripts/frozen_selftest.py"), "MLDatasetProfiler-selftest", console=True
)
