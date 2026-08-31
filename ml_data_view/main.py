"""Entry point — launches the pywebview window."""
from __future__ import annotations

from pathlib import Path

import webview

from ml_data_view.api import Api

FRONTEND_DIR = Path(__file__).parent / "frontend"
# Multi-size .ico (16/32/48/64/128/256), built from assets/otter.png by
# scripts/make_icons.py. Windows picks a different frame from the same file
# for the title bar, taskbar and Alt-Tab, which is why it is an .ico rather
# than a PNG — .NET's System.Drawing.Icon, which pywebview hands this to,
# only accepts .ico anyway.
ICON_PATH = Path(__file__).resolve().parents[1] / "assets" / "otter.ico"


def main() -> None:
    api = Api()
    webview.create_window(
        "ML Dataset Profiler",
        str(FRONTEND_DIR / "index.html"),
        js_api=api,
        width=1200,
        height=800,
    )
    # pywebview's own docs say `icon` is "supported only on GTK and QT", but
    # that is out of date: the installed winforms backend (6.2.1,
    # platforms/winforms.py) reads the same _state['icon'] and assigns
    # self.Icon = Icon(path), falling back to extracting python.exe's icon
    # when unset — which is why the window used to show a Python logo.
    # Verified against a running window, not just the source.
    #
    # A missing file would make that fall back silently, so it is passed
    # only when it is actually there.
    icon = str(ICON_PATH) if ICON_PATH.is_file() else None
    webview.start(icon=icon)


if __name__ == "__main__":
    main()
