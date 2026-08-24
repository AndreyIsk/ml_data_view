"""Entry point — launches the pywebview window."""
from __future__ import annotations

from pathlib import Path

import webview

from ml_data_view.api import Api

FRONTEND_DIR = Path(__file__).parent / "frontend"


def main() -> None:
    api = Api()
    webview.create_window(
        "ML Dataset Profiler",
        str(FRONTEND_DIR / "index.html"),
        js_api=api,
        width=1200,
        height=800,
    )
    webview.start()


if __name__ == "__main__":
    main()
