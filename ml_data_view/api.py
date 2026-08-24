"""pywebview-facing API layer.

Thin wrapper around data_loader.py — exposes methods pywebview binds to
`window.pywebview.api` in the frontend. Keep this file free of business
logic so data_loader.py stays importable/testable without a window.

Known Phase 0 limitation: date/datetime columns are not yet stringified,
so a source file containing them may fail to serialize across the js_api
bridge. Not a concern for the CSV/Parquet/Arrow test fixtures used here;
revisit if/when a dataset with date columns is opened.
"""
from __future__ import annotations

from typing import Any

import polars as pl
import webview

from ml_data_view.chart_data import build_chart_payload
from ml_data_view.data_loader import (
    UnsupportedFileTypeError,
    dataset_info_from_df,
    read_dataframe,
)


class Api:
    """Methods exposed to the JS frontend via pywebview's js_api binding."""

    def __init__(self) -> None:
        self._current_df: pl.DataFrame | None = None

    def open_file_dialog(self) -> str | None:
        """Show a native file-open dialog restricted to supported types.

        Returns the selected path, or None if the user cancelled. Requires
        an active pywebview window, so this is not unit-tested headlessly —
        verify manually per the Phase 0 plan's manual-verification task.
        """
        window = webview.windows[0]
        result = window.create_file_dialog(
            webview.FileDialog.OPEN,
            file_types=(
                "Dataset files (*.parquet;*.arrow;*.csv)",
                "All files (*.*)",
            ),
        )
        if not result:
            return None
        return result[0]

    def load_file(self, path: str) -> dict[str, Any]:
        """Load a dataset file, cache it, and return its profile as a dict.

        On failure, returns {"error": "<message>"} instead of raising, so
        the frontend always gets a plain dict back across the js_api bridge.
        Caches the loaded DataFrame on success so get_chart_data() can reuse
        it without re-reading the file from disk.
        """
        try:
            df = read_dataframe(path)
        except (UnsupportedFileTypeError, FileNotFoundError) as exc:
            return {"error": str(exc)}
        self._current_df = df
        return dataset_info_from_df(df, path).to_dict()

    def get_chart_data(self) -> dict[str, Any]:
        """Return Phase 1 chart data for the most recently loaded dataset."""
        if self._current_df is None:
            return {"error": "No dataset loaded yet. Open a file first."}
        return build_chart_payload(self._current_df)
