"""Runtime self-test, meant to be run from INSIDE the frozen build.

A PyInstaller build that compiles is not a build that runs. The failures
that matter here are all invisible from source: a lazily-imported package
that never made it into the bundle, a data file whose bundled path does
not match what resource_path() joins, a binary that kaleido shells out to
and cannot find, a log directory that is read-only once packaged.

This is built as its own console executable next to the windowed app
(see MLDatasetProfiler.spec) from the same Analysis, so it exercises the
same bundle the product ships. Run it on a clean machine to find out what
actually works there:

    MLDatasetProfiler-selftest.exe

Exit code is 0 only if every check passed.
"""
from __future__ import annotations

import sys
import tempfile
import traceback
from pathlib import Path

# When frozen, ml_data_view lives in the PYZ archive and imports normally.
# From source this file sits in scripts/, so the repository root has to go
# on sys.path first — running it from source is the useful baseline to
# compare a frozen run against.
if not hasattr(sys, "_MEIPASS"):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

RESULTS: list[tuple[str, bool, str]] = []


def check(name: str):
    """Decorator: run a check, record pass/fail, never let one abort the rest."""

    def decorate(fn):
        try:
            detail = fn() or ""
            RESULTS.append((name, True, str(detail)))
        except Exception as exc:
            RESULTS.append((name, False, f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}"))
        return fn

    return decorate


# --- 1. are we actually frozen, and where did the data go? -------------


@check("frozen / _MEIPASS")
def _frozen():
    from ml_data_view.resources import bundle_root, is_frozen

    if not is_frozen():
        return f"NOT FROZEN (running from source) — bundle_root={bundle_root()}"
    return f"frozen, _MEIPASS={bundle_root()}"


@check("bundled frontend + assets resolve")
def _assets():
    from ml_data_view.resources import resource_path

    wanted = [
        ("ml_data_view", "frontend", "index.html"),
        ("ml_data_view", "frontend", "app.js"),
        ("ml_data_view", "frontend", "style.css"),
        ("ml_data_view", "frontend", "findings.js"),
        ("ml_data_view", "frontend", "plotly.min.js"),
        ("assets", "otter.ico"),
        ("assets", "icons", "otter-256.png"),
    ]
    missing = [str(Path(*w)) for w in wanted if not resource_path(*w).is_file()]
    if missing:
        raise FileNotFoundError(f"not in bundle: {missing}")
    return f"all {len(wanted)} present"


@check("writable dir is outside the bundle")
def _writable():
    from ml_data_view.resources import bundle_root, is_frozen, writable_dir

    target = writable_dir()
    target.mkdir(parents=True, exist_ok=True)
    probe = target / "_selftest_write_probe.tmp"
    probe.write_text("ok", encoding="utf-8")
    probe.unlink()
    if is_frozen() and bundle_root() in target.parents:
        raise AssertionError(f"writable dir {target} is inside the bundle — it vanishes on exit")
    return f"{target} (writable)"


# --- 2. the lazy/streaming data layer ----------------------------------


@check("polars + pyarrow: lazy scan stays bounded")
def _data_layer():
    import polars as pl

    from ml_data_view.data_loader import dataset_info_from_lazy, row_count, scan_dataframe

    n = 50_000
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "probe.parquet"
        pl.DataFrame(
            {"id": range(n), "label": ["a", "b"] * (n // 2)}
        ).write_parquet(path)

        collected: list[int] = []
        original = pl.LazyFrame.collect

        def tracking(self, *args, **kwargs):
            result = original(self, *args, **kwargs)
            collected.append(result.height)
            return result

        pl.LazyFrame.collect = tracking
        try:
            lf = scan_dataframe(path)
            total = row_count(lf)
            info = dataset_info_from_lazy(lf, path, total_rows=total)
        finally:
            pl.LazyFrame.collect = original

    if total != n:
        raise AssertionError(f"row_count returned {total}, expected {n}")
    biggest = max(collected) if collected else 0
    if biggest > 20_000:
        raise AssertionError(f"a collect() pulled {biggest} of {n} rows — not bounded")
    return f"{total:,} rows counted; largest collect {biggest} rows; preview {len(info.preview_rows)} rows"


@check("sklearn (lazy import) projects an embedding")
def _sklearn():
    import polars as pl

    from ml_data_view.embedding_data import embedding_projection_data

    frame = pl.DataFrame(
        {"vec": [[float(i + j) for i in range(16)] for j in range(50)]},
        schema={"vec": pl.List(pl.Float64)},
    ).lazy()
    result = embedding_projection_data(frame, "vec", categorical_columns=[])
    if len(result["x"]) != 50:
        raise AssertionError(f"expected 50 projected points, got {len(result['x'])}")
    return f"PCA projected {len(result['x'])} points"


# --- 3. reporting: kaleido is the one most likely to be missing --------


@check("kaleido: browser discovery")
def _kaleido_browser():
    from choreographer.browsers.chromium import Chromium

    path = Chromium.find_browser(skip_local=False)
    if not path:
        raise RuntimeError(
            "no Chromium-based browser found. kaleido does not bundle one; PDF chart "
            "export needs Chrome/Edge/Chromium installed on this machine."
        )
    return f"using {path}"


@check("kaleido: rasterize a chart to PNG")
def _kaleido_render():
    from ml_data_view.report_charts import figure_png_bytes, numeric_histogram_figure

    figure = numeric_histogram_figure(
        {"column": "probe", "bin_edges": [0.0, 1.0, 2.0], "counts": [3, 5], "x_scale": "linear"}
    )
    png = figure_png_bytes(figure)
    if not png or not png.startswith(b"\x89PNG"):
        raise AssertionError(f"not a PNG ({len(png) if png else 0} bytes)")
    return f"{len(png):,} bytes of PNG"


@check("reportlab: write a full PDF report")
def _pdf():
    import polars as pl

    from ml_data_view.chart_data import build_chart_payload
    from ml_data_view.data_loader import dataset_info_from_lazy, scan_dataframe
    from ml_data_view.report import generate_report

    with tempfile.TemporaryDirectory() as tmp:
        source = Path(tmp) / "probe.parquet"
        pl.DataFrame({"score": [1.0, 2.0, 3.0, 4.0], "label": ["a", "b", "a", "b"]}).write_parquet(
            source
        )
        lf = scan_dataframe(source)
        info = dataset_info_from_lazy(lf, source).to_dict()
        charts = build_chart_payload(lf, total_rows=4)
        out = Path(tmp) / "report.pdf"
        generate_report(
            str(out),
            dataset_info=info,
            chart_data=charts,
            sections=["overview", "distributions"],
        )
        if not out.exists() or out.stat().st_size < 1000:
            raise AssertionError("PDF missing or implausibly small")
        size = out.stat().st_size
    return f"{size:,} byte PDF written"


# --- 4. the API layer and its worker threads ----------------------------
#
# Everything above calls the headless functions directly. These go through
# Api, the object pywebview binds to the JS bridge, including the two
# background jobs — worker threads are a common frozen-build casualty.


@check("Api: open a file, build charts, run findings")
def _api():
    import polars as pl

    from ml_data_view.api import Api

    with tempfile.TemporaryDirectory() as tmp:
        source = Path(tmp) / "probe.parquet"
        pl.DataFrame(
            {
                "score": [float(i) for i in range(200)],
                "label": ["a"] * 190 + ["b"] * 10,
                "dead": [None] * 200,
            },
            schema_overrides={"dead": pl.Float64},
        ).write_parquet(source)

        api = Api(settings_dir=Path(tmp) / "settings")
        info = api.load_file(str(source))
        if info.get("error"):
            raise RuntimeError(info["error"])
        api.get_chart_data()
        findings = api.run_instant_checks()
        if findings.get("pending"):
            raise AssertionError("chart data did not cache")
        critical = findings["summary"]["critical"]
        stat_line = info["load_stats"]["line"]
    return f"{info['row_count']} rows, {critical} critical finding(s); stat line: {stat_line}"


@check("Api: background jobs (folder load + chart build) on worker threads")
def _jobs():
    import polars as pl

    from ml_data_view.api import Api

    with tempfile.TemporaryDirectory() as tmp:
        folder = Path(tmp) / "parts"
        folder.mkdir()
        for index in range(3):
            pl.DataFrame({"v": list(range(100))}).write_parquet(folder / f"part{index}.parquet")

        api = Api(settings_dir=Path(tmp) / "settings")
        discovery = api.discover_folder(str(folder))
        if discovery.get("error"):
            raise RuntimeError(discovery["error"])
        api.start_folder_load(0)
        state = api.wait_for_folder_load(timeout=120)
        if not state["finished"] or state.get("error"):
            raise RuntimeError(f"folder load failed: {state.get('error')}")
        rows = state["result"]["row_count"]

        api.start_chart_build()
        chart_state = api.wait_for_chart_build(timeout=180)
        if chart_state.get("error"):
            raise RuntimeError(f"chart build failed: {chart_state['error']}")

        api.start_deep_checks()
        deep = api.wait_for_deep_checks(timeout=180)
        done = sum(1 for c in deep["checks"] if c["status"] == "done")
    return f"folder load {rows} rows across 3 files; chart build ok; {done} deep checks ran"


# --- 5. settings / audit log persistence --------------------------------


@check("settings round-trip in the per-user app-data dir")
def _settings():
    from ml_data_view.settings import AppSettings, load_settings, save_settings, settings_file_path

    with tempfile.TemporaryDirectory() as tmp:
        save_settings(AppSettings(theme="dark", audit_log_enabled=True), Path(tmp))
        loaded = load_settings(Path(tmp))
        if loaded.theme != "dark":
            raise AssertionError(f"theme did not persist: {loaded.theme}")
    return f"default location would be {settings_file_path()}"


@check("checkpoint log writes without crashing")
def _checkpoint():
    from ml_data_view.checkpoint_log import LOG_PATH, checkpoint
    from ml_data_view.resources import bundle_root, is_frozen

    if is_frozen() and bundle_root() in LOG_PATH.parents:
        raise AssertionError(f"log path {LOG_PATH} is inside the bundle")
    checkpoint("frozen_selftest", "probe")
    return f"wrote to {LOG_PATH}"


# --- report -------------------------------------------------------------


def main() -> int:
    print("=" * 72)
    print("ML Dataset Profiler — frozen runtime self-test")
    print(f"python {sys.version.split()[0]}   frozen={hasattr(sys, '_MEIPASS')}")
    print(f"executable: {sys.executable}")
    print("=" * 72)

    failed = 0
    for name, ok, detail in RESULTS:
        status = "PASS" if ok else "FAIL"
        if not ok:
            failed += 1
        first, _, rest = detail.partition("\n")
        print(f"[{status}] {name}")
        if first:
            print(f"       {first}")
        if not ok and rest:
            for line in rest.strip().splitlines()[-6:]:
                print(f"       | {line}")

    print("=" * 72)
    print(f"{len(RESULTS) - failed}/{len(RESULTS)} checks passed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
