"""Proves the lazy/streaming architecture note in PROJECT_BRIEF.md: every
operation the app performs on a loaded dataset — table preview, row count,
distributions, embedding sampling, media previews — only ever collect()s a
bounded slice or a small streaming aggregation, never the full frame.

Uses a "few hundred thousand rows" fixture (not anywhere near the ~200GB
target this is meant to scale to) big enough that a full collect() would be
unmistakable in the tracked results below.
"""
from __future__ import annotations

import io

import numpy as np
import polars as pl
from PIL import Image

from ml_data_view.chart_data import build_chart_payload
from ml_data_view.data_loader import dataset_info_from_lazy, row_count, scan_dataframe

N_ROWS = 300_000
EMBEDDING_DIMENSIONS = 16
# The largest bounded read any single operation is allowed to make (PCA
# sampling's PCA_SAMPLE_MAX_ROWS is the biggest, at 10_000) — anything at
# or above this, in a dataset of N_ROWS, could only mean a full/near-full
# collect happened somewhere.
MAX_ALLOWED_COLLECTED_HEIGHT = 20_000


def _png_bytes() -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", (4, 4), (255, 0, 0)).save(buffer, format="PNG")
    return buffer.getvalue()


def _write_big_fixture(path) -> None:
    rng = np.random.default_rng(0)
    thumb = _png_bytes()  # one shared blob, reused for every row — cheap to build at this scale
    df = pl.DataFrame(
        {
            "id": np.arange(N_ROWS, dtype=np.int64),
            "score": rng.normal(50, 10, N_ROWS),
            "category": rng.choice(["alpha", "beta", "gamma"], N_ROWS),
            "comment": [f"a fairly unique free-text sentence number {i} of many words" for i in range(N_ROWS)],
            "vec": rng.random((N_ROWS, EMBEDDING_DIMENSIONS)).tolist(),
            "thumb": [thumb] * N_ROWS,
        },
        schema={
            "id": pl.Int64,
            "score": pl.Float64,
            "category": pl.String,
            "comment": pl.String,
            "vec": pl.List(pl.Float64),
            "thumb": pl.Binary,
        },
    )
    df.write_parquet(path)


def test_full_pipeline_never_collects_the_full_unbounded_frame(tmp_path, monkeypatch):
    parquet_path = tmp_path / "big.parquet"
    _write_big_fixture(parquet_path)

    collected_heights: list[int] = []
    original_collect = pl.LazyFrame.collect

    def tracking_collect(self, *args, **kwargs):
        result = original_collect(self, *args, **kwargs)
        collected_heights.append(result.height)
        return result

    monkeypatch.setattr(pl.LazyFrame, "collect", tracking_collect)

    # Table preview (Overview) + row count, via the same path api.py uses.
    lf = scan_dataframe(parquet_path)
    total_rows = row_count(lf)
    dataset_info = dataset_info_from_lazy(lf, parquet_path, total_rows=total_rows)

    # Distributions, Embeddings, Media — one call, like get_chart_data().
    chart_data = build_chart_payload(lf, total_rows=total_rows, dataset_dir=tmp_path)

    # Sanity: this only means something if the fixture is actually as big
    # as claimed, and if collect() was actually exercised at all.
    assert total_rows == N_ROWS
    assert dataset_info.row_count == N_ROWS
    assert len(dataset_info.preview_rows) == 50
    assert collected_heights

    # The real assertion: every single collect(), across the whole
    # pipeline, pulled a bounded number of rows — never anything close to
    # the full N_ROWS frame.
    over_limit = [h for h in collected_heights if h > MAX_ALLOWED_COLLECTED_HEIGHT]
    assert not over_limit, (
        f"a collect() pulled {max(collected_heights)} rows (limit "
        f"{MAX_ALLOWED_COLLECTED_HEIGHT}) out of {N_ROWS} total — looks like "
        f"a full/unbounded materialization, not a bounded slice or streaming "
        f"aggregation. All collected heights: {sorted(collected_heights)}"
    )

    # And the chart data is real, not empty — proves the boundedness above
    # isn't just because nothing ran.
    assert chart_data["null_heatmap"]["total_rows"] == N_ROWS
    assert chart_data["null_heatmap"]["sampled_rows"] <= 500
    assert [h["column"] for h in chart_data["numeric_histograms"]] == ["id", "score"]
    assert [b["column"] for b in chart_data["categorical_bars"]] == ["category"]
    assert [t["column"] for t in chart_data["text_length_histograms"]] == ["comment"]
    assert chart_data["embedding_columns"] == [{"column": "vec", "dimensions": EMBEDDING_DIMENSIONS}]
    projection = chart_data["embedding_projections"][0]
    assert projection["is_sample"] is True
    assert projection["total_rows"] == N_ROWS
    assert len(projection["x"]) <= MAX_ALLOWED_COLLECTED_HEIGHT
    assert chart_data["image_columns"] == [{"column": "thumb", "source": "bytes"}]
    assert len(chart_data["image_previews"][0]["rows"]) == 50
    assert chart_data["image_previews"][0]["total_rows"] == N_ROWS
