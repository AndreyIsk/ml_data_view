"""Deep dataset checks (deep_checks.py) — the ones that read data.

Unlike findings.py's instant checks, these touch the LazyFrame, so every
one has to obey PROJECT_BRIEF.md's bounded-slice-or-streaming-aggregation
rule. The last test in this file enforces that directly, with the same
collect()-monitoring harness tests/test_lazy_loading.py uses.
"""
from __future__ import annotations

import io

import numpy as np
import polars as pl
import pytest
from PIL import Image

from ml_data_view.deep_checks import (
    DEEP_CHECKS,
    DeepContext,
    SENTINEL_MIN_SHARE,
    SENTINEL_VALUES,
    duplicate_rows,
    sentinel_values,
    unresolvable_images,
)


def _png_bytes() -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", (4, 4), (0, 128, 255)).save(buffer, format="PNG")
    return buffer.getvalue()


# --- duplicate_rows ----------------------------------------------------


def test_duplicate_rows_finds_planted_duplicates(tmp_path):
    df = pl.DataFrame({"a": [1, 2, 3, 1, 2], "b": ["x", "y", "z", "x", "y"]})
    path = tmp_path / "dupes.parquet"
    df.write_parquet(path)

    findings = duplicate_rows(DeepContext(lf=pl.scan_parquet(path)))

    assert len(findings) == 1
    assert findings[0].severity == "warning"
    # Two rows are repeats of an earlier row: (1,"x") and (2,"y").
    assert "2 duplicate rows" in findings[0].detail


def test_duplicate_rows_silent_when_every_row_is_unique(tmp_path):
    path = tmp_path / "unique.parquet"
    pl.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]}).write_parquet(path)

    assert duplicate_rows(DeepContext(lf=pl.scan_parquet(path))) == []


def test_duplicate_rows_states_the_sample_it_actually_scanned(tmp_path):
    """The count is over a bounded sample, never the dataset. Quoting it as
    a dataset-wide figure is exactly the mistake the wording must prevent."""
    path = tmp_path / "big.parquet"
    pl.DataFrame({"a": [1] * 50}).write_parquet(path)

    findings = duplicate_rows(DeepContext(lf=pl.scan_parquet(path)), sample_rows=10)

    detail = findings[0].detail
    assert "first 10" in detail
    assert "Not a dataset-wide count" in detail


def test_duplicate_rows_never_reads_past_the_sample(tmp_path):
    """A sample smaller than the file must bound what is compared: 10 rows
    scanned can yield at most 9 duplicates, never the file's 49."""
    path = tmp_path / "big.parquet"
    pl.DataFrame({"a": [1] * 50}).write_parquet(path)

    findings = duplicate_rows(DeepContext(lf=pl.scan_parquet(path)), sample_rows=10)

    assert "9 duplicate rows" in findings[0].detail


def test_duplicate_rows_excludes_binary_and_embedding_columns_and_says_so(tmp_path):
    """100,000 rows of image blobs or 768-float embeddings is not a bounded
    read. Those columns are left out of row identity — which changes what
    "duplicate" means, so the finding has to declare it."""
    path = tmp_path / "wide.parquet"
    pl.DataFrame(
        {
            "a": [1, 1],
            "blob": [_png_bytes(), b"different bytes entirely"],
            "vec": [[1.0] * 16, [2.0] * 16],
        },
        schema={"a": pl.Int64, "blob": pl.Binary, "vec": pl.List(pl.Float64)},
    ).write_parquet(path)

    findings = duplicate_rows(DeepContext(lf=pl.scan_parquet(path)))

    # Identical on 'a' alone, so they count as duplicates once blob/vec are out.
    assert "1 duplicate rows" in findings[0].detail or "1 duplicate row" in findings[0].detail
    assert "blob" in findings[0].detail and "vec" in findings[0].detail


def test_duplicate_rows_silent_when_every_column_was_excluded(tmp_path):
    """Nothing left to compare is not "no duplicates" — it is no answer, so
    the check must not claim a clean result."""
    path = tmp_path / "blobs.parquet"
    pl.DataFrame({"blob": [b"a", b"a"]}, schema={"blob": pl.Binary}).write_parquet(path)

    assert duplicate_rows(DeepContext(lf=pl.scan_parquet(path))) == []


# --- sentinel_values ---------------------------------------------------


def test_sentinel_values_finds_a_missing_code_posing_as_data(tmp_path):
    path = tmp_path / "sentinel.parquet"
    pl.DataFrame({"age": [25, 30, -999, 41, -999, 33, 28, 39, 44, 51]}).write_parquet(path)

    findings = sentinel_values(DeepContext(lf=pl.scan_parquet(path), files=[path]))

    assert len(findings) == 1
    assert findings[0].severity == "warning"
    assert findings[0].columns == ["age"]
    assert "-999" in findings[0].detail


def test_sentinel_values_counts_exactly_across_several_files(tmp_path):
    """Sentinel counts add exactly across files — unlike the sampled checks,
    this one is not an estimate."""
    files = []
    for index in range(3):
        path = tmp_path / f"part{index}.parquet"
        pl.DataFrame({"v": [-999, -999, 1, 2, 3, 4, 5, 6, 7, 8]}).write_parquet(path)
        files.append(path)

    lf = pl.concat([pl.scan_parquet(f) for f in files], how="vertical")
    findings = sentinel_values(DeepContext(lf=lf, files=files))

    assert "6" in findings[0].detail  # 2 per file x 3 files, exactly


def test_sentinel_values_silent_below_the_share_threshold(tmp_path):
    """A handful of -1s is ordinary data. A sentinel encoding "missing"
    shows up in a meaningful fraction of rows."""
    path = tmp_path / "rare.parquet"
    values = [-1] + list(range(1, 1000))

    pl.DataFrame({"v": values}).write_parquet(path)
    findings = sentinel_values(DeepContext(lf=pl.scan_parquet(path), files=[path]))

    assert findings == []


def test_sentinel_values_fires_at_a_share_above_the_threshold(tmp_path):
    path = tmp_path / "common.parquet"
    total = 1000
    planted = int(total * SENTINEL_MIN_SHARE) + 10
    values = [-1] * planted + list(range(total - planted))

    pl.DataFrame({"v": values}).write_parquet(path)
    findings = sentinel_values(DeepContext(lf=pl.scan_parquet(path), files=[path]))

    assert len(findings) == 1


def test_sentinel_values_ignores_string_columns(tmp_path):
    path = tmp_path / "strings.parquet"
    pl.DataFrame({"s": ["-999"] * 10}).write_parquet(path)

    assert sentinel_values(DeepContext(lf=pl.scan_parquet(path), files=[path])) == []


def test_sentinel_values_silent_on_a_clean_numeric_column(tmp_path):
    path = tmp_path / "clean.parquet"
    pl.DataFrame({"v": list(range(100))}).write_parquet(path)

    assert sentinel_values(DeepContext(lf=pl.scan_parquet(path), files=[path])) == []


@pytest.mark.parametrize("sentinel", SENTINEL_VALUES)
def test_every_catalogued_sentinel_is_detected(tmp_path, sentinel):
    path = tmp_path / f"s{abs(sentinel)}.parquet"
    pl.DataFrame({"v": [sentinel] * 20 + list(range(80))}).write_parquet(path)

    findings = sentinel_values(DeepContext(lf=pl.scan_parquet(path), files=[path]))

    assert str(sentinel) in findings[0].detail


# --- unresolvable_images -----------------------------------------------


def test_unresolvable_images_separates_missing_files_from_missing_values(tmp_path):
    """This is the check's whole reason to exist: the instant check can
    only say "no usable image", because image_previews records one null
    thumbnail for three different causes."""
    good = tmp_path / "real.png"
    good.write_bytes(_png_bytes())
    path = tmp_path / "images.parquet"
    pl.DataFrame({"img": ["real.png", "gone.png", None, "also_gone.png"]}).write_parquet(path)

    findings = unresolvable_images(
        DeepContext(
            lf=pl.scan_parquet(path),
            dataset_dir=tmp_path,
            image_columns=[{"column": "img", "source": "path"}],
        )
    )

    detail = findings[0].detail
    assert "2" in detail  # two paths that do not resolve
    assert "missing file" in detail
    assert "1" in detail  # one null value, reported separately


def test_unresolvable_images_counts_undecodable_bytes(tmp_path):
    path = tmp_path / "bytes.parquet"
    pl.DataFrame(
        {"img": [_png_bytes(), b"not an image at all"]}, schema={"img": pl.Binary}
    ).write_parquet(path)

    findings = unresolvable_images(
        DeepContext(
            lf=pl.scan_parquet(path), image_columns=[{"column": "img", "source": "bytes"}]
        )
    )

    assert "undecodable" in findings[0].detail


def test_unresolvable_images_silent_when_everything_resolves(tmp_path):
    good = tmp_path / "real.png"
    good.write_bytes(_png_bytes())
    path = tmp_path / "ok.parquet"
    pl.DataFrame({"img": ["real.png"]}).write_parquet(path)

    findings = unresolvable_images(
        DeepContext(
            lf=pl.scan_parquet(path),
            dataset_dir=tmp_path,
            image_columns=[{"column": "img", "source": "path"}],
        )
    )

    assert findings == []


def test_unresolvable_images_states_the_sample_it_scanned(tmp_path):
    path = tmp_path / "many.parquet"
    pl.DataFrame({"img": ["gone.png"] * 100}).write_parquet(path)

    findings = unresolvable_images(
        DeepContext(
            lf=pl.scan_parquet(path),
            dataset_dir=tmp_path,
            image_columns=[{"column": "img", "source": "path"}],
        ),
        sample_rows=20,
    )

    assert "first 20" in findings[0].detail


def test_unresolvable_images_does_nothing_without_image_columns(tmp_path):
    path = tmp_path / "plain.parquet"
    pl.DataFrame({"a": [1, 2]}).write_parquet(path)

    assert unresolvable_images(DeepContext(lf=pl.scan_parquet(path))) == []


# --- catalogue ---------------------------------------------------------


def test_deep_catalogue_order_is_stable():
    assert [spec.id for spec in DEEP_CHECKS] == [
        "duplicate_rows",
        "sentinel_values",
        "unresolvable_images",
    ]


def test_every_deep_check_is_marked_deep_and_titled():
    assert all(spec.kind == "deep" and spec.title for spec in DEEP_CHECKS)


# --- boundedness -------------------------------------------------------

_N_ROWS = 300_000
# The largest bounded read any deep check may make. DUPLICATE_SAMPLE_ROWS
# is the biggest at 100,000; anything at or above this, against a
# _N_ROWS-row fixture, could only mean an unbounded materialization.
_MAX_ALLOWED_COLLECTED_HEIGHT = 120_000


def test_no_deep_check_ever_collects_the_full_frame(tmp_path, monkeypatch):
    """The same harness as tests/test_lazy_loading.py: every collect() a
    deep check makes must pull a bounded number of rows, never anything
    near the full frame."""
    rng = np.random.default_rng(0)
    path = tmp_path / "big.parquet"
    pl.DataFrame(
        {
            "id": np.arange(_N_ROWS, dtype=np.int64),
            "score": rng.normal(50, 10, _N_ROWS),
            "label": rng.choice(["a", "b"], _N_ROWS),
        }
    ).write_parquet(path)

    collected_heights: list[int] = []
    original_collect = pl.LazyFrame.collect

    def tracking_collect(self, *args, **kwargs):
        result = original_collect(self, *args, **kwargs)
        collected_heights.append(result.height)
        return result

    monkeypatch.setattr(pl.LazyFrame, "collect", tracking_collect)

    ctx = DeepContext(lf=pl.scan_parquet(path), files=[path], dataset_dir=tmp_path)
    for spec in DEEP_CHECKS:
        spec.fn(ctx)

    assert collected_heights, "no collect() ran — the assertion below would be vacuous"
    over_limit = [h for h in collected_heights if h > _MAX_ALLOWED_COLLECTED_HEIGHT]
    assert not over_limit, (
        f"a deep check collected {max(collected_heights)} rows out of {_N_ROWS} — "
        f"that is an unbounded materialization, not a bounded slice or a streaming "
        f"aggregation. All collected heights: {sorted(collected_heights)}"
    )
