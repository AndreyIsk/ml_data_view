import io

import polars as pl
from PIL import Image

from ml_data_view.image_data import (
    detect_image_columns,
    image_previews_data,
    media_grid_png,
    resolve_image_full,
)


def _png_bytes(color: tuple[int, int, int] = (255, 0, 0)) -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", (4, 4), color).save(buffer, format="PNG")
    return buffer.getvalue()


def _jpeg_bytes() -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", (4, 4), (0, 255, 0)).save(buffer, format="JPEG")
    return buffer.getvalue()


def test_detect_image_columns_finds_png_magic_bytes():
    df = pl.DataFrame({"thumb": [_png_bytes(), _png_bytes((0, 0, 255))]}, schema={"thumb": pl.Binary})

    assert detect_image_columns(df.lazy()) == [{"column": "thumb", "source": "bytes"}]


def test_detect_image_columns_finds_jpeg_magic_bytes():
    df = pl.DataFrame({"thumb": [_jpeg_bytes(), _jpeg_bytes()]}, schema={"thumb": pl.Binary})

    assert detect_image_columns(df.lazy()) == [{"column": "thumb", "source": "bytes"}]


def test_detect_image_columns_rejects_non_image_bytes():
    df = pl.DataFrame({"blob": [b"not an image", b"also not an image"]}, schema={"blob": pl.Binary})

    assert detect_image_columns(df.lazy()) == []


def test_detect_image_columns_tolerates_null_bytes_entries():
    df = pl.DataFrame({"thumb": [_png_bytes(), None, _png_bytes()]}, schema={"thumb": pl.Binary})

    assert detect_image_columns(df.lazy()) == [{"column": "thumb", "source": "bytes"}]


def test_detect_image_columns_finds_file_path_extensions():
    df = pl.DataFrame(
        {"path": ["images/cat.jpg", "images/dog.PNG", "images/bird.webp", "images/fox.gif"]}
    )

    assert detect_image_columns(df.lazy()) == [{"column": "path", "source": "path"}]


def test_detect_image_columns_rejects_non_image_path_strings():
    df = pl.DataFrame({"path": ["docs/report.pdf", "docs/notes.txt"]})

    assert detect_image_columns(df.lazy()) == []


def test_detect_image_columns_neither_numeric_nor_categorical_column():
    df = pl.DataFrame({"score": [1.0, 2.0, 3.0], "label": ["cat", "dog", "cat"]})

    assert detect_image_columns(df.lazy()) == []


def _is_png_data_uri(value) -> bool:
    return isinstance(value, str) and value.startswith("data:image/png;base64,")


def test_image_previews_data_builds_a_thumbnail_per_row():
    df = pl.DataFrame({"thumb": [_png_bytes(), _jpeg_bytes()]}, schema={"thumb": pl.Binary})

    result = image_previews_data(df.lazy(), "thumb", source="bytes")

    assert result["column"] == "thumb"
    assert result["total_rows"] == 2
    assert [row["row_index"] for row in result["rows"]] == [0, 1]
    assert all(_is_png_data_uri(row["thumbnail"]) for row in result["rows"])


def test_image_previews_data_null_value_has_no_thumbnail():
    df = pl.DataFrame({"thumb": [_png_bytes(), None]}, schema={"thumb": pl.Binary})

    result = image_previews_data(df.lazy(), "thumb", source="bytes")

    assert result["rows"][1]["thumbnail"] is None


def test_image_previews_data_respects_row_limit_but_reports_total_rows():
    df = pl.DataFrame({"thumb": [_png_bytes() for _ in range(5)]}, schema={"thumb": pl.Binary})

    result = image_previews_data(df.lazy(), "thumb", source="bytes", row_limit=3)

    assert len(result["rows"]) == 3
    assert result["total_rows"] == 5


def test_image_previews_data_resolves_relative_path_against_dataset_dir(tmp_path):
    (tmp_path / "images").mkdir()
    (tmp_path / "images" / "cat.png").write_bytes(_png_bytes())
    df = pl.DataFrame({"path": ["images/cat.png"]})

    result = image_previews_data(df.lazy(), "path", source="path", dataset_dir=tmp_path)

    assert _is_png_data_uri(result["rows"][0]["thumbnail"])


def test_image_previews_data_missing_file_has_no_thumbnail(tmp_path):
    df = pl.DataFrame({"path": ["images/does_not_exist.png"]})

    result = image_previews_data(df.lazy(), "path", source="path", dataset_dir=tmp_path)

    assert result["rows"][0]["thumbnail"] is None


def test_resolve_image_full_returns_data_uri_for_valid_row():
    df = pl.DataFrame({"thumb": [_png_bytes()]}, schema={"thumb": pl.Binary})

    result = resolve_image_full(df.lazy(), "thumb", source="bytes", row_index=0)

    assert _is_png_data_uri(result["image"])


def test_resolve_image_full_errors_on_out_of_range_row_index():
    df = pl.DataFrame({"thumb": [_png_bytes()]}, schema={"thumb": pl.Binary})

    result = resolve_image_full(df.lazy(), "thumb", source="bytes", row_index=5)

    assert "error" in result


def test_resolve_image_full_errors_when_value_is_null():
    df = pl.DataFrame({"thumb": [None]}, schema={"thumb": pl.Binary})

    result = resolve_image_full(df.lazy(), "thumb", source="bytes", row_index=0)

    assert "error" in result


def test_resolve_image_full_errors_when_path_file_is_missing(tmp_path):
    df = pl.DataFrame({"path": ["images/does_not_exist.png"]})

    result = resolve_image_full(df.lazy(), "path", source="path", row_index=0, dataset_dir=tmp_path)

    assert "error" in result


def test_media_grid_png_returns_none_when_no_decodable_thumbnails():
    preview = {"column": "thumb", "rows": [{"row_index": 0, "thumbnail": None}], "total_rows": 1}

    assert media_grid_png(preview) is None


def test_media_grid_png_returns_valid_png_for_decodable_thumbnails():
    df = pl.DataFrame({"thumb": [_png_bytes(), _png_bytes((0, 255, 0)), None]}, schema={"thumb": pl.Binary})
    preview = image_previews_data(df.lazy(), "thumb", source="bytes")

    result = media_grid_png(preview)

    assert result is not None
    assert result[:8] == b"\x89PNG\r\n\x1a\n"


def test_media_grid_png_caps_at_the_given_limit():
    df = pl.DataFrame({"thumb": [_png_bytes() for _ in range(20)]}, schema={"thumb": pl.Binary})
    preview = image_previews_data(df.lazy(), "thumb", source="bytes")

    grid_two = media_grid_png(preview, limit=2, columns=2)
    grid_many = media_grid_png(preview, limit=20, columns=2)

    # More items -> a taller grid image -> a larger encoded file. Not a
    # precise size check (PNG compression varies), just "more rows happened".
    assert len(grid_many) > len(grid_two)
