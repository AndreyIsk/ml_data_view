"""Headless image-column detection and preview generation for Phase 3.

Pure functions over a polars LazyFrame (plus, for preview/full-image
resolution, the dataset file's own directory for relative-path lookups) —
no pywebview or Plotly imports here, mirroring embedding_data.py's split
between data computation and rendering (frontend/app.js). See
data_loader.py's module docstring for the lazy/streaming architecture note.
"""
from __future__ import annotations

import base64
import io
import math
from pathlib import Path
from typing import Any

import polars as pl
from PIL import Image

from ml_data_view.data_loader import row_count as _dataset_row_count

IMAGE_PATH_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".gif"}
IMAGE_PREVIEW_ROW_LIMIT = 50
THUMBNAIL_SIZE = (100, 100)
FULL_IMAGE_MAX_DIMENSION = 800

# Stage 3 (Reports/PDF export): a simple, non-interactive thumbnail grid of
# an image column's first few items, for embedding in the PDF — same source
# thumbnails as the live preview grid, just flattened into one static image.
MEDIA_GRID_ITEM_LIMIT = 12
MEDIA_GRID_COLUMNS = 4
MEDIA_GRID_PADDING = 6
# How many non-null sample values to sniff when deciding whether a column
# looks like images — checking every row would be wasteful (and, for bytes
# columns, means decoding every image just to classify the column).
_DETECTION_SAMPLE_SIZE = 20

_BINARY_DTYPE_NAME = "Binary"
_STRING_DTYPE_NAMES = {"String", "Utf8"}


def _looks_like_image_bytes(value: bytes) -> bool:
    """True if PIL can identify `value` as an image via its magic bytes."""
    try:
        with Image.open(io.BytesIO(value)) as img:
            img.verify()
        return True
    except Exception:
        return False


def _looks_like_image_path(value: str) -> bool:
    return Path(value).suffix.lower() in IMAGE_PATH_EXTENSIONS


def _sample_non_null(
    lf: pl.LazyFrame, column: str, files: list[Path] | None = None
) -> list[Any]:
    """A bounded (`_DETECTION_SAMPLE_SIZE`-row) collect — never the whole
    column, even for a bytes column, which would mean pulling every image
    in the dataset through this classification step.

    The head is taken at FRAME level, not inside the select expression.
    That difference is not cosmetic: `select(col.drop_nulls().head(n))`
    makes polars evaluate drop_nulls() over the entire frame before
    truncating, so on a multi-file concat it reads every row of the
    column. Measured over 26 string columns, that grew from 975MB at 2
    files to 2872MB at 50 — the single largest remaining source of
    file-count-dependent memory in chart building. A frame-level head
    pushes the limit down into the scan.

    `files` narrows the sample to the FIRST file. Detection is a heuristic
    over a handful of values; reading one file answers it just as well as
    reading 250.
    """
    source = pl.scan_parquet(files[0]) if files else lf
    sample = source.select(column).drop_nulls().head(_DETECTION_SAMPLE_SIZE).collect()
    return sample[column].to_list()


def detect_image_columns(
    lf: pl.LazyFrame, files: list[Path] | None = None
) -> list[dict[str, str]]:
    """Find columns that look like images: either binary values starting
    with a known image format's magic bytes, or strings that look like
    file paths ending in a known image extension.

    Returns a list of {"column": name, "source": "bytes" | "path"} dicts,
    in column order.
    """
    candidates = []
    for column, dtype in lf.collect_schema().items():
        dtype_name = str(dtype)
        if dtype_name == _BINARY_DTYPE_NAME:
            sample = _sample_non_null(lf, column, files=files)
            if sample and all(_looks_like_image_bytes(value) for value in sample):
                candidates.append({"column": column, "source": "bytes"})
        elif dtype_name in _STRING_DTYPE_NAMES:
            sample = _sample_non_null(lf, column, files=files)
            if sample and all(_looks_like_image_path(value) for value in sample):
                candidates.append({"column": column, "source": "path"})

    return candidates


def _resolve_path_bytes(value: str, dataset_dir: Path | None) -> bytes | None:
    """Read the bytes at `value`, resolved against dataset_dir if relative.

    Returns None if the path can't be resolved (no dataset_dir for a
    relative path) or the file can't be read.
    """
    path = Path(value)
    if not path.is_absolute():
        if dataset_dir is None:
            return None
        path = dataset_dir / path
    try:
        return path.read_bytes()
    except OSError:
        return None


def _encode_thumbnail(image_bytes: bytes, max_dimension: tuple[int, int] | int) -> str | None:
    """Downsize `image_bytes` and return it as a base64 PNG data URI, or
    None if it can't be decoded."""
    size = max_dimension if isinstance(max_dimension, tuple) else (max_dimension, max_dimension)
    try:
        with Image.open(io.BytesIO(image_bytes)) as img:
            img = img.convert("RGBA")
            img.thumbnail(size)
            buffer = io.BytesIO()
            img.save(buffer, format="PNG")
    except Exception:
        return None
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def _image_bytes_for_row(value: Any, source: str, dataset_dir: Path | None) -> bytes | None:
    if value is None:
        return None
    if source == "bytes":
        return value
    return _resolve_path_bytes(value, dataset_dir)


def image_previews_data(
    lf: pl.LazyFrame,
    column: str,
    source: str,
    dataset_dir: Path | None = None,
    row_limit: int = IMAGE_PREVIEW_ROW_LIMIT,
    total_rows: int | None = None,
) -> dict[str, Any]:
    """Build thumbnail data for the first `row_limit` rows of an image column.

    Each row gets a {"row_index", "thumbnail"} entry; `thumbnail` is None
    when the value is null, its file is missing, or it fails to decode —
    the frontend shows a broken-image placeholder for those. `.head()` is a
    bounded slice — at most `row_limit` (50) rows are ever collected here,
    regardless of the source file's size. `total_rows`, if the caller
    already has it, is reused instead of triggering another count.
    """
    values = lf.select(pl.col(column).head(row_limit)).collect()[column].to_list()
    rows = []
    for row_index, value in enumerate(values):
        image_bytes = _image_bytes_for_row(value, source, dataset_dir)
        thumbnail = _encode_thumbnail(image_bytes, THUMBNAIL_SIZE) if image_bytes is not None else None
        rows.append({"row_index": row_index, "thumbnail": thumbnail})

    return {
        "column": column,
        "source": source,
        "rows": rows,
        "total_rows": total_rows if total_rows is not None else _dataset_row_count(lf),
    }


def media_grid_png(
    preview: dict[str, Any],
    limit: int = MEDIA_GRID_ITEM_LIMIT,
    columns: int = MEDIA_GRID_COLUMNS,
) -> bytes | None:
    """Build a static, non-interactive thumbnail grid PNG for one image
    column's `preview` (an image_previews_data()-shaped dict), using its
    first `limit` rows that have a decodable thumbnail. Returns None if
    there are none — e.g. every row was null or failed to decode.
    """
    thumbnails: list[Image.Image] = []
    for row in preview.get("rows", []):
        if len(thumbnails) >= limit:
            break
        data_uri = row.get("thumbnail")
        if not data_uri:
            continue
        try:
            _, encoded = data_uri.split(",", 1)
            thumbnails.append(Image.open(io.BytesIO(base64.b64decode(encoded))).convert("RGBA"))
        except Exception:
            continue

    if not thumbnails:
        return None

    cell_w, cell_h = THUMBNAIL_SIZE
    row_count = math.ceil(len(thumbnails) / columns)
    grid = Image.new(
        "RGBA",
        (
            columns * cell_w + (columns + 1) * MEDIA_GRID_PADDING,
            row_count * cell_h + (row_count + 1) * MEDIA_GRID_PADDING,
        ),
        (255, 255, 255, 255),
    )
    for index, thumbnail in enumerate(thumbnails):
        col, row = index % columns, index // columns
        x = MEDIA_GRID_PADDING + col * (cell_w + MEDIA_GRID_PADDING)
        y = MEDIA_GRID_PADDING + row * (cell_h + MEDIA_GRID_PADDING)
        # Thumbnails may be smaller than THUMBNAIL_SIZE (PIL's thumbnail()
        # preserves aspect ratio) — center each one in its cell.
        x_offset = max((cell_w - thumbnail.width) // 2, 0)
        y_offset = max((cell_h - thumbnail.height) // 2, 0)
        grid.paste(thumbnail, (x + x_offset, y + y_offset), thumbnail)

    buffer = io.BytesIO()
    grid.convert("RGB").save(buffer, format="PNG")
    return buffer.getvalue()


def resolve_image_full(
    lf: pl.LazyFrame,
    column: str,
    source: str,
    row_index: int,
    dataset_dir: Path | None = None,
) -> dict[str, Any]:
    """Resolve a single row's image at a larger (but still capped) size,
    for the preview grid's "view larger" click. Computed on demand rather
    than eagerly for all rows — some source images could be large.

    A single-row `.slice(row_index, 1)` collect — never needs the total row
    count or any other row: an out-of-range offset just comes back as an
    empty (0-row) result, which reads the same as "out of range" here.
    """
    if row_index < 0:
        return {"error": "Row index out of range."}

    result = lf.select(pl.col(column)).slice(row_index, 1).collect()
    if result.height == 0:
        return {"error": "Row index out of range."}

    value = result[column][0]
    image_bytes = _image_bytes_for_row(value, source, dataset_dir)
    if image_bytes is None:
        return {"error": "Image not available for this row."}

    image = _encode_thumbnail(image_bytes, FULL_IMAGE_MAX_DIMENSION)
    if image is None:
        return {"error": "Could not decode image."}
    return {"image": image}
