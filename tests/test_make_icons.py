"""Icon generation (scripts/make_icons.py) and the shipped .ico.

The size list is the whole point of the file: Windows pulls a different
frame out of the same .ico for the title bar, the taskbar, Alt-Tab and
Explorer's large tiles, so a frame going missing shows up as one blurry
surface somewhere and nowhere else. That is easy to miss by eye and cheap
to assert.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from make_icons import (  # noqa: E402
    ICO_PATH,
    ICO_SIZES,
    PNG_DIR,
    PNG_SIZES,
    SOURCE,
    load_square_rgba,
    write_ico,
    write_pngs,
)


# --- padding to square -------------------------------------------------


def test_a_square_source_is_left_alone(tmp_path):
    path = tmp_path / "square.png"
    Image.new("RGBA", (64, 64), (10, 20, 30, 255)).save(path)

    assert load_square_rgba(path).size == (64, 64)


def test_a_wide_source_is_padded_not_stretched(tmp_path):
    """Stretching would distort the artwork, and every generated size is
    derived from this one image."""
    path = tmp_path / "wide.png"
    Image.new("RGBA", (100, 40), (10, 20, 30, 255)).save(path)

    result = load_square_rgba(path)

    assert result.size == (100, 100)
    # The original pixels survive at their own aspect ratio, centred.
    assert result.getpixel((50, 50)) == (10, 20, 30, 255)


def test_a_tall_source_is_padded_to_its_longest_side(tmp_path):
    path = tmp_path / "tall.png"
    Image.new("RGBA", (30, 90), (10, 20, 30, 255)).save(path)

    assert load_square_rgba(path).size == (90, 90)


def test_padding_is_transparent_not_filled(tmp_path):
    path = tmp_path / "wide.png"
    Image.new("RGBA", (100, 40), (10, 20, 30, 255)).save(path)

    result = load_square_rgba(path)

    assert result.getpixel((50, 2))[3] == 0, "padding must be transparent"


def test_a_source_without_alpha_still_loads(tmp_path):
    """The source may be a JPEG (or a JPEG named .png) with no alpha
    channel at all — converting to RGBA must not fail."""
    path = tmp_path / "flat.jpg"
    Image.new("RGB", (50, 50), (200, 100, 50)).save(path, format="JPEG")

    result = load_square_rgba(path)

    assert result.mode == "RGBA"
    assert result.size == (50, 50)


# --- .ico contents -----------------------------------------------------


def test_write_ico_contains_every_requested_size(tmp_path):
    source = tmp_path / "src.png"
    Image.new("RGBA", (512, 512), (10, 20, 30, 255)).save(source)
    target = tmp_path / "out.ico"

    write_ico(load_square_rgba(source), target)

    assert sorted(Image.open(target).ico.sizes()) == sorted((s, s) for s in ICO_SIZES)


def test_write_pngs_writes_one_file_per_size(tmp_path):
    source = tmp_path / "src.png"
    Image.new("RGBA", (512, 512), (10, 20, 30, 255)).save(source)

    written = write_pngs(load_square_rgba(source), tmp_path / "icons")

    assert len(written) == len(PNG_SIZES)
    for path, size in zip(written, PNG_SIZES):
        assert Image.open(path).size == (size, size)


# --- the assets actually committed -------------------------------------


@pytest.mark.skipif(not SOURCE.exists(), reason="source artwork not present")
def test_the_shipped_ico_exists_and_has_every_size():
    assert ICO_PATH.exists(), "run scripts/make_icons.py"

    assert sorted(Image.open(ICO_PATH).ico.sizes()) == sorted((s, s) for s in ICO_SIZES)


@pytest.mark.skipif(not SOURCE.exists(), reason="source artwork not present")
@pytest.mark.parametrize("size", PNG_SIZES)
def test_the_shipped_png_set_is_complete(size):
    target = PNG_DIR / f"otter-{size}.png"

    assert target.exists(), "run scripts/make_icons.py"
    assert Image.open(target).size == (size, size)


@pytest.mark.skipif(not ICO_PATH.exists(), reason="icon not generated yet")
def test_main_points_at_an_ico_that_exists():
    """pywebview falls back to python.exe's own icon when the path is
    missing, silently — so a broken path would show up only as the wrong
    icon on someone's screen, never as an error."""
    from ml_data_view.main import ICON_PATH

    assert ICON_PATH.is_file()
    assert ICON_PATH.suffix == ".ico", "System.Drawing.Icon only accepts .ico"
