"""Generate the app's Windows .ico and PNG icon set from one source image.

A script rather than a one-off, because the icon will be regenerated every
time the source art changes, and because the size list is a decision worth
keeping written down: Windows picks a different frame from the same .ico
for the title bar (16), taskbar (32), Alt-Tab (48), and large Explorer
views (256), so a single-size .ico looks resampled and muddy in most of
those places.

Run:  python scripts/make_icons.py
"""
from __future__ import annotations

import sys
from pathlib import Path

from PIL import Image

ASSETS_DIR = Path(__file__).resolve().parents[1] / "assets"
SOURCE = ASSETS_DIR / "otter.png"
ICO_PATH = ASSETS_DIR / "otter.ico"
PNG_DIR = ASSETS_DIR / "icons"

# The sizes Windows actually asks for. 16 title bar / 32 taskbar / 48
# Alt-Tab and medium Explorer icons / 64 / 128 / 256 large Explorer tiles
# and the installer.
ICO_SIZES = (16, 32, 48, 64, 128, 256)
# Standalone PNGs for everything that is not a Windows shell surface —
# docs, a web page, or a future non-Windows packaging target.
PNG_SIZES = (16, 32, 48, 128, 256, 512)


def load_square_rgba(path: Path) -> Image.Image:
    """Open `path` as RGBA, padded to a square on a transparent canvas.

    Padded rather than stretched: a non-square source scaled to a square
    frame distorts the artwork, and every size below is derived from this
    one image. A source that is already square passes through untouched.

    Note the source may not carry alpha at all (a JPEG, say) — converting
    to RGBA still gives the padding a transparent margin to sit in, and
    leaves the artwork itself unchanged.
    """
    image = Image.open(path).convert("RGBA")
    width, height = image.size
    if width == height:
        return image

    side = max(width, height)
    canvas = Image.new("RGBA", (side, side), (0, 0, 0, 0))
    canvas.paste(image, ((side - width) // 2, (side - height) // 2))
    return canvas


def write_ico(image: Image.Image, path: Path, sizes: tuple[int, ...] = ICO_SIZES) -> None:
    """Write one .ico containing a frame per size in `sizes`.

    Each frame is resampled from the full-resolution source rather than
    from the previous (smaller) step, so a 16px frame is a clean reduction
    of the original instead of a reduction of a reduction.
    """
    frames = [image.resize((size, size), Image.LANCZOS) for size in sizes]
    largest = frames[-1]
    largest.save(path, format="ICO", sizes=[(size, size) for size in sizes], append_images=frames)


def write_pngs(image: Image.Image, directory: Path, sizes: tuple[int, ...] = PNG_SIZES) -> list[Path]:
    directory.mkdir(parents=True, exist_ok=True)
    written = []
    for size in sizes:
        target = directory / f"otter-{size}.png"
        image.resize((size, size), Image.LANCZOS).save(target, format="PNG")
        written.append(target)
    return written


def main() -> int:
    if not SOURCE.exists():
        print(f"Source image not found: {SOURCE}", file=sys.stderr)
        return 1

    source = Image.open(SOURCE)
    print(f"source: {SOURCE.name}  {source.size[0]}x{source.size[1]}  "
          f"format={source.format}  mode={source.mode}")

    image = load_square_rgba(SOURCE)
    write_ico(image, ICO_PATH)
    print(f"wrote {ICO_PATH.relative_to(ASSETS_DIR.parent)} "
          f"({', '.join(f'{s}px' for s in ICO_SIZES)}) — {ICO_PATH.stat().st_size:,} bytes")

    for target in write_pngs(image, PNG_DIR):
        print(f"wrote {target.relative_to(ASSETS_DIR.parent)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
