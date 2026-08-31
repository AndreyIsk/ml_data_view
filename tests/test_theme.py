"""Contrast gates for both theme palettes (frontend/style.css).

A hand-picked dark palette fails in one specific way: muted text that
looked fine against white disappears against a dark ground. That failure is
computable, so it is computed here rather than left to whoever next opens
the app at midnight.

The palettes are parsed straight out of style.css, so these tests break if
someone edits a color without checking it — which is the point. Nothing
here renders anything; it is arithmetic on the declared values.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

STYLE_SHEET = Path(__file__).resolve().parents[1] / "ml_data_view" / "frontend" / "style.css"

# WCAG AA for body text. Large/bold text is allowed 3:1, but nothing in this
# app's palette is large-only, so one gate covers everything.
AA_CONTRAST = 4.5
# WCAG AA for non-text UI: borders, gridlines, chart marks. A gridline is
# deliberately recessive, so it is exempt (see NON_TEXT_EXEMPT).
AA_NON_TEXT = 3.0


def _parse_block(css: str, selector: str) -> dict[str, str]:
    """Custom properties declared in one selector block."""
    match = re.search(re.escape(selector) + r"\s*\{(.*?)\}", css, re.S)
    assert match, f"no {selector} block found in style.css"
    return dict(re.findall(r"(--[\w-]+)\s*:\s*(#[0-9a-fA-F]{3,8})\s*;", match.group(1)))


def _load_palettes() -> tuple[dict[str, str], dict[str, str]]:
    css = STYLE_SHEET.read_text(encoding="utf-8")
    light = _parse_block(css, ":root")
    dark = {**light, **_parse_block(css, '[data-theme="dark"]')}
    return light, dark


LIGHT, DARK = _load_palettes()


def _channel(value: float) -> float:
    return value / 12.92 if value <= 0.03928 else ((value + 0.055) / 1.055) ** 2.4


def relative_luminance(hex_color: str) -> float:
    """WCAG relative luminance."""
    raw = hex_color.lstrip("#")
    if len(raw) == 3:
        raw = "".join(c * 2 for c in raw)
    r, g, b = (int(raw[i : i + 2], 16) / 255 for i in (0, 2, 4))
    return 0.2126 * _channel(r) + 0.7152 * _channel(g) + 0.0722 * _channel(b)


def contrast_ratio(a: str, b: str) -> float:
    """WCAG contrast ratio between two hex colors, always >= 1."""
    la, lb = relative_luminance(a), relative_luminance(b)
    lighter, darker = max(la, lb), min(la, lb)
    return (lighter + 0.05) / (darker + 0.05)


# --- the ratio helper itself, pinned against known values --------------


def test_contrast_ratio_of_black_on_white_is_maximal():
    assert contrast_ratio("#000000", "#ffffff") == pytest.approx(21.0, abs=0.01)


def test_contrast_ratio_of_a_color_with_itself_is_one():
    assert contrast_ratio("#2a78d6", "#2a78d6") == pytest.approx(1.0, abs=0.001)


def test_contrast_ratio_is_symmetric():
    assert contrast_ratio("#1a1a19", "#c3c2b7") == pytest.approx(
        contrast_ratio("#c3c2b7", "#1a1a19")
    )


def test_three_digit_hex_expands():
    assert contrast_ratio("#fff", "#000000") == pytest.approx(21.0, abs=0.01)


# --- both palettes actually exist --------------------------------------

# Every role that must be defined in BOTH palettes. Listed explicitly so a
# role added to light but forgotten in dark fails here rather than silently
# inheriting the light value.
THEMED_ROLES = [
    "--surface-1",
    "--surface-2",
    "--panel-bg",
    "--text-primary",
    "--text-secondary",
    "--text-muted",
    "--border",
    "--accent",
    "--accent-hover",
    "--accent-strong",
    "--accent-strong-hover",
    "--on-accent",
    "--error-bg",
    "--error-border",
    "--error-text",
    "--warning-bg",
    "--warning-border",
    "--warning-text",
    "--success-bg",
    "--success-border",
    "--success-text",
    "--info-bg",
    "--info-border",
    "--info-text",
    "--row-tint-strong",
    "--row-tint-soft",
    "--chart-series",
    "--chart-missing",
    "--chart-present",
    "--chart-gridline",
    "--chart-axis-text",
    "--chart-cat-1",
    "--chart-cat-2",
    "--chart-cat-3",
    "--chart-other",
    "--chart-missing-bucket",
    "--chart-marker-ring",
]


@pytest.mark.parametrize("role", THEMED_ROLES)
def test_role_is_defined_in_the_light_palette(role):
    assert role in LIGHT, f"{role} missing from :root"


@pytest.mark.parametrize("role", THEMED_ROLES)
def test_role_is_redefined_in_the_dark_palette(role):
    css = STYLE_SHEET.read_text(encoding="utf-8")
    dark_only = _parse_block(css, '[data-theme="dark"]')
    assert role in dark_only, f"{role} not overridden for dark — it would inherit the light value"


# --- text contrast, both palettes --------------------------------------

# (ink role, ground role) pairs that actually occur on screen.
TEXT_PAIRS = [
    ("--text-primary", "--surface-1"),
    ("--text-primary", "--surface-2"),
    ("--text-secondary", "--surface-1"),
    ("--text-secondary", "--surface-2"),
    ("--text-muted", "--surface-1"),
    ("--text-muted", "--surface-2"),
    ("--error-text", "--error-bg"),
    ("--warning-text", "--warning-bg"),
    ("--success-text", "--success-bg"),
    ("--info-text", "--info-bg"),
    ("--on-accent", "--accent-strong"),
    ("--chart-axis-text", "--panel-bg"),
    ("--text-primary", "--row-tint-strong"),
    ("--text-primary", "--row-tint-soft"),
]


@pytest.mark.parametrize("ink,ground", TEXT_PAIRS, ids=lambda v: v.lstrip("-"))
@pytest.mark.parametrize("theme", ["light", "dark"])
def test_text_meets_wcag_aa(theme, ink, ground):
    palette = LIGHT if theme == "light" else DARK
    ratio = contrast_ratio(palette[ink], palette[ground])
    assert ratio >= AA_CONTRAST, (
        f"{theme}: {ink} on {ground} is {ratio:.2f}:1, below AA's {AA_CONTRAST}:1 "
        f"({palette[ink]} on {palette[ground]})"
    )


# --- non-text contrast --------------------------------------------------

# Chart marks and interactive borders must clear 3:1 against the surface
# they sit on.
NON_TEXT_PAIRS = [
    ("--chart-series", "--panel-bg"),
    ("--chart-missing", "--panel-bg"),
    ("--chart-cat-1", "--panel-bg"),
    ("--chart-cat-2", "--panel-bg"),
    ("--chart-cat-3", "--panel-bg"),
    ("--chart-other", "--panel-bg"),
    ("--chart-missing-bucket", "--panel-bg"),
    ("--accent", "--surface-1"),
]

# Marks deliberately below 3:1, each with the reason it is allowed to be.
# Listed rather than dropped from NON_TEXT_PAIRS so an exemption is a
# visible decision with a justification, not a silent gap in coverage.
NON_TEXT_EXEMPT = {
    # Aqua on a light surface measures 2.82:1. This is a documented
    # property of the validated categorical palette, not a slip: the three
    # slots were chosen for colorblind separation across ALL pairs, and
    # re-stepping aqua to clear 3:1 would break that. The palette's relief
    # rule applies instead — the scatter always renders a legend (one
    # Plotly trace per label bucket), so identity is never color-alone.
    ("light", "--chart-cat-3", "--panel-bg"),
    # The "Missing" bucket is recessive by design, exactly like a gridline:
    # a neutral that met 3:1 would compete with the data it sits behind.
    # The legend labels it, so it is never identified by color alone.
    ("light", "--chart-missing-bucket", "--panel-bg"),
    ("dark", "--chart-missing-bucket", "--panel-bg"),
}


@pytest.mark.parametrize("mark,ground", NON_TEXT_PAIRS, ids=lambda v: v.lstrip("-"))
@pytest.mark.parametrize("theme", ["light", "dark"])
def test_chart_marks_meet_wcag_aa_non_text(theme, mark, ground):
    if (theme, mark, ground) in NON_TEXT_EXEMPT:
        pytest.skip("documented exemption — see NON_TEXT_EXEMPT")
    palette = LIGHT if theme == "light" else DARK
    ratio = contrast_ratio(palette[mark], palette[ground])
    assert ratio >= AA_NON_TEXT, (
        f"{theme}: {mark} on {ground} is {ratio:.2f}:1, below {AA_NON_TEXT}:1 "
        f"({palette[mark]} on {palette[ground]})"
    )


# --- the scatter's three categorical slots --------------------------------

# The validated categorical slots, per mode. Hue separation is NOT a
# luminance property — blue and orange can share a luminance and still be
# trivially distinguishable — so it cannot be checked with contrast_ratio
# above. It is measured as CVD Delta E in OKLab by the dataviz skill's
# validate_palette.js, which was run against both rows below:
#
#   light (surface #ffffff, --pairs all): worst CVD dE 9.2, normal 24.0
#   dark  (surface #1a1a19, --pairs all): worst CVD dE 9.4, normal 20.9
#
# Both clear the >=8 CVD target and >=15 normal-vision floor. Pinning the
# values here means a hue cannot be swapped without this test failing and
# sending whoever did it back to the validator.
VALIDATED_SCATTER_SLOTS = {
    "light": ["#2a78d6", "#eb6834", "#1baf7a"],
    "dark": ["#3987e5", "#d95926", "#199e70"],
}


@pytest.mark.parametrize("theme", ["light", "dark"])
def test_scatter_slots_match_the_validated_palette(theme):
    """The embedding scatter is an all-pairs form — any two categories can
    land adjacent anywhere in the cloud, which is why the backend caps
    color-by at three (embedding_data.EMBEDDING_COLOR_TOP_N)."""
    palette = LIGHT if theme == "light" else DARK
    slots = [palette[f"--chart-cat-{i}"].lower() for i in (1, 2, 3)]

    assert slots == VALIDATED_SCATTER_SLOTS[theme], (
        f"{theme} scatter slots changed — re-run the dataviz validator "
        f"(--pairs all, surface {palette['--panel-bg']}) before updating this list"
    )


@pytest.mark.parametrize("theme", ["light", "dark"])
def test_other_and_missing_buckets_are_distinct_neutrals(theme):
    """"Other" and "Missing" are different statements and must not read as
    the same bucket."""
    palette = LIGHT if theme == "light" else DARK

    assert palette["--chart-other"] != palette["--chart-missing-bucket"]


@pytest.mark.parametrize("theme", ["light", "dark"])
def test_missing_bucket_recedes_relative_to_other(theme):
    """On light the recessive direction is lighter; on dark it is darker.
    The relationship must survive the flip rather than inverting, or
    "Missing" becomes the loudest thing in the plot."""
    palette = LIGHT if theme == "light" else DARK
    ground = relative_luminance(palette["--panel-bg"])
    other = abs(relative_luminance(palette["--chart-other"]) - ground)
    missing = abs(relative_luminance(palette["--chart-missing-bucket"]) - ground)

    assert missing < other, (
        f"{theme}: the Missing bucket stands out more from the surface than Other does"
    )
