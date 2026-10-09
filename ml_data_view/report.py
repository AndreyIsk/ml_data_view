"""PDF report generation for Stage 3's Reports section.

Assembles the same computations the live dashboard uses — chart_data.py's
per-column charts and null heatmap, embedding_data.py's PCA projections,
image_data.py's decoded thumbnails — into a PDF via reportlab. Chart images
are produced by report_charts.py (Plotly figure + kaleido rasterization);
no chart logic is reimplemented here.
"""
from __future__ import annotations

import io
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from reportlab.lib import colors
from reportlab.lib.pagesizes import LETTER
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import (
    Image as RLImage,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)
from PIL import Image as PILImage

from ml_data_view.image_data import media_grid_png
from ml_data_view.report_charts import (
    categorical_bar_figure,
    embedding_scatter_figure,
    figure_png_bytes,
    null_heatmap_figure,
    numeric_histogram_figure,
    text_length_histogram_figure,
)

PAGE_MARGIN = 0.75 * inch
BANNER_HEIGHT = 0.32 * inch
CHART_EMBED_WIDTH = 6.5 * inch
MEDIA_GRID_EMBED_WIDTH = 4 * inch

# Per-column charts are laid out in a grid rather than one-per-row: at full
# width only 1-2 fitted per page, which made the report absurdly long for a
# wide dataset — the exact case this tool exists for. Two columns of ~3.1in
# charts fit 8 per page instead.
CHART_GRID_COLUMNS = 2
CHART_GRID_EMBED_WIDTH = 3.1 * inch
CHART_GRID_GUTTER = 0.15 * inch
# Rendered pixel size for grid charts. Smaller than the full-width figures
# while the font stays at PLOT_FONT's 11px, so text stays legible once the
# PNG is scaled down into a grid cell — rendering at full size and shrinking
# would make every label unreadable.
GRID_CHART_PX_WIDTH = 420
GRID_CHART_PX_HEIGHT = 260


def available_sections(chart_data: dict[str, Any]) -> dict[str, bool]:
    """Which Reports checkboxes should be offered for the currently loaded
    dataset. Mirrors the frontend's own "is there anything to show" checks
    (renderColumnCharts / renderEmbeddingProjections / renderImagePreviews
    in app.js) so a section only appears when it would actually render
    something live, too."""
    has_distributions = bool(
        chart_data.get("numeric_histograms")
        or chart_data.get("categorical_bars")
        or chart_data.get("text_length_histograms")
        or (chart_data.get("null_heatmap") or {}).get("z")
    )
    has_embeddings = any(
        projection.get("x") for projection in chart_data.get("embedding_projections") or []
    )
    has_media = bool(chart_data.get("image_previews"))
    # Health is offered on exactly the condition its checks declare: a
    # timestamp column was detected and the analysis ran. No time axis, no
    # section — rather than a section that says nothing.
    has_health = bool((chart_data.get("timeseries") or {}).get("units"))
    # The catalogue is offered above UNIT scope only. With one unit loaded
    # there is one row, and the Recording Health section beside it already
    # says everything that row would.
    has_catalog = len(((chart_data.get("catalog") or {}).get("rows")) or []) > 1
    return {
        "overview": True,
        "distributions": has_distributions,
        "embeddings": has_embeddings,
        "media": has_media,
        "health": has_health,
        "catalog": has_catalog,
    }


def _contrasting_text_color(hex_color: str) -> colors.Color:
    """Same relative-luminance heuristic as the frontend's
    contrastingTextColor() (app.js) — dark text on a light banner color,
    white text on a dark one."""
    hex_color = hex_color.lstrip("#")
    r, g, b = (int(hex_color[i : i + 2], 16) / 255 for i in (0, 2, 4))
    luminance = 0.299 * r + 0.587 * g + 0.114 * b
    return colors.HexColor("#1a1a1a") if luminance > 0.6 else colors.HexColor("#ffffff")


def _draw_header_footer(
    canvas, doc, *, filename: str, generated_at: str, banner: dict[str, Any] | None
) -> None:
    """Drawn on every page (SimpleDocTemplate's onFirstPage/onLaterPages):
    filename + generation timestamp top and bottom, plus a reproduced
    classification banner strip top and bottom when Settings has one
    enabled."""
    canvas.saveState()
    width, height = doc.pagesize

    canvas.setFont("Helvetica", 8)
    canvas.setFillColor(colors.HexColor("#52514e"))
    for y in (height - 0.45 * inch, 0.35 * inch):
        canvas.drawString(PAGE_MARGIN, y, filename)
        canvas.drawRightString(width - PAGE_MARGIN, y, f"Generated {generated_at}")

    if banner and banner.get("enabled") and banner.get("label"):
        background = colors.HexColor(banner["color"])
        foreground = _contrasting_text_color(banner["color"])
        for y in (height - BANNER_HEIGHT, 0):
            canvas.setFillColor(background)
            canvas.rect(0, y, width, BANNER_HEIGHT, fill=1, stroke=0)
            canvas.setFillColor(foreground)
            canvas.setFont("Helvetica-Bold", 10)
            canvas.drawCentredString(width / 2, y + BANNER_HEIGHT / 2 - 3, banner["label"])

    canvas.restoreState()


def _schema_table(dataset_info: dict[str, Any], body_style) -> Table:
    header = ["Column", "Type", "Null count"]
    rows = [
        [Paragraph(column["name"], body_style), column["dtype"], str(column["null_count"])]
        for column in dataset_info["columns"]
    ]
    table = Table([header, *rows], colWidths=[3.5 * inch, 2 * inch, 1.5 * inch], hAlign="LEFT", repeatRows=1)
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#faf9f8")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.HexColor("#52514e")),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, -1), 8),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#e2e1dc")),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ]
        )
    )
    return table


def _chart_image_flowables(fig) -> list:
    png_bytes = figure_png_bytes(fig)
    width = CHART_EMBED_WIDTH
    height = width * fig.layout.height / fig.layout.width
    return [RLImage(io.BytesIO(png_bytes), width=width, height=height), Spacer(1, 12)]


def _chart_grid_flowable(figures: list) -> list:
    """Lay figures out in a CHART_GRID_COLUMNS-wide grid.

    One Table for all of them rather than one per page: reportlab splits a
    table's rows across page boundaries by itself, so pages fill densely
    without this code having to know how much vertical space is left.
    """
    if not figures:
        return []
    cells = []
    for fig in figures:
        png_bytes = figure_png_bytes(fig)
        width = CHART_GRID_EMBED_WIDTH
        height = width * fig.layout.height / fig.layout.width
        cells.append(RLImage(io.BytesIO(png_bytes), width=width, height=height))

    rows = [cells[i : i + CHART_GRID_COLUMNS] for i in range(0, len(cells), CHART_GRID_COLUMNS)]
    # Pad the final row so every row has the same cell count — reportlab
    # requires a rectangular table.
    if len(rows[-1]) < CHART_GRID_COLUMNS:
        rows[-1] = rows[-1] + [""] * (CHART_GRID_COLUMNS - len(rows[-1]))

    table = Table(
        rows,
        colWidths=[CHART_GRID_EMBED_WIDTH + CHART_GRID_GUTTER] * CHART_GRID_COLUMNS,
        hAlign="LEFT",
    )
    table.setStyle(
        TableStyle(
            [
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("LEFTPADDING", (0, 0), (-1, -1), 0),
                ("RIGHTPADDING", (0, 0), (-1, -1), 0),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
                ("TOPPADDING", (0, 0), (-1, -1), 0),
            ]
        )
    )
    return [table, Spacer(1, 8)]


def _correlated_missingness_flowables(groups: list[dict[str, Any]], heading_style, body_style) -> list:
    """Columns whose nulls land on the same rows (null_patterns.py) — a
    shared upstream cause is worth stating outright rather than leaving a
    reader to notice matching null counts in the schema table."""
    if not groups:
        return []
    flowables = [
        Paragraph("Correlated missingness", heading_style),
        Paragraph(
            "These column groups are missing on the same rows, which usually means a shared "
            "upstream cause rather than independent gaps.",
            body_style,
        ),
    ]
    header = ["Columns", "Rows missing", "Overlap"]
    rows = [
        [
            Paragraph(", ".join(group["columns"]), body_style),
            f"{group['null_count']:,}",
            f"{group['overlap'] * 100:.1f}%",
        ]
        for group in groups
    ]
    table = Table([header, *rows], colWidths=[4 * inch, 1.5 * inch, 1 * inch], hAlign="LEFT", repeatRows=1)
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#faf9f8")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.HexColor("#52514e")),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, -1), 8),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#e2e1dc")),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ]
        )
    )
    return flowables + [table, Spacer(1, 12)]


def _media_grid_flowables(preview: dict[str, Any], heading_style, body_style) -> list:
    grid_png = media_grid_png(preview)
    if grid_png is None:
        return [Paragraph(preview["column"], heading_style), Paragraph("No previewable images.", body_style)]
    with PILImage.open(io.BytesIO(grid_png)) as pil_image:
        aspect = pil_image.height / pil_image.width
    width = MEDIA_GRID_EMBED_WIDTH
    return [
        Paragraph(preview["column"], heading_style),
        RLImage(io.BytesIO(grid_png), width=width, height=width * aspect),
        Spacer(1, 12),
    ]


def _seconds(value: Any) -> str:
    """A duration as a reader says it. Shared shape with ts_checks._seconds;
    duplicated rather than imported because the report deliberately depends
    on nothing in the findings layer."""
    if value is None:
        return "—"
    if value < 1:
        return f"{value * 1000:.0f}ms"
    if value < 90:
        return f"{value:.2f}s"
    minutes, seconds = divmod(value, 60)
    return f"{int(minutes)}m {seconds:04.1f}s"


def _health_flowables(
    timeseries: dict[str, Any], heading_style, sub_heading_style, body_style
) -> list:
    """Recording Health: a rollup, then one row per unit.

    ROLLUP FIRST, because at anything above recording scope that is the
    answer — "three of nineteen recordings are missing a chunk" is what a
    reader wants before the table, and at recording scope the rollup is a
    single row that says the same thing as the table beneath it.

    Every figure here was computed during the chart build and cached per
    chunk; nothing is read to produce this section.
    """
    units = timeseries.get("units") or {}
    if not units:
        return []

    story = [Paragraph("Recording Health", heading_style)]
    column = timeseries.get("timestamp_column")
    story.append(
        Paragraph(
            f"Ordered by <b>{column}</b>. "
            f"{len(units)} recording(s); "
            f"{len(timeseries.get('channels_analysed') or [])} channel(s) analysed"
            + (
                f", {len(timeseries.get('channels_skipped') or [])} beyond the configured limit"
                if timeseries.get("channels_skipped")
                else ""
            )
            + ".",
            body_style,
        )
    )

    totals = {
        "missing chunks": 0,
        "overlaps": 0,
        "backwards steps": 0,
        "gaps": 0,
        "duplicate stamps": 0,
    }
    affected = {key: 0 for key in totals}
    rows = [["Recording", "Rows", "Duration", "Rate", "Gaps", "Missing", "Overlap", "Back"]]
    for unit_id, unit in sorted(units.items()):
        gaps = unit.get("gaps") or []
        missing = sum(1 for gap in gaps if gap.get("at_boundary"))
        ordinary = len(gaps) - missing
        counts = {
            "missing chunks": missing,
            "overlaps": len(unit.get("overlaps") or []),
            "backwards steps": unit.get("backwards") or 0,
            "gaps": ordinary,
            "duplicate stamps": unit.get("duplicates") or 0,
        }
        for key, value in counts.items():
            totals[key] += value
            affected[key] += 1 if value else 0
        rows.append(
            [
                unit_id,
                f"{unit.get('rows', 0):,}",
                _seconds(unit.get("duration")),
                f"{unit['hz']:.1f}Hz" if unit.get("hz") else "—",
                str(ordinary),
                str(missing),
                str(counts["overlaps"]),
                str(counts["backwards steps"]),
            ]
        )

    summary = ", ".join(
        f"{affected[key]} of {len(units)} with {key}" for key in totals if totals[key]
    )
    story.append(
        Paragraph(
            f"<b>Rollup:</b> {summary}." if summary else "<b>Rollup:</b> nothing to report.",
            body_style,
        )
    )
    story.append(Spacer(1, 8))

    table = Table(rows, hAlign="LEFT")
    table.setStyle(
        TableStyle(
            [
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, -1), 8),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
                ("TOPPADDING", (0, 0), (-1, -1), 3),
                ("GRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#d0d0d0")),
                ("ALIGN", (1, 1), (-1, -1), "RIGHT"),
            ]
        )
    )
    story.append(table)
    return story


def _bytes(value: Any) -> str:
    if value is None:
        return "—"
    size = float(value)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if size < 1024 or unit == "TB":
            return f"{size:,.0f} {unit}" if unit == "B" else f"{size:,.1f} {unit}"
        size /= 1024
    return f"{size:,.1f} TB"


def _catalog_flowables(
    chart_data: dict[str, Any], heading_style, sub_heading_style, body_style
) -> list:
    """The catalogue as a SUMMARY, not as a table of every row.

    A fleet of nineteen systems and two hundred recordings is a report
    nobody prints. What a reader wants above unit scope is the top-level
    numbers, a line per group, and then the recordings that are actually
    unhealthy — so that is what this is: totals, groups, and the unhealthy
    rows named. The full table is one CSV export away, which is the right
    place for two hundred rows.

    Every figure was computed during the chart build from cached per-file
    statistics; nothing is read to produce this section.
    """
    catalog = chart_data.get("catalog") or {}
    rows = catalog.get("rows") or []
    if not rows:
        return []

    unit_label = catalog.get("unit_label") or "Unit"
    totals = catalog.get("totals") or {}
    story = [Paragraph(f"{_plural(unit_label)} catalogue", heading_style)]
    story.append(
        Paragraph(
            f"{totals.get('unit_count', len(rows)):,} {_plural(unit_label).lower()}, "
            f"{totals.get('chunk_count', 0):,} chunk(s), "
            f"{_count(totals.get('rows'))} row(s), "
            f"{_seconds(totals.get('duration'))} of data, "
            f"{_bytes(totals.get('bytes'))} on disk.",
            body_style,
        )
    )

    breakdown = catalog.get("health") or {}
    named = ", ".join(
        f"{breakdown[key]} {key}" for key in ("critical", "warning", "info", "ok")
        if breakdown.get(key)
    )
    if named:
        story.append(Paragraph(f"<b>Health:</b> {named}.", body_style))
    story.append(Spacer(1, 8))

    groups = catalog.get("groups") or []
    if groups:
        group_label = (catalog.get("group_labels") or ["Group"])[0]
        story.append(Paragraph(f"By {group_label.lower()}", sub_heading_style))
        table = [[group_label, _plural(unit_label), "Rows", "Duration", "Health"]]
        for group in groups:
            health = group.get("health") or {}
            worst = next(
                (key for key in ("critical", "warning", "info") if health.get(key)), "ok"
            )
            table.append(
                [
                    str(group.get("value")),
                    f"{group.get('unit_count', 0):,}",
                    _count(group.get("rows")),
                    _seconds(group.get("duration")),
                    f"{worst} ({health.get(worst, 0)})" if worst != "ok" else "ok",
                ]
            )
        story.append(_grid(table))
        story.append(Spacer(1, 8))

    unhealthy = [
        row for row in rows if (row.get("health") or {}).get("status") not in (None, "ok")
    ]
    story.append(
        Paragraph(
            f"Unhealthy {_plural(unit_label).lower()}"
            + (f" — {len(unhealthy)} of {len(rows)}" if unhealthy else ""),
            sub_heading_style,
        )
    )
    if not unhealthy:
        story.append(
            Paragraph(
                f"None. Every {unit_label.lower()} passed every check that applied to it.",
                body_style,
            )
        )
        return story

    table = [[unit_label, "Chunks", "Rows", "Duration", "Gaps", "Health"]]
    for row in unhealthy:
        health = row.get("health") or {}
        table.append(
            [
                row.get("label", ""),
                f"{row.get('chunk_count', 0):,}",
                _count(row.get("rows")),
                _seconds(row.get("duration")),
                "—" if row.get("gap_count") is None else f"{row['gap_count']:,}",
                f"{health.get('status')} ({health.get('count', 0)})",
            ]
        )
    story.append(_grid(table))
    return story


def _plural(word: str) -> str:
    """A label's plural, for a heading. Naive on purpose — the labels are
    the user's own words and a real pluraliser would still get some of
    them wrong; this at least never turns "Series" into "Seriess"."""
    lowered = word.lower()
    if lowered.endswith(("s", "x", "z", "ch", "sh")):
        return word
    if lowered.endswith("y") and not lowered.endswith(("ay", "ey", "iy", "oy", "uy")):
        return f"{word[:-1]}ies"
    return f"{word}s"


def _count(value: Any) -> str:
    return "unknown" if value is None else f"{value:,}"


def _grid(rows: list[list[str]]) -> Table:
    table = Table(rows, hAlign="LEFT")
    table.setStyle(
        TableStyle(
            [
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, -1), 8),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
                ("TOPPADDING", (0, 0), (-1, -1), 3),
                ("GRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#d0d0d0")),
                ("ALIGN", (1, 1), (-1, -1), "RIGHT"),
            ]
        )
    )
    return table


def generate_report(
    output_path: str | Path,
    *,
    dataset_info: dict[str, Any],
    chart_data: dict[str, Any],
    sections: Iterable[str],
    banner: dict[str, Any] | None = None,
    generated_at: str | None = None,
) -> Path:
    """Write a PDF report to `output_path`.

    `sections` is any subset of ("overview", "distributions", "embeddings",
    "media", "health", "catalog"); a section is only actually included if
    available_sections() also finds content for it — requesting an
    unavailable section is a silent no-op for that section, not an error.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    filename = Path(dataset_info["file_path"]).name
    generated_at = generated_at or datetime.now(timezone.utc).isoformat()
    include = {section for section in sections if available_sections(chart_data).get(section)}

    styles = getSampleStyleSheet()
    title_style = ParagraphStyle("ReportTitle", parent=styles["Heading1"], spaceAfter=4)
    heading_style = ParagraphStyle("SectionHeading", parent=styles["Heading2"], spaceBefore=18, spaceAfter=8)
    sub_heading_style = ParagraphStyle("ChartHeading", parent=styles["Heading3"], spaceBefore=10, spaceAfter=4)
    body_style = styles["BodyText"]

    banner_enabled = bool(banner and banner.get("enabled") and banner.get("label"))
    top_margin = PAGE_MARGIN + (BANNER_HEIGHT if banner_enabled else 0)
    bottom_margin = PAGE_MARGIN + (BANNER_HEIGHT if banner_enabled else 0)

    doc = SimpleDocTemplate(
        str(output_path),
        pagesize=LETTER,
        leftMargin=PAGE_MARGIN,
        rightMargin=PAGE_MARGIN,
        topMargin=top_margin,
        bottomMargin=bottom_margin,
        title=f"{filename} — ML Dataset Profiler Report",
    )

    story: list = [
        Paragraph("ML Dataset Profiler Report", title_style),
        Paragraph(f"{filename} &mdash; {dataset_info['row_count']:,} rows", body_style),
        Spacer(1, 16),
    ]

    if "overview" in include:
        story.append(Paragraph("Overview", heading_style))
        story.append(_schema_table(dataset_info, body_style))

    if "distributions" in include:
        story.append(Paragraph("Distributions", heading_style))
        story += _correlated_missingness_flowables(
            chart_data.get("correlated_missingness") or [], sub_heading_style, body_style
        )
        # The heatmap keeps full width — it is one chart whose whole point
        # is comparing many columns side by side, so shrinking it into a
        # grid cell would defeat it. Per-column charts go in the grid.
        heatmap_fig = null_heatmap_figure(chart_data.get("null_heatmap") or {})
        if heatmap_fig is not None:
            story += _chart_image_flowables(heatmap_fig)

        grid_figures = []
        for chart in chart_data.get("numeric_histograms") or []:
            grid_figures.append(
                numeric_histogram_figure(chart, GRID_CHART_PX_WIDTH, GRID_CHART_PX_HEIGHT)
            )
        for chart in chart_data.get("categorical_bars") or []:
            grid_figures.append(
                categorical_bar_figure(chart, GRID_CHART_PX_WIDTH, GRID_CHART_PX_HEIGHT)
            )
        for chart in chart_data.get("text_length_histograms") or []:
            grid_figures.append(
                text_length_histogram_figure(chart, GRID_CHART_PX_WIDTH, GRID_CHART_PX_HEIGHT)
            )
        story += _chart_grid_flowable([fig for fig in grid_figures if fig is not None])

    # The catalogue before the per-recording health section: it is the
    # wider view, and a reader coming to a fleet report wants "how many of
    # these are wrong" before "here is every one of them".
    if "catalog" in include:
        story += _catalog_flowables(chart_data, heading_style, sub_heading_style, body_style)

    if "health" in include:
        story += _health_flowables(
            chart_data.get("timeseries") or {}, heading_style, sub_heading_style, body_style
        )

    if "embeddings" in include:
        story.append(Paragraph("Embeddings", heading_style))
        for projection in chart_data.get("embedding_projections") or []:
            fig = embedding_scatter_figure(projection)
            if fig is not None:
                story += _chart_image_flowables(fig)

    if "media" in include:
        story.append(Paragraph("Media", heading_style))
        for preview in chart_data.get("image_previews") or []:
            story += _media_grid_flowables(preview, sub_heading_style, body_style)

    def on_page(canvas, doc_) -> None:
        _draw_header_footer(canvas, doc_, filename=filename, generated_at=generated_at, banner=banner)

    doc.build(story, onFirstPage=on_page, onLaterPages=on_page)
    return output_path
