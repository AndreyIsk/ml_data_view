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
    return {
        "overview": True,
        "distributions": has_distributions,
        "embeddings": has_embeddings,
        "media": has_media,
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
    "media"); a section is only actually included if available_sections()
    also finds content for it — requesting an unavailable section is a
    silent no-op for that section, not an error.
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
