"""Python-side Plotly figure builders for PDF export (Stage 3's Reports
section).

Reuses chart_data.py/embedding_data.py's already-computed data (bin edges,
counts, PCA coordinates, ...) as-is — this module only turns that data into
a Plotly Figure and, via kaleido, a static PNG. No chart *logic* (binning,
top-N bucketing, PCA) lives here; that all stays in chart_data.py/
embedding_data.py so the live dashboard and the PDF report are always
looking at the same numbers.

Color roles mirror frontend/app.js's dataviz-skill palette for visual
consistency between the live dashboard and the PDF, but pixel-for-pixel
fidelity isn't the goal — this is a static, non-interactive embed.
"""
from __future__ import annotations

from typing import Any

import plotly.graph_objects as go

SERIES_COLOR = "#2a78d6"
MISSING_COLOR = "#d03b3b"
PRESENT_COLOR = "#f0efec"
GRIDLINE_COLOR = "#e2e1dc"
AXIS_TEXT_COLOR = "#52514e"
PLOT_FONT = dict(family="Helvetica, Arial, sans-serif", color=AXIS_TEXT_COLOR, size=11)

# Fixed per-figure size (not just a default): report.py reads a figure's own
# layout.width/height back to compute its embedded aspect ratio, so every
# figure builder below must set both.
CHART_WIDTH = 700
CHART_HEIGHT = 380


def _bin_centers(bin_edges: list[float]) -> list[float]:
    return [(edge + bin_edges[i + 1]) / 2 for i, edge in enumerate(bin_edges[:-1])]


def format_axis_value(value: float) -> str:
    """Compact tick label for log-scale bin edges, whose original-unit
    values span orders of magnitude and would otherwise overlap."""
    magnitude = abs(value)
    if magnitude >= 1e9:
        return f"{value / 1e9:.1f}B"
    if magnitude >= 1e6:
        return f"{value / 1e6:.1f}M"
    if magnitude >= 1e3:
        return f"{value / 1e3:.1f}k"
    if magnitude >= 10:
        return f"{value:.0f}"
    return f"{value:.3g}"


def _bar_layout(title: str, width: int = CHART_WIDTH, height: int = CHART_HEIGHT) -> dict[str, Any]:
    return dict(
        title=dict(text=title, font=PLOT_FONT),
        font=PLOT_FONT,
        width=width,
        height=height,
        bargap=0.15,
        paper_bgcolor="white",
        plot_bgcolor="white",
        margin=dict(t=40, l=50, r=20, b=40),
        xaxis=dict(tickfont=PLOT_FONT, gridcolor=GRIDLINE_COLOR, zeroline=False),
        yaxis=dict(tickfont=PLOT_FONT, gridcolor=GRIDLINE_COLOR, zeroline=False),
    )


def numeric_histogram_figure(
    chart: dict[str, Any], width: int = CHART_WIDTH, height: int = CHART_HEIGHT
) -> go.Figure | None:
    if not chart.get("counts"):
        return None
    counts = chart["counts"]
    edges = chart["bin_edges"]

    if chart.get("x_scale") == "log":
        # Log-spaced bins are equal-width in log space, so the bars are
        # plotted evenly spaced by index and the axis is labelled with the
        # original-unit edges. Plotting the original values on a Plotly
        # log axis instead would fail outright on the zeros and negatives
        # these columns contain (chart_data offsets before log1p for the
        # same reason).
        positions = list(range(len(counts)))
        step = max(1, len(counts) // 6)
        tickvals = positions[::step]
        layout = _bar_layout(f"{chart['column']} (log scale)", width, height)
        layout["xaxis"].update(
            tickmode="array",
            tickvals=tickvals,
            ticktext=[format_axis_value(edges[i]) for i in tickvals],
        )
        return go.Figure(data=[go.Bar(x=positions, y=counts, marker_color=SERIES_COLOR)], layout=layout)

    return go.Figure(
        data=[go.Bar(x=_bin_centers(edges), y=counts, marker_color=SERIES_COLOR)],
        layout=_bar_layout(chart["column"], width, height),
    )


def categorical_bar_figure(
    chart: dict[str, Any], width: int = CHART_WIDTH, height: int = CHART_HEIGHT
) -> go.Figure | None:
    if not chart.get("counts"):
        return None
    return go.Figure(
        data=[go.Bar(x=chart["categories"], y=chart["counts"], marker_color=SERIES_COLOR)],
        layout=_bar_layout(chart["column"], width, height),
    )


def text_length_histogram_figure(
    chart: dict[str, Any], width: int = CHART_WIDTH, height: int = CHART_HEIGHT
) -> go.Figure | None:
    if not chart.get("counts"):
        return None
    return go.Figure(
        data=[go.Bar(x=_bin_centers(chart["bin_edges"]), y=chart["counts"], marker_color=SERIES_COLOR)],
        layout=_bar_layout(chart["column"], width, height),
    )


def null_heatmap_figure(data: dict[str, Any]) -> go.Figure | None:
    if not data.get("z"):
        return None
    y_labels = list(range(data["display_rows"]))
    return go.Figure(
        data=[
            go.Heatmap(
                z=data["z"],
                x=data["columns"],
                y=y_labels,
                zmin=0,
                zmax=1,
                colorscale=[[0, PRESENT_COLOR], [1, MISSING_COLOR]],
                showscale=False,
                xgap=2,
                ygap=1,
            )
        ],
        layout=dict(
            title=dict(text="Missing-value heatmap", font=PLOT_FONT),
            font=PLOT_FONT,
            width=CHART_WIDTH,
            height=CHART_HEIGHT,
            paper_bgcolor="white",
            plot_bgcolor="white",
            margin=dict(t=40, l=60, r=20, b=50),
            xaxis=dict(tickfont=PLOT_FONT),
            yaxis=dict(
                title=dict(text="Row bin", font=PLOT_FONT), autorange="reversed", tickfont=PLOT_FONT
            ),
        ),
    )


def embedding_scatter_figure(projection: dict[str, Any]) -> go.Figure | None:
    if not projection.get("x"):
        return None
    return go.Figure(
        data=[
            go.Scatter(
                x=projection["x"],
                y=projection["y"],
                mode="markers",
                marker=dict(size=6, color=SERIES_COLOR, opacity=0.85),
            )
        ],
        layout=dict(
            title=dict(text=f"Embedding projection: {projection['column']}", font=PLOT_FONT),
            font=PLOT_FONT,
            width=CHART_WIDTH,
            height=CHART_HEIGHT,
            paper_bgcolor="white",
            plot_bgcolor="white",
            margin=dict(t=40, l=50, r=20, b=40),
            xaxis=dict(
                title=dict(text="PC1", font=PLOT_FONT),
                tickfont=PLOT_FONT,
                gridcolor=GRIDLINE_COLOR,
                zeroline=False,
            ),
            yaxis=dict(
                title=dict(text="PC2", font=PLOT_FONT),
                tickfont=PLOT_FONT,
                gridcolor=GRIDLINE_COLOR,
                zeroline=False,
            ),
        ),
    )


def figure_png_bytes(fig: go.Figure) -> bytes:
    """Rasterize a figure to PNG bytes via kaleido. Each figure builder
    above sets its own layout.width/height, so no size args are needed
    here."""
    return fig.to_image(format="png")
