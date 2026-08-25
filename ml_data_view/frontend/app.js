const openFileBtn = document.getElementById("open-file-btn");
const filePathSpan = document.getElementById("file-path");
const errorBanner = document.getElementById("error-banner");
const summarySection = document.getElementById("summary");
const rowCountSpan = document.getElementById("row-count");
const fileSizeSpan = document.getElementById("file-size");
const schemaSection = document.getElementById("schema-section");
const schemaTableBody = document.querySelector("#schema-table tbody");
const previewSection = document.getElementById("preview-section");
const previewTableHead = document.querySelector("#preview-table thead");
const previewTableBody = document.querySelector("#preview-table tbody");

// --- Chart color roles (dataviz skill's validated reference palette) ---
// One accent color for every single-series bar/histogram — a dashboard of
// small multiples reads as one coherent system, not a rainbow. The null
// heatmap uses a separate severity ramp (status "critical" red, never a
// categorical hue) since it encodes "how bad is the missingness", not identity.
const SERIES_COLOR = "#2a78d6"; // categorical slot 1 (blue)
const MISSING_COLOR = "#d03b3b"; // status: critical
const PRESENT_COLOR = "#f0efec"; // neutral surface-adjacent gray
const GRIDLINE_COLOR = "#e2e1dc"; // one-step-off-surface, hairline
const AXIS_TEXT_COLOR = "#52514e"; // text-secondary — axis text never wears the data color

const PLOT_FONT = { family: "system-ui, -apple-system, 'Segoe UI', sans-serif", color: AXIS_TEXT_COLOR, size: 11 };

// Plotly's chart containers need to react to ANY change in their own
// rendered size — a window resize, a CSS grid reflow that changes the
// column count, a sidebar toggle — not just a raw browser "resize" event
// (which pywebview's native window doesn't reliably fire for every OS-level
// transition, e.g. restoring from maximized). ResizeObserver is the right
// primitive for that — BUT it must watch a STABLE element that Plotly
// itself never resizes. Observing the exact div that gets resized creates a
// feedback loop (each resize re-triggers the observer on itself) — verified
// empirically: it hung the page outright. Instead, each chart's *stable
// parent* (the .chart-grid container, or the heatmap's section) is what
// gets observed; its own box size is pure CSS/grid layout, never written to
// by Plotly. .chart-plot/#null-heatmap-chart also get a fixed CSS height
// (see style.css) so Plotly's height handling can't feed back into the
// grid's auto-row-height either.
//
// Resizing itself uses a full Plotly.newPlot() redraw, not
// Plotly.Plots.resize(): the latter was verified (by decoding actual
// rendered widths) to leave charts at wrong, inconsistent sizes when
// called across several divs in sequence — each subsequent chart ended up
// narrower than the last in a way that didn't match any real container
// width, and it never self-corrected. A full newPlot() redraw with the
// same data/layout/config has been reliable in every test, including this
// exact multi-chart sequential pattern, so resize reuses it instead of a
// second, less-trustworthy code path. Each div's original arguments are
// cached in `plotArgs` so a redraw can reuse them.
const plotArgs = new WeakMap(); // div -> { data, layout, config }
const activePlotDivs = new Set();

async function drawPlot(div, data, layout, config) {
  plotArgs.set(div, { data, layout, config });
  activePlotDivs.add(div);
  await Plotly.newPlot(div, data, layout, config);
}

function untrackPlot(div) {
  activePlotDivs.delete(div);
  plotArgs.delete(div);
}

let resizeFlushScheduled = false;

async function flushPendingResizes() {
  resizeFlushScheduled = false;
  // Awaited sequentially, not fired concurrently: Plotly.newPlot() is
  // async, and unawaited concurrent calls race — verified empirically
  // (concurrent calls left one chart's width matching neither its own
  // container nor any other chart's).
  for (const div of activePlotDivs) {
    const args = plotArgs.get(div);
    if (div.isConnected && args) {
      await Plotly.newPlot(div, args.data, args.layout, args.config);
    }
  }
}

function scheduleResizeFlush() {
  if (!resizeFlushScheduled) {
    resizeFlushScheduled = true;
    setTimeout(flushPendingResizes, 120);
  }
}

const containerResizeObserver = new ResizeObserver(scheduleResizeFlush);

function observeContainer(container) {
  containerResizeObserver.observe(container);
}

function formatBytes(bytes) {
  if (bytes < 1024) return `${bytes} B`;
  const units = ["KB", "MB", "GB", "TB"];
  let value = bytes / 1024;
  let unitIndex = 0;
  while (value >= 1024 && unitIndex < units.length - 1) {
    value /= 1024;
    unitIndex += 1;
  }
  return `${value.toFixed(1)} ${units[unitIndex]}`;
}

function showError(message) {
  errorBanner.textContent = message;
  errorBanner.classList.remove("hidden");
  summarySection.classList.add("hidden");
  schemaSection.classList.add("hidden");
  previewSection.classList.add("hidden");
  document.getElementById("charts-section").classList.add("hidden");
}

function clearError() {
  errorBanner.classList.add("hidden");
  errorBanner.textContent = "";
}

function renderSchema(columns) {
  schemaTableBody.innerHTML = "";
  for (const column of columns) {
    const row = document.createElement("tr");
    row.innerHTML = `<td>${column.name}</td><td>${column.dtype}</td><td>${column.null_count}</td>`;
    schemaTableBody.appendChild(row);
  }
  schemaSection.classList.remove("hidden");
}

function renderPreview(columns, rows) {
  const headRow = document.createElement("tr");
  headRow.innerHTML = columns.map((c) => `<th>${c.name}</th>`).join("");
  previewTableHead.innerHTML = "";
  previewTableHead.appendChild(headRow);

  previewTableBody.innerHTML = "";
  for (const row of rows) {
    const tr = document.createElement("tr");
    tr.innerHTML = columns
      .map((c) => {
        const value = row[c.name];
        return `<td>${value === null || value === undefined ? "" : value}</td>`;
      })
      .join("");
    previewTableBody.appendChild(tr);
  }
  previewSection.classList.remove("hidden");
}

function renderDataset(data) {
  filePathSpan.textContent = data.file_path;
  rowCountSpan.textContent = data.row_count.toLocaleString();
  fileSizeSpan.textContent = formatBytes(data.file_size_bytes);
  summarySection.classList.remove("hidden");
  renderSchema(data.columns);
  renderPreview(data.columns, data.preview_rows);
}

async function handleOpenFile() {
  const path = await window.pywebview.api.open_file_dialog();
  if (!path) return;

  clearError();
  const data = await window.pywebview.api.load_file(path);
  if (data.error) {
    showError(data.error);
    return;
  }
  renderDataset(data);

  const chartData = await window.pywebview.api.get_chart_data();
  if (chartData.error) {
    showError(chartData.error);
    return;
  }
  await renderCharts(chartData);
}

openFileBtn.addEventListener("click", handleOpenFile);

async function renderNullHeatmap(data) {
  const section = document.getElementById("null-heatmap-section");
  if (!data.z.length) {
    section.classList.add("hidden");
    return;
  }
  section.classList.remove("hidden");

  // z rows are bins (each bin's cell is the *fraction* missing, 0.0-1.0),
  // not necessarily individual source rows — see null_heatmap_data()'s
  // docstring for why (Plotly can't rasterize more z-rows than the plot
  // has pixels of height). display_rows == sampled_rows when no binning
  // was needed, in which case fractions are always exactly 0 or 1.
  const isBinned = data.display_rows < data.sampled_rows;
  const yLabels = Array.from({ length: data.display_rows }, (_, i) => i);
  const heatmapDiv = document.getElementById("null-heatmap-chart");
  observeContainer(section); // stable — Plotly never resizes #null-heatmap-section itself
  await drawPlot(
    heatmapDiv,
    [
      {
        z: data.z,
        x: data.columns,
        y: yLabels,
        type: "heatmap",
        zmin: 0,
        zmax: 1,
        colorscale: [
          [0, PRESENT_COLOR],
          [1, MISSING_COLOR],
        ],
        showscale: false,
        xgap: 2,
        ygap: 1,
      },
    ],
    {
      font: PLOT_FONT,
      margin: { t: 10, l: 50, r: 10, b: 70 },
      height: 320,
      paper_bgcolor: "transparent",
      plot_bgcolor: "transparent",
      xaxis: { tickfont: PLOT_FONT },
      yaxis: {
        title: { text: isBinned ? "Row bin" : "Row index", font: PLOT_FONT },
        autorange: "reversed",
        tickfont: PLOT_FONT,
      },
    },
    { displayModeBar: false, responsive: true }
  );

  const note = document.getElementById("null-heatmap-note");
  if (isBinned) {
    note.textContent = `Rows 0-${data.sampled_rows} of ${data.total_rows} grouped into ${data.display_rows} bins (color intensity = % missing per bin).`;
  } else {
    note.textContent =
      data.total_rows > data.sampled_rows
        ? `Showing first ${data.sampled_rows} of ${data.total_rows} rows (missing = red).`
        : `All ${data.total_rows} rows shown (missing = red).`;
  }
}

async function renderColumnCharts(containerId, sectionId, charts, plotBuilder) {
  const container = document.getElementById(containerId);
  const section = document.getElementById(sectionId);
  for (const oldDiv of container.querySelectorAll(".chart-plot")) {
    untrackPlot(oldDiv);
  }
  container.innerHTML = "";

  const plottable = charts.filter((chart) => chart.counts && chart.counts.length);
  if (!plottable.length) {
    section.classList.add("hidden");
    return;
  }
  section.classList.remove("hidden");
  observeContainer(container); // stable — Plotly never resizes .chart-grid itself

  // Awaited sequentially, not fired concurrently: Plotly.newPlot() is async,
  // and unawaited concurrent calls race — verified empirically (concurrent
  // calls left one chart's width matching neither its own container nor
  // any other chart's).
  for (const chart of plottable) {
    const card = document.createElement("div");
    card.className = "chart-box";

    // Plain HTML title, not Plotly's layout.title: a title Plotly lays out
    // itself can only ever be as correct as Plotly's current size
    // computation for that div — if that's stale or wrong (see the resize
    // fixes above), the title moves with the bug. Plain HTML in normal
    // document flow can't be mispositioned by a Plotly layout bug.
    const title = document.createElement("div");
    title.className = "chart-title";
    title.textContent = chart.column;
    title.title = chart.column; // native tooltip if the name is truncated
    card.appendChild(title);

    const plotDiv = document.createElement("div");
    plotDiv.className = "chart-plot";
    card.appendChild(plotDiv);

    container.appendChild(card);
    await plotBuilder(plotDiv, chart);
  }
}

function binCenters(binEdges) {
  return binEdges.slice(0, -1).map((edge, i) => (edge + binEdges[i + 1]) / 2);
}

// A function, not a shared const object: Plotly.newPlot() mutates the
// layout object it's given in place (writing computed fields like `range`,
// `autorange`, `type` directly onto layout.xaxis/yaxis for its own
// bookkeeping). A single BAR_LAYOUT_DEFAULTS const reused via `{...spread}`
// only shallow-copies the top level — every chart's xaxis/yaxis pointed at
// the SAME nested objects, so each chart silently overwrote the previous
// one's axis state. Verified concretely: after rendering, the shared
// xaxis object had frozen in the *last* chart's numeric range
// (`{type: "linear", range: [2007, 2009]}`, from a "year" column), which
// every other chart — including categorical ones — then inherited,
// showing numeric year-like ticks instead of their own category labels.
// A factory returning fresh objects on every call closes that off.
function barLayout(margin) {
  return {
    font: PLOT_FONT,
    height: 220,
    bargap: 0.15,
    paper_bgcolor: "transparent",
    plot_bgcolor: "transparent",
    margin,
    xaxis: { tickfont: PLOT_FONT, gridcolor: GRIDLINE_COLOR, zeroline: false },
    yaxis: { tickfont: PLOT_FONT, gridcolor: GRIDLINE_COLOR, zeroline: false },
  };
}

async function plotNumericHistogram(div, chart) {
  await drawPlot(
    div,
    [{ x: binCenters(chart.bin_edges), y: chart.counts, type: "bar", marker: { color: SERIES_COLOR } }],
    barLayout({ t: 10, l: 40, r: 10, b: 30 }),
    { displayModeBar: false, responsive: true }
  );
}

async function plotCategoricalBar(div, chart) {
  await drawPlot(
    div,
    [{ x: chart.categories, y: chart.counts, type: "bar", marker: { color: SERIES_COLOR } }],
    barLayout({ t: 10, l: 40, r: 10, b: 50 }),
    { displayModeBar: false, responsive: true }
  );
}

async function plotTextLengthHistogram(div, chart) {
  await drawPlot(
    div,
    [{ x: binCenters(chart.bin_edges), y: chart.counts, type: "bar", marker: { color: SERIES_COLOR } }],
    barLayout({ t: 10, l: 40, r: 10, b: 30 }),
    { displayModeBar: false, responsive: true }
  );
}

async function renderCharts(chartData) {
  document.getElementById("charts-section").classList.remove("hidden");
  await renderNullHeatmap(chartData.null_heatmap);
  await renderColumnCharts(
    "numeric-histograms",
    "numeric-histograms-section",
    chartData.numeric_histograms,
    plotNumericHistogram
  );
  await renderColumnCharts(
    "categorical-bars",
    "categorical-bars-section",
    chartData.categorical_bars,
    plotCategoricalBar
  );
  await renderColumnCharts(
    "text-length-histograms",
    "text-length-histograms-section",
    chartData.text_length_histograms,
    plotTextLengthHistogram
  );
}
