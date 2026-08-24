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

// Plotly's `responsive: true` relies on an internal ResizeObserver that, in
// practice, does not reliably fire when a plot's container shrinks because
// of a layout reflow (e.g. the pywebview window un-maximizing) rather than
// a direct browser resize — the chart's SVG is left stale, oversized, and
// overflowing its container, which reads as charts overlapping one
// another. Tracking every plotted div and forcing an explicit resize on
// window "resize" (debounced) fixes that regardless of which internal
// mechanism failed to catch the shrink.
const plottedDivs = new Set();
let resizeDebounceTimer = null;

function trackPlot(div) {
  plottedDivs.add(div);
}

async function resizeTrackedPlots() {
  // Plotly.Plots.resize() is async (returns a Promise) — awaiting each one
  // in turn, rather than firing them all concurrently, avoids a race where
  // overlapping redraws read each other's in-progress layout and end up
  // with the wrong width (verified empirically: unawaited concurrent
  // resizes left one chart's SVG matching neither its own container nor
  // any other chart's width).
  for (const div of plottedDivs) {
    if (div.isConnected) {
      await Plotly.Plots.resize(div);
    } else {
      plottedDivs.delete(div);
    }
  }
}

window.addEventListener("resize", () => {
  clearTimeout(resizeDebounceTimer);
  resizeDebounceTimer = setTimeout(resizeTrackedPlots, 150);
});

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
  renderCharts(chartData);
}

openFileBtn.addEventListener("click", handleOpenFile);

function renderNullHeatmap(data) {
  const section = document.getElementById("null-heatmap-section");
  if (!data.z.length) {
    section.classList.add("hidden");
    return;
  }
  section.classList.remove("hidden");

  const yLabels = Array.from({ length: data.sampled_rows }, (_, i) => i);
  const heatmapDiv = document.getElementById("null-heatmap-chart");
  trackPlot(heatmapDiv);
  Plotly.newPlot(
    heatmapDiv,
    [
      {
        z: data.z,
        x: data.columns,
        y: yLabels,
        type: "heatmap",
        colorscale: [
          [0, "#f5f5f5"],
          [1, "#c0392b"],
        ],
        showscale: false,
        xgap: 1,
        ygap: 1,
      },
    ],
    {
      margin: { t: 10, l: 50, r: 10, b: 80 },
      height: 320,
      yaxis: { title: "Row index", autorange: "reversed" },
    },
    { displayModeBar: false, responsive: true }
  );

  const note = document.getElementById("null-heatmap-note");
  note.textContent =
    data.total_rows > data.sampled_rows
      ? `Showing first ${data.sampled_rows} of ${data.total_rows} rows (missing = red).`
      : `All ${data.total_rows} rows shown (missing = red).`;
}

function renderColumnCharts(containerId, sectionId, charts, plotBuilder) {
  const container = document.getElementById(containerId);
  const section = document.getElementById(sectionId);
  container.innerHTML = "";

  const plottable = charts.filter((chart) => chart.counts && chart.counts.length);
  if (!plottable.length) {
    section.classList.add("hidden");
    return;
  }
  section.classList.remove("hidden");

  for (const chart of plottable) {
    const chartDiv = document.createElement("div");
    chartDiv.className = "chart-box";
    container.appendChild(chartDiv);
    plotBuilder(chartDiv, chart);
  }
}

function binCenters(binEdges) {
  return binEdges.slice(0, -1).map((edge, i) => (edge + binEdges[i + 1]) / 2);
}

function plotNumericHistogram(div, chart) {
  trackPlot(div);
  Plotly.newPlot(
    div,
    [{ x: binCenters(chart.bin_edges), y: chart.counts, type: "bar" }],
    { title: chart.column, margin: { t: 30, l: 40, r: 10, b: 40 }, height: 260 },
    { displayModeBar: false, responsive: true }
  );
}

function plotCategoricalBar(div, chart) {
  trackPlot(div);
  Plotly.newPlot(
    div,
    [{ x: chart.categories, y: chart.counts, type: "bar" }],
    { title: chart.column, margin: { t: 30, l: 40, r: 10, b: 60 }, height: 260 },
    { displayModeBar: false, responsive: true }
  );
}

function plotTextLengthHistogram(div, chart) {
  trackPlot(div);
  Plotly.newPlot(
    div,
    [{ x: binCenters(chart.bin_edges), y: chart.counts, type: "bar" }],
    {
      title: `${chart.column} (length)`,
      margin: { t: 30, l: 40, r: 10, b: 40 },
      height: 260,
    },
    { displayModeBar: false, responsive: true }
  );
}

function renderCharts(chartData) {
  document.getElementById("charts-section").classList.remove("hidden");
  renderNullHeatmap(chartData.null_heatmap);
  renderColumnCharts(
    "numeric-histograms",
    "numeric-histograms-section",
    chartData.numeric_histograms,
    plotNumericHistogram
  );
  renderColumnCharts(
    "categorical-bars",
    "categorical-bars-section",
    chartData.categorical_bars,
    plotCategoricalBar
  );
  renderColumnCharts(
    "text-length-histograms",
    "text-length-histograms-section",
    chartData.text_length_histograms,
    plotTextLengthHistogram
  );
}
