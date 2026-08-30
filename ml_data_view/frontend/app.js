const openFileBtn = document.getElementById("open-file-btn");
const openFolderBtn = document.getElementById("open-folder-btn");
const filePathSpan = document.getElementById("file-path");
const errorBanner = document.getElementById("error-banner");
const summarySection = document.getElementById("summary");
const rowCountSpan = document.getElementById("row-count");
const fileSizeSpan = document.getElementById("file-size");
const summaryFileCountRow = document.getElementById("summary-file-count-row");
const summaryFileCountSpan = document.getElementById("summary-file-count");
const schemaSection = document.getElementById("schema-section");
const schemaTableBody = document.querySelector("#schema-table tbody");
const previewSection = document.getElementById("preview-section");
const previewTableHead = document.querySelector("#preview-table thead");
const previewTableBody = document.querySelector("#preview-table tbody");
const loadingIndicator = document.getElementById("loading-indicator");
const loadingMessage = document.getElementById("loading-message");
const loadProgress = document.getElementById("load-progress");
const loadProgressBar = document.getElementById("load-progress-bar");
const loadProgressPct = document.getElementById("load-progress-pct");
const stopLoadBtn = document.getElementById("stop-load-btn");
const partialBanner = document.getElementById("partial-banner");

const TEXT_TRUNCATE_LENGTH = 100; // Phase 3: preview-table cells longer than this get a "…" + click-to-expand

// One shared modal, reused for both the text-cell popup and the image
// "view larger" click — only one of these is ever open at a time.
const detailModal = document.getElementById("detail-modal");
const modalBody = document.getElementById("modal-body");
const modalCloseBtn = document.getElementById("modal-close-btn");

function setModalContent(build) {
  modalBody.innerHTML = "";
  build(modalBody);
}

function showModal() {
  detailModal.classList.remove("hidden");
}

function closeModal() {
  detailModal.classList.add("hidden");
  modalBody.innerHTML = "";
}

modalCloseBtn.addEventListener("click", closeModal);
detailModal.querySelector(".modal-backdrop").addEventListener("click", closeModal);
document.addEventListener("keydown", (event) => {
  if (event.key === "Escape" && !detailModal.classList.contains("hidden")) closeModal();
});

function showTextModal(fullText) {
  setModalContent((body) => {
    const p = document.createElement("p");
    p.className = "modal-text";
    p.textContent = fullText;
    body.appendChild(p);
  });
  showModal();
}

async function showImageModal(column, source, rowIndex) {
  setModalContent((body) => {
    const loading = document.createElement("p");
    loading.className = "chart-note";
    loading.textContent = "Loading…";
    body.appendChild(loading);
  });
  showModal();

  const result = await window.pywebview.api.get_image_full(column, source, rowIndex);
  setModalContent((body) => {
    if (result.error) {
      const error = document.createElement("p");
      error.className = "chart-note";
      error.textContent = result.error;
      body.appendChild(error);
      return;
    }
    const img = document.createElement("img");
    img.className = "modal-image";
    img.src = result.image;
    img.alt = `${column} row ${rowIndex}`;
    body.appendChild(img);
  });
}

// A large file's read + chart computation can take a visible moment (still
// far faster than it used to be — see chart_data.py's polars-native
// histogram binning — but "far faster" isn't "instant" for a many-million-
// row file). Comfort here means never leaving the user looking at an
// unresponsive button with no feedback: show what's happening, and don't
// let a second click start a second load while one is in flight.
function showLoading(message) {
  loadingMessage.textContent = message;
  loadingIndicator.classList.remove("hidden");
  openFileBtn.disabled = true;
  // Any new load invalidates a previous load's "partial" warning — it must
  // never outlive the dataset it described.
  partialBanner.classList.add("hidden");
}

function hideLoading() {
  loadingIndicator.classList.add("hidden");
  openFileBtn.disabled = false;
}

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

// Embedding scatter color-by (dataviz skill's validated palette, palette.md):
// scatter is an "all-pairs" chart form — any two categories can land next to
// each other anywhere in the cloud, unlike a bar chart where only neighboring
// bars are compared — so only the first 3 categorical slots (of the 8-slot
// reference palette) validate colorblind-safety across every pair. The
// backend buckets each color-by column to top-3 + "Other" to match (see
// embedding_data.EMBEDDING_COLOR_TOP_N); "Other"/"Missing" get non-identity
// neutrals, never a 4th hue.
const EMBEDDING_CATEGORICAL_COLORS = ["#2a78d6", "#eb6834", "#1baf7a"]; // slots 1-3: blue, orange, aqua
const EMBEDDING_OTHER_COLOR = "#898781"; // muted ink — non-identity bucket
const EMBEDDING_MISSING_COLOR = "#c3c2b7"; // baseline/border gray — distinct neutral from "Other"
const MARKER_SURFACE_RING = "#ffffff"; // surface color — separates overlapping markers, not a data-carrying stroke

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
  document.querySelectorAll(".chart-content").forEach((el) => el.classList.add("hidden"));
  showReportsEmptyState();
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

function renderPreview(columns, rows, textColumns) {
  const headRow = document.createElement("tr");
  headRow.innerHTML = columns.map((c) => `<th>${c.name}</th>`).join("");
  previewTableHead.innerHTML = "";
  previewTableHead.appendChild(headRow);

  previewTableBody.innerHTML = "";
  for (const row of rows) {
    const tr = document.createElement("tr");
    for (const c of columns) {
      const value = row[c.name];
      const td = document.createElement("td");
      // Phase 3: only Phase 1's "text"-classified columns get truncated —
      // a categorical/id column with one unusually long value stays as-is.
      if (textColumns.has(c.name) && typeof value === "string" && value.length > TEXT_TRUNCATE_LENGTH) {
        td.textContent = `${value.slice(0, TEXT_TRUNCATE_LENGTH)}…`;
        td.className = "truncated-cell";
        td.title = "Click to view full text";
        td.addEventListener("click", () => showTextModal(value));
      } else {
        td.textContent = value === null || value === undefined ? "" : value;
      }
      tr.appendChild(td);
    }
    previewTableBody.appendChild(tr);
  }
  previewSection.classList.remove("hidden");
}

function renderDataset(data) {
  filePathSpan.textContent = data.file_path;
  rowCountSpan.textContent = data.row_count.toLocaleString();
  fileSizeSpan.textContent = formatBytes(data.file_size_bytes);
  summarySection.classList.remove("hidden");
  // Folder loads (Stage 2) only — a single opened file has no "file count"
  // of its own, so this stays hidden for the ordinary Open File path.
  if (data.source_type === "folder") {
    summaryFileCountSpan.textContent = data.file_count.toLocaleString();
    summaryFileCountRow.classList.remove("hidden");
  } else {
    summaryFileCountRow.classList.add("hidden");
  }
  renderSchema(data.columns);
  renderPreview(data.columns, data.preview_rows, new Set(data.text_columns || []));
  // search.js loads after this file; the call happens at runtime, so the
  // function is defined by the time a dataset is opened.
  resetSearchForDataset(data.columns);
}

// Shared by handleOpenFile and handleOpenFolder once each has its own
// `data` (the load_file()/load_folder_group() result): render it, then
// compute and render charts + the Reports section from the same loaded
// dataset. Returns false (having already shown the error) on any failure.
async function finishLoadingDataset(data) {
  if (data.error) {
    showError(data.error);
    return false;
  }
  renderDataset(data);

  // Chart generation runs ~0.54s per file (measured), so on a large folder
  // this is minutes — the same start-and-poll shape as the folder load,
  // rather than one blocking call behind a static spinner.
  showLoading("Computing charts…");
  showProgress({ stoppable: false });
  let chartData;
  try {
    const started = await window.pywebview.api.start_chart_build();
    if (started.error) {
      showError(started.error);
      return false;
    }
    const state = await pollChartBuild();
    if (state.error) {
      showError(state.error);
      return false;
    }
    chartData = state.result;
  } finally {
    hideProgress();
  }
  if (!chartData) {
    showError("Chart generation produced no data.");
    return false;
  }
  await renderCharts(chartData);
  await refreshReportsSection();
  return true;
}

async function handleOpenFile() {
  const path = await window.pywebview.api.open_file_dialog();
  if (!path) return;

  clearError();
  showLoading("Reading file…");
  try {
    const data = await window.pywebview.api.load_file(path);
    await finishLoadingDataset(data);
  } finally {
    hideLoading();
  }
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

// Compact label for log-scale tick values, which span orders of magnitude.
// Mirrors report_charts.format_axis_value() so the live chart and the PDF
// label the same axis the same way.
function formatAxisValue(value) {
  const magnitude = Math.abs(value);
  if (magnitude >= 1e9) return `${(value / 1e9).toFixed(1)}B`;
  if (magnitude >= 1e6) return `${(value / 1e6).toFixed(1)}M`;
  if (magnitude >= 1e3) return `${(value / 1e3).toFixed(1)}k`;
  if (magnitude >= 10) return value.toFixed(0);
  return String(Number(value.toPrecision(3)));
}

async function plotNumericHistogram(div, chart) {
  const layout = barLayout({ t: 10, l: 40, r: 10, b: 30 });

  if (chart.x_scale === "log") {
    // Bins are already log-spaced (chart_data bins in log1p space), so the
    // bars go at even positions and the axis carries original-unit labels.
    // Plotly's own log axis type can't be used: these columns contain
    // zeros and negatives, which a log axis drops silently.
    const positions = chart.counts.map((_, i) => i);
    const step = Math.max(1, Math.floor(chart.counts.length / 6));
    const tickvals = positions.filter((i) => i % step === 0);
    layout.xaxis.tickmode = "array";
    layout.xaxis.tickvals = tickvals;
    layout.xaxis.ticktext = tickvals.map((i) => formatAxisValue(chart.bin_edges[i]));
    layout.title = { text: "log scale", font: PLOT_FONT, x: 0, xanchor: "left" };
    await drawPlot(
      div,
      [{ x: positions, y: chart.counts, type: "bar", marker: { color: SERIES_COLOR } }],
      layout,
      { displayModeBar: false, responsive: true }
    );
    return;
  }

  await drawPlot(
    div,
    [{ x: binCenters(chart.bin_edges), y: chart.counts, type: "bar", marker: { color: SERIES_COLOR } }],
    layout,
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

function scatterLayout(margin) {
  return {
    font: PLOT_FONT,
    height: 320,
    paper_bgcolor: "transparent",
    plot_bgcolor: "transparent",
    margin,
    xaxis: { title: { text: "PC1", font: PLOT_FONT }, tickfont: PLOT_FONT, gridcolor: GRIDLINE_COLOR, zeroline: false },
    yaxis: { title: { text: "PC2", font: PLOT_FONT }, tickfont: PLOT_FONT, gridcolor: GRIDLINE_COLOR, zeroline: false },
    legend: { font: PLOT_FONT },
  };
}

// One Plotly trace per label bucket, so Plotly generates the legend for free
// (dataviz skill: identity is never color-alone — the legend's text label is
// what makes "Other"/"Missing" and the top-3 categories distinguishable, on
// top of color). `labels` is null when no color-by column is selected.
function buildEmbeddingTraces(x, y, labels) {
  if (!labels) {
    return [
      {
        x,
        y,
        type: "scattergl",
        mode: "markers",
        name: "points",
        showlegend: false,
        marker: { size: 8, color: SERIES_COLOR, opacity: 0.85, line: { width: 2, color: MARKER_SURFACE_RING } },
        hovertemplate: "PC1: %{x:.3f}<br>PC2: %{y:.3f}<extra></extra>",
      },
    ];
  }

  const groups = new Map();
  labels.forEach((label, i) => {
    if (!groups.has(label)) groups.set(label, { x: [], y: [] });
    const group = groups.get(label);
    group.x.push(x[i]);
    group.y.push(y[i]);
  });

  // Backend caps color-by at top-3 categories + "Other"/"Missing" (see
  // embedding_data.EMBEDDING_COLOR_TOP_N), so at most 3 non-bucket labels —
  // order of first appearance is enough to hand them out slots 1-3.
  const categoryOrder = [...groups.keys()].filter((label) => label !== "Other" && label !== "Missing");
  const orderedLabels = [...categoryOrder, ...(groups.has("Other") ? ["Other"] : []), ...(groups.has("Missing") ? ["Missing"] : [])];

  return orderedLabels.map((label) => {
    let color;
    if (label === "Other") color = EMBEDDING_OTHER_COLOR;
    else if (label === "Missing") color = EMBEDDING_MISSING_COLOR;
    else color = EMBEDDING_CATEGORICAL_COLORS[categoryOrder.indexOf(label)];

    const group = groups.get(label);
    return {
      x: group.x,
      y: group.y,
      type: "scattergl",
      mode: "markers",
      name: label,
      marker: { size: 8, color, opacity: 0.85, line: { width: 2, color: MARKER_SURFACE_RING } },
      hovertemplate: `${label}<br>PC1: %{x:.3f}<br>PC2: %{y:.3f}<extra></extra>`,
    };
  });
}

async function drawEmbeddingScatter(plotDiv, projection, colorByColumn) {
  const labels = colorByColumn ? projection.categorical_colors[colorByColumn] : null;
  const traces = buildEmbeddingTraces(projection.x, projection.y, labels);
  const layout = scatterLayout({ t: 10, l: 50, r: 10, b: 40 });
  layout.showlegend = traces.length > 1;
  await drawPlot(plotDiv, traces, layout, { displayModeBar: false, responsive: true });
}

function renderEmbeddingCard(projection) {
  const card = document.createElement("div");
  card.className = "embedding-card";

  const plotDiv = document.createElement("div");
  plotDiv.className = "embedding-plot chart-plot";

  const header = document.createElement("div");
  header.className = "embedding-card-header";

  const title = document.createElement("div");
  title.className = "chart-title";
  title.textContent = projection.column;
  title.title = projection.column;
  header.appendChild(title);

  const colorByColumns = Object.keys(projection.categorical_colors);
  if (colorByColumns.length) {
    const colorByWrap = document.createElement("label");
    colorByWrap.className = "embedding-color-by";
    colorByWrap.textContent = "Color by ";
    const select = document.createElement("select");
    select.appendChild(new Option("None", ""));
    for (const column of colorByColumns) {
      select.appendChild(new Option(column, column));
    }
    colorByWrap.appendChild(select);
    header.appendChild(colorByWrap);

    select.addEventListener("change", () => {
      drawEmbeddingScatter(plotDiv, projection, select.value || null);
    });
  }
  card.appendChild(header);

  if (projection.is_sample) {
    const note = document.createElement("p");
    note.className = "chart-note";
    note.textContent = `Showing a random sample of ${projection.sampled_rows.toLocaleString()} of ${projection.total_rows.toLocaleString()} rows.`;
    card.appendChild(note);
  }

  card.appendChild(plotDiv);
  return { card, plotDiv };
}

async function renderEmbeddingProjections(projections) {
  const container = document.getElementById("embedding-projections");
  const section = document.getElementById("embedding-projections-section");
  for (const oldDiv of container.querySelectorAll(".embedding-plot")) {
    untrackPlot(oldDiv);
  }
  container.innerHTML = "";

  const plottable = (projections || []).filter((p) => p.x.length);
  if (!plottable.length) {
    section.classList.add("hidden");
    return;
  }
  section.classList.remove("hidden");
  observeContainer(container); // stable — Plotly never resizes #embedding-projections itself

  for (const projection of plottable) {
    const { card, plotDiv } = renderEmbeddingCard(projection);
    container.appendChild(card);
    await drawEmbeddingScatter(plotDiv, projection, null);
  }
}

function renderImagePreviewCard(preview) {
  const card = document.createElement("div");
  card.className = "image-preview-card";

  const title = document.createElement("div");
  title.className = "chart-title";
  title.textContent = preview.column;
  title.title = preview.column;
  card.appendChild(title);

  const shown = preview.rows.length;
  const note = document.createElement("p");
  note.className = "chart-note";
  note.textContent =
    preview.total_rows > shown
      ? `Showing the first ${shown} of ${preview.total_rows.toLocaleString()} rows.`
      : `Showing all ${shown.toLocaleString()} row${shown === 1 ? "" : "s"}.`;
  card.appendChild(note);

  const grid = document.createElement("div");
  grid.className = "image-thumbnail-grid";
  for (const row of preview.rows) {
    const cell = document.createElement("div");
    cell.className = "image-thumbnail-cell";
    if (row.thumbnail) {
      // loading="lazy": thumbnails are all generated server-side up front
      // (cheap at 50 x 100x100px), but the browser defers decode/paint of
      // off-screen ones until they scroll into view.
      const img = document.createElement("img");
      img.loading = "lazy";
      img.src = row.thumbnail;
      img.alt = `${preview.column} row ${row.row_index}`;
      img.addEventListener("click", () => showImageModal(preview.column, preview.source, row.row_index));
      cell.appendChild(img);
    } else {
      cell.classList.add("thumbnail-broken");
      cell.textContent = "No preview";
    }
    grid.appendChild(cell);
  }
  card.appendChild(grid);

  return card;
}

function renderImagePreviews(imagePreviews) {
  const container = document.getElementById("image-previews");
  const section = document.getElementById("image-previews-section");
  container.innerHTML = "";

  const plottable = imagePreviews || [];
  if (!plottable.length) {
    section.classList.add("hidden");
    return;
  }
  section.classList.remove("hidden");

  for (const preview of plottable) {
    container.appendChild(renderImagePreviewCard(preview));
  }
}

function renderCorrelatedMissingness(groups) {
  const section = document.getElementById("correlated-missingness-section");
  const list = document.getElementById("correlated-missingness-list");
  list.textContent = "";
  if (!groups || groups.length === 0) {
    section.classList.add("hidden");
    return;
  }
  for (const group of groups) {
    const row = document.createElement("div");
    row.className = "correlated-group";

    const names = document.createElement("span");
    names.className = "correlated-group-columns";
    names.textContent = group.columns.join(", ");

    const detail = document.createElement("span");
    detail.className = "correlated-group-detail";
    detail.textContent =
      `${group.null_count.toLocaleString()} rows missing · ` +
      `${(group.overlap * 100).toFixed(1)}% overlap`;

    row.append(names, detail);
    list.appendChild(row);
  }
  section.classList.remove("hidden");
}

async function renderCharts(chartData) {
  document.querySelectorAll(".chart-content").forEach((el) => el.classList.remove("hidden"));
  renderCorrelatedMissingness(chartData.correlated_missingness);
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
  await renderEmbeddingProjections(chartData.embedding_projections);
  renderImagePreviews(chartData.image_previews);
}

// --- Left-hand nav: pure layout switching, no effect on data/chart state ---
// Sections stay in their normal DOM position at all times; only the active
// one is unhidden via the existing `.hidden` (display:none) convention.
// Switching sections changes a chart's container size (0 while hidden, real
// once shown) — that's exactly what the ResizeObserver + full-redraw resize
// pipeline above already exists to handle (see the long comment on
// `plotArgs`/`flushPendingResizes`), so no extra redraw hook is needed here.
const navButtons = document.querySelectorAll(".nav-item");
const appSections = document.querySelectorAll(".app-section");

function activateSection(sectionName) {
  for (const button of navButtons) {
    button.classList.toggle("active", button.dataset.section === sectionName);
  }
  for (const section of appSections) {
    section.classList.toggle("hidden", section.id !== `section-${sectionName}`);
  }
}

for (const button of navButtons) {
  button.addEventListener("click", () => activateSection(button.dataset.section));
}

// --- Settings: classification banner + audit log ---
// Settings live entirely on the Python side (ml_data_view/settings.py) —
// loaded once on startup, re-saved (and re-read back) on every change, so
// this file only ever renders whatever the backend confirms was persisted.
const classificationBannerSlot = document.getElementById("classification-banner-slot");
const bottomStrip = document.getElementById("bottom-strip");
const settingsBannerEnabled = document.getElementById("settings-banner-enabled");
const settingsBannerLabel = document.getElementById("settings-banner-label");
const settingsBannerColor = document.getElementById("settings-banner-color");
const settingsAuditEnabled = document.getElementById("settings-audit-enabled");
const settingsAuditLogPath = document.getElementById("settings-audit-log-path");

// Simple relative-luminance check (WCAG-style, not exact) so a light banner
// color gets dark text and a dark one gets white text — the label stays
// readable regardless of what color the user picks.
function contrastingTextColor(hexColor) {
  const hex = hexColor.replace("#", "");
  const r = parseInt(hex.slice(0, 2), 16) / 255;
  const g = parseInt(hex.slice(2, 4), 16) / 255;
  const b = parseInt(hex.slice(4, 6), 16) / 255;
  const luminance = 0.299 * r + 0.587 * g + 0.114 * b;
  return luminance > 0.6 ? "#1a1a1a" : "#ffffff";
}

// Applies the persisted settings to both banner slots (top strip +
// #bottom-strip). Disabled means genuinely empty/hidden in both spots — no
// leftover height or color — so the app looks exactly as it does today.
function renderClassificationBanner(settings) {
  if (!settings.classification_banner_enabled) {
    classificationBannerSlot.innerHTML = "";
    bottomStrip.classList.add("hidden");
    bottomStrip.textContent = "";
    return;
  }

  const label = settings.classification_banner_label || "";
  const color = settings.classification_banner_color;
  const textColor = contrastingTextColor(color);

  classificationBannerSlot.innerHTML = "";
  const chip = document.createElement("span");
  chip.className = "classification-banner-chip";
  chip.style.background = color;
  chip.style.color = textColor;
  chip.textContent = label;
  classificationBannerSlot.appendChild(chip);

  bottomStrip.textContent = label;
  bottomStrip.style.background = color;
  bottomStrip.style.color = textColor;
  bottomStrip.classList.remove("hidden");
}

function applySettingsToForm(settings) {
  settingsBannerEnabled.checked = settings.classification_banner_enabled;
  settingsBannerLabel.value = settings.classification_banner_label;
  settingsBannerColor.value = settings.classification_banner_color;
  settingsAuditEnabled.checked = settings.audit_log_enabled;
  settingsAuditLogPath.textContent = settings.audit_log_path;
}

// Saves immediately (no explicit Save button, per the settings spec) and
// re-renders from whatever the backend confirms it wrote — never from the
// raw form state — so the UI can't drift from what's actually on disk.
async function saveSettings(update) {
  const settings = await window.pywebview.api.save_settings(update);
  applySettingsToForm(settings);
  renderClassificationBanner(settings);
}

settingsBannerEnabled.addEventListener("change", () => {
  saveSettings({ classification_banner_enabled: settingsBannerEnabled.checked });
});
settingsBannerLabel.addEventListener("change", () => {
  saveSettings({ classification_banner_label: settingsBannerLabel.value });
});
settingsBannerColor.addEventListener("change", () => {
  saveSettings({ classification_banner_color: settingsBannerColor.value });
});
settingsAuditEnabled.addEventListener("change", () => {
  saveSettings({ audit_log_enabled: settingsAuditEnabled.checked });
});

async function initSettings() {
  const settings = await window.pywebview.api.get_settings();
  applySettingsToForm(settings);
  renderClassificationBanner(settings);
}

// pywebview injects window.pywebview.api asynchronously; `pywebviewready`
// fires once it's actually available (it may already have fired by the
// time this script runs, hence the direct check first).
if (window.pywebview) {
  initSettings();
} else {
  window.addEventListener("pywebviewready", initSettings);
}

// --- Reports: PDF export ---
// Section availability, chart images, and the audit-log entry all come
// from the backend (ml_data_view/report.py) — this file only offers the
// checkboxes it's told about and hands the user's choice back.
const reportsEmptyState = document.getElementById("reports-empty-state");
const reportsPanel = document.getElementById("reports-panel");
const reportsSectionCheckboxes = document.getElementById("reports-section-checkboxes");
const generateReportBtn = document.getElementById("generate-report-btn");
const reportsResult = document.getElementById("reports-result");

const REPORT_SECTION_ORDER = ["overview", "distributions", "embeddings", "media"];
const REPORT_SECTION_LABELS = {
  overview: "Overview",
  distributions: "Distributions",
  embeddings: "Embeddings",
  media: "Media",
};

function showReportsEmptyState() {
  reportsEmptyState.classList.remove("hidden");
  reportsPanel.classList.add("hidden");
}

// Builds one checkbox per available section (default checked, per the
// Reports spec) — "available" already means "this dataset actually has
// content there" (report.available_sections mirrors the same checks
// renderColumnCharts/renderEmbeddingProjections/renderImagePreviews use
// above), so nothing here re-decides that.
function renderReportSectionCheckboxes(sections) {
  reportsSectionCheckboxes.innerHTML = "";
  for (const key of REPORT_SECTION_ORDER) {
    if (!sections[key]) continue;
    const label = document.createElement("label");
    label.className = "settings-field-checkbox";
    const checkbox = document.createElement("input");
    checkbox.type = "checkbox";
    checkbox.value = key;
    checkbox.checked = true;
    label.appendChild(checkbox);
    label.appendChild(document.createTextNode(REPORT_SECTION_LABELS[key]));
    reportsSectionCheckboxes.appendChild(label);
  }
}

// Called once chart data is ready (see handleOpenFile) and whenever a load
// fails (see showError) — the Reports tab always reflects whether there's
// currently a dataset to report on, regardless of which tab is active.
async function refreshReportsSection() {
  const sections = await window.pywebview.api.get_report_sections();
  if (sections.error) {
    showReportsEmptyState();
    return;
  }
  renderReportSectionCheckboxes(sections);
  reportsResult.classList.add("hidden");
  reportsEmptyState.classList.add("hidden");
  reportsPanel.classList.remove("hidden");
}

function defaultReportFilename() {
  const loadedPath = filePathSpan.textContent || "dataset";
  const base = loadedPath.split(/[\\/]/).pop().replace(/\.[^./\\]+$/, "") || "dataset";
  return `${base}_report.pdf`;
}

generateReportBtn.addEventListener("click", async () => {
  const selectedSections = Array.from(
    reportsSectionCheckboxes.querySelectorAll("input[type=checkbox]:checked")
  ).map((checkbox) => checkbox.value);

  const outputPath = await window.pywebview.api.save_report_dialog(defaultReportFilename());
  if (!outputPath) return;

  generateReportBtn.disabled = true;
  reportsResult.classList.add("hidden");
  try {
    const result = await window.pywebview.api.generate_report(outputPath, selectedSections);
    reportsResult.textContent = result.error ? result.error : `Saved to ${result.path}`;
    reportsResult.classList.remove("hidden");
  } finally {
    generateReportBtn.disabled = false;
  }
});

// --- Open Folder (Stage 2: folder loading) ---
// discover_folder() only ever reads schema + cheap format metadata (see
// folder_loader.py) — this file's job is just presenting the groups it
// found and, when there's more than one, letting the user pick.

// Every file NOT in `groups[selectedIndex]` — other groups' files (schema
// mismatch) plus anything discovery couldn't even read — with a reason.
// Pure client-side recompute over already-fetched data, so switching the
// radio selection in the chooser modal needs no extra backend round trip.
function computeExcludedFiles(discovery, selectedIndex) {
  const excluded = [];
  discovery.groups.forEach((group, index) => {
    if (index === selectedIndex) return;
    const columnNames = group.columns.map((c) => c.name).join(", ");
    const reason = `Different schema than the selected group (${group.columns.length} columns: ${columnNames})`;
    for (const file of group.files) excluded.push({ file, reason });
  });
  return excluded.concat(discovery.unreadable);
}

function describeGroup(group) {
  const fileWord = group.file_count === 1 ? "file" : "files";
  const rowsText = group.row_count != null ? `${group.row_count.toLocaleString()} rows` : "row count unknown";
  const columnNames = group.columns.map((c) => c.name).join(", ");
  return `${group.file_count} ${fileWord}, ${rowsText} — columns: ${columnNames}`;
}

// Renders the group-choice UI into the shared #detail-modal and resolves
// with the chosen group's index once the user clicks "Load selected
// group" — or null if they cancel (X, backdrop click, or Escape), any of
// which must resolve the promise so the caller's await doesn't hang.
function chooseFolderGroup(discovery) {
  return new Promise((resolve) => {
    let resolved = false;
    let selectedIndex = discovery.default_group_index;

    function onEscape(event) {
      if (event.key === "Escape") finish(null);
    }

    function finish(value) {
      if (resolved) return;
      resolved = true;
      document.removeEventListener("keydown", onEscape);
      closeModal();
      resolve(value);
    }

    setModalContent((body) => {
      const heading = document.createElement("h2");
      heading.textContent = "Choose which files to load";
      body.appendChild(heading);

      const note = document.createElement("p");
      note.className = "chart-note";
      note.textContent = `Found ${discovery.groups.length} different schemas in this folder. The largest is selected by default.`;
      body.appendChild(note);

      const choices = document.createElement("div");
      const excludedSection = document.createElement("div");

      function renderExcluded() {
        excludedSection.innerHTML = "";
        const excluded = computeExcludedFiles(discovery, selectedIndex);
        if (!excluded.length) return;
        const excludedHeading = document.createElement("h3");
        excludedHeading.textContent = `Excluded files (${excluded.length})`;
        excludedSection.appendChild(excludedHeading);
        const list = document.createElement("ul");
        list.className = "folder-excluded-list";
        for (const item of excluded) {
          const li = document.createElement("li");
          li.textContent = `${item.file} — ${item.reason}`;
          list.appendChild(li);
        }
        excludedSection.appendChild(list);
      }

      discovery.groups.forEach((group, index) => {
        const label = document.createElement("label");
        label.className = "folder-group-option";
        const radio = document.createElement("input");
        radio.type = "radio";
        radio.name = "folder-group-choice";
        radio.value = String(index);
        radio.checked = index === selectedIndex;
        radio.addEventListener("change", () => {
          selectedIndex = index;
          renderExcluded();
        });
        const text = document.createElement("span");
        text.textContent = describeGroup(group);
        label.appendChild(radio);
        label.appendChild(text);
        choices.appendChild(label);
      });
      body.appendChild(choices);
      body.appendChild(excludedSection);
      renderExcluded();

      const actions = document.createElement("div");
      actions.className = "folder-modal-actions";
      const loadBtn = document.createElement("button");
      loadBtn.className = "btn-primary";
      loadBtn.textContent = "Load selected group";
      loadBtn.addEventListener("click", () => finish(selectedIndex));
      actions.appendChild(loadBtn);
      body.appendChild(actions);
    });

    modalCloseBtn.addEventListener("click", () => finish(null), { once: true });
    detailModal.querySelector(".modal-backdrop").addEventListener("click", () => finish(null), { once: true });
    document.addEventListener("keydown", onEscape);
    showModal();
  });
}

async function handleOpenFolder() {
  const path = await window.pywebview.api.open_folder_dialog();
  if (!path) return;

  clearError();
  showLoading("Scanning folder…");
  let discovery;
  try {
    discovery = await window.pywebview.api.discover_folder(path);
  } finally {
    hideLoading();
  }
  if (discovery.error) {
    showError(discovery.error);
    return;
  }

  let groupIndex = discovery.default_group_index;
  if (discovery.groups.length > 1) {
    groupIndex = await chooseFolderGroup(discovery);
    if (groupIndex === null) return; // cancelled — nothing loaded, leave the app as it was
  }

  const started = await window.pywebview.api.start_folder_load(groupIndex);
  if (started.error) {
    showError(started.error);
    return;
  }

  showLoading("Loading folder…");
  showProgress();
  try {
    const state = await pollFolderLoad();
    if (state.error) {
      showError(state.error);
      return;
    }
    if (!state.result) {
      // Stopped before a single file finished — nothing coherent to show,
      // so leave the app exactly as it was rather than half-loading it.
      return;
    }
    await finishLoadingDataset(state.result);
    renderPartialBanner(state.result);
  } finally {
    hideProgress();
    hideLoading();
  }
}

// Poll cadence: fast enough to feel live, slow enough not to spam the
// js_api bridge across a ~108s load (~500 polls at 200ms).
const LOAD_POLL_MS = 200;

function pollFolderLoad() {
  return new Promise((resolve) => {
    const tick = async () => {
      let state;
      try {
        state = await window.pywebview.api.get_folder_load_progress();
      } catch (err) {
        resolve({ error: String(err) });
        return;
      }
      updateProgress(state);
      if (state.finished) {
        resolve(state);
        return;
      }
      setTimeout(tick, LOAD_POLL_MS);
    };
    tick();
  });
}

function pollChartBuild() {
  return new Promise((resolve) => {
    const tick = async () => {
      let state;
      try {
        state = await window.pywebview.api.get_chart_build_progress();
      } catch (err) {
        resolve({ error: String(err) });
        return;
      }
      updateProgress(state);
      if (state.finished) {
        resolve(state);
        return;
      }
      setTimeout(tick, LOAD_POLL_MS);
    };
    tick();
  });
}

// `stoppable` is false for chart generation: a half-built chart payload
// has no coherent meaning (some columns charted, others silently absent),
// unlike a truncated folder load, which is still an honest dataset. So the
// bar is shown without a Stop button rather than offering one that would
// leave the UI in an unexplainable state.
function showProgress({ stoppable = true } = {}) {
  loadProgress.classList.remove("hidden");
  stopLoadBtn.classList.toggle("hidden", !stoppable);
  stopLoadBtn.disabled = false;
  stopLoadBtn.textContent = "Stop";
  updateProgress({ fraction: 0, label: "Starting…" });
}

function hideProgress() {
  loadProgress.classList.add("hidden");
}

function updateProgress(state) {
  const pct = Math.round((state.fraction || 0) * 100);
  loadProgressBar.value = pct;
  loadProgressPct.textContent = `${pct}%`;
  if (state.label) loadingMessage.textContent = state.label;
}

async function handleStopLoad() {
  // Cancellation lands at the next file/column boundary, so the click is
  // acknowledged immediately even though work continues briefly.
  stopLoadBtn.disabled = true;
  stopLoadBtn.textContent = "Stopping…";
  await window.pywebview.api.cancel_folder_load();
}

function renderPartialBanner(result) {
  if (!result.partial) {
    partialBanner.classList.add("hidden");
    return;
  }
  const used = result.files_used;
  const total = result.files_total;
  partialBanner.textContent =
    `Stopped early — showing ${used.toLocaleString()} of ${total.toLocaleString()} files ` +
    `(${(result.row_count || 0).toLocaleString()} rows). Every figure below describes ` +
    `only those files. Re-open the folder to load all of it.`;
  partialBanner.classList.remove("hidden");
}

openFolderBtn.addEventListener("click", handleOpenFolder);
stopLoadBtn.addEventListener("click", handleStopLoad);
