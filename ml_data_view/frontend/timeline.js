// The channel timeline: one strip per channel on a shared time axis.
//
// Everything drawn here arrives already downsampled (timeline.py). A
// bucket carries a count, a min, a max and a mean; RAW ROWS NEVER REACH
// THIS FILE at any zoom, which is what lets a 200-million-row recording
// draw in the same time as a 200-row one.
//
// Zooming is a REFETCH, not a redraw. Plotly could rescale the axis by
// itself, but that would only stretch the buckets already on screen —
// asking Python again is what turns a wider bucket into a narrower one,
// and what makes the backend read the row groups the new window touches
// and nothing else.
//
// No threshold and no severity rule lives here. Which stretches are gaps,
// which gaps are missing chunks, and which extremes count as out of range
// are decided in Python beside the checks that report the same things —
// the rule the Health tab's other charts already follow.

const timelinePanel = document.getElementById("timeline-panel");
const timelinePlot = document.getElementById("timeline-plot");
const timelineChannelSelect = document.getElementById("timeline-channels");
const timelineStatus = document.getElementById("timeline-status");
const timelineNote = document.getElementById("timeline-note");
const timelineMarksPanel = document.getElementById("timeline-marks-panel");
const timelineMarksList = document.getElementById("timeline-marks");
const timelineSelection = document.getElementById("timeline-selection");

// A zoom gesture fires a stream of relayout events; only the last one is
// a question worth asking Python.
const TIMELINE_REFETCH_MS = 220;

let timelinePayload = null;
let timelineUnit = null;
let timelineWindow = null;
let timelineRefetchTimer = null;
let timelineInFlight = false;
let timelinePending = null;

const MARK_COLORS = {
  gap: "--chart-missing-bucket",
  missing: "--chart-missing",
  overlap: "--chart-missing",
  nan: "--chart-missing-bucket",
  flat: "--chart-cat-2",
  out_of_range: "--chart-cat-3",
};

function timelineBucketCount() {
  // Tied to pixel width: a bucket narrower than a pixel is a number
  // nobody can see, and asking for it would be asking for the raw rows
  // this whole design exists to avoid sending.
  const width = timelinePlot.clientWidth || 900;
  return Math.max(120, Math.min(2000, Math.round(width)));
}

function centers(t0, t1) {
  return t0.map((value, index) => (value + t1[index]) / 2);
}

function timelineTraces(channels) {
  const traces = [];
  channels.forEach((channel, index) => {
    const axis = index === 0 ? "y" : `y${index + 1}`;
    const x = centers(channel.t0, channel.t1);
    // Max first, then min with `tonexty`, so the band is the envelope
    // between them. A min/max envelope is the only downsample that cannot
    // hide a one-sample spike, which is what someone opens this view to
    // find.
    traces.push({
      type: "scatter",
      mode: "lines",
      x,
      y: channel.max,
      yaxis: axis,
      name: `${channel.column} max`,
      line: { color: themeColor("--chart-series"), width: 0.5 },
      hoverinfo: "skip",
      showlegend: false,
      connectgaps: false,
    });
    traces.push({
      type: "scatter",
      mode: "lines",
      x,
      y: channel.min,
      yaxis: axis,
      name: `${channel.column} range`,
      line: { color: themeColor("--chart-series"), width: 0.5 },
      fill: "tonexty",
      fillcolor: themeColor("--chart-missing-bucket"),
      hoverinfo: "skip",
      showlegend: false,
      connectgaps: false,
    });
    traces.push({
      type: "scatter",
      mode: "lines",
      x,
      y: channel.mean,
      yaxis: axis,
      name: channel.column,
      line: { color: themeColor("--chart-series"), width: 1.4 },
      hovertemplate: `${channel.column}: %{y:.6g}<extra></extra>`,
      showlegend: false,
      connectgaps: false,
    });
  });
  return traces;
}

function timelineLayout(channels, marks, span) {
  const gap = 0.14; // share of a strip's height left as a gutter
  const height = Math.max(160, channels.length * 130);
  const layout = {
    height,
    margin: { l: 62, r: 12, t: 8, b: 34 },
    paper_bgcolor: "rgba(0,0,0,0)",
    plot_bgcolor: "rgba(0,0,0,0)",
    font: PLOT_FONT,
    showlegend: false,
    hovermode: "x unified",
    dragmode: "zoom",
    // A horizontal box select IS the lasso here: a time window is one
    // dimension, and letting somebody draw a shape in the value axis
    // would offer a selection the backend cannot answer.
    selectdirection: "h",
    xaxis: {
      range: [span[0], span[1]],
      gridcolor: GRIDLINE_COLOR,
      zeroline: false,
      domain: [0, 1],
      // The bottom-most strip owns the shared time axis. With one channel
      // that axis is "y", not "y1".
      anchor: channels.length === 1 ? "y" : `y${channels.length}`,
    },
    shapes: [],
  };
  // The FIRST channel gets the TOP strip, which is the order the picker
  // lists them in and the order somebody reads them in.
  channels.forEach((channel, index) => {
    const slot = channels.length - 1 - index;
    const bottom = slot / channels.length;
    const top = bottom + (1 - gap) / channels.length;
    layout[index === 0 ? "yaxis" : `yaxis${index + 1}`] = {
      domain: [bottom, Math.min(top, 1)],
      title: { text: channel.column, standoff: 6 },
      gridcolor: GRIDLINE_COLOR,
      zeroline: false,
    };
  });
  for (const mark of marks) {
    layout.shapes.push({
      type: "rect",
      xref: "x",
      yref: "paper",
      x0: mark.t0,
      // A zero-width mark (an instant, like a spike) would be invisible,
      // so it is drawn as a hairline rather than as nothing.
      x1: mark.t1 > mark.t0 ? mark.t1 : mark.t0 + (span[1] - span[0]) / 400,
      y0: 0,
      y1: 1,
      fillcolor: themeColor(MARK_COLORS[mark.kind] || "--chart-missing-bucket"),
      opacity: 0.28,
      line: { width: 0 },
      layer: "below",
    });
  }
  return layout;
}

function renderTimelineMarks(marks) {
  if (!marks.length) {
    timelineMarksPanel.classList.add("hidden");
    timelineMarksList.replaceChildren();
    return;
  }
  timelineMarksPanel.classList.remove("hidden");
  timelineMarksList.replaceChildren(
    ...marks.map((mark) => {
      const item = document.createElement("li");
      item.className = "timeline-mark";

      const badge = document.createElement("span");
      badge.className = `severity-badge severity-${mark.severity}`;
      badge.textContent = mark.severity;

      const label = document.createElement("span");
      label.className = "timeline-mark-label";
      label.textContent = mark.channel ? `${mark.channel}: ${mark.label}` : mark.label;

      const detail = document.createElement("span");
      detail.className = "timeline-mark-detail";
      detail.textContent = mark.detail;

      const jump = document.createElement("button");
      jump.type = "button";
      jump.className = "btn-secondary";
      jump.textContent = "Jump to it";
      jump.addEventListener("click", () => {
        // A little padding either side: landing with the problem filling
        // the whole axis shows it without showing what it interrupted.
        const span = Math.max(mark.t1 - mark.t0, 1e-6);
        fetchTimeline({ start: mark.t0 - span, end: mark.t1 + span });
      });

      item.append(badge, label, detail, jump);
      return item;
    })
  );
}

function renderTimelineStatus(payload) {
  const parts = [];
  if (payload.source === "scan") {
    parts.push(
      `Read ${payload.rows_read.toLocaleString()} row(s) from ` +
        `${payload.chunks_read} of ${payload.chunks_total} chunk(s)`
    );
  } else {
    parts.push(`From the cache — ${payload.cached_buckets} bucket(s), nothing read`);
  }
  timelineStatus.textContent = parts.join(" · ");
  timelineNote.textContent =
    payload.note ||
    "Min/max envelope and mean per time bucket, computed chunk by chunk. " +
      "Zoom to re-fetch at finer resolution; drag a box to select a window.";
}

async function drawTimeline(payload) {
  const channels = payload.channels || [];
  if (!channels.length) {
    timelinePlot.replaceChildren();
    return;
  }
  await drawPlot(
    timelinePlot,
    timelineTraces(channels),
    timelineLayout(channels, payload.marks || [], payload.window),
    { displayModeBar: true, responsive: true, displaylogo: false }
  );
  bindTimelineEvents();
}

// Re-bound after every draw, not once. drawPlot() re-initialises the div
// with Plotly.newPlot (see app.js on why a resize is a full redraw), and a
// re-initialised div does not reliably keep its listeners — so they are
// cleared and set again rather than assumed to have survived.
function bindTimelineEvents() {
  for (const event of ["plotly_relayout", "plotly_selected", "plotly_deselect"]) {
    if (typeof timelinePlot.removeAllListeners === "function") {
      timelinePlot.removeAllListeners(event);
    }
  }
  timelinePlot.on("plotly_relayout", (event) => {
    if (event["xaxis.autorange"]) {
      scheduleTimelineFetch({ start: null, end: null });
      return;
    }
    const start = event["xaxis.range[0]"];
    const end = event["xaxis.range[1]"];
    if (start === undefined || end === undefined) return;
    scheduleTimelineFetch({ start, end });
  });
  timelinePlot.on("plotly_selected", (event) => {
    if (!event || !event.range) return;
    showTimelineSelection(event.range.x[0], event.range.x[1]);
  });
  timelinePlot.on("plotly_deselect", () => timelineSelection.classList.add("hidden"));
}

function scheduleTimelineFetch(range) {
  clearTimeout(timelineRefetchTimer);
  timelineRefetchTimer = setTimeout(() => fetchTimeline(range), TIMELINE_REFETCH_MS);
}

async function fetchTimeline(range) {
  // A zoom that arrives while the previous one is still being answered is
  // not dropped: it replaces whatever was queued, and runs next. Dropping
  // it would leave the chart showing a window the user has already zoomed
  // past.
  if (timelineInFlight) {
    timelinePending = range;
    return;
  }
  timelineInFlight = true;
  timelinePending = null;
  timelineStatus.textContent = "Reading…";
  try {
    const selected = Array.from(timelineChannelSelect.selectedOptions).map((o) => o.value);
    const payload = await window.pywebview.api.get_timeline(
      timelineUnit,
      selected.length ? selected : null,
      range && range.start !== undefined ? range.start : null,
      range && range.end !== undefined ? range.end : null,
      timelineBucketCount()
    );
    if (!payload || payload.available === false) {
      timelinePanel.classList.add("hidden");
      return;
    }
    timelinePanel.classList.remove("hidden");
    timelinePayload = payload;
    timelineWindow = payload.window;
    renderTimelineStatus(payload);
    renderTimelineMarks(payload.marks || []);
    await drawTimeline(payload);
  } finally {
    timelineInFlight = false;
    if (timelinePending) fetchTimeline(timelinePending);
  }
}

async function showTimelineSelection(start, end) {
  const stats = await window.pywebview.api.get_timeline_window(start, end);
  if (stats.error) return;
  timelineSelection.classList.remove("hidden");
  document.getElementById("timeline-selection-note").textContent =
    `${formatSeconds(stats.duration)} selected` +
    (stats.exact
      ? "."
      : " — summarised from whole buckets, so the figures describe " +
        `${formatSeconds((stats.covered && stats.covered[1] - stats.covered[0]) || 0)} ` +
        "of data around the selection. Zoom in for an exact one.");
  document.getElementById("timeline-selection-stats").replaceChildren(
    ...stats.channels.flatMap((channel) => [
      healthStat(`${channel.column} rows`, (channel.n || 0).toLocaleString()),
      healthStat(
        `${channel.column} range`,
        channel.min === null
          ? "—"
          : `${channel.min.toPrecision(4)} … ${channel.max.toPrecision(4)}`
      ),
      healthStat(
        `${channel.column} mean`,
        channel.mean === null ? "—" : channel.mean.toPrecision(4)
      ),
    ])
  );
  document.getElementById("timeline-search-btn").onclick = async () => {
    const result = await window.pywebview.api.timeline_window_conditions(start, end);
    if (result.error) {
      showError(result.error);
      return;
    }
    // Straight into the feature that already knows how to count matches
    // and write a CSV — with the query on screen and editable, rather than
    // as a second, invisible export path.
    activateSection("search");
    await applySearchConditions(result.conditions);
  };
}

timelineChannelSelect.addEventListener("change", () =>
  fetchTimeline(timelineWindow ? { start: timelineWindow[0], end: timelineWindow[1] } : {})
);
document
  .getElementById("timeline-reset-btn")
  .addEventListener("click", () => fetchTimeline({ start: null, end: null }));

// Called by app.js whenever the Health tab switches unit — the timeline
// is the same unit's data as everything else on that tab.
async function renderTimeline(unitId) {
  timelineUnit = unitId;
  timelineWindow = null;
  timelineSelection.classList.add("hidden");
  const payload = await window.pywebview.api.get_timeline(unitId, null, null, null, timelineBucketCount());
  if (!payload || payload.available === false) {
    timelinePanel.classList.add("hidden");
    return;
  }
  timelinePanel.classList.remove("hidden");
  timelinePayload = payload;
  timelineWindow = payload.window;

  const previously = new Set(
    Array.from(timelineChannelSelect.selectedOptions).map((option) => option.value)
  );
  timelineChannelSelect.replaceChildren();
  for (const column of payload.channels_available || []) {
    const option = document.createElement("option");
    option.value = column;
    option.textContent = column;
    option.selected = previously.size
      ? previously.has(column)
      : (payload.channels_selected || []).includes(column);
    timelineChannelSelect.appendChild(option);
  }

  renderTimelineStatus(payload);
  renderTimelineMarks(payload.marks || []);
  await drawTimeline(payload);
}

// Chart colors are read out of the stylesheet at draw time, so a theme
// switch redraws. Nothing is re-fetched: the buckets did not change.
async function redrawTimeline() {
  if (timelinePayload) await drawTimeline(timelinePayload);
}
