// The catalogue and the group comparison.
//
// Both payloads arrive fully computed (catalog.py, fleet.py) — this file
// renders them and nothing else. No threshold, no severity rule and no
// "which gaps count" decision lives here, for the same reason none of the
// Health tab's do: a second copy of those rules in JS is how a chart and
// the finding beside it end up telling different stories.
//
// NOTHING HERE NAMES A LEVEL. Every heading, column label and button
// caption comes from the mapping the user confirmed, with generic
// defaults ("File", "Group") when there is no mapping. "System" and
// "Recording" appear nowhere in this file — they are two of the words
// somebody's mapping might supply.

const catalogEmpty = document.getElementById("catalog-empty");
const catalogContent = document.getElementById("catalog-content");
const catalogTable = document.getElementById("catalog-table");
const catalogGroupsPanel = document.getElementById("catalog-groups-panel");
const catalogGroupsTable = document.getElementById("catalog-groups-table");
const catalogFilter = document.getElementById("catalog-filter");
const catalogGroupBy = document.getElementById("catalog-group-by");
const catalogUnhealthyOnly = document.getElementById("catalog-unhealthy-only");
const catalogRowCount = document.getElementById("catalog-row-count");

let catalogPayload = null;
let catalogSort = { key: null, descending: false };
const selectedUnits = new Set();
const selectedGroups = new Set();

const HEALTH_ORDER = ["critical", "warning", "info", "ok", "unknown"];

function plural(word) {
  const lower = (word || "").toLowerCase();
  if (!lower) return "";
  if (/(s|x|z|ch|sh)$/.test(lower)) return word;
  if (/[^aeiou]y$/.test(lower)) return `${word.slice(0, -1)}ies`;
  return `${word}s`;
}

function formatCount(value) {
  return value === null || value === undefined ? "—" : Number(value).toLocaleString();
}

function formatTime(value) {
  if (value === null || value === undefined) return "—";
  // Timestamps reach here as seconds (timeseries.py normalises every unit
  // to them). Anything in the plausible epoch range is shown as a date,
  // because that is what it is; anything else is a bare offset and
  // pretending otherwise would invent a calendar the data never had.
  if (value > 1e8 && value < 4e9) {
    return new Date(value * 1000).toISOString().replace("T", " ").slice(0, 19);
  }
  return Number(value).toFixed(3);
}

function formatCell(row, column) {
  const value = row[column.key];
  switch (column.kind) {
    case "int":
      return formatCount(value);
    case "time":
      return formatTime(value);
    case "duration":
      return formatSeconds(value);
    case "rate":
      return value === null || value === undefined ? "—" : `${Number(value).toFixed(1)} Hz`;
    case "percent":
      return value === null || value === undefined ? "—" : `${(value * 100).toFixed(1)}%`;
    case "bytes":
      return value === null || value === undefined ? "—" : formatBytes(value);
    default:
      return value === null || value === undefined ? "—" : String(value);
  }
}

function healthBadge(health) {
  const status = (health && health.status) || "unknown";
  const count = (health && health.count) || 0;
  const badge = document.createElement("span");
  badge.className = `severity-badge health-${status}`;
  badge.textContent = count ? `${status} (${count})` : status;
  return badge;
}

function checkboxCell(set, key, onChange) {
  const cell = document.createElement("td");
  cell.className = "catalog-select";
  const box = document.createElement("input");
  box.type = "checkbox";
  box.checked = set.has(key);
  box.addEventListener("click", (event) => event.stopPropagation());
  box.addEventListener("change", () => {
    if (box.checked) set.add(key);
    else set.delete(key);
    onChange();
  });
  cell.appendChild(box);
  return cell;
}

// --- sorting and filtering ------------------------------------------------

function comparableValue(row, key) {
  if (key === "health") return HEALTH_ORDER.indexOf((row.health || {}).status || "unknown");
  const value = row[key];
  return value === null || value === undefined ? -Infinity : value;
}

function visibleRows() {
  const needle = (catalogFilter.value || "").trim().toLowerCase();
  let rows = catalogPayload.rows.filter((row) => {
    if (catalogUnhealthyOnly.checked && ((row.health || {}).status || "unknown") === "ok") {
      return false;
    }
    if (!needle) return true;
    return [row.label, ...(row.group_values || [])]
      .join(" ")
      .toLowerCase()
      .includes(needle);
  });
  const chosen = catalogGroupBy.value;
  if (chosen) rows = rows.filter((row) => (row.group_values || [])[0] === chosen);
  if (catalogSort.key) {
    const key = catalogSort.key;
    rows = [...rows].sort((a, b) => {
      const left = comparableValue(a, key);
      const right = comparableValue(b, key);
      const order = typeof left === "string" || typeof right === "string"
        ? String(left).localeCompare(String(right))
        : left - right;
      return catalogSort.descending ? -order : order;
    });
  }
  return rows;
}

// --- the tables -----------------------------------------------------------

function renderCatalogHead() {
  const head = catalogTable.querySelector("thead");
  const row = document.createElement("tr");
  row.appendChild(document.createElement("th"));
  for (const label of catalogPayload.group_labels || []) {
    const cell = document.createElement("th");
    cell.textContent = label;
    row.appendChild(cell);
  }
  for (const column of catalogPayload.columns) {
    const cell = document.createElement("th");
    cell.textContent = column.label;
    cell.className = "catalog-sortable";
    if (catalogSort.key === column.key) {
      cell.textContent = `${column.label} ${catalogSort.descending ? "▾" : "▴"}`;
    }
    cell.addEventListener("click", () => {
      catalogSort =
        catalogSort.key === column.key
          ? { key: column.key, descending: !catalogSort.descending }
          : { key: column.key, descending: false };
      renderCatalogTable();
    });
    row.appendChild(cell);
  }
  head.replaceChildren(row);
}

function renderCatalogTable() {
  renderCatalogHead();
  const body = catalogTable.querySelector("tbody");
  const rows = visibleRows();
  body.replaceChildren(
    ...rows.map((row) => {
      const tr = document.createElement("tr");
      tr.className = "catalog-row";
      tr.appendChild(checkboxCell(selectedUnits, row.label, refreshCompareButtons));
      for (const value of row.group_values || []) {
        const cell = document.createElement("td");
        cell.textContent = value;
        tr.appendChild(cell);
      }
      for (const column of catalogPayload.columns) {
        const cell = document.createElement("td");
        if (column.kind === "health") cell.appendChild(healthBadge(row.health));
        else cell.textContent = formatCell(row, column);
        if (column.kind !== "text" && column.kind !== "health") cell.className = "numeric";
        tr.appendChild(cell);
      }
      // A row is a scope, so clicking it loads that scope — the same
      // operation the breadcrumb performs, reached from the table that
      // made you want it.
      tr.addEventListener("click", () => changeScope(row.group_values || [], row.unit_id));
      return tr;
    })
  );
  catalogRowCount.textContent =
    rows.length === catalogPayload.rows.length
      ? `${rows.length.toLocaleString()} row(s)`
      : `${rows.length.toLocaleString()} of ${catalogPayload.rows.length.toLocaleString()} row(s)`;
  refreshCompareButtons();
}

function renderCatalogGroups() {
  const groups = catalogPayload.groups || [];
  if (!groups.length) {
    catalogGroupsPanel.classList.add("hidden");
    return;
  }
  catalogGroupsPanel.classList.remove("hidden");
  const label = (catalogPayload.group_labels || ["Group"])[0];
  document.getElementById("catalog-groups-heading").textContent = `By ${label.toLowerCase()}`;

  const head = catalogGroupsTable.querySelector("thead");
  const headings = ["", label, plural(catalogPayload.unit_label), "Chunks", "Rows"];
  if (catalogPayload.has_time) headings.push("Duration");
  headings.push("Size", "Health");
  head.replaceChildren(
    (() => {
      const row = document.createElement("tr");
      for (const text of headings) {
        const cell = document.createElement("th");
        cell.textContent = text;
        row.appendChild(cell);
      }
      return row;
    })()
  );

  const body = catalogGroupsTable.querySelector("tbody");
  body.replaceChildren(
    ...groups.map((group) => {
      const tr = document.createElement("tr");
      tr.appendChild(checkboxCell(selectedGroups, group.value, refreshCompareButtons));
      const cells = [
        group.value,
        formatCount(group.unit_count),
        formatCount(group.chunk_count),
        formatCount(group.rows),
      ];
      if (catalogPayload.has_time) cells.push(formatSeconds(group.duration));
      cells.push(group.bytes === null ? "—" : formatBytes(group.bytes));
      for (const value of cells) {
        const cell = document.createElement("td");
        cell.textContent = value;
        tr.appendChild(cell);
      }
      const healthCell = document.createElement("td");
      healthCell.className = "catalog-health-breakdown";
      for (const status of HEALTH_ORDER) {
        const count = (group.health || {})[status] || 0;
        if (!count) continue;
        const chip = document.createElement("span");
        chip.className = `severity-badge health-${status}`;
        chip.textContent = `${count} ${status}`;
        healthCell.appendChild(chip);
      }
      tr.appendChild(healthCell);
      tr.addEventListener("click", () => changeScope([group.value], null));
      tr.className = "catalog-row";
      return tr;
    })
  );
}

function renderCatalogTotals() {
  const totals = catalogPayload.totals || {};
  const stats = [
    healthStat(plural(catalogPayload.unit_label), formatCount(totals.unit_count)),
    healthStat("Chunks", formatCount(totals.chunk_count)),
    healthStat("Rows", formatCount(totals.rows)),
  ];
  if (catalogPayload.has_time) {
    stats.push(healthStat("Total duration", formatSeconds(totals.duration)));
  }
  stats.push(
    healthStat("On disk", totals.bytes === null ? "—" : formatBytes(totals.bytes))
  );
  const breakdown = catalogPayload.health || {};
  for (const status of ["critical", "warning"]) {
    if (breakdown[status]) {
      stats.push(healthStat(plural(status), breakdown[status], true));
    }
  }
  document.getElementById("catalog-totals").replaceChildren(...stats);
}

function refreshCompareButtons() {
  const units = document.getElementById("catalog-compare-units-btn");
  const groups = document.getElementById("catalog-compare-groups-btn");
  units.disabled = selectedUnits.size !== 2;
  groups.disabled = selectedGroups.size !== 2;
  units.textContent =
    selectedUnits.size === 2
      ? `Compare ${[...selectedUnits].join(" with ")}`
      : `Compare two ${plural(catalogPayload ? catalogPayload.unit_label : "unit").toLowerCase()}`;
  groups.textContent =
    selectedGroups.size === 2
      ? `Compare ${[...selectedGroups].join(" with ")}`
      : "Compare the two selected";
}

function unitScope(label) {
  const row = catalogPayload.rows.find((entry) => entry.label === label);
  return row ? { values: row.group_values || [], unit_id: row.unit_id } : null;
}

async function renderCatalog() {
  const payload = await window.pywebview.api.get_catalog();
  catalogPayload = payload && payload.available ? payload : null;
  const unitLabel = payload ? payload.unit_label || "File" : "File";
  document.getElementById("nav-catalog").textContent = plural(unitLabel);
  document.getElementById("catalog-heading").textContent = plural(unitLabel);
  document.getElementById("catalog-table-heading").textContent = plural(unitLabel);

  if (!catalogPayload) {
    catalogContent.classList.add("hidden");
    catalogEmpty.classList.remove("hidden");
    catalogEmpty.textContent = `No catalogue for this dataset — ${
      (payload && payload.reason) || "nothing to catalogue"
    }.`;
    return;
  }
  catalogEmpty.classList.add("hidden");
  catalogContent.classList.remove("hidden");
  selectedUnits.clear();
  selectedGroups.clear();

  catalogGroupBy.replaceChildren();
  const all = document.createElement("option");
  all.value = "";
  all.textContent = catalogPayload.group_labels.length
    ? `All ${catalogPayload.group_labels[0].toLowerCase()}s`
    : "Everything";
  catalogGroupBy.appendChild(all);
  for (const group of catalogPayload.groups || []) {
    const option = document.createElement("option");
    option.value = group.value;
    option.textContent = group.value;
    catalogGroupBy.appendChild(option);
  }
  catalogGroupBy.disabled = !(catalogPayload.groups || []).length;

  renderCatalogTotals();
  renderCatalogGroups();
  renderCatalogTable();
}

catalogFilter.addEventListener("input", () => renderCatalogTable());
catalogUnhealthyOnly.addEventListener("change", () => renderCatalogTable());
catalogGroupBy.addEventListener("change", () => renderCatalogTable());

document.getElementById("catalog-export-btn").addEventListener("click", async () => {
  const result = document.getElementById("catalog-export-result");
  const path = await window.pywebview.api.catalog_export_dialog();
  if (!path) return;
  const written = await window.pywebview.api.export_catalog(path);
  if (written.error) {
    showError(written.error);
    return;
  }
  result.textContent = `Wrote ${written.rows.toLocaleString()} row(s) to ${written.path}`;
});

document.getElementById("catalog-compare-units-btn").addEventListener("click", () => {
  const [first, second] = [...selectedUnits];
  const a = unitScope(first);
  const b = unitScope(second);
  if (a && b) startScopeComparison(a, b);
});

document.getElementById("catalog-compare-groups-btn").addEventListener("click", () => {
  const [first, second] = [...selectedGroups];
  startScopeComparison({ values: [first], unit_id: null }, { values: [second], unit_id: null });
});

// --- the group comparison -------------------------------------------------

const fleetEmpty = document.getElementById("fleet-empty");
const fleetContent = document.getElementById("fleet-content");
const fleetTable = document.getElementById("fleet-table");
const fleetChannelSelect = document.getElementById("fleet-channel");
const fleetBaselineControl = document.getElementById("fleet-baseline-control");
const fleetBaseline = document.getElementById("fleet-baseline");

let fleetPayload = null;

function severityClass(name) {
  return `severity-badge severity-${name || "stable"}`;
}

function renderFleetTable() {
  const previous = fleetBaseline.value === "previous";
  const head = fleetTable.querySelector("thead");
  const headRow = document.createElement("tr");
  for (const text of ["Channel", ...fleetPayload.groups.map((group) => group.value)]) {
    const cell = document.createElement("th");
    cell.textContent = text;
    headRow.appendChild(cell);
  }
  head.replaceChildren(headRow);

  const body = fleetTable.querySelector("tbody");
  body.replaceChildren(
    ...fleetPayload.channels.map((channel) => {
      const tr = document.createElement("tr");
      const name = document.createElement("td");
      name.textContent = channel.column;
      tr.appendChild(name);
      const byValue = new Map(channel.groups.map((entry) => [entry.value, entry]));
      for (const group of fleetPayload.groups) {
        const cell = document.createElement("td");
        const entry = byValue.get(group.value);
        const score = entry ? (previous ? entry.psi_previous : entry.psi) : undefined;
        const level = entry ? (previous ? entry.severity_previous : entry.severity) : undefined;
        if (score === undefined || score === null) {
          cell.textContent = "—";
        } else {
          const badge = document.createElement("span");
          badge.className = severityClass(level);
          badge.textContent = score.toFixed(2);
          cell.appendChild(badge);
        }
        cell.className = "numeric";
        tr.appendChild(cell);
      }
      return tr;
    })
  );
}

function binCentersFromEdges(edges) {
  const centers = [];
  for (let i = 0; i + 1 < edges.length; i += 1) centers.push((edges[i] + edges[i + 1]) / 2);
  return centers;
}

function proportions(counts) {
  const total = counts.reduce((sum, value) => sum + value, 0);
  return total ? counts.map((value) => value / total) : counts.map(() => 0);
}

// One compact distribution per group, the aggregate outlined over it. Drawn
// from the cached bin counts the fleet payload carries — never from rows,
// which is what keeps this bounded however large the dataset is.
async function renderFleetCharts() {
  const container = document.getElementById("fleet-small-multiples");
  container.replaceChildren();
  const channel = (fleetPayload.channels || []).find(
    (entry) => entry.column === fleetChannelSelect.value
  );
  if (!channel || !channel.edges) return;

  const centers = binCentersFromEdges(channel.edges);
  const aggregate = proportions(channel.fleet_counts || []);
  for (const entry of channel.groups) {
    if (!entry.counts) continue;
    const card = document.createElement("div");
    card.className = "fleet-multiple";

    const title = document.createElement("div");
    title.className = "fleet-multiple-title";
    const name = document.createElement("span");
    name.textContent = entry.value;
    const badge = document.createElement("span");
    badge.className = severityClass(entry.severity);
    badge.textContent = entry.psi.toFixed(2);
    title.append(name, badge);

    const plot = document.createElement("div");
    plot.className = "fleet-multiple-plot";
    card.append(title, plot);
    container.appendChild(card);
    observeContainer(card);

    await drawPlot(
      plot,
      [
        {
          type: "bar",
          x: centers,
          y: proportions(entry.counts),
          name: entry.value,
          marker: { color: themeColor("--chart-series") },
          hovertemplate: "%{x:.4g}: %{y:.1%}<extra></extra>",
        },
        {
          type: "scatter",
          mode: "lines",
          x: centers,
          y: aggregate,
          name: "All groups",
          line: { color: themeColor("--chart-axis-text"), width: 1.5, dash: "dot" },
          hovertemplate: "all: %{y:.1%}<extra></extra>",
        },
      ],
      {
        ...barLayout({ l: 40, r: 8, t: 6, b: 28 }),
        showlegend: false,
        height: 150,
        bargap: 0.05,
      },
      { displayModeBar: false, responsive: true }
    );
  }
}

function renderFleetUnits() {
  const panel = document.getElementById("fleet-units-panel");
  const units = fleetPayload.units || {};
  if (!units.available) {
    panel.classList.add("hidden");
    return;
  }
  panel.classList.remove("hidden");
  const label = fleetPayload.group_label || "Group";
  const unitLabel = plural(fleetPayload.unit_label || "Unit");
  panel.querySelector("h2").textContent = `${unitLabel} lengths`;
  document.getElementById("fleet-units-note").textContent =
    `Median ${(fleetPayload.unit_label || "unit").toLowerCase()} length per ` +
    `${label.toLowerCase()}, against the median of the OTHER ${plural(label).toLowerCase()} — ` +
    "so a group cannot dilute the baseline it is being compared with. Reported past " +
    `${Math.round((units.duration_ratio_threshold || 0.25) * 100)}% either way.`;

  const table = document.getElementById("fleet-units-table");
  const headRow = document.createElement("tr");
  for (const text of [label, unitLabel, "Median length", "Others", "Ratio"]) {
    const cell = document.createElement("th");
    cell.textContent = text;
    headRow.appendChild(cell);
  }
  table.querySelector("thead").replaceChildren(headRow);
  table.querySelector("tbody").replaceChildren(
    ...(units.groups || []).map((group) => {
      const tr = document.createElement("tr");
      const ratio = group.duration_ratio;
      const notable =
        group.comparable &&
        ratio !== null &&
        Math.abs(ratio - 1) >= (units.duration_ratio_threshold || 0.25);
      const cells = [
        group.value,
        `${group.unit_count}${group.comparable ? "" : " (too few to judge)"}`,
        formatSeconds(group.median_duration),
        formatSeconds(group.reference_duration),
        ratio === null || ratio === undefined ? "—" : `${(ratio * 100).toFixed(0)}%`,
      ];
      for (const value of cells) {
        const cell = document.createElement("td");
        cell.textContent = value;
        tr.appendChild(cell);
      }
      if (notable) tr.className = "compare-significant";
      return tr;
    })
  );
}

function renderFleetAnomalies() {
  const panel = document.getElementById("fleet-anomalies-panel");
  const anomalies = fleetPayload.unit_anomalies || [];
  if (!anomalies.length) {
    panel.classList.add("hidden");
    return;
  }
  panel.classList.remove("hidden");
  const table = document.getElementById("fleet-anomalies-table");
  const headRow = document.createElement("tr");
  for (const text of [
    fleetPayload.unit_label || "Unit",
    fleetPayload.group_label || "Group",
    "Measure",
    "Value",
    "Group median",
    "Distance",
  ]) {
    const cell = document.createElement("th");
    cell.textContent = text;
    headRow.appendChild(cell);
  }
  table.querySelector("thead").replaceChildren(headRow);
  table.querySelector("tbody").replaceChildren(
    ...anomalies.map((item) => {
      const tr = document.createElement("tr");
      const show = (value) =>
        item.metric === "duration"
          ? formatSeconds(value)
          : item.metric === "gap_share"
            ? `${(value * 100).toFixed(1)}%`
            : `${Number(value).toFixed(2)} Hz`;
      for (const value of [
        item.unit,
        item.group,
        item.metric_label,
        show(item.value),
        show(item.group_median),
        `${Math.abs(item.z).toFixed(0)}σ`,
      ]) {
        const cell = document.createElement("td");
        cell.textContent = value;
        tr.appendChild(cell);
      }
      return tr;
    })
  );
}

async function renderFleet() {
  const payload = await window.pywebview.api.get_fleet();
  fleetPayload = payload && payload.available ? payload : null;
  const label = fleetPayload ? fleetPayload.group_label || "Group" : "Group";
  document.getElementById("nav-groups").textContent = plural(label);
  document.getElementById("fleet-heading").textContent = `${plural(label)}`;

  if (!fleetPayload) {
    fleetContent.classList.add("hidden");
    fleetEmpty.classList.remove("hidden");
    fleetEmpty.textContent =
      `No group comparison for this dataset — ${(payload && payload.reason) || "no grouping column"}. ` +
      "A grouping can be a directory level, a hive partition key, or a column you " +
      "nominate in the Findings tab.";
    return;
  }
  fleetEmpty.classList.add("hidden");
  fleetContent.classList.remove("hidden");

  document.getElementById("fleet-method").textContent = fleetPayload.method_detail;
  document.getElementById("fleet-totals").replaceChildren(
    healthStat(plural(label), fleetPayload.groups.length),
    healthStat("Channels scored", (fleetPayload.channels_scored || []).length),
    healthStat("Channels charted", (fleetPayload.channels_charted || []).length)
  );
  const notCharted =
    (fleetPayload.channels_scored || []).length - (fleetPayload.channels_charted || []).length;
  document.getElementById("fleet-table-note").textContent =
    `Every channel is scored. ${
      notCharted > 0
        ? `${notCharted} of them carry no histogram below — a score is five numbers, a per-group histogram is hundreds — but none of them is missing from this table.`
        : "All of them can also be drawn below."
    }`;
  document.getElementById("fleet-charts-note").textContent =
    `One distribution per ${label.toLowerCase()}, with the aggregate of all ` +
    `${plural(label).toLowerCase()} outlined over it. Drawn from the cached bin counts, ` +
    "never from rows — so this costs the same on a 200-million-row dataset as on a small one.";

  fleetBaselineControl.hidden = !fleetPayload.time_ordered;
  if (!fleetPayload.time_ordered) fleetBaseline.value = "typical";

  fleetChannelSelect.replaceChildren();
  for (const column of fleetPayload.channels_charted || []) {
    const option = document.createElement("option");
    option.value = column;
    option.textContent = column;
    fleetChannelSelect.appendChild(option);
  }
  // The most divergent channel first: the payload is already ranked, so
  // opening the tab lands on the thing worth looking at rather than on
  // whichever column happens to sort first alphabetically.
  const ranked = (fleetPayload.channels || []).find((channel) =>
    (fleetPayload.channels_charted || []).includes(channel.column)
  );
  if (ranked) fleetChannelSelect.value = ranked.column;

  renderFleetTable();
  renderFleetUnits();
  renderFleetAnomalies();
  await renderFleetCharts();
}

fleetChannelSelect.addEventListener("change", () => renderFleetCharts());
fleetBaseline.addEventListener("change", () => renderFleetTable());

async function renderCatalogAndFleet() {
  await renderCatalog();
  await renderFleet();
}

// Chart colors are read back out of the stylesheet AT DRAW TIME, so a
// theme switch has to redraw. app.js re-renders everything in renderCharts;
// the small multiples are not in there — they are drawn after the findings
// scan — so they get their own hook.
async function redrawFleetCharts() {
  if (fleetPayload) await renderFleetCharts();
}
