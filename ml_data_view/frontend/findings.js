// Findings tab UI.
//
// The tab reads as a scan: every check is painted by name with a pending
// marker BEFORE any work happens (list_checks() is deliberately
// independent of a loaded dataset), then each resolves to a finding or a
// clean tick.
//
// One honest distinction runs through this file. The INSTANT checks take
// about 5ms in total — they read figures the chart build already computed
// — so their stagger below is a reveal animation and nothing more. It is
// capped at REVEAL_STAGGER_MS and results are never withheld beyond it.
// The DEEP checks are the opposite: their pending/running/done comes from
// DeepCheckJob's real per-check status, polled while the worker runs.
// Dressing the first up as the second would be theatre, so it isn't.

const REVEAL_STAGGER_MS = 120;

const SEVERITY_ORDER = ["critical", "warning", "info"];
const SEVERITY_LABELS = { critical: "critical", warning: "warning", info: "info" };

// Results for every check seen so far, keyed by id, so the summary counts
// instant and deep findings together rather than one replacing the other.
let findingsResults = new Map();
let findingsCatalogue = [];
let deepChecksRunning = false;

function findingsElements() {
  return {
    list: document.getElementById("findings-checks"),
    summary: document.getElementById("findings-summary"),
    empty: document.getElementById("findings-empty"),
    deepBtn: document.getElementById("deep-checks-btn"),
    progress: document.getElementById("deep-checks-progress"),
    bar: document.getElementById("deep-checks-bar"),
    pct: document.getElementById("deep-checks-pct"),
    stopBtn: document.getElementById("deep-checks-stop-btn"),
  };
}

function sleep(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

// --- rendering ---------------------------------------------------------

function checkRow(spec) {
  const row = document.createElement("li");
  row.className = "findings-check";
  row.dataset.checkId = spec.id;

  const marker = document.createElement("span");
  marker.className = "findings-marker";
  row.appendChild(marker);

  const body = document.createElement("div");
  body.className = "findings-check-body";

  const title = document.createElement("span");
  title.className = "findings-check-title";
  title.textContent = spec.title;
  body.appendChild(title);

  const status = document.createElement("span");
  status.className = "findings-check-status";
  body.appendChild(status);

  const list = document.createElement("div");
  list.className = "findings-list";
  body.appendChild(list);

  row.appendChild(body);
  return row;
}

function severityBadge(severity) {
  const badge = document.createElement("span");
  // Severity is carried by this text label, not by color alone — the same
  // rule compare.js's badges follow.
  badge.className = `severity-badge severity-${severity}`;
  badge.textContent = SEVERITY_LABELS[severity] || severity;
  return badge;
}

function renderFinding(finding) {
  const item = document.createElement("div");
  item.className = `finding finding-${finding.severity}`;

  const head = document.createElement("div");
  head.className = "finding-head";
  head.appendChild(severityBadge(finding.severity));

  const title = document.createElement("span");
  title.className = "finding-title";
  title.textContent = finding.title;
  head.appendChild(title);
  item.appendChild(head);

  const detail = document.createElement("p");
  detail.className = "finding-detail";
  detail.textContent = finding.detail;
  item.appendChild(detail);

  return item;
}

// `state` is one of: pending, running, done, error, cancelled.
function applyCheckResult(result) {
  const row = document.querySelector(`.findings-check[data-check-id="${result.id}"]`);
  if (!row) return;

  row.dataset.status = result.status;
  const status = row.querySelector(".findings-check-status");
  const list = row.querySelector(".findings-list");
  list.innerHTML = "";

  if (result.status === "pending") {
    status.textContent = "waiting";
    return;
  }
  if (result.status === "running") {
    status.textContent = "checking…";
    return;
  }
  if (result.status === "cancelled") {
    status.textContent = "not run";
    return;
  }
  if (result.status === "error") {
    // A check that could not run is not a check that passed, so it never
    // renders as a tick.
    status.textContent = `could not run — ${result.error || "unknown error"}`;
    return;
  }

  const findings = result.findings || [];
  status.textContent = findings.length ? "" : "clean";
  for (const finding of findings) list.appendChild(renderFinding(finding));
}

function renderSummary() {
  const { summary, empty, list } = findingsElements();
  const all = [];
  let checksDone = 0;
  for (const result of findingsResults.values()) {
    if (result.status === "done") checksDone += 1;
    for (const finding of result.findings || []) all.push(finding);
  }

  if (!checksDone) {
    summary.classList.add("hidden");
    empty.classList.add("hidden");
    return;
  }

  if (!all.length) {
    // The count is the checks ACTUALLY run — 9 after the instant scan, 12
    // once deep checks have also run. A fixed number would misreport the
    // moment the button is used.
    empty.textContent = `All ${checksDone} checks passed — no issues found.`;
    empty.classList.remove("hidden");
    summary.classList.add("hidden");
    list.classList.add("findings-clean");
    return;
  }

  list.classList.remove("findings-clean");
  empty.classList.add("hidden");
  summary.innerHTML = "";
  for (const severity of SEVERITY_ORDER) {
    const count = all.filter((f) => f.severity === severity).length;
    const chip = document.createElement("span");
    chip.className = `findings-count severity-${severity}`;
    chip.textContent = `${count} ${SEVERITY_LABELS[severity]}`;
    summary.appendChild(chip);
  }
  summary.classList.remove("hidden");
}

// --- instant scan ------------------------------------------------------

// Called from app.js's renderDataset(): paints the catalogue as pending so
// the tab is never blank, and drops any previous dataset's results.
async function resetFindingsForDataset() {
  const { list, summary, empty, deepBtn } = findingsElements();
  findingsResults = new Map();
  summary.classList.add("hidden");
  empty.classList.add("hidden");
  list.classList.remove("findings-clean");
  deepBtn.disabled = false;

  if (!findingsCatalogue.length) {
    const response = await window.pywebview.api.list_checks();
    findingsCatalogue = response.checks || [];
  }

  list.innerHTML = "";
  for (const spec of findingsCatalogue) {
    const row = checkRow(spec);
    row.dataset.status = "pending";
    list.appendChild(row);
    applyCheckResult({ id: spec.id, status: "pending", findings: [] });
  }
}

// Called from app.js once the chart build has finished.
async function runFindingsScan() {
  const result = await window.pywebview.api.run_instant_checks();
  if (result.error) {
    setInstantCheckNote(result.error);
    return;
  }
  if (result.pending) {
    // Never a clean state before the checks have run.
    setInstantCheckNote("Waiting for chart data…");
    return;
  }

  for (const check of result.checks || []) {
    applyCheckResult({ ...check, status: "running" });
    // Reveal pacing only — the work is already done. See the file header.
    await sleep(REVEAL_STAGGER_MS);
    findingsResults.set(check.id, check);
    applyCheckResult(check);
    renderSummary();
  }
}

function setInstantCheckNote(message) {
  for (const spec of findingsCatalogue) {
    if (spec.kind !== "instant") continue;
    const row = document.querySelector(`.findings-check[data-check-id="${spec.id}"]`);
    if (!row) continue;
    row.dataset.status = "waiting";
    row.querySelector(".findings-check-status").textContent = message;
  }
}

// --- deep checks -------------------------------------------------------

function updateDeepProgress(state) {
  const { bar, pct } = findingsElements();
  const percent = Math.round((state.fraction || 0) * 100);
  bar.value = percent;
  pct.textContent = `${percent}%`;
}

function pollDeepChecks() {
  return new Promise((resolve) => {
    const tick = async () => {
      let state;
      try {
        state = await window.pywebview.api.get_deep_check_progress();
      } catch (err) {
        resolve({ error: String(err) });
        return;
      }
      updateDeepProgress(state);
      // Per-check status here is REAL, unlike the instant reveal above:
      // each row updates as the worker thread actually finishes it.
      for (const check of state.checks || []) {
        applyCheckResult(check);
        if (check.status === "done") findingsResults.set(check.id, check);
      }
      renderSummary();
      if (state.finished) {
        resolve(state);
        return;
      }
      setTimeout(tick, LOAD_POLL_MS);
    };
    tick();
  });
}

async function handleRunDeepChecks() {
  if (deepChecksRunning) return;
  const { deepBtn, progress, stopBtn } = findingsElements();

  const started = await window.pywebview.api.start_deep_checks();
  if (started.error) {
    showError(started.error);
    return;
  }

  deepChecksRunning = true;
  deepBtn.disabled = true;
  stopBtn.disabled = false;
  stopBtn.textContent = "Stop";
  progress.classList.remove("hidden");
  try {
    const state = await pollDeepChecks();
    if (state.error) showError(state.error);
  } finally {
    deepChecksRunning = false;
    deepBtn.disabled = false;
    progress.classList.add("hidden");
    renderSummary();
  }
}

async function handleStopDeepChecks() {
  const { stopBtn } = findingsElements();
  stopBtn.disabled = true;
  stopBtn.textContent = "Stopping…";
  await window.pywebview.api.cancel_deep_checks();
}

document.getElementById("deep-checks-btn").addEventListener("click", handleRunDeepChecks);
document.getElementById("deep-checks-stop-btn").addEventListener("click", handleStopDeepChecks);
