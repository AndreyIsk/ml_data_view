// Dataset comparison UI.
//
// Two things make this feel responsive rather than a long blank wait:
// the schema diff comes back from start_comparison() itself (footer
// metadata only, so it is instant) and paints before any distribution work
// begins; and the column table fills in row by row, because the job carries
// completed rows in its progress state rather than only in its final result.

let compareSeenColumns = 0;

function comparePanels() {
  return {
    status: document.getElementById("compare-status"),
    progress: document.getElementById("compare-progress"),
    bar: document.getElementById("compare-progress-bar"),
    pct: document.getElementById("compare-progress-pct"),
    schemaPanel: document.getElementById("compare-schema-panel"),
    schemaBody: document.getElementById("compare-schema-body"),
    resultsPanel: document.getElementById("compare-results-panel"),
    resultsBody: document.getElementById("compare-results-body"),
  };
}

function renderSchemaDiff(diff) {
  const { schemaPanel, schemaBody } = comparePanels();
  schemaBody.innerHTML = "";

  const blocks = [
    ["Only in A (loaded dataset)", diff.only_in_a, "removed"],
    ["Only in B (compared dataset)", diff.only_in_b, "added"],
  ];
  let any = false;
  for (const [title, names, kind] of blocks) {
    if (!names || names.length === 0) continue;
    any = true;
    const block = document.createElement("div");
    block.className = "schema-diff-block";
    const heading = document.createElement("span");
    heading.className = "schema-diff-title";
    heading.textContent = `${title}:`;
    block.appendChild(heading);
    for (const name of names) {
      const chip = document.createElement("span");
      chip.className = `schema-chip schema-chip-${kind}`;
      chip.textContent = name;
      block.appendChild(chip);
    }
    schemaBody.appendChild(block);
  }

  if (diff.type_changed && diff.type_changed.length) {
    any = true;
    const block = document.createElement("div");
    block.className = "schema-diff-block";
    const heading = document.createElement("span");
    heading.className = "schema-diff-title";
    heading.textContent = "Type changed:";
    block.appendChild(heading);
    for (const entry of diff.type_changed) {
      const chip = document.createElement("span");
      chip.className = "schema-chip schema-chip-retyped";
      // Retyped columns are excluded from the distribution table on
      // purpose — two different types have no comparable distribution —
      // so naming the types here is the only place that is explained.
      chip.textContent = `${entry.column}: ${entry.dtype_a} → ${entry.dtype_b}`;
      block.appendChild(chip);
    }
    schemaBody.appendChild(block);
  }

  if (!any) {
    const ok = document.createElement("p");
    ok.className = "chart-note";
    ok.textContent = "Both datasets have identical schemas.";
    schemaBody.appendChild(ok);
  }
  schemaPanel.classList.remove("hidden");
}

function formatStat(row, side) {
  if (row.kind === "numeric") {
    const mean = side === "a" ? row.mean_a : row.mean_b;
    return mean === null || mean === undefined ? "—" : `mean ${Number(mean).toFixed(2)}`;
  }
  const extra = side === "b" ? row.new_categories : row.missing_categories;
  if (extra && extra.length) {
    const shown = extra.slice(0, 3).join(", ");
    return extra.length > 3 ? `${shown} +${extra.length - 3} more` : shown;
  }
  return "—";
}

function renderCompareRows(rows) {
  const { resultsBody, resultsPanel } = comparePanels();
  // Only append what is new: the job's progress carries a growing list, and
  // rebuilding the whole table on every poll would fight the user's scroll.
  for (let i = compareSeenColumns; i < rows.length; i += 1) {
    const row = rows[i];
    const tr = document.createElement("tr");
    tr.className = `compare-row compare-${row.severity}`;

    const cells = [
      row.column,
      formatStat(row, "a"),
      formatStat(row, "b"),
      `${(row.null_rate_a * 100).toFixed(1)}%`,
      `${(row.null_rate_b * 100).toFixed(1)}%`,
      row.psi.toFixed(3),
    ];
    for (const value of cells) {
      const td = document.createElement("td");
      td.textContent = value;
      tr.appendChild(td);
    }

    const badgeCell = document.createElement("td");
    const badge = document.createElement("span");
    badge.className = `severity-badge severity-${row.severity}`;
    badge.textContent = row.severity;
    badgeCell.appendChild(badge);
    tr.appendChild(badgeCell);

    resultsBody.appendChild(tr);
  }
  compareSeenColumns = rows.length;
  if (rows.length) resultsPanel.classList.remove("hidden");
}

// Rows arrive in completion order while running, but the final result is
// sorted by PSI. Re-render once at the end so the biggest change is at the
// top, which is the whole point of the table.
function renderCompareFinal(result) {
  const { resultsBody, resultsPanel } = comparePanels();
  resultsBody.innerHTML = "";
  compareSeenColumns = 0;
  renderCompareRows(result.columns || []);
  if (!result.columns || result.columns.length === 0) {
    resultsPanel.classList.add("hidden");
  }
}

async function startComparison(kind) {
  const { status, progress, resultsBody } = comparePanels();
  const path = await window.pywebview.api.compare_picker_dialog(kind);
  if (!path) return;

  clearError();
  resultsBody.innerHTML = "";
  compareSeenColumns = 0;
  document.getElementById("compare-results-panel").classList.add("hidden");

  const compareAll = document.getElementById("compare-all-files").checked;
  // Called with one argument when sampling, so the backend's own default
  // applies. Passing `undefined` would serialize to null across the bridge,
  // and null is what the backend reads as "compare every file" — the exact
  // opposite of what the unchecked box means.
  const started = compareAll
    ? await window.pywebview.api.start_comparison(path, null)
    : await window.pywebview.api.start_comparison(path);
  if (started.error) {
    showError(started.error);
    return;
  }

  renderSchemaDiff(started.schema);
  const sampledNote =
    started.sampled && (started.files_a > 1 || started.files_b > 1)
      ? ` · sampled ${started.files_a} and ${started.files_b} files, spread across each dataset`
      : "";
  status.textContent = `Comparing against ${path}${sampledNote}`;
  progress.classList.remove("hidden");
  document.getElementById("compare-stop-btn").disabled = false;
  document.getElementById("compare-stop-btn").textContent = "Stop";

  const poll = async () => {
    const state = await window.pywebview.api.get_comparison_progress();
    const { bar, pct } = comparePanels();
    bar.value = Math.round((state.fraction || 0) * 100);
    pct.textContent = `${Math.round((state.fraction || 0) * 100)}%`;
    renderCompareRows(state.rows || []);

    if (!state.finished) {
      status.textContent = `${state.label}${sampledNote}`;
      setTimeout(poll, LOAD_POLL_MS);
      return;
    }
    progress.classList.add("hidden");
    if (state.error) {
      showError(state.error);
      return;
    }
    if (state.result) {
      renderCompareFinal(state.result);
      const counts =
        `${state.result.row_count_a.toLocaleString()} rows (A) vs ` +
        `${state.result.row_count_b.toLocaleString()} rows (B)`;
      status.textContent = `${state.label} · ${counts}${sampledNote}`;
    }
  };
  poll();
}

document.getElementById("compare-folder-btn").addEventListener("click", () => startComparison("folder"));
document.getElementById("compare-file-btn").addEventListener("click", () => startComparison("file"));
document.getElementById("compare-stop-btn").addEventListener("click", async () => {
  const btn = document.getElementById("compare-stop-btn");
  btn.disabled = true;
  btn.textContent = "Stopping…";
  await window.pywebview.api.cancel_comparison();
});
