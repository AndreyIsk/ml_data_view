// Row search UI.
//
// Kept out of app.js, which is already ~1250 lines: this is a
// self-contained section with its own state (the condition rows) and only
// borrows showError/clearError/LOAD_POLL_MS from app.js, which loads
// first. Conditions are built here as plain dicts and validated entirely
// in search.py, so the frontend never has to know polars dtypes — an
// invalid value comes back with its condition index, which is what lets
// the message land on the row that caused it rather than on the search.

const OPERATORS_BY_KIND = {
  numeric: ["is", "is not", ">", ">=", "<", "<=", "between", "is null", "is not null"],
  string: ["is", "is not", "contains", "is null", "is not null"],
  boolean: ["is", "is null", "is not null"],
  other: ["is", "is not", "is null", "is not null"],
};
const NO_VALUE_OPERATORS = new Set(["is null", "is not null"]);
const TWO_VALUE_OPERATORS = new Set(["between"]);
const NUMERIC_DTYPES = new Set([
  "Int8", "Int16", "Int32", "Int64", "Int128",
  "UInt8", "UInt16", "UInt32", "UInt64", "Float32", "Float64",
]);

let searchSchema = []; // [{name, dtype}] for the currently loaded dataset

function dtypeKind(dtype) {
  if (NUMERIC_DTYPES.has(dtype)) return "numeric";
  if (dtype === "String" || dtype === "Utf8") return "string";
  if (dtype === "Boolean") return "boolean";
  return "other";
}

function conditionRows() {
  return Array.from(document.querySelectorAll("#search-conditions .search-condition"));
}

function readConditions() {
  return conditionRows().map((row) => ({
    column: row.querySelector(".cond-column").value,
    operator: row.querySelector(".cond-operator").value,
    values: Array.from(row.querySelectorAll(".cond-value"))
      .filter((input) => !input.classList.contains("hidden"))
      .map((input) => input.value),
  }));
}

// Operators differ by how many values they need, so the inputs follow the
// operator rather than always showing two boxes the user must ignore.
function syncValueInputs(row) {
  const operator = row.querySelector(".cond-operator").value;
  const wanted = NO_VALUE_OPERATORS.has(operator)
    ? 0
    : TWO_VALUE_OPERATORS.has(operator)
    ? 2
    : 1;
  row.querySelectorAll(".cond-value").forEach((input, i) => {
    input.classList.toggle("hidden", i >= wanted);
  });
}

// Only offer operators that can apply to the column's type — "contains" on
// a number is rejected by the backend anyway, so never show it.
function syncOperators(row) {
  const column = row.querySelector(".cond-column").value;
  const entry = searchSchema.find((c) => c.name === column);
  const operators = OPERATORS_BY_KIND[dtypeKind(entry ? entry.dtype : "")];
  const select = row.querySelector(".cond-operator");
  const previous = select.value;
  select.innerHTML = operators.map((op) => `<option value="${op}">${op}</option>`).join("");
  if (operators.includes(previous)) select.value = previous;
  syncValueInputs(row);
}

function addConditionRow(preset) {
  const row = document.createElement("div");
  row.className = "search-condition";

  const columnSelect = document.createElement("select");
  columnSelect.className = "cond-column";
  columnSelect.innerHTML = searchSchema
    .map((c) => `<option value="${c.name}">${c.name}</option>`)
    .join("");

  const operatorSelect = document.createElement("select");
  operatorSelect.className = "cond-operator";

  const value1 = document.createElement("input");
  value1.type = "text";
  value1.className = "cond-value";
  value1.placeholder = "value";
  const value2 = value1.cloneNode();
  value2.placeholder = "and";

  const remove = document.createElement("button");
  remove.type = "button";
  remove.className = "btn-remove";
  remove.textContent = "×";
  remove.title = "Remove this condition";
  remove.addEventListener("click", () => row.remove());

  const error = document.createElement("span");
  error.className = "cond-error hidden";

  row.append(columnSelect, operatorSelect, value1, value2, remove, error);
  document.getElementById("search-conditions").appendChild(row);

  columnSelect.addEventListener("change", () => syncOperators(row));
  operatorSelect.addEventListener("change", () => syncValueInputs(row));

  if (preset) {
    // A saved search may name a column this dataset does not have (a
    // folder can be reloaded with a different schema group). Disable that
    // row with an explanation instead of failing the whole search.
    const known = searchSchema.some((c) => c.name === preset.column);
    if (!known) {
      row.classList.add("has-error");
      error.textContent = `Column "${preset.column}" is not in this dataset.`;
      error.classList.remove("hidden");
      columnSelect.disabled = true;
      operatorSelect.disabled = true;
      return row;
    }
    columnSelect.value = preset.column;
    syncOperators(row);
    operatorSelect.value = preset.operator;
    syncValueInputs(row);
    (preset.values || []).forEach((value, i) => {
      const input = row.querySelectorAll(".cond-value")[i];
      if (input) input.value = value;
    });
  } else {
    syncOperators(row);
  }
  return row;
}

function clearConditionErrors() {
  conditionRows().forEach((row) => {
    const error = row.querySelector(".cond-error");
    error.textContent = "";
    error.classList.add("hidden");
    row.classList.remove("has-error");
  });
}

function markConditionError(index, message) {
  const row = conditionRows()[index];
  if (!row) {
    showError(message);
    return;
  }
  const error = row.querySelector(".cond-error");
  error.textContent = message;
  error.classList.remove("hidden");
  row.classList.add("has-error");
}

function renderSearchColumns() {
  const container = document.getElementById("search-columns");
  container.innerHTML = "";
  for (const column of searchSchema) {
    const label = document.createElement("label");
    const box = document.createElement("input");
    box.type = "checkbox";
    box.value = column.name;
    box.checked = true;
    label.append(box, document.createTextNode(` ${column.name}`));
    container.appendChild(label);
  }
}

function selectedSearchColumns() {
  const boxes = Array.from(document.querySelectorAll("#search-columns input:checked"));
  // Everything selected means "no projection" — send null rather than
  // listing all 40 columns.
  return boxes.length === searchSchema.length ? null : boxes.map((b) => b.value);
}

function renderSearchResults(result) {
  const head = document.getElementById("search-results-head");
  const body = document.getElementById("search-results-body");
  head.innerHTML = `<tr>${result.columns.map((c) => `<th>${c}</th>`).join("")}</tr>`;
  body.innerHTML = "";
  for (const row of result.rows) {
    const tr = document.createElement("tr");
    for (const column of result.columns) {
      const td = document.createElement("td");
      const value = row[column];
      td.textContent = value === null || value === undefined ? "" : String(value);
      tr.appendChild(td);
    }
    body.appendChild(tr);
  }
  document.getElementById("search-results-panel").classList.remove("hidden");
}

async function handleRunSearch() {
  clearError();
  clearConditionErrors();
  const result = await window.pywebview.api.run_search(readConditions(), selectedSearchColumns());
  if (result.error) {
    if (typeof result.condition_index === "number") {
      markConditionError(result.condition_index, result.error);
    } else {
      showError(result.error);
    }
    return;
  }
  renderSearchResults(result);

  const countLine = document.getElementById("search-count-line");
  const shown = result.rows.length.toLocaleString();
  if (!result.truncated) {
    // Fewer rows than the limit means we already saw them all.
    countLine.textContent = `${shown} matches`;
    return;
  }

  // Rows are on screen immediately. The exact total needs a full scan when
  // filtering on a string column (Spark-written files carry no statistics
  // for those, so no row group can be skipped), so it arrives separately
  // rather than blocking the results.
  countLine.textContent = `${shown} shown · counting…`;
  await window.pywebview.api.start_match_count();
  const poll = async () => {
    const state = await window.pywebview.api.get_match_count_progress();
    if (!state.finished) {
      const pct = Math.round((state.fraction || 0) * 100);
      countLine.textContent = `${shown} shown · counting… ${pct}%`;
      setTimeout(poll, LOAD_POLL_MS);
      return;
    }
    if (state.cancelled || state.error) {
      // Never show a partial count as if it were the answer.
      countLine.textContent = `${shown} shown · total not counted`;
      return;
    }
    countLine.textContent = `${shown} shown of ${state.result.toLocaleString()} matches`;
  };
  poll();
}

async function handleExportSearch() {
  const path = await window.pywebview.api.search_export_dialog();
  if (!path) return;
  const countLine = document.getElementById("search-count-line");
  const started = await window.pywebview.api.start_search_export(path);
  if (started.error) {
    showError(started.error);
    return;
  }
  const poll = async () => {
    const state = await window.pywebview.api.get_search_export_progress();
    if (!state.finished) {
      countLine.textContent = state.label;
      setTimeout(poll, LOAD_POLL_MS);
      return;
    }
    if (state.error) showError(state.error);
    // The job's own label says "incomplete" when cancelled — handing
    // someone a truncated CSV that looks whole is the failure to avoid.
    countLine.textContent = state.label;
  };
  poll();
}

async function renderSavedSearches() {
  const searches = await window.pywebview.api.list_saved_searches();
  const list = document.getElementById("saved-search-list");
  list.innerHTML = "";
  for (const saved of searches) {
    const chip = document.createElement("span");
    chip.className = "saved-search-chip";

    const run = document.createElement("button");
    run.type = "button";
    run.className = "saved-search-run";
    run.textContent = saved.name;
    run.addEventListener("click", () => {
      document.getElementById("search-conditions").innerHTML = "";
      (saved.conditions || []).forEach((condition) => addConditionRow(condition));
      handleRunSearch();
    });

    const remove = document.createElement("button");
    remove.type = "button";
    remove.className = "btn-remove";
    remove.textContent = "×";
    remove.title = `Delete "${saved.name}"`;
    remove.addEventListener("click", async () => {
      await window.pywebview.api.delete_saved_search(saved.name);
      renderSavedSearches();
    });

    chip.append(run, remove);
    list.appendChild(chip);
  }
}

async function handleSaveSearch() {
  const nameInput = document.getElementById("saved-search-name");
  const result = await window.pywebview.api.save_search(
    nameInput.value,
    readConditions(),
    selectedSearchColumns()
  );
  if (result.error) {
    showError(result.error);
    return;
  }
  nameInput.value = "";
  renderSavedSearches();
}

// Called by app.js's renderDataset(), so the builder always offers the
// columns of the dataset actually loaded — including after a folder load
// that picked a different schema group.
function resetSearchForDataset(columns) {
  searchSchema = columns.map((c) => ({ name: c.name, dtype: c.dtype }));
  document.getElementById("search-conditions").innerHTML = "";
  document.getElementById("search-results-panel").classList.add("hidden");
  document.getElementById("search-count-line").textContent = "";
  renderSearchColumns();
  addConditionRow();
  renderSavedSearches();
}

document.getElementById("add-condition-btn").addEventListener("click", () => addConditionRow());
document.getElementById("run-search-btn").addEventListener("click", handleRunSearch);
document.getElementById("export-search-btn").addEventListener("click", handleExportSearch);
document.getElementById("save-search-btn").addEventListener("click", handleSaveSearch);
