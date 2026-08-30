import polars as pl

from ml_data_view.compare_jobs import CompareJob


def _write(tmp_path, name, frame, count=2):
    files = []
    for i in range(count):
        path = tmp_path / f"{name}{i}.parquet"
        frame.write_parquet(path)
        files.append(path)
    return files


def _frame(offset=0.0, rows=200):
    return pl.DataFrame(
        {
            "value": [float(n % 40) + offset for n in range(rows)],
            "other": [float(n % 7) for n in range(rows)],
            "kind": [f"k{n % 3}" for n in range(rows)],
        }
    )


def _run(job):
    job.start()
    job.wait(timeout=120)
    return job.progress()


def test_compare_job_completes_with_rows_and_full_progress(tmp_path):
    a = _write(tmp_path, "a", _frame())
    b = _write(tmp_path, "b", _frame(offset=900.0))

    state = _run(CompareJob(a, b))

    assert state["finished"] is True
    assert state["error"] is None
    assert state["cancelled"] is False
    assert state["fraction"] == 1.0
    assert state["result"]["columns"][0]["column"] == "value"  # biggest change first
    assert state["result"]["columns"][0]["severity"] == "significant"


def test_progress_carries_rows_as_columns_complete(tmp_path):
    """The table fills in row by row rather than appearing all at once, so
    partial rows must be visible in progress before the job finishes."""
    a = _write(tmp_path, "a", _frame())
    b = _write(tmp_path, "b", _frame())
    job = CompareJob(a, b)
    row_counts = []
    job._on_progress_for_test = lambda s: row_counts.append(len(s.get("rows") or []))

    _run(job)

    assert max(row_counts) >= 3  # value, other, kind
    assert row_counts == sorted(row_counts), "rows must only ever accumulate"


def test_cancelling_keeps_the_columns_already_compared(tmp_path):
    a = _write(tmp_path, "a", _frame())
    b = _write(tmp_path, "b", _frame())
    job = CompareJob(a, b)
    job._on_progress_for_test = lambda s: job.cancel() if len(s.get("rows") or []) >= 1 else None

    state = _run(job)

    assert state["cancelled"] is True
    assert len(state["result"]["columns"]) >= 1
    assert state["result"]["cancelled"] is True
    assert "compared" in state["label"]


def test_error_is_recorded_not_raised(tmp_path):
    a = _write(tmp_path, "a", _frame())
    missing = [tmp_path / "does_not_exist.parquet"]

    state = _run(CompareJob(a, missing))

    assert state["finished"] is True
    assert state["error"] is not None
