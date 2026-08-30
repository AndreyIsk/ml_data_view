import polars as pl

from ml_data_view.search import Condition, build_predicate
from ml_data_view.search_jobs import SearchCountJob, SearchExportJob


def _write(tmp_path, count=4, rows=20):
    files = []
    for i in range(count):
        path = tmp_path / f"p{i}.parquet"
        pl.DataFrame(
            {"track_id": [f"T-{n % 2}" for n in range(rows)], "height": [float(n) for n in range(rows)]}
        ).write_parquet(path)
        files.append(path)
    return files


def _predicate(files, conditions):
    return build_predicate(conditions, pl.scan_parquet(files[0]).collect_schema())


def _run(job):
    job.start()
    job.wait(timeout=60)
    return job.progress()


def test_count_job_returns_the_total_and_full_progress(tmp_path):
    files = _write(tmp_path, count=4, rows=20)
    job = SearchCountJob(files, _predicate(files, [Condition("track_id", "is", ["T-0"])]))

    state = _run(job)

    assert state["finished"] is True
    assert state["cancelled"] is False
    assert state["error"] is None
    assert state["fraction"] == 1.0
    assert state["result"] == 40  # 10 per file x 4 files


def test_count_job_progress_is_monotonic(tmp_path):
    files = _write(tmp_path, count=5)
    job = SearchCountJob(files, _predicate(files, []))
    seen = []
    job._on_progress_for_test = lambda s: seen.append(s["fraction"])

    _run(job)

    assert seen == sorted(seen)
    assert all(0.0 <= f <= 1.0 for f in seen)


def test_cancelled_count_is_flagged_rather_than_reporting_a_low_number(tmp_path):
    """A partial count that looks like an answer is worse than no answer,
    so `cancelled` must be set for the UI to show 'not counted'."""
    files = _write(tmp_path, count=6)
    job = SearchCountJob(files, _predicate(files, []))
    job._on_progress_for_test = lambda s: job.cancel() if s["files_done"] >= 2 else None

    state = _run(job)

    assert state["cancelled"] is True
    assert state["fraction"] < 1.0


def test_export_job_writes_matching_rows(tmp_path):
    files = _write(tmp_path, count=3, rows=20)
    out = tmp_path / "out.csv"
    job = SearchExportJob(files, _predicate(files, [Condition("track_id", "is", ["T-1"])]), None, out)

    state = _run(job)

    assert state["finished"] is True
    assert state["error"] is None
    assert state["result"] == str(out)
    written = pl.read_csv(out)
    assert written.height == 30
    assert set(written["track_id"].to_list()) == {"T-1"}


def test_export_job_respects_selected_columns(tmp_path):
    files = _write(tmp_path, count=2)
    out = tmp_path / "cols.csv"

    _run(SearchExportJob(files, _predicate(files, []), ["height"], out))

    assert pl.read_csv(out).columns == ["height"]


def test_cancelled_export_says_the_file_is_incomplete(tmp_path):
    files = _write(tmp_path, count=6)
    out = tmp_path / "partial.csv"
    job = SearchExportJob(files, _predicate(files, []), None, out)
    job._on_progress_for_test = lambda s: job.cancel() if s["files_done"] >= 2 else None

    state = _run(job)

    assert state["cancelled"] is True
    assert "incomplete" in state["label"].lower()
    assert out.exists()  # a valid, partial file


def test_export_job_records_an_error_instead_of_raising(tmp_path):
    files = _write(tmp_path, count=1)
    # A directory path can't be opened as a file.
    job = SearchExportJob(files, _predicate(files, []), None, tmp_path)

    state = _run(job)

    assert state["finished"] is True
    assert state["error"] is not None
