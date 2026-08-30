import polars as pl

from ml_data_view.folder_load_job import CLASSIFY_WEIGHT, NULL_COUNT_WEIGHT, FolderLoadJob
from ml_data_view.folder_loader import scan_folder


def _write_folder(tmp_path, file_count, rows=30):
    for i in range(file_count):
        pl.DataFrame(
            {
                "id": list(range(rows)),
                "label": [f"c{n % 3}" for n in range(rows)],
                "score": [1.0 if n % 4 else None for n in range(rows)],
            }
        ).write_parquet(tmp_path / f"part{i}.parquet")
    return tmp_path


def _run_to_completion(job):
    job.start()
    job.wait(timeout=60)
    return job.progress()


def test_job_reports_completion_and_full_progress(tmp_path):
    _write_folder(tmp_path, 4)
    group = scan_folder(tmp_path).groups[0]

    state = _run_to_completion(FolderLoadJob(group, tmp_path))

    assert state["finished"] is True
    assert state["cancelled"] is False
    assert state["error"] is None
    assert state["fraction"] == 1.0
    assert state["files_total"] == 4
    assert state["files_done"] == 4
    assert state["result"]["row_count"] == 120  # 4 files x 30 rows
    assert state["result"]["partial"] is False


def test_job_progress_is_monotonic_and_bounded(tmp_path):
    _write_folder(tmp_path, 5)
    group = scan_folder(tmp_path).groups[0]
    job = FolderLoadJob(group, tmp_path)
    seen = []
    job._on_progress_for_test = lambda s: seen.append(s["fraction"])

    _run_to_completion(job)

    assert seen == sorted(seen), "progress must never go backwards"
    assert all(0.0 <= f <= 1.0 for f in seen)


def test_phase_weights_match_measured_cost_split():
    # 77% of a real 250-file load was null counting, 22% classification.
    assert abs((NULL_COUNT_WEIGHT + CLASSIFY_WEIGHT) - 1.0) < 1e-9
    assert NULL_COUNT_WEIGHT > CLASSIFY_WEIGHT


def test_cancel_during_null_counts_truncates_to_finished_files(tmp_path):
    _write_folder(tmp_path, 6)
    group = scan_folder(tmp_path).groups[0]
    job = FolderLoadJob(group, tmp_path)
    # Cancel as soon as two files have been counted.
    job._on_progress_for_test = lambda s: job.cancel() if s["files_done"] >= 2 else None

    state = _run_to_completion(job)

    assert state["cancelled"] is True
    assert state["finished"] is True
    result = state["result"]
    assert result["partial"] is True
    assert result["files_used"] < 6
    assert result["files_total"] == 6
    # Row count and null counts describe ONLY the files actually used, so
    # every number on screen agrees with every other.
    assert result["row_count"] == result["files_used"] * 30
    nulls = {c["name"]: c["null_count"] for c in result["columns"]}
    assert nulls["score"] == result["files_used"] * 8  # 8 nulls per 30-row file


def test_cancelled_load_does_not_report_full_progress(tmp_path):
    """A stopped load must not snap the bar to 100% — that reads as
    'finished successfully' when it is the opposite. Caught on a real
    250-file run that stopped at 9% and reported FINAL 100%."""
    _write_folder(tmp_path, 8)
    group = scan_folder(tmp_path).groups[0]
    job = FolderLoadJob(group, tmp_path)
    job._on_progress_for_test = lambda s: job.cancel() if s["files_done"] >= 2 else None

    state = _run_to_completion(job)

    assert state["cancelled"] is True
    assert state["fraction"] < 1.0
    assert state["fraction"] > 0.0


def test_cancelled_counts_match_a_direct_load_of_the_same_files(tmp_path):
    """The truncated result must be indistinguishable from having opened
    just those files in the first place — otherwise "stopped" silently
    means "wrong" rather than "smaller"."""
    from ml_data_view.data_loader import dataset_info_from_lazy

    _write_folder(tmp_path, 6)
    group = scan_folder(tmp_path).groups[0]
    job = FolderLoadJob(group, tmp_path)
    job._on_progress_for_test = lambda s: job.cancel() if s["files_done"] >= 3 else None

    result = _run_to_completion(job)["result"]
    used = group.files[: result["files_used"]]

    direct = dataset_info_from_lazy(
        pl.concat([pl.scan_parquet(f) for f in used], how="vertical"), tmp_path, files=used
    ).to_dict()

    assert result["row_count"] == direct["row_count"]
    assert {c["name"]: c["null_count"] for c in result["columns"]} == {
        c["name"]: c["null_count"] for c in direct["columns"]
    }


def test_cancel_during_classify_keeps_every_file(tmp_path):
    """Null counts already finished by then, so stopping in the classify
    phase must not throw away files — only leave columns unclassified."""
    _write_folder(tmp_path, 3)
    group = scan_folder(tmp_path).groups[0]
    job = FolderLoadJob(group, tmp_path)
    job._on_progress_for_test = lambda s: job.cancel() if s["stage"] == "classify" else None

    state = _run_to_completion(job)
    result = state["result"]

    assert state["cancelled"] is True
    assert result["files_used"] == 3
    assert result["row_count"] == 90
    assert result["partial"] is True


def test_cancel_before_start_finishes_immediately_without_result(tmp_path):
    _write_folder(tmp_path, 3)
    group = scan_folder(tmp_path).groups[0]
    job = FolderLoadJob(group, tmp_path)
    job.cancel()

    state = _run_to_completion(job)

    assert state["cancelled"] is True
    assert state["finished"] is True
    assert state["result"] is None  # nothing completed, so nothing to show


def test_job_records_error_instead_of_raising_in_the_thread(tmp_path):
    _write_folder(tmp_path, 2)
    group = scan_folder(tmp_path).groups[0]
    for f in group.files:
        f.write_bytes(b"not parquet at all")  # corrupt after discovery

    state = _run_to_completion(FolderLoadJob(group, tmp_path))

    assert state["finished"] is True
    assert state["error"] is not None
    assert state["result"] is None
