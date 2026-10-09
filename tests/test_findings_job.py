"""DeepCheckJob — per-check status while deep checks run (findings_job.py).

The point of the job, beyond not blocking the js_api bridge, is that the
Findings tab can render each check resolving as it finishes rather than
waiting for all three. So these tests assert on the SEQUENCE of states, not
just the final result.
"""
from __future__ import annotations

import threading

import polars as pl

from ml_data_view.deep_checks import DEEP_CHECKS, DeepContext
from ml_data_view.findings import Finding
from ml_data_view.findings_job import DeepCheckJob


def _context(tmp_path) -> DeepContext:
    path = tmp_path / "data.parquet"
    pl.DataFrame({"a": [1, 2, 3, 1], "b": ["x", "y", "z", "x"]}).write_parquet(path)
    return DeepContext(lf=pl.scan_parquet(path), files=[path])


def test_every_check_starts_pending(tmp_path):
    """Before the job runs, the tab must be able to paint every check name
    with a pending marker — that is what makes it read as a scan."""
    job = DeepCheckJob(_context(tmp_path))

    checks = job.progress()["checks"]
    assert [c["id"] for c in checks] == [spec.id for spec in DEEP_CHECKS]
    assert all(c["status"] == "pending" for c in checks)


CANONICAL_SEQUENCE = ["pending", "running", "done"]


def test_each_check_passes_through_running_to_done(tmp_path):
    """Each check advances along pending -> running -> done and never goes
    backwards.

    A check may not be *observed* in every state: the first check is
    already running by the time the hook below is attached, so its history
    legitimately starts at "running". What must hold is that whatever was
    seen is in canonical order, includes "running", and ends at "done".
    (test_every_check_starts_pending covers the pending state itself.)
    """
    job = DeepCheckJob(_context(tmp_path))
    seen: dict[str, list[str]] = {spec.id: [] for spec in DEEP_CHECKS}

    def record(state):
        for check in state.get("checks", []):
            history = seen[check["id"]]
            if not history or history[-1] != check["status"]:
                history.append(check["status"])

    job._on_progress_for_test = record
    job.start()
    job.wait(timeout=60)
    record(job.progress())

    for check_id, history in seen.items():
        assert history == [s for s in CANONICAL_SEQUENCE if s in history], (
            f"{check_id} went {history}, which is not in pending -> running -> done order"
        )
        assert "running" in history, f"{check_id} never reported running"
        assert history[-1] == "done", f"{check_id} ended at {history[-1]}"


def test_finished_checks_are_visible_before_the_job_completes(tmp_path):
    """A completed check must appear in progress state while later checks
    are still running — otherwise the tab may as well wait for the end."""
    job = DeepCheckJob(_context(tmp_path))
    saw_partial = threading.Event()

    def record(state):
        statuses = [c["status"] for c in state.get("checks", [])]
        if "done" in statuses and ("pending" in statuses or "running" in statuses):
            saw_partial.set()

    job._on_progress_for_test = record
    job.start()
    job.wait(timeout=60)

    assert saw_partial.is_set()


def test_findings_accumulate_into_progress_state(tmp_path):
    job = DeepCheckJob(_context(tmp_path))
    job.start()
    job.wait(timeout=60)

    state = job.progress()
    duplicate = next(c for c in state["checks"] if c["id"] == "duplicate_rows")
    assert duplicate["status"] == "done"
    assert duplicate["findings"], "the fixture has a planted duplicate row"
    assert state["finished"] is True


def test_result_carries_every_check_and_a_severity_summary(tmp_path):
    job = DeepCheckJob(_context(tmp_path))
    job.start()
    job.wait(timeout=60)

    result = job.progress()["result"]
    assert [c["id"] for c in result["checks"]] == [spec.id for spec in DEEP_CHECKS]
    assert set(result["summary"]) == {"critical", "warning", "info"}
    assert result["summary"]["warning"] >= 1


def test_a_check_that_raises_is_an_error_row_not_a_dead_job(tmp_path, monkeypatch):
    """One failing check must not take the other two down, and must not be
    reported as having passed."""
    def boom(ctx):
        raise ValueError("check exploded")

    monkeypatch.setattr(DEEP_CHECKS[0], "fn", boom)

    job = DeepCheckJob(_context(tmp_path))
    job.start()
    job.wait(timeout=60)

    checks = job.progress()["checks"]
    assert checks[0]["status"] == "error"
    assert "check exploded" in checks[0]["error"]
    assert all(c["status"] == "done" for c in checks[1:])


def test_cancelling_keeps_completed_checks(tmp_path):
    """Half a findings list is half an answer — unlike a half-built chart
    payload, what finished is kept rather than discarded."""
    job = DeepCheckJob(_context(tmp_path))
    job.cancel()
    job.start()
    job.wait(timeout=60)

    state = job.progress()
    assert state["cancelled"] is True
    statuses = {c["status"] for c in state["checks"]}
    assert statuses <= {"pending", "running", "done", "cancelled", "error"}
    # Nothing that never ran may be reported as done.
    assert all(c["findings"] == [] for c in state["checks"] if c["status"] != "done")


def test_cancelled_before_anything_ran_reports_no_findings(tmp_path):
    job = DeepCheckJob(_context(tmp_path))
    job.cancel()
    job.start()
    job.wait(timeout=60)

    result = job.progress()["result"]
    if result is not None:
        assert result["summary"]["critical"] == 0


def test_progress_fraction_advances_across_checks(tmp_path):
    job = DeepCheckJob(_context(tmp_path))
    fractions: list[float] = []
    job._on_progress_for_test = lambda state: fractions.append(state.get("fraction", 0.0))
    job.start()
    job.wait(timeout=60)

    assert fractions == sorted(fractions), "progress must never go backwards"
    assert job.progress()["fraction"] == 1.0


def test_findings_are_plain_dicts_for_the_js_bridge(tmp_path):
    """Everything crossing pywebview's js_api must be JSON-safe — a
    Finding dataclass is not."""
    job = DeepCheckJob(_context(tmp_path))
    job.start()
    job.wait(timeout=60)

    for check in job.progress()["checks"]:
        for finding in check["findings"]:
            assert isinstance(finding, dict)
            assert not isinstance(finding, Finding)
            assert set(finding) == {
                "check_id",
                "severity",
                "title",
                "detail",
                "columns",
                # The unit a finding is about, when it is about one. Empty
                # for a deep check, which reads the whole dataset.
                "unit",
            }
