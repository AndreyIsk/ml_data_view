"""Worker-thread wrapper for the Findings tab's deep checks.

Same shape as every other long operation here: start() returns at once,
the UI polls progress(), cancel() takes effect at a check boundary.

Like CompareJob — and unlike ChartBuildJob — this carries per-check status
in its progress state, so the tab renders each check resolving as it
finishes rather than staying blank until all three are done. That is the
whole point of running these as a job rather than one blocking call.

Cancelling keeps completed checks. Half a findings list is half an answer,
the same reasoning that lets a stopped folder load keep its whole files;
a half-built chart payload is the case where that does NOT hold, which is
why ChartBuildJob has no Stop button.
"""
from __future__ import annotations

from typing import Any

from ml_data_view.background_job import BackgroundJob
from ml_data_view.deep_checks import DEEP_CHECKS, DeepContext
from ml_data_view.findings import Finding, severity_summary


class DeepCheckJob(BackgroundJob):
    def __init__(self, context: DeepContext) -> None:
        # BackgroundJob's file-based progress helpers don't fit here — the
        # unit of work is a check, not a file — so it is given the check
        # list as its "files" and the fraction is driven per check below.
        super().__init__([])
        self._context = context
        self._checks: list[dict[str, Any]] = [
            {
                "id": spec.id,
                "title": spec.title,
                "kind": spec.kind,
                "status": "pending",
                "findings": [],
                "error": None,
            }
            for spec in DEEP_CHECKS
        ]
        # Painted by the UI before any work starts, which is what makes the
        # tab read as a scan rather than a list that appears at the end.
        self._update(checks=self._snapshot(), checks_done=0, checks_total=len(DEEP_CHECKS))

    def _snapshot(self) -> list[dict[str, Any]]:
        """A deep-enough copy: the UI polls this while the worker mutates
        the originals."""
        return [dict(check, findings=list(check["findings"])) for check in self._checks]

    def _set(self, index: int, **changes: Any) -> None:
        self._checks[index].update(changes)
        done = sum(1 for check in self._checks if check["status"] in ("done", "error"))
        total = len(self._checks) or 1
        self._update(
            stage="checking",
            checks=self._snapshot(),
            checks_done=done,
            fraction=round(done / total, 4),
            label=f"Running deep checks — {done} of {len(self._checks)}",
        )

    def _work(self) -> dict[str, Any]:
        for index, spec in enumerate(DEEP_CHECKS):
            if self._cancel.is_set():
                # Left pending rather than marked done: a check that never
                # ran is not a check that passed.
                self._set(index, status="cancelled")
                continue
            self._set(index, status="running")
            try:
                findings = spec.fn(self._context)
                self._set(
                    index,
                    status="done",
                    findings=[finding.to_dict() for finding in findings],
                )
            except Exception as exc:  # one bad check must not end the scan
                self._set(index, status="error", error=str(exc))

        findings = [
            Finding(**finding) for check in self._checks for finding in check["findings"]
        ]
        return {"checks": self._snapshot(), "summary": severity_summary(findings)}

    def _final_label(self, cancelled: bool) -> str:
        done = sum(1 for check in self._checks if check["status"] == "done")
        if cancelled:
            return f"Stopped — {done} of {len(self._checks)} checks completed."
        return "Deep checks complete."
