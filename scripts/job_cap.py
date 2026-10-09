"""Run a subprocess under a hard, OS-enforced Windows memory ceiling.

This started as a diagnostics script — every memory measurement quoted in
PROJECT_BRIEF.md was taken through it — and the app now depends on the
same enforcement for real work, so the implementation moved into the
package as ml_data_view/jobs.py. What is left here is the CLI and the
original import surface, unchanged:

    from scripts.job_cap import run_capped
    result = run_capped([sys.executable, "script.py"], limit_mb=2048)
    print(result.exit_code, result.killed_by_memory_limit, result.peak_mb)

    python scripts/job_cap.py --limit-mb 2048 -- python script.py arg1

See ml_data_view/jobs.py for how the ceiling is enforced (a Job Object
commit-charge limit plus a polling watchdog backstop) and for the job
runner built on top of it.
"""
from __future__ import annotations

import sys
from pathlib import Path

# Runnable as a bare script (`python scripts/job_cap.py`), where the
# repository root is not on sys.path.
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ml_data_view.jobs import (  # noqa: E402
    DEFAULT_POLL_INTERVAL_S,
    CappedResult,
    main,
    run_capped,
)

# The dataclass was named JobCapResult when it lived here; keep the name
# working for anything that imported it.
JobCapResult = CappedResult
DEFAULT_LIMIT_MB = 2048

__all__ = [
    "CappedResult",
    "JobCapResult",
    "DEFAULT_LIMIT_MB",
    "DEFAULT_POLL_INTERVAL_S",
    "main",
    "run_capped",
]


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
