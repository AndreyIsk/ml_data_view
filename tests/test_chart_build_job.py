import polars as pl

from ml_data_view.chart_build_job import ChartBuildJob


def _frame(columns=6, rows=40):
    data = {f"num_{i}": [float(n % 7) for n in range(rows)] for i in range(columns)}
    data["label"] = [f"c{n % 3}" for n in range(rows)]
    return pl.DataFrame(data).lazy()


def _run(job):
    job.start()
    job.wait(timeout=60)
    return job.progress()


def test_job_completes_and_returns_the_payload():
    state = _run(ChartBuildJob(_frame()))

    assert state["finished"] is True
    assert state["error"] is None
    assert state["fraction"] == 1.0
    assert "numeric_histograms" in state["result"]


def test_progress_is_monotonic_and_within_bounds():
    job = ChartBuildJob(_frame(columns=10))
    seen = []
    job._on_progress_for_test = lambda s: seen.append(s["fraction"])

    _run(job)

    assert seen == sorted(seen), "progress must never go backwards"
    assert all(0.0 <= f <= 1.0 for f in seen)
    assert seen[-1] == 1.0


def test_progress_labels_name_the_current_step():
    job = ChartBuildJob(_frame(columns=4))
    labels = []
    job._on_progress_for_test = lambda s: labels.append(s["label"])

    _run(job)

    assert any("Charting columns" in label for label in labels)
    # The trailing whole-dataset steps must be named too, or the bar looks
    # stalled for the last stretch of a multi-minute build.
    assert any("heatmap" in label.lower() for label in labels)


def test_progress_before_start_is_idle():
    job = ChartBuildJob(_frame())
    state = job.progress()
    assert state["stage"] == "pending"
    assert state["finished"] is False


def test_error_is_recorded_not_raised_in_the_thread():
    class Broken:
        def collect_schema(self):
            raise RuntimeError("schema exploded")

    state = _run(ChartBuildJob(Broken()))

    assert state["finished"] is True
    assert "schema exploded" in state["error"]
    assert state["result"] is None
