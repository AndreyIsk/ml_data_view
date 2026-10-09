"""Per-file profiling and the merge (ml_data_view/file_profile.py).

Correctness before speed. The claim this feature makes is that a folder
profile assembled by MERGING cached per-file statistics is the same profile
a full rescan would produce — so the first tests here pin the merged result
against the direct one, statistic by statistic, and the rest prove that the
work actually skipped was only the work that was safe to skip.
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from ml_data_view import file_profile, profile_cache
from ml_data_view.chart_data import build_chart_payload
from ml_data_view.file_profile import (
    HistogramSpec,
    load_folder_profile,
    merge_load,
    profile_load_section,
)
from ml_data_view.folder_loader import load_files
from ml_data_view.profile_cache import SECTION_HIST, MemoryProfileCache, ProfileCache

# Chart families that are assembled by merging per-file statistics. Every
# one of them must equal what a direct recompute produces — that is the
# whole contract, and listing them here means a new mergeable chart has to
# be added to the check rather than quietly escaping it.
MERGEABLE_CHART_KEYS = [
    "numeric_histograms",
    "categorical_bars",
    "text_length_histograms",
    "value_frequency_histograms",
]


def _write_fixture(folder: Path, file_count: int = 4, seed: int = 7) -> list[Path]:
    """A multi-file folder with one column of each kind that merges.

    Per-file row counts differ and the numeric ranges overlap without
    matching, so a merge that quietly used one file's extent for everything
    would show up immediately.
    """
    folder.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    files = []
    for index in range(file_count):
        rows = 200 + index * 40
        frame = pl.DataFrame(
            {
                "score": rng.normal(50 + index * 5, 10, rows),
                # Long-tailed, so the skew heuristic puts it on a log scale
                # and the log-space merge gets exercised too.
                "skewed": rng.exponential(400 * (index + 1), rows),
                "flag": rng.choice(["alpha", "beta", "gamma"], rows),
                "note": ["x" * (3 + (position + index) % 37) for position in range(rows)],
                "holes": [
                    None if position % 3 == 0 else float(position) for position in range(rows)
                ],
                "nanny": [
                    float("nan") if position % 5 == 0 else float(position) for position in range(rows)
                ],
            }
        )
        path = folder / f"part{index}.parquet"
        frame.write_parquet(path)
        files.append(path)
    return files


@pytest.fixture
def folder(tmp_path):
    return _write_fixture(tmp_path / "data")


@pytest.fixture
def cache(tmp_path):
    with ProfileCache(tmp_path / "cache.sqlite") as store:
        yield store


# --- the merge equals a full recompute ----------------------------------


def test_merged_chart_payload_equals_a_full_recompute(folder, tmp_path):
    """The core claim. The direct per-file path stays the reference
    implementation; the cached path has to agree with it exactly, not
    approximately, for every statistic designed to merge."""
    lf = load_files(folder)

    direct = build_chart_payload(lf, files=folder)
    merged = build_chart_payload(lf, files=folder, cache_path=tmp_path / "cache.sqlite")

    for key in MERGEABLE_CHART_KEYS:
        assert merged[key] == direct[key], key


def test_merged_histograms_share_bin_edges_with_a_full_recompute(folder, tmp_path):
    """Per-file counts only add up if the bins are the same bins, so the
    edges are the part that actually has to match — a histogram with the
    right total and the wrong edges is a different chart."""
    lf = load_files(folder)

    direct = {h["column"]: h for h in build_chart_payload(lf, files=folder)["numeric_histograms"]}
    merged = {
        h["column"]: h
        for h in build_chart_payload(
            lf, files=folder, cache_path=tmp_path / "cache.sqlite"
        )["numeric_histograms"]
    }

    assert set(merged) == set(direct)
    for column, histogram in merged.items():
        assert histogram["bin_edges"] == direct[column]["bin_edges"], column
        assert histogram["counts"] == direct[column]["counts"], column
        assert histogram["x_scale"] == direct[column]["x_scale"], column
    # The fixture is built so at least one column is long-tailed; if the log
    # path stopped being taken, this test would silently stop covering it.
    assert any(h["x_scale"] == "log" for h in merged.values())


def test_merged_counts_and_nulls_equal_a_direct_computation(folder):
    """Row counts, null counts, NaN counts and missing counts are all plain
    sums — the easy half of the merge, and the half every other number on
    screen is derived from."""
    schema = load_files(folder).collect_schema()
    merged = merge_load(
        [profile_load_section(file, schema) for file in folder], schema
    )

    whole = pl.concat([pl.scan_parquet(f) for f in folder], how="vertical")
    expected = whole.select(
        pl.len().alias("rows"),
        pl.col("holes").null_count().alias("holes_null"),
        pl.col("nanny").is_nan().sum().alias("nanny_nan"),
        pl.col("nanny").null_count().alias("nanny_null"),
    ).collect(engine="streaming")

    assert merged.row_count == expected[0, "rows"]
    assert merged.null_counts["holes"] == expected[0, "holes_null"]
    assert merged.nan_counts["nanny"] == expected[0, "nanny_nan"]
    # "Missing" is nulls plus NaN, which is what the schema table and every
    # chart in this app mean by it.
    assert merged.missing_counts["nanny"] == expected[0, "nanny_nan"] + expected[0, "nanny_null"]
    assert merged.missing_counts["holes"] == expected[0, "holes_null"]


def test_merged_min_and_max_equal_a_direct_computation(folder, cache):
    """min/max merge by min/max — and they are what the histogram bin edges
    are derived from, so an error here moves every bar."""
    schema = load_files(folder).collect_schema()
    plan = {"score": "numeric", "skewed": "numeric"}
    profile = load_folder_profile(folder, schema, cache)
    file_profile._fill_chart_sections(profile.profiles, schema, plan, cache)

    whole = pl.concat([pl.scan_parquet(f) for f in folder], how="vertical")
    for column in plan:
        sections = [
            p.payload[profile_cache.SECTION_CHART][column] for p in profile.profiles
        ]
        stats = file_profile._merge_numeric(sections)
        expected = whole.select(
            pl.col(column).min().alias("mn"),
            pl.col(column).max().alias("mx"),
            pl.col(column).count().alias("n"),
        ).collect(engine="streaming")
        assert stats.minimum == pytest.approx(expected[0, "mn"])
        assert stats.maximum == pytest.approx(expected[0, "mx"])
        assert stats.n == expected[0, "n"]


def test_a_removed_file_simply_drops_out_of_the_merge(folder, tmp_path):
    """No bookkeeping needed: the merge is over the files discovered now,
    so a file that is gone contributes nothing — and the cache row it left
    behind cannot be reached by anything."""
    cache_path = tmp_path / "cache.sqlite"
    lf_all = load_files(folder)
    build_chart_payload(lf_all, files=folder, cache_path=cache_path)

    remaining = folder[:-1]
    merged = build_chart_payload(load_files(remaining), files=remaining, cache_path=cache_path)
    direct = build_chart_payload(load_files(remaining), files=remaining)

    for key in MERGEABLE_CHART_KEYS:
        assert merged[key] == direct[key], key


# --- what actually gets read --------------------------------------------


class ReadMonitor:
    """Records every per-file read the profiler makes.

    Patches the three functions that are the only places a source file's
    DATA is touched on this path — the load-section scan, a column's chart
    statistics, and a file's histogram bin counts. Parquet footer reads are
    deliberately not counted: metadata-only reads are free by this
    codebase's own standard (see PROJECT_BRIEF.md's folder-discovery note),
    and counting them would make every assertion here about the wrong thing.
    """

    def __init__(self, monkeypatch):
        self.files: list[Path] = []
        self.histograms: list[tuple[Path, str]] = []
        self.columns: list[tuple[Path, str]] = []
        self._patch(monkeypatch)

    def _patch(self, monkeypatch):
        real_load = file_profile.profile_load_section
        real_column = file_profile.profile_chart_column
        real_hist = file_profile._file_histogram

        def load(file, schema, timestamp_column=None):
            self.files.append(Path(file))
            return real_load(file, schema, timestamp_column)

        def column(file, name, kind, dtype):
            self.files.append(Path(file))
            self.columns.append((Path(file), name))
            return real_column(file, name, kind, dtype)

        def histogram(file, name, kind, spec):
            self.files.append(Path(file))
            self.histograms.append((Path(file), name))
            return real_hist(file, name, kind, spec)

        monkeypatch.setattr(file_profile, "profile_load_section", load)
        monkeypatch.setattr(file_profile, "profile_chart_column", column)
        monkeypatch.setattr(file_profile, "_file_histogram", histogram)

    @property
    def touched(self) -> set[Path]:
        return {path.resolve() for path in self.files}


def test_adding_one_file_reads_only_that_file(tmp_path, monkeypatch):
    """The point of caching per FILE rather than per folder."""
    folder = _write_fixture(tmp_path / "data", file_count=4)
    cache_path = tmp_path / "cache.sqlite"
    build_chart_payload(load_files(folder), files=folder, cache_path=cache_path)

    newcomer = tmp_path / "data" / "part4.parquet"
    pl.DataFrame(
        {
            "score": [50.0, 51.0, 49.0],
            "skewed": [10.0, 20.0, 30.0],
            "flag": ["alpha", "beta", "alpha"],
            "note": ["xxx", "xxxx", "xxxxx"],
            "holes": [None, 1.0, 2.0],
            "nanny": [float("nan"), 1.0, 2.0],
        }
    ).write_parquet(newcomer)
    grown = folder + [newcomer]

    monitor = ReadMonitor(monkeypatch)
    payload = build_chart_payload(load_files(grown), files=grown, cache_path=cache_path)

    assert monitor.touched == {newcomer.resolve()}, (
        f"files other than the new one were read: {sorted(monitor.touched)}"
    )
    assert payload["cache_stats"]["files_from_cache"] == 4
    assert payload["cache_stats"]["files_scanned"] == 1


def test_modifying_one_file_reads_only_that_file(tmp_path, monkeypatch):
    folder = _write_fixture(tmp_path / "data", file_count=4)
    cache_path = tmp_path / "cache.sqlite"
    build_chart_payload(load_files(folder), files=folder, cache_path=cache_path)

    changed = folder[1]
    stat = changed.stat()
    os.utime(changed, ns=(stat.st_atime_ns, stat.st_mtime_ns + 10**9))

    monitor = ReadMonitor(monkeypatch)
    payload = build_chart_payload(load_files(folder), files=folder, cache_path=cache_path)

    assert monitor.touched == {changed.resolve()}
    assert payload["cache_stats"]["files_scanned"] == 1


def test_an_unchanged_folder_is_not_read_at_all(tmp_path, monkeypatch):
    folder = _write_fixture(tmp_path / "data", file_count=3)
    cache_path = tmp_path / "cache.sqlite"
    build_chart_payload(load_files(folder), files=folder, cache_path=cache_path)

    monitor = ReadMonitor(monkeypatch)
    payload = build_chart_payload(load_files(folder), files=folder, cache_path=cache_path)

    assert monitor.touched == set()
    assert payload["cache_stats"]["files_from_cache"] == 3


def test_a_schema_version_bump_reads_everything_again(tmp_path, monkeypatch):
    folder = _write_fixture(tmp_path / "data", file_count=3)
    cache_path = tmp_path / "cache.sqlite"
    build_chart_payload(load_files(folder), files=folder, cache_path=cache_path)

    monkeypatch.setattr(profile_cache, "SCHEMA_VERSION", profile_cache.SCHEMA_VERSION + 1)
    monitor = ReadMonitor(monkeypatch)
    payload = build_chart_payload(load_files(folder), files=folder, cache_path=cache_path)

    assert monitor.touched == {path.resolve() for path in folder}
    assert payload["cache_stats"]["files_from_cache"] == 0


# --- range expansion -----------------------------------------------------


def test_a_widened_range_rebins_only_the_affected_column(tmp_path, monkeypatch):
    """The case the whole histogram design exists for.

    A new file whose values fall outside the cached bin edges makes those
    edges wrong for every file, so that column's counts have to be redone
    across the folder. Every OTHER column — and every other statistic of
    the same column — still comes from the cache.
    """
    folder = tmp_path / "data"
    folder.mkdir()
    files = []
    for index in range(3):
        path = folder / f"part{index}.parquet"
        pl.DataFrame(
            {
                "narrow": [float(v) for v in range(100)],
                "steady": [float(v % 10) for v in range(100)],
                "flag": ["a" if v % 2 else "b" for v in range(100)],
            }
        ).write_parquet(path)
        files.append(path)

    cache_path = tmp_path / "cache.sqlite"
    build_chart_payload(load_files(files), files=files, cache_path=cache_path)

    # Same schema, but "narrow" now reaches far past the cached maximum.
    # "steady" keeps exactly the range it had.
    outlier = folder / "part3.parquet"
    pl.DataFrame(
        {
            "narrow": [500_000.0, 1.0, 2.0],
            "steady": [1.0, 2.0, 3.0],
            "flag": ["a", "b", "a"],
        }
    ).write_parquet(outlier)
    grown = files + [outlier]

    monitor = ReadMonitor(monkeypatch)
    payload = build_chart_payload(load_files(grown), files=grown, cache_path=cache_path)

    assert payload["cache_stats"]["range_expanded_columns"] == ["narrow"]
    # The already-cached files were re-binned for "narrow" and nothing else
    # — not "steady", whose range did not move. The new file is binned for
    # every column because it has no cached histogram at all, which is the
    # one file that genuinely has to be read.
    rebinned = {(path.resolve(), column) for path, column in monitor.histograms}
    assert rebinned == {(path.resolve(), "narrow") for path in grown} | {
        (outlier.resolve(), "steady")
    }
    # And no cached file had its non-histogram statistics recomputed: min,
    # max, quantiles and category counts all still came from the cache.
    assert monitor.columns == [
        (outlier, name) for name in ["narrow", "steady", "flag"]
    ], "a cached file's per-column statistics were recomputed"

    # And the answer is still right.
    direct = build_chart_payload(load_files(grown), files=grown)
    for key in MERGEABLE_CHART_KEYS:
        assert payload[key] == direct[key], key


def test_a_steady_range_is_not_reported_as_expanded(tmp_path):
    """`range_expanded` has to mean something. A first profile has no
    cached edges to invalidate, so it is not an expansion."""
    folder = _write_fixture(tmp_path / "data", file_count=2)

    payload = build_chart_payload(
        load_files(folder), files=folder, cache_path=tmp_path / "cache.sqlite"
    )

    assert payload["cache_stats"]["range_expanded_columns"] == []


# --- the approximations, stated ------------------------------------------


def test_a_truncated_top_k_is_flagged(tmp_path, monkeypatch):
    """A category below the per-file shortlist in every file is missed by
    the merge. The chart says so rather than presenting the number as a
    count."""
    monkeypatch.setattr("ml_data_view.chart_data.CATEGORICAL_PER_FILE_TOP_K", 3)
    monkeypatch.setattr(file_profile, "CACHE_CATEGORICAL_TOP_K", 3)
    folder = tmp_path / "data"
    folder.mkdir()
    files = []
    for index in range(2):
        path = folder / f"part{index}.parquet"
        pl.DataFrame({"many": [f"v{v % 20}" for v in range(200)]}).write_parquet(path)
        files.append(path)

    payload = build_chart_payload(
        load_files(files), files=files, cache_path=tmp_path / "cache.sqlite"
    )

    bars = next(b for b in payload["categorical_bars"] if b["column"] == "many")
    assert bars["truncated"] is True
    # The chart still adds up to every row: whatever the shortlist missed
    # lands in "Other" rather than vanishing.
    assert sum(bars["counts"]) == 400


def test_an_untruncated_top_k_is_not_flagged(folder, tmp_path):
    payload = build_chart_payload(
        load_files(folder), files=folder, cache_path=tmp_path / "cache.sqlite"
    )

    bars = next(b for b in payload["categorical_bars"] if b["column"] == "flag")
    assert bars["truncated"] is False


# --- histogram specs -----------------------------------------------------


def test_a_spec_generates_the_same_edges_chart_data_computes():
    """The two have to agree bit for bit, or two payloads describing the
    same histogram compare unequal (and they are compared, above)."""
    spec = HistogramSpec(2.0, 98.0, 30, "linear", 0.0)

    expected = [2.0 + (98.0 - 2.0) * index / 30 for index in range(31)]
    assert spec.edges() == expected


def test_a_log_spec_reports_its_edges_in_original_units():
    """`bin_edges` is always in the column's own units, whatever space the
    binning happened in — consumers label axes without knowing about the
    transform."""
    spec = file_profile.numeric_histogram_spec(-5.0, 100_000.0, -4.0, 1.0, 30)

    assert spec.scale == "log"
    edges = spec.display_edges()
    assert edges[0] == pytest.approx(-5.0)
    assert edges[-1] == pytest.approx(100_000.0)


def test_a_constant_column_has_no_spec_and_gets_one_bin(tmp_path):
    folder = tmp_path / "data"
    folder.mkdir()
    files = []
    for index in range(2):
        path = folder / f"part{index}.parquet"
        pl.DataFrame({"same": [7.0] * 20}).write_parquet(path)
        files.append(path)

    payload = build_chart_payload(
        load_files(files), files=files, cache_path=tmp_path / "cache.sqlite"
    )

    histogram = next(h for h in payload["numeric_histograms"] if h["column"] == "same")
    assert histogram["counts"] == [40]
    assert histogram["bin_edges"] == [7.0, 7.0]


def test_specs_are_stored_rather_than_edge_lists(folder, tmp_path, cache):
    """31 floats per column per file across 250 files is megabytes of JSON
    to say what five numbers say."""
    build_chart_payload(load_files(folder), files=folder, cache_path=cache.path)

    stored = cache.get(folder[0])[SECTION_HIST]["score"]
    assert HistogramSpec.from_json(stored["spec"]) is not None
    assert len(stored["spec"]) == 5


# --- a filtered build never uses the cache -------------------------------


def test_a_filtered_build_does_not_touch_the_cache(folder, tmp_path, monkeypatch):
    """A chart filter changes which rows every file contributes, so a
    cached per-file statistic is simply not about the same rows."""
    from ml_data_view.search import Condition, build_predicate

    lf = load_files(folder)
    predicate = build_predicate(
        [Condition(column="score", operator=">=", values=["55"])], lf.collect_schema()
    )

    monitor = ReadMonitor(monkeypatch)
    payload = build_chart_payload(
        lf, files=folder, predicate=predicate, cache_path=tmp_path / "cache.sqlite"
    )

    assert monitor.touched == set(), "the filtered build went through the cached path"
    assert "cache_stats" not in payload


# --- the no-cache path stays the same code -------------------------------


def test_the_memory_cache_produces_the_same_merge(folder):
    """A caller with nowhere to persist still goes through one profiling
    path — it just has no yesterday to draw on."""
    schema = load_files(folder).collect_schema()
    store = MemoryProfileCache()

    first = load_folder_profile(folder, schema, store)
    second = load_folder_profile(folder, schema, store)

    assert first.files_scanned == len(folder)
    assert second.files_from_cache == len(folder)
    assert first.merged.row_count == second.merged.row_count
    assert first.merged.missing_counts == second.merged.missing_counts


# --- correlated missingness' cacheable half ------------------------------


def _co_null_folder(tmp_path, file_count=3, rows=60):
    """Two columns that go missing on exactly the same rows, across files."""
    folder = tmp_path / "conull"
    folder.mkdir()
    files = []
    for index in range(file_count):
        missing = [position % 4 == 0 for position in range(rows)]
        path = folder / f"part{index}.parquet"
        pl.DataFrame(
            {
                "a": [None if gone else 1.0 for gone in missing],
                "b": [None if gone else "x" for gone in missing],
                "keep": list(range(rows)),
            }
        ).write_parquet(path)
        files.append(path)
    return files


def test_co_null_counts_are_cached_per_file_and_summed(tmp_path, cache):
    """The group STRUCTURE is recomputed every load — which pairs to check
    depends on merged null counts. The per-file counts for a chosen pair
    are a plain sum, so they cache, and a re-opened folder pays nothing for
    what is otherwise the one real scan in a chart build."""
    from ml_data_view.null_patterns import correlated_missingness

    files = _co_null_folder(tmp_path)
    lf = load_files(files)
    null_counts = {"a": 45, "b": 45, "keep": 0}
    store = file_profile.CachedCoNulls(cache)

    first = correlated_missingness(lf, null_counts, 180, files=files, co_null_cache=store)
    second = correlated_missingness(lf, null_counts, 180, files=files, co_null_cache=store)

    assert first == second
    assert first and set(first[0]["columns"]) == {"a", "b"}
    # Every file now carries the pair it was asked about.
    for file in files:
        assert store.get(file, "a", "b") == 15


def test_a_cached_co_null_count_is_not_recomputed(tmp_path, cache, monkeypatch):
    from ml_data_view import null_patterns

    files = _co_null_folder(tmp_path)
    lf = load_files(files)
    null_counts = {"a": 45, "b": 45, "keep": 0}
    store = file_profile.CachedCoNulls(cache)
    correlated_first = null_patterns.correlated_missingness(
        lf, null_counts, 180, files=files, co_null_cache=store
    )

    collected = []
    original = pl.LazyFrame.collect

    def tracking(self, *args, **kwargs):
        collected.append(1)
        return original(self, *args, **kwargs)

    monkeypatch.setattr(pl.LazyFrame, "collect", tracking)
    correlated_second = null_patterns.correlated_missingness(
        lf, null_counts, 180, files=files, co_null_cache=store
    )

    assert correlated_second == correlated_first
    assert collected == [], "a cached co-null pair was scanned again"


def test_a_filtered_run_never_writes_co_null_counts(tmp_path, cache):
    """Counts under a filter describe different rows, so they must not end
    up in an entry that an unfiltered load would then trust."""
    from ml_data_view.null_patterns import correlated_missingness
    from ml_data_view.search import Condition, build_predicate

    files = _co_null_folder(tmp_path)
    lf = load_files(files)
    predicate = build_predicate(
        [Condition(column="keep", operator=">=", values=["10"])], lf.collect_schema()
    )
    store = file_profile.CachedCoNulls(cache)

    correlated_missingness(
        lf, {"a": 45, "b": 45}, 180, files=files, predicate=predicate, co_null_cache=store
    )

    assert all(store.get(file, "a", "b") is None for file in files)
