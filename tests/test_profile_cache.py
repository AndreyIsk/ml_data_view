"""The profile cache's store (ml_data_view/profile_cache.py).

Everything here is about ONE question: when is a cached profile still the
truth about the file it describes? A cache that answers that wrongly is
worse than no cache — it puts numbers on screen that no longer describe
anything on disk, and nothing about them looks wrong.
"""
from __future__ import annotations

import json
import os
import sqlite3

import polars as pl
import pytest

from ml_data_view import profile_cache
from ml_data_view.profile_cache import (
    SECTION_CHART,
    SECTION_LOAD,
    FileKey,
    MemoryProfileCache,
    ProfileCache,
    cache_file_path,
)


@pytest.fixture
def data_file(tmp_path):
    path = tmp_path / "part.parquet"
    pl.DataFrame({"v": [1, 2, 3]}).write_parquet(path)
    return path


@pytest.fixture
def cache(tmp_path):
    with ProfileCache(tmp_path / "cache.sqlite") as store:
        yield store


# --- the key ------------------------------------------------------------


def test_the_key_is_path_size_and_mtime(data_file):
    key = FileKey.for_file(data_file)

    assert key.path == str(data_file.resolve())
    assert key.size_bytes == data_file.stat().st_size
    assert key.mtime_ns == data_file.stat().st_mtime_ns


def test_mtime_is_taken_in_nanoseconds(data_file):
    """st_mtime is a float and loses precision at real-world timestamps, so
    two genuinely different modification times can compare equal."""
    key = FileKey.for_file(data_file)

    assert isinstance(key.mtime_ns, int)
    assert key.mtime_ns > 10**15


# --- round trip ---------------------------------------------------------


def test_a_stored_profile_comes_back(cache, data_file):
    cache.put(data_file, {SECTION_LOAD: {"row_count": 3}})

    assert cache.get(data_file) == {SECTION_LOAD: {"row_count": 3}}


def test_an_unknown_file_is_a_miss(cache, tmp_path):
    assert cache.get(tmp_path / "never-profiled.parquet") is None


def test_sections_are_filled_independently(cache, data_file):
    """A folder load writes "load"; the chart build that follows adds
    "chart" to the SAME row, so the file is not read twice."""
    cache.merge_sections(data_file, {SECTION_LOAD: {"row_count": 3}})
    cache.merge_sections(data_file, {SECTION_CHART: {"v": {"kind": "numeric"}}})

    payload = cache.get(data_file)
    assert payload[SECTION_LOAD] == {"row_count": 3}
    assert payload[SECTION_CHART] == {"v": {"kind": "numeric"}}


# --- invalidation -------------------------------------------------------


def test_a_changed_size_invalidates_the_entry(cache, data_file):
    cache.put(data_file, {SECTION_LOAD: {"row_count": 3}})

    pl.DataFrame({"v": list(range(500))}).write_parquet(data_file)

    assert cache.get(data_file) is None


def test_a_changed_mtime_alone_invalidates_the_entry(cache, data_file):
    """Touched but the same length — the case a size check alone would
    miss, and the reason mtime is part of the key at all."""
    cache.put(data_file, {SECTION_LOAD: {"row_count": 3}})
    stat = data_file.stat()
    os.utime(data_file, ns=(stat.st_atime_ns, stat.st_mtime_ns + 10**9))

    assert cache.get(data_file) is None
    assert data_file.stat().st_size == stat.st_size


def test_rewriting_a_file_replaces_rather_than_duplicates(cache, data_file):
    cache.put(data_file, {SECTION_LOAD: {"row_count": 3}})
    pl.DataFrame({"v": list(range(500))}).write_parquet(data_file)
    cache.put(data_file, {SECTION_LOAD: {"row_count": 500}})

    assert cache.entry_count() == 1
    assert cache.get(data_file)[SECTION_LOAD]["row_count"] == 500


def test_a_schema_version_bump_invalidates_everything(tmp_path, data_file, monkeypatch):
    """The guarantee that a code change cannot serve a stale statistic: a
    number computed differently by new code must not come back from a row
    the old code wrote."""
    path = tmp_path / "cache.sqlite"
    with ProfileCache(path) as store:
        store.put(data_file, {SECTION_LOAD: {"row_count": 3}})
        assert store.get(data_file) is not None

    monkeypatch.setattr(profile_cache, "SCHEMA_VERSION", profile_cache.SCHEMA_VERSION + 1)

    with ProfileCache(path) as store:
        assert store.get(data_file) is None


def test_rows_from_an_older_version_are_not_left_occupying_disk(tmp_path, data_file, monkeypatch):
    """They can never be served, so leaving them would make Settings report
    disk as "cache" that will never save anyone a read."""
    path = tmp_path / "cache.sqlite"
    with ProfileCache(path) as store:
        store.put(data_file, {SECTION_LOAD: {"row_count": 3}})

    monkeypatch.setattr(profile_cache, "SCHEMA_VERSION", profile_cache.SCHEMA_VERSION + 1)

    with ProfileCache(path) as store:
        assert store.entry_count() == 0


def test_a_corrupt_payload_is_a_miss_not_a_crash(tmp_path, data_file):
    path = tmp_path / "cache.sqlite"
    with ProfileCache(path) as store:
        store.put(data_file, {SECTION_LOAD: {"row_count": 3}})
    connection = sqlite3.connect(str(path))
    connection.execute("UPDATE file_profile SET payload = ?", ("{not json",))
    connection.commit()
    connection.close()

    with ProfileCache(path) as store:
        assert store.get(data_file) is None


def test_a_vanished_file_is_a_miss_not_an_oserror(cache, data_file):
    cache.put(data_file, {SECTION_LOAD: {"row_count": 3}})
    data_file.unlink()

    assert cache.get(data_file) is None


# --- maintenance --------------------------------------------------------


def test_clear_empties_the_cache_and_reports_what_went(cache, tmp_path):
    for index in range(3):
        path = tmp_path / f"p{index}.parquet"
        pl.DataFrame({"v": [index]}).write_parquet(path)
        cache.put(path, {SECTION_LOAD: {"row_count": 1}})

    assert cache.clear() == 3
    assert cache.entry_count() == 0


def test_forget_drops_only_the_files_named(cache, tmp_path):
    """What "Force full re-profile" does: this folder is re-read, every
    other folder keeps its profiles."""
    paths = []
    for index in range(3):
        path = tmp_path / f"p{index}.parquet"
        pl.DataFrame({"v": [index]}).write_parquet(path)
        cache.put(path, {SECTION_LOAD: {"row_count": 1}})
        paths.append(path)

    cache.forget(paths[:2])

    assert cache.get(paths[0]) is None
    assert cache.get(paths[1]) is None
    assert cache.get(paths[2]) is not None


def test_size_on_disk_is_reported(cache, data_file):
    cache.put(data_file, {SECTION_LOAD: {"row_count": 3}})

    assert cache.size_bytes() > 0


def test_the_cache_lives_beside_the_settings_file(tmp_path):
    from ml_data_view.settings import settings_file_path

    assert cache_file_path(tmp_path).parent == settings_file_path(tmp_path).parent


def test_the_payload_is_stored_as_json_not_pickled(tmp_path, data_file):
    """A cache of numbers is data, not code. Reading it must not be able to
    execute anything, and it has to stay inspectable with sqlite3."""
    path = tmp_path / "cache.sqlite"
    with ProfileCache(path) as store:
        store.put(data_file, {SECTION_LOAD: {"row_count": 3}})

    connection = sqlite3.connect(str(path))
    payload = connection.execute("SELECT payload FROM file_profile").fetchone()[0]
    connection.close()

    assert json.loads(payload) == {SECTION_LOAD: {"row_count": 3}}


# --- the in-memory stand-in ---------------------------------------------


def test_the_memory_cache_answers_the_same_way(data_file):
    """Used when no cache is configured. It must be interchangeable, or the
    no-cache path becomes a second implementation that drifts."""
    store = MemoryProfileCache()

    assert store.get(data_file) is None
    store.merge_sections(data_file, {SECTION_LOAD: {"row_count": 3}})
    store.merge_sections(data_file, {SECTION_CHART: {}})

    assert store.get(data_file) == {SECTION_LOAD: {"row_count": 3}, SECTION_CHART: {}}
    assert store.entry_count() == 1
    assert store.clear() == 1
    assert store.get(data_file) is None
