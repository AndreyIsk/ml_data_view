import polars as pl
import pytest

from ml_data_view.folder_loader import discover_files, load_group, scan_folder


def _write(path, df: pl.DataFrame, fmt: str = "parquet") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fmt == "parquet":
        df.write_parquet(path)
    elif fmt == "csv":
        df.write_csv(path)
    elif fmt == "arrow":
        df.write_ipc(path)


def _sample_df(n: int = 3) -> pl.DataFrame:
    return pl.DataFrame({"id": list(range(n)), "label": [f"row{i}" for i in range(n)]})


# --- discover_files ---------------------------------------------------


def test_discover_files_finds_files_recursively_in_nested_subfolders(tmp_path):
    _write(tmp_path / "a" / "b" / "c" / "one.parquet", _sample_df())
    _write(tmp_path / "x" / "two.csv", _sample_df(), fmt="csv")
    _write(tmp_path / "three.arrow", _sample_df(), fmt="arrow")

    found = discover_files(tmp_path)

    assert {p.name for p in found} == {"one.parquet", "two.csv", "three.arrow"}


def test_discover_files_ignores_unsupported_extensions(tmp_path):
    _write(tmp_path / "data.parquet", _sample_df())
    (tmp_path / "notes.txt").write_text("not a dataset")
    (tmp_path / "config.json").write_text("{}")

    found = discover_files(tmp_path)

    assert [p.name for p in found] == ["data.parquet"]


def test_discover_files_recognizes_feather_extension(tmp_path):
    _write(tmp_path / "data.feather", _sample_df(), fmt="arrow")

    found = discover_files(tmp_path)

    assert [p.name for p in found] == ["data.feather"]


def test_discover_files_returns_empty_list_for_empty_folder(tmp_path):
    assert discover_files(tmp_path) == []


# --- scan_folder: schema grouping --------------------------------------


def test_scan_folder_groups_matching_schema_files_together(tmp_path):
    _write(tmp_path / "a" / "one.parquet", _sample_df(3))
    _write(tmp_path / "b" / "c" / "two.csv", _sample_df(4), fmt="csv")
    _write(tmp_path / "three.arrow", _sample_df(5), fmt="arrow")

    result = scan_folder(tmp_path)

    assert len(result.groups) == 1
    assert result.groups[0].file_count == 3
    assert result.unreadable == []


def test_scan_folder_puts_mismatched_schema_file_in_its_own_group(tmp_path):
    _write(tmp_path / "one.parquet", _sample_df(3))
    _write(tmp_path / "two.parquet", _sample_df(4))
    mismatched_df = pl.DataFrame({"different_column": [1, 2], "extra": [1.0, 2.0]})
    _write(tmp_path / "odd_one_out.parquet", mismatched_df)

    result = scan_folder(tmp_path)

    assert len(result.groups) == 2
    file_counts = sorted(g.file_count for g in result.groups)
    assert file_counts == [1, 2]
    odd_group = next(g for g in result.groups if g.file_count == 1)
    assert odd_group.files[0].name == "odd_one_out.parquet"


def test_scan_folder_default_group_is_the_largest_by_file_count(tmp_path):
    _write(tmp_path / "one.parquet", _sample_df())
    _write(tmp_path / "two.parquet", _sample_df())
    _write(tmp_path / "three.parquet", _sample_df())
    mismatched_df = pl.DataFrame({"other": [1, 2]})
    _write(tmp_path / "odd.parquet", mismatched_df)

    result = scan_folder(tmp_path)

    assert result.groups[0].file_count == 3  # sorted largest-first
    assert result.default_group_index == 0


def test_scan_folder_empty_folder_returns_no_groups(tmp_path):
    result = scan_folder(tmp_path)

    assert result.groups == []
    assert result.unreadable == []


def test_scan_folder_reports_unreadable_files_with_a_reason(tmp_path):
    _write(tmp_path / "good.parquet", _sample_df())
    bad_path = tmp_path / "corrupt.parquet"
    bad_path.write_bytes(b"not actually parquet data")

    result = scan_folder(tmp_path)

    assert len(result.groups) == 1
    assert len(result.unreadable) == 1
    assert result.unreadable[0].path.name == "corrupt.parquet"
    assert result.unreadable[0].reason  # non-empty, short explanation


def test_scan_folder_row_count_cheap_for_parquet_and_arrow(tmp_path):
    _write(tmp_path / "one.parquet", _sample_df(3))
    _write(tmp_path / "two.parquet", _sample_df(4))
    _write(tmp_path / "three.arrow", _sample_df(5), fmt="arrow")

    result = scan_folder(tmp_path)

    assert result.groups[0].file_count == 3
    assert result.groups[0].row_count == 12


def test_scan_folder_row_count_none_when_group_includes_a_csv_file(tmp_path):
    # CSV has no metadata-only row count, so a mixed-format group's total
    # is reported as unknown rather than forcing a full scan during discovery.
    _write(tmp_path / "one.parquet", _sample_df(3))
    _write(tmp_path / "two.csv", _sample_df(4), fmt="csv")

    result = scan_folder(tmp_path)

    assert result.groups[0].file_count == 2
    assert result.groups[0].row_count is None


# --- excluded_from ------------------------------------------------------


def test_excluded_from_lists_other_groups_files_with_a_reason(tmp_path):
    _write(tmp_path / "one.parquet", _sample_df())
    _write(tmp_path / "two.parquet", _sample_df())
    mismatched_df = pl.DataFrame({"other": [1, 2]})
    _write(tmp_path / "odd.parquet", mismatched_df)

    result = scan_folder(tmp_path)
    excluded = result.excluded_from(result.default_group_index)

    assert len(excluded) == 1
    assert excluded[0]["file"].endswith("odd.parquet")
    assert "schema" in excluded[0]["reason"].lower()


def test_excluded_from_lists_unreadable_files_with_their_reason(tmp_path):
    _write(tmp_path / "good.parquet", _sample_df())
    (tmp_path / "corrupt.parquet").write_bytes(b"garbage")

    result = scan_folder(tmp_path)
    excluded = result.excluded_from(result.default_group_index)

    assert len(excluded) == 1
    assert excluded[0]["file"].endswith("corrupt.parquet")
    assert excluded[0]["reason"]


def test_excluded_from_is_empty_when_only_one_group_and_nothing_unreadable(tmp_path):
    _write(tmp_path / "one.parquet", _sample_df())
    _write(tmp_path / "two.parquet", _sample_df())

    result = scan_folder(tmp_path)
    excluded = result.excluded_from(result.default_group_index)

    assert excluded == []


# --- load_group -----------------------------------------------------------


def test_load_group_concatenates_files_into_one_lazyframe(tmp_path):
    _write(tmp_path / "a" / "one.parquet", _sample_df(3))
    _write(tmp_path / "b" / "two.parquet", _sample_df(4))

    result = scan_folder(tmp_path)
    lf = load_group(result.groups[0])

    assert isinstance(lf, pl.LazyFrame)
    assert lf.select(pl.len()).collect(engine="streaming")[0, 0] == 7


def test_load_group_raises_for_empty_group():
    from ml_data_view.folder_loader import SchemaGroup

    empty_group = SchemaGroup(signature=(), files=[], row_count=0)
    with pytest.raises(ValueError):
        load_group(empty_group)


# --- laziness invariant ----------------------------------------------------


def test_scan_folder_never_calls_collect_during_discovery(tmp_path, monkeypatch):
    # Discovery reads schema (collect_schema, metadata-only) and a cheap row
    # count — parquet's via pyarrow file metadata directly (no collect() at
    # all), arrow/feather's via a bounded streaming length aggregation (one
    # collect() per file, but its result is a single scalar count, never row
    # data). A decent number of files, across nested folders and formats, to
    # make this a meaningful check.
    for i in range(30):
        _write(tmp_path / f"group_a" / f"nested_{i % 5}" / f"file_{i}.parquet", _sample_df(10))
    for i in range(5):
        _write(tmp_path / "group_b" / f"file_{i}.csv", _sample_df(10), fmt="csv")
    for i in range(5):
        _write(tmp_path / "group_a" / f"arrow_{i}.arrow", _sample_df(10), fmt="arrow")

    collected_heights = []
    original_collect = pl.LazyFrame.collect

    def tracking_collect(self, *args, **kwargs):
        result = original_collect(self, *args, **kwargs)
        collected_heights.append(result.height)
        return result

    monkeypatch.setattr(pl.LazyFrame, "collect", tracking_collect)

    result = scan_folder(tmp_path)

    # Sanity: this only means something if discovery actually found and
    # grouped the 40 files (all one schema here — parquet, CSV, and arrow
    # all produced the same {id, label} columns from _sample_df).
    assert sum(g.file_count for g in result.groups) == 40
    # The only collect() calls allowed during discovery are the arrow/
    # feather row-count aggregations (one per arrow file, 5 here), and each
    # must pull back a single scalar row count — never anything resembling
    # actual row data (_sample_df(10) rows per file would show up as height
    # 10 if a full read snuck in).
    assert len(collected_heights) == 5, (
        f"expected exactly 5 collect() calls (one per arrow file's row-count "
        f"aggregation), got {len(collected_heights)}: {collected_heights}"
    )
    assert all(h == 1 for h in collected_heights), (
        f"a collect() during discovery pulled back more than a single scalar "
        f"row count — looks like real row data got materialized. "
        f"Heights: {collected_heights}"
    )


def test_scan_folder_discovery_stays_metadata_only_at_scale(tmp_path, monkeypatch):
    # Regression test for the OOM-during-folder-selection bug: discovery
    # must never materialize row data, no matter the file format. A "few
    # hundred MB" fixture (nowhere near a real 12GB folder, but big enough
    # that a full/near-full read of any one file would be unmistakable in
    # the collected heights below) across all three supported formats.
    n_rows = 2_000_000
    big_df = pl.DataFrame(
        {"id": list(range(n_rows)), "label": [f"row number {i} of many" for i in range(n_rows)]}
    )
    _write(tmp_path / "a.parquet", big_df)
    _write(tmp_path / "b.csv", big_df, fmt="csv")
    _write(tmp_path / "c.arrow", big_df, fmt="arrow")
    _write(tmp_path / "d.arrow", big_df, fmt="arrow")

    total_bytes = sum(p.stat().st_size for p in tmp_path.iterdir())
    assert total_bytes > 200_000_000  # sanity: this fixture is actually sizeable

    collected_heights = []
    original_collect = pl.LazyFrame.collect

    def tracking_collect(self, *args, **kwargs):
        result = original_collect(self, *args, **kwargs)
        collected_heights.append(result.height)
        return result

    monkeypatch.setattr(pl.LazyFrame, "collect", tracking_collect)

    result = scan_folder(tmp_path)

    # The real assertion: every collect() during discovery pulled back at
    # most a single scalar (a row count), never anything close to the
    # n_rows actually sitting in each file.
    over_limit = [h for h in collected_heights if h > 1]
    assert not over_limit, (
        f"a collect() during discovery pulled {max(collected_heights)} rows "
        f"out of {n_rows} in the file — looks like a full/unbounded "
        f"materialization snuck into discovery. All heights: {collected_heights}"
    )

    # And the pipeline still did real work — proves the boundedness above
    # isn't just because nothing ran (parquet: metadata-only, no collect;
    # arrow x2: one bounded collect each; csv: no cheap count available).
    assert sum(g.file_count for g in result.groups) == 4
    assert len(collected_heights) == 2
    assert result.groups[0].row_count is None  # csv in the group forces "unknown"
