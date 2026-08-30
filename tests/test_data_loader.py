import pyarrow as pa
import pyarrow.parquet as pq
import polars as pl
import pytest

import ml_data_view.data_loader as data_loader_module
from ml_data_view.data_loader import (
    ColumnInfo,
    DatasetInfo,
    UnsupportedFileTypeError,
    dataset_info_from_lazy,
    load_dataset,
    row_count,
    scan_dataframe,
)


def _sample_df() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "id": [1, 2, 3, 4],
            "label": ["cat", "dog", None, "cat"],
            "score": [0.9, None, 0.75, 0.5],
        }
    )


def test_dataset_info_to_dict_is_json_safe_shape():
    dataset_info = DatasetInfo(
        file_path="C:/data/sample.csv",
        file_size_bytes=1234,
        row_count=2,
        columns=[
            ColumnInfo(name="id", dtype="Int64", null_count=0),
            ColumnInfo(name="label", dtype="String", null_count=1),
        ],
        preview_rows=[{"id": 1, "label": "cat"}, {"id": 2, "label": None}],
    )

    result = dataset_info.to_dict()

    assert result == {
        "file_path": "C:/data/sample.csv",
        "file_size_bytes": 1234,
        "row_count": 2,
        "columns": [
            {"name": "id", "dtype": "Int64", "null_count": 0},
            {"name": "label", "dtype": "String", "null_count": 1},
        ],
        "preview_rows": [{"id": 1, "label": "cat"}, {"id": 2, "label": None}],
    }


def test_load_csv_reads_schema_nulls_and_row_count(tmp_path):
    csv_path = tmp_path / "sample.csv"
    _sample_df().write_csv(csv_path)

    dataset_info = load_dataset(csv_path)

    assert dataset_info.row_count == 4
    assert dataset_info.file_size_bytes == csv_path.stat().st_size
    assert dataset_info.file_path == str(csv_path)

    columns_by_name = {c.name: c for c in dataset_info.columns}
    assert set(columns_by_name) == {"id", "label", "score"}
    assert columns_by_name["label"].null_count == 1
    assert columns_by_name["score"].null_count == 1
    assert columns_by_name["id"].null_count == 0


def test_load_parquet_reads_schema_nulls_and_row_count(tmp_path):
    parquet_path = tmp_path / "sample.parquet"
    _sample_df().write_parquet(parquet_path)

    dataset_info = load_dataset(parquet_path)

    assert dataset_info.row_count == 4
    columns_by_name = {c.name: c for c in dataset_info.columns}
    assert columns_by_name["label"].null_count == 1
    assert columns_by_name["id"].dtype == "Int64"


def test_load_arrow_reads_schema_nulls_and_row_count(tmp_path):
    arrow_path = tmp_path / "sample.arrow"
    _sample_df().write_ipc(arrow_path)

    dataset_info = load_dataset(arrow_path)

    assert dataset_info.row_count == 4
    columns_by_name = {c.name: c for c in dataset_info.columns}
    assert columns_by_name["score"].null_count == 1


def test_load_dataset_raises_on_unsupported_extension(tmp_path):
    bad_path = tmp_path / "sample.txt"
    bad_path.write_text("not a dataset")

    with pytest.raises(UnsupportedFileTypeError):
        load_dataset(bad_path)


def test_load_dataset_raises_on_missing_file(tmp_path):
    missing_path = tmp_path / "does_not_exist.csv"

    with pytest.raises(FileNotFoundError):
        load_dataset(missing_path)


def test_preview_rows_capped_at_50_for_larger_files(tmp_path):
    big_df = pl.DataFrame({"id": list(range(60)), "value": [x * 2 for x in range(60)]})
    csv_path = tmp_path / "big.csv"
    big_df.write_csv(csv_path)

    dataset_info = load_dataset(csv_path)

    assert dataset_info.row_count == 60
    assert len(dataset_info.preview_rows) == 50
    assert dataset_info.preview_rows[0]["id"] == 0
    assert dataset_info.preview_rows[-1]["id"] == 49


def test_preview_rows_returns_all_when_fewer_than_50(tmp_path):
    small_df = pl.DataFrame({"id": [1, 2, 3]})
    csv_path = tmp_path / "small.csv"
    small_df.write_csv(csv_path)

    dataset_info = load_dataset(csv_path)

    assert dataset_info.row_count == 3
    assert len(dataset_info.preview_rows) == 3


def test_dataset_info_from_lazy_counts_nan_as_null_for_float_columns(tmp_path):
    df = pl.DataFrame({"score": [1.0, None, float("nan"), 2.0]})
    parquet_path = tmp_path / "sample.parquet"
    df.write_parquet(parquet_path)  # parquet round-trips NaN exactly, unlike CSV

    dataset_info = dataset_info_from_lazy(pl.scan_parquet(parquet_path), parquet_path)

    columns_by_name = {c.name: c for c in dataset_info.columns}
    # 1 real null + 1 NaN = 2 "missing" values, consistent with how the
    # Phase 1 null heatmap treats NaN as missing too.
    assert columns_by_name["score"].null_count == 2


def test_scan_dataframe_returns_lazyframe(tmp_path):
    csv_path = tmp_path / "sample.csv"
    _sample_df().write_csv(csv_path)

    lf = scan_dataframe(csv_path)

    assert isinstance(lf, pl.LazyFrame)
    assert row_count(lf) == 4


def test_preview_rows_replaces_binary_values_with_placeholder(tmp_path):
    # Raw bytes aren't JSON-safe across the pywebview js_api bridge — a
    # Binary column (e.g. an image-as-bytes column, Phase 3) must not put
    # bytes objects into preview_rows.
    df = pl.DataFrame({"id": [1, 2], "blob": [b"\xff\xd8\xff\x00abc", None]})
    parquet_path = tmp_path / "sample.parquet"
    df.write_parquet(parquet_path)

    dataset_info = dataset_info_from_lazy(pl.scan_parquet(parquet_path), parquet_path)

    assert dataset_info.preview_rows[0]["blob"] == "<binary: 7 bytes>"
    assert dataset_info.preview_rows[1]["blob"] is None


def test_dataset_info_from_lazy_matches_load_dataset(tmp_path):
    csv_path = tmp_path / "sample.csv"
    _sample_df().write_csv(csv_path)

    lf = scan_dataframe(csv_path)
    via_helper = dataset_info_from_lazy(lf, csv_path)
    via_load_dataset = load_dataset(csv_path)

    assert via_helper.to_dict() == via_load_dataset.to_dict()


def test_parquet_null_counts_come_from_metadata_not_a_decode(tmp_path, monkeypatch):
    """The whole point of the fix: a parquet file whose columns all have
    footer statistics must never build a decode-based null_count_expr —
    proves the metadata path is actually taken, not just correct by
    coincidence."""
    parquet_path = tmp_path / "sample.parquet"
    _sample_df().write_parquet(parquet_path)  # pyarrow writes stats by default

    def _fail_if_called(name, dtype):
        raise AssertionError(f"decode-based null count fallback used for column {name!r}")

    monkeypatch.setattr(data_loader_module, "_null_count_expr", _fail_if_called)

    dataset_info = load_dataset(parquet_path)

    columns_by_name = {c.name: c for c in dataset_info.columns}
    assert columns_by_name["label"].null_count == 1
    assert columns_by_name["id"].null_count == 0


def test_parquet_null_counts_still_count_nan_for_float_columns_via_metadata_path(tmp_path):
    df = pl.DataFrame({"score": [1.0, None, float("nan"), 2.0]})
    parquet_path = tmp_path / "sample.parquet"
    df.write_parquet(parquet_path)

    dataset_info = load_dataset(parquet_path)

    columns_by_name = {c.name: c for c in dataset_info.columns}
    # 1 real null (from metadata) + 1 NaN (from the small float-only
    # streaming aggregation) = 2, same result as the old fully-decoded path.
    assert columns_by_name["score"].null_count == 2


def test_parquet_null_counts_sum_across_multiple_files(tmp_path):
    file1 = tmp_path / "part1.parquet"
    file2 = tmp_path / "part2.parquet"
    pl.DataFrame({"id": [1, 2, None]}).write_parquet(file1)
    pl.DataFrame({"id": [None, None, 5]}).write_parquet(file2)

    lf = pl.concat([pl.scan_parquet(file1), pl.scan_parquet(file2)], how="vertical")
    dataset_info = dataset_info_from_lazy(lf, tmp_path, files=[file1, file2])

    columns_by_name = {c.name: c for c in dataset_info.columns}
    assert columns_by_name["id"].null_count == 3
    assert dataset_info.row_count == 6


def test_parquet_null_counts_fall_back_per_column_when_stats_missing(tmp_path):
    """Parquet statistics are optional per the spec. A file written
    without them must still get a correct null count for that column, via
    the decode-based fallback — scoped to just that column, not the whole
    file (verified indirectly: the untouched-by-fallback column's count is
    also still correct, meaning it came from metadata as normal)."""
    table = pa.table({"no_stats": [1, None, 3], "has_stats": [None, 2, 3]})
    parquet_path = tmp_path / "sample.parquet"
    pq.write_table(table, parquet_path, column_encoding=None, use_dictionary=False, write_statistics=["has_stats"])

    dataset_info = load_dataset(parquet_path)

    columns_by_name = {c.name: c for c in dataset_info.columns}
    assert columns_by_name["no_stats"].null_count == 1
    assert columns_by_name["has_stats"].null_count == 1


def test_non_parquet_multi_file_null_counts_still_use_decode_path(tmp_path):
    """files= is given but they're not parquet (arrow) — the metadata path
    must not be attempted; falls back to the pre-existing decode behavior
    for every column, unchanged."""
    file1 = tmp_path / "part1.arrow"
    file2 = tmp_path / "part2.arrow"
    pl.DataFrame({"id": [1, 2, None]}).write_ipc(file1)
    pl.DataFrame({"id": [None, None, 5]}).write_ipc(file2)

    lf = pl.concat([pl.scan_ipc(file1), pl.scan_ipc(file2)], how="vertical")
    dataset_info = dataset_info_from_lazy(lf, tmp_path, files=[file1, file2])

    columns_by_name = {c.name: c for c in dataset_info.columns}
    assert columns_by_name["id"].null_count == 3


def _write_parts(tmp_path, count):
    files = []
    for i in range(count):
        path = tmp_path / f"part{i}.parquet"
        pl.DataFrame({"id": [1, None, 3], "s": ["a", "b", None]}).write_parquet(path)
        files.append(path)
    return files


def test_null_counts_reports_each_file_as_it_completes(tmp_path):
    files = _write_parts(tmp_path, 4)
    schema = pl.scan_parquet(files[0]).collect_schema()
    seen = []

    data_loader_module._null_counts(
        pl.scan_parquet(files[0]), schema, files=files, on_file=lambda i, p: seen.append((i, p))
    )

    # One callback per file, in order, AFTER that file is fully counted.
    assert [i for i, _ in seen] == [0, 1, 2, 3]
    assert [p for _, p in seen] == files


def test_null_counts_stops_early_when_cancelled_and_counts_only_finished_files(tmp_path):
    files = _write_parts(tmp_path, 4)
    schema = pl.scan_parquet(files[0]).collect_schema()
    done = []

    # Cancel once two files are complete.
    counts = data_loader_module._null_counts(
        pl.scan_parquet(files[0]),
        schema,
        files=files,
        on_file=lambda i, p: done.append(i),
        should_cancel=lambda: len(done) >= 2,
    )

    assert done == [0, 1]
    # 1 null per column per file x 2 completed files — NOT all 4.
    assert counts["id"] == 2
    assert counts["s"] == 2


def test_dataset_info_from_lazy_accepts_precomputed_null_counts(tmp_path):
    parquet_path = tmp_path / "sample.parquet"
    _sample_df().write_parquet(parquet_path)

    # Deliberately wrong values come straight through, proving they are
    # reused rather than recomputed (same contract as total_rows).
    info = dataset_info_from_lazy(
        pl.scan_parquet(parquet_path),
        parquet_path,
        files=[parquet_path],
        null_counts={"id": 111, "label": 222, "score": 333},
    )

    assert {c.name: c.null_count for c in info.columns} == {"id": 111, "label": 222, "score": 333}


def test_dataset_info_from_lazy_reuses_a_precomputed_total_rows(tmp_path):
    csv_path = tmp_path / "sample.csv"
    _sample_df().write_csv(csv_path)
    lf = scan_dataframe(csv_path)

    # A wrong precomputed value should come straight through unchecked —
    # proves the function trusts the caller's total_rows instead of
    # recomputing it when one is given.
    dataset_info = dataset_info_from_lazy(lf, csv_path, total_rows=999)

    assert dataset_info.row_count == 999


# --- filtered profiles (charts following a search filter) ---


def test_null_counts_under_a_filter_describe_only_matching_rows(tmp_path):
    """Parquet footer null counts describe the WHOLE file, so they are
    invalid once a filter is applied — this pins that the filtered path is
    used instead of the metadata shortcut."""
    path = tmp_path / "f.parquet"
    pl.DataFrame(
        {"keep": [1, 1, 0, 0], "v": [None, 5.0, None, None]}
    ).write_parquet(path)

    counts = data_loader_module._null_counts(
        pl.scan_parquet(path),
        pl.scan_parquet(path).collect_schema(),
        files=[path],
        predicate=pl.col("keep") == 1,
    )

    assert counts["v"] == 1  # only the matching rows, not the file's 3
    assert counts["keep"] == 0


def test_null_counts_under_a_filter_sum_across_files(tmp_path):
    files = []
    for i in range(3):
        path = tmp_path / f"p{i}.parquet"
        pl.DataFrame({"keep": [1, 0], "v": [None, None]}).write_parquet(path)
        files.append(path)

    counts = data_loader_module._null_counts(
        pl.scan_parquet(files[0]),
        pl.scan_parquet(files[0]).collect_schema(),
        files=files,
        predicate=pl.col("keep") == 1,
    )

    assert counts["v"] == 3  # one matching null row per file


def test_dataset_info_under_a_filter_reports_filtered_rows_and_nulls(tmp_path):
    path = tmp_path / "d.parquet"
    pl.DataFrame({"keep": [1, 1, 0], "v": [None, 2.0, None]}).write_parquet(path)

    info = dataset_info_from_lazy(
        pl.scan_parquet(path), path, files=[path], predicate=pl.col("keep") == 1
    )

    assert info.row_count == 2
    assert {c.name: c.null_count for c in info.columns}["v"] == 1
    assert len(info.preview_rows) == 2  # preview shows matching rows only
