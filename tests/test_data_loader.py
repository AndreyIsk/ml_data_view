import polars as pl
import pytest

from ml_data_view.data_loader import (
    ColumnInfo,
    DatasetInfo,
    UnsupportedFileTypeError,
    load_dataset,
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
