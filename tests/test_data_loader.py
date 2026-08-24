from ml_data_view.data_loader import ColumnInfo, DatasetInfo


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
