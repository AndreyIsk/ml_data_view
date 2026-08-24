import polars as pl

from ml_data_view.api import Api


def test_load_file_returns_dataset_dict_for_valid_file(tmp_path):
    df = pl.DataFrame({"id": [1, 2], "label": ["a", None]})
    csv_path = tmp_path / "sample.csv"
    df.write_csv(csv_path)

    api = Api()
    result = api.load_file(str(csv_path))

    assert "error" not in result
    assert result["row_count"] == 2
    assert result["columns"][0]["name"] == "id"
    assert result["preview_rows"] == [{"id": 1, "label": "a"}, {"id": 2, "label": None}]


def test_load_file_returns_error_dict_for_unsupported_extension(tmp_path):
    bad_path = tmp_path / "sample.txt"
    bad_path.write_text("not a dataset")

    api = Api()
    result = api.load_file(str(bad_path))

    assert "error" in result


def test_load_file_returns_error_dict_for_missing_file(tmp_path):
    missing_path = tmp_path / "does_not_exist.csv"

    api = Api()
    result = api.load_file(str(missing_path))

    assert "error" in result
