import polars as pl

from ml_data_view.chart_data import (
    build_chart_payload,
    categorical_bar_data,
    classify_column,
    null_heatmap_data,
    numeric_histogram_data,
    text_length_histogram_data,
)


def test_classify_column_numeric_dtypes():
    df = pl.DataFrame({"i": [1, 2, 3], "f": [1.0, 2.0, 3.0]})
    assert classify_column(df, "i") == "numeric"
    assert classify_column(df, "f") == "numeric"


def test_classify_column_boolean_is_categorical():
    df = pl.DataFrame({"flag": [True, False, True]})
    assert classify_column(df, "flag") == "categorical"


def test_classify_column_low_cardinality_string_is_categorical():
    df = pl.DataFrame({"label": ["cat", "dog", "cat", "dog", "cat"]})
    assert classify_column(df, "label") == "categorical"


def test_classify_column_high_cardinality_string_is_text():
    # 20 distinct long strings out of 20 rows: distinct-count 20 > 50? no,
    # but distinct/row ratio is 1.0 > 0.5 and count 20 <= 50 would normally
    # say categorical, so use enough distinct rows to exceed both thresholds.
    values = [f"a fairly unique free-text sentence number {i}" for i in range(60)]
    df = pl.DataFrame({"comment": values})
    assert classify_column(df, "comment") == "text"


def test_classify_column_all_null_string_is_categorical():
    df = pl.DataFrame({"empty": pl.Series("empty", [None, None], dtype=pl.String)})
    assert classify_column(df, "empty") == "categorical"


def test_numeric_histogram_data_bins_and_sums_to_row_count():
    df = pl.DataFrame({"score": [1.0, 2.0, 3.0, 4.0, 5.0]})

    result = numeric_histogram_data(df, "score", bins=5)

    assert result["column"] == "score"
    assert len(result["bin_edges"]) == 6
    assert len(result["counts"]) == 5
    assert sum(result["counts"]) == 5


def test_numeric_histogram_data_ignores_nulls():
    df = pl.DataFrame({"score": [1.0, None, 3.0, None, 5.0]})

    result = numeric_histogram_data(df, "score", bins=3)

    assert sum(result["counts"]) == 3


def test_numeric_histogram_data_empty_after_dropping_nulls():
    df = pl.DataFrame({"score": pl.Series("score", [None, None], dtype=pl.Float64)})

    result = numeric_histogram_data(df, "score")

    assert result == {"column": "score", "bin_edges": [], "counts": []}


def test_numeric_histogram_data_all_identical_values():
    df = pl.DataFrame({"score": [7.0, 7.0, 7.0]})

    result = numeric_histogram_data(df, "score", bins=5)

    assert sum(result["counts"]) == 3


def test_categorical_bar_data_sorted_by_count_descending():
    df = pl.DataFrame({"label": ["cat", "dog", "cat", "bird", "cat", "dog"]})

    result = categorical_bar_data(df, "label")

    assert result["column"] == "label"
    assert result["categories"] == ["cat", "dog", "bird"]
    assert result["counts"] == [3, 2, 1]


def test_categorical_bar_data_groups_rest_as_other_beyond_top_n():
    labels = []
    for i in range(15):
        labels += [f"cat_{i}"] * (15 - i)  # cat_0 most frequent, cat_14 least
    df = pl.DataFrame({"label": labels})

    result = categorical_bar_data(df, "label", top_n=10)

    assert len(result["categories"]) == 11  # top 10 + "Other"
    assert result["categories"][-1] == "Other"
    top_10_total = sum(15 - i for i in range(10))
    rest_total = sum(15 - i for i in range(10, 15))
    assert result["counts"][:10] == [15 - i for i in range(10)]
    assert result["counts"][-1] == rest_total
    assert sum(result["counts"]) == top_10_total + rest_total


def test_categorical_bar_data_ignores_nulls():
    df = pl.DataFrame({"label": ["a", None, "a", None]})

    result = categorical_bar_data(df, "label")

    assert result["categories"] == ["a"]
    assert result["counts"] == [2]


def test_categorical_bar_data_boolean_column():
    df = pl.DataFrame({"flag": [True, True, False]})

    result = categorical_bar_data(df, "flag")

    assert set(result["categories"]) == {"True", "False"}
    assert sum(result["counts"]) == 3


def test_text_length_histogram_data_bins_by_character_length():
    df = pl.DataFrame({"comment": ["a", "bb", "ccc", "dddd", "eeeee"]})

    result = text_length_histogram_data(df, "comment", bins=5)

    assert result["column"] == "comment"
    assert len(result["counts"]) == 5
    assert sum(result["counts"]) == 5


def test_text_length_histogram_data_ignores_nulls():
    df = pl.DataFrame({"comment": ["hello", None, "hi"]})

    result = text_length_histogram_data(df, "comment", bins=3)

    assert sum(result["counts"]) == 2


def test_text_length_histogram_data_empty_column():
    df = pl.DataFrame({"comment": pl.Series("comment", [], dtype=pl.String)})

    result = text_length_histogram_data(df, "comment")

    assert result == {"column": "comment", "bin_edges": [], "counts": []}


def test_null_heatmap_data_marks_nulls_as_one():
    df = pl.DataFrame({"a": [1, None, 3], "b": [None, 2, 3]})

    result = null_heatmap_data(df)

    assert result["columns"] == ["a", "b"]
    assert result["total_rows"] == 3
    assert result["sampled_rows"] == 3
    assert result["z"] == [[0, 1], [1, 0], [0, 0]]


def test_null_heatmap_data_caps_sampled_rows():
    df = pl.DataFrame({"a": list(range(700))})

    result = null_heatmap_data(df, max_rows=500)

    assert result["total_rows"] == 700
    assert result["sampled_rows"] == 500
    assert len(result["z"]) == 500


def test_null_heatmap_data_no_cap_needed_when_under_limit():
    df = pl.DataFrame({"a": [1, 2, 3]})

    result = null_heatmap_data(df, max_rows=500)

    assert result["sampled_rows"] == 3
    assert result["total_rows"] == 3


def test_build_chart_payload_routes_columns_by_type():
    df = pl.DataFrame(
        {
            "id": [1, 2, 3, 4],
            "label": ["cat", "dog", "cat", "dog"],
            "comment": [f"unique free text entry number {i} of many words" for i in range(4)],
        }
    )

    payload = build_chart_payload(df)

    assert payload["null_heatmap"]["columns"] == ["id", "label", "comment"]
    assert [h["column"] for h in payload["numeric_histograms"]] == ["id"]
    # "comment" has only 4 distinct values, which is <= CATEGORICAL_MAX_DISTINCT
    # (50), so by the documented rule it's still classified as categorical at
    # this tiny sample size — both "label" and "comment" land here.
    assert [b["column"] for b in payload["categorical_bars"]] == ["label", "comment"]
    assert payload["text_length_histograms"] == []


def test_build_chart_payload_skips_unsupported_dtypes():
    df = pl.DataFrame({"id": [1, 2, 3]})
    df = df.with_columns(pl.date(2024, 1, 1).alias("created_at"))

    payload = build_chart_payload(df)

    numeric_columns = [h["column"] for h in payload["numeric_histograms"]]
    categorical_columns = [b["column"] for b in payload["categorical_bars"]]
    text_columns = [t["column"] for t in payload["text_length_histograms"]]
    assert "created_at" not in numeric_columns + categorical_columns + text_columns
    assert "created_at" in payload["null_heatmap"]["columns"]
