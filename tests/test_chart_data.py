import polars as pl

from ml_data_view.chart_data import categorical_bar_data, classify_column, numeric_histogram_data


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
