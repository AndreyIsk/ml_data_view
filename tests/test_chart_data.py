import polars as pl

from ml_data_view.chart_data import classify_column


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
