import polars as pl

from ml_data_view.null_patterns import correlated_missingness


def _null_counts(df):
    return {name: int(df[name].is_null().sum()) for name in df.columns}


def test_detects_columns_that_are_null_on_the_same_rows():
    mask = [i % 5 == 0 for i in range(200)]
    df = pl.DataFrame(
        {
            "a": [None if m else 1.0 for m in mask],
            "b": [None if m else 2.0 for m in mask],
            "unrelated": [3.0] * 200,
        }
    )

    groups = correlated_missingness(df.lazy(), _null_counts(df), total_rows=200)

    assert len(groups) == 1
    assert set(groups[0]["columns"]) == {"a", "b"}
    assert groups[0]["null_count"] == 40
    assert groups[0]["overlap"] == 1.0


def test_does_not_flag_similar_null_counts_with_different_null_rows():
    """The cheap first pass groups by null COUNT, so these two look like
    candidates. Only the row-level second pass can tell them apart — this
    is the test that proves the second pass is actually doing something."""
    df = pl.DataFrame(
        {
            "a": [None if i < 40 else 1.0 for i in range(200)],
            "b": [None if i >= 160 else 2.0 for i in range(200)],  # same count, disjoint rows
        }
    )
    counts = _null_counts(df)
    assert counts["a"] == counts["b"] == 40  # identical counts, so pass 1 pairs them

    assert correlated_missingness(df.lazy(), counts, total_rows=200) == []


def test_partially_overlapping_nulls_are_not_reported():
    # 40 nulls each, only half on shared rows — real but not a shared cause.
    df = pl.DataFrame(
        {
            "a": [None if i < 40 else 1.0 for i in range(200)],
            "b": [None if 20 <= i < 60 else 2.0 for i in range(200)],
        }
    )

    assert correlated_missingness(df.lazy(), _null_counts(df), total_rows=200) == []


def test_groups_three_columns_sharing_one_pattern():
    mask = [i % 4 == 0 for i in range(120)]
    df = pl.DataFrame(
        {name: [None if m else 1.0 for m in mask] for name in ("x", "y", "z")}
        | {"other": list(range(120))}
    )

    groups = correlated_missingness(df.lazy(), _null_counts(df), total_rows=120)

    assert len(groups) == 1
    assert set(groups[0]["columns"]) == {"x", "y", "z"}


def test_columns_without_nulls_are_never_grouped():
    df = pl.DataFrame({"a": [1.0] * 50, "b": [2.0] * 50})
    assert correlated_missingness(df.lazy(), _null_counts(df), total_rows=50) == []


def test_all_null_columns_are_grouped_together():
    df = pl.DataFrame(
        {
            "a": pl.Series("a", [None] * 30, dtype=pl.Float64),
            "b": pl.Series("b", [None] * 30, dtype=pl.Float64),
        }
    )

    groups = correlated_missingness(df.lazy(), _null_counts(df), total_rows=30)

    assert len(groups) == 1
    assert set(groups[0]["columns"]) == {"a", "b"}


def test_near_equal_null_counts_still_count_as_candidates():
    """Real upstream causes rarely produce EXACTLY equal counts, so the
    first-pass grouping has to tolerate a small difference."""
    df = pl.DataFrame(
        {
            "a": [None if i % 10 == 0 else 1.0 for i in range(1000)],   # 100 nulls
            "b": [None if i % 10 == 0 or i == 3 else 2.0 for i in range(1000)],  # 101 nulls
        }
    )

    groups = correlated_missingness(df.lazy(), _null_counts(df), total_rows=1000)

    assert len(groups) == 1
    assert set(groups[0]["columns"]) == {"a", "b"}


def test_a_single_column_is_never_a_group():
    df = pl.DataFrame({"a": [None if i < 10 else 1.0 for i in range(100)], "b": [1.0] * 100})
    assert correlated_missingness(df.lazy(), _null_counts(df), total_rows=100) == []


def test_empty_dataset_returns_nothing():
    df = pl.DataFrame({"a": pl.Series("a", [], dtype=pl.Float64)})
    assert correlated_missingness(df.lazy(), {"a": 0}, total_rows=0) == []
