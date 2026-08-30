import polars as pl

from ml_data_view.embedding_data import detect_embedding_columns, embedding_projection_data


def test_detect_embedding_columns_finds_consistent_float_list_above_threshold():
    df = pl.DataFrame(
        {"vec": [[float(i) for i in range(16)] for _ in range(3)]},
        schema={"vec": pl.List(pl.Float64)},
    )

    result = detect_embedding_columns(df.lazy())

    assert result == [{"column": "vec", "dimensions": 16}]


def test_detect_embedding_columns_rejects_length_at_threshold():
    df = pl.DataFrame(
        {"vec": [[float(i) for i in range(8)] for _ in range(3)]},
        schema={"vec": pl.List(pl.Float64)},
    )

    assert detect_embedding_columns(df.lazy()) == []


def test_detect_embedding_columns_rejects_inconsistent_lengths():
    df = pl.DataFrame(
        {"vec": [[1.0] * 16, [1.0] * 12, [1.0] * 16]},
        schema={"vec": pl.List(pl.Float64)},
    )

    assert detect_embedding_columns(df.lazy()) == []


def test_detect_embedding_columns_finds_fixed_size_array_above_threshold():
    df = pl.DataFrame(
        {"vec": [[float(i) for i in range(32)] for _ in range(3)]},
        schema={"vec": pl.Array(pl.Float32, 32)},
    )

    result = detect_embedding_columns(df.lazy())

    assert result == [{"column": "vec", "dimensions": 32}]


def test_detect_embedding_columns_rejects_small_fixed_size_array():
    df = pl.DataFrame(
        {"vec": [[1.0, 2.0, 3.0, 4.0] for _ in range(3)]},
        schema={"vec": pl.Array(pl.Float32, 4)},
    )

    assert detect_embedding_columns(df.lazy()) == []


def test_detect_embedding_columns_rejects_integer_lists():
    df = pl.DataFrame(
        {"vec": [[i for i in range(16)] for _ in range(3)]},
        schema={"vec": pl.List(pl.Int64)},
    )

    assert detect_embedding_columns(df.lazy()) == []


def test_detect_embedding_columns_tolerates_null_list_entries():
    df = pl.DataFrame(
        {"vec": [[1.0] * 16, None, [2.0] * 16]},
        schema={"vec": pl.List(pl.Float64)},
    )

    result = detect_embedding_columns(df.lazy())

    assert result == [{"column": "vec", "dimensions": 16}]


def test_detect_embedding_columns_ignores_non_list_columns():
    df = pl.DataFrame({"score": [1.0, 2.0, 3.0], "label": ["a", "b", "c"]})

    assert detect_embedding_columns(df.lazy()) == []


def _embedding_df(n_rows: int, dimensions: int = 16) -> pl.DataFrame:
    return pl.DataFrame(
        {"vec": [[float(row * dimensions + d) for d in range(dimensions)] for row in range(n_rows)]},
        schema={"vec": pl.List(pl.Float64)},
    )


def test_embedding_projection_data_returns_a_2d_point_per_row():
    df = _embedding_df(5)

    result = embedding_projection_data(df.lazy(), "vec", categorical_columns=[])

    assert len(result["x"]) == 5
    assert len(result["y"]) == 5
    assert all(isinstance(v, float) for v in result["x"] + result["y"])
    assert result["is_sample"] is False
    assert result["sampled_rows"] == 5
    assert result["total_rows"] == 5


def test_embedding_projection_data_excludes_rows_with_null_embedding():
    df = _embedding_df(5)
    df = df.with_columns(
        pl.when(pl.int_range(pl.len()) == 2).then(None).otherwise(pl.col("vec")).alias("vec")
    )

    result = embedding_projection_data(df.lazy(), "vec", categorical_columns=[])

    assert len(result["x"]) == 4
    assert len(result["y"]) == 4
    assert result["total_rows"] == 5


def test_embedding_projection_data_samples_when_over_max():
    df = _embedding_df(8)

    result = embedding_projection_data(df.lazy(), "vec", categorical_columns=[], sample_max=5)

    assert result["is_sample"] is True
    assert result["sampled_rows"] == 5
    assert result["total_rows"] == 8
    assert len(result["x"]) == 5


def test_embedding_projection_data_does_not_sample_at_or_below_max():
    df = _embedding_df(5)

    result = embedding_projection_data(df.lazy(), "vec", categorical_columns=[], sample_max=5)

    assert result["is_sample"] is False
    assert result["sampled_rows"] == 5


def test_embedding_projection_data_buckets_categorical_labels_top_n_and_missing():
    # Scatter is an all-pairs chart form (dataviz skill's palette.md): only
    # the first 3 categorical color slots validate colorblind-safety across
    # every pair, so the embedding scatter's color-by caps at top-3 + Other,
    # unlike the class-balance bar chart's top-10 (an adjacent-pairs form).
    categories = ["c0"] * 5 + ["c1"] * 4 + ["c2"] * 3 + ["c3"] * 1 + [None, None]
    df = _embedding_df(len(categories)).with_columns(pl.Series("cat", categories))

    result = embedding_projection_data(df.lazy(), "vec", categorical_columns=["cat"], sample_max=1000)

    labels = result["categorical_colors"]["cat"]
    assert len(labels) == len(categories)
    assert labels.count("Missing") == 2
    assert labels.count("Other") == 1  # the single "c3" row, past the top-3 cap
    assert labels.count("c0") == 5
    assert labels.count("c1") == 4
    assert labels.count("c2") == 3
    assert set(labels) == {"c0", "c1", "c2", "Other", "Missing"}


def test_embedding_projection_data_handles_fewer_than_two_usable_rows():
    df = _embedding_df(1)

    result = embedding_projection_data(df.lazy(), "vec", categorical_columns=[])

    assert result["x"] == []
    assert result["y"] == []
    assert result["total_rows"] == 1
