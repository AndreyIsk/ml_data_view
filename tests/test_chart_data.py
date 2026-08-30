import polars as pl
import pytest

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
    assert classify_column(df.lazy(), "i") == "numeric"
    assert classify_column(df.lazy(), "f") == "numeric"


def test_classify_column_boolean_is_categorical():
    df = pl.DataFrame({"flag": [True, False, True]})
    assert classify_column(df.lazy(), "flag") == "categorical"


def test_classify_column_low_cardinality_string_is_categorical():
    df = pl.DataFrame({"label": ["cat", "dog", "cat", "dog", "cat"]})
    assert classify_column(df.lazy(), "label") == "categorical"


def test_classify_column_high_cardinality_string_is_text():
    # 20 distinct long strings out of 20 rows: distinct-count 20 > 50? no,
    # but distinct/row ratio is 1.0 > 0.5 and count 20 <= 50 would normally
    # say categorical, so use enough distinct rows to exceed both thresholds.
    values = [f"a fairly unique free-text sentence number {i}" for i in range(60)]
    df = pl.DataFrame({"comment": values})
    assert classify_column(df.lazy(), "comment") == "text"


def test_classify_column_high_cardinality_but_repeating_is_categorical():
    """Many distinct values, but each repeated — the shape of a hashed
    categorical feature (e.g. Criteo's categorical_feature_*), where
    distinct count is far above CATEGORICAL_MAX_DISTINCT yet the
    distinct/row ratio stays well under CATEGORICAL_MAX_DISTINCT_RATIO.

    Pins the behavior that distinguishes this from free text, because the
    distinct count is estimated (approx_n_unique) rather than exact: a
    sample-based estimate would report a near-1.0 ratio here and
    misclassify the column as "text", changing which chart it gets.
    """
    values = [f"hash_{i % 200}" for i in range(600)]  # 200 distinct / 600 rows = 0.33
    df = pl.DataFrame({"feature": values})
    assert classify_column(df.lazy(), "feature") == "categorical"


def test_classify_column_per_file_matches_single_frame_for_categorical(tmp_path):
    """With `files`, classification runs per file and combines ratios
    instead of querying the multi-file concat (which grows ~28-42MB per
    file — measured; 100 files exceeded a 2GB cap on one column). The
    verdict must still match the concat's for a clearly categorical
    column."""
    file1 = tmp_path / "p1.parquet"
    file2 = tmp_path / "p2.parquet"
    pl.DataFrame({"feature": [f"hash_{i % 200}" for i in range(600)]}).write_parquet(file1)
    pl.DataFrame({"feature": [f"hash_{i % 200}" for i in range(600)]}).write_parquet(file2)

    lf = pl.concat([pl.scan_parquet(file1), pl.scan_parquet(file2)], how="vertical")

    assert classify_column(lf, "feature", files=[file1, file2]) == "categorical"
    assert classify_column(lf, "feature") == "categorical"  # concat path agrees


def test_classify_column_per_file_still_detects_free_text(tmp_path):
    """Free text sits near a 1.0 distinct/row ratio at ANY scale, which is
    what makes the per-file ratio a valid discriminator — this pins that
    the bounded path doesn't collapse everything to "categorical"."""
    file1 = tmp_path / "t1.parquet"
    file2 = tmp_path / "t2.parquet"
    pl.DataFrame({"c": [f"a unique sentence number {i}" for i in range(300)]}).write_parquet(file1)
    pl.DataFrame({"c": [f"another unique sentence {i}" for i in range(300)]}).write_parquet(file2)

    lf = pl.concat([pl.scan_parquet(file1), pl.scan_parquet(file2)], how="vertical")

    assert classify_column(lf, "c", files=[file1, file2]) == "text"


def test_classify_column_per_file_ignores_files_for_non_string_dtypes(tmp_path):
    parquet_path = tmp_path / "n.parquet"
    pl.DataFrame({"i": [1, 2, 3], "f": [1.0, 2.0, 3.0]}).write_parquet(parquet_path)
    lf = pl.scan_parquet(parquet_path)

    assert classify_column(lf, "i", files=[parquet_path]) == "numeric"
    assert classify_column(lf, "f", files=[parquet_path]) == "numeric"


def test_classify_column_all_null_string_is_categorical():
    df = pl.DataFrame({"empty": pl.Series("empty", [None, None], dtype=pl.String)})
    assert classify_column(df.lazy(), "empty") == "categorical"


def test_numeric_histogram_data_bins_and_sums_to_row_count():
    df = pl.DataFrame({"score": [1.0, 2.0, 3.0, 4.0, 5.0]})

    result = numeric_histogram_data(df.lazy(), "score", bins=5)

    assert result["column"] == "score"
    assert len(result["bin_edges"]) == 6
    assert len(result["counts"]) == 5
    assert sum(result["counts"]) == 5


def test_numeric_histogram_data_ignores_nulls():
    df = pl.DataFrame({"score": [1.0, None, 3.0, None, 5.0]})

    result = numeric_histogram_data(df.lazy(), "score", bins=3)

    assert sum(result["counts"]) == 3


def test_numeric_histogram_data_empty_after_dropping_nulls():
    df = pl.DataFrame({"score": pl.Series("score", [None, None], dtype=pl.Float64)})

    result = numeric_histogram_data(df.lazy(), "score")

    assert result == {"column": "score", "bin_edges": [], "counts": [], "x_scale": "linear"}


def test_numeric_histogram_data_all_identical_values():
    df = pl.DataFrame({"score": [7.0, 7.0, 7.0]})

    result = numeric_histogram_data(df.lazy(), "score", bins=5)

    assert sum(result["counts"]) == 3


def test_numeric_histogram_data_excludes_nan_values():
    df = pl.DataFrame({"score": [1.0, 2.0, float("nan"), 3.0]})

    result = numeric_histogram_data(df.lazy(), "score", bins=3)

    assert sum(result["counts"]) == 3


def test_numeric_histogram_data_does_not_crash_when_nan_sorts_first():
    # A NaN value earlier in iteration order breaks naive min()/max() (NaN
    # compares unequal to everything), so this must not raise or produce
    # NaN bin edges.
    df = pl.DataFrame({"score": [float("nan"), 0.5, 1.2, 3.0]})

    result = numeric_histogram_data(df.lazy(), "score", bins=3)

    assert sum(result["counts"]) == 3
    assert all(edge == edge for edge in result["bin_edges"])  # no NaN edges


def test_numeric_histogram_data_correct_at_large_scale():
    # Exercises the polars-native binning path (Series.hist()) at a size
    # where the old pure-Python loop implementation was measurably slow
    # (~0.6s/column at 5M rows; kept smaller here to keep the suite fast).
    n = 200_000
    df = pl.DataFrame({"x": [float(i) for i in range(n)]})

    result = numeric_histogram_data(df.lazy(), "x", bins=30)

    assert len(result["counts"]) == 30
    assert len(result["bin_edges"]) == 31
    assert sum(result["counts"]) == n
    assert result["bin_edges"][0] == 0.0
    assert result["bin_edges"][-1] == pytest.approx(n - 1, rel=1e-6)


def test_categorical_bar_data_sorted_by_count_descending():
    df = pl.DataFrame({"label": ["cat", "dog", "cat", "bird", "cat", "dog"]})

    result = categorical_bar_data(df.lazy(), "label")

    assert result["column"] == "label"
    assert result["categories"] == ["cat", "dog", "bird"]
    assert result["counts"] == [3, 2, 1]


def test_categorical_bar_data_groups_rest_as_other_beyond_top_n():
    labels = []
    for i in range(15):
        labels += [f"cat_{i}"] * (15 - i)  # cat_0 most frequent, cat_14 least
    df = pl.DataFrame({"label": labels})

    result = categorical_bar_data(df.lazy(), "label", top_n=10)

    assert len(result["categories"]) == 11  # top 10 + "Other"
    assert result["categories"][-1] == "Other"
    top_10_total = sum(15 - i for i in range(10))
    rest_total = sum(15 - i for i in range(10, 15))
    assert result["counts"][:10] == [15 - i for i in range(10)]
    assert result["counts"][-1] == rest_total
    assert sum(result["counts"]) == top_10_total + rest_total


def test_categorical_bar_data_ignores_nulls():
    df = pl.DataFrame({"label": ["a", None, "a", None]})

    result = categorical_bar_data(df.lazy(), "label")

    assert result["categories"] == ["a"]
    assert result["counts"] == [2]


def test_categorical_bar_data_boolean_column():
    df = pl.DataFrame({"flag": [True, True, False]})

    result = categorical_bar_data(df.lazy(), "flag")

    assert set(result["categories"]) == {"True", "False"}
    assert sum(result["counts"]) == 3


def test_text_length_histogram_data_bins_by_character_length():
    df = pl.DataFrame({"comment": ["a", "bb", "ccc", "dddd", "eeeee"]})

    result = text_length_histogram_data(df.lazy(), "comment", bins=5)

    assert result["column"] == "comment"
    assert len(result["counts"]) == 5
    assert sum(result["counts"]) == 5


def test_text_length_histogram_data_ignores_nulls():
    df = pl.DataFrame({"comment": ["hello", None, "hi"]})

    result = text_length_histogram_data(df.lazy(), "comment", bins=3)

    assert sum(result["counts"]) == 2


def test_text_length_histogram_data_empty_column():
    df = pl.DataFrame({"comment": pl.Series("comment", [], dtype=pl.String)})

    result = text_length_histogram_data(df.lazy(), "comment")

    assert result == {"column": "comment", "bin_edges": [], "counts": []}


def test_null_heatmap_data_marks_nulls_as_one():
    df = pl.DataFrame({"a": [1, None, 3], "b": [None, 2, 3]})

    result = null_heatmap_data(df.lazy())

    assert result["columns"] == ["a", "b"]
    assert result["total_rows"] == 3
    assert result["sampled_rows"] == 3
    assert result["z"] == [[0, 1], [1, 0], [0, 0]]


def test_null_heatmap_data_marks_nan_as_missing_too():
    df = pl.DataFrame({"a": [1.0, float("nan"), 3.0]})

    result = null_heatmap_data(df.lazy())

    assert result["z"] == [[0], [1], [0]]


def test_null_heatmap_data_caps_sampled_rows():
    df = pl.DataFrame({"a": list(range(700))})

    result = null_heatmap_data(df.lazy(), max_rows=500)

    assert result["total_rows"] == 700
    assert result["sampled_rows"] == 500


def test_null_heatmap_data_bins_display_rows_below_plotly_rasterizer_limit():
    # Plotly's heatmap trace silently renders a 100%-transparent image once
    # the z-matrix has more rows than the plot has pixels of height (empirically
    # confirmed: works at 150 rows / 230px height, produces zero opaque pixels
    # at 230+ rows / 230px height). Capping display rows well below any
    # realistic chart pixel height avoids ever hitting that cliff.
    df = pl.DataFrame({"a": list(range(700))})

    result = null_heatmap_data(df.lazy(), max_rows=500, max_display_rows=100)

    assert result["sampled_rows"] == 500
    assert result["display_rows"] == 100
    assert len(result["z"]) == 100


def test_null_heatmap_data_bin_values_are_fraction_missing_not_binary():
    # 10 source rows binned into 2 display rows (5 source rows per bin);
    # column "a" has 1 missing out of 5 in the first bin -> 0.2, none in the
    # second bin -> 0.0.
    df = pl.DataFrame({"a": [None, 1, 2, 3, 4, 5, 6, 7, 8, 9]})

    result = null_heatmap_data(df.lazy(), max_display_rows=2)

    assert result["display_rows"] == 2
    assert result["z"] == [[0.2], [0.0]]


def test_null_heatmap_data_no_binning_when_rows_fit_within_display_cap():
    df = pl.DataFrame({"a": [1, None, 3]})

    result = null_heatmap_data(df.lazy(), max_display_rows=100)

    assert result["display_rows"] == 3
    assert result["z"] == [[0], [1], [0]]


def test_null_heatmap_data_no_cap_needed_when_under_limit():
    df = pl.DataFrame({"a": [1, 2, 3]})

    result = null_heatmap_data(df.lazy(), max_rows=500)

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

    payload = build_chart_payload(df.lazy())

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

    payload = build_chart_payload(df.lazy())

    numeric_columns = [h["column"] for h in payload["numeric_histograms"]]
    categorical_columns = [b["column"] for b in payload["categorical_bars"]]
    text_columns = [t["column"] for t in payload["text_length_histograms"]]
    assert "created_at" not in numeric_columns + categorical_columns + text_columns
    assert "created_at" in payload["null_heatmap"]["columns"]


def test_build_chart_payload_includes_embedding_projection_for_detected_column():
    df = pl.DataFrame(
        {
            "vec": [[float(row * 16 + d) for d in range(16)] for row in range(4)],
            "label": ["cat", "dog", "cat", "dog"],
        },
        schema={"vec": pl.List(pl.Float64), "label": pl.String},
    )

    payload = build_chart_payload(df.lazy())

    assert payload["embedding_columns"] == [{"column": "vec", "dimensions": 16}]
    assert len(payload["embedding_projections"]) == 1
    projection = payload["embedding_projections"][0]
    assert projection["column"] == "vec"
    assert len(projection["x"]) == 4
    assert "label" in projection["categorical_colors"]


def test_build_chart_payload_embedding_lists_empty_without_embedding_column():
    df = pl.DataFrame({"id": [1, 2, 3], "label": ["cat", "dog", "cat"]})

    payload = build_chart_payload(df.lazy())

    assert payload["embedding_columns"] == []
    assert payload["embedding_projections"] == []


def _png_bytes() -> bytes:
    import io

    from PIL import Image

    buffer = io.BytesIO()
    Image.new("RGB", (4, 4), (255, 0, 0)).save(buffer, format="PNG")
    return buffer.getvalue()


def test_build_chart_payload_includes_image_preview_for_detected_column():
    df = pl.DataFrame({"thumb": [_png_bytes(), _png_bytes()]}, schema={"thumb": pl.Binary})

    payload = build_chart_payload(df.lazy())

    assert payload["image_columns"] == [{"column": "thumb", "source": "bytes"}]
    assert len(payload["image_previews"]) == 1
    assert payload["image_previews"][0]["column"] == "thumb"
    assert len(payload["image_previews"][0]["rows"]) == 2


def test_build_chart_payload_image_lists_empty_without_image_column():
    df = pl.DataFrame({"id": [1, 2, 3], "label": ["cat", "dog", "cat"]})

    payload = build_chart_payload(df.lazy())

    assert payload["image_columns"] == []
    assert payload["image_previews"] == []


def test_build_chart_payload_resolves_path_image_columns_against_dataset_dir(tmp_path):
    (tmp_path / "images").mkdir()
    (tmp_path / "images" / "cat.png").write_bytes(_png_bytes())
    df = pl.DataFrame({"path": ["images/cat.png"]})

    payload = build_chart_payload(df.lazy(), dataset_dir=tmp_path)

    assert payload["image_columns"] == [{"column": "path", "source": "path"}]
    assert payload["image_previews"][0]["rows"][0]["thumbnail"] is not None


# --- log-scale histograms for skewed numeric columns ---


def test_skewed_column_is_flagged_and_binned_on_a_log_scale():
    # Long right tail: 99% of rows near zero, a few very large — exactly
    # the shape that crams every row into one bar on a linear axis.
    values = [float(i % 3) for i in range(9900)] + [float(500_000 + i) for i in range(100)]
    df = pl.DataFrame({"skewed": values})

    chart = numeric_histogram_data(df.lazy(), "skewed")

    assert chart["x_scale"] == "log"
    assert sum(chart["counts"]) == 10_000
    assert len(chart["bin_edges"]) == len(chart["counts"]) + 1
    # Log binning must spread the mass out instead of dumping it in bin 0.
    assert chart["counts"][0] < 9900


def test_normally_distributed_column_stays_linear():
    import random

    random.seed(7)
    values = [random.gauss(100.0, 15.0) for _ in range(5000)]
    df = pl.DataFrame({"normal": values})

    chart = numeric_histogram_data(df.lazy(), "normal")

    assert chart["x_scale"] == "linear"


def test_log_scale_handles_zero_and_negative_values(tmp_path):
    """log1p(x) breaks below -1 and a raw log breaks at 0, so the offset
    has to be applied before the transform, not after."""
    values = [-50.0, -1.0, 0.0] + [0.0] * 4000 + [float(1_000_000)] * 20
    df = pl.DataFrame({"mixed": values})

    chart = numeric_histogram_data(df.lazy(), "mixed")

    assert chart["x_scale"] == "log"
    assert sum(chart["counts"]) == len(values)
    assert all(isinstance(e, float) and e == e for e in chart["bin_edges"])  # no NaN
    assert chart["bin_edges"][0] <= -50.0 + 1e-6  # edges reported in ORIGINAL units


def test_constant_column_is_not_flagged_as_skewed():
    df = pl.DataFrame({"flat": [5.0] * 100})
    assert numeric_histogram_data(df.lazy(), "flat")["x_scale"] == "linear"


def test_all_null_numeric_column_still_returns_empty_chart():
    df = pl.DataFrame({"n": pl.Series("n", [None, None], dtype=pl.Float64)})
    chart = numeric_histogram_data(df.lazy(), "n")
    assert chart["counts"] == []
    assert chart["x_scale"] == "linear"


# --- value-frequency ("detections per value") histogram for ID-like columns ---


def test_value_frequency_histogram_counts_values_by_how_often_they_recur():
    from ml_data_view.chart_data import value_frequency_histogram_data

    # 5 values seen once, 1 seen twice, 1 seen three times.
    df = pl.DataFrame({"track": ["A", "A", "A", "B", "B", "C", "D", "E", "F", "G"]})

    result = value_frequency_histogram_data(df.lazy(), "track")

    by_bucket = dict(zip(result["buckets"], result["counts"]))
    assert by_bucket["1"] == 5      # C, D, E, F, G
    assert by_bucket["2-3"] == 2    # B (2) and A (3)
    assert sum(result["counts"]) == 7  # 7 distinct values total
    assert result["per_file"] is False


def test_value_frequency_histogram_is_bucketed_across_magnitudes():
    from ml_data_view.chart_data import value_frequency_histogram_data

    values = ["once"] + ["mid"] * 20 + ["heavy"] * 500
    df = pl.DataFrame({"track": values})

    by_bucket = dict(
        zip(*[value_frequency_histogram_data(df.lazy(), "track")[k] for k in ("buckets", "counts")])
    )

    assert by_bucket["1"] == 1
    assert by_bucket["11-50"] == 1
    assert by_bucket["201-1000"] == 1


def test_value_frequency_histogram_per_file_sums_buckets(tmp_path):
    from ml_data_view.chart_data import value_frequency_histogram_data

    f1 = tmp_path / "a.parquet"
    f2 = tmp_path / "b.parquet"
    pl.DataFrame({"track": ["A", "B", "C"]}).write_parquet(f1)
    pl.DataFrame({"track": ["D", "E", "F"]}).write_parquet(f2)
    lf = pl.concat([pl.scan_parquet(f1), pl.scan_parquet(f2)], how="vertical")

    result = value_frequency_histogram_data(lf, "track", files=[f1, f2])

    assert dict(zip(result["buckets"], result["counts"]))["1"] == 6
    assert result["per_file"] is True


def test_high_cardinality_column_gets_a_frequency_histogram_not_a_bar():
    # >CATEGORICAL_BAR_MAX_DISTINCT distinct values, each recurring — the
    # track-ID shape. A top-10 bar of these names arbitrary IDs.
    values = [f"track_{i % 500}" for i in range(5000)]
    df = pl.DataFrame({"track": values, "kind": ["a", "b"] * 2500})

    payload = build_chart_payload(df.lazy())

    charted = {c["column"] for c in payload["value_frequency_histograms"]}
    barred = {c["column"] for c in payload["categorical_bars"]}
    assert "track" in charted
    assert "track" not in barred
    assert "kind" in barred  # 2 distinct — a bar is still the right chart


def test_categorical_bar_per_file_matches_the_single_frame_result(tmp_path):
    """The bounded per-file top-K path must agree with the exact
    single-frame computation on data where both can run."""
    values = [f"c{i % 5}" for i in range(300)]
    f1 = tmp_path / "p1.parquet"
    f2 = tmp_path / "p2.parquet"
    pl.DataFrame({"label": values}).write_parquet(f1)
    pl.DataFrame({"label": values}).write_parquet(f2)
    lf = pl.concat([pl.scan_parquet(f1), pl.scan_parquet(f2)], how="vertical")

    per_file = categorical_bar_data(lf, "label", files=[f1, f2])
    single = categorical_bar_data(lf, "label")

    assert dict(zip(per_file["categories"], per_file["counts"])) == dict(
        zip(single["categories"], single["counts"])
    )


def test_numeric_histogram_per_file_is_exact_not_approximate(tmp_path):
    """Fixed bin edges make per-file histograms summable, so the folder
    path must produce EXACTLY the single-frame counts \u2014 unlike the
    cardinality-based paths, this one involves no approximation."""
    f1 = tmp_path / "n1.parquet"
    f2 = tmp_path / "n2.parquet"
    pl.DataFrame({"x": [float(i) for i in range(500)]}).write_parquet(f1)
    pl.DataFrame({"x": [float(i) for i in range(500, 1000)]}).write_parquet(f2)
    lf = pl.concat([pl.scan_parquet(f1), pl.scan_parquet(f2)], how="vertical")

    per_file = numeric_histogram_data(lf, "x", bins=10, files=[f1, f2])
    single = numeric_histogram_data(lf, "x", bins=10)

    assert sum(per_file["counts"]) == 1000
    assert per_file["counts"] == single["counts"]
    assert per_file["bin_edges"] == pytest.approx(single["bin_edges"], rel=1e-9)


def test_numeric_histogram_per_file_detects_skew_and_stays_exact(tmp_path):
    f1 = tmp_path / "s1.parquet"
    f2 = tmp_path / "s2.parquet"
    pl.DataFrame({"x": [float(i % 3) for i in range(2000)]}).write_parquet(f1)
    pl.DataFrame({"x": [float(500_000 + i) for i in range(40)]}).write_parquet(f2)
    lf = pl.concat([pl.scan_parquet(f1), pl.scan_parquet(f2)], how="vertical")

    result = numeric_histogram_data(lf, "x", files=[f1, f2])

    assert result["x_scale"] == "log"
    assert sum(result["counts"]) == 2040  # every row accounted for


def test_correlated_missingness_per_file_matches_the_concat(tmp_path):
    from ml_data_view.null_patterns import correlated_missingness

    mask = [i % 4 == 0 for i in range(200)]
    frame = pl.DataFrame(
        {"a": [None if m else 1.0 for m in mask], "b": [None if m else 2.0 for m in mask]}
    )
    f1 = tmp_path / "c1.parquet"
    f2 = tmp_path / "c2.parquet"
    frame.write_parquet(f1)
    frame.write_parquet(f2)
    lf = pl.concat([pl.scan_parquet(f1), pl.scan_parquet(f2)], how="vertical")
    counts = {"a": 100, "b": 100}

    per_file = correlated_missingness(lf, counts, 400, files=[f1, f2])
    concat = correlated_missingness(lf, counts, 400)

    assert per_file == concat
    assert set(per_file[0]["columns"]) == {"a", "b"}
