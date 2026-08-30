from ml_data_view.report_charts import (
    categorical_bar_figure,
    embedding_scatter_figure,
    null_heatmap_figure,
    numeric_histogram_figure,
    text_length_histogram_figure,
)


def test_numeric_histogram_figure_returns_none_for_empty_chart():
    assert numeric_histogram_figure({"column": "x", "bin_edges": [], "counts": []}) is None


def test_numeric_histogram_figure_builds_bar_trace_from_bin_centers():
    chart = {"column": "x", "bin_edges": [0.0, 1.0, 2.0], "counts": [3, 5]}

    fig = numeric_histogram_figure(chart)

    assert fig is not None
    assert list(fig.data[0].x) == [0.5, 1.5]
    assert list(fig.data[0].y) == [3, 5]


def test_categorical_bar_figure_returns_none_for_empty_chart():
    assert categorical_bar_figure({"column": "c", "categories": [], "counts": []}) is None


def test_categorical_bar_figure_builds_bar_trace():
    chart = {"column": "c", "categories": ["a", "b"], "counts": [4, 2]}

    fig = categorical_bar_figure(chart)

    assert list(fig.data[0].x) == ["a", "b"]
    assert list(fig.data[0].y) == [4, 2]


def test_text_length_histogram_figure_returns_none_for_empty_chart():
    assert text_length_histogram_figure({"column": "t", "bin_edges": [], "counts": []}) is None


def test_null_heatmap_figure_returns_none_when_no_data():
    assert null_heatmap_figure({"columns": [], "z": [], "display_rows": 0}) is None


def test_null_heatmap_figure_builds_heatmap_trace():
    data = {"columns": ["a", "b"], "z": [[0.0, 1.0]], "display_rows": 1}

    fig = null_heatmap_figure(data)

    assert fig.data[0].type == "heatmap"
    assert list(fig.data[0].x) == ["a", "b"]


def test_embedding_scatter_figure_returns_none_when_no_points():
    assert embedding_scatter_figure({"column": "vec", "x": [], "y": []}) is None


def test_embedding_scatter_figure_builds_scatter_trace():
    projection = {"column": "vec", "x": [1.0, 2.0], "y": [3.0, 4.0]}

    fig = embedding_scatter_figure(projection)

    assert list(fig.data[0].x) == [1.0, 2.0]
    assert list(fig.data[0].y) == [3.0, 4.0]
