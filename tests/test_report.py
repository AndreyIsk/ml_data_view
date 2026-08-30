import io

import numpy as np
import polars as pl
from PIL import Image

from ml_data_view.chart_data import build_chart_payload
from ml_data_view.data_loader import dataset_info_from_lazy
from ml_data_view.report import available_sections, generate_report


def _png_bytes(color: tuple[int, int, int] = (255, 0, 0)) -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", (4, 4), color).save(buffer, format="PNG")
    return buffer.getvalue()


def test_available_sections_true_for_overview_and_distributions_on_a_plain_dataset():
    df = pl.DataFrame({"id": [1, 2, 3], "label": ["a", "b", "a"]})

    sections = available_sections(build_chart_payload(df.lazy()))

    assert sections == {
        "overview": True,
        "distributions": True,
        "embeddings": False,
        "media": False,
    }


def test_available_sections_true_for_embeddings_when_an_embedding_column_is_present():
    embeddings = [list(np.random.rand(16)) for _ in range(20)]
    df = pl.DataFrame({"id": list(range(20)), "vec": embeddings, "label": ["a", "b"] * 10})

    sections = available_sections(build_chart_payload(df.lazy()))

    assert sections["embeddings"] is True


def test_available_sections_true_for_media_when_an_image_column_is_present():
    df = pl.DataFrame({"thumb": [_png_bytes()]}, schema={"thumb": pl.Binary})

    sections = available_sections(build_chart_payload(df.lazy()))

    assert sections["media"] is True


def test_generate_report_writes_a_valid_nonempty_pdf(tmp_path):
    df = pl.DataFrame({"id": [1, 2, 3, 4], "label": ["a", "b", "a", "c"]})
    csv_path = tmp_path / "sample.csv"
    df.write_csv(csv_path)
    dataset_info = dataset_info_from_lazy(df.lazy(), csv_path).to_dict()
    chart_data = build_chart_payload(df.lazy())
    output_path = tmp_path / "report.pdf"

    result = generate_report(
        output_path,
        dataset_info=dataset_info,
        chart_data=chart_data,
        sections=["overview", "distributions", "embeddings", "media"],
        generated_at="2026-08-28T12:00:00+00:00",
    )

    assert result == output_path
    assert output_path.stat().st_size > 0
    assert output_path.read_bytes()[:5] == b"%PDF-"


def test_generate_report_reproduces_the_classification_banner_without_error(tmp_path):
    df = pl.DataFrame({"id": [1, 2, 3]})
    csv_path = tmp_path / "sample.csv"
    df.write_csv(csv_path)
    dataset_info = dataset_info_from_lazy(df.lazy(), csv_path).to_dict()
    chart_data = build_chart_payload(df.lazy())
    output_path = tmp_path / "report.pdf"

    generate_report(
        output_path,
        dataset_info=dataset_info,
        chart_data=chart_data,
        sections=["overview"],
        banner={"enabled": True, "label": "INTERNAL USE ONLY", "color": "#eb6834"},
        generated_at="2026-08-28T12:00:00+00:00",
    )

    assert output_path.read_bytes()[:5] == b"%PDF-"
    assert output_path.stat().st_size > 0


def test_generate_report_silently_skips_sections_unavailable_for_this_dataset(tmp_path):
    df = pl.DataFrame({"id": [1, 2, 3]})
    csv_path = tmp_path / "sample.csv"
    df.write_csv(csv_path)
    dataset_info = dataset_info_from_lazy(df.lazy(), csv_path).to_dict()
    chart_data = build_chart_payload(df.lazy())  # no embedding or image column
    output_path = tmp_path / "report.pdf"

    # Requesting embeddings/media even though this dataset has neither
    # should not raise, and should still produce a valid PDF.
    generate_report(
        output_path,
        dataset_info=dataset_info,
        chart_data=chart_data,
        sections=["overview", "distributions", "embeddings", "media"],
        generated_at="2026-08-28T12:00:00+00:00",
    )

    assert output_path.read_bytes()[:5] == b"%PDF-"


def test_generate_report_grows_with_more_sections_included(tmp_path):
    df = pl.DataFrame(
        {"id": list(range(20)), "score": [float(i) for i in range(20)], "label": ["a", "b"] * 10}
    )
    csv_path = tmp_path / "sample.csv"
    df.write_csv(csv_path)
    dataset_info = dataset_info_from_lazy(df.lazy(), csv_path).to_dict()
    chart_data = build_chart_payload(df.lazy())

    overview_only = tmp_path / "overview_only.pdf"
    generate_report(
        overview_only, dataset_info=dataset_info, chart_data=chart_data, sections=["overview"]
    )

    with_distributions = tmp_path / "with_distributions.pdf"
    generate_report(
        with_distributions,
        dataset_info=dataset_info,
        chart_data=chart_data,
        sections=["overview", "distributions"],
    )

    assert with_distributions.stat().st_size > overview_only.stat().st_size


# --- denser grid layout + correlated missingness ---


def _wide_dataset(columns=12, rows=400):
    data = {}
    for i in range(columns):
        data[f"num_{i}"] = [float((n * (i + 1)) % 97) for n in range(rows)]
    return pl.DataFrame(data)


def test_grid_layout_uses_far_fewer_pages_than_one_chart_per_row(tmp_path):
    """The whole point of the change: a wide dataset must not produce a
    report with 1-2 charts per page."""
    from pypdf import PdfReader

    df = _wide_dataset(columns=12)
    lf = df.lazy()
    info = dataset_info_from_lazy(lf, tmp_path / "wide.parquet", file_size_bytes=0).to_dict()
    charts = build_chart_payload(lf)

    out = generate_report(
        tmp_path / "wide.pdf", dataset_info=info, chart_data=charts, sections=["distributions"]
    )

    pages = len(PdfReader(str(out)).pages)
    # 12 charts + heatmap. At the old full-width size this needed ~7 pages;
    # two-up grid must bring it well under that.
    assert pages <= 4, f"expected a dense layout, got {pages} pages for 12 charts"
    assert out.stat().st_size > 1000


def test_generated_pdf_draws_the_header_once_per_page(tmp_path):
    """Header text appears at top AND bottom by design (header + footer),
    so it occurs twice per page — but it must not occur FOUR times, which
    is what a double-drawn header would produce."""
    from pypdf import PdfReader

    df = _wide_dataset(columns=4)
    lf = df.lazy()
    info = dataset_info_from_lazy(lf, tmp_path / "hdr.parquet", file_size_bytes=0).to_dict()
    charts = build_chart_payload(lf)

    out = generate_report(
        tmp_path / "hdr.pdf", dataset_info=info, chart_data=charts, sections=["overview"]
    )

    for page in PdfReader(str(out)).pages:
        text = page.extract_text()
        # "Generated <timestamp>" is drawn only by _draw_header_footer, once
        # at the top and once at the bottom, so exactly 2 per page. Counting
        # the filename instead would count 3 — the report's own subtitle
        # repeats it — which reads as a double-drawn header but is not one.
        assert text.count("Generated ") == 2


def test_report_includes_a_correlated_missingness_section(tmp_path):
    mask = [i % 5 == 0 for i in range(300)]
    df = pl.DataFrame(
        {
            "a": [None if m else 1.0 for m in mask],
            "b": [None if m else 2.0 for m in mask],
            "c": list(range(300)),
        }
    )
    lf = df.lazy()
    info = dataset_info_from_lazy(lf, tmp_path / "corr.parquet", file_size_bytes=0).to_dict()
    null_counts = {c["name"]: c["null_count"] for c in info["columns"]}
    charts = build_chart_payload(lf, null_counts=null_counts)

    assert charts["correlated_missingness"], "detection should have found the a/b pair"

    from pypdf import PdfReader

    out = generate_report(
        tmp_path / "corr.pdf", dataset_info=info, chart_data=charts, sections=["distributions"]
    )
    text = "".join(page.extract_text() for page in PdfReader(str(out)).pages)
    assert "Correlated missingness" in text


def test_log_scale_histogram_renders_into_the_pdf(tmp_path):
    values = [float(i % 3) for i in range(4000)] + [float(900_000 + i) for i in range(60)]
    df = pl.DataFrame({"skewed": values})
    lf = df.lazy()
    info = dataset_info_from_lazy(lf, tmp_path / "log.parquet", file_size_bytes=0).to_dict()
    charts = build_chart_payload(lf)
    assert charts["numeric_histograms"][0]["x_scale"] == "log"

    out = generate_report(
        tmp_path / "log.pdf", dataset_info=info, chart_data=charts, sections=["distributions"]
    )
    assert out.stat().st_size > 1000
