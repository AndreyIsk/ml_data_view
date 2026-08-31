import json

import polars as pl

from ml_data_view import load_stats
from ml_data_view.api import Api
from ml_data_view.audit_log import audit_log_file_path
from ml_data_view.settings import DEFAULT_BANNER_COLOR, settings_file_path


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


def test_load_file_caches_lazyframe_for_chart_data(tmp_path):
    df = pl.DataFrame({"id": [1, 2, 3]})
    csv_path = tmp_path / "sample.csv"
    df.write_csv(csv_path)

    api = Api()
    api.load_file(str(csv_path))

    assert api._current_lazy is not None
    assert api._current_row_count == 3


def test_load_file_does_not_cache_lazyframe_on_error(tmp_path):
    bad_path = tmp_path / "sample.txt"
    bad_path.write_text("not a dataset")

    api = Api()
    api.load_file(str(bad_path))

    assert api._current_lazy is None


def test_get_chart_data_returns_error_when_nothing_loaded():
    api = Api()

    result = api.get_chart_data()

    assert "error" in result


def test_get_chart_data_returns_payload_after_load_file(tmp_path):
    df = pl.DataFrame({"id": [1, 2, 3], "label": ["a", "b", "a"]})
    csv_path = tmp_path / "sample.csv"
    df.write_csv(csv_path)

    api = Api()
    api.load_file(str(csv_path))
    result = api.get_chart_data()

    assert "error" not in result
    assert result["null_heatmap"]["columns"] == ["id", "label"]
    assert [h["column"] for h in result["numeric_histograms"]] == ["id"]


def test_load_file_returns_text_columns(tmp_path):
    values = [f"a fairly unique free-text sentence number {i} of many words" for i in range(60)]
    df = pl.DataFrame({"comment": values})
    csv_path = tmp_path / "sample.csv"
    df.write_csv(csv_path)

    api = Api()
    result = api.load_file(str(csv_path))

    assert result["text_columns"] == ["comment"]


def test_get_chart_data_resolves_relative_image_paths_against_loaded_file_dir(tmp_path):
    import io

    from PIL import Image

    (tmp_path / "images").mkdir()
    buffer = io.BytesIO()
    Image.new("RGB", (4, 4), (255, 0, 0)).save(buffer, format="PNG")
    (tmp_path / "images" / "cat.png").write_bytes(buffer.getvalue())

    df = pl.DataFrame({"path": ["images/cat.png"]})
    csv_path = tmp_path / "sample.csv"
    df.write_csv(csv_path)

    api = Api()
    api.load_file(str(csv_path))
    result = api.get_chart_data()

    assert result["image_columns"] == [{"column": "path", "source": "path"}]
    assert result["image_previews"][0]["rows"][0]["thumbnail"] is not None


def test_get_image_full_returns_error_when_nothing_loaded():
    api = Api()

    result = api.get_image_full("thumb", "bytes", 0)

    assert "error" in result


def test_get_image_full_returns_data_uri_for_bytes_column(tmp_path):
    import io

    from PIL import Image

    buffer = io.BytesIO()
    Image.new("RGB", (4, 4), (0, 255, 0)).save(buffer, format="PNG")
    png_bytes = buffer.getvalue()

    df = pl.DataFrame({"thumb": [png_bytes]}, schema={"thumb": pl.Binary})
    csv_path = tmp_path / "sample.parquet"
    df.write_parquet(csv_path)

    api = Api()
    api.load_file(str(csv_path))
    result = api.get_image_full("thumb", "bytes", 0)

    assert result["image"].startswith("data:image/png;base64,")


def test_get_settings_returns_defaults_when_nothing_saved_yet(tmp_path):
    api = Api(settings_dir=tmp_path)

    result = api.get_settings()

    assert result["classification_banner_enabled"] is False
    assert result["classification_banner_label"] == ""
    assert result["classification_banner_color"] == DEFAULT_BANNER_COLOR
    assert result["audit_log_enabled"] is False
    assert result["audit_log_path"] == str(audit_log_file_path(tmp_path))


def test_save_settings_persists_to_disk_and_is_reflected_by_get_settings(tmp_path):
    api = Api(settings_dir=tmp_path)

    result = api.save_settings(
        {
            "classification_banner_enabled": True,
            "classification_banner_label": "INTERNAL USE ONLY",
            "classification_banner_color": "#eb6834",
        }
    )

    assert result["classification_banner_enabled"] is True
    assert result["classification_banner_label"] == "INTERNAL USE ONLY"
    assert result["classification_banner_color"] == "#eb6834"
    assert settings_file_path(tmp_path).exists()
    assert api.get_settings()["classification_banner_label"] == "INTERNAL USE ONLY"


def test_save_settings_loaded_fresh_by_a_new_api_instance(tmp_path):
    Api(settings_dir=tmp_path).save_settings({"audit_log_enabled": True})

    reloaded = Api(settings_dir=tmp_path)

    assert reloaded.get_settings()["audit_log_enabled"] is True


def test_load_file_writes_audit_entry_when_audit_log_enabled(tmp_path, monkeypatch):
    monkeypatch.setattr("ml_data_view.audit_log.getpass.getuser", lambda: "mocked-user")
    api = Api(settings_dir=tmp_path)
    api.save_settings({"audit_log_enabled": True})

    df = pl.DataFrame({"id": [1, 2]})
    csv_path = tmp_path / "sample.csv"
    df.write_csv(csv_path)
    api.load_file(str(csv_path))

    lines = audit_log_file_path(tmp_path).read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    entry = json.loads(lines[0])
    assert entry["action"] == "file_open"
    assert entry["filename"] == "sample.csv"
    assert entry["user"] == "mocked-user"


def test_load_file_writes_no_audit_entry_when_audit_log_disabled(tmp_path):
    api = Api(settings_dir=tmp_path)

    df = pl.DataFrame({"id": [1, 2]})
    csv_path = tmp_path / "sample.csv"
    df.write_csv(csv_path)
    api.load_file(str(csv_path))

    assert not audit_log_file_path(tmp_path).exists()


def test_get_report_sections_returns_error_when_nothing_loaded():
    api = Api()

    result = api.get_report_sections()

    assert "error" in result


def test_get_report_sections_reflects_the_loaded_dataset(tmp_path):
    df = pl.DataFrame({"id": [1, 2, 3], "label": ["a", "b", "a"]})
    csv_path = tmp_path / "sample.csv"
    df.write_csv(csv_path)

    api = Api(settings_dir=tmp_path)
    api.load_file(str(csv_path))
    api.get_chart_data()
    result = api.get_report_sections()

    assert result == {"overview": True, "distributions": True, "embeddings": False, "media": False}


def test_generate_report_returns_error_when_nothing_loaded(tmp_path):
    api = Api(settings_dir=tmp_path)

    result = api.generate_report(str(tmp_path / "report.pdf"), ["overview"])

    assert "error" in result


def test_generate_report_writes_a_pdf_and_returns_its_path(tmp_path):
    df = pl.DataFrame({"id": [1, 2, 3], "label": ["a", "b", "a"]})
    csv_path = tmp_path / "sample.csv"
    df.write_csv(csv_path)
    report_path = tmp_path / "out" / "report.pdf"

    api = Api(settings_dir=tmp_path)
    api.load_file(str(csv_path))
    api.get_chart_data()
    result = api.generate_report(str(report_path), ["overview", "distributions"])

    assert result["path"] == str(report_path)
    assert report_path.read_bytes()[:5] == b"%PDF-"


def test_generate_report_writes_audit_entry_when_audit_log_enabled(tmp_path, monkeypatch):
    monkeypatch.setattr("ml_data_view.audit_log.getpass.getuser", lambda: "mocked-user")
    df = pl.DataFrame({"id": [1, 2, 3]})
    csv_path = tmp_path / "sample.csv"
    df.write_csv(csv_path)

    api = Api(settings_dir=tmp_path)
    api.save_settings({"audit_log_enabled": True})
    api.load_file(str(csv_path))
    api.get_chart_data()
    api.generate_report(str(tmp_path / "report.pdf"), ["overview"])

    lines = audit_log_file_path(tmp_path).read_text(encoding="utf-8").splitlines()
    # One entry for load_file's file_open, one for generate_report's
    # report_generated — both audit-logged the same way.
    actions = [json.loads(line)["action"] for line in lines]
    assert actions == ["file_open", "report_generated"]
    assert json.loads(lines[1])["filename"] == "sample.csv"
    assert json.loads(lines[1])["user"] == "mocked-user"


def test_generate_report_writes_no_audit_entry_when_audit_log_disabled(tmp_path):
    df = pl.DataFrame({"id": [1, 2, 3]})
    csv_path = tmp_path / "sample.csv"
    df.write_csv(csv_path)

    api = Api(settings_dir=tmp_path)
    api.load_file(str(csv_path))
    api.get_chart_data()
    api.generate_report(str(tmp_path / "report.pdf"), ["overview"])

    assert not audit_log_file_path(tmp_path).exists()


def test_discover_folder_returns_error_for_empty_folder(tmp_path):
    settings_dir = tmp_path / "settings"
    empty_folder = tmp_path / "empty"
    empty_folder.mkdir()

    api = Api(settings_dir=settings_dir)
    result = api.discover_folder(str(empty_folder))

    assert "error" in result


def test_discover_folder_returns_groups_and_default_index(tmp_path):
    settings_dir = tmp_path / "settings"
    data_dir = tmp_path / "dataset_folder"
    (data_dir / "nested").mkdir(parents=True)
    pl.DataFrame({"id": [1, 2], "label": ["a", "b"]}).write_parquet(data_dir / "one.parquet")
    pl.DataFrame({"id": [3, 4], "label": ["c", "d"]}).write_parquet(data_dir / "nested" / "two.parquet")

    api = Api(settings_dir=settings_dir)
    result = api.discover_folder(str(data_dir))

    assert "error" not in result
    assert result["default_group_index"] == 0
    assert result["groups"][0]["file_count"] == 2
    assert result["unreadable"] == []


def test_load_folder_group_returns_error_when_nothing_discovered(tmp_path):
    api = Api(settings_dir=tmp_path / "settings")

    result = api.load_folder_group(0)

    assert "error" in result


def test_load_folder_group_loads_combined_data(tmp_path):
    settings_dir = tmp_path / "settings"
    data_dir = tmp_path / "dataset_folder"
    (data_dir / "a").mkdir(parents=True)
    (data_dir / "b").mkdir(parents=True)
    pl.DataFrame({"id": [1, 2], "label": ["a", "b"]}).write_parquet(data_dir / "a" / "one.parquet")
    pl.DataFrame({"id": [3, 4, 5], "label": ["c", "d", "e"]}).write_parquet(data_dir / "b" / "two.parquet")

    api = Api(settings_dir=settings_dir)
    api.discover_folder(str(data_dir))
    result = api.load_folder_group(0)

    assert "error" not in result
    assert result["row_count"] == 5
    assert {c["name"] for c in result["columns"]} == {"id", "label"}
    assert result["source_type"] == "folder"
    assert result["file_count"] == 2
    assert result["file_path"] == str(data_dir)


def test_load_folder_group_excludes_mismatched_schema_file(tmp_path):
    settings_dir = tmp_path / "settings"
    data_dir = tmp_path / "dataset_folder"
    data_dir.mkdir(parents=True)
    pl.DataFrame({"id": [1, 2], "label": ["a", "b"]}).write_parquet(data_dir / "one.parquet")
    pl.DataFrame({"id": [3, 4], "label": ["c", "d"]}).write_parquet(data_dir / "two.parquet")
    pl.DataFrame({"other_col": [1.0]}).write_parquet(data_dir / "odd.parquet")

    api = Api(settings_dir=settings_dir)
    discovery = api.discover_folder(str(data_dir))

    assert len(discovery["groups"]) == 2
    excluded = discovery["excluded"]
    assert len(excluded) == 1
    assert excluded[0]["file"].endswith("odd.parquet")
    assert excluded[0]["reason"]


def test_load_folder_group_lets_chart_data_run_against_the_combined_frame(tmp_path):
    settings_dir = tmp_path / "settings"
    data_dir = tmp_path / "dataset_folder"
    data_dir.mkdir(parents=True)
    pl.DataFrame({"id": [1, 2, 3], "label": ["a", "b", "a"]}).write_parquet(data_dir / "one.parquet")
    pl.DataFrame({"id": [4, 5], "label": ["b", "a"]}).write_parquet(data_dir / "two.parquet")

    api = Api(settings_dir=settings_dir)
    api.discover_folder(str(data_dir))
    api.load_folder_group(0)
    chart_data = api.get_chart_data()

    assert "error" not in chart_data
    assert chart_data["null_heatmap"]["total_rows"] == 5
    assert [h["column"] for h in chart_data["numeric_histograms"]] == ["id"]


def test_load_folder_group_writes_audit_entry_when_enabled(tmp_path, monkeypatch):
    monkeypatch.setattr("ml_data_view.audit_log.getpass.getuser", lambda: "mocked-user")
    settings_dir = tmp_path / "settings"
    data_dir = tmp_path / "dataset_folder"
    data_dir.mkdir(parents=True)
    pl.DataFrame({"id": [1, 2]}).write_parquet(data_dir / "one.parquet")

    api = Api(settings_dir=settings_dir)
    api.save_settings({"audit_log_enabled": True})
    api.discover_folder(str(data_dir))
    api.load_folder_group(0)

    lines = audit_log_file_path(settings_dir).read_text(encoding="utf-8").splitlines()
    entry = json.loads(lines[0])
    assert entry["action"] == "folder_open"
    assert entry["filename"] == data_dir.name
    assert entry["user"] == "mocked-user"


def test_load_folder_group_writes_no_audit_entry_when_disabled(tmp_path):
    settings_dir = tmp_path / "settings"
    data_dir = tmp_path / "dataset_folder"
    data_dir.mkdir(parents=True)
    pl.DataFrame({"id": [1, 2]}).write_parquet(data_dir / "one.parquet")

    api = Api(settings_dir=settings_dir)
    api.discover_folder(str(data_dir))
    api.load_folder_group(0)

    assert not audit_log_file_path(settings_dir).exists()


# --- cancellable folder loading (progress bar + stop button) ---


def _make_folder(tmp_path, file_count=4, rows=30):
    folder = tmp_path / "data"
    folder.mkdir()
    for i in range(file_count):
        pl.DataFrame(
            {"id": list(range(rows)), "label": [f"c{n % 3}" for n in range(rows)]}
        ).write_parquet(folder / f"part{i}.parquet")
    return folder


def test_start_folder_load_returns_immediately_and_reports_progress(tmp_path):
    folder = _make_folder(tmp_path)
    api = Api(settings_dir=tmp_path)
    api.discover_folder(str(folder))

    started = api.start_folder_load(0)
    assert started["started"] is True
    assert started["files_total"] == 4

    state = api.wait_for_folder_load(timeout=60)
    assert state["finished"] is True
    assert state["fraction"] == 1.0
    assert state["result"]["row_count"] == 120


def test_folder_load_adopts_result_so_charts_can_follow(tmp_path):
    folder = _make_folder(tmp_path)
    api = Api(settings_dir=tmp_path)
    api.discover_folder(str(folder))
    api.start_folder_load(0)
    api.wait_for_folder_load(timeout=60)

    # get_chart_data must work off the loaded frame, exactly as it does
    # after the synchronous load_folder_group().
    assert api.get_chart_data().get("error") is None


def test_cancel_folder_load_yields_a_partial_dataset(tmp_path):
    folder = _make_folder(tmp_path, file_count=6)
    api = Api(settings_dir=tmp_path)
    api.discover_folder(str(folder))
    api.start_folder_load(0)
    api.cancel_folder_load()
    state = api.wait_for_folder_load(timeout=60)

    assert state["cancelled"] is True
    result = state["result"]
    if result is not None:  # cancelled after at least one file finished
        assert result["partial"] is True
        assert result["files_used"] <= 6
        assert result["row_count"] == result["files_used"] * 30


def test_start_folder_load_rejects_a_bad_group_index(tmp_path):
    folder = _make_folder(tmp_path)
    api = Api(settings_dir=tmp_path)
    api.discover_folder(str(folder))

    assert "error" in api.start_folder_load(99)


def test_start_folder_load_requires_a_scanned_folder(tmp_path):
    api = Api(settings_dir=tmp_path)
    assert "error" in api.start_folder_load(0)


def test_progress_before_any_load_is_idle_not_an_error(tmp_path):
    api = Api(settings_dir=tmp_path)
    state = api.get_folder_load_progress()
    assert state["stage"] == "idle"
    assert state["finished"] is False


def test_start_chart_build_reports_progress_and_caches_the_payload(tmp_path):
    folder = _make_folder(tmp_path)
    api = Api(settings_dir=tmp_path)
    api.discover_folder(str(folder))
    api.start_folder_load(0)
    api.wait_for_folder_load(timeout=60)

    assert api.start_chart_build()["started"] is True
    state = api.wait_for_chart_build(timeout=120)

    assert state["finished"] is True
    assert state["error"] is None
    assert state["fraction"] == 1.0
    assert "numeric_histograms" in state["result"]
    # Cached the same way get_chart_data() would, so Reports still works.
    assert api.get_report_sections().get("error") is None


def test_start_chart_build_requires_a_loaded_dataset(tmp_path):
    assert "error" in Api(settings_dir=tmp_path).start_chart_build()


def test_chart_build_progress_before_any_build_is_idle(tmp_path):
    state = Api(settings_dir=tmp_path).get_chart_build_progress()
    assert state["stage"] == "idle"
    assert state["finished"] is False


# --- row search ---


def _searchable_folder(tmp_path, files=3, rows=30):
    folder = tmp_path / "search_data"
    folder.mkdir()
    for i in range(files):
        pl.DataFrame(
            {
                "track_id": [f"T-{n % 3}" for n in range(rows)],
                "rec_key": [n % 5 for n in range(rows)],
                "height": [float(n * 100) for n in range(rows)],
            }
        ).write_parquet(folder / f"p{i}.parquet")
    return folder


def _loaded_api(tmp_path):
    folder = _searchable_folder(tmp_path)
    api = Api(settings_dir=tmp_path)
    api.discover_folder(str(folder))
    api.start_folder_load(0)
    api.wait_for_folder_load(timeout=60)
    return api


def test_run_search_returns_matching_rows(tmp_path):
    api = _loaded_api(tmp_path)

    result = api.run_search([{"column": "track_id", "operator": "is", "values": ["T-1"]}])

    assert "error" not in result
    assert len(result["rows"]) == 30  # 10 per file x 3 files
    assert {row["track_id"] for row in result["rows"]} == {"T-1"}


def test_run_search_with_a_composite_key_and_a_range(tmp_path):
    api = _loaded_api(tmp_path)

    result = api.run_search(
        [
            {"column": "track_id", "operator": "is", "values": ["T-2"]},
            {"column": "rec_key", "operator": "is", "values": ["2"]},
            {"column": "height", "operator": "between", "values": ["0", "1000"]},
        ]
    )

    assert "error" not in result
    for row in result["rows"]:
        assert row["track_id"] == "T-2"
        assert row["rec_key"] == 2
        assert 0 <= row["height"] <= 1000


def test_run_search_reports_the_offending_condition_index(tmp_path):
    api = _loaded_api(tmp_path)

    result = api.run_search(
        [
            {"column": "track_id", "operator": "is", "values": ["T-1"]},
            {"column": "rec_key", "operator": "is", "values": ["not a number"]},
        ]
    )

    assert "error" in result
    assert result["condition_index"] == 1


def test_run_search_projects_selected_columns_only(tmp_path):
    api = _loaded_api(tmp_path)

    result = api.run_search([], columns=["track_id"], limit=5)

    assert all(set(row) == {"track_id"} for row in result["rows"])


def test_run_search_flags_a_truncated_result(tmp_path):
    api = _loaded_api(tmp_path)
    assert api.run_search([], limit=5)["truncated"] is True


def test_match_count_runs_after_a_search(tmp_path):
    api = _loaded_api(tmp_path)
    api.run_search([{"column": "track_id", "operator": "is", "values": ["T-0"]}])

    assert api.start_match_count()["started"] is True
    state = api.wait_for_match_count(timeout=60)
    assert state["result"] == 30


def test_match_count_requires_a_search_first(tmp_path):
    assert "error" in _loaded_api(tmp_path).start_match_count()


def test_search_export_writes_a_csv(tmp_path):
    api = _loaded_api(tmp_path)
    api.run_search([{"column": "track_id", "operator": "is", "values": ["T-1"]}])
    out = tmp_path / "export.csv"

    assert api.start_search_export(str(out))["started"] is True
    state = api.wait_for_search_export(timeout=60)
    assert state["error"] is None
    assert pl.read_csv(out).height == 30


def test_search_works_on_a_single_opened_file(tmp_path):
    """A single-file open has no folder file list; search must still work
    by treating the one path as a one-element list."""
    path = tmp_path / "one.parquet"
    pl.DataFrame({"track_id": ["A", "B", "A"], "height": [1.0, 2.0, 3.0]}).write_parquet(path)
    api = Api(settings_dir=tmp_path)
    api.load_file(str(path))

    result = api.run_search([{"column": "track_id", "operator": "is", "values": ["A"]}])

    assert len(result["rows"]) == 2


def test_saved_searches_round_trip_through_the_api(tmp_path):
    api = Api(settings_dir=tmp_path)
    conditions = [{"column": "height", "operator": ">=", "values": ["1000"]}]

    saved = api.save_search("high", conditions, ["height"])
    assert [s["name"] for s in saved["saved_searches"]] == ["high"]
    assert [s["name"] for s in Api(settings_dir=tmp_path).list_saved_searches()] == ["high"]

    assert api.delete_saved_search("high")["saved_searches"] == []


def test_saving_a_search_without_a_name_is_an_error(tmp_path):
    assert "error" in Api(settings_dir=tmp_path).save_search("  ", [], None)


# --- dataset comparison ---


def _compare_folder(tmp_path, name, offset=0.0, files=3, rows=200):
    folder = tmp_path / name
    folder.mkdir()
    for i in range(files):
        pl.DataFrame(
            {
                "value": [float(n % 40) + offset for n in range(rows)],
                "kind": [f"k{n % 3}" for n in range(rows)],
            }
        ).write_parquet(folder / f"p{i}.parquet")
    return folder


def _api_with_folder(tmp_path, folder):
    api = Api(settings_dir=tmp_path)
    api.discover_folder(str(folder))
    api.start_folder_load(0)
    api.wait_for_folder_load(timeout=60)
    return api


def test_start_comparison_returns_the_schema_diff_immediately(tmp_path):
    a = _compare_folder(tmp_path, "a")
    b = _compare_folder(tmp_path, "b")
    api = _api_with_folder(tmp_path, a)

    started = api.start_comparison(str(b))

    assert started["started"] is True
    assert started["schema"]["only_in_a"] == []
    assert started["schema"]["only_in_b"] == []
    api.wait_for_comparison(timeout=120)


def test_comparison_flags_a_shifted_column(tmp_path):
    a = _compare_folder(tmp_path, "a", offset=0.0)
    b = _compare_folder(tmp_path, "b", offset=900.0)
    api = _api_with_folder(tmp_path, a)

    api.start_comparison(str(b))
    state = api.wait_for_comparison(timeout=120)

    assert state["error"] is None
    top = state["result"]["columns"][0]
    assert top["column"] == "value"
    assert top["severity"] == "significant"


def test_comparing_a_dataset_against_itself_is_stable(tmp_path):
    """A useful sanity check, not an error."""
    a = _compare_folder(tmp_path, "a")
    api = _api_with_folder(tmp_path, a)

    api.start_comparison(str(a))
    state = api.wait_for_comparison(timeout=120)

    assert all(row["severity"] == "stable" for row in state["result"]["columns"])


def test_comparison_reports_schema_differences(tmp_path):
    a = _compare_folder(tmp_path, "a")
    b = tmp_path / "b"
    b.mkdir()
    pl.DataFrame({"value": [1.0], "extra": ["x"]}).write_parquet(b / "p0.parquet")
    api = _api_with_folder(tmp_path, a)

    started = api.start_comparison(str(b))

    assert started["schema"]["only_in_a"] == ["kind"]
    assert started["schema"]["only_in_b"] == ["extra"]
    api.wait_for_comparison(timeout=120)


def test_start_comparison_requires_a_loaded_dataset(tmp_path):
    assert "error" in Api(settings_dir=tmp_path).start_comparison(str(tmp_path))


def test_start_comparison_rejects_a_path_with_no_datasets(tmp_path):
    a = _compare_folder(tmp_path, "a")
    empty = tmp_path / "empty"
    empty.mkdir()
    api = _api_with_folder(tmp_path, a)

    assert "error" in api.start_comparison(str(empty))


def test_comparison_against_a_single_file(tmp_path):
    a = _compare_folder(tmp_path, "a")
    single = tmp_path / "one.parquet"
    pl.DataFrame({"value": [1.0, 2.0], "kind": ["k0", "k1"]}).write_parquet(single)
    api = _api_with_folder(tmp_path, a)

    started = api.start_comparison(str(single))
    assert started["started"] is True
    assert started["files_b"] == 1
    state = api.wait_for_comparison(timeout=120)
    assert state["error"] is None


def _api_with_folder_at(tmp_path, folder):
    api = Api(settings_dir=tmp_path)
    api.discover_folder(str(folder))
    api.start_folder_load(0)
    api.wait_for_folder_load(timeout=60)
    return api


# --- charts following the search filter ---


def test_apply_chart_filter_makes_every_figure_describe_matching_rows(tmp_path):
    folder = _searchable_folder(tmp_path, files=2, rows=30)
    api = _api_with_folder_at(tmp_path, folder)
    api.run_search([{"column": "track_id", "operator": "is", "values": ["T-1"]}])

    filtered = api.apply_chart_filter()

    assert filtered["filter_active"] is True
    assert filtered["row_count"] == 20  # 10 per file x 2 files, not 60
    api.start_chart_build()
    state = api.wait_for_chart_build(timeout=120)
    total = sum(state["result"]["categorical_bars"][0]["counts"])
    assert total == 20  # charts describe the filtered rows too


def test_clearing_the_filter_restores_the_whole_dataset(tmp_path):
    folder = _searchable_folder(tmp_path, files=2, rows=30)
    api = _api_with_folder_at(tmp_path, folder)
    api.run_search([{"column": "track_id", "operator": "is", "values": ["T-1"]}])
    api.apply_chart_filter()

    restored = api.clear_chart_filter()

    assert restored["filter_active"] is False
    assert restored["row_count"] == 60


def test_apply_chart_filter_requires_a_search(tmp_path):
    folder = _searchable_folder(tmp_path, files=1)
    api = _api_with_folder_at(tmp_path, folder)
    assert "error" in api.apply_chart_filter()


def test_null_counts_follow_the_filter(tmp_path):
    """Footer null counts describe whole files, so a filtered dataset info
    must not report them."""
    folder = tmp_path / "nulls"
    folder.mkdir()
    pl.DataFrame(
        {"keep": [1, 1, 0, 0], "v": [None, 5.0, None, None]}
    ).write_parquet(folder / "p0.parquet")
    api = _api_with_folder_at(tmp_path, folder)
    api.run_search([{"column": "keep", "operator": "is", "values": ["1"]}])

    filtered = api.apply_chart_filter()

    assert {c["name"]: c["null_count"] for c in filtered["columns"]}["v"] == 1


def test_loading_a_new_dataset_clears_any_active_filter(tmp_path):
    """A filter belonging to a previous dataset must never silently apply
    to the next one."""
    folder = _searchable_folder(tmp_path, files=1)
    api = _api_with_folder_at(tmp_path, folder)
    api.run_search([{"column": "track_id", "operator": "is", "values": ["T-1"]}])
    api.apply_chart_filter()
    assert api.get_chart_filter()["filter_active"] is True

    other = tmp_path / "other.parquet"
    pl.DataFrame({"track_id": ["T-9"], "height": [1.0]}).write_parquet(other)
    api.load_file(str(other))

    assert api.get_chart_filter()["filter_active"] is False


# --- load performance stat line ---------------------------------------


def test_load_file_reports_load_stats_for_a_single_file(tmp_path):
    pl.DataFrame({"id": [1, 2, 3]}).write_csv(tmp_path / "sample.csv")

    api = Api(settings_dir=tmp_path)
    result = api.load_file(str(tmp_path / "sample.csv"))

    stats = result["load_stats"]
    assert stats["row_count"] == 3
    assert stats["file_count"] == 1  # a single open is one file, not "unknown"
    assert stats["elapsed_seconds"] >= 0
    assert "Profiled 3 rows across 1 file in " in stats["line"]


def test_load_file_stat_line_omits_memory_when_the_reading_is_unavailable(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(load_stats, "peak_memory_bytes", lambda: None)
    pl.DataFrame({"id": [1]}).write_csv(tmp_path / "sample.csv")

    api = Api(settings_dir=tmp_path)
    result = api.load_file(str(tmp_path / "sample.csv"))

    assert "memory" not in result["load_stats"]["line"]


def test_load_file_times_the_load_itself_not_zero(tmp_path, monkeypatch):
    """The clock must wrap the real load path — a stat line that always
    says 0.0s would be worse than no stat line."""
    clock = iter([100.0, 102.5])
    monkeypatch.setattr("ml_data_view.api.perf_counter", lambda: next(clock))
    pl.DataFrame({"id": [1]}).write_csv(tmp_path / "sample.csv")

    api = Api(settings_dir=tmp_path)
    result = api.load_file(str(tmp_path / "sample.csv"))

    assert result["load_stats"]["elapsed_seconds"] == 2.5
    assert "in 2.5s" in result["load_stats"]["line"]


def test_load_folder_group_reports_stats_across_every_file(tmp_path):
    data_dir = tmp_path / "dataset_folder"
    data_dir.mkdir()
    pl.DataFrame({"id": [1, 2]}).write_parquet(data_dir / "one.parquet")
    pl.DataFrame({"id": [3, 4, 5]}).write_parquet(data_dir / "two.parquet")

    api = Api(settings_dir=tmp_path / "settings")
    api.discover_folder(str(data_dir))
    result = api.load_folder_group(0)

    stats = result["load_stats"]
    assert stats["row_count"] == 5
    assert stats["file_count"] == 2
    assert "Profiled 5 rows across 2 files in " in stats["line"]


def test_folder_load_job_result_carries_stats_for_the_worker_thread_load(tmp_path):
    folder = _make_folder(tmp_path)
    api = Api(settings_dir=tmp_path)
    api.discover_folder(str(folder))
    api.start_folder_load(0)
    state = api.wait_for_folder_load(timeout=60)

    stats = state["result"]["load_stats"]
    assert stats["row_count"] == 120
    assert stats["file_count"] == 4
    assert stats["elapsed_seconds"] >= 0


def test_stopped_folder_load_stats_count_only_the_files_actually_used(tmp_path):
    """A stopped load is a smaller dataset — the stat line must agree with
    the partial banner rather than quoting the folder's full file count."""
    folder = _make_folder(tmp_path, file_count=6)
    api = Api(settings_dir=tmp_path)
    api.discover_folder(str(folder))
    api.start_folder_load(0)
    api.cancel_folder_load()
    state = api.wait_for_folder_load(timeout=60)

    result = state["result"]
    if result is None:  # cancelled before any file finished — nothing to report
        return
    stats = result["load_stats"]
    assert stats["file_count"] == result["files_used"]
    assert stats["row_count"] == result["row_count"]


# --- Findings tab -----------------------------------------------------


def test_list_checks_returns_the_full_catalogue_before_any_dataset_is_loaded(tmp_path):
    """The tab paints every check name as pending before work starts, so
    the catalogue must not depend on a loaded dataset."""
    api = Api(settings_dir=tmp_path)

    checks = api.list_checks()["checks"]

    assert [c["id"] for c in checks][:3] == ["empty_dataset", "all_null_column", "high_missingness"]
    assert {c["kind"] for c in checks} == {"instant", "deep"}


def test_run_instant_checks_requires_a_dataset(tmp_path):
    api = Api(settings_dir=tmp_path)

    assert "error" in api.run_instant_checks()


def test_run_instant_checks_reports_pending_before_charts_are_built(tmp_path):
    """Never a clean state before the checks have run — that is a lie at
    exactly the moment the user is most likely to trust it."""
    pl.DataFrame({"id": [1, 2, 3]}).write_csv(tmp_path / "sample.csv")
    api = Api(settings_dir=tmp_path)
    api.load_file(str(tmp_path / "sample.csv"))

    result = api.run_instant_checks()

    assert result["pending"] == "charts"
    assert "findings" not in result


def test_run_instant_checks_finds_a_planted_problem(tmp_path):
    csv_path = tmp_path / "sample.csv"
    pl.DataFrame({"id": [1, 2, 3, 4], "dead": [None, None, None, None]}).write_csv(csv_path)

    api = Api(settings_dir=tmp_path)
    api.load_file(str(csv_path))
    api.get_chart_data()
    result = api.run_instant_checks()

    all_null = next(c for c in result["checks"] if c["id"] == "all_null_column")
    assert all_null["status"] == "done"
    assert all_null["findings"][0]["columns"] == ["dead"]
    assert result["summary"]["critical"] >= 1


def test_run_instant_checks_reports_a_clean_dataset_as_clean(tmp_path):
    csv_path = tmp_path / "clean.csv"
    pl.DataFrame({"a": [1, 2, 3, 4], "b": ["w", "x", "y", "z"]}).write_csv(csv_path)

    api = Api(settings_dir=tmp_path)
    api.load_file(str(csv_path))
    api.get_chart_data()
    result = api.run_instant_checks()

    assert result["summary"] == {"critical": 0, "warning": 0, "info": 0}
    assert all(c["status"] == "done" for c in result["checks"])


def test_start_deep_checks_requires_a_dataset(tmp_path):
    api = Api(settings_dir=tmp_path)

    assert "error" in api.start_deep_checks()


def test_deep_checks_run_and_report_per_check_status(tmp_path):
    parquet_path = tmp_path / "dupes.parquet"
    pl.DataFrame({"a": [1, 2, 1], "b": ["x", "y", "x"]}).write_parquet(parquet_path)

    api = Api(settings_dir=tmp_path)
    api.load_file(str(parquet_path))
    assert api.start_deep_checks()["started"] is True
    state = api.wait_for_deep_checks(timeout=60)

    assert state["finished"] is True
    duplicate = next(c for c in state["checks"] if c["id"] == "duplicate_rows")
    assert duplicate["status"] == "done"
    assert duplicate["findings"], "the fixture has a planted duplicate row"


def test_deep_check_progress_is_idle_before_any_run(tmp_path):
    api = Api(settings_dir=tmp_path)

    state = api.get_deep_check_progress()

    assert state["stage"] == "idle"
    assert state["finished"] is False


def test_cancel_deep_checks_without_a_running_job_is_an_error(tmp_path):
    api = Api(settings_dir=tmp_path)

    assert "error" in api.cancel_deep_checks()


def test_deep_checks_see_the_folder_file_list(tmp_path):
    """Per-file scanning, not a query over the concat — the same rule every
    other multi-file computation follows."""
    data_dir = tmp_path / "folder"
    data_dir.mkdir()
    for index in range(2):
        pl.DataFrame({"v": [-999] * 5 + list(range(5))}).write_parquet(
            data_dir / f"part{index}.parquet"
        )

    api = Api(settings_dir=tmp_path / "settings")
    api.discover_folder(str(data_dir))
    api.load_folder_group(0)
    api.start_deep_checks()
    state = api.wait_for_deep_checks(timeout=60)

    sentinel = next(c for c in state["checks"] if c["id"] == "sentinel_values")
    assert sentinel["status"] == "done"
    # 5 sentinels per file across 2 files, counted exactly.
    assert "10" in sentinel["findings"][0]["detail"]
