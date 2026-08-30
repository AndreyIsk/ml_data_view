import polars as pl
import pytest

import ml_data_view.search as search_module
from ml_data_view.search import (
    SEARCH_ROW_LIMIT,
    Condition,
    SearchConditionError,
    build_predicate,
    count_matches,
    export_matches,
    search_rows,
)


def _frame(rows=40):
    return pl.DataFrame(
        {
            "track_id": [f"T-{n % 5}" for n in range(rows)],
            "rec_key": [n % 4 for n in range(rows)],
            "height": [float(n * 10) for n in range(rows)],
            "flagged": [n % 2 == 0 for n in range(rows)],
            "note": [None if n % 3 == 0 else f"note {n}" for n in range(rows)],
        }
    )


def _write(tmp_path, count=3, rows=40):
    files = []
    for i in range(count):
        path = tmp_path / f"part{i}.parquet"
        _frame(rows).write_parquet(path)
        files.append(path)
    return files


def _schema(files):
    return pl.scan_parquet(files[0]).collect_schema()


# --- build_predicate ---------------------------------------------------


def _apply(condition, frame=None):
    frame = frame if frame is not None else _frame()
    predicate = build_predicate([condition], frame.lazy().collect_schema())
    return frame.filter(predicate)


def test_is_operator_matches_exact_value():
    assert _apply(Condition("track_id", "is", ["T-2"])).height == 8


def test_is_not_operator_excludes_the_value():
    assert _apply(Condition("track_id", "is not", ["T-2"])).height == 32


def test_comparison_operators_on_a_numeric_column():
    assert _apply(Condition("height", ">", ["100"])).height == 29
    assert _apply(Condition("height", ">=", ["100"])).height == 30
    assert _apply(Condition("height", "<", ["100"])).height == 10
    assert _apply(Condition("height", "<=", ["100"])).height == 11


def test_between_is_inclusive_on_both_ends():
    assert _apply(Condition("height", "between", ["100", "200"])).height == 11


def test_is_null_and_is_not_null():
    assert _apply(Condition("note", "is null", [])).height == 14
    assert _apply(Condition("note", "is not null", [])).height == 26


def test_contains_matches_a_substring():
    # "note 1" also matches note 10..19; the multiples of 3 in that span are
    # null, leaving 1, 10, 11, 13, 14, 16, 17, 19.
    assert _apply(Condition("note", "contains", ["note 1"])).height == 8


def test_conditions_are_combined_with_and():
    frame = _frame()
    predicate = build_predicate(
        [Condition("track_id", "is", ["T-1"]), Condition("height", ">=", ["100"])],
        frame.lazy().collect_schema(),
    )
    result = frame.filter(predicate)

    assert result.height == 6
    assert set(result["track_id"].to_list()) == {"T-1"}
    assert min(result["height"].to_list()) >= 100


def test_composite_key_selects_one_unique_entity():
    """The motivating case: uniqueness defined by the user as a combination
    of columns, not a fixed property of the schema."""
    frame = _frame()
    predicate = build_predicate(
        [Condition("track_id", "is", ["T-3"]), Condition("rec_key", "is", ["3"])],
        frame.lazy().collect_schema(),
    )
    result = frame.filter(predicate)

    assert result.height >= 1
    assert set(result["track_id"].to_list()) == {"T-3"}
    assert set(result["rec_key"].to_list()) == {3}


def test_no_conditions_matches_everything():
    frame = _frame()
    assert frame.filter(build_predicate([], frame.lazy().collect_schema())).height == 40


def test_boolean_column_accepts_true_and_false_case_insensitively():
    assert _apply(Condition("flagged", "is", ["TRUE"])).height == 20
    assert _apply(Condition("flagged", "is", ["false"])).height == 20


# --- build_predicate failures -----------------------------------------


def test_non_integer_value_for_an_integer_column_is_rejected():
    """Rejected rather than silently truncated — '1.9' meaning rec_key 1 is
    a guess we should not make on the user's behalf."""
    with pytest.raises(SearchConditionError) as exc:
        _apply(Condition("rec_key", "is", ["1.9"]))
    assert exc.value.condition_index == 0
    assert "rec_key" in str(exc.value)


def test_non_numeric_value_for_a_numeric_column_is_rejected():
    with pytest.raises(SearchConditionError) as exc:
        _apply(Condition("height", ">=", ["abc"]))
    assert exc.value.condition_index == 0


def test_unknown_column_is_rejected_with_its_index():
    frame = _frame()
    with pytest.raises(SearchConditionError) as exc:
        build_predicate(
            [Condition("height", ">", ["1"]), Condition("nope", "is", ["x"])],
            frame.lazy().collect_schema(),
        )
    assert exc.value.condition_index == 1


def test_unknown_operator_is_rejected():
    with pytest.raises(SearchConditionError):
        _apply(Condition("height", "roughly", ["1"]))


def test_between_requires_two_values():
    with pytest.raises(SearchConditionError):
        _apply(Condition("height", "between", ["100"]))


def test_contains_is_rejected_on_a_non_string_column():
    with pytest.raises(SearchConditionError) as exc:
        _apply(Condition("height", "contains", ["1"]))
    assert "contains" in str(exc.value).lower()


# --- search_rows -------------------------------------------------------


def test_search_rows_matches_a_single_frame_filter(tmp_path):
    files = _write(tmp_path, count=3)
    predicate = build_predicate([Condition("track_id", "is", ["T-1"])], _schema(files))

    rows = search_rows(files, predicate)
    expected = (
        pl.concat([pl.scan_parquet(f) for f in files], how="vertical")
        .filter(predicate)
        .head(SEARCH_ROW_LIMIT)
        .collect()
        .to_dicts()
    )

    assert rows == expected


def test_search_rows_respects_the_limit(tmp_path):
    files = _write(tmp_path, count=3)
    predicate = build_predicate([], _schema(files))

    assert len(search_rows(files, predicate, limit=7)) == 7


def test_search_rows_projects_only_requested_columns(tmp_path):
    files = _write(tmp_path, count=2)
    predicate = build_predicate([], _schema(files))

    rows = search_rows(files, predicate, columns=["track_id", "height"], limit=3)

    assert all(set(row) == {"track_id", "height"} for row in rows)


def test_search_rows_stops_opening_files_once_the_limit_is_reached(tmp_path, monkeypatch):
    """Early exit is what makes a folder search instant — measured at 500
    matches found in file 1 of 50. This pins that it actually stops."""
    files = _write(tmp_path, count=10, rows=40)
    predicate = build_predicate([], _schema(files))
    opened = []
    real_scan = search_module._scan
    monkeypatch.setattr(
        search_module, "_scan", lambda path: (opened.append(path), real_scan(path))[1]
    )

    search_rows(files, predicate, limit=10)

    assert len(opened) == 1, f"should have stopped after one file, opened {len(opened)}"


def test_search_rows_spans_files_when_one_is_not_enough(tmp_path):
    files = _write(tmp_path, count=3, rows=10)
    predicate = build_predicate([], _schema(files))

    assert len(search_rows(files, predicate, limit=25)) == 25


# --- count_matches -----------------------------------------------------


def test_count_matches_equals_a_single_frame_count(tmp_path):
    files = _write(tmp_path, count=4)
    predicate = build_predicate([Condition("height", ">=", ["100"])], _schema(files))

    counted = count_matches(files, predicate)
    expected = (
        pl.concat([pl.scan_parquet(f) for f in files], how="vertical")
        .filter(predicate)
        .select(pl.len())
        .collect()[0, 0]
    )

    assert counted == expected


def test_count_matches_reports_progress_once_per_file(tmp_path):
    files = _write(tmp_path, count=5)
    predicate = build_predicate([], _schema(files))
    seen = []

    count_matches(files, predicate, on_progress=lambda i, p: seen.append(i))

    assert seen == [0, 1, 2, 3, 4]


def test_count_matches_cancelled_covers_only_completed_files(tmp_path):
    files = _write(tmp_path, count=6, rows=10)
    predicate = build_predicate([], _schema(files))
    done = []

    total = count_matches(
        files,
        predicate,
        on_progress=lambda i, p: done.append(i),
        should_cancel=lambda: len(done) >= 2,
    )

    assert done == [0, 1]
    assert total == 20  # exactly the two completed files, nothing partial


# --- export_matches ----------------------------------------------------


def test_export_writes_exactly_the_matching_rows(tmp_path):
    files = _write(tmp_path, count=3)
    predicate = build_predicate([Condition("track_id", "is", ["T-2"])], _schema(files))
    out = tmp_path / "out.csv"

    export_matches(files, predicate, None, out)

    written = pl.read_csv(out)
    assert written.height == 24  # 8 per file x 3 files
    assert set(written["track_id"].to_list()) == {"T-2"}


def test_export_can_exceed_the_display_limit(tmp_path):
    files = _write(tmp_path, count=3, rows=1000)
    predicate = build_predicate([], _schema(files))
    out = tmp_path / "big.csv"

    export_matches(files, predicate, None, out)

    assert pl.read_csv(out).height == 3000  # far above SEARCH_ROW_LIMIT


def test_export_writes_only_selected_columns(tmp_path):
    files = _write(tmp_path, count=2)
    predicate = build_predicate([], _schema(files))
    out = tmp_path / "cols.csv"

    export_matches(files, predicate, ["track_id", "height"], out)

    assert pl.read_csv(out).columns == ["track_id", "height"]


def test_export_of_no_matches_still_writes_a_header_only_file(tmp_path):
    files = _write(tmp_path, count=2)
    predicate = build_predicate([Condition("track_id", "is", ["nope"])], _schema(files))
    out = tmp_path / "empty.csv"

    export_matches(files, predicate, ["track_id"], out)

    assert out.exists()
    assert pl.read_csv(out).height == 0
