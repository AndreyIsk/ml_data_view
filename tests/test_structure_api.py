"""Structure and scope through the Api, across every input shape.

hierarchy.py's own tests cover the model. These cover what the APP does
with it — and, at least as importantly, what it does NOT do when there is
no structure to speak of.

The five shapes Otter must keep working for are built by `shape` below:

    SINGLE   one file
    FLAT     a folder of files
    HIVE     key=value partitions, no series level
    NESTED   an arbitrary tree with nothing recognisable in it
    FULL     group levels plus a series of ordered chunks

The section at the end is the important one: it pins that a flat input
produces exactly the payload it produced before any of this existed —
same columns, same charts, no scope, no structure, no derived anything.
"""
from __future__ import annotations

from pathlib import Path

import polars as pl
import pytest

from ml_data_view.api import Api

BASE_TS = 1_700_000_000

def _conditional_checks() -> set[str]:
    """Every check that declares a precondition, read from the registry.

    Derived rather than listed: the point being tested is that a check
    without its precondition SKIPS, and a new check should inherit that
    assertion rather than quietly escaping it.
    """
    from ml_data_view.findings import INSTANT_CHECKS

    return {spec.id for spec in INSTANT_CHECKS if spec.applies is not None}


def _series_checks() -> set[str]:
    """The subset that additionally needs a series of chunks."""
    from ml_data_view.findings import INSTANT_CHECKS

    return {
        spec.id for spec in INSTANT_CHECKS if spec.requires == "a series of chunks"
    }


def _timestamp_checks() -> set[str]:
    from ml_data_view.findings import INSTANT_CHECKS

    return {spec.id for spec in INSTANT_CHECKS if spec.requires == "a timestamp column"}


def _chunk(path: Path, start_ts: int = BASE_TS, rows: int = 10, deviant: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data: dict[str, list] = {
        "ts": [start_ts + index for index in range(rows)],
        "value": [float(index) for index in range(rows)],
    }
    if deviant:
        data.pop("value")
        data["other"] = [float(index) for index in range(rows)]
    pl.DataFrame(data).write_parquet(path)


def build_flat(root: Path) -> Path:
    for index in range(3):
        _chunk(root / f"part{index}.parquet", BASE_TS + index * 100)
    return root


def build_hive(root: Path) -> Path:
    for day in ("day=2026-03-01", "day=2026-03-02"):
        for index in range(2):
            _chunk(root / day / f"part{index}.parquet", BASE_TS + index * 100)
    return root


def build_nested(root: Path) -> Path:
    for outer in ("alpha", "beta"):
        for inner in ("x", "y"):
            _chunk(root / outer / inner / "data.parquet")
    return root


def build_full(root: Path) -> Path:
    for index, name in enumerate(["part-1.parquet", "part-2.parquet", "part-10.parquet"]):
        _chunk(root / "sys_key=123" / "rec_a" / "raw" / name, BASE_TS + index * 100)
    _chunk(root / "sys_key=123" / "rec_b" / "part-1.parquet", BASE_TS + 9_000)
    _chunk(root / "sys_key=123" / "rec_b" / "part-2.parquet", BASE_TS + 8_000)
    _chunk(root / "sys_key=456" / "rec_c" / "part-1.parquet", BASE_TS + 1_000)
    _chunk(root / "sys_key=456" / "rec_c" / "part-2.parquet", BASE_TS + 1_100, deviant=True)
    return root


FOLDER_SHAPES = {
    "flat": build_flat,
    "hive": build_hive,
    "nested": build_nested,
    "full": build_full,
}


@pytest.fixture
def shape(tmp_path, request):
    return FOLDER_SHAPES[request.param](tmp_path / request.param)


def _confirm(api: Api, discovery: dict, **overrides) -> dict:
    """Accept the suggested mapping, the way the dialog's default does."""
    proposal = discovery["structure_proposal"]
    payload = {
        "root": proposal["root"],
        "roles": {level["depth"]: level["role"] for level in proposal["levels"]},
        "labels": {level["depth"]: level["label"] for level in proposal["levels"]},
        "fingerprint": proposal["fingerprint"],
        "timestamp_column": proposal["suggested_timestamp"],
        "timestamp_unit": proposal["suggested_unit"],
    }
    payload.update(overrides)
    return api.confirm_structure(payload)


def _open(api: Api, folder: Path) -> dict:
    """Open a folder the way the frontend does, whatever its shape."""
    discovery = api.discover_folder(str(folder))
    if discovery.get("source") == "structure":
        if discovery.get("needs_confirmation"):
            _confirm(api, discovery)
        api.start_scope_load()
    else:
        api.start_folder_load(0)
    return api.wait_for_folder_load(timeout=180)


def _statuses(api: Api) -> dict[str, str]:
    api.start_chart_build()
    api.wait_for_chart_build(timeout=300)
    return {c["id"]: c["status"] for c in api.run_instant_checks()["checks"]}


# --- what each shape is detected as ---------------------------------------


@pytest.mark.parametrize("shape", ["flat", "nested"], indirect=True)
def test_shapes_with_nothing_recognisable_take_the_flat_path(shape, tmp_path):
    """The schema-group behaviour, untouched. No dialog, no scope, no
    structure — the app opens them exactly as it always has."""
    api = Api(settings_dir=tmp_path / "settings")

    discovery = api.discover_folder(str(shape))

    assert discovery["source"] == "flat"
    assert discovery["needs_confirmation"] is False
    assert "groups" in discovery
    assert api.get_structure() == {"detected": False}
    assert api.get_scope() == {"active": False}


def test_a_hive_folder_is_detected_with_groups_and_no_series(shape_hive_api):
    api, _folder = shape_hive_api

    structure = api.get_structure()

    assert structure["detected"] is True
    assert structure["has_groups"] is True
    assert structure["has_series"] is False
    assert structure["group_labels"] == ["day"]
    assert structure["unit_label"] == "File"


@pytest.fixture
def shape_hive_api(tmp_path):
    folder = build_hive(tmp_path / "hive")
    api = Api(settings_dir=tmp_path / "settings")
    _open(api, folder)
    return api, folder


@pytest.fixture
def full_api(tmp_path):
    folder = build_full(tmp_path / "full")
    api = Api(settings_dir=tmp_path / "settings")
    state = _open(api, folder)
    return api, folder, state


def test_a_full_tree_is_detected_with_both_roles(full_api):
    api, _folder, _state = full_api

    structure = api.get_structure()

    assert structure["has_groups"] is True
    assert structure["has_series"] is True
    assert structure["unit_count"] == 3
    assert structure["derived_columns"] == ["sys_key", "unit_id", "chunk_index", "chunk_file"]


def test_a_single_file_has_no_structure_and_no_scope(tmp_path):
    path = tmp_path / "one.csv"
    pl.DataFrame({"ts": [1, 2, 3], "value": [1.0, 2.0, 3.0]}).write_csv(path)
    api = Api(settings_dir=tmp_path / "settings")

    api.load_file(str(path))

    assert api.get_structure() == {"detected": False}
    assert api.get_scope() == {"active": False}


# --- labels are the user's, with generic defaults -------------------------


def test_the_confirmed_labels_are_what_the_breadcrumb_shows(tmp_path):
    """"System" and "Recording" are one dataset's words for its own levels.
    Nothing in the code supplies them."""
    folder = build_full(tmp_path / "full")
    api = Api(settings_dir=tmp_path / "settings")
    discovery = api.discover_folder(str(folder))
    _confirm(
        api,
        discovery,
        labels={0: "System", 1: "Recording"},
        root_label="Fleet",
    )
    api.start_scope_load()
    api.wait_for_folder_load(timeout=180)

    scope = api.get_scope()

    assert scope["crumbs"][0]["label"] == "Fleet"
    assert scope["group_labels"] == ["System"]
    assert scope["unit_label"] == "Recording"
    assert scope["option_label"] == "System"


def test_unnamed_levels_fall_back_to_generic_defaults(tmp_path):
    folder = build_full(tmp_path / "full")
    api = Api(settings_dir=tmp_path / "settings")
    discovery = api.discover_folder(str(folder))
    _confirm(api, discovery, labels={})
    api.start_scope_load()
    api.wait_for_folder_load(timeout=180)

    scope = api.get_scope()

    assert scope["crumbs"][0]["label"] == "All"
    assert scope["group_labels"] == ["Group"]
    assert scope["unit_label"] == "Series"


# --- mapping persistence and override -------------------------------------


def test_the_mapping_is_not_asked_for_twice(tmp_path):
    folder = build_full(tmp_path / "full")
    api = Api(settings_dir=tmp_path / "settings")
    _confirm(api, api.discover_folder(str(folder)))

    reopened = Api(settings_dir=tmp_path / "settings")
    discovery = reopened.discover_folder(str(folder))

    assert discovery["needs_confirmation"] is False
    assert discovery["structure"]["unit_count"] == 3


def test_adding_a_file_does_not_make_it_ask_again(tmp_path):
    folder = build_full(tmp_path / "full")
    api = Api(settings_dir=tmp_path / "settings")
    _confirm(api, api.discover_folder(str(folder)))
    _chunk(folder / "sys_key=123" / "rec_b" / "part-3.parquet", BASE_TS + 9_500)

    reopened = Api(settings_dir=tmp_path / "settings")

    assert reopened.discover_folder(str(folder))["needs_confirmation"] is False


def test_a_changed_shape_asks_again(tmp_path):
    folder = build_full(tmp_path / "full")
    api = Api(settings_dir=tmp_path / "settings")
    _confirm(api, api.discover_folder(str(folder)))
    _chunk(folder / "sys_key=789" / "rec_z" / "a" / "b" / "c" / "p0.parquet")

    reopened = Api(settings_dir=tmp_path / "settings")

    assert reopened.discover_folder(str(folder))["needs_confirmation"] is True


def test_an_override_is_remembered_not_the_suggestion(tmp_path):
    folder = build_full(tmp_path / "full")
    api = Api(settings_dir=tmp_path / "settings")
    proposal = api.discover_folder(str(folder))["structure_proposal"]
    roles = {level["depth"]: level["role"] for level in proposal["levels"]}
    roles[1] = "ignore"
    roles[2] = "series"
    api.confirm_structure(
        {"root": proposal["root"], "roles": roles, "fingerprint": proposal["fingerprint"]}
    )

    reopened = Api(settings_dir=tmp_path / "settings")
    structure = reopened.discover_folder(str(folder))["structure"]

    assert [unit["unit_id"] for unit in structure["units"]] == ["raw"]


def test_a_mapping_with_neither_role_is_rejected(tmp_path):
    """A structure with no group and no series is not a structure, and
    applying one would produce nothing while looking like it worked."""
    folder = build_full(tmp_path / "full")
    api = Api(settings_dir=tmp_path / "settings")
    proposal = api.discover_folder(str(folder))["structure_proposal"]

    result = api.confirm_structure(
        {
            "root": proposal["root"],
            "roles": {level["depth"]: "ignore" for level in proposal["levels"]},
            "fingerprint": proposal["fingerprint"],
        }
    )

    assert "error" in result


def test_a_detected_structure_can_be_declined(tmp_path):
    """The escape hatch that keeps structure optional in practice as well
    as in principle. Declining returns the schema-group discovery."""
    folder = build_full(tmp_path / "full")
    api = Api(settings_dir=tmp_path / "settings")
    _confirm(api, api.discover_folder(str(folder)))
    assert api.get_structure()["detected"] is True

    flat = api.clear_structure()

    assert flat["source"] == "flat"
    assert api.get_structure() == {"detected": False}
    assert "groups" in flat


def test_an_undetected_tree_can_still_be_mapped_by_hand(tmp_path):
    """Detection failing is not the same as there being nothing to map.
    An arbitrary tree's levels are offered so a person can assign roles the
    code refused to guess."""
    folder = build_nested(tmp_path / "nested")
    api = Api(settings_dir=tmp_path / "settings")
    discovery = api.discover_folder(str(folder))
    proposal = discovery["structure_proposal"]
    assert proposal["detected"] is False
    assert len(proposal["levels"]) == 2

    api.confirm_structure(
        {
            "root": proposal["root"],
            "roles": {0: "group", 1: "series"},
            "labels": {0: "Site", 1: "Run"},
            "fingerprint": proposal["fingerprint"],
        }
    )

    structure = api.get_structure()
    assert structure["detected"] is True
    assert structure["group_labels"] == ["Site"]
    assert structure["derived_columns"][:2] == ["site_id", "run_id"]


# --- scope ----------------------------------------------------------------


def test_scope_narrows_by_group_then_by_unit(full_api):
    api, _folder, state = full_api
    assert state["result"]["row_count"] == 60  # six chunks of ten, one excluded

    api.set_scope(["sys_key=123"])
    assert api.wait_for_folder_load(timeout=180)["result"]["row_count"] == 50

    api.set_scope(["sys_key=123"], "rec_b")
    narrowed = api.wait_for_folder_load(timeout=180)

    assert [Path(p).name for p in narrowed["result"]["used_files"]] == [
        "part-2.parquet",
        "part-1.parquet",
    ]


def test_a_hive_folder_scopes_by_its_partition(shape_hive_api):
    """No series level, so the step below a group is the FILE — and that
    still scopes."""
    api, _folder = shape_hive_api

    assert api.get_scope()["options"] == ["day=2026-03-01", "day=2026-03-02"]
    api.set_scope(["day=2026-03-01"])
    state = api.wait_for_folder_load(timeout=180)

    assert state["result"]["row_count"] == 20
    scope = api.get_scope()
    assert scope["option_kind"] == "unit"
    assert scope["options"] == ["part0.parquet", "part1.parquet"]


def test_an_impossible_scope_is_an_error_not_an_empty_load(full_api):
    api, _folder, _state = full_api

    assert "error" in api.set_scope(["sys_key=nope"])
    assert "error" in api.set_scope(["sys_key=123"], "nope")
    assert "error" in api.set_scope(["a", "b", "c"])


def test_scope_methods_are_inert_without_a_structure(tmp_path):
    folder = build_flat(tmp_path / "flat")
    api = Api(settings_dir=tmp_path / "settings")
    api.discover_folder(str(folder))

    assert api.get_scope() == {"active": False}
    assert "error" in api.set_scope(["x"])
    assert "error" in api.start_scope_load()


def test_moving_between_scopes_reads_nothing_twice(full_api):
    api, _folder, _state = full_api

    api.set_scope(["sys_key=123"])
    stats = api.wait_for_folder_load(timeout=180)["result"]["load_stats"]

    assert stats["files_scanned"] == 0
    assert stats["files_from_cache"] == 5


# --- what each check may assume, per shape --------------------------------


def test_a_single_csv_with_a_time_column_runs_the_time_series_check(tmp_path):
    """The precondition is a timestamp column and NOTHING else — no
    structure, no series, no folder."""
    path = tmp_path / "one.csv"
    pl.DataFrame(
        {"ts": [1, 2, None, 4], "value": [1.0, 2.0, 3.0, 4.0]},
        schema={"ts": pl.Int64, "value": pl.Float64},
    ).write_csv(path)
    api = Api(settings_dir=tmp_path / "settings")
    api.load_file(str(path))

    assert api.get_capabilities()["timestamp_column"] == "ts"
    statuses = _statuses(api)

    # EVERY timestamp check runs — including the time-series family, which
    # needs nothing but a time column.
    for check_id in _timestamp_checks():
        assert statuses[check_id] == "done", check_id
    assert statuses["structure"] == "skipped"
    assert statuses["group_balance"] == "skipped"
    # ...and the ones that need a series of chunks do not.
    for check_id in _series_checks():
        assert statuses[check_id] == "skipped", check_id
    ids = {f["check_id"] for f in api.run_instant_checks()["findings"]}
    assert "timestamp_integrity" in ids


def test_a_single_csv_without_a_time_column_skips_it(tmp_path):
    path = tmp_path / "one.csv"
    pl.DataFrame({"speed": [1, 2], "label": ["a", "b"]}).write_csv(path)
    api = Api(settings_dir=tmp_path / "settings")
    api.load_file(str(path))

    statuses = _statuses(api)

    for check_id in _timestamp_checks():
        assert statuses[check_id] == "skipped", check_id
    assert api.get_capabilities()["timestamp_column"] is None


@pytest.mark.parametrize("shape", ["flat", "nested"], indirect=True)
def test_the_structural_checks_are_skipped_on_a_flat_input(shape, tmp_path):
    """Skipped, not failed. A check whose precondition is absent had
    nothing to look at.

    `timestamp_integrity` is deliberately NOT in this list: these fixtures
    have a `ts` column, and a time-series check requires a timestamp column
    and nothing else. That it runs here — on a folder with no structure at
    all — is the point, not an exception to it.
    """
    api = Api(settings_dir=tmp_path / "settings")
    _open(api, shape)

    statuses = _statuses(api)

    for check_id in {"structure", "group_balance"} | _series_checks():
        assert statuses[check_id] == "skipped", check_id
    assert statuses["timestamp_integrity"] == "done"
    assert not any(status == "error" for status in statuses.values())


def test_every_conditional_check_is_skipped_when_nothing_applies(tmp_path):
    """A flat folder of files with no time column satisfies none of the
    four preconditions, and all four say so."""
    folder = tmp_path / "plain"
    folder.mkdir()
    for index in range(2):
        pl.DataFrame({"speed": [1.0, 2.0], "label": ["a", "b"]}).write_parquet(
            folder / f"part{index}.parquet"
        )
    api = Api(settings_dir=tmp_path / "settings")
    _open(api, folder)

    statuses = _statuses(api)

    for check_id in _conditional_checks():
        assert statuses[check_id] == "skipped", check_id
    assert not any(status == "error" for status in statuses.values())


def test_a_hive_folder_skips_the_chunk_boundary_check(shape_hive_api):
    """No series level means a unit is one file, and the boundary between
    chunks is not a thing that exists. The group check DOES run — the hive
    key is a grouping column by definition."""
    api, _folder = shape_hive_api

    statuses = _statuses(api)

    for check_id in _series_checks():
        assert statuses[check_id] == "skipped", check_id
    assert statuses["structure"] == "done"
    assert statuses["group_balance"] == "done"
    assert api.get_capabilities()["grouping_column"] == "day"


def test_a_full_tree_runs_every_conditional_check(full_api):
    api, _folder, _state = full_api

    statuses = _statuses(api)

    for check_id in _conditional_checks():
        assert statuses[check_id] == "done", check_id


def test_structure_problems_reach_the_findings_tab(full_api):
    api, _folder, _state = full_api
    api.start_chart_build()
    api.wait_for_chart_build(timeout=300)

    ids = {f["check_id"] for f in api.run_instant_checks()["findings"]}

    assert "chunk_order_vs_time" in ids
    assert "deviant_chunk_schema" in ids


def test_the_skip_reason_says_what_was_missing(tmp_path):
    """"This check did not run, and here is why" is information; a green
    tick would be a lie."""
    folder = build_flat(tmp_path / "flat")
    api = Api(settings_dir=tmp_path / "settings")
    _open(api, folder)
    api.start_chart_build()
    api.wait_for_chart_build(timeout=300)

    checks = {c["id"]: c for c in api.run_instant_checks()["checks"]}

    assert checks["missing_chunk"]["reason"] == "needs a series of chunks"
    assert checks["group_balance"]["reason"] == "needs a grouping column"


def test_any_column_can_be_nominated_as_the_grouping_column(tmp_path):
    """"A directory role, a hive key, or any column the user picks" — the
    third is not a lesser case, and a dataset can be grouped by something
    that was never a directory."""
    path = tmp_path / "one.csv"
    pl.DataFrame(
        {"label": ["a"] * 50 + ["b"], "value": list(range(51))}
    ).write_csv(path)
    api = Api(settings_dir=tmp_path / "settings")
    api.load_file(str(path))
    api.get_chart_data()

    assert api.get_capabilities()["grouping_column"] is None
    result = api.set_grouping_column("label")

    assert result["grouping_column"] == "label"
    statuses = _statuses(api)
    assert statuses["group_balance"] == "done"
    assert "group_balance" in {f["check_id"] for f in api.run_instant_checks()["findings"]}


def test_nominating_a_column_that_is_not_there_is_an_error(tmp_path):
    path = tmp_path / "one.csv"
    pl.DataFrame({"a": [1, 2]}).write_csv(path)
    api = Api(settings_dir=tmp_path / "settings")
    api.load_file(str(path))

    assert "error" in api.set_grouping_column("nope")


# --- flat inputs are UNCHANGED --------------------------------------------
#
# The generality requirement, stated as a test. A folder with no structure
# must produce exactly what it produced before any of this existed: the
# same frame, the same columns, the same charts — not "equivalent", equal.


@pytest.mark.parametrize("shape", ["flat", "nested"], indirect=True)
def test_a_flat_input_gains_no_derived_columns(shape, tmp_path):
    api = Api(settings_dir=tmp_path / "settings")
    state = _open(api, shape)

    assert [c["name"] for c in state["result"]["columns"]] == ["ts", "value"]
    assert set(state["result"]["preview_rows"][0]) == {"ts", "value"}
    assert "scope" not in state["result"]
    assert "structure_findings" not in state["result"]


@pytest.mark.parametrize("shape", ["flat", "nested"], indirect=True)
def test_a_flat_inputs_charts_equal_the_reference_payload(shape, tmp_path):
    """Byte for byte against build_chart_payload's own per-file path — the
    implementation that predates every structure feature and is still the
    one a flat folder takes."""
    from ml_data_view.chart_data import build_chart_payload
    from ml_data_view.folder_loader import discover_files, load_files

    api = Api(settings_dir=tmp_path / "settings")
    _open(api, shape)
    api.start_chart_build()
    produced = api.wait_for_chart_build(timeout=300)["result"]

    files = discover_files(shape)
    reference = build_chart_payload(load_files(files), files=files)

    for key in (
        "numeric_histograms",
        "categorical_bars",
        "text_length_histograms",
        "value_frequency_histograms",
    ):
        assert produced[key] == reference[key], key
    # No synthesized bars at all: there is no structure to synthesize from.
    assert not any(bar.get("derived") for bar in produced["categorical_bars"])


def test_a_flat_folder_still_loads_through_the_schema_group_path(tmp_path):
    """Right down to the group chooser's own payload."""
    folder = build_flat(tmp_path / "flat")
    api = Api(settings_dir=tmp_path / "settings")

    discovery = api.discover_folder(str(folder))

    assert discovery["groups"][0]["file_count"] == 3
    assert discovery["default_group_index"] == 0
    assert "excluded" in discovery
    api.start_folder_load(0)
    assert api.wait_for_folder_load(timeout=180)["result"]["row_count"] == 30


def test_a_single_file_loads_exactly_as_before(tmp_path):
    path = tmp_path / "one.csv"
    pl.DataFrame({"ts": [1, 2], "value": [1.0, 2.0]}).write_csv(path)
    api = Api(settings_dir=tmp_path / "settings")

    info = api.load_file(str(path))

    assert [c["name"] for c in info["columns"]] == ["ts", "value"]
    assert info["source_type"] == "file"
    assert info["preview_rows"] == [{"ts": 1, "value": 1.0}, {"ts": 2, "value": 2.0}]


# --- the chunk is still the cache unit ------------------------------------


def test_each_chunk_caches_its_own_time_bounds(full_api):
    from ml_data_view.profile_cache import SECTION_LOAD, ProfileCache

    api, folder, _state = full_api
    chunk = folder / "sys_key=123" / "rec_b" / "part-1.parquet"

    with ProfileCache(api._cache_path()) as cache:
        payload = cache.get(chunk)

    assert payload[SECTION_LOAD]["time_bounds"]["ts"] == [BASE_TS + 9_000, BASE_TS + 9_009]


def test_the_cache_is_keyed_on_files_whatever_the_shape(shape_hive_api):
    """A chunk is one file, so a hive folder with no series caches exactly
    as a full tree does — four entries for four files."""
    from ml_data_view.profile_cache import ProfileCache

    api, _folder = shape_hive_api

    with ProfileCache(api._cache_path()) as cache:
        assert cache.entry_count() == 4


# --- derived columns, named by the mapping --------------------------------


def test_derived_columns_are_named_by_the_structure(full_api):
    api, _folder, state = full_api

    names = [c["name"] for c in state["result"]["columns"]]
    assert names == ["ts", "value", "sys_key", "unit_id", "chunk_index", "chunk_file"]
    assert state["result"]["preview_rows"][0]["sys_key"] == "sys_key=123"


def test_a_hive_folder_gets_its_partition_as_a_column(shape_hive_api):
    api, _folder = shape_hive_api

    info = api._dataset_info
    names = [c["name"] for c in info["columns"]]

    assert names == ["ts", "value", "day", "unit_id", "chunk_index", "chunk_file"]


def test_the_charts_get_a_bar_per_group_and_unit(full_api):
    api, _folder, _state = full_api
    api.start_chart_build()
    payload = api.wait_for_chart_build(timeout=300)["result"]

    derived = {bar["column"]: bar for bar in payload["categorical_bars"] if bar.get("derived")}

    assert set(derived) == {"sys_key", "unit_id"}
    assert dict(zip(derived["sys_key"]["categories"], derived["sys_key"]["counts"])) == {
        "sys_key=123": 50,
        "sys_key=456": 10,
    }


def test_a_search_can_filter_on_a_derived_column(full_api):
    api, _folder, _state = full_api

    result = api.run_search([{"column": "unit_id", "operator": "is", "values": ["rec_b"]}])

    assert result["rows"]
    assert {row["unit_id"] for row in result["rows"]} == {"rec_b"}


# --- the flat fallback explains itself ------------------------------------
#
# "unrecognizable tree -> treated as flat with a note". Silently flattening
# someone's carefully arranged directories, with no word about it, is how
# they conclude the app cannot read their data.


def test_an_unrecognisable_tree_loads_flat_with_a_note(tmp_path):
    folder = build_nested(tmp_path / "nested")
    api = Api(settings_dir=tmp_path / "settings")

    discovery = api.discover_folder(str(folder))

    assert discovery["source"] == "flat"
    assert "no group or series level was recognised" in discovery["note"]
    assert "2 directory level(s)" in discovery["note"]
    # The levels exist and are mappable; the code just declined to guess.
    assert discovery["mappable"] is True
    assert len(discovery["structure_proposal"]["levels"]) == 2


def test_a_folder_of_files_has_nothing_to_explain(tmp_path):
    """A flat folder is not a tree that failed to be recognised, so there
    is no note — the app behaves exactly as it always has, silently."""
    folder = build_flat(tmp_path / "flat")
    api = Api(settings_dir=tmp_path / "settings")

    discovery = api.discover_folder(str(folder))

    assert discovery["note"] == ""
    assert discovery["mappable"] is False


def test_declining_a_structure_says_so(tmp_path):
    folder = build_full(tmp_path / "full")
    api = Api(settings_dir=tmp_path / "settings")
    _confirm(api, api.discover_folder(str(folder)))

    flat = api.clear_structure()

    assert "you chose not to use this folder's structure" in flat["note"]
    assert flat["mappable"] is True


# --- the series rule, through the Api -------------------------------------


def test_a_series_with_no_group_above_it_is_detected(tmp_path):
    """Evidence, not position: a run of chunk directories at the top of a
    folder is a series even with nothing above it."""
    folder = tmp_path / "runs"
    for run, offset in (("run_a", 0), ("run_b", 5_000)):
        for index in range(3):
            _chunk(folder / run / f"part-{index}.parquet", BASE_TS + offset + index * 100)

    api = Api(settings_dir=tmp_path / "settings")
    _open(api, folder)

    structure = api.get_structure()
    assert structure["has_groups"] is False
    assert structure["has_series"] is True
    assert structure["unit_count"] == 2
    assert api.get_capabilities()["has_series"] is True
    # No grouping column: there are no group levels and nothing was picked.
    assert api.get_capabilities()["grouping_column"] is None


def test_a_series_with_no_groups_still_scopes_to_one_unit(tmp_path):
    folder = tmp_path / "runs"
    for run, offset in (("run_a", 0), ("run_b", 5_000)):
        for index in range(3):
            _chunk(folder / run / f"part-{index}.parquet", BASE_TS + offset + index * 100)
    api = Api(settings_dir=tmp_path / "settings")
    _open(api, folder)

    assert api.get_scope()["options"] == ["run_a", "run_b"]
    api.set_scope([], "run_b")
    state = api.wait_for_folder_load(timeout=180)

    assert state["result"]["row_count"] == 30
    assert all("run_b" in path for path in state["result"]["used_files"])


# --- labels become the derived column names -------------------------------


def test_labelling_the_levels_names_the_derived_columns(tmp_path):
    """The motivating tree, labelled by its owner, produces exactly the
    columns its owner would have written by hand — with none of those
    words in the code."""
    folder = build_full(tmp_path / "full")
    api = Api(settings_dir=tmp_path / "settings")
    discovery = api.discover_folder(str(folder))
    _confirm(api, discovery, labels={0: "System", 1: "Recording"}, root_label="Fleet")
    api.start_scope_load()
    state = api.wait_for_folder_load(timeout=180)

    names = [column["name"] for column in state["result"]["columns"]]
    assert names[-4:] == ["system_id", "recording_id", "chunk_index", "chunk_file"]
    assert state["result"]["preview_rows"][0]["system_id"] == "sys_key=123"
    # And the breadcrumb reads in those words too.
    assert [c["label"] for c in api.get_scope()["crumbs"]] == ["Fleet"]


def test_different_labels_give_different_columns(tmp_path):
    """Nothing about the names is fixed — they follow the mapping."""
    folder = build_full(tmp_path / "full")
    api = Api(settings_dir=tmp_path / "settings")
    discovery = api.discover_folder(str(folder))
    _confirm(api, discovery, labels={0: "Site", 1: "Trip"})
    api.start_scope_load()
    state = api.wait_for_folder_load(timeout=180)

    names = [column["name"] for column in state["result"]["columns"]]
    assert names[-4:] == ["site_id", "trip_id", "chunk_index", "chunk_file"]
