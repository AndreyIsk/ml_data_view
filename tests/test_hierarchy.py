"""Directory structure detection and scoping (ml_data_view/hierarchy.py).

The point of this module is that structure is OPTIONAL, so the tests are
organised around the shapes Otter has to keep working for:

    SINGLE      one file, no directories at all
    FLAT        a folder of files, no directories
    HIVE        key=value partitions, no series level
    NESTED      an arbitrary tree with nothing recognisable in it
    FULL        group levels plus a series of ordered chunks

`shape` below builds all five, and the parametrised tests at the end run
the same assertions across every one of them. A rule that only holds for
the shape it was written for is not a rule.
"""
from __future__ import annotations

from pathlib import Path

import polars as pl
import pytest

from ml_data_view import hierarchy as h
from ml_data_view.hierarchy import (
    ROLE_GROUP,
    ROLE_IGNORE,
    ROLE_SERIES,
    Scope,
    StructureMapping,
    build_structure,
    detect_timestamp_column,
    identity_columns,
    natural_key,
    propose_mapping,
    scope_files,
    scope_label,
    structure_fingerprint,
)

BASE_TS = 1_700_000_000


def _chunk(path: Path, start_ts: int = BASE_TS, rows: int = 10, deviant: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data: dict[str, list] = {
        "ts": [start_ts + index for index in range(rows)],
        "value": [float(index) for index in range(rows)],
    }
    if deviant:
        # Same column count, different name: the kind of drift a loose "do
        # the shapes match" check would let through.
        data.pop("value")
        data["other"] = [float(index) for index in range(rows)]
    pl.DataFrame(data).write_parquet(path)


# --- the five shapes ------------------------------------------------------


def build_single(root: Path) -> Path:
    _chunk(root / "one.parquet")
    return root


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
    """Two groups, two series each, several chunks per series.

    rec_a  chunks nested two levels deeper, named part-1/2/10 so a
           lexicographic sort would get them wrong
    rec_b  chunk names DISAGREE with time order
    rec_c  one chunk with a deviant schema
    rec_d  ordinary
    """
    for index, name in enumerate(["part-1.parquet", "part-2.parquet", "part-10.parquet"]):
        _chunk(root / "sys_key=123" / "rec_a" / "raw" / "telemetry" / name, BASE_TS + index * 100)
    _chunk(root / "sys_key=123" / "rec_b" / "part-1.parquet", BASE_TS + 9_000)
    _chunk(root / "sys_key=123" / "rec_b" / "part-2.parquet", BASE_TS + 8_000)
    _chunk(root / "sys_key=456" / "rec_c" / "part-1.parquet", BASE_TS + 1_000)
    _chunk(root / "sys_key=456" / "rec_c" / "part-2.parquet", BASE_TS + 1_100)
    _chunk(root / "sys_key=456" / "rec_c" / "part-3.parquet", BASE_TS + 1_200, deviant=True)
    _chunk(root / "sys_key=456" / "rec_d" / "part-1.parquet", BASE_TS + 2_000)
    _chunk(root / "sys_key=456" / "rec_d" / "part-2.parquet", BASE_TS + 2_100)
    return root


SHAPES = {
    "single": build_single,
    "flat": build_flat,
    "hive": build_hive,
    "nested": build_nested,
    "full": build_full,
}


@pytest.fixture
def shape(tmp_path, request):
    """Builds whichever shape the test asked for by name."""
    return SHAPES[request.param](tmp_path / request.param)


@pytest.fixture
def full(tmp_path):
    return build_full(tmp_path / "full")


def _mapping_from(proposal) -> StructureMapping:
    return StructureMapping(
        root=proposal.root,
        roles=proposal.roles(),
        fingerprint=proposal.fingerprint,
        labels=proposal.labels(),
        timestamp_column=proposal.suggested_timestamp,
        timestamp_unit=proposal.suggested_unit,
    )


@pytest.fixture
def structure(full):
    return build_structure(full, _mapping_from(propose_mapping(full)))


# --- detection, per shape -------------------------------------------------


@pytest.mark.parametrize("shape", ["single", "flat", "nested"], indirect=True)
def test_shapes_with_nothing_recognisable_are_not_detected(shape):
    """No hive level means no anchor, and the code does not invent one. The
    caller keeps its schema-group behaviour, untouched — which is what
    makes a single file, a flat folder and an arbitrary tree all still
    work exactly as they did."""
    assert propose_mapping(shape).detected is False


def test_an_undetected_tree_still_offers_its_levels(tmp_path):
    """Detection failing is not the same as there being nothing to map.
    Someone with an arbitrary tree marks the levels themselves, so the
    dialog still needs to know what levels exist."""
    proposal = propose_mapping(build_nested(tmp_path / "nested"))

    assert proposal.detected is False
    assert [level.depth for level in proposal.levels] == [0, 1]
    assert all(level.role == ROLE_IGNORE for level in proposal.levels)
    assert proposal.levels[0].examples == ["alpha", "beta"]


def test_a_hive_tree_is_detected_as_groups_with_no_series(tmp_path):
    """Hive-only partitions are a first-class shape, not a degenerate
    version of a fuller one: there is simply no level below the partition,
    so there is no series and the unit is the file."""
    proposal = propose_mapping(build_hive(tmp_path / "hive"))

    assert proposal.detected is True
    assert proposal.roles() == {0: ROLE_GROUP}
    assert proposal.levels[0].hive_key == "day"
    assert proposal.levels[0].label == "day"


def test_a_full_tree_is_detected_as_a_group_and_a_series(full):
    proposal = propose_mapping(full)

    assert proposal.detected is True
    # Four levels, because rec_a's chunks sit two directories deeper —
    # everything below the series is structure inside one dataset.
    assert proposal.roles() == {0: ROLE_GROUP, 1: ROLE_SERIES, 2: ROLE_IGNORE, 3: ROLE_IGNORE}
    assert proposal.levels[1].label == "Series"


def test_the_suggested_labels_are_generic_defaults(full):
    """"System" and "Recording" are one dataset's words. The code's
    defaults say what the ROLE is and leave the naming to the user."""
    proposal = propose_mapping(full)

    assert proposal.levels[0].label == "sys_key"  # the hive key, which IS a name
    assert proposal.levels[1].label == "Series"


def test_a_buried_hive_level_is_not_the_anchor(tmp_path):
    """A hive directory under three arbitrary levels is a partition INSIDE
    something, not the thing itself — so it does not silently become the
    top of a hierarchy."""
    root = tmp_path / "buried"
    _chunk(root / "a" / "b" / "day=2026-03-01" / "part0.parquet")

    assert propose_mapping(root).detected is False


def test_nested_hive_levels_are_all_groups(tmp_path):
    """Any number of group levels — "two systems" is not a special case of
    "one"."""
    root = tmp_path / "double"
    _chunk(root / "site=north" / "day=2026-03-01" / "part0.parquet")
    _chunk(root / "site=south" / "day=2026-03-01" / "part0.parquet")

    proposal = propose_mapping(root)

    assert proposal.roles() == {0: ROLE_GROUP, 1: ROLE_GROUP}
    assert proposal.detected is True


# --- applying a mapping, per shape ----------------------------------------


@pytest.mark.parametrize("shape", ["single", "flat", "hive", "nested", "full"], indirect=True)
def test_every_shape_can_be_mapped_without_error(shape):
    """Whatever the shape, applying its own proposal produces something
    coherent or nothing at all — never an exception."""
    proposal = propose_mapping(shape)
    structure = build_structure(shape, _mapping_from(proposal))

    if proposal.detected:
        assert structure.units
    # Units always cover every chunk they claim.
    assert all(unit.chunks for unit in structure.units)


def test_without_a_series_level_a_unit_is_one_file(tmp_path):
    """Which is exactly how the app behaved before any of this existed —
    a flat folder and a single file fall out of the same model rather than
    needing their own path."""
    root = build_hive(tmp_path / "hive")
    structure = build_structure(root, _mapping_from(propose_mapping(root)))

    assert len(structure.units) == 4
    assert all(len(unit.chunks) == 1 for unit in structure.units)
    assert structure.has_series is False
    assert sorted({unit.key[0] for unit in structure.units}) == [
        "day=2026-03-01",
        "day=2026-03-02",
    ]


def test_with_a_series_level_a_unit_is_a_directory_of_chunks(structure):
    assert structure.has_series is True
    assert sorted(unit.unit_id for unit in structure.units) == [
        "rec_a",
        "rec_b",
        "rec_c",
        "rec_d",
    ]
    assert len(structure.unit(("sys_key=123",), "rec_a").chunks) == 3


def test_a_series_with_no_groups_is_a_valid_structure(tmp_path):
    """Both roles are optional, independently. A single run of chunks with
    nothing above it is a structure too."""
    root = tmp_path / "one_run"
    for index, name in enumerate(["part-1.parquet", "part-2.parquet", "part-10.parquet"]):
        _chunk(root / "run" / name, BASE_TS + index * 100)

    structure = build_structure(
        root,
        StructureMapping(root=str(root), roles={0: ROLE_SERIES}, fingerprint="", labels={0: "Run"}),
    )

    assert len(structure.units) == 1
    assert structure.units[0].key == ()
    assert [c.path.name for c in structure.units[0].chunks] == [
        "part-1.parquet",
        "part-2.parquet",
        "part-10.parquet",
    ]
    assert structure.has_series is True
    assert structure.group_labels == []


def test_an_override_is_honoured_over_the_suggestion(full):
    """The proposal is a suggestion. Somebody who says the series level is
    one deeper than suggested gets exactly that."""
    proposal = propose_mapping(full)
    roles = dict(proposal.roles())
    roles[1] = ROLE_IGNORE
    roles[2] = ROLE_SERIES

    structure = build_structure(
        full, StructureMapping(str(full), roles, proposal.fingerprint, timestamp_column="ts")
    )

    assert [unit.unit_id for unit in structure.units] == ["raw"]


# --- labels are the user's ------------------------------------------------


def test_the_mapping_carries_the_users_words(full):
    structure = build_structure(
        full,
        StructureMapping(
            root=str(full),
            roles={0: ROLE_GROUP, 1: ROLE_SERIES},
            fingerprint="",
            labels={0: "System", 1: "Recording"},
            root_label="Fleet",
            timestamp_column="ts",
        ),
    )

    assert structure.group_labels == ["System"]
    assert structure.unit_label == "Recording"
    assert scope_label(structure, Scope())[0]["label"] == "Fleet"


def test_labels_fall_back_to_generic_defaults(full):
    """Nobody is forced to name anything."""
    structure = build_structure(
        full,
        StructureMapping(str(full), {0: ROLE_GROUP, 1: ROLE_SERIES}, "", timestamp_column="ts"),
    )

    assert structure.group_labels == ["Group"]
    assert structure.unit_label == "Series"
    assert scope_label(structure, Scope())[0]["label"] == "All"


def test_a_mapping_round_trips_through_json(full):
    """It is stored in settings.json, so depths come back as strings and
    have to survive being ints again — labels included."""
    mapping = StructureMapping(
        root=str(full),
        roles={0: ROLE_GROUP, 1: ROLE_SERIES},
        fingerprint="abc",
        labels={0: "System", 1: "Recording"},
        root_label="Fleet",
        timestamp_column="ts",
        timestamp_unit="s",
    )

    restored = StructureMapping.from_dict(mapping.to_dict())

    assert restored.roles == mapping.roles
    assert restored.labels == mapping.labels
    assert all(isinstance(depth, int) for depth in restored.roles)
    assert restored.root_label == "Fleet"
    assert restored.timestamp_column == "ts"


# --- the fingerprint ------------------------------------------------------


@pytest.mark.parametrize("shape", ["single", "flat", "hive", "nested", "full"], indirect=True)
def test_adding_a_file_never_changes_the_fingerprint(shape):
    """"Never asked again unless the structure changes" is only true if
    adding files does not count as a change — in ANY shape."""
    before = structure_fingerprint(shape, h._chunk_files(shape))
    existing = h._chunk_files(shape)[0]
    _chunk(existing.parent / "added-99.parquet")

    assert structure_fingerprint(shape, h._chunk_files(shape)) == before


def test_a_new_level_changes_the_fingerprint(full):
    before = structure_fingerprint(full, h._chunk_files(full))
    _chunk(full / "sys_key=789" / "rec_z" / "a" / "b" / "c" / "p0.parquet")

    assert structure_fingerprint(full, h._chunk_files(full)) != before


# --- chunk ordering (a SERIES concern) ------------------------------------


def test_chunks_are_ordered_naturally_by_name(structure):
    unit = structure.unit(("sys_key=123",), "rec_a")

    assert [chunk.path.name for chunk in unit.chunks] == [
        "part-1.parquet",
        "part-2.parquet",
        "part-10.parquet",
    ]
    assert unit.order_source == "name"
    assert unit.order_conflict is False


def test_natural_key_sorts_digit_runs_as_numbers():
    assert sorted(["part-10", "part-2", "part-1"], key=natural_key) == [
        "part-1",
        "part-2",
        "part-10",
    ]


def test_time_order_wins_when_it_disagrees_with_the_names(structure):
    """A filename is a label; a timestamp is a measurement."""
    unit = structure.unit(("sys_key=123",), "rec_b")

    assert [chunk.path.name for chunk in unit.chunks] == ["part-2.parquet", "part-1.parquet"]
    assert unit.order_source == "timestamp"
    assert unit.order_conflict is True
    assert [chunk.index for chunk in unit.chunks] == [0, 1]


def test_a_disagreement_raises_a_finding(structure):
    findings = [f for f in structure.findings if f["check_id"] == "chunk_order_vs_time"]

    assert len(findings) == 1
    assert "rec_b" in findings[0]["detail"]


@pytest.mark.parametrize("shape", ["single", "flat", "hive"], indirect=True)
def test_a_shape_with_one_chunk_per_unit_raises_no_ordering_findings(shape):
    """A unit of one chunk cannot be mis-ordered, so these shapes produce
    none of these findings at all — not a passing check, no check."""
    proposal = propose_mapping(shape)
    structure = build_structure(shape, _mapping_from(proposal))

    assert [f for f in structure.findings if f["check_id"] == "chunk_order_vs_time"] == []


def test_without_time_bounds_the_name_order_stands(full):
    """A partial ordering by time is not an ordering."""
    proposal = propose_mapping(full)
    structure = build_structure(
        full,
        StructureMapping(str(full), proposal.roles(), proposal.fingerprint),
        timestamp_column="",
    )
    unit = structure.unit(("sys_key=123",), "rec_b")

    assert [chunk.path.name for chunk in unit.chunks] == ["part-1.parquet", "part-2.parquet"]
    assert unit.order_conflict is False


# --- timestamp detection --------------------------------------------------


def test_a_named_numeric_column_is_a_candidate_with_a_unit(full):
    proposal = propose_mapping(full)

    assert proposal.suggested_timestamp == "ts"
    assert proposal.suggested_unit == "s"


@pytest.mark.parametrize(
    "value,unit",
    [
        (1_700_000_000, "s"),
        (1_700_000_000_000, "ms"),
        (1_700_000_000_000_000, "us"),
        (1_700_000_000_000_000_000, "ns"),
    ],
)
def test_numeric_timestamp_units_come_from_magnitude(value, unit):
    assert h.detect_unit(value) == unit


def test_a_datetime_column_wins_over_a_named_numeric_one(tmp_path):
    """The dtype IS the answer; nothing about it has to be inferred."""
    import datetime

    path = tmp_path / "day=1" / "part0.parquet"
    path.parent.mkdir(parents=True)
    pl.DataFrame(
        {
            "ts": [BASE_TS, BASE_TS + 1],
            "event_time": [datetime.datetime(2026, 3, 1), datetime.datetime(2026, 3, 2)],
        }
    ).write_parquet(path)

    proposal = propose_mapping(tmp_path)

    assert proposal.suggested_timestamp == "event_time"


def test_a_timestamp_is_detectable_from_a_schema_alone():
    """No footer, no structure — a single CSV has to be able to report a
    time column, or a time-series check could never run on one."""
    frame = pl.DataFrame({"ts": [1, 2], "value": [1.0, 2.0]})

    found = detect_timestamp_column(frame.lazy().collect_schema())

    assert found["column"] == "ts"


def test_a_schema_with_no_time_column_detects_nothing():
    frame = pl.DataFrame({"optimistic": [1, 2], "speed": [1.0, 2.0]})

    assert detect_timestamp_column(frame.lazy().collect_schema()) is None


# --- deviant chunks (also a SERIES concern) -------------------------------


def test_a_deviant_chunk_is_excluded_with_a_reason(structure):
    unit = structure.unit(("sys_key=456",), "rec_c")

    assert [chunk.path.name for chunk in unit.chunks] == ["part-1.parquet", "part-2.parquet"]
    assert [item.path.name for item in unit.excluded] == ["part-3.parquet"]
    assert "missing value" in unit.excluded[0].reason
    assert "unexpected other" in unit.excluded[0].reason


def test_the_reference_schema_is_the_majority_not_the_first_file(tmp_path):
    """"The first chunk happened to be the broken one" is a real way to
    lose a whole unit to a single bad file.

    The series role is set explicitly rather than suggested: a directory of
    mixed schemas is exactly what detection declines to call a series, and
    this is a test of what the BUILDER does once a person has said it is
    one anyway.
    """
    root = tmp_path / "maj"
    _chunk(root / "sys_key=1" / "run" / "part-1.parquet", BASE_TS, deviant=True)
    _chunk(root / "sys_key=1" / "run" / "part-2.parquet", BASE_TS + 100)
    _chunk(root / "sys_key=1" / "run" / "part-3.parquet", BASE_TS + 200)

    structure = build_structure(
        root,
        StructureMapping(
            str(root), {0: ROLE_GROUP, 1: ROLE_SERIES}, "", timestamp_column="ts"
        ),
    )

    assert [c.path.name for c in structure.units[0].chunks] == [
        "part-2.parquet",
        "part-3.parquet",
    ]
    assert [e.path.name for e in structure.units[0].excluded] == ["part-1.parquet"]


def test_schemas_may_differ_across_units(tmp_path):
    """Grouping by schema cuts a dataset across its own structure. Keep it
    as information."""
    root = tmp_path / "across"
    _chunk(root / "sys_key=1" / "run_a" / "part-1.parquet", BASE_TS)
    _chunk(root / "sys_key=1" / "run_a" / "part-2.parquet", BASE_TS + 100)
    _chunk(root / "sys_key=1" / "run_b" / "part-1.parquet", BASE_TS + 200, deviant=True)
    _chunk(root / "sys_key=1" / "run_b" / "part-2.parquet", BASE_TS + 300, deviant=True)

    structure = build_structure(root, _mapping_from(propose_mapping(root)))

    assert len(structure.units) == 2
    assert all(not unit.excluded for unit in structure.units)
    assert [group["unit_count"] for group in structure.schema_groups()] == [1, 1]


# --- scope, at every depth ------------------------------------------------


def test_an_empty_scope_is_everything(structure):
    assert len(scope_files(structure, Scope())) == 9


def test_a_group_value_narrows(structure):
    files = scope_files(structure, Scope(values=["sys_key=123"]))

    assert len(files) == 5
    assert all("sys_key=123" in str(path) for path in files)


def test_a_unit_narrows_to_its_chunks_in_final_order(structure):
    files = scope_files(structure, Scope(values=["sys_key=123"], unit_id="rec_b"))

    assert [path.name for path in files] == ["part-2.parquet", "part-1.parquet"]


def test_a_scope_that_matches_nothing_selects_nothing(structure):
    assert scope_files(structure, Scope(values=["sys_key=999"])) == []
    assert scope_files(structure, Scope(unit_id="nope")) == []


def test_scope_works_with_no_group_levels_at_all(tmp_path):
    """Depth zero is a depth. A series with nothing above it still scopes."""
    root = tmp_path / "one_run"
    _chunk(root / "run" / "part-1.parquet")
    structure = build_structure(
        root, StructureMapping(str(root), {0: ROLE_SERIES}, "", timestamp_column="ts")
    )

    assert len(scope_files(structure, Scope())) == 1
    assert len(scope_files(structure, Scope(unit_id="run"))) == 1


def test_the_breadcrumb_uses_the_mappings_labels(full):
    structure = build_structure(
        full,
        StructureMapping(
            str(full),
            {0: ROLE_GROUP, 1: ROLE_SERIES},
            "",
            labels={0: "System", 1: "Recording"},
            root_label="Fleet",
            timestamp_column="ts",
        ),
    )

    crumbs = scope_label(structure, Scope(values=["sys_key=123"], unit_id="rec_b"))

    assert [c["label"] for c in crumbs] == ["Fleet", "sys_key=123", "rec_b"]
    assert [c["level_label"] for c in crumbs] == ["", "System", "Recording"]


# --- derived columns ------------------------------------------------------


def test_derived_column_names_come_from_the_structure(structure):
    """A hive key becomes a column called after itself. No module
    downstream may hold a fixed list of these names."""
    assert structure.derived_columns() == ["sys_key", "unit_id", "chunk_index", "chunk_file"]


def test_a_labelled_group_level_names_its_own_column(tmp_path):
    root = tmp_path / "named"
    _chunk(root / "north" / "run" / "part-1.parquet")
    structure = build_structure(
        root,
        StructureMapping(
            str(root), {0: ROLE_GROUP, 1: ROLE_SERIES}, "", labels={0: "Site"}, timestamp_column="ts"
        ),
    )

    # "Site" becomes `site_id` — the label is the user's word, and the
    # column is that word as an identifier.
    assert structure.derived_columns()[0] == "site_id"


def test_a_derived_name_never_shadows_a_real_column(tmp_path):
    """A derived column must not quietly replace one the file actually
    contains."""
    root = tmp_path / "clash"
    path = root / "unit_id=1" / "run" / "part0.parquet"
    path.parent.mkdir(parents=True)
    pl.DataFrame({"unit_id": ["a", "b"], "ts": [1, 2]}).write_parquet(path)

    structure = build_structure(
        root,
        StructureMapping(str(root), {0: ROLE_GROUP, 1: ROLE_SERIES}, "", timestamp_column="ts"),
    )

    names = structure.derived_columns()
    assert "unit_id" not in names[1:] or names[1] != "unit_id"
    assert len(set(names)) == len(names)


def test_the_scope_frame_carries_the_derived_columns(structure):
    identity = h.chunk_identity(structure, Scope(values=["sys_key=123"], unit_id="rec_b"))
    frame = h.scope_lazyframe(identity)

    schema = frame.collect_schema()
    for name in structure.derived_columns():
        assert name in schema

    rows = frame.select(structure.derived_columns()).collect().to_dicts()
    assert {row["unit_id"] for row in rows} == {"rec_b"}
    # chunk_index follows the FINAL order, so the first chunk of rec_b is
    # index 0 even though its filename says part-2.
    assert rows[0]["chunk_index"] == 0
    assert rows[0]["chunk_file"] == "part-2.parquet"


def test_identity_reports_its_own_column_names(structure):
    identity = h.chunk_identity(structure, Scope())

    assert identity_columns(identity) == structure.derived_columns()


def test_identity_columns_of_nothing_is_nothing():
    """The no-structure case, which every consumer hits on a flat input."""
    assert identity_columns(None) == []
    assert identity_columns([]) == []


def test_chunk_identity_is_json_safe(structure):
    import json

    identity = h.chunk_identity(structure, Scope())

    assert json.loads(json.dumps(identity)) == identity


# --- the constraint that shapes all of it ---------------------------------


@pytest.mark.parametrize("shape", ["single", "flat", "hive", "nested", "full"], indirect=True)
def test_nothing_reads_row_data_during_discovery_or_ordering(shape, monkeypatch):
    """Detection, schema comparison, row counts and time bounds all come
    from the parquet footer — in EVERY shape. The moment that stops being
    true, a tree of thousands of chunks becomes unopenable.
    """
    collected = []
    original = pl.LazyFrame.collect

    def tracking(self, *args, **kwargs):
        result = original(self, *args, **kwargs)
        collected.append(result.height)
        return result

    monkeypatch.setattr(pl.LazyFrame, "collect", tracking)

    proposal = propose_mapping(shape)
    structure = build_structure(shape, _mapping_from(proposal))
    files = scope_files(structure, Scope())

    # Sanity: this only means anything if the work actually happened.
    assert files or not proposal.detected
    assert collected == [], f"row data was collected during discovery: {collected}"


def test_row_counts_and_time_bounds_come_from_the_footer(structure):
    unit = structure.unit(("sys_key=123",), "rec_a")

    assert unit.row_count == 30
    assert unit.first_timestamp == BASE_TS
    assert unit.last_timestamp == BASE_TS + 200 + 9


def test_an_unreadable_file_is_excluded_rather_than_fatal(full):
    """One bad file must not take a tree down."""
    (full / "sys_key=456" / "rec_c" / "broken.parquet").write_bytes(b"not parquet at all")

    structure = build_structure(full, _mapping_from(propose_mapping(full)))

    assert any(f["check_id"] == "unreadable_chunk" for f in structure.findings)
    assert len(scope_files(structure, Scope())) == 9

# --- the series rule is EVIDENCE, not position ----------------------------
#
# A level is suggested as a series when its directories mostly hold
# same-schema files with non-overlapping time ranges — all of it readable
# from parquet footers. The alternative, "whatever level sits below the
# groups", names a series on no evidence and cannot see one that has no
# group above it.


def test_a_series_is_found_with_no_group_level_above_it(tmp_path):
    """The case a positional rule structurally cannot reach."""
    root = tmp_path / "runs"
    for run, offset in (("run_a", 0), ("run_b", 5_000)):
        for index in range(3):
            _chunk(root / run / f"part-{index}.parquet", BASE_TS + offset + index * 100)

    proposal = propose_mapping(root)

    assert proposal.detected is True
    assert proposal.roles() == {0: ROLE_SERIES}
    assert proposal.levels[0].label == "Series"


def test_overlapping_time_ranges_disqualify_a_series(tmp_path):
    """Chunks of one dataset partition time; they do not cover the same
    moments twice. Overlap is the disqualifier."""
    root = tmp_path / "overlap"
    for run in ("run_a", "run_b"):
        for index in range(3):
            # Every chunk covers the SAME window.
            _chunk(root / run / f"part-{index}.parquet", BASE_TS)

    assert propose_mapping(root).detected is False


def test_a_gap_between_chunks_does_not_disqualify_a_series(tmp_path):
    """A recording can legitimately pause. Treating a pause as evidence
    against would reject most real data."""
    root = tmp_path / "gappy"
    for run in ("run_a", "run_b"):
        _chunk(root / run / "part-1.parquet", BASE_TS)
        _chunk(root / run / "part-2.parquet", BASE_TS + 1_000_000)

    assert propose_mapping(root).roles() == {0: ROLE_SERIES}


def test_mixed_schemas_disqualify_a_series(tmp_path):
    """A directory holding two shapes of file is holding two things."""
    root = tmp_path / "mixed"
    for run in ("run_a", "run_b"):
        _chunk(root / run / "part-1.parquet", BASE_TS)
        _chunk(root / run / "part-2.parquet", BASE_TS + 100, deviant=True)

    assert propose_mapping(root).detected is False


def test_one_file_per_directory_is_not_a_series(tmp_path):
    """That is just a file in a folder — no claim about ordering is being
    made by the tree."""
    root = tmp_path / "singles"
    for run in ("run_a", "run_b", "run_c"):
        _chunk(root / run / "only.parquet", BASE_TS)

    assert propose_mapping(root).detected is False


def test_one_bad_directory_does_not_cost_the_level_its_role(tmp_path):
    """A majority, not unanimity — the same reasoning as the majority
    schema rule for chunks within a unit."""
    root = tmp_path / "majority"
    for run, offset in (("run_a", 0), ("run_b", 5_000), ("run_c", 10_000)):
        for index in range(2):
            _chunk(root / run / f"part-{index}.parquet", BASE_TS + offset + index * 100)
    # One directory that is not a series at all.
    _chunk(root / "run_bad" / "a.parquet", BASE_TS)
    _chunk(root / "run_bad" / "b.parquet", BASE_TS, deviant=True)

    assert propose_mapping(root).roles()[0] == ROLE_SERIES


def test_the_shallowest_qualifying_level_wins(structure):
    """A series is the OUTERMOST directory holding one dataset's chunks;
    every level below it would qualify on the same evidence and is
    structure inside that dataset."""
    proposal = propose_mapping(structure.root)

    assert proposal.roles()[1] == ROLE_SERIES
    assert proposal.roles()[2] == ROLE_IGNORE
    assert proposal.roles()[3] == ROLE_IGNORE


def test_a_hive_level_is_never_suggested_as_a_series(tmp_path):
    """`key=value` says "this identifies something", which is a group — and
    a group of well-behaved series would otherwise qualify as one itself."""
    root = tmp_path / "hive_series"
    for day, offset in (("day=2026-03-01", 0), ("day=2026-03-02", 5_000)):
        for index in range(3):
            _chunk(root / day / f"part-{index}.parquet", BASE_TS + offset + index * 100)

    proposal = propose_mapping(root)

    assert proposal.roles() == {0: ROLE_GROUP}


def test_a_tree_with_no_timestamp_column_suggests_no_series(tmp_path):
    """The evidence is about time. With no time column there is none, and
    no evidence is not evidence of a series."""
    root = tmp_path / "timeless"
    for run in ("run_a", "run_b"):
        for index in range(3):
            path = root / run / f"part-{index}.parquet"
            path.parent.mkdir(parents=True, exist_ok=True)
            pl.DataFrame({"speed": [1.0, 2.0], "label": ["a", "b"]}).write_parquet(path)

    assert propose_mapping(root).detected is False


# --- derived column names come from the labels ----------------------------


def test_labelled_levels_name_their_own_id_columns(full):
    """Labelling the motivating tree's levels "System" and "Recording"
    produces exactly the column names its owner would have written by
    hand — with neither word appearing anywhere in the code."""
    structure = build_structure(
        full,
        StructureMapping(
            str(full),
            {0: ROLE_GROUP, 1: ROLE_SERIES},
            "",
            labels={0: "System", 1: "Recording"},
            timestamp_column="ts",
        ),
    )

    assert structure.derived_columns() == [
        "system_id",
        "recording_id",
        "chunk_index",
        "chunk_file",
    ]


def test_an_unlabelled_hive_level_keeps_its_own_key(full):
    """The key is already the name the data itself uses, and the default
    label is just that key echoed back — which is not a choice."""
    structure = build_structure(full, _mapping_from(propose_mapping(full)))

    assert structure.derived_columns()[0] == "sys_key"
    assert structure.derived_columns()[1] == "unit_id"


def test_an_unnamed_group_level_falls_back_to_its_position(tmp_path):
    root = tmp_path / "unnamed"
    for outer in ("north", "south"):
        for index in range(2):
            _chunk(root / outer / "run" / f"part-{index}.parquet", BASE_TS + index * 100)

    structure = build_structure(
        root,
        StructureMapping(str(root), {0: ROLE_GROUP, 1: ROLE_SERIES}, "", timestamp_column="ts"),
    )

    assert structure.derived_columns() == [
        "group_1_id",
        "unit_id",
        "chunk_index",
        "chunk_file",
    ]
