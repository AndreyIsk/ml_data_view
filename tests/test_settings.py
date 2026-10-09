from ml_data_view.jobs import default_memory_ceiling_mb
from ml_data_view.settings import (
    DEFAULT_BANNER_COLOR,
    MIN_MEMORY_CEILING_MB,
    THEMES,
    AppSettings,
    load_settings,
    resolve_memory_ceiling_mb,
    save_settings,
    settings_file_path,
)


def test_load_settings_returns_defaults_when_no_file_exists(tmp_path):
    settings = load_settings(tmp_path)

    assert settings == AppSettings(
        classification_banner_enabled=False,
        classification_banner_label="",
        classification_banner_color=DEFAULT_BANNER_COLOR,
        audit_log_enabled=False,
    )


def test_save_then_load_round_trips_all_fields(tmp_path):
    settings = AppSettings(
        classification_banner_enabled=True,
        classification_banner_label="TOP SECRET // NOFORN",
        classification_banner_color="#d03b3b",
        audit_log_enabled=True,
    )

    save_settings(settings, tmp_path)
    loaded = load_settings(tmp_path)

    assert loaded == settings


def test_save_settings_writes_file_at_settings_file_path(tmp_path):
    save_settings(AppSettings(audit_log_enabled=True), tmp_path)

    assert settings_file_path(tmp_path).exists()


def test_save_settings_creates_missing_parent_directories(tmp_path):
    nested_dir = tmp_path / "does" / "not" / "exist" / "yet"

    save_settings(AppSettings(), nested_dir)

    assert settings_file_path(nested_dir).exists()


def test_load_settings_merges_partial_file_with_defaults(tmp_path):
    settings_file_path(tmp_path).parent.mkdir(parents=True, exist_ok=True)
    settings_file_path(tmp_path).write_text('{"audit_log_enabled": true}', encoding="utf-8")

    loaded = load_settings(tmp_path)

    assert loaded.audit_log_enabled is True
    assert loaded.classification_banner_enabled is False
    assert loaded.classification_banner_color == DEFAULT_BANNER_COLOR


def test_load_settings_ignores_unknown_fields_in_file(tmp_path):
    settings_file_path(tmp_path).parent.mkdir(parents=True, exist_ok=True)
    settings_file_path(tmp_path).write_text(
        '{"audit_log_enabled": true, "future_field": "surprise"}', encoding="utf-8"
    )

    loaded = load_settings(tmp_path)

    assert loaded.audit_log_enabled is True


def test_load_settings_returns_defaults_on_corrupt_json(tmp_path):
    settings_file_path(tmp_path).parent.mkdir(parents=True, exist_ok=True)
    settings_file_path(tmp_path).write_text("not valid json {{{", encoding="utf-8")

    loaded = load_settings(tmp_path)

    assert loaded == AppSettings()


# --- saved searches ---


def test_saved_search_round_trips_through_the_settings_file(tmp_path):
    from ml_data_view.settings import upsert_saved_search

    settings = upsert_saved_search(
        AppSettings(), "high tracks", [{"column": "height", "operator": ">=", "values": ["1000"]}], ["height"]
    )
    save_settings(settings, tmp_path)

    reloaded = load_settings(tmp_path)
    assert len(reloaded.saved_searches) == 1
    assert reloaded.saved_searches[0]["name"] == "high tracks"
    assert reloaded.saved_searches[0]["columns"] == ["height"]


def test_saving_an_existing_name_overwrites_rather_than_duplicating():
    from ml_data_view.settings import upsert_saved_search

    settings = upsert_saved_search(AppSettings(), "s", [{"a": 1}], None)
    settings = upsert_saved_search(settings, "s", [{"b": 2}], None)

    assert len(settings.saved_searches) == 1
    assert settings.saved_searches[0]["conditions"] == [{"b": 2}]


def test_removing_a_saved_search_and_removing_a_missing_one(tmp_path):
    from ml_data_view.settings import remove_saved_search, upsert_saved_search

    settings = upsert_saved_search(AppSettings(), "a", [], None)
    settings = upsert_saved_search(settings, "b", [], None)

    settings = remove_saved_search(settings, "a")
    assert [s["name"] for s in settings.saved_searches] == ["b"]

    # A stale UI list may ask to remove something already gone.
    assert remove_saved_search(settings, "gone").saved_searches == settings.saved_searches


def test_an_unnamed_saved_search_is_rejected():
    import pytest

    from ml_data_view.settings import upsert_saved_search

    with pytest.raises(ValueError):
        upsert_saved_search(AppSettings(), "   ", [], None)


# --- theme ------------------------------------------------------------


def test_theme_defaults_to_system(tmp_path):
    """First launch should match the machine, not impose a choice nobody
    made."""
    assert load_settings(tmp_path).theme == "system"


def test_theme_round_trips(tmp_path):
    save_settings(AppSettings(theme="dark"), tmp_path)

    assert load_settings(tmp_path).theme == "dark"


def test_an_unrecognised_theme_falls_back_to_system(tmp_path):
    """A hand-edited or future settings file must not leave the app with an
    unresolvable theme — every other field already tolerates that."""
    settings_file_path(tmp_path).parent.mkdir(parents=True, exist_ok=True)
    settings_file_path(tmp_path).write_text('{"theme": "midnight"}', encoding="utf-8")

    assert load_settings(tmp_path).theme == "system"


def test_every_valid_theme_is_accepted(tmp_path):
    for theme in THEMES:
        save_settings(AppSettings(theme=theme), tmp_path)
        assert load_settings(tmp_path).theme == theme


# --- memory ceiling ------------------------------------------------------


def test_memory_ceiling_defaults_to_automatic():
    """Stored as None rather than as a resolved number, so the default
    follows the hardware: a settings file copied to a bigger machine, or
    one written before more RAM was installed, must not pin an old ceiling
    forever."""
    assert AppSettings().memory_ceiling_mb is None


def test_a_memory_ceiling_below_the_minimum_is_raised_to_it():
    """Clamped, not rejected: the field is a number box, and someone who
    types 1 wants the smallest ceiling available, not an error."""
    assert AppSettings.from_dict({"memory_ceiling_mb": 512}).memory_ceiling_mb == (
        MIN_MEMORY_CEILING_MB
    )


def test_clearing_the_memory_ceiling_returns_to_automatic():
    assert AppSettings.from_dict({"memory_ceiling_mb": 0}).memory_ceiling_mb is None
    assert AppSettings.from_dict({"memory_ceiling_mb": None}).memory_ceiling_mb is None


def test_a_nonsense_memory_ceiling_falls_back_to_automatic():
    """A hand-edited or future-version settings file must not leave the job
    runner with a ceiling it cannot use."""
    assert AppSettings.from_dict({"memory_ceiling_mb": "lots"}).memory_ceiling_mb is None


def test_a_memory_ceiling_round_trips_through_the_file(tmp_path):
    save_settings(AppSettings(memory_ceiling_mb=8192), tmp_path)

    assert load_settings(tmp_path).memory_ceiling_mb == 8192


def test_resolving_the_ceiling_prefers_the_users_choice():
    assert resolve_memory_ceiling_mb(AppSettings(memory_ceiling_mb=8192)) == 8192


def test_resolving_an_automatic_ceiling_uses_this_machines_default():
    assert resolve_memory_ceiling_mb(AppSettings()) == default_memory_ceiling_mb()


# --- timeline bounds ---------------------------------------------------


def test_timeline_bounds_have_defaults():
    settings = AppSettings()

    assert settings.timeline_max_channels == 6
    assert settings.timeline_max_rows == 2_000_000


def test_timeline_bounds_round_trip_through_the_file(tmp_path):
    save_settings(AppSettings(timeline_max_channels=2, timeline_max_rows=500_000), tmp_path)

    loaded = load_settings(tmp_path)

    assert loaded.timeline_max_channels == 2
    assert loaded.timeline_max_rows == 500_000
