import json

from ml_data_view.audit_log import (
    audit_log_file_path,
    record_file_open,
    record_folder_open,
    record_report_generated,
    write_audit_entry,
)
from ml_data_view.settings import AppSettings


def test_write_audit_entry_appends_one_json_line_with_expected_fields(tmp_path):
    write_audit_entry(
        "file_open",
        "sample.csv",
        base_dir=tmp_path,
        username="jdoe",
        timestamp="2026-08-28T12:00:00+00:00",
    )

    lines = audit_log_file_path(tmp_path).read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    entry = json.loads(lines[0])
    assert entry == {
        "timestamp": "2026-08-28T12:00:00+00:00",
        "user": "jdoe",
        "action": "file_open",
        "filename": "sample.csv",
    }


def test_write_audit_entry_appends_without_overwriting_previous_entries(tmp_path):
    write_audit_entry("file_open", "a.csv", base_dir=tmp_path, username="jdoe", timestamp="t1")
    write_audit_entry("file_open", "b.csv", base_dir=tmp_path, username="jdoe", timestamp="t2")

    lines = audit_log_file_path(tmp_path).read_text(encoding="utf-8").splitlines()
    filenames = [json.loads(line)["filename"] for line in lines]
    assert filenames == ["a.csv", "b.csv"]


def test_write_audit_entry_uses_real_username_and_timestamp_by_default(tmp_path, monkeypatch):
    monkeypatch.setattr("ml_data_view.audit_log.getpass.getuser", lambda: "mocked-user")

    write_audit_entry("file_open", "sample.csv", base_dir=tmp_path)

    entry = json.loads(audit_log_file_path(tmp_path).read_text(encoding="utf-8").splitlines()[0])
    assert entry["user"] == "mocked-user"
    # ISO8601 with an explicit UTC offset, not a bare/naive timestamp.
    assert entry["timestamp"].endswith("+00:00")


def test_record_file_open_writes_entry_when_audit_log_enabled(tmp_path, monkeypatch):
    monkeypatch.setattr("ml_data_view.audit_log.getpass.getuser", lambda: "mocked-user")
    settings = AppSettings(audit_log_enabled=True)

    record_file_open("sample.csv", settings, base_dir=tmp_path)

    assert audit_log_file_path(tmp_path).exists()
    entry = json.loads(audit_log_file_path(tmp_path).read_text(encoding="utf-8").splitlines()[0])
    assert entry["action"] == "file_open"
    assert entry["filename"] == "sample.csv"
    assert entry["user"] == "mocked-user"


def test_record_file_open_writes_nothing_when_audit_log_disabled(tmp_path):
    settings = AppSettings(audit_log_enabled=False)

    record_file_open("sample.csv", settings, base_dir=tmp_path)

    assert not audit_log_file_path(tmp_path).exists()


def test_record_report_generated_writes_entry_when_audit_log_enabled(tmp_path, monkeypatch):
    monkeypatch.setattr("ml_data_view.audit_log.getpass.getuser", lambda: "mocked-user")
    settings = AppSettings(audit_log_enabled=True)

    record_report_generated("sample.csv", settings, base_dir=tmp_path)

    entry = json.loads(audit_log_file_path(tmp_path).read_text(encoding="utf-8").splitlines()[0])
    assert entry["action"] == "report_generated"
    assert entry["filename"] == "sample.csv"
    assert entry["user"] == "mocked-user"


def test_record_report_generated_writes_nothing_when_audit_log_disabled(tmp_path):
    settings = AppSettings(audit_log_enabled=False)

    record_report_generated("sample.csv", settings, base_dir=tmp_path)

    assert not audit_log_file_path(tmp_path).exists()


def test_record_folder_open_writes_entry_when_audit_log_enabled(tmp_path, monkeypatch):
    monkeypatch.setattr("ml_data_view.audit_log.getpass.getuser", lambda: "mocked-user")
    settings = AppSettings(audit_log_enabled=True)

    record_folder_open("my_dataset_folder", settings, base_dir=tmp_path)

    entry = json.loads(audit_log_file_path(tmp_path).read_text(encoding="utf-8").splitlines()[0])
    assert entry["action"] == "folder_open"
    assert entry["filename"] == "my_dataset_folder"
    assert entry["user"] == "mocked-user"


def test_record_folder_open_writes_nothing_when_audit_log_disabled(tmp_path):
    settings = AppSettings(audit_log_enabled=False)

    record_folder_open("my_dataset_folder", settings, base_dir=tmp_path)

    assert not audit_log_file_path(tmp_path).exists()
