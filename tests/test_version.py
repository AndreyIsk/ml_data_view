"""Version metadata agreement across the three places it appears.

`__version__` is the source; the .exe's VERSIONINFO resource and the
installer's AppVersion are derived. If they drift, Properties > Details
and Add/Remove Programs disagree with each other and with reality, which
is only ever discovered by a customer reporting a bug against a build
that was never made.
"""
from __future__ import annotations

import re
from pathlib import Path

import ml_data_view
from ml_data_view.resources import PROJECT_ROOT

ISS = PROJECT_ROOT / "installer" / "MLDatasetProfiler.iss"


def test_version_is_a_three_part_number():
    assert re.fullmatch(r"\d+\.\d+\.\d+", ml_data_view.__version__), ml_data_view.__version__


def test_app_identity_constants_are_set():
    assert ml_data_view.APP_NAME
    assert ml_data_view.PUBLISHER


def test_installer_reads_the_version_from_the_package():
    """The .iss must not hard-code a number of its own — it defines
    MyAppVersion from a value the build injects, so there is one source."""
    text = ISS.read_text(encoding="utf-8")

    assert "MyAppVersion" in text
    assert '#define MyAppVersion "0.0.0"' not in text


def test_installer_version_matches_the_package_when_built(tmp_path):
    from scripts.build_release import iss_defines

    defines = iss_defines()

    assert defines["MyAppVersion"] == ml_data_view.__version__
    assert defines["MyAppName"] == ml_data_view.APP_NAME


def test_version_info_resource_carries_the_same_version():
    from scripts.build_release import render_version_info

    rendered = render_version_info()

    assert ml_data_view.__version__ in rendered
    assert ml_data_view.APP_NAME in rendered
    # filevers/prodvers are a 4-tuple; 0.1.0 -> (0, 1, 0, 0)
    major, minor, patch = ml_data_view.__version__.split(".")
    assert f"({major}, {minor}, {patch}, 0)" in rendered


# --- installer invariants ----------------------------------------------


def _iss() -> str:
    return ISS.read_text(encoding="utf-8")


def test_installer_never_prompts_for_admin():
    """PrivilegesRequired=lowest is what keeps the install UAC-free. A
    regression here reintroduces the admin prompt that per-user install
    exists to avoid."""
    assert "PrivilegesRequired=lowest" in _iss()


def test_installer_defaults_under_localappdata():
    assert r"DefaultDirName={localappdata}\Programs\MLDatasetProfiler" in _iss()


def test_installer_has_a_stable_appid():
    """AppId is how Windows recognises an existing install. If it ever
    changes, a new Setup.exe installs a second copy beside the old one
    instead of upgrading it."""
    match = re.search(r"^AppId=\{\{([0-9A-F-]{36})\}?", _iss(), re.M)

    assert match, "AppId missing or not a GUID"
    assert match.group(1) == "7F3C9A21-4E8B-4C6D-9F52-1A7D3B8E5C40"


def test_the_full_wizard_pages_are_not_disabled():
    """The destination and Start Menu pages are deliberately present —
    this is the classic wizard, not the minimal one."""
    text = _iss()

    assert "DisableDirPage=yes" not in text
    assert "DisableProgramGroupPage=yes" not in text
    assert "DisableReadyPage=yes" not in text


def test_uninstall_leaves_user_data_alone():
    """Settings, saved searches and the audit log are the user's data, not
    the installer's to remove. An audit log in particular exists precisely
    so that it survives.

    An [UninstallDelete] section is the only way the .iss could delete
    them, so its absence is the whole assertion. Comments are stripped
    first: the file explains in prose why it has no such section, and that
    sentence is not a section."""
    directives = "\n".join(
        line for line in _iss().splitlines() if not line.lstrip().startswith(";")
    )

    assert "[UninstallDelete]" not in directives


def test_signing_is_wired_but_disabled():
    """SignTool must stay commented until a certificate exists — an active
    SignTool line with no configured tool fails the build."""
    text = _iss()

    assert "; SignTool=signtool" in text
    assert not re.search(r"^SignTool=", text, re.M)
