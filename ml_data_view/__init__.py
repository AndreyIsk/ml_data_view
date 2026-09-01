"""ML Dataset Profiler."""

# Single source of truth for the version. Three things derive from it and a
# test asserts they agree: the Win32 VERSIONINFO resource embedded in the
# .exe (so Properties > Details is populated), the installer's AppVersion
# (so Add/Remove Programs matches), and the Setup.exe filename.
#
# Version drift between an executable and the installer shipping it is the
# classic packaging bug — the installer claims one number, Properties says
# another, and a customer reports against a build that never existed.
__version__ = "0.1.0"

APP_NAME = "ML Dataset Profiler"
PUBLISHER = "ML Dataset Profiler"
COPYRIGHT = "Copyright (C) 2026"
