# Windows installer — design

**Date:** 2026-08-31
**Status:** approved, implementing
**Phase:** 5, part 2 (part 1 — the standalone .exe — shipped in `6209c65`)

## Purpose

Turn the loose 193MB `MLDatasetProfiler.exe` into an install a paying
customer would recognise as a real desktop application: one `Setup.exe`,
a Start Menu entry, an Add/Remove Programs entry that actually uninstalls,
and a launch that feels instant.

The benchmark named by the user is Claude Desktop. What that pattern
actually consists of on Windows — and what this spec adopts — is a
per-user install to `%LOCALAPPDATA%\Programs`, no UAC prompt at any point,
almost no wizard, and a shortcut that starts the app immediately.

## Scope

1. Switch the PyInstaller build from `onefile` to `onedir`.
2. Add version and file metadata, which the project currently has none of.
3. An Inno Setup installer, per-user, scripted and committed, with the
   full classic seven-screen wizard (destination picker included).
4. One build command producing `Setup.exe`.
5. A pre-flight browser check in the Reports tab.

**Out of scope: auto-update.** Claude Desktop has it; this should not, yet.
It is a subsystem of its own — an update feed, versioned release
artifacts, signature verification, and a background updater — and for an
on-prem, one-time-purchase tool aimed at teams who often forbid
self-updating software, it is as likely to be a liability as a feature.
Upgrades happen by running a newer `Setup.exe`, which installs over the
previous version in place.

**Out of scope: code signing itself.** The wiring is added and left
disabled; see "Signing" below.

**Out of scope: MSI / Group Policy deployment.** If an enterprise customer
later requires silent rollout via Intune or SCCM, that is a second
artifact built from the same `dist/` tree, not a change to this one.

## Why onedir, and why it comes first

`onefile` is a self-extracting archive: every launch unpacks ~193MB into a
temp directory before a line of app code runs. Measured on the dev
machine, the same work took **11.4 / 11.5 / 11.7s frozen versus 5.3 / 6.3s
from source** — roughly 5–6 seconds of pure extraction, on every single
start.

No amount of installer polish survives that. It is also the one property
that most reliably reads as "this is a script someone wrapped up", so it
is fixed before anything else.

`onedir` produces `dist/MLDatasetProfiler/` containing the executable and
an `_internal/` payload directory. Startup drops to normal application
launch time because nothing is extracted. The trade — a folder instead of
a single file — is exactly what an installer wants to lay down, and stops
mattering the moment one exists.

`sys._MEIPASS` is still set under `onedir` (it points at `_internal/`
rather than a temp directory), so `resources.resource_path()` needs no
change. That is worth stating explicitly because it is the obvious thing
to worry about, and `tests/test_resources.py` already pins the behaviour.

## Version and file metadata

The project has no version string. A consequence visible today: right-click
the built `.exe` → Properties → Details shows no product name, no
publisher, and no version. Windows also has nothing to put in Add/Remove
Programs beyond a name.

**`ml_data_view/__init__.py` gains `__version__`**, starting at `0.1.0`,
and becomes the single source of truth. Two things derive from it:

- **`build/version_info.txt`** — the Win32 VERSIONINFO resource PyInstaller
  embeds, generated at build time rather than hand-maintained. Carries
  `ProductName`, `FileDescription`, `CompanyName`, `FileVersion`,
  `ProductVersion`, `LegalCopyright`.
- **`installer/MLDatasetProfiler.iss`** — `AppVersion`, and the version
  shown in Add/Remove Programs.

A test asserts all three agree. Version drift between the executable and
the installer that ships it is the classic packaging bug: the installer
claims 0.2.0, Properties says 0.1.0, and nobody notices until a customer
reports a bug against a version that was never built.

## The installer

`installer/MLDatasetProfiler.iss`, committed, built by Inno Setup 6
(`winget install JRSoftware.InnoSetup`, package verified present at 6.7.3).

### Install layout and privileges

```
%LOCALAPPDATA%\Programs\MLDatasetProfiler\
    MLDatasetProfiler.exe
    _internal\...
    unins000.exe
```

`PrivilegesRequired=lowest` with `DefaultDirName={localappdata}\Programs\MLDatasetProfiler`.
**No UAC prompt at any point** — this is the single most important setting
in the file. It is what lets an evaluator double-click and be running,
rather than stopping to find someone with admin rights. It is also what
Claude Desktop, VS Code's User Installer, Slack and Discord all do.

The consequence, stated so it is a decision rather than a surprise: the
app is installed for the current user only and is invisible to other user
accounts on the machine. That is the correct default for this product;
per-machine deployment is the MSI follow-up, not a checkbox here.

### Wizard

The full classic seven-screen flow, the one commercial desktop software
uses:

1. **Welcome**
2. **Select Destination Location** — defaults to
   `%LOCALAPPDATA%\Programs\MLDatasetProfiler`, with a Browse button and
   the usual required/available disk-space line
3. **Select Start Menu Folder** — defaults to `ML Dataset Profiler`, with
   a "don't create a Start Menu folder" option
4. **Select Additional Tasks** — "Create a desktop shortcut", unchecked by
   default
5. **Ready to Install** — a summary of destination, Start Menu folder and
   selected tasks, with Back still available
6. **Installing** — progress
7. **Finished** — "Launch ML Dataset Profiler", checked

No licence page. There is no EULA text in the repository, and a wizard
page backed by placeholder text is worse than no page. Adding one later is
`LicenseFile=installer\EULA.txt` plus the file — one line, no structural
change.

An earlier draft of this spec specified a three-screen wizard with no
directory page, on the grounds that Claude Desktop (the reference the user
named) has essentially no wizard at all. That was the wrong reading:
Claude Desktop is a free consumer app that optimises for "running in five
seconds", whereas this is bought, installed deliberately, and may need to
go somewhere other than the system drive. The directory page is a real
control here, not ceremony.

### Destination page and per-user privileges

The directory page and `PrivilegesRequired=lowest` interact in a way worth
handling rather than discovering. Setup runs unelevated, so if the user
browses to `C:\Program Files`, the install cannot write there.

Inno's default behaviour is to fail partway through with a permissions
error. Instead, a `NextButtonClick` check on the destination page tests
whether the chosen directory is actually writable and, if not, says so
immediately:

> Setup cannot write to that folder. Choose a location inside your user
> profile, or re-run Setup as an administrator to install for all users.

Caught on the page where the choice was made, with the two ways out named.
That is the difference between an installer that guides and one that
merely refuses.

### Shortcuts and registry

- Start Menu: `ML Dataset Profiler`, icon from the bundled `otter.ico`
- Desktop shortcut only if the checkbox was ticked
- Add/Remove Programs entry with `AppPublisher`, `AppVersion`, the otter
  icon, and an estimated size — all from the same version metadata

### Upgrades and running instances

A stable `AppId` GUID, fixed for the life of the product, is what makes a
newer `Setup.exe` upgrade in place rather than installing a second copy
beside the first.

`CloseApplications=yes` so an upgrade over a running instance offers to
close it rather than failing on a locked file. This uses the Windows
Restart Manager, which detects the running executable by the files it has
open — deliberately **not** `AppMutex`, which would require the app to
create a named mutex at startup and would therefore be a change to
`main.py` for the installer's benefit. Restart Manager needs nothing from
the application, so there is no reason to add one.

### Uninstall leaves user data alone

Uninstall removes the install directory and the shortcuts. It does **not**
touch `%APPDATA%\MLDatasetProfiler`, which holds settings, saved searches,
the audit log and the diagnostics log. Those are the user's data, not the
installer's to delete — and an audit log, in particular, exists precisely
so it survives.

### Signing

`SignTool=` is present in the `.iss` but commented out, with a note naming
what to uncomment. Nothing about the installer's structure changes when a
certificate arrives; it becomes a build-time flag.

Until then `Setup.exe` is unsigned and Windows SmartScreen will show
"Windows protected your PC — Unknown publisher" on first run, requiring
*More info → Run anyway*. This is documented in PROJECT_BRIEF.md rather
than left to be discovered, because it is the largest remaining gap
between this and a commercial install, and no amount of installer work
closes it.

## Build command

`scripts/build_release.py` runs the whole pipeline and reports what it
produced:

```
python scripts/build_release.py
  -> generates build/version_info.txt from __version__
  -> PyInstaller  (onedir)      -> dist/MLDatasetProfiler/
  -> ISCC.exe     (Inno Setup)  -> dist/MLDatasetProfiler-Setup-0.1.0.exe
  -> prints both artifact paths and sizes
```

It fails loudly if Inno Setup is not installed, naming the winget command,
rather than half-succeeding and leaving a `dist/` that looks complete.

## Reports tab: a pre-flight browser check

PDF export drives a real Chromium through kaleido/choreographer, which is
**not bundled** (established in Phase 5 part 1 — `kaleido/executable/bin/`
holds only log files). On a machine with no Chromium-based browser, the
current behaviour is that Generate runs, works for a while, and then fails
with whatever kaleido raises.

The Reports tab gains a check, run when the section renders: if
`choreographer` can resolve no browser, the Generate button is disabled
and a short line explains that PDF export needs Chrome or Edge installed.
Failing before the user commits, with a sentence they can act on, is the
difference between a missing dependency and a broken feature.

In practice Windows 10 and 11 ship Edge, which choreographer detects, so
this path is rare — but "rare" is exactly when a clear message matters
most, because nobody will have seen it before.

## Testing

Packaging is verified by building, not by unit tests, so the tests here
cover only what can genuinely go wrong silently:

- **Version consistency** — `__version__`, the generated `version_info`
  resource, and the `.iss` `AppVersion` all agree.
- **`.iss` invariants** — `PrivilegesRequired=lowest` (a regression here
  reintroduces the UAC prompt), `DefaultDirName` under `{localappdata}`,
  a non-empty stable `AppId`, that the destination and Start Menu pages
  are NOT disabled (the wizard is deliberately the full one), and that the
  uninstall section contains no reference to `%APPDATA%`.
- **The spec/`resource_path` coupling** already covered in
  `tests/test_resources.py` continues to apply unchanged under onedir.

Everything else — that the installer runs, that shortcuts appear, that
Add/Remove Programs works, that an upgrade replaces in place — is
manual, and goes on the clean-machine checklist in PROJECT_BRIEF.md
alongside the Phase 5 part 1 items that are still outstanding.

## Follow-ups deliberately left open

- **Code signing**, once a certificate exists. OV (~$200–400/yr, SmartScreen
  warning fades as install reputation accrues) versus EV (~$400–600/yr,
  trusted immediately, hardware token or cloud HSM required). Azure Trusted
  Signing is cheaper (~$10/mo) but restricted to US/CA organisations with
  a 3+ year history.
- **MSI for enterprise deployment**, built from the same `dist/` tree.
- **Auto-update.** See Scope.
- **Shipping a Chromium** so PDF export is fully self-contained. Adds
  well over 150MB and a browser to keep patched; only worth it if a real
  customer turns out to lack Edge.
