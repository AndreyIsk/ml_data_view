; Inno Setup script for ML Dataset Profiler.
;
; Built by scripts/build_release.py, which passes MyAppName/MyAppVersion/
; MyAppPublisher in with /D so no release number is written down twice.
; Design: docs/superpowers/specs/2026-08-31-windows-installer-design.md
;
; Build by hand with:
;   "%LOCALAPPDATA%\Programs\Inno Setup 6\ISCC.exe" ^
;       /DMyAppName="ML Dataset Profiler" /DMyAppVersion=0.1.0 ^
;       /DMyAppPublisher="ML Dataset Profiler" installer\MLDatasetProfiler.iss

#ifndef MyAppName
  #define MyAppName "ML Dataset Profiler"
#endif
#ifndef MyAppVersion
  ; Deliberately not a plausible-looking number: if this ever ends up in a
  ; shipped installer it means the build did not pass /DMyAppVersion, and
  ; that should be obvious rather than silently wrong.
  #define MyAppVersion "0.0.0-UNSET"
#endif
#ifndef MyAppPublisher
  #define MyAppPublisher "ML Dataset Profiler"
#endif

#define MyAppExeName "MLDatasetProfiler.exe"

[Setup]
; AppId is the identity Windows uses to recognise an existing install.
; FIXED FOR THE LIFE OF THE PRODUCT — changing it makes a new Setup.exe
; install a second copy beside the old one instead of upgrading it.
AppId={{7F3C9A21-4E8B-4C6D-9F52-1A7D3B8E5C40}
AppName={#MyAppName}
AppVersion={#MyAppVersion}
AppVerName={#MyAppName} {#MyAppVersion}
AppPublisher={#MyAppPublisher}
VersionInfoVersion={#MyAppVersion}

; --- per-user install, no UAC prompt ---------------------------------
; The single most important pair of settings in this file. Setup runs
; unelevated and installs under the user's own profile, so an evaluator
; can double-click and be running rather than stopping to find an admin.
; Same pattern as Claude Desktop, VS Code's User Installer, Slack.
; Consequence, accepted: the app is installed for this user only.
PrivilegesRequired=lowest
DefaultDirName={localappdata}\Programs\MLDatasetProfiler
DefaultGroupName={#MyAppName}
UninstallDisplayName={#MyAppName}
UninstallDisplayIcon={app}\{#MyAppExeName}

OutputDir=..\dist
OutputBaseFilename=MLDatasetProfiler-Setup-{#MyAppVersion}
SetupIconFile=..\assets\otter.ico
Compression=lzma2/max
SolidCompression=yes
WizardStyle=modern
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible

; An upgrade over a running instance offers to close it rather than
; failing on a locked file. Uses the Windows Restart Manager, which
; detects the running executable by the files it holds open — deliberately
; not AppMutex, which would mean adding a named mutex to main.py purely
; for the installer's benefit.
CloseApplications=yes
RestartApplications=no

; --- signing ----------------------------------------------------------
; UNSIGNED. Windows SmartScreen shows "Windows protected your PC — Unknown
; publisher" on first run and the user must click More info > Run anyway.
; No installer setting closes that gap; it needs a code-signing
; certificate. When one exists, configure a "signtool" entry in Inno
; Setup's IDE (Tools > Configure Sign Tools) and uncomment:
; SignTool=signtool
; SignedUninstaller=yes

[Languages]
Name: "english"; MessagesFile: "compiler:Default.isl"

[Tasks]
Name: "desktopicon"; Description: "{cm:CreateDesktopIcon}"; GroupDescription: "{cm:AdditionalIcons}"; Flags: unchecked

[Files]
; The whole PyInstaller onedir tree. recursesubdirs picks up _internal\,
; which holds the Python runtime, polars/pyarrow/sklearn, the frontend and
; the assets — everything resources.resource_path() resolves at runtime.
Source: "..\dist\MLDatasetProfiler\{#MyAppExeName}"; DestDir: "{app}"; Flags: ignoreversion
Source: "..\dist\MLDatasetProfiler\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs

[Icons]
Name: "{group}\{#MyAppName}"; Filename: "{app}\{#MyAppExeName}"
Name: "{group}\{cm:UninstallProgram,{#MyAppName}}"; Filename: "{uninstallexe}"
Name: "{autodesktop}\{#MyAppName}"; Filename: "{app}\{#MyAppExeName}"; Tasks: desktopicon

[Run]
Filename: "{app}\{#MyAppExeName}"; Description: "{cm:LaunchProgram,{#StringChange(MyAppName, '&', '&&')}}"; Flags: nowait postinstall skipifsilent

; NOTE: there is deliberately no [UninstallDelete] entry for
; {userappdata}\MLDatasetProfiler. Settings, saved searches and the audit
; log are the user's data, not the installer's to remove — and an audit
; log in particular exists precisely so that it survives.

[Code]
function IsDirWritable(Dir: String): Boolean;
var
  Probe: String;
begin
  { Walk up to the nearest existing ancestor: the chosen directory itself
    usually does not exist yet, so testing it directly proves nothing. }
  while (Dir <> '') and not DirExists(Dir) do
    Dir := ExtractFileDir(Dir);
  if Dir = '' then
  begin
    Result := False;
    Exit;
  end;
  Probe := AddBackslash(Dir) + 'mldp_write_probe.tmp';
  Result := SaveStringToFile(Probe, 'probe', False);
  if Result then
    DeleteFile(Probe);
end;

function NextButtonClick(CurPageID: Integer): Boolean;
begin
  Result := True;
  { Setup runs unelevated (PrivilegesRequired=lowest), so browsing to
    Program Files would otherwise fail partway through the file copy with
    a bare permissions error. Catch it on the page where the choice was
    made and name both ways out. }
  if CurPageID = wpSelectDir then
  begin
    if not IsDirWritable(WizardDirValue) then
    begin
      MsgBox('Setup cannot write to that folder.'#13#10#13#10
             + 'Choose a location inside your user profile, or re-run Setup '
             + 'as an administrator to install for all users.',
             mbError, MB_OK);
      Result := False;
    end;
  end;
end;
