#ifndef RepoRoot
  #define RepoRoot ".."
#endif
#ifndef AppId
  #define AppId "ClusterLens"
#endif
#ifndef AppName
  #define AppName "ClusterLens"
#endif
#ifndef AppVersion
  #define AppVersion "0.4.0a1"
#endif
#ifndef BuildVariant
  #define BuildVariant "cpu"
#endif
#ifndef AppPublisher
  #define AppPublisher "d3v4shish"
#endif
#ifndef AppExeName
  #define AppExeName "ClusterLens.exe"
#endif
#ifndef SourceDir
  #define SourceDir "..\\dist\\production\\windows-x64"
#endif
#ifndef OutputDir
  #define OutputDir "..\\dist\\installer"
#endif

[Setup]
AppId={#AppId}
AppName={#AppName}
AppVersion={#AppVersion}
AppPublisher={#AppPublisher}
DefaultDirName={localappdata}\Programs\{#AppName}
DefaultGroupName={#AppName}
UninstallDisplayIcon={app}\{#AppExeName}
SetupIconFile={#RepoRoot}\apps\pyqt_production\assets\app_icon.ico
OutputDir={#OutputDir}
OutputBaseFilename=ClusterLens-{#AppVersion}-{#BuildVariant}-setup
Compression=lzma2
SolidCompression=yes
WizardStyle=modern
PrivilegesRequired=lowest
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
DisableProgramGroupPage=yes
UsePreviousAppDir=yes

[Tasks]
Name: "desktopicon"; Description: "Create a desktop shortcut"; GroupDescription: "Additional shortcuts:"; Flags: unchecked

[Files]
Source: "{#SourceDir}\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs

[Icons]
Name: "{autoprograms}\{#AppName}"; Filename: "{app}\{#AppExeName}"
Name: "{autodesktop}\{#AppName}"; Filename: "{app}\{#AppExeName}"; Tasks: desktopicon

[Run]
Filename: "{app}\{#AppExeName}"; Description: "Launch {#AppName}"; Flags: nowait postinstall skipifsilent
