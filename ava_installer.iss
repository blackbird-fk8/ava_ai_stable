[Setup]
AppName=AVA Control Center
AppVersion=1.0
AppPublisher=Team Artemis
AppPublisherURL=https://github.com/blackbird-fk8/ava_ai_stable
DefaultDirName={autopf}\AVA Control Center
DefaultGroupName=AVA Control Center
OutputDir=C:\scare_ai\installer_output
OutputBaseFilename=AVA_Control_Center_Setup
SetupIconFile=C:\scare_ai\assets\ava_icon.ico
Compression=lzma
SolidCompression=yes
WizardStyle=modern
WizardImageFile=C:\scare_ai\assets\ava_installer_side.bmp
WizardSmallImageFile=C:\scare_ai\assets\ava_installer_small.bmp
DisableWelcomePage=no
DisableDirPage=no
DisableProgramGroupPage=yes
PrivilegesRequired=admin
UninstallDisplayIcon={app}\AVA_Control_Center.exe

[Files]
Source: "C:\scare_ai\dist\AVA_Control_Center\*"; DestDir: "{app}"; Flags: recursesubdirs ignoreversion

[Icons]
Name: "{autodesktop}\AVA Control Center"; Filename: "{app}\AVA_Control_Center.exe"
Name: "{group}\AVA Control Center"; Filename: "{app}\AVA_Control_Center.exe"

[Run]
Filename: "{app}\AVA_Control_Center.exe"; Description: "Launch AVA Control Center"; Flags: nowait postinstall skipifsilent