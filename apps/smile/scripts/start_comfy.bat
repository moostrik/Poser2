@REM Starts only the Image Generator (ComfyUI), in this window
@REM extra arguments go to ComfyUI: scripts\start_comfy.bat --enable-manager

@echo off

set "COMFY_DIR=%~dp0..\ComfyUI"
set "LOG_DIR=%~dp0..\output\logs"

curl -s -o nul http://127.0.0.1:8188/system_stats
if "%ERRORLEVEL%"=="0" (
    echo ComfyUI is already running.
    exit /b 0
)

if not exist "%COMFY_DIR%\.venv\Scripts\python.exe" (
    echo ComfyUI is not installed in %COMFY_DIR%, see comfy\README.md
    exit /b 1
)

@REM no line per image from the face detector of the grin (ultralytics)
set "YOLO_VERBOSE=false"
@REM harmless deprecation warnings inside packages: three on every run, two at startup
set "PYTHONWARNINGS=ignore::FutureWarning:insightface.utils.face_align,ignore::UserWarning:torchvision.transforms.v2._deprecated,ignore::UserWarning:torchsde._brownian.brownian_interval,ignore::FutureWarning:torch.cuda,ignore::FutureWarning:torch.jit._script"

@REM a log file per start next to the app's, with a time on every line
if not exist "%LOG_DIR%" mkdir "%LOG_DIR%"
for /f %%i in ('powershell -NoProfile -Command "Get-Date -Format yyyyMMdd_HHmmss"') do set "STAMP=%%i"

pushd "%COMFY_DIR%"
.venv\Scripts\python.exe main.py --preview-method auto --verbose INFO "%LOG_DIR%\comfy_%STAMP%.log" %*
popd
