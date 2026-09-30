@REM Starts only the Image Generator (ComfyUI), in this window
@REM extra arguments go to ComfyUI: scripts\start_comfy.bat --enable-manager

@echo off

set "COMFY_DIR=%~dp0..\ComfyUI"

curl -s -o nul http://127.0.0.1:8188/system_stats
if "%ERRORLEVEL%"=="0" (
    echo ComfyUI is already running.
    exit /b 0
)

if not exist "%COMFY_DIR%\.venv\Scripts\python.exe" (
    echo ComfyUI is not installed in %COMFY_DIR%, see comfy\README.md
    exit /b 1
)

pushd "%COMFY_DIR%"
.venv\Scripts\python.exe main.py --preview-method auto %*
popd
