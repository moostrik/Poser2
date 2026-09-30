@echo off
@REM Starts (or restarts) the image players only (PLAYERS below: 1 or 2); SMILE, ComfyUI and Max are left alone.
@REM Extra arguments go to both players, e.g.:  start_players.bat -n 12 --pool 30
@REM Use  start_players.bat console  to run them with a console for the logs (Ctrl+C to stop).

set "PLAYERS=1"
set "WINDOWED=1"
set "PLAYER_1_MONITOR=1"
set "PLAYER_2_MONITOR=2"
@REM WINDOWED=1: each player in a plain window on the primary monitor (testing) instead of fullscreen on its own monitor
if "%WINDOWED%"=="1" (
    set "PLAYER_1_MONITOR=0"
    set "PLAYER_2_MONITOR=0"
    set "MODE=--window"
) else (
    set "MODE="
)
set "PLAYER_1_OSC_PORT=4115"
set "PLAYER_2_OSC_PORT=4116"

set "CURRENT_DIR=%~dp0..\"
cd /d "%CURRENT_DIR%"

set "PYTHON=%CURRENT_DIR%.venv\Scripts\pythonw.exe"
set "WINDOW="
if /i "%~1"=="console" (
    set "PYTHON=%CURRENT_DIR%.venv\Scripts\python.exe"
    set "WINDOW=cmd /k"
    shift
)
set "ARGS="
:collect
if "%~1"=="" goto collected
set "ARGS=%ARGS% %1"
shift
goto collect
:collected

echo Closing running players
powershell -NoProfile -Command "Get-CimInstance Win32_Process -Filter \"Name='pythonw.exe' or Name='python.exe'\" | Where-Object { $_.CommandLine -like '*pl_player.py*' } | ForEach-Object { Stop-Process -Id $_.ProcessId -Force }"
timeout /t 1 /nobreak >nul

echo Opening Video Player 1
start "Player 1" %WINDOW% "%PYTHON%" pl_player.py -m %PLAYER_1_MONITOR% -osc %PLAYER_1_OSC_PORT% %MODE%%ARGS%
timeout /t 2 /nobreak >nul

if %PLAYERS% lss 2 goto :eof
echo Opening Video Player 2
start "Player 2" %WINDOW% "%PYTHON%" pl_player.py -m %PLAYER_2_MONITOR% -osc %PLAYER_2_OSC_PORT% %MODE%%ARGS%
