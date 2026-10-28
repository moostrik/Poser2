@echo off
@REM Timing session: runs an app in simulation, writes logs\timing_<app>_<date>_<time>.json, then quits.
@REM   scripts\timing.bat APP [SECONDS]      (default 300 s after a 30 s warm-up)
@REM Compare two reports: .venv\Scripts\python scripts\timing_compare.py BASELINE.json NEW.json

set APP=%1
set SECONDS=%2
if "%APP%"=="" (
    echo Usage: %~nx0 APP [SECONDS]
    echo   APP    hd_trio, white_space, deep_flow
    exit /b 1
)
if "%SECONDS%"=="" set SECONDS=300

cd /d "%~dp0.."
".venv\Scripts\python.exe" launcher.py -app %APP% -sim -ni -timing %SECONDS%
