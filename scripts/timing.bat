@echo off
@REM Timing session: runs an app in simulation, writes logs\timing_<app>_<date>_<time>.json, then quits.
@REM   scripts\timing.bat APP [SECONDS]      (default 300 s after a 30 s warm-up)
@REM Compare two reports: .venv\Scripts\python scripts\timing_compare.py BASELINE.json NEW.json
@REM The app runs in its venv (scripts\app_venv.bat).

set APP=%1
set SECONDS=%2
if "%APP%"=="" (
    echo Usage: %~nx0 APP [SECONDS]
    echo   APP    deep_flow, hd_trio, low_flow, smile, white_space
    exit /b 1
)
if "%SECONDS%"=="" set SECONDS=300

call "%~dp0app_venv.bat" %APP% || (echo Unknown app: %APP% & exit /b 1)

cd /d "%~dp0.."
"%VENV%\Scripts\python.exe" launcher.py -app %APP% -sim -ni -timing %SECONDS%
