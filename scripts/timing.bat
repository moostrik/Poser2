@echo off
@REM Timing session: runs an app in simulation, writes logs\timing_<app>_<date>_<time>.json, then quits.
@REM   scripts\timing.bat APP [SECONDS]      (default 300 s after a 30 s warm-up)
@REM Compare two reports: .venv\Scripts\python scripts\timing_compare.py BASELINE.json NEW.json
@REM The app runs in the venv of its OAK backend: .venv for the OAK4 apps, .venv-oakd for the OAK-D ones.

set APP=%1
set SECONDS=%2
if "%APP%"=="" (
    echo Usage: %~nx0 APP [SECONDS]
    echo   APP    deep_flow, hd_trio, low_flow, smile, white_space
    exit /b 1
)
if "%SECONDS%"=="" set SECONDS=300

set VENV=.venv
if "%APP%"=="hd_trio" set VENV=.venv-oakd
if "%APP%"=="white_space" set VENV=.venv-oakd

cd /d "%~dp0.."
"%VENV%\Scripts\python.exe" launcher.py -app %APP% -sim -ni -timing %SECONDS%
