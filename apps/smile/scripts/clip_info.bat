@REM Shows how a saved visitor was made: drop a clip (output\grins) or its portrait (output\portraits) on this file
@REM the record (settings, description, seed, grin peaks) is written to output\temp and opened

@echo off

cd /d "%~dp0..\"
.venv\Scripts\python.exe -m modules.save.record "%~1" --open
if errorlevel 1 pause
