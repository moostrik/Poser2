@REM Shows how a saved visitor was made: drop a clip (apps\smile\output\grins) or its portrait (apps\smile\output\portraits) on this file
@REM the record (settings, description, seed, grin peaks) is written to apps\smile\output\temp and opened

@echo off

cd /d "%~dp0..\..\..\"
.venv\Scripts\python.exe -m apps.smile.save.record "%~1" --open
if errorlevel 1 pause
