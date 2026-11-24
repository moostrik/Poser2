@REM Writes apps\smile\data\comfy\smile_test.json: the app's workflow for tests in the ComfyUI page (drag the file in)
@REM arguments pass through: apps\smile\scripts\make_test_workflow.bat -p apps/smile/data/settings/Grintest.json -s b

@echo off

cd /d "%~dp0..\..\..\"
.venv\Scripts\python.exe -m apps.smile.comfy.testflow %*
pause
