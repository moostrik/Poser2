@REM Writes data\comfy\smile_test.json: the app's workflow for tests in the ComfyUI page (drag the file in)
@REM arguments pass through: scripts\make_test_workflow.bat -p data/settings/Grintest.json -s b

@echo off

cd /d "%~dp0..\"
.venv\Scripts\python.exe -m modules.comfy.testflow %*
pause
