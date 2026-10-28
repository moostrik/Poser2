@REM The relay of SMILE Online on this PC (a test; the real one runs on the VPS): the phone page,
@REM the waiting line and the messages between the phones and SMILE, on port 8800, in a window of
@REM its own (close it, or Ctrl+C in it, to stop the relay: python runs there without a batch file, so
@REM cmd does not ask "Terminate batch job?").
@REM Needs apps\smile\relay\token.txt (one line, the same as apps\smile\data\online_token.txt); see relay\README.md
@echo off
set "ROOT_DIR=%~dp0..\..\.."
cd /d "%ROOT_DIR%"
if not exist "apps\smile\relay\token.txt" (
    echo No apps\smile\relay\token.txt: write one line of random text into it and the same into apps\smile\data\online_token.txt
    pause
    exit /b 1
)
start "SMILE relay" "%ROOT_DIR%\.venv\Scripts\python.exe" -m apps.smile.relay.server --port 8800 %*
