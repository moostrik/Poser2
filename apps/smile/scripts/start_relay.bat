@REM The relay of SMILE Online on this PC (a test; the real one runs on the VPS): the phone page,
@REM the waiting line and the messages between the phones and SMILE, on port 8800, in a window of
@REM its own (close it, or Ctrl+C in it, to stop the relay: python runs there without a batch file, so
@REM cmd does not ask "Terminate batch job?").
@REM Needs relay\token.txt (one line, the same as data\online_token.txt); see relay\README.md
@echo off
set "ROOT_DIR=%~dp0.."
cd /d "%ROOT_DIR%"
if not exist "relay\token.txt" (
    echo No relay\token.txt: write one line of random text into it and the same into data\online_token.txt
    pause
    exit /b 1
)
start "SMILE relay" "%ROOT_DIR%\.venv\Scripts\python.exe" -m relay.server --port 8800 %*
