@REM The relay of SMILE Online on this PC (a test; the real one runs on the VPS): the phone page,
@REM the waiting line and the messages between the phones and SMILE, on port 8800.
@REM Needs relay\token.txt (one line, the same as data\online_token.txt); see relay\README.md
@echo off
set "ROOT_DIR=%~dp0.."
cd /d "%ROOT_DIR%"
"%ROOT_DIR%\.venv\Scripts\python.exe" -m relay.server --port 8800 %*
pause
