@REM Puts the relay on the VPS and restarts it there: the code, the page and the token (the same line as
@REM data\online_token.txt), never the VPS's own data (its passes, the visitors' files, the log).
@REM     apps\smile\scripts\deploy_relay.bat [--library] [user@host]
@REM --library also sends the music tracks and the laughter samples of this PC's relay (relay\data\tracks and
@REM laughs, about 64 MB), so the first visitors wait for no uploads; the host is the login user TransIP made,
@REM default moostrik@smile.oostrik.net (an IP before the DNS is there). The machine is prepared once by
@REM relay\deploy\setup_vps.sh.
@echo off
setlocal
set "SMILE_DIR=%~dp0.."
set "SSH=C:\Windows\System32\OpenSSH\ssh.exe"
set "HOST="
set "LIBRARY="
:args
if "%~1"=="" goto run
if /i "%~1"=="--library" (set "LIBRARY=1") else (set "HOST=%~1")
shift
goto args
:run
if "%HOST%"=="" set "HOST=moostrik@smile.oostrik.net"
if not exist "%SMILE_DIR%\relay\token.txt" (
    echo No apps\smile\relay\token.txt: the relay needs the token, the same line as apps\smile\data\online_token.txt
    exit /b 1
)
set "EXCLUDE=--exclude=relay/tests --exclude=relay/deploy --exclude=relay/__pycache__ --exclude=relay/data/relay.log* --exclude=relay/data/passes.json --exclude=relay/data/grins --exclude=relay/data/portraits"
if not defined LIBRARY set "EXCLUDE=%EXCLUDE% --exclude=relay/data"
cd /d "%SMILE_DIR%"
echo Deploying the relay to %HOST% ...
tar -cf - %EXCLUDE% relay | "%SSH%" %HOST% "sudo tar -xf - -C /home/smile && sudo chown -R smile:smile /home/smile/relay && sudo chmod 600 /home/smile/relay/token.txt && sudo systemctl restart smile-relay && sleep 2 && echo relay: $(systemctl is-active smile-relay) && sudo tail -n 3 /home/smile/relay/data/relay.log"
if errorlevel 1 (
    echo Deploy failed, see above
    exit /b 1
)
echo Done.
