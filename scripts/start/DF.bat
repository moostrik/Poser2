@REM Windows Key + R, type shell:startup, place shortcut in folder

@echo off
setlocal EnableDelayedExpansion
for /f %%a in ('copy /z "%~f0" nul') do set "CR=%%a"

@REM echo Current Applications:
@REM tasklist

tasklist /FI "IMAGENAME eq python.exe" 2>NUL | find /I /N "python.exe">NUL
if "%ERRORLEVEL%"=="0" (
    echo Deep Flow is still running, please shut it down...
    set "PY_CLOSED="
    for /L %%I in (1,1,30) do (
        if not defined PY_CLOSED (
            timeout /t 1 /nobreak >nul
            tasklist /FI "IMAGENAME eq python.exe" 2>NUL | find /I /N "python.exe">NUL
            if errorlevel 1 (
                set "PY_CLOSED=1"
                echo Python closed cleanly [%%I/30]
            ) else (
                if %%I==30 (
                    echo Python still running after 30s, force killing
                    taskkill /IM "python.exe" /F
                    timeout /t 1 /nobreak >nul
                ) else (
                    <nul set /p ="Waiting for Python to shut down... [%%I/30]!CR!"
                )
            )
        )
    )
)

set "CURRENT_DIR=%~dp0..\..\"
cd /d "%CURRENT_DIR%"

echo Opening Main App
start "Main App" cmd /k "cd /d "%CURRENT_DIR%" && "%CURRENT_DIR%.venv\Scripts\python.exe" "%CURRENT_DIR%launcher.py" -app deep_flow"
timeout /t 2 /nobreak >nul

@REM echo Press any key to continue . . .
@REM pause >nul
echo Exit this script
timeout /t 1 /nobreak >nul
exit

@REM start cmd /k # leaves the cmd window open after finishing the command
@REM start cmd /c # closes the cmd window after finishing the command
