@REM Windows Key + R, type shell:startup, place shortcut in folder

@echo off
setlocal EnableDelayedExpansion
for /f %%a in ('copy /z "%~f0" nul') do set "CR=%%a"

@REM echo Current Applications:
@REM tasklist

tasklist /FI "IMAGENAME eq python.exe" 2>NUL | find /I /N "python.exe">NUL
if "%ERRORLEVEL%"=="0" (
    echo Harmonic Dissonance is still running, please shut it down...
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

tasklist /FI "IMAGENAME eq Max.exe" 2>NUL | find /I /N "Max.exe">NUL
if "%ERRORLEVEL%"=="0" (
    echo Closing Max MSP gracefully
    taskkill /IM "Max.exe"
    set "MAX_CLOSED="
    for /L %%I in (1,1,10) do (
        if not defined MAX_CLOSED (
            timeout /t 1 /nobreak >nul
            tasklist /FI "IMAGENAME eq Max.exe" 2>NUL | find /I /N "Max.exe">NUL
            if errorlevel 1 (
                set "MAX_CLOSED=1"
                echo Waiting for Max to close... [%%I/10] done
            ) else (
                if %%I==10 (
                    echo Waiting for Max to close... [%%I/10] force killing
                    taskkill /IM "Max.exe" /F
                    timeout /t 1 /nobreak >nul
                ) else (
                    <nul set /p ="Waiting for Max to close... [%%I/10]!CR!"
                    taskkill /IM "Max.exe" >nul 2>&1
                )
            )
        )
    )
)

set "CURRENT_DIR=%~dp0..\..\"
cd /d "%CURRENT_DIR%"

echo Opening umu MAIN in Max MSP
cmd /c start "" "%CURRENT_DIR%apps\hd_trio\data\audio\umu MAIN.maxpat"
timeout /t 5 /nobreak >nul

echo Opening Main App
start "" "%CURRENT_DIR%.venv\Scripts\python.exe" launcher.py -app hd_trio
timeout /t 2 /nobreak >nul

@REM echo Press any key to continue . . .
@REM pause >nul
echo Exit this script
timeout /t 1 /nobreak >nul
exit

@REM start cmd /k # leaves the cmd window open after finishing the command
@REM start cmd /c # closes the cmd window after finishing the command