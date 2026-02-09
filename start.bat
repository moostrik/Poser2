@REM Windows Key + R, type shell:startup, place shortcut in folder

@echo off

@REM echo Current Applications:
@REM tasklist

tasklist /FI "IMAGENAME eq pythonw.exe" 2>NUL | find /I /N "pythonw.exe">NUL
if "%ERRORLEVEL%"=="0" (
    echo Closing Phyton Apps
    taskkill /IM "pythonw.exe" /F
    timeout /t 2 /nobreak >nul
)

set "CURRENT_DIR=%~dp0"

echo Opening Main App
start "" "%CURRENT_DIR%.venv\Scripts\python.exe" launcher.py
timeout /t 2 /nobreak >nul

@REM echo Press any key to continue . . .
@REM pause >nul
echo Exit this script
timeout /t 1 /nobreak >nul
exit

@REM start cmd /k # leaves the cmd window open after finishing the command
@REM start cmd /c # closes the cmd window after finishing the command