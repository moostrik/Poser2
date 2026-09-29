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

@REM tasklist /FI "IMAGENAME eq Max.exe" 2>NUL | find /I /N "Max.exe">NUL
@REM IF "%ERRORLEVEL%"=="0" (
@REM     echo Closing Music Player
@REM     WMIC PROCESS where "name='Max.exe'" call terminate
@REM     @REM taskkill /IM "Max.exe" /F
@REM timeout /t 4 /nobreak >nul
@REM )

@REM taskkill /IM "python.exe" /F

set "PLAYER_1_POSITION=2000"
set "PLAYER_2_POSITION=6000"
set "PLAYER_1_OSC_PORT=4115"
set "PLAYER_2_OSC_PORT=4116"

set "CURRENT_DIR=%~dp0..\"
cd /d "%CURRENT_DIR%"

echo Opening Music Player
tasklist /FI "IMAGENAME eq Max.exe" 2>NUL | find /I /N "Max.exe">NUL
IF "%ERRORLEVEL%"=="0" (
    echo Music Player is already running.
) ELSE (
    explorer "%CURRENT_DIR%audio\Smile-Music-Player.maxpat"
    timeout /t 3 /nobreak >nul
)
timeout /t 1 /nobreak >nul

echo Opening Video Player 1
start "" "%CURRENT_DIR%.venv\Scripts\pythonw.exe" pl_player.py -pX %PLAYER_1_POSITION% -osc %PLAYER_1_OSC_PORT%
timeout /t 2 /nobreak >nul

echo Opening Video Player 2
start "" "%CURRENT_DIR%.venv\Scripts\pythonw.exe" pl_player.py -pX %PLAYER_2_POSITION% -osc %PLAYER_2_OSC_PORT%
timeout /t 2 /nobreak >nul

echo Opening Main App
start "" "%CURRENT_DIR%.venv\Scripts\pythonw.exe" smile.py -pm
timeout /t 2 /nobreak >nul

@REM echo Press any key to continue . . .
@REM pause >nul
echo Exit this script
timeout /t 1 /nobreak >nul
exit

@REM start cmd /k # leaves the cmd window open after finishing the command
@REM start cmd /c # closes the cmd window after finishing the command
