@REM The main startup: closes whatever runs of the installation and starts all of it again
@REM (ComfyUI, the music player, the image players, SMILE).
@REM Windows Key + R, type shell:startup, place shortcut in folder

@echo off

set "CURRENT_DIR=%~dp0..\"
cd /d "%CURRENT_DIR%"

echo Closing SMILE and the image players
powershell -NoProfile -Command "Get-CimInstance Win32_Process -Filter \"Name='pythonw.exe' or Name='python.exe'\" | Where-Object { $_.CommandLine -like '*smile.py*' -or $_.CommandLine -like '*player.py*' } | ForEach-Object { Stop-Process -Id $_.ProcessId -Force }"

@REM whatever listens on ComfyUI's port, however it was started; then wait until the port is free
echo Closing Image Generator (ComfyUI)
powershell -NoProfile -Command "Get-NetTCPConnection -LocalPort 8188 -State Listen -ErrorAction SilentlyContinue | ForEach-Object { Stop-Process -Id $_.OwningProcess -Force -ErrorAction SilentlyContinue }; foreach ($i in 1..20) { if (-not (Get-NetTCPConnection -LocalPort 8188 -State Listen -ErrorAction SilentlyContinue)) { break }; Start-Sleep -Milliseconds 500 }"

@REM Max is asked to close; forced only when it has not after 10 s
tasklist /FI "IMAGENAME eq Max.exe" 2>NUL | find /I "Max.exe">NUL
if not errorlevel 1 (
    echo Closing Music Player
    taskkill /IM "Max.exe" >nul 2>&1
    set "MAX_CLOSED="
    for /L %%I in (1,1,10) do (
        if not defined MAX_CLOSED (
            timeout /t 1 /nobreak >nul
            tasklist /FI "IMAGENAME eq Max.exe" 2>NUL | find /I "Max.exe">NUL
            if errorlevel 1 (
                set "MAX_CLOSED=1"
            ) else (
                if %%I==10 (
                    echo Max did not close, forcing it
                    taskkill /IM "Max.exe" /F >nul 2>&1
                    timeout /t 1 /nobreak >nul
                ) else (
                    taskkill /IM "Max.exe" >nul 2>&1
                )
            )
        )
    )
)

@REM ComfyUI first: it takes the longest to load. SMILE waits for it and shows that on its screen
echo Opening Image Generator (ComfyUI)
start "ComfyUI" /MIN cmd /c "%CURRENT_DIR%scripts\start_comfy.bat"

echo Opening Music Player
explorer "%CURRENT_DIR%data\audio\Smile-Music-Player.maxpat"
timeout /t 4 /nobreak >nul

call "%CURRENT_DIR%scripts\start_players.bat"
timeout /t 2 /nobreak >nul

echo Opening Main App
start "" "%CURRENT_DIR%.venv\Scripts\pythonw.exe" smile.py -pm -fs
timeout /t 2 /nobreak >nul

echo Exit this script
timeout /t 1 /nobreak >nul
exit

@REM start cmd /k # leaves the cmd window open after finishing the command
@REM start cmd /c # closes the cmd window after finishing the command
