@echo off
@REM Installs Poser2 on this PC: the PC part every app needs, then the venv.
@REM   scripts\install.bat          the PC part and .venv (depthai 3, the OAK4 apps and smile)
@REM   scripts\install.bat oakd     the PC part and .venv-oakd (depthai 2, white_space and hd_trio)
@REM   scripts\install.bat APP      the PC part, the app's venv (scripts\app_venv.bat), its TensorRT engines and, for
@REM                                the OAK4 apps, the OAK4's archives
@REM The PC part offers what winget can install when it is missing (asked, Y/N) and checks what it cannot.
setlocal EnableDelayedExpansion
cd /d "%~dp0.."
echo.
set "MISSING="
set "APP="
set "OAKD="
if /i "%~1"=="oakd" (
    set "OAKD=1"
) else if not "%~1"=="" (
    call scripts\app_venv.bat %1 || (echo [91mUnknown app: %~1[0m & goto endofscript)
    set "APP=%~1"
    if "!VENV!"==".venv-oakd" set "OAKD=1"
)

echo [44m PC [0m
where git >nul 2>nul
if %errorlevel% neq 0 (
    echo [91mGit is not installed or not in PATH: install it from https://git-scm.com/download/win[0m
    goto endofscript
)
git lfs install >nul
if %errorlevel% neq 0 (
    echo [91mGit LFS is not installed: it comes with Git for Windows[0m
    goto endofscript
)
echo [92mGit and Git LFS found[0m

reg query "HKLM\SOFTWARE\Microsoft\VisualStudio\14.0\VC\Runtimes\x64" /v Installed 2>nul | find "0x1" >nul
if %errorlevel% neq 0 (
    call :offer "Visual C++ Redistributable" Microsoft.VCRedist.2015+.x64
) else (
    echo [92mVisual C++ Redistributable found[0m
)

where uv >nul 2>nul
if %errorlevel% neq 0 (
    call :offer "uv" astral-sh.uv
    where uv >nul 2>nul || (echo [91muv is needed for the venv[0m & goto endofscript)
) else (
    echo [92muv found[0m
)

where ffmpeg >nul 2>nul
if %errorlevel% neq 0 (
    call :offer "ffmpeg" Gyan.FFmpeg
) else (
    echo [92mffmpeg found[0m
)

where code >nul 2>nul
if %errorlevel% neq 0 (
    call :offer "VS Code" Microsoft.VisualStudioCode
) else (
    echo [92mVS Code found[0m
)

@REM The NVIDIA driver: torch's CUDA 13 needs 580 or newer. Checked only: a driver is the PC's own.
set "DRIVER="
for /f "tokens=1 delims=." %%v in ('nvidia-smi --query-gpu^=driver_version --format^=csv^,noheader 2^>nul') do if not defined DRIVER set "DRIVER=%%v"
if not defined DRIVER (
    echo [91mNo NVIDIA driver found ^(nvidia-smi^): install it from https://www.nvidia.com/drivers[0m
    set "MISSING=!MISSING! NVIDIA-driver"
) else if !DRIVER! lss 580 (
    echo [91mNVIDIA driver !DRIVER! is too old: CUDA 13 needs 580 or newer, from https://www.nvidia.com/drivers[0m
    set "MISSING=!MISSING! NVIDIA-driver"
) else (
    echo [92mNVIDIA driver !DRIVER! found[0m
)

@REM Quick Sync: every app records and decodes on the iGPU (h264_qsv). Checked only: the Intel graphics driver.
where ffmpeg >nul 2>nul
if %errorlevel% equ 0 (
    ffmpeg -hide_banner -loglevel error -f lavfi -i color=black:s=256x256 -frames:v 1 -c:v h264_qsv -f null - >nul 2>nul
    if !errorlevel! neq 0 (
        echo [91mNo Quick Sync: the Intel graphics driver is missing or the iGPU is off in the BIOS[0m
        set "MISSING=!MISSING! QuickSync"
    ) else (
        echo [92mQuick Sync found[0m
    )
)
echo.

echo [44m Git LFS [0m
echo [33mPulling LFS files...[0m
git lfs pull
if %errorlevel% neq 0 (
    echo [91mFailed to pull LFS files![0m
    goto endofscript
) else (
    echo [92mLFS files downloaded successfully[0m
)
echo.

echo [44m Packages [0m
@REM Builds or updates .venv to match uv.lock exactly (Python 3.12 included), with depthai 3 for the OAK4 backend.
@REM "install.bat oakd" builds a second venv, .venv-oakd, with depthai 2 for the OAK-D backend in place of depthai 3
@REM (pyproject.toml, dependency-groups); .venv keeps depthai 3, so both live side by side.
set SYNC_GROUPS=
if defined OAKD (
    set SYNC_GROUPS=--no-group oak4 --group oakd
    set UV_PROJECT_ENVIRONMENT=.venv-oakd
    echo [33mOAK-D: depthai 2 in .venv-oakd[0m
    if not exist ".venv-oakd" uv venv .venv-oakd --prompt poser2-oakd
)
uv sync --locked %SYNC_GROUPS%
if %errorlevel% neq 0 (
    echo [91mPackage install failed![0m
    goto endofscript
)
echo.

if defined APP (
    echo [44m Models: %APP% [0m
    if /i "%APP%"=="deep_flow" call :oak4_archives || goto endofscript
    if /i "%APP%"=="low_flow"  call :oak4_archives || goto endofscript
    set "NOPAUSE=1"
    call scripts\makemodels.bat %APP%
    echo.
)

:success
echo.
if defined MISSING (
    echo [93mInstall complete, but missing:!MISSING! ^(see above^)[0m
) else (
    echo [92mInstall complete.[0m
)
if not defined APP (
    echo.
    echo An app's TensorRT engines: scripts\install.bat ^<app^>, or scripts\makemodels.bat ^<app^>
)
echo.
pause
goto :eof

@REM The OAK4's archives from Luxonis's zoo, the ones not there yet (scripts\oak4_models.py).
:oak4_archives
.venv\Scripts\python.exe scripts\oak4_models.py
if !errorlevel! neq 0 (
    echo [91mThe OAK4's archives could not all be fetched ^(internet?^)[0m
    exit /b 1
)
exit /b 0

:endofscript
echo.
echo [91mInstall not successful[0m
echo.
pause
exit /b 1

@REM A missing piece winget can install: asked, and installed on Y. %1 its name, %2 its winget id.
:offer
set "ANSWER="
set /p "ANSWER=%~1 is missing. Install it with winget (%~2)? [Y/N] "
if /i not "!ANSWER!"=="Y" (
    echo [93m%~1 not installed[0m
    set "MISSING=!MISSING! %~1"
    exit /b 0
)
winget install --id %~2 --exact --silent --accept-source-agreements --accept-package-agreements
if !errorlevel! neq 0 (
    echo [91mwinget could not install %~1[0m
    set "MISSING=!MISSING! %~1"
    exit /b 0
)
@REM winget puts what it installs on the PATH of new terminals; this run reads that PATH again
for /f "delims=" %%p in ('powershell -NoProfile -Command "[Environment]::GetEnvironmentVariable('Path','Machine') + ';' + [Environment]::GetEnvironmentVariable('Path','User')"') do set "PATH=%%p;%PATH%"
echo [92m%~1 installed[0m
exit /b 0
