@echo off
cd /d "%~dp0.."
echo.

echo [44m Git LFS [0m
where git >nul 2>nul
if %errorlevel% neq 0 (
    echo [91mGit is not installed or not in PATH![0m
    echo Please install Git from https://git-scm.com/download/win
    echo And run "git lfs install" once installed.
    goto endofscript
)

git lfs version >nul 2>nul
if %errorlevel% neq 0 (
    echo [91mGit LFS is not installed![0m
    echo Did you run "git lfs install" after installing Git?
    goto endofscript
) else (
    echo [92mGit LFS is installed[0m
)

echo [33mPulling LFS files...[0m
git lfs pull
if %errorlevel% neq 0 (
    echo [91mFailed to pull LFS files![0m
    goto endofscript
) else (
    echo [92mLFS files downloaded successfully[0m
)
echo.

echo [44m Packages [0m
where uv >nul 2>nul
if %errorlevel% neq 0 (
    echo [91muv is not installed![0m
    echo Install it with "winget install astral-sh.uv" and open a new terminal
    goto endofscript
)

@REM Builds or updates .venv to match uv.lock exactly (Python 3.12 included).
@REM "install.bat oak4" installs depthai 3 for the OAK4 backend in place of depthai 2 (pyproject.toml, dependency-groups)
set SYNC_GROUPS=
if /i "%~1"=="oak4" (
    set SYNC_GROUPS=--no-group oak --group oak4
    echo [33mOAK4: depthai 3 instead of depthai 2[0m
)
uv sync --locked %SYNC_GROUPS%
if %errorlevel% neq 0 (
    echo [91mPackage install failed![0m
    goto endofscript
)


:success
echo.
echo [92mInstall complete.[0m
echo.
echo Run makemodels.bat to build TensorRT engines:
echo   scripts/makemodels.bat ^<app_name^>
echo or
echo   scripts/makemodels.bat all
echo.
pause
goto :eof

:endofscript
echo.
echo [91mInstall not successful[0m
echo.
pause