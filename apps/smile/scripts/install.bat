@REM Installs SMILE, or checks an install: the LFS files, the app venv, ComfyUI (scripts\install_comfy.ps1),
@REM the models (modules\comfy\models.py), then a smoke test that runs the warm-up in ComfyUI once.
@REM Safe to run again: everything in place is skipped. Extra arguments go to the model tool; without --mirror
@REM it asks for a folder with the models to copy from (the backup's ComfyUI\models, a USB stick):
@REM     scripts\install.bat --mirror E:\SMILE\ComfyUI\models
@echo off
setlocal EnableDelayedExpansion
set "ROOT_DIR=%~dp0.."
set "VENV_DIR=%ROOT_DIR%\.venv"
echo.

echo [44m Prerequisites [0m
where git >nul 2>nul
if %errorlevel% neq 0 (
    echo [91mGit is not installed or not in PATH: install it from https://git-scm.com/download/win and run "git lfs install"[0m
    goto fail
)
git lfs version >nul 2>nul
if %errorlevel% neq 0 (
    echo [91mGit LFS is not installed: run "git lfs install"[0m
    goto fail
)
py -3.12 --version >nul 2>nul
if %errorlevel% neq 0 (
    echo [91mPython 3.12 is not installed with the py launcher: see README.md[0m
    goto fail
)
where ffmpeg >nul 2>nul
if %errorlevel% neq 0 (
    echo [93mffmpeg is not on the PATH: visitors cannot be saved as clips and the players cannot play them.[0m
    echo Download the "release full" build from https://www.gyan.dev/ffmpeg/builds/ , extract it and add its bin folder to the PATH.
) else (
    echo [92mgit, git lfs, py -3.12 and ffmpeg found[0m
)
echo.

echo [44m Git LFS [0m
git -C "%ROOT_DIR%" lfs pull
if %errorlevel% neq 0 (
    echo [91mFailed to pull the LFS files[0m
    goto fail
)
echo [92mLFS files in place[0m
echo.

echo [44m App venv [0m
if not exist "%VENV_DIR%\Scripts\python.exe" (
    echo [33mCreating %VENV_DIR%[0m
    py -3.12 -m venv "%VENV_DIR%"
    if %errorlevel% neq 0 (
        echo [91mCould not create the venv[0m
        goto fail
    )
)
"%VENV_DIR%\Scripts\python.exe" -m pip install --quiet --upgrade pip
"%VENV_DIR%\Scripts\python.exe" -m pip install -r "%ROOT_DIR%\requirements.txt"
if %errorlevel% neq 0 (
    echo [91mpip install -r requirements.txt failed[0m
    goto fail
)
echo [92mapp venv ok[0m
echo.

echo [44m ComfyUI [0m
powershell -NoProfile -ExecutionPolicy Bypass -File "%ROOT_DIR%\scripts\install_comfy.ps1"
if %errorlevel% neq 0 (
    echo [91mComfyUI install failed, see above[0m
    goto fail
)
echo.

echo [44m Models [0m
set "MODEL_ARGS=%*"
echo %MODEL_ARGS% | find "--mirror" >nul
if errorlevel 1 (
    echo The models go into ComfyUI\models. A folder in the same layout to copy them from, for instance the backup's
    echo ComfyUI\models or a USB stick, saves the download of what it holds; what it lacks comes from the sources.
    set "MIRROR="
    set /p "MIRROR=Folder with the models to copy from (Enter for none): "
    if defined MIRROR set "MODEL_ARGS=!MODEL_ARGS! --mirror "!MIRROR!""
)
pushd "%ROOT_DIR%"
py -3.12 -m modules.comfy.models install !MODEL_ARGS!
set "RC=%errorlevel%"
popd
if not "%RC%"=="0" (
    echo [91mNot every model is in place, see above[0m
    goto fail
)
echo.

echo [44m Smoke test [0m
pushd "%ROOT_DIR%"
"%VENV_DIR%\Scripts\python.exe" -m modules.comfy.smoke
set "RC=%errorlevel%"
popd
if not "%RC%"=="0" (
    echo [91mSmoke test failed, see above[0m
    goto fail
)

echo.
echo [92mInstall complete.[0m
echo.
pause
exit /b 0

:fail
echo.
echo [91mInstall not successful[0m
echo.
pause
exit /b 1
