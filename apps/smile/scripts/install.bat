@REM Installs SMILE's image generator, or checks an install: ComfyUI (scripts\install_comfy.ps1), the models
@REM (comfy\models.py), then a smoke test that runs the warm-up in ComfyUI once. The repo's own install
@REM comes first: scripts\install.bat at the Poser2 root (the LFS files, the venv).
@REM Safe to run again: everything in place is skipped. Extra arguments go to the model tool; without --mirror
@REM it asks for a folder with the models to copy from (the backup's ComfyUI\models, a USB stick):
@REM     apps\smile\scripts\install.bat --mirror E:\SMILE\ComfyUI\models
@echo off
setlocal EnableDelayedExpansion
set "SMILE_DIR=%~dp0.."
set "ROOT_DIR=%~dp0..\..\.."
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
if not exist "%VENV_DIR%\Scripts\python.exe" (
    echo [91mNo venv at %VENV_DIR%: run scripts\install.bat at the Poser2 root first[0m
    goto fail
)
echo [92mgit, git lfs and the venv found[0m
where ffmpeg >nul 2>nul
if %errorlevel% neq 0 (
    echo [93mffmpeg is not on the PATH: visitors cannot be saved as clips and the players cannot play them.[0m
    echo Download the "release full" build from https://www.gyan.dev/ffmpeg/builds/ , extract it and add its bin folder to the PATH.
) else (
    echo [92mffmpeg found[0m
)
set "VIMBA_DLL="
if defined VIMBA_HOME set "VIMBA_DLL=%VIMBA_HOME%\VimbaC\Bin\Win64\VimbaC.dll"
if defined VIMBA_DLL if exist "%VIMBA_DLL%" (
    echo [92mVimba SDK found: %VIMBA_HOME%[0m
    goto vimba_done
)
echo [93mVimba SDK not found ^(VIMBA_HOME, VimbaC^): the Mako camera cannot open; SMILE runs with -wc or the panel's use_image.[0m
echo Install the Vimba 6 SDK ^(not Vimba X, with its C API, not the Viewer alone^) as administrator from
echo https://www.alliedvision.com/en/products/vimba-sdk/ and open a new terminal.
:vimba_done
echo.

echo [44m ComfyUI [0m
powershell -NoProfile -ExecutionPolicy Bypass -File "%SMILE_DIR%\scripts\install_comfy.ps1"
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
"%VENV_DIR%\Scripts\python.exe" -m apps.smile.comfy.models install !MODEL_ARGS!
set "RC=%errorlevel%"
popd
if not "%RC%"=="0" (
    echo [91mNot every model is in place, see above[0m
    goto fail
)
echo.

echo [44m Smoke test [0m
pushd "%ROOT_DIR%"
"%VENV_DIR%\Scripts\python.exe" -m apps.smile.comfy.smoke
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
