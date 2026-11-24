@REM Installs SMILE's image generator, or checks an install: ComfyUI (scripts\install_comfy.ps1), the models
@REM (comfy\models.py), then a smoke test that runs the warm-up in ComfyUI once. The repo's own install
@REM comes first: scripts\install.bat at the Poser2 root (the LFS files, the venv SMILE runs in, .venv).
@REM Safe to run again: everything in place is skipped. --from takes the site folder, what git does not bring:
@REM     apps\smile\scripts\install.bat --from F:\_scratch\SMILE
@REM its models\ (ComfyUI's models, copied and verified, the rest fetched from the sources), online_token.txt and
@REM input\ (the photos of the panel's still image), each copied where it is missing, never over a file that is there.
@REM Without --from it asks for the folder. Further arguments go to the model tool.
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
    echo [93mffmpeg is not on the PATH: visitors cannot be saved as clips and the clip player cannot play them.[0m
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
set "FROM="
set "MODEL_ARGS="
if /i "%~1"=="--from" (
    set "FROM=%~2"
    shift
    shift
) else (
    echo The site folder holds what git does not bring: models\ ^(ComfyUI's models^), online_token.txt and input\.
    echo Its models save the download of what it holds; what it lacks comes from the sources.
    set /p "FROM=Site folder (Enter for none): "
)
:model_args
if not "%~1"=="" (
    set "MODEL_ARGS=!MODEL_ARGS! %1"
    shift
    goto model_args
)
if defined FROM if not exist "!FROM!\" (
    echo [91mNo folder !FROM![0m
    goto fail
)
if defined FROM if exist "!FROM!\models\" set "MODEL_ARGS=!MODEL_ARGS! --mirror "!FROM!\models""
pushd "%ROOT_DIR%"
"%VENV_DIR%\Scripts\python.exe" -m apps.smile.comfy.models install !MODEL_ARGS!
set "RC=%errorlevel%"
popd
if not "%RC%"=="0" (
    echo [91mNot every model is in place, see above[0m
    goto fail
)
echo.

echo [44m Site files [0m
if not defined FROM (
    echo No site folder: the online token and the photos stay as they are
) else (
    if exist "%SMILE_DIR%\data\online_token.txt" (
        echo [92monline_token.txt there[0m
    ) else if exist "!FROM!\online_token.txt" (
        copy /y "!FROM!\online_token.txt" "%SMILE_DIR%\data\online_token.txt" >nul && echo [92monline_token.txt copied[0m
    ) else (
        echo [93mNo online_token.txt in !FROM!: SMILE Online cannot connect without it[0m
    )
    if exist "!FROM!\input\" (
        if not exist "%SMILE_DIR%\data\input\" mkdir "%SMILE_DIR%\data\input"
        set "COPIED=0"
        for %%f in ("!FROM!\input\*") do if not exist "%SMILE_DIR%\data\input\%%~nxf" (
            copy /y "%%f" "%SMILE_DIR%\data\input\" >nul && set /a COPIED+=1
        )
        echo [92minput: !COPIED! photo^(s^) copied[0m
    ) else (
        echo [93mNo input\ in !FROM!: the panel's still image has no photos[0m
    )
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
