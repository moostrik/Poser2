@echo off
set PYTHON=py -3.12
set "ROOT_DIR=%~dp0.."
set "VENV_DIR=%ROOT_DIR%\.venv"
echo.

echo [44m Git LFS [0m
where git >nul 2>nul
if %errorlevel% neq 0 (
    echo [91mGit is not installed or not in PATH![0m
    echo Please install Git from https://git-scm.com/download/win
    echo And run "git lfs install" once installed.
    goto endofscript
)

git lfs version >nul 2>nul
if %errorlevel% neq 0 (
    echo [91mGit LFS is not installed![0m
    echo Did you run "git lfs install" after installing Git?
    goto endofscript
) else (
    echo [92mGit LFS is installed[0m
)

echo [33mPulling LFS files...[0m
git -C "%ROOT_DIR%" lfs pull
if %errorlevel% neq 0 (
    echo [91mFailed to pull LFS files![0m
    goto endofscript
) else (
    echo [92mLFS files downloaded successfully[0m
)
echo.

echo [44m FFmpeg [0m
where ffmpeg >nul 2>nul
if %errorlevel% neq 0 (
    echo [91mffmpeg is not on the PATH: visitors cannot be saved as clips and the players cannot play them.[0m
    echo Download the "release full" build from https://www.gyan.dev/ffmpeg/builds/ , extract it to
    echo C:\Developer\ffmpeg and add C:\Developer\ffmpeg\bin to the PATH.
) else (
    echo [92mffmpeg is installed[0m
)
echo.

echo [44m Virtual Environment [0m

if exist "%VENV_DIR%" (
    echo [91mvirtual environment already exists at %VENV_DIR%[0m
    goto endofscript
)

:venv_create
echo [33mCreating and activating virtual environment %VENV_DIR%[0m
%PYTHON% -m venv "%VENV_DIR%"
call "%VENV_DIR%\Scripts\activate"
echo.

echo [44m Requirements [0m
python -m pip install --upgrade pip
pip install -r "%ROOT_DIR%\requirements.txt"

call "%VENV_DIR%\Scripts\deactivate"


:success
echo.
echo [92mInstall complete.[0m
echo.
pause
goto :eof

:endofscript
echo.
echo [91mInstall not successful[0m
echo.
pause
