@echo off
set PYTHON=py -3.12
echo.

echo [44m CUDA 12.9 [0m
echo %PATH% | find /I "CUDA\v12.9\bin" >nul
if %errorlevel%==0 (
    echo [92mCUDA 12.9 bin directory found in PATH.[0m
) else (
    echo [91mCUDA 12.9 bin directory NOT found in PATH![0m
    echo Please add "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.9\bin" to your PATH.
    goto endofscript
)
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

echo [44m Virtual Environment [0m
set "VENV_DIR=%~dp0.venv"

if exist "%VENV_DIR%" (
    echo [33mvirtual environment already exists at %VENV_DIR%[0m
    goto endofscript
)

:venv_create
echo [33mCreating and activating virtual environment %VENV_DIR%[0m
%PYTHON% -m venv "%VENV_DIR%"
call "%VENV_DIR%\Scripts\activate"
echo.

python -m pip install --upgrade pip
echo.

echo [44m Requirements [0m
pip install -r requirements.txt

call "%VENV_DIR%\Scripts\deactivate"

:makemodels
echo.
call makemodels.bat
if %errorlevel% neq 0 (
    echo [91mModel conversion failed[0m
    goto endofscript
)


:success
echo.
echo [92mLaunch successful[0m
echo Exiting.
pause
exit

:endofscript
echo.
echo [91mLaunch not successful[0m
echo Exiting.
pause