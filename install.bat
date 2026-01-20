@echo off
set PYTHON=python
echo.

set INSTALL_MMCV=0
set FORCE_RECREATE=0
:parse_args
if "%~1"=="" goto args_done
if /I "%~1"=="--mmcv" set INSTALL_MMCV=1
if /I "%~1"=="--force" set FORCE_RECREATE=1
if /I "%~1"=="--help" (
    echo Usage: %~nx0 [--mmcv] [--force]
    goto endofscript
)
shift
goto parse_args
:args_done


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

echo [44m Virtual Environment [0m
set "VENV_DIR=%~dp0%.venv"
if exist "%VENV_DIR%" (
    if "%FORCE_RECREATE%"=="1" (
        echo [33mDeleting existing virtual environment %VENV_DIR%[0m
        rmdir /s /q "%VENV_DIR%"
    ) else (
        echo [91mvirtual environment %VENV_DIR% already exists.[0m
        goto endofscript
    )
) else (
    echo [33mCreating and activating virtual environment %VENV_DIR%[0m
    python -m venv %VENV_DIR%
    call %VENV_DIR%\Scripts\activate

    @rem python -m pip install --upgrade pip THIS BREAKS CHUMPY

    echo.
    echo [44m PyOpenGL [0m
    pip install files\PyOpenGL-3.1.6-cp310-cp310-win_amd64.whl
    pip install files\PyOpenGL_accelerate-3.1.6-cp310-cp310-win_amd64.whl

    echo.
    echo [44m General Requirements [0m
    pip install -r requirements.txt

    echo.
    echo [44m Torch for CUDA 12.9 [0m
    pip install torch==2.8.0+cu129 torchvision==0.23.0+cu129 --index-url https://download.pytorch.org/whl/cu129

    echo.
    if "%INSTALL_MMCV%"=="1" (
        echo [44m mmcv [0m
        echo [33mBuild and install mmcv for CUDA 12.9, this takes about 20 minutes...[0m
        @rem these versions are compatible
        pip install mmdet==3.2.0
        pip install mmpose==1.3.2
        pip install mmcv==2.1.0
    ) else (
        echo [44m mmcv [0m
        echo [33mSkipping mmcv installation (use --mmcv to enable)[0m
    )

    call %VENV_DIR%\Scripts\deactivate
)
echo.


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