@echo off
set PYTHON=python
echo.


echo CHECK AND ACTIVATE VENV
set "VENV_DIR=%~dp0%.venv"
if exist "%VENV_DIR%" (
    echo FOUND %VENV_DIR%
) else (
    echo %VENV_DIR%
    python -m venv %VENV_DIR%
    call %VENV_DIR%\Scripts\activate
    python -m pip install --upgrade pip
    @REM pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
    pip install files\PyOpenGL-3.1.6-cp310-cp310-win_amd64.whl
    pip install files\PyOpenGL_accelerate-3.1.6-cp310-cp310-win_amd64.whl
    pip install -r requirements.txt
    call %VENV_DIR%\Scripts\deactivate
)
echo.


:succes
echo Launch successful.
echo Exiting.
pause
exit /b

:endofscript
echo.
echo Launch not successful.
echo Exiting.
pause