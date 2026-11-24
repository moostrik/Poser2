@echo off
@REM The venv an app runs in, for the scripts that start or build for an app: call scripts\app_venv.bat APP
@REM sets VENV (.venv-oakd for the OAK-D apps on depthai 2, .venv for the others) and INSTALL, the command that
@REM builds it; an unknown app leaves VENV empty and returns 1.
set "VENV="
set "INSTALL="
if /i "%~1"=="deep_flow"   set "VENV=.venv"
if /i "%~1"=="low_flow"    set "VENV=.venv"
if /i "%~1"=="smile"       set "VENV=.venv"
if /i "%~1"=="white_space" set "VENV=.venv-oakd"
if /i "%~1"=="hd_trio"     set "VENV=.venv-oakd"
if not defined VENV exit /b 1
if "%VENV%"==".venv" (set "INSTALL=scripts\install.bat") else (set "INSTALL=scripts\install.bat oakd")
exit /b 0
