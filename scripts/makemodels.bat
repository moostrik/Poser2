@echo off
echo.
echo  TensorRT Model Conversion

set APP=%1
set "VENV_DIR=%~dp0..\.venv"

if "%APP%"=="" (
    echo Usage: %~nx0 APP
    echo   APP    hd_trio, white_space, deep_flow, all
    goto endofscript
)

if not exist "%VENV_DIR%\Scripts\activate.bat" (
    echo Virtual environment not found! Run scripts/install.bat first.
    goto endofscript
)
call "%VENV_DIR%\Scripts\activate"

if /I "%APP%"=="hd_trio"      call :section_hd_trio
if /I "%APP%"=="white_space"  call :section_white_space
if /I "%APP%"=="deep_flow"    call :section_deep_flow
if /I "%APP%"=="all"          call :section_hd_trio
if /I "%APP%"=="all"          call :section_white_space
if /I "%APP%"=="all"          call :section_deep_flow

echo.
echo TensorRT conversion complete
goto endofscript

:section_hd_trio
echo.
echo hd_trio
call :build "python modules\inference\tools\export_rtm_onnx_to_trt.py --onnx data\models\rtmpose-l_256x192.onnx --output apps\hd_trio\data\models\rtmpose-l_256x192.trt --opt-batch 3 --max-batch 3"
call :build "python modules\inference\tools\export_rvm_onnx_to_trt.py --onnx data\models\rvm_mobilenetv3_256x192.onnx --output apps\hd_trio\data\models\rvm_mobilenetv3_256x192.trt --opt-batch 3 --max-batch 3"
call :build "python modules\inference\tools\export_rvm_onnx_to_trt.py --onnx data\models\rvm_mobilenetv3_384x288.onnx --output apps\hd_trio\data\models\rvm_mobilenetv3_384x288.trt --opt-batch 3 --max-batch 3"
call :build "python modules\inference\tools\export_rvm_onnx_to_trt.py --onnx data\models\rvm_mobilenetv3_512x384.onnx --output apps\hd_trio\data\models\rvm_mobilenetv3_512x384.trt --opt-batch 3 --max-batch 3"
call :build "python modules\inference\tools\export_rvm_onnx_to_trt.py --onnx data\models\rvm_mobilenetv3_1024x768.onnx --output apps\hd_trio\data\models\rvm_mobilenetv3_1024x768.trt --opt-batch 3 --max-batch 3"
exit /b 0

:section_white_space
echo.
echo white_space
call :build "python modules\inference\tools\export_rtm_onnx_to_trt.py --onnx data\models\rtmpose-l_256x192.onnx --output apps\white_space\data\models\rtmpose-l_256x192.trt --opt-batch 6 --max-batch 8"
call :build "python modules\inference\tools\export_rvm_onnx_to_trt.py --onnx data\models\rvm_mobilenetv3_256x192.onnx --output apps\white_space\data\models\rvm_mobilenetv3_256x192.trt --opt-batch 6 --max-batch 8"
exit /b 0

:section_deep_flow
echo.
echo deep_flow
call :build "python modules\inference\tools\export_rtm_onnx_to_trt.py --onnx data\models\rtmpose-l_256x192.onnx --output apps\deep_flow\data\models\rtmpose-l_256x192.trt --opt-batch 1 --max-batch 1"

call :build "python modules\inference\tools\export_rvm_onnx_to_trt.py --onnx data\models\rvm_mobilenetv3_256x192.onnx --output apps\deep_flow\data\models\rvm_mobilenetv3_256x192.trt --opt-batch 1 --max-batch 1"
call :build "python modules\inference\tools\export_rvm_onnx_to_trt.py --onnx data\models\rvm_mobilenetv3_384x288.onnx --output apps\deep_flow\data\models\rvm_mobilenetv3_384x288.trt --opt-batch 1 --max-batch 1"
call :build "python modules\inference\tools\export_rvm_onnx_to_trt.py --onnx data\models\rvm_mobilenetv3_512x384.onnx --output apps\deep_flow\data\models\rvm_mobilenetv3_512x384.trt --opt-batch 1 --max-batch 1"
call :build "python modules\inference\tools\export_rvm_onnx_to_trt.py --onnx data\models\rvm_mobilenetv3_1024x768.onnx --output apps\deep_flow\data\models\rvm_mobilenetv3_1024x768.trt --opt-batch 1 --max-batch 1"

call :build "python modules\inference\tools\export_raft_onnx_to_trt.py --onnx data\models\raft-sintel_256x192_i12.onnx --output apps\deep_flow\data\models\raft-sintel_256x192_i12.trt --opt-batch 1 --max-batch 1"
call :build "python modules\inference\tools\export_raft_onnx_to_trt.py --onnx data\models\raft-sintel_384x288_i12.onnx --output apps\deep_flow\data\models\raft-sintel_384x288_i12.trt --opt-batch 1 --max-batch 1"
call :build "python modules\inference\tools\export_raft_onnx_to_trt.py --onnx data\models\raft-sintel_512x384_i12.onnx --output apps\deep_flow\data\models\raft-sintel_512x384_i12.trt --opt-batch 1 --max-batch 1"
exit /b 0

:build
setlocal enabledelayedexpansion
set _ARGS=%~1
set _OUT=
set _PREV=
for %%A in (!_ARGS!) do (
    if "!_PREV!"=="--output" set "_OUT=%%A"
    set "_PREV=%%A"
)
if exist "!_OUT!" (
    echo Skipping !_OUT! - already exists
    endlocal & exit /b 0
)
%~1
if errorlevel 1 (echo Failed: !_OUT!) else (echo Built !_OUT!)
endlocal
exit /b 0

:endofscript
pause
exit /b 0