@echo off
echo.
echo [44m TensorRT Model Conversion [0m

set FORCE_REBUILD=0
:parse_args
if "%~1"=="" goto args_done
if /I "%~1"=="--force" set FORCE_REBUILD=1
if /I "%~1"=="--help" (
    echo Usage: %~nx0 [--force]
    echo   --force    Rebuild all models even if they exist
    goto endofscript
)
shift
goto parse_args
:args_done

if "%FORCE_REBUILD%"=="1" (
    echo [33mForce rebuild enabled - will recreate all models[0m
    echo.
)

REM Check if virtual environment exists
set "VENV_DIR=%~dp0.venv"
if not exist "%VENV_DIR%\Scripts\activate.bat" (
    echo [91mVirtual environment not found![0m
    echo Please run install.bat first.
    goto fail
)

REM Activate virtual environment
call "%VENV_DIR%\Scripts\activate"

echo.
echo [33mRTM pose estimation[0m
call :build "data\models\rtmpose-l_256x192_b3.trt"  "python modules\inference\tools\export_rtm_onnx_to_trt.py --onnx data\models\rtmpose-l_256x192.onnx --output data\models\rtmpose-l_256x192_b3.trt"
call :build "data\models\rtmpose-l_256x192_b8.trt"  "python modules\inference\tools\export_rtm_onnx_to_trt.py --onnx data\models\rtmpose-l_256x192.onnx --output data\models\rtmpose-l_256x192_b8.trt --opt-batch 6 --max-batch 8"

echo.
echo [33mRVM matting / segmentation[0m
call :build "data\models\rvm_mobilenetv3_256x192_b3.trt"  "python modules\inference\tools\export_rvm_onnx_to_trt.py --onnx data\models\rvm_mobilenetv3_256x192.onnx --output data\models\rvm_mobilenetv3_256x192_b3.trt"
call :build "data\models\rvm_mobilenetv3_256x192_b8.trt"  "python modules\inference\tools\export_rvm_onnx_to_trt.py --onnx data\models\rvm_mobilenetv3_256x192.onnx --output data\models\rvm_mobilenetv3_256x192_b8.trt --opt-batch 6 --max-batch 8"
call :build "data\models\rvm_mobilenetv3_384x288_b3.trt"  "python modules\inference\tools\export_rvm_onnx_to_trt.py --onnx data\models\rvm_mobilenetv3_384x288.onnx --output data\models\rvm_mobilenetv3_384x288_b3.trt"
call :build "data\models\rvm_mobilenetv3_512x384_b3.trt"  "python modules\inference\tools\export_rvm_onnx_to_trt.py --onnx data\models\rvm_mobilenetv3_512x384.onnx --output data\models\rvm_mobilenetv3_512x384_b3.trt"
call :build "data\models\rvm_mobilenetv3_1024x768_b3.trt" "python modules\inference\tools\export_rvm_onnx_to_trt.py --onnx data\models\rvm_mobilenetv3_1024x768.onnx --output data\models\rvm_mobilenetv3_1024x768_b3.trt"

@REM echo.
@REM echo [33mRAFT optical flow[0m
@REM call :build "data\models\raft-sintel_256x192_i12_b3.trt" "python modules\inference\tools\export_raft_onnx_to_trt.py --onnx data\models\raft-sintel_256x192_i12.onnx --output data\models\raft-sintel_256x192_i12_b3.trt"
@REM call :build "data\models\raft-sintel_384x288_i12_b3.trt" "python modules\inference\tools\export_raft_onnx_to_trt.py --onnx data\models\raft-sintel_384x288_i12.onnx --output data\models\raft-sintel_384x288_i12_b3.trt"
@REM call :build "data\models\raft-sintel_512x384_i12_b3.trt" "python modules\inference\tools\export_raft_onnx_to_trt.py --onnx data\models\raft-sintel_512x384_i12.onnx --output data\models\raft-sintel_512x384_i12_b3.trt"

echo.
echo [92mTensorRT conversion complete[0m
goto endofscript

:fail
echo.
echo [91mConversion failed[0m

:endofscript
pause
exit /b 0

:build
if not exist "%~1" goto do_build
if "%FORCE_REBUILD%"=="1" goto do_build
echo [90mSkipping %~nx1 - already exists[0m
exit /b 0
:do_build
%~2
if errorlevel 1 (
    echo [91mFailed: %~nx1[0m
) else (
    echo [92mBuilt %~nx1[0m
)
exit /b 0