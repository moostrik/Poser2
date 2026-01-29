@echo off
echo.
echo [44m TensorRT Model Conversion [0m
echo.

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
set "VENV_DIR=%~dp0%.venv"
if not exist "%VENV_DIR%\Scripts\activate.bat" (
    echo [91mVirtual environment not found![0m
    echo Please run install.bat first.
    goto fail
)

REM Activate virtual environment
call "%VENV_DIR%\Scripts\activate"

echo [33mConverting RTM pose estimation models to TensorRT engines.[0m

rem rtmpose-l_256x192
if not exist "models\rtmpose-l_256x192_b3.trt" goto build_rtmpose_l_256x192
if "%FORCE_REBUILD%"=="1" goto build_rtmpose_l_256x192
echo [90mSkipping rtmpose-l_256x192_b3.trt (already exists)[0m
goto after_rtmpose_l_256x192

:build_rtmpose_l_256x192
python modules\pose\batch\detection\export_rtm_onnx_to_trt.py --onnx models\rtmpose-l_256x192.onnx --output models\rtmpose-l_256x192_b3.trt
if %errorlevel% neq 0 echo [91mFailed to convert rtmpose-l_256x192.onnx[0m
if %errorlevel% neq 0 echo.
if %errorlevel%==0 echo [92mBuilt rtmpose-l_256x192_b3.trt[0m
:after_rtmpose_l_256x192



echo.
echo [33mConverting RVM segmentation models to TensorRT engines.[0m

rem rvm 256x192
if not exist "models\rvm_mobilenetv3_256x192_b3.trt" goto build_rvm_256x192
if "%FORCE_REBUILD%"=="1" goto build_rvm_256x192
echo [90mSkipping rvm_mobilenetv3_256x192_b3.trt (already exists)[0m
goto after_rvm_256x192

:build_rvm_256x192
python modules\pose\batch\segmentation\export_rvm_onnx_to_trt.py --onnx models\rvm_mobilenetv3_256x192.onnx --output models\rvm_mobilenetv3_256x192_b3.trt
if %errorlevel% neq 0 echo [91mFailed to convert rvm_mobilenetv3_256x192.onnx[0m
if %errorlevel% neq 0 echo.
if %errorlevel%==0 echo [92mBuilt rvm_mobilenetv3_256x192_b3.trt[0m
:after_rvm_256x192

rem rvm 384x288
if not exist "models\rvm_mobilenetv3_384x288_b3.trt" goto build_rvm_384x288
if "%FORCE_REBUILD%"=="1" goto build_rvm_384x288
echo [90mSkipping rvm_mobilenetv3_384x288_b3.trt (already exists)[0m
goto after_rvm_384x288

:build_rvm_384x288
python modules\pose\batch\segmentation\export_rvm_onnx_to_trt.py --onnx models\rvm_mobilenetv3_384x288.onnx --output models\rvm_mobilenetv3_384x288_b3.trt
if %errorlevel% neq 0 echo [91mFailed to convert rvm_mobilenetv3_384x288.onnx[0m
if %errorlevel% neq 0 echo.
if %errorlevel%==0 echo [92mBuilt rvm_mobilenetv3_384x288_b3.trt[0m
:after_rvm_384x288

rem rvm 512x384
if not exist "models\rvm_mobilenetv3_512x384_b3.trt" goto build_rvm_512x384
if "%FORCE_REBUILD%"=="1" goto build_rvm_512x384
echo [90mSkipping rvm_mobilenetv3_512x384_b3.trt (already exists)[0m
goto after_rvm_512x384

:build_rvm_512x384
python modules\pose\batch\segmentation\export_rvm_onnx_to_trt.py --onnx models\rvm_mobilenetv3_512x384.onnx --output models\rvm_mobilenetv3_512x384_b3.trt
if %errorlevel% neq 0 echo [91mFailed to convert rvm_mobilenetv3_512x384.onnx[0m
if %errorlevel% neq 0 echo.
if %errorlevel%==0 echo [92mBuilt rvm_mobilenetv3_512x384_b3.trt[0m
:after_rvm_512x384

rem rvm 1024x768
if not exist "models\rvm_mobilenetv3_1024x768_b3.trt" goto build_rvm_1024x768
if "%FORCE_REBUILD%"=="1" goto build_rvm_1024x768
echo [90mSkipping rvm_mobilenetv3_1024x768_b4.trt (already exists)[0m
goto after_rvm_1024x768

:build_rvm_1024x768
python modules\pose\batch\segmentation\export_rvm_onnx_to_trt.py --onnx models\rvm_mobilenetv3_1024x768.onnx --output models\rvm_mobilenetv3_1024x768_b3.trt
if %errorlevel% neq 0 echo [91mFailed to convert rvm_mobilenetv3_1024x768.onnx[0m
if %errorlevel% neq 0 echo.
if %errorlevel%==0 echo [92mBuilt rvm_mobilenetv3_1024x768_b3.trt[0m
:after_rvm_1024x768



echo.
echo [33mConverting RAFT optical flow models to TensorRT engines.[0m

rem raft 256x192
if not exist "models\raft-sintel_256x192_i12_b3.trt" goto build_raft_256x192
if "%FORCE_REBUILD%"=="1" goto build_raft_256x192
echo [90mSkipping raft-sintel_256x192_i12_b3.trt (already exists)[0m
goto after_raft_256x192

:build_raft_256x192
python modules\pose\batch\flow\export_raft_onnx_to_trt.py --onnx models\raft-sintel_256x192_i12.onnx --output models\raft-sintel_256x192_i12_b3.trt
if %errorlevel% neq 0 echo [91mFailed to convert raft-sintel_256x192_i12.onnx[0m
if %errorlevel% neq 0 echo.
if %errorlevel%==0 echo [92mBuilt raft-sintel_256x192_i12_b3.trt[0m
:after_raft_256x192

rem raft 384x288
if not exist "models\raft-sintel_384x288_i12_b3.trt" goto build_raft_384x288
if "%FORCE_REBUILD%"=="1" goto build_raft_384x288
echo [90mSkipping raft-sintel_384x288_i12_b3.trt (already exists)[0m
goto after_raft_384x288

:build_raft_384x288
python modules\pose\batch\flow\export_raft_onnx_to_trt.py --onnx models\raft-sintel_384x288_i12.onnx --output models\raft-sintel_384x288_i12_b3.trt
if %errorlevel% neq 0 echo [91mFailed to convert raft-sintel_384x288_i12.onnx[0m
if %errorlevel% neq 0 echo.
if %errorlevel%==0 echo [92mBuilt raft-sintel_384x288_i12_b3.trt[0m
:after_raft_384x288

rem raft 512x384
if not exist "models\raft-sintel_512x384_i12_b3.trt" goto build_raft_512x384
if "%FORCE_REBUILD%"=="1" goto build_raft_512x384
echo [90mSkipping raft-sintel_512x384_i12_b3.trt (already exists)[0m
goto after_raft_512x384

:build_raft_512x384
python modules\pose\batch\flow\export_raft_onnx_to_trt.py --onnx models\raft-sintel_512x384_i12.onnx --output models\raft-sintel_512x384_i12_b3.trt
if %errorlevel% neq 0 echo [91mFailed to convert raft-sintel_512x384_i12.onnx[0m
if %errorlevel% neq 0 echo.
if %errorlevel%==0 echo [92mBuilt raft-sintel_512x384_i12_b3.trt[0m
:after_raft_512x384

echo.
echo [92mTensorRT conversion complete[0m

@REM call "%VENV_DIR%\Scripts\deactivate"
goto endofscript

:fail
echo.
echo [91mConversion failed[0m

:endofscript
pause
exit /b 0