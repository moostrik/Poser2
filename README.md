# Poser2
depth cam pose synchony detection

## INSTALLATION

### New PC
1. Git for Windows from [site](https://git-scm.com/download/win), then ```git clone https://github.com/moostrik/Poser2.git```
2. ```scripts\install.bat <app>``` (deep_flow, low_flow, smile, white_space, hd_trio), with ```--startup``` for a show PC: it offers what is missing and winget installs (Visual C++ Redistributable, uv, ffmpeg, VS Code), runs `git lfs install`, checks the NVIDIA driver (580 or newer, CUDA 13) and Quick Sync, pulls the LFS files, builds the app's venv (`scripts\app_venv.bat`), its TensorRT engines and, for deep_flow and low_flow, the OAK4's archives; ```--startup``` puts the app's start script in the Startup folder
3. By hand, what no script installs:
    * every app: the NVIDIA driver from [nvidia.com/drivers](https://www.nvidia.com/drivers) and the Intel graphics driver with the iGPU on in the BIOS (the apps record and decode on Quick Sync); `install.bat` says which is missing
    * every app: Max (Cycling '74) for the sound
    * deep_flow, low_flow: the OAK4 on Luxonis OS 1.45.0 (`oakctl device update`, [OAK4.md](modules/camera/oak4/OAK4.md))
    * smile: the Vimba 6 SDK, the Mako's UserSet2, the Windows sound and Max's audio, the PC section, then ```apps\smile\scripts\install.bat --from <site folder>``` ([smile's README](apps/smile/README.md))
    * a show machine: [docs/WINDOWS.md](docs/WINDOWS.md)
4. Development: VS Code's extensions, PowerShell 7 and Git's user name and email (below)

### ~~Visual C++ Redistributable and Visual Studio~~
* ~~Maybe it is necessary to instal visual studio with python and c++ modules (investigate on next clean install)~~
* ~~install latest Visual C++ Redistributable [download](https://aka.ms/vs/17/release/vc_redist.x86.exe) or [site](https://learn.microsoft.com/en-us/cpp/windows/latest-supported-vc-redist?view=msvc-170#latest-supported-redistributable-version) and RESTART~~

### ~~uv~~
* ~~In a terminal run ```winget install astral-sh.uv``` and open a new terminal; uv installs Python 3.12 and the packages~~

### Git for Windows
* ~~Install latest version from [site](https://git-scm.com/download/win)~~
* Configure Global User Name and Email
```git config --global user.name "M.Oostrik"```
```git config --global user.email "m.oostrik@gmail.com"```
* ~~Initialize Git LFS~~
~~```git lfs install```~~

### VSCode
* ~~Install latest version from [site](https://code.visualstudio.com/Download)~~
* install extentions
    * Error Lens
    * GLSL Lint
    * Pylance
    * Python
    * Python Debugger
    * ~~Python Environment Manager (deprecated)~~
    * Python Environments
    * Python Indent
    * Shader languages support for VS Code
* Install PowerShell 7 ```winget install Microsoft.PowerShell``` and make it the default terminal: Command Palette > Terminal: Select Default Profile > PowerShell

### ~~ExecutionPolicy~~
* ~~open powershell as administrator and type ```Set-ExecutionPolicy -ExecutionPolicy RemoteSigned```~~

### ~~FFmpeg~~
* ~~Install latest version from [site](https://ffmpeg.org/download.html#build-windows)~~
* ~~And add its 'bin' to path~~

### ~~Clone Project~~
* ~~```git clone https://github.com/moostrik/Poser2.git```~~

### Install Script
* in the terminal run ```scripts\install.bat```: it pulls the LFS files and builds `.venv` from `uv.lock`, with depthai 3 for the OAK4 (deep_flow, low_flow, smile); run it again after every pull that changes `uv.lock`; ```scripts\install.bat <app>``` does the same for the app's venv and builds its models (New PC, above)
* for the OAK-D apps (white_space, hd_trio) also run ```scripts\install.bat oakd```: it builds a second venv, `.venv-oakd`, with depthai 2 in place of depthai 3; `.venv` keeps depthai 3, so both stay installed. An app started with `.venv\Scripts\python` uses the `oak4` backend (```.venv\Scripts\python launcher.py -app deep_flow```), one started with `.venv-oakd` the OAK-D's `oak` backend (```.venv-oakd\Scripts\python launcher.py -app white_space```). Run it again after every pull that changes `uv.lock`, like `install.bat`
* and make the models for an app ```scripts\makemodels.bat <app_name>``` or all apps ```scripts\makemodels.bat all```, each app's in its own venv; it skips the engines that exist, ```scripts\makemodels.bat all rebuild``` builds them again (needed after a TensorRT upgrade: an engine only runs on the version that built it); run it again after a pull that changes the models in `data/models`, as the app looks for the engines of the models it names (the pose runs FP16, `rtmpose-l_<size>_fp16.trt`)
* a pose model's FP16 file is made from its FP32 original once, when a model is added, and kept in LFS beside it: ```uv run --with onnxconverter-common python tools/models/convert_onnx_to_fp16.py --input data/models/<name>.onnx --output data/models/<name>_fp16.onnx --fp32-norms --keep-io-types```
* The CUDA runtime comes with torch's wheel (CUDA 13) and TensorRT builds the engines from Python, so no CUDA toolkit is installed; the NVIDIA driver has to support CUDA 13

### Upgrade Packages
* ```uv lock --upgrade``` moves `uv.lock` to the newest versions that fit `pyproject.toml`; then ```scripts\install.bat```, the timing check below, and commit `uv.lock`
* The exact pins in `pyproject.toml` (depthai, torch, TensorRT) never move this way; the GPU stack moves only together with rebuilt TensorRT engines

### Run Project
* activate venv and run ```python launcher.py```
* ```python launcher.py -app <app> -sim``` plays the app's recordings through an OAK device (its detector and tracker); with `replay: true` under `camera.simulator` in the settings file they play on the host without a device, the people found by `modules/inference/person`, on ONNX or TensorRT by its `model_type`, at the input size of `resolution` (the ONNX files exported from the blobs' own weights by ```uv run --with ultralytics python tools/models/export_yolo_to_onnx.py```, the engines built by `scripts\makemodels.bat`) and followed by `modules/camera/replay`

### Timing
* ```scripts\timing.bat <app> [seconds]``` runs the app in simulation for a timing session (300 s after a 30 s warm-up, no settings window) and writes `logs/timing_<app>_<date>_<time>.json`: count, mean, p50, p95, p99 and max per timer, with the commit and the package versions
* ```.venv\Scripts\python scripts\timing_compare.py <baseline>.json <new>.json``` compares two reports; a timer fails when its mean is more than 5% slower or its p95 more than 10%
* The timers: inference (pose estimation, segmentation, optical flow estimation, person and face detection), GPU image upload, window similarity and correlation, tracker tick, each pose stage's filters, render update, draw and frame interval
* ```.venv\Scripts\python scripts\model_benchmark.py --app <app> [--model-type onnx] [--reference <earlier>.npz]``` times the app's enabled models through their workers without cameras (the startup settings file's resolutions, batch and crop size; seeded input or `--image`) and writes `logs/inference_<app>_<type>_<date>_<time>.json` in the same format, with each model's GPU memory, and an `.npz` of the outputs; `--reference` reports how far the outputs moved
* A package upgrade lands only when its report passes against a baseline of the build before it, both from the same simulation settings file on the same PC

### ~~Startup~~
* ~~make a shortcut to `scripts\start\<app>.bat`~~
* ~~Press Windows+R to open the Run dialog, type shell:Common Startup~~
* ~~move shortcut to startup folder~~


## HINTS
* .\.venv\Scripts\Activate.ps1