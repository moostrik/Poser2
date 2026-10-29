# Poser2
depth cam pose synchony detection

## INSTALLATION

### Visual C++ Redistributable and Visual Studio
* Maybe it is necessary to instal visual studio with python and c++ modules (investigate on next clean install)
* install latest Visual C++ Redistributable [download](https://aka.ms/vs/17/release/vc_redist.x86.exe) or [site](https://learn.microsoft.com/en-us/cpp/windows/latest-supported-vc-redist?view=msvc-170#latest-supported-redistributable-version) and RESTART

### uv
* In a terminal run ```winget install astral-sh.uv``` and open a new terminal; uv installs Python 3.12 and the packages

### Git for Windows
* Install latest version from [site](https://git-scm.com/download/win)
* Configure Global User Name and Email
```git config --global user.name "M.Oostrik"```
```git config --global user.email "m.oostrik@gmail.com"```
* Initialize Git LFS
```git lfs install```

### VSCode
* Install latest version from [site](https://code.visualstudio.com/Download)
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

### ExecutionPolicy
* open powershell as administrator and type ```Set-ExecutionPolicy -ExecutionPolicy RemoteSigned```

### FFmpeg
* Install latest version from [site](https://ffmpeg.org/download.html#build-windows)
* And add its 'bin' to path

### Clone Project
* ```git clone https://github.com/moostrik/Poser2.git```

### Install Script
* in the terminal run ```scripts\install.bat```: it pulls the LFS files and builds `.venv` from `uv.lock` (run it again after every pull that changes `uv.lock`)
* and make the models for an app ```scripts/makemodels.bat <app_name>``` or all apps ```scripts/makemodels.bat all``` using the .venv environment; it skips the engines that exist, ```scripts/makemodels.bat all rebuild``` builds them again (needed after a TensorRT upgrade: an engine only runs on the version that built it)
* The CUDA runtime comes with torch's wheel (CUDA 13) and TensorRT builds the engines from Python, so no CUDA toolkit is installed; the NVIDIA driver has to support CUDA 13

### Upgrade Packages
* ```uv lock --upgrade``` moves `uv.lock` to the newest versions that fit `pyproject.toml`; then ```scripts\install.bat```, the timing check below, and commit `uv.lock`
* The exact pins in `pyproject.toml` (depthai, torch, TensorRT) never move this way; the GPU stack moves only together with rebuilt TensorRT engines

### Run Project
* activate venv and run ```python launcher.py```
* ```python launcher.py -app <app> -sim``` plays the app's recordings through an OAK device (its detector and tracker); with `replay: true` under `camera.simulator` in the preset they play on the host without a device, the people found by `data/models/yolov8n_coco_512x448.onnx` (exported from the blobs' own weights by ```uv run --with ultralytics python tools/models/export_yolo_to_onnx.py```) and followed by `modules/camera/replay`

### Timing
* ```scripts\timing.bat <app> [seconds]``` runs the app in simulation for a timing session (300 s after a 30 s warm-up, no settings window) and writes `logs/timing_<app>_<date>_<time>.json`: count, mean, p50, p95, p99 and max per timer, with the commit and the package versions
* ```.venv\Scripts\python scripts\timing_compare.py <baseline>.json <new>.json``` compares two reports; a timer fails when its mean is more than 5% slower or its p95 more than 10%
* The timers: inference (RTM pose, RVM, RAFT), GPU image upload, window similarity and correlation, tracker tick, each pose stage's filters, render update, draw and frame interval
* ```.venv\Scripts\python -m modules.inference.tools.benchmark --app <app> [--model-type onnx] [--reference <earlier>.npz]``` times the app's enabled models through its runners without cameras (the startup preset's resolutions, batch and crop size; seeded input or `--image`) and writes `logs/inference_<app>_<type>_<date>_<time>.json` in the same format, with each model's GPU memory, and an `.npz` of the outputs; `--reference` reports how far the outputs moved
* A package upgrade lands only when its report passes against a baseline of the build before it, both from the same simulation preset on the same PC

### Startup
* make a shortcut to `scripts\start\<app>.bat`
* Press Windows+R to open the Run dialog, type shell:Common Startup
* move shortcut to startup folder


## HINTS
* je moeder