# Poser2
depth cam pose synchony detection

## INSTALLATION

### Visual C++ Redistributable and Visual Studio
* Maybe it is necessary to instal visual studio with python and c++ modules (investigate on next clean install)
* install latest Visual C++ Redistributable [download](https://aka.ms/vs/17/release/vc_redist.x86.exe) or [site](https://learn.microsoft.com/en-us/cpp/windows/latest-supported-vc-redist?view=msvc-170#latest-supported-redistributable-version) and RESTART

### Python
* ~~Install python 3.10 from Microsoft Store (or find a better method)~~
* Install python 3.12 Windows Installer (64 bit) [download](https://www.python.org/ftp/python/3.12.10/python-3.12.10-amd64.exe) or [site](https://www.python.org/downloads/release/python-31210/)

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

### Cuda
* install CUDA 12.9.1 [download](https://developer.download.nvidia.com/compute/cuda/12.9.1/local_installers/cuda_12.9.1_576.57_windows.exe) or [site](https://developer.nvidia.com/cuda-12-9-1-download-archive?target_os=Windows&target_arch=x86_64&target_version=11&target_type=exe_local)

### FFmpeg
* Install latest version from [site](https://ffmpeg.org/download.html#build-windows)
* And add its 'bin' to path

### Clone Project
* ```git clone https://github.com/moostrik/Poser2.git```

### Install Script
* in the terminal run ```scripts\install.bat```
* and make the models for an app ```scripts/makemodels.bat <app_name>``` or all apps ```scripts/makemodels.bat all``` using the .venv environment

### Run Project
* activate venv and run ```python launcher.py```

### Timing
* ```scripts\timing.bat <app> [seconds]``` runs the app in simulation for a timing session (300 s after a 30 s warm-up, no settings window) and writes `logs/timing_<app>_<date>_<time>.json`: count, mean, p50, p95, p99 and max per timer, with the commit and the package versions
* ```.venv\Scripts\python scripts\timing_compare.py <baseline>.json <new>.json``` compares two reports; a timer fails when its mean or p95 is more than 5% slower
* The timers: inference (RTM pose, RVM, RAFT), GPU image upload, window similarity and correlation, tracker tick, each pose stage's filters, render update, draw and frame interval
* A package upgrade lands only when its report passes against a baseline of the build before it, both from the same simulation preset on the same PC

### Startup
* make a shortcut to `scripts\start\<app>.bat`
* Press Windows+R to open the Run dialog, type shell:Common Startup
* move shortcut to startup folder


## HINTS
* je moeder