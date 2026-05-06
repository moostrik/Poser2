# Poser2
depth cam pose synchony detection

## INSTALLATION

### Visual C++ Redistributable and Visual Studio
* Maybe it is necessary to instal visual studio with python and c++ modules (investigate on next clean install)
* install Visual C++ Redistributable [Download](https://aka.ms/vs/17/release/vc_redist.x86.exe) and RESTART

### Python
* ~~Install python 3.10 from Microsoft Store (or find a better method)~~
* Download python 3.12 Windows Installer (64 bit) from [site](https://www.python.org/downloads/release/python-31210/)

### Git for Windows
* Download from [site](https://git-scm.com/download/win)
* Configure Global User Name and Email
```git config --global user.name "M.Oostrik"```
```git config --global user.email "m.oostrik@gmail.com"```
* Initialize Git LFS
```git lfs install```

### VSCode
* Download from [site](https://code.visualstudio.com/Download)
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
* for now use CUDA 12.9 [site](https://developer.nvidia.com/cuda-12-9-0-download-archive)

### FFmpeg
* Download from [site](https://ffmpeg.org/download.html#build-windows)
* And add its 'bin' to path

### Clone Project
* ```https://github.com/moostrik/Poser2.git```

### Install Script
* in the terminal run ```scripts\install.bat```

### Run Project
* activate venv and run ```python launcher.py```

### Startup
* make a shortcut to `scripts\start\<app>.bat`
* Press Windows+R to open the Run dialog, type shell:Common Startup
* move shortcut to startup folder


## HINTS
* je moeder