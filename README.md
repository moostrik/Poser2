# Poser2
depth cam pose synchony detection

## INSTALLATION

# Visual Studio
* Maybe it is necessary to instal visual studio with python and c++ modules
* investigate on next install

# Git for Windows
* Download from [site](https://git-scm.com/download/win)
* Configure Global User Name and Email
```git config --global user.name "M.Oostrik"```
```git config --global user.email "m.oostrik@gmail.com"```
* Initialize Git LFS
```git lfs install```

# SSH Github
* create ssh key
```ssh-keygen -t ed25519 -C "m.oostrik@gmail.com"```
* copy key and add to github
* test

# Python
* Install python 3.10 from Microsoft Store (or find a better method)

# VSCode
* Download from [site](https://code.visualstudio.com/Download)
* install extentions
    * Pylance
    * Python
    * Python Debugger
    * Python Environment Manager (deprecated) or Python Environments
    * Python Indent

# Visual C++ Redistributable
* install and RESTART [Download](https://aka.ms/vs/17/release/vc_redist.x86.exe)

# Set-ExecutionPolicy
* open powershell as administrator and type ```Set-ExecutionPolicy -ExecutionPolicy RemoteSigned```

# FFmpeg
* Download from [site](https://ffmpeg.org/download.html#build-windows)
* And add its 'bin' to path

# Cuda
* for now use CUDA 12.9 [site](https://developer.nvidia.com/cuda-12-9-0-download-archive)

# Clone Project
* ```git clone git@github.com:moostrik/DepthPose.git```

# Install Project
* in powershell run ```.\install.bat```

# Run Project
* activate venv and run ```python .\launcher.py```
* or in powershell run ```.\start.bat```

# Startup
* make a shortcut of start.bat
* Press Windows+R to open the Run dialog, type shell:Common Startup
* move shortcut to startup folder


## HINTS

# shotcut vscode
* ctrl+shift+p
* Tasks:Open User Tasks
```json
{
  "version": "2.0.0",
  "tasks": [
    {
      "label": "Copy Python Command",
      "type": "shell",
      "command": "echo python launcher.py -s studio -sim | clip",
      "presentation": {
        "reveal": "never",
        "panel": "dedicated",
      },
      "problemMatcher": []
    }
  ]
}
```
* Preferences:Open Keyboard Shortcuts (JSON)
```json
[
    {
        "key": "ctrl+alt+d",
        "command": "workbench.action.toggleLightDarkThemes"
    },
    {
      "key": "ctrl+1",
      "command": "workbench.action.tasks.runTask",
      "args": "Copy Python Command"
    }
]
```
