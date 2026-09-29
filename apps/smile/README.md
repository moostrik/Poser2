# SMILE
semi real time webcam diffusers

## INSTALLATION

### Visual C++ Redistributable
* install latest Visual C++ Redistributable from [site](https://learn.microsoft.com/en-us/cpp/windows/latest-supported-vc-redist?view=msvc-170#latest-supported-redistributable-version) and RESTART

### Python
* Install python 3.12 Windows Installer (64 bit) [download](https://www.python.org/ftp/python/3.12.10/python-3.12.10-amd64.exe) or [site](https://www.python.org/downloads/release/python-31210/)
* choose **Customize installation**, uncheck "Add python.exe to PATH" and keep the py launcher
* verify with ```py -0p``` that `-3.12` is listed

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
    * Python
    * Python Environments
    * Python Indent
    * Pylance

### ExecutionPolicy
* open powershell as administrator and type ```Set-ExecutionPolicy -ExecutionPolicy RemoteSigned```

### Allied Vision camera
* Download from [site](https://www.alliedvision.com/en/products/vimba-sdk/)
* run installer as administrator
* restart

### Install Script
* in the terminal run ```scripts\install.bat```

### Run Project
* run ```scripts\start.bat```

### Startup
* make a shortcut to `scripts\start.bat`
* Press Windows+R to open the Run dialog, type shell:Common Startup
* move shortcut to startup folder


## HINTS
* Could not find the DLL(s) 'msvcp140.dll: https://learn.microsoft.com/en-US/cpp/windows/latest-supported-vc-redist?view=msvc-170
