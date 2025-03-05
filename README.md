# DepthPose
depth cam pose synchony detection

## INSTALLATION

# install git for windows
* Download from [site](https://git-scm.com/download/win)

# create ssh key for github
* create ssh key
```ssh-keygen -t ed25519 -C "m.oostrik@gmail.com"```
* copy key and add to github

# install visual studio code and extentions
https://code.visualstudio.com/Download

* Python
* python Indent
* Python Environment Manager
* IntelliCode

# clone project
```git@github.com:moostrik/camDiffusers.git```
* init and update submodules

# install python 3.10.6
* Download from [site](https://www.python.org/downloads/release/python-3106/)
* start webui to downoad dependencies

# run install.bat

# run start.bat

# add start.bat to startup
* make a shortcut of start.bat
* Press Windows+R to open the Run dialog, type shell:Common Startup
* move shortcut to startup folder

# onnx CUDA install
* Using onnxruntime-gpy 1.18.1
* [instructions](https://onnxruntime.ai/docs/install/#cuda-and-cudnn) follow 'CUDA and CuDNN', not 'Pyhton installs'
* express install cuda 11.8 from [site](https://developer.nvidia.com/cuda-11-8-0-download-archive?target_os=Windows&target_arch=x86_64&target_version=11&target_type=exe_local)
* install cudnn 8.9.2 from [site](https://developer.nvidia.com/rdp/cudnn-archive) instructions [here](https://docs.nvidia.com/deeplearning/cudnn/archives/cudnn-890/install-guide/index.html) not [here](https://docs.nvidia.com/deeplearning/cudnn/latest/installation/windows.html)
* install Zlib [download](http://www.winimage.com/zLibDll/zlib123dllx64.zip), following instructions from cudnn
* add cudnn and Zlib dll folders to Path (RUN, control sysdm.cpl, Advanced, Environment Variables, add multiple values to variable 'path' )

# Hints
* RUNNING SCRIPTS IS DISABLED: ```Set-ExecutionPolicy -ExecutionPolicy RemoteSigned```
* Could not find the DLL(s) 'msvcp140.dll: https://learn.microsoft.com/en-US/cpp/windows/latest-supported-vc-redist?view=msvc-170