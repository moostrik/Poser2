# DepthPose
depth cam pose synchony detection

## INSTALLATION

# Git for Windows
* Download from [site](https://git-scm.com/download/win)

# SSH Github
* create ssh key
```ssh-keygen -t ed25519 -C "m.oostrik@gmail.com"```
* copy key and add to github

# VSCode
* Download from [site](https://code.visualstudio.com/Download)
* install extentions
    * Pylance
    * Python
    * Python Debugger
    * Python Environments
    * Python Indent

# Python
* Install python 3.10 from Microsoft Store (or find a better method)

# Visual C++ Redistributable
* install and RESTART [Download](https://aka.ms/vs/17/release/vc_redist.x86.exe)

# Set-ExecutionPolicy
* open powershell as administrator and type ```Set-ExecutionPolicy -ExecutionPolicy RemoteSigned```

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


## OPTIONAL

# build MMPose (for cuda 12.1, probably works for newer cuda)
* install torch with cuda ```pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121```
(or, i forgot what i did) ```pip install torch==2.1.0 torchvision==0.16.0 --index-url https://download.pytorch.org/whl/cu121``` )
* install openmim ```pip install openmim```
* install mmcv ```mim install mmcv==2.1.0```   (this will build mmcv, takes abaout 15 minutes)
* install mmdet ```mim install mmdet==3.2.0```
* install mmpose ```mim install mmpose==1.3.2```
* mim can be used to download the models (If your use case is controlled poses / standard humans, coco is fine.
If your use case has varied poses or unusual angles, aic-coco is better.)
  * mim download mmpose --config rtmpose-l_8xb256-420e_aic-coco-256x192  --dest .\models
  * mim download mmpose --config rtmpose-l_8xb256-420e_aic-coco-384x288  --dest .\models
  * mim download mmpose --config rtmpose-m_8xb256-420e_aic-coco-256x192  --dest .\models
  * mim download mmpose --config rtmpose-m_8xb256-420e_aic-coco-384x288  --dest .\models
  * mim download mmpose --config rtmpose-s_8xb256-420e_aic-coco-256x192  --dest .\models
  * mim download mmpose --config rtmpose-t_8xb256-420e_aic-coco-256x192  --dest .\models
* the following links are helpful:
  * https://pytorch.org/get-started/locally/
  * https://pypi.org/project/torchvision/
  * https://github.com/pytorch/pytorch/wiki/PyTorch-Versions
  * https://mmpose.readthedocs.io/en/latest/installation.html
  * https://mmcv.readthedocs.io/en/latest/get_started/installation.html
  * https://mmpose.readthedocs.io/en/latest/model_zoo/body_2d_keypoint.html#coco-dataset

