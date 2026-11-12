---
description: The camera package: its thread and what it hands over
paths:
  - "modules/camera/**"
---
# Camera

- The camera's thread runs only the camera; it hands frames to the uploader and the recorder
- `modules/camera/oak/` runs on depthai 2 in `.venv` (`install.bat`) and `modules/camera/oak4/` on depthai 3 in a second venv, `.venv-oak4` (`install.bat oak4`); `oak_backend()` says which the running venv has, so an app never imports both
