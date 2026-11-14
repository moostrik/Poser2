---
description: The camera package: its thread and what it hands over
paths:
  - "modules/camera/**"
---
# Camera

- The camera's thread runs only the camera
- `modules/camera/oak/` runs on depthai 2 in `.venv` (`install.bat`) and `modules/camera/oak4/` on depthai 3 in `.venv-oak4` (`install.bat oak4`); an app imports one of them, never both
