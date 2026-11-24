"""The oak backend: an OAK-D camera on depthai 2. It imports only on a depthai 2 install
(`.venv-oakd`, built by `scripts\\install.bat oakd`); the OAK4 runs on depthai 3 in `modules.camera.oak4`."""

from .. import require_backend

require_backend('oak')

from .camera import     Camera
from .settings import   OakCameraSettings
