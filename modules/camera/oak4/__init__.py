"""The oak4 backend: an OAK4 camera on depthai 3, implementing `modules.camera.CameraSource`
behind the same contract as the OAK-D's `modules.camera.oak`. It imports only on a depthai 3
install: `.venv-oak4`, built by `scripts\\install.bat oak4`."""

from .. import require_backend

require_backend('oak4')

from .camera import     Camera
from .device import     CameraLink, get_device_list
from .models import     ModelChain, POSE_MODEL, POSE_ARCHIVE, DETECTOR_MODEL, DETECTOR_ARCHIVE, TOP_DOWN_MODEL, TOP_DOWN_ARCHIVE
from .sensor import     DepthModel
from .settings import   Oak4CameraSettings
