"""The oak4 backend: an OAK4 camera on depthai 3, implementing `modules.camera.CameraSource`
behind the same contract as the OAK-D's `modules.camera.oak`. It imports only on a depthai 3
install: `.venv-oak4`, built by `scripts\\install.bat oak4`."""

from .. import require_backend

require_backend('oak4')

from .camera import     Camera
from .control import    SensorControl
from .definitions import ModelChain, CameraLink, DepthModel, get_device_list, \
                        POSE_MODEL, POSE_ARCHIVE, DETECTOR_MODEL, DETECTOR_ARCHIVE, TOP_DOWN_MODEL, TOP_DOWN_ARCHIVE
from .settings import   Oak4CameraSettings
