"""The oak4 backend: an OAK4 camera on depthai 3, implementing `modules.camera.CameraSource`
behind the same contract as the OAK-D's `modules.camera.oak`. Frames, the device detector and
tracker, the sensor controls and the IMU come in the next steps of the plan; this is the package
and its guard: it imports only on a depthai 3 install (`scripts\\install.bat oak4`)."""

from .. import require_backend

require_backend('oak4')
