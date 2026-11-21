"""SMILE's render board: the thread-safe store the app's threads write and the render reads at its tick."""

from modules.board import CameraImageStoreMixin

from .capture import CaptureStoreMixin
from .faces import FaceStoreMixin
from .generator_images import GeneratorImageStoreMixin
from .online_cell import OnlineCellStoreMixin
from .status import StatusStoreMixin


class RenderBoard(CameraImageStoreMixin, FaceStoreMixin, CaptureStoreMixin, GeneratorImageStoreMixin, StatusStoreMixin,
                  OnlineCellStoreMixin):
    """The camera's full uploaded frame (the module slice), and smile's own: the judged faces, the capture and its
    depth, the generator's image and depth, the status line and warm-up, and the online cell."""

    def __init__(self) -> None:
        CameraImageStoreMixin.__init__(self)
        FaceStoreMixin.__init__(self)
        CaptureStoreMixin.__init__(self)
        GeneratorImageStoreMixin.__init__(self)
        StatusStoreMixin.__init__(self)
        OnlineCellStoreMixin.__init__(self)
