# Standard library imports
from typing import TYPE_CHECKING

# Third-party imports
import torch

# Local application imports
from modules.board import HasCaptures
from modules.gl import Tensor, Texture
from ..layer_base import LayerBase, DataCache

if TYPE_CHECKING:
    from modules.inference import Capture, GpuImage


class CaptureImageLayer(LayerBase):
    """The color image of the camera's newest capture, the one its depth and poses belong to, as the
    uploader made it (straightened, mirrored). New with each capture."""

    def __init__(self, cam_id: int, board: HasCaptures) -> None:
        self._cam_id: int = cam_id
        self._board: HasCaptures = board
        self._image: Tensor = Tensor()
        self._data_cache: DataCache["GpuImage"] = DataCache["GpuImage"]()
        self._timestamp: float | None = None      # the capture time of the image it holds

    @property
    def texture(self) -> Texture:
        return self._image

    @property
    def timestamp(self) -> float | None:
        """The capture time of the image it holds (`Capture.timestamp`), None before the first."""
        return self._timestamp

    def deallocate(self) -> None:
        if self._image.allocated:
            self._image.deallocate()

    def update(self) -> None:
        capture: "Capture | None" = self._board.get_capture(self._cam_id)
        gpu_image: "GpuImage | None" = capture.image if capture is not None else None
        self._data_cache.set(gpu_image)

        if self._data_cache.lost:
            self._image.clear()
            self._timestamp = None

        if self._data_cache.idle or gpu_image is None or capture is None:
            return

        # The image was written on the uploader's stream: wait for it there, and keep its memory
        # from being reused until this stream's copy is done.
        stream: torch.cuda.Stream = torch.cuda.current_stream()
        stream.wait_event(gpu_image.ready)
        gpu_image.image.record_stream(stream)
        self._image.set_tensor(gpu_image.image)
        self._image.update()
        self._timestamp = capture.timestamp
