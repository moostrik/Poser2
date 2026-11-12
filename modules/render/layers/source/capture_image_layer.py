# Standard library imports
from typing import TYPE_CHECKING

# Third-party imports
import torch

# Local application imports
from modules.gl import Tensor, Texture
from ..layer_base import LayerBase

if TYPE_CHECKING:
    from modules.inference import Capture


class CaptureImageLayer(LayerBase):
    """The color image of the capture it is given (`set_capture`), the one its depth and poses belong to, as the
    uploader made it (straightened, mirrored). Taken into GL at the next update."""

    def __init__(self) -> None:
        self._image: Tensor = Tensor()
        self._capture: "Capture | None" = None      # the capture to take in at the next update
        self._timestamp: float | None = None        # the capture time of the image it holds

    @property
    def texture(self) -> Texture:
        return self._image

    @property
    def timestamp(self) -> float | None:
        """The capture time of the image it holds (`Capture.timestamp`), None before the first."""
        return self._timestamp

    def set_capture(self, capture: "Capture") -> None:
        """The capture whose image the next update takes in; render thread."""
        self._capture = capture

    def deallocate(self) -> None:
        if self._image.allocated:
            self._image.deallocate()

    def update(self) -> None:
        capture: "Capture | None" = self._capture
        if capture is None or capture.timestamp == self._timestamp:
            return
        # The image was written on the uploader's stream: wait for it there, and keep its memory
        # from being reused until this stream's copy is done.
        stream: torch.cuda.Stream = torch.cuda.current_stream()
        stream.wait_event(capture.image.ready)
        capture.image.image.record_stream(stream)
        self._image.set_tensor(capture.image.image)
        self._image.update()
        self._timestamp = capture.timestamp
