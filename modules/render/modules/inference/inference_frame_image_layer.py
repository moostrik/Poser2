# Standard library imports
from typing import TYPE_CHECKING

# Third-party imports
import torch

# Local application imports
from modules.gl import Tensor, Texture
from ...layer_base import LayerBase

if TYPE_CHECKING:
    from modules.inference import InferenceFrame


class InferenceFrameImageLayer(LayerBase):
    """The color image of the frame it is given (`set_inference_frame`), the one its depth and poses belong to, as the
    uploader made it (straightened, mirrored). Taken into GL at the next update."""

    def __init__(self) -> None:
        self._image: Tensor = Tensor()
        self._frame: "InferenceFrame | None" = None      # the frame to take in at the next update
        self._timestamp: float | None = None        # the capture time of the image it holds

    @property
    def texture(self) -> Texture:
        return self._image

    @property
    def timestamp(self) -> float | None:
        """The capture time of the image it holds (`InferenceFrame.timestamp`), None before the first."""
        return self._timestamp

    def set_inference_frame(self, frame: "InferenceFrame") -> None:
        """The frame whose image the next update takes in; render thread."""
        self._frame = frame

    def deallocate(self) -> None:
        if self._image.allocated:
            self._image.deallocate()

    def update(self) -> None:
        frame: "InferenceFrame | None" = self._frame
        if frame is None or frame.timestamp == self._timestamp:
            return
        # Handed on written; its memory is kept from being reused until this stream's copy is done
        stream: torch.cuda.Stream = torch.cuda.current_stream()
        frame.image.image.record_stream(stream)
        self._image.set_tensor(frame.image.image)
        self._image.update()
        self._timestamp = frame.timestamp
