# Standard library imports
from typing import TYPE_CHECKING

# Third-party imports
import torch
from OpenGL.GL import GL_CLAMP_TO_BORDER, GL_NEAREST

# Local application imports
from modules.gl import Tensor, Texture
from ..layer_base import LayerBase

if TYPE_CHECKING:
    from modules.inference import Capture


class CaptureDepthLayer(LayerBase):
    """The depth of the capture it is given (`set_capture`), the one its image and poses belong to, as the uploader
    made it: metres in red, 0 where there is none, on the color image's view at the depth's own grid. Read NEAREST,
    so an edge never reads as a distance in between; beyond it is no depth. Taken into GL at the next update."""

    def __init__(self) -> None:
        self._depth: Tensor = Tensor(interpolation=GL_NEAREST, wrap=GL_CLAMP_TO_BORDER, border_color=(0.0, 0.0, 0.0, 0.0))
        self._capture: "Capture | None" = None      # the capture to take in at the next update
        self._timestamp: float | None = None        # the capture time of the depth it holds

    @property
    def texture(self) -> Texture:
        return self._depth

    @property
    def timestamp(self) -> float | None:
        """The capture time of the depth it holds (`Capture.timestamp`), None before the first."""
        return self._timestamp

    def set_capture(self, capture: "Capture") -> None:
        """The capture whose depth the next update takes in; render thread."""
        self._capture = capture

    def deallocate(self) -> None:
        if self._depth.allocated:
            self._depth.deallocate()

    def update(self) -> None:
        capture: "Capture | None" = self._capture
        if capture is None or capture.timestamp == self._timestamp:
            return
        # Written on the uploader's stream: wait for it there, and keep its memory until this stream's copy is done
        stream: torch.cuda.Stream = torch.cuda.current_stream()
        stream.wait_event(capture.depth.ready)
        capture.depth.depth.record_stream(stream)
        self._depth.set_tensor(capture.depth.depth)
        self._depth.update()
        self._timestamp = capture.timestamp
