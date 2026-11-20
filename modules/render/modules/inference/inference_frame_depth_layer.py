# Standard library imports
from typing import TYPE_CHECKING

# Third-party imports
import torch
from OpenGL.GL import GL_CLAMP_TO_BORDER, GL_NEAREST

# Local application imports
from modules.gl import Tensor, Texture
from modules.inference import Depth
from ...layer_base import LayerBase

if TYPE_CHECKING:
    from modules.inference import GpuDepth, InferenceFrame


class InferenceFrameDepthLayer(LayerBase):
    """The depth of the frame it is given (`set_inference_frame`), the one its image and poses belong to, as the uploader
    made it: metres in red, 0 where there is none, on the color image's view at the depth's own grid. Read NEAREST,
    so an edge never reads as a distance in between; beyond it is no depth. Taken into GL at the next update."""

    def __init__(self) -> None:
        self._depth: Tensor = Tensor(interpolation=GL_NEAREST, wrap=GL_CLAMP_TO_BORDER, border_color=(0.0, 0.0, 0.0, 0.0))
        self._frame: "InferenceFrame | None" = None      # the frame to take in at the next update
        self._timestamp: float | None = None        # the capture time of the depth it holds

    @property
    def texture(self) -> Texture:
        return self._depth

    @property
    def timestamp(self) -> float | None:
        """The capture time of the depth it holds (`InferenceFrame.timestamp`), None before the first."""
        return self._timestamp

    def set_inference_frame(self, frame: "InferenceFrame") -> None:
        """The frame whose depth the next update takes in; main thread."""
        self._frame = frame

    def deallocate(self) -> None:
        if self._depth.allocated:
            self._depth.deallocate()

    def update(self) -> None:
        frame: "InferenceFrame | None" = self._frame
        if frame is None or frame.timestamp == self._timestamp:
            return
        # Written on the uploader's stream: wait for it there, and keep its memory until this stream's copy is done
        stream: torch.cuda.Stream = torch.cuda.current_stream()
        depth: GpuDepth | None = frame[Depth].depth
        if depth is None:
            return
        depth.depth.record_stream(stream)       # handed on written; its memory kept until this stream's copy is done
        self._depth.set_tensor(depth.depth)
        self._depth.update()
        self._timestamp = frame.timestamp
