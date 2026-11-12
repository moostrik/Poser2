# Standard library imports
from typing import TYPE_CHECKING

# Third-party imports
import torch
from OpenGL.GL import GL_CLAMP_TO_BORDER, GL_NEAREST

# Local application imports
from modules.board import HasCaptures
from modules.gl import Tensor, Texture
from ..layer_base import LayerBase, DataCache

if TYPE_CHECKING:
    from modules.inference import Capture, GpuDepth


class CaptureDepthLayer(LayerBase):
    """The depth of the camera's newest capture, the one its image and poses belong to, as the uploader made it:
    metres in red, 0 where there is none, on the color image's view at the depth's own grid. Read NEAREST, so an
    edge never reads as a distance in between; beyond it is no depth. New with each capture."""

    def __init__(self, cam_id: int, board: HasCaptures) -> None:
        self._cam_id: int = cam_id
        self._board: HasCaptures = board
        self._depth: Tensor = Tensor(interpolation=GL_NEAREST, wrap=GL_CLAMP_TO_BORDER, border_color=(0.0, 0.0, 0.0, 0.0))
        self._data_cache: DataCache["GpuDepth"] = DataCache["GpuDepth"]()
        self._timestamp: float | None = None      # the capture time of the depth it holds

    @property
    def texture(self) -> Texture:
        return self._depth

    @property
    def timestamp(self) -> float | None:
        """The capture time of the depth it holds (`Capture.timestamp`), None before the first."""
        return self._timestamp

    def deallocate(self) -> None:
        if self._depth.allocated:
            self._depth.deallocate()

    def update(self) -> None:
        capture: "Capture | None" = self._board.get_capture(self._cam_id)
        gpu_depth: "GpuDepth | None" = capture.depth if capture is not None else None
        self._data_cache.set(gpu_depth)

        if self._data_cache.lost:
            self._depth.clear()
            self._timestamp = None

        if self._data_cache.idle or gpu_depth is None or capture is None:
            return

        # Written on the uploader's stream: wait for it there, and keep its memory until this stream's copy is done
        stream: torch.cuda.Stream = torch.cuda.current_stream()
        stream.wait_event(gpu_depth.ready)
        gpu_depth.depth.record_stream(stream)
        self._depth.set_tensor(gpu_depth.depth)
        self._depth.update()
        self._timestamp = capture.timestamp
