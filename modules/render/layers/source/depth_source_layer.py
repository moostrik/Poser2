# Standard library imports
import numpy as np

# Local application imports
from modules.board import HasCameraDepth
from modules.camera import CameraFrame, mirrored_image
from modules.gl import Image, Texture
from ..layer_base import LayerBase, DataCache

DEPTH_RANGE_MM: float = 8000.0      # the distance shaded black


def depth_shade(depth: np.ndarray) -> np.ndarray:
    """uint16 millimetres (H, W) as float32 in [0, 1]: white up close, black at `DEPTH_RANGE_MM`
    and where there is no depth."""
    shade: np.ndarray = 1.0 - depth.astype(np.float32) / DEPTH_RANGE_MM
    shade[depth == 0] = 0.0
    return np.clip(shade, 0.0, 1.0)


class DepthSourceLayer(LayerBase):
    """The camera's depth as a texture, on the view of the camera image. Stays empty on a source
    without depth."""

    def __init__(self, cam_id: int, board: HasCameraDepth) -> None:
        self._cam_id: int = cam_id
        self._board: HasCameraDepth = board
        self._image: Image = Image()
        self._data_cache: DataCache[CameraFrame] = DataCache[CameraFrame]()
        self._dirty: bool = False

    @property
    def texture(self) -> Texture:
        return self._image

    @property
    def dirty(self) -> bool:
        return self._dirty

    def allocate(self, width: int | None = None, height: int | None = None, internal_format: int | None = None) -> None:
        pass  # Lazy allocation on first update

    def deallocate(self) -> None:
        self._image.deallocate()

    def update(self) -> None:
        self._dirty = False
        frame: CameraFrame | None = self._board.get_camera_depth(self._cam_id)
        self._data_cache.set(frame)

        if self._data_cache.lost:
            self._image.clear()

        if self._data_cache.idle or frame is None:
            return

        self._image.set_image(depth_shade(mirrored_image(frame) if frame.flip_h else frame.image))
        self._image.update()
        self._dirty = True
