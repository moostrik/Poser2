# Standard library imports
import numpy as np
import time

# Third-party imports
from OpenGL.GL import * # type: ignore

# Local application imports
from modules.gl import Tensor, Texture

from modules.board import HasCameraImages
from modules.render.layers.LayerBase import LayerBase, DataCache
from modules.inference.camera_image import CameraImage


class ImageSourceLayer(LayerBase):
    def __init__(self, cam_id: int, board: HasCameraImages) -> None:
        self._cam_id: int = cam_id
        self._board: HasCameraImages = board
        self._cuda_image: Tensor = Tensor()
        self._data_cache: DataCache[CameraImage]= DataCache[CameraImage]()
        self._dirty: bool = False

    @property
    def texture(self) -> Texture:
        return self._cuda_image

    @property
    def dirty(self) -> bool:
        return self._dirty

    def deallocate(self) -> None:
        if self._cuda_image.allocated:
            self._cuda_image.deallocate()

    def update(self) -> None:
        self._dirty = False
        gpu_frame: CameraImage | None = self._board.get_camera_image(self._cam_id)
        self._data_cache.update(gpu_frame)

        if self._data_cache.lost:
            self._cuda_image.clear()

        if self._data_cache.idle or gpu_frame is None:
            return

        self._cuda_image.set_tensor(gpu_frame.image)
        self._cuda_image.update()
        self._dirty = True
