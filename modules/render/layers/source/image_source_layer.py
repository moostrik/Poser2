# Standard library imports
import numpy as np
import time

# Local application imports
from modules.gl import Tensor, Texture

import torch
from typing import TYPE_CHECKING

from modules.board import HasCameraImages
from ..layer_base import LayerBase, DataCache

if TYPE_CHECKING:
    from modules.inference import GpuImage


class ImageSourceLayer(LayerBase):
    def __init__(self, cam_id: int, board: HasCameraImages) -> None:
        self._cam_id: int = cam_id
        self._board: HasCameraImages = board
        self._cuda_image: Tensor = Tensor()
        self._data_cache: DataCache["GpuImage"] = DataCache["GpuImage"]()

    @property
    def texture(self) -> Texture:
        return self._cuda_image

    def deallocate(self) -> None:
        if self._cuda_image.allocated:
            self._cuda_image.deallocate()

    def update(self) -> None:
        gpu_image: "GpuImage | None" = self._board.get_camera_image(self._cam_id)
        self._data_cache.set(gpu_image)

        if self._data_cache.lost:
            self._cuda_image.clear()

        if self._data_cache.idle or gpu_image is None:
            return

        # Written on the uploader's stream: wait for it there, and keep its memory from being reused until this
        # stream's copy is done
        stream: torch.cuda.Stream = torch.cuda.current_stream()
        stream.wait_event(gpu_image.ready)
        gpu_image.image.record_stream(stream)
        self._cuda_image.set_tensor(gpu_image.image)
        self._cuda_image.update()
