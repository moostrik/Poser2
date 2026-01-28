# Standard library imports
import numpy as np
import time

# Third-party imports
from OpenGL.GL import * # type: ignore

# Local application imports
from modules.gl import Tensor, Texture

from modules.DataHub import DataHub, DataHubType
from modules.render.layers.LayerBase import LayerBase, DataCache
from modules.pose.batch.GPUFrame import GPUFrame


class ImageSourceLayer(LayerBase):
    def __init__(self, cam_id: int, data: DataHub) -> None:
        self._cam_id: int = cam_id
        self._data_hub: DataHub = data
        self._cuda_image: Tensor = Tensor()
        self._data_cache: DataCache[GPUFrame]= DataCache[GPUFrame]()


    @property
    def texture(self) -> Texture:
        return self._cuda_image

    def deallocate(self) -> None:
        if self._cuda_image.allocated:
            self._cuda_image.deallocate()

    def update(self) -> None:
        gpu_frame: GPUFrame | None = self._data_hub.get_item(DataHubType.gpu_frames, self._cam_id)
        self._data_cache.update(gpu_frame)

        # if self._data_cache.lost:
        #     self._image.clear()

        if self._data_cache.idle or gpu_frame is None:
            return

        self._cuda_image.set_tensor(gpu_frame.full_image)
        self._cuda_image.update()


