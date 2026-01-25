# Standard library imports
import numpy as np

# Third-party imports
from OpenGL.GL import * # type: ignore

# Local application imports
from modules.gl.Image import Image, Texture

from modules.DataHub import DataHub, DataHubType
from modules.render.layers.LayerBase import LayerBase, DataCache, Rect


class ImageSourceLayer(LayerBase):
    def __init__(self, cam_id: int, data: DataHub) -> None:
        self._cam_id: int = cam_id
        self._data: DataHub = data
        self._image: Image = Image('BGR')
        self._data_cache: DataCache[np.ndarray]= DataCache[np.ndarray]()

    @property
    def texture(self) -> Texture:
        return self._image

    def deallocate(self) -> None:
        if self._image.allocated:
            self._image.deallocate()

    def update(self) -> None:
        frame: np.ndarray | None = self._data.get_item(DataHubType.cam_image, self._cam_id)
        self._data_cache.update(frame)

        if self._data_cache.lost:
            self._image.clear()

        if self._data_cache.idle or frame is None:
            return

        self._image.set_image(frame)
        self._image.update()


