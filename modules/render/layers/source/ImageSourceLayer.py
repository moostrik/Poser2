# Standard library imports
import numpy as np

# Third-party imports
from OpenGL.GL import * # type: ignore

# Local application imports
from modules.gl.Image import Image, Texture

from modules.DataHub import DataHub, DataHubType
from modules.render.layers.LayerBase import LayerBase, Rect


class ImageSourceLayer(LayerBase):
    def __init__(self, cam_id: int, data: DataHub) -> None:
        self._cam_id: int = cam_id
        self._data: DataHub = data
        self._image: Image = Image('BGR')
        self._p_frame: np.ndarray | None = None

    @property
    def texture(self) -> Texture:
        return self._image

    def deallocate(self) -> None:
        if self._image.allocated:
            self._image.deallocate()

    def update(self) -> None:
        frame: np.ndarray | None = self._data.get_item(DataHubType.cam_image, self._cam_id)

        if frame is None: # frames not initialized yet
            return

        if frame is not self._p_frame:
            self._image.set_image(frame)
            self._image.update()
            self._p_frame = frame


