# Standard library imports
import numpy as np

# Third-party imports
from OpenGL.GL import * # type: ignore

# Local application imports
from modules.gl.Image import Image

from modules.DataHub import DataHub, DataType
from modules.gl.LayerBase import LayerBase, Rect


class CamImageLayer(LayerBase):
    def __init__(self, cam_id: int, data: DataHub) -> None:
        self._cam_id: int = cam_id
        self._data: DataHub = data
        self._image: Image = Image()
        self._p_frame: np.ndarray | None = None


    def allocate(self, width: int, height: int, internal_format: int) -> None:
        pass


    def deallocate(self) -> None:
        pass


    def draw(self, rect: Rect) -> None:
        if self._image.allocated:
             self._image.draw(rect.x, rect.y, rect.width, rect.height)


    def update(self) -> None:
        frame: np.ndarray | None = self._data.get_item(DataType.cam_image, self._cam_id)

        if frame is None: # frames not initialized yet
            return

        if frame is not self._p_frame:
            self._image.set_image(frame)
            self._image.update()
            self._p_frame = frame


