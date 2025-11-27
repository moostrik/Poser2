# Standard library imports
import numpy as np

# Third-party imports
from OpenGL.GL import * # type: ignore

# Local application imports
from modules.gl.Image import Image

from modules.DataHub import DataHub, DataHubType
from modules.render.renderers.RendererBase import RendererBase
from modules.utils.PointsAndRects import Rect


class CamImageRenderer(RendererBase):
    def __init__(self, cam_id: int, data: DataHub) -> None:
        self._cam_id: int = cam_id
        self._data: DataHub = data
        self._image: Image = Image()
        self._p_frame: np.ndarray | None = None

    def allocate(self) -> None:
        pass

    def deallocate(self) -> None:
        if self._image.allocated:
            self._image.deallocate()

    def draw(self, rect: Rect) -> None:
        if self._image.allocated:
             self._image.draw(rect.x, rect.y, rect.width, rect.height)

    def draw_roi(self, draw_rect: Rect, text_rect: Rect) -> None:
        if self._image.allocated:
             self._image.draw_roi(draw_rect.x, draw_rect.y, draw_rect.width, draw_rect.height,
                                  text_rect.x, text_rect.y, text_rect.width, text_rect.height)

    def update(self) -> None:
        frame: np.ndarray | None = self._data.get_item(DataHubType.cam_image, self._cam_id)

        if frame is None: # frames not initialized yet
            return

        if frame is not self._p_frame:
            self._image.set_image(frame)
            self._image.update()
            self._p_frame = frame


