# Standard library imports
import numpy as np

# Third-party imports
from OpenGL.GL import * # type: ignore

# Local application imports
from modules.gl.Image import Image
from modules.gl.Fbo import Fbo, SwapFbo
from modules.gl.Text import draw_box_string, text_init


from modules.DataHub import DataHub, DataType, PoseDataTypes
from modules.pose.Frame import Frame
from modules.pose.features.Points2D import PointLandmark
from modules.render.renderers import CamImageRenderer

from modules.DataHub import DataHub
from modules.gl.LayerBase import LayerBase, Rect

from modules.utils.HotReloadMethods import HotReloadMethods


class PoseCamLayer(LayerBase):
    def __init__(self, cam_id: int, data_hub: DataHub, data_type: PoseDataTypes, image_renderer: CamImageRenderer,) -> None:
        self._cam_id: int = cam_id
        self._data_hub: DataHub = data_hub
        self._fbo: Fbo = Fbo()
        self._image_renderer: CamImageRenderer = image_renderer
        self._p_pose: Frame | None = None

        self.data_type: PoseDataTypes = data_type

        text_init()
        hot_reload = HotReloadMethods(self.__class__, True, True)

    def allocate(self, width: int, height: int, internal_format: int) -> None:
        self._fbo.allocate(width, height, internal_format)

    def deallocate(self) -> None:
        self._fbo.deallocate()

    def draw(self, rect: Rect) -> None:
        self._fbo.draw(rect.x, rect.y, rect.width, rect.height)

    def update(self) -> None:
        key: int = self._cam_id

        pose: Frame | None = self._data_hub.get_item(DataType(self.data_type), key)

        if pose is self._p_pose:
            return # no update needed

        LayerBase.setView(self._fbo.width, self._fbo.height)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

        self._fbo.clear(0.0, 0.0, 0.0, 0.0)
        if pose is None:
            return

        pose_rect = pose.bbox.to_rect()

        self._fbo.begin()
        self._image_renderer.draw_roi(Rect(0, 0, self._fbo.width, self._fbo.height), pose_rect)

        self._fbo.end()


    def get_fbo(self) -> Fbo:
        return self._fbo
