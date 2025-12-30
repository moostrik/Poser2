# Standard library imports
import numpy as np

# Third-party imports
from OpenGL.GL import * # type: ignore

# Local application imports
from modules.gl import Fbo

from modules.pose.Frame import Frame

from modules.DataHub import DataHub, DataHubType, PoseDataHubTypes
from modules.render.layers.LayerBase import LayerBase, Rect
from modules.utils.PointsAndRects import Rect

from modules.gl.shaders.PosePointDots import PosePointDots as shader

from modules.utils.HotReloadMethods import HotReloadMethods


class PoseDotLayer(LayerBase):
    def __init__(self, track_id: int, data: DataHub, data_type: PoseDataHubTypes,
                 dot_size: float = 4.0, dot_smooth: float = 2.0,
                 color: tuple[float, float, float, float] | None = None) -> None:
        self._track_id: int = track_id
        self._data: DataHub = data
        self._fbo: Fbo = Fbo()
        self._p_pose: Frame | None = None

        self.data_type: PoseDataHubTypes = data_type
        self.dot_size: float = dot_size
        self.dot_smooth: float = dot_smooth
        self.color: tuple[float, float, float, float] | None = color

        self._shader: shader = shader()

        hot_reload = HotReloadMethods(self.__class__, True, True)

    def allocate(self, width: int, height: int, internal_format: int) -> None:
        self._fbo.allocate(width, height, internal_format)
        self._shader.allocate()

    def deallocate(self) -> None:
        self._fbo.deallocate()
        self._shader.deallocate()

    def draw(self, rect: Rect) -> None:
        self._fbo.draw(rect.x, rect.y, rect.width, rect.height)

    def update(self) -> None:

        pose: Frame | None = self._data.get_item(DataHubType(self.data_type), self._track_id)

        if pose is self._p_pose:
            return # no update needed
        self._p_pose = pose

        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

        self._fbo.clear(0.0, 0.0, 0.0, 0.0)
        if pose is None:
            return

        dot_size: float = 1.0 / self._fbo.height * self.dot_size * 2
        dot_smooth: float = 1.0 / self._fbo.height * self.dot_smooth * 2

        self._shader.use(self._fbo.fbo_id, pose.points, dot_size=dot_size, dot_smooth=dot_smooth)