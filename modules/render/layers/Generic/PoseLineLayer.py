# Standard library imports
import numpy as np

# Third-party imports
from OpenGL.GL import * # type: ignore

# Local application imports
from modules.gl.Fbo import Fbo

from modules.pose.Frame import Frame

from modules.DataHub import DataHub, DataHubType, PoseDataHubTypes
from modules.gl.LayerBase import LayerBase, Rect

from modules.gl.shaders.PosePointLines import PosePointLines as shader

from modules.utils.HotReloadMethods import HotReloadMethods


class PoseLineLayer(LayerBase):
    _shader = shader()

    def __init__(self, track_id: int, data: DataHub, data_type: PoseDataHubTypes,
                 line_width: float = 4.0, line_smooth: float = 2.0, use_scores: bool = True, use_bbox: bool = False,
                 color: tuple[float, float, float, float] | None = None) -> None:
        self._track_id: int = track_id
        self._data: DataHub = data
        self._fbo: Fbo = Fbo()
        self._p_pose: Frame | None = None

        self.data_type: PoseDataHubTypes = data_type
        self.line_width: float = line_width
        self.line_smooth: float = line_smooth
        self.use_scores: bool = use_scores
        self.use_bbox: bool = use_bbox
        self.color: tuple[float, float, float, float] | None = color

        hot_reload = HotReloadMethods(self.__class__, True, True)

    def allocate(self, width: int, height: int, internal_format: int) -> None:
        self._fbo.allocate(width, height, internal_format)
        if not PoseLineLayer._shader.allocated:
            PoseLineLayer._shader.allocate(monitor_file=True)

    def deallocate(self) -> None:
        self._fbo.deallocate()
        if PoseLineLayer._shader.allocated:
            PoseLineLayer._shader.deallocate()

    def draw(self, rect: Rect) -> None:
        if self.use_bbox and self._p_pose is not None:
            box_rect: Rect = self._p_pose.bbox.to_rect()
            rect = box_rect.affine_transform(rect)

        self._fbo.draw(rect.x, rect.y, rect.width, rect.height)

    def update(self) -> None:
        if not PoseLineLayer._shader.allocated:
            PoseLineLayer._shader.allocate(monitor_file=True)

        pose: Frame | None = self._data.get_item(DataHubType(self.data_type), self._track_id)

        if pose is self._p_pose:
            return # no update needed
        self._p_pose = pose

        LayerBase.setView(self._fbo.width, self._fbo.height)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

        self._fbo.clear(0.0, 0.0, 0.0, 0.0)
        if pose is None:
            return

        line_width: float = 1.0 / self._fbo.height * self.line_width * 2
        line_smooth: float = 1.0 / self._fbo.height * self.line_smooth * 2

        PoseLineLayer._shader.use(self._fbo.fbo_id, pose.points, line_width=line_width, line_smooth=line_smooth, color=self.color, use_scores=self.use_scores)