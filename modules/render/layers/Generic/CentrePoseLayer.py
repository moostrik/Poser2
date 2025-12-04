# Standard library imports
from dataclasses import replace

# Third-party imports
import numpy as np
from OpenGL.GL import * # type: ignore

# Local application imports
from modules.gl.Fbo import Fbo

from modules.pose.Frame import Frame

from modules.DataHub import DataHub, DataHubType, PoseDataHubTypes
from modules.gl.LayerBase import LayerBase, Rect

from modules.render.layers.generic.PoseLineLayer import PoseLineLayer
from modules.pose.features import Points2D

from modules.gl.shaders.PosePointLines import PosePointLines as shader

from modules.utils.HotReloadMethods import HotReloadMethods


class CentrePoseLayer(PoseLineLayer):
    def __init__(self, track_id: int, data: DataHub, data_type: PoseDataHubTypes,
             line_width: float = 4.0, line_smooth: float = 2.0, use_scores: bool = True, use_bbox: bool = False,
             color: tuple[float, float, float, float] | None = None) -> None:
        super().__init__(track_id, data, data_type, line_width, line_smooth, use_scores, use_bbox, color)
        self._points: Points2D = Points2D.create_dummy()

    def setCentrePoints(self, points: Points2D) -> None:
        self._points = points

    def update(self) -> None:
        if not PoseLineLayer._shader.allocated:
            PoseLineLayer._shader.allocate(monitor_file=True)

        pose: Frame | None = self._data.get_item(DataHubType(self.data_type), self._track_id)

        if pose is not None:
            pose = replace(pose, points=self._points)

        if pose is self._p_pose:
            return # no update needed
        self._p_pose = pose

        LayerBase.setView(self._fbo.width, self._fbo.height)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

        self._fbo.clear(0.0, 0.0, 0.0, 0.0)
        if pose is None:
            return

        line_width: float = 1.0 / self._fbo.height * self.line_width
        line_smooth: float = 1.0 / self._fbo.height * self.line_smooth

        PoseLineLayer._shader.use(self._fbo.fbo_id, pose.points, line_width=line_width, line_smooth=line_smooth, color=self.color, use_scores=self.use_scores)