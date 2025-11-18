# Standard library imports
import numpy as np

# Third-party imports
from OpenGL.GL import * # type: ignore

# Local application imports
from modules.gl.Fbo import Fbo
from modules.gl.Mesh import Mesh

from modules.pose.Pose import Pose

from modules.DataHub import DataHub, DataType, PoseDataTypes
from modules.render.renderers.PoseMeshUtils import PoseMeshUtils, POSE_VERTEX_INDICES, PoseVertexData
from modules.render.renderers.RendererBase import RendererBase
from modules.utils.PointsAndRects import Rect


class PoseMeshRenderer(RendererBase):
    def __init__(self, track_id: int, data: DataHub, type: PoseDataTypes, line_width: float = 2.0,
                 color: tuple[float, float, float] | None = None) -> None:
        self._track_id: int = track_id
        self._data: DataHub = data
        self._mesh: Mesh = Mesh()
        self._is_active: bool = False
        self._p_cam_poses: set[Pose] = set()

        self.type: PoseDataTypes = type
        self.line_width: float = line_width
        self.color: tuple[float, float, float] | None = color

    def allocate(self) -> None:
        self._mesh.allocate()
        self._mesh.set_indices(POSE_VERTEX_INDICES)

    def deallocate(self) -> None:
        self._mesh.deallocate()

    def draw(self, rect: Rect) -> None:
        if not self._is_active:
            return
        self._mesh.draw(rect.x, rect.y, rect.width, rect.height)

    def update(self) -> None:
        pose: Pose | None = self._data.get_item(DataType(self.type), self._track_id)

        if pose is None:
            self._is_active = False
            return
        else:
            self._is_active = True

        vertex_data: PoseVertexData = PoseMeshUtils.compute_vertices(pose.points, self.color)
        self._mesh.set_vertices(vertex_data.vertices)
        self._mesh.set_colors(vertex_data.colors)
        self._mesh.update()