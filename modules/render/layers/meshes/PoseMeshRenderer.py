# Standard library imports
import numpy as np

# Third-party imports
from OpenGL.GL import * # type: ignore

# Local application imports
from modules.gl.Fbo import Fbo
from modules.gl.Mesh import Mesh

from modules.pose.Frame import Frame

from modules.DataHub import DataHub, DataHubType, PoseDataHubTypes
from modules.render.layers.meshes.PoseMeshUtils import PoseMeshUtils, POSE_VERTEX_INDICES, PoseVertexData
from modules.render.layers.LayerBase import LayerBase, Rect


class PoseMeshRenderer(LayerBase):
    def __init__(self, track_id: int, data: DataHub, data_type: PoseDataHubTypes, line_width: float = 10.0,
                 color: tuple[float, float, float, float] | None = None) -> None:
        self._track_id: int = track_id
        self._data: DataHub = data
        self._mesh: Mesh = Mesh()
        self._is_active: bool = False
        self._p_cam_poses: set[Frame] = set()

        self.data_type: PoseDataHubTypes = data_type
        self.line_width: float = line_width
        self.color: tuple[float, float, float, float] | None = color

    def allocate(self, width: int | None = None, height: int | None = None, internal_format: int | None = None) -> None:
        self._mesh.allocate()
        self._mesh.set_indices(POSE_VERTEX_INDICES)

    def deallocate(self) -> None:
        self._mesh.deallocate()

    def draw(self, rect: Rect) -> None:
        if not self._is_active:
            return
        self._mesh.draw(rect.x, rect.y, rect.width, rect.height)

    def update(self) -> None:
        pose: Frame | None = self._data.get_item(DataHubType(self.data_type), self._track_id)

        if not self._mesh.allocated:
            return

        if pose is None:
            self._is_active = False
            return
        else:
            self._is_active = True

        vertex_data: PoseVertexData = PoseMeshUtils.compute_vertices_and_colors(pose.points, color = self.color)
        self._mesh.set_vertices(vertex_data.vertices)
        self._mesh.set_colors(vertex_data.colors)
        self._mesh.update()