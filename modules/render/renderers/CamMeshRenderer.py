# Standard library imports
import numpy as np

# Third-party imports
from OpenGL.GL import * # type: ignore

# Local application imports
from modules.gl.Mesh import Mesh

from modules.DataHub import DataHub, DataType, PoseDataTypes
from modules.pose.Pose import Pose
from modules.render.renderers.PoseMeshUtils import PoseVertexData, PoseMeshUtils, POSE_VERTEX_INDICES
from modules.render.renderers.RendererBase import RendererBase
from modules.utils.PointsAndRects import Rect


class CamMeshRenderer(RendererBase):
    def __init__(self, cam_id: int, data: DataHub, type: PoseDataTypes, line_width: int = 2,
                 mesh_color: tuple[float, float, float, float] | None = None) -> None:
        self._data: DataHub = data
        self._cam_id: int = cam_id
        self._cam_meshes: dict[int, Mesh] = {}  # Lazily initialized, track_ids 1-8
        self._active_track_ids: set[int] = set()
        self.type: PoseDataTypes = type
        self.line_width: int = int(line_width)
        self.mesh_color: tuple[float, float, float, float] | None = mesh_color

    def allocate(self) -> None:
        pass

    def deallocate(self) -> None:
        """Deallocate all meshes when renderer is destroyed."""
        for mesh in self._cam_meshes.values():
            mesh.deallocate()
        self._cam_meshes.clear()

    def draw(self, rect: Rect) -> None:
        glLineWidth(self.line_width)

        for track_id in self._active_track_ids:
            if track_id in self._cam_meshes:
                self._cam_meshes[track_id].draw(rect.x, rect.y, rect.width, rect.height)

    def update(self) -> None:
        """Update meshes for active poses. Meshes are lazily initialized and cached (track_ids 1-8)."""
        cam_poses: set[Pose] = self._data.get_items_for_cam(DataType(self.type), self._cam_id)

        self._active_track_ids = {pose.track_id for pose in cam_poses}

        for pose in cam_poses:
            track_id = pose.track_id

            # Lazy initialization
            if track_id not in self._cam_meshes:
                mesh = Mesh()
                mesh.allocate()
                mesh.set_indices(POSE_VERTEX_INDICES)
                self._cam_meshes[track_id] = mesh

            # Update mesh data
            mesh: Mesh = self._cam_meshes[track_id]
            vertex_data: PoseVertexData = PoseMeshUtils.compute_vertices_and_colors(pose.points, pose.bbox.to_rect(), self.mesh_color)
            mesh.set_vertices(vertex_data.vertices)
            mesh.set_colors(vertex_data.colors)
            mesh.update()