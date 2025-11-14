# Standard library imports
import numpy as np

# Third-party imports

# Local application imports
from modules.gl.Mesh import Mesh
from modules.pose.Pose import Pose
from modules.deprecated.PoseVertices import PoseVertexData, PoseVertexFactory
from modules.deprecated.PoseVertices import POSE_VERTEX_INDICES
from modules.DataHub import DataHub, DataType, POSE_ENUMS
from modules.gl.LayerBase import LayerBase, Rect

class PoseMesh(LayerBase):
    """Methods for updating meshes based on pose data."""
    def __init__(self, amount: int, data: DataHub, type: DataType) -> None:
        self._amount: int = amount
        self._data: DataHub = data
        if type not in POSE_ENUMS:
            raise ValueError(f"Invalid DataType for CamTrackPoseLayer: {type}")
        self._type: DataType = type
        self.meshes: dict[int, Mesh] = {}

        self._p_pose: dict[int, Pose] = {}

    def allocate(self) -> None:
        for i in range(self._amount):
            if i not in self.meshes:
                mesh = Mesh()
                mesh.allocate()
                mesh.set_indices(POSE_VERTEX_INDICES)
                self.meshes[i] = mesh

    def deallocate(self) -> None:
        for mesh in self.meshes.values():
            mesh.deallocate()
        self.meshes.clear()

    def draw(self, rect: Rect) -> None:
        pass

    def update(self) -> None:
        for id in range(self._amount):
            pose: Pose | None = self._data.get_item(self._type, id)
            if pose == self._p_pose.get(id, None):
                continue
            pose_mesh: Mesh | None = self.meshes.get(id, None)
            if pose is not None and pose_mesh is not None:
                vertex_data: PoseVertexData = PoseVertexFactory.compute_angled_vertices(pose.points, pose.angles)
                if vertex_data is not None:
                    pose_mesh.set_vertices(vertex_data.vertices)
                    pose_mesh.set_colors(vertex_data.colors)
                    pose_mesh.update()