# Standard library imports
import numpy as np

# Third-party imports

# Local application imports
from modules.gl.Mesh import Mesh
from modules.pose.Pose import Pose, PoseVertexData
from modules.pose.features.PoseVertices import POSE_VERTEX_INDICES
from modules.CaptureDataHub import CaptureDataHub
from modules.gl.LayerBase import LayerBase, Rect

class PoseMeshes(LayerBase):
    """Methods for updating meshes based on pose data."""
    def __init__(self, data: CaptureDataHub, amount: int) -> None:
        self.data: CaptureDataHub = data
        self.data_consumer_key: str = data.get_unique_consumer_key()
        self.amount: int = amount
        self.meshes: dict[int, Mesh] = {}

    def allocate(self) -> None:
        for i in range(self.amount):
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
        for id in range(self.amount):
            pose: Pose | None = self.data.get_pose(id, True, self.data_consumer_key)
            pose_mesh: Mesh | None = self.meshes.get(id, None)
            if pose is not None and pose_mesh is not None:
                vertex_data: PoseVertexData | None = pose.vertex_data
                if vertex_data is not None:
                    pose_mesh.set_vertices(vertex_data.vertices)
                    pose_mesh.set_colors(vertex_data.colors)
                    pose_mesh.update()