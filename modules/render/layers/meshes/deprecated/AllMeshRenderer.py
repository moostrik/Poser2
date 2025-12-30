# Standard library imports
import numpy as np

# Third-party imports

# Local application imports
from modules.gl.Mesh import Mesh
from modules.pose.Frame import Frame
from ..PoseMeshUtils import PoseVertexData, PoseMeshUtils, POSE_VERTEX_INDICES
from modules.DataHub import DataHub, DataHubType, PoseDataHubTypes
from modules.render.layers.LayerBase import LayerBase, Rect

class AllMeshRenderer(LayerBase):
    """Methods for updating meshes based on pose data."""
    def __init__(self, amount: int, data: DataHub, type: PoseDataHubTypes) -> None:
        self._amount: int = amount
        self._data: DataHub = data
        self._type: PoseDataHubTypes = type
        self.meshes: dict[int, Mesh] = {}

        self._p_pose: dict[int, Frame] = {}

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

    def draw(self, track_id: int, rect: Rect) -> None:
        pass

    def update(self) -> None:
        for id in range(self._amount):
            pose: Frame | None = self._data.get_item(DataHubType(self._type), id)
            if pose == self._p_pose.get(id, None):
                continue
            pose_mesh: Mesh | None = self.meshes.get(id, None)
            if pose is not None and pose_mesh is not None:
                vertex_data: PoseVertexData = PoseMeshUtils.compute_angled_vertices(pose.points, pose.angles)
                if vertex_data is not None:
                    pose_mesh.set_vertices(vertex_data.vertices)
                    pose_mesh.set_colors(vertex_data.colors)
                    pose_mesh.update()