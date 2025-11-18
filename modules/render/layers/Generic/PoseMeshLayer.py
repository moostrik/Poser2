# Standard library imports
import numpy as np

# Third-party imports
from OpenGL.GL import * # type: ignore

# Local application imports
from modules.gl.Fbo import Fbo
from modules.gl.Mesh import Mesh

from modules.pose.Pose import Pose

from modules.DataHub import DataHub, DataType, PoseDataTypes
from modules.gl.LayerBase import LayerBase, Rect
from modules.render.meshes.PoseMeshes import PoseMeshes


class PoseMeshLayer(LayerBase):
    def __init__(self, track_id: int, data: DataHub, type: PoseDataTypes) -> None:
        # for now make sure the pose meshes are for the correct data type
        self._data: DataHub = data
        self._mesh: Mesh = Mesh()
        self._fbo: Fbo = Fbo()
        self._track_id: int = track_id
        self._type: PoseDataTypes = type
        self._p_cam_poses: set[Pose] = set()

    def allocate(self, width: int, height: int, internal_format: int) -> None:
        self._fbo.allocate(width, height, internal_format)

    def deallocate(self) -> None:
        self._fbo.deallocate()

    def draw(self, rect: Rect) -> None:
        self._fbo.draw(rect.x, rect.y, rect.width, rect.height)

    def update(self) -> None:
        cam_poses: set[Pose] = self._data.get_items_for_cam(DataType(self._type), self._track_id)
        if cam_poses is self._p_cam_poses:
            # Sets contain the same pose objects (by pointer), no update needed
            return
        self._p_cam_poses = cam_poses

        LayerBase.setView(self._fbo.width, self._fbo.height)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

        if not cam_poses: # no poses available
            self._fbo.clear(0.0, 0.0, 0.0, 0.0) # Clear with transparent color
            return

        meshes: dict[int, Mesh] = self._pose_meshes.meshes
        cam_meshes: dict[int, Mesh] = {pose.track_id: meshes[pose.track_id] for pose in cam_poses if pose.track_id in meshes}
        cam_bboxes: dict[int, Rect] = {pose.track_id: pose.bbox.to_rect() for pose in cam_poses}
        for box in cam_bboxes.values():
            box.x *= self._fbo.width
            box.y *= self._fbo.height
            box.width *= self._fbo.width
            box.height *= self._fbo.height

        glLineWidth(3.0)
        self._fbo.clear(0.0, 0.0, 0.0, 0.0) # Clear with transparent color
        self._fbo.begin()

        for track_id in cam_meshes:
            mesh = cam_meshes[track_id]
            bbox = cam_bboxes[track_id]
            if mesh.isInitialized():
                mesh.draw(bbox.x, bbox.y, bbox.width, bbox.height)

        self._fbo.end()
