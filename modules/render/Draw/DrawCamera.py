# Standard library imports
import numpy as np

# Third-party imports
from OpenGL.GL import * # type: ignore

# Local application imports
from modules.gl.Fbo import Fbo
from modules.gl.Image import Image
from modules.gl.Mesh import Mesh
from modules.gl.Text import draw_string, draw_box_string, text_init
from modules.gl.shaders.WS_PoseStream import WS_PoseStream

from modules.cam.depthcam.Definitions import Tracklet as DepthTracklet
from modules.pose.PoseDefinitions import Pose, PosePoints, PoseEdgeIndices, PoseAngleNames
from modules.tracker.TrackerBase import TrackerType, TrackerMetadata
from modules.tracker.Tracklet import Tracklet, TrackletIdColor, TrackingStatus
from modules.utils.PointsAndRects import Rect, Point2f

from modules.render.DataManager import DataManager
from modules.render.Draw.DrawBase import DrawBase, Rect
from modules.render.Mesh.PoseMeshes import PoseMeshes
from modules.render.DrawMethods import DrawMethods


class DrawCamera(DrawBase):
    def __init__(self, data: DataManager, pose_meshes: PoseMeshes, cam_id: int) -> None:
        self.data: DataManager = data
        self.fbo: Fbo = Fbo()
        self.image: Image = Image()
        self.cam_id: int = cam_id

        self.pose_meshes: PoseMeshes = pose_meshes

    def allocate(self, width: int, height: int, internal_format: int) -> None:
        self.fbo.allocate(width, height, internal_format)

    def deallocate(self) -> None:
        self.fbo.deallocate()
        self.image.deallocate()

    def update(self, only_if_dirty: bool) -> None:
        frame: np.ndarray | None = self.data.get_cam_image(self.cam_id)
        if frame is not None:
            self.image.set_image(frame)
            self.image.update()
        fbo: Fbo = self.fbo
        depth_tracklets: list[DepthTracklet] | None = self.data.get_depth_tracklets(self.cam_id, False)
        poses: list[Pose] = self.data.get_poses_for_cam(self.cam_id)
        meshes: dict[int, Mesh] = self.pose_meshes.meshes

        DrawBase.setView(fbo.width, fbo.height)
        fbo.begin()
        glClearColor(0.0, 0.0, 0.0, 1.0)
        self.image.draw(0, 0, fbo.width, fbo.height)
        DrawCamera.draw_camera_overlay(depth_tracklets, poses, meshes, 0, 0, fbo.width, fbo.height)
        glFlush()
        fbo.end()

    def draw(self, rect: Rect) -> None:
        self.fbo.draw(rect.x, rect.y, rect.width, rect.height)


    @staticmethod
    def draw_camera_overlay(depth_tracklets: list[DepthTracklet], poses: list[Pose], pose_meshes: dict[int, Mesh], x: float, y: float, width: float, height: float) -> None:
        for pose in poses:
            tracklet: Tracklet | None = pose.tracklet
            if tracklet is None or tracklet.is_removed or tracklet.is_lost:
                continue
            roi: Rect | None = pose.crop_rect
            mesh: Mesh = pose_meshes[pose.id]
            if roi is None or not mesh.isInitialized():
                continue
            roi_x, roi_y, roi_w, roi_h = roi.x, roi.y, roi.width, roi.height
            roi_x: float = x + roi_x * width
            roi_y: float = y + roi_y * height
            roi_w: float = roi_w * width
            roi_h: float = roi_h * height
            DrawMethods.draw_tracklet(tracklet, mesh, roi_x, roi_y, roi_w, roi_h, True, True, False)

        for depth_tracklet in depth_tracklets:
            DrawMethods.draw_depth_tracklet(depth_tracklet, 0, 0, width, height)

