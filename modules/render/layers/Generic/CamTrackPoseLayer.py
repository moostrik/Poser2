# Standard library imports
import numpy as np

# Third-party imports
from OpenGL.GL import * # type: ignore

# Local application imports
from modules.gl.Fbo import Fbo
from modules.gl.Image import Image
from modules.gl.Mesh import Mesh
from modules.gl.Text import draw_box_string, text_init

from modules.cam.depthcam.Definitions import Tracklet as DepthTracklet
from modules.pose.Pose import Pose
from modules.tracker.Tracklet import Tracklet

from modules.data.CaptureDataHub import DataManager
from modules.gl.LayerBase import LayerBase, Rect
from modules.render.meshes.PoseMeshes import PoseMeshes

from modules.utils.HotReloadMethods import HotReloadMethods


class CamTrackPoseLayer(LayerBase):
    def __init__(self, data: DataManager, pose_meshes: PoseMeshes, cam_id: int) -> None:
        self.data: DataManager = data
        self.data_consumer_key: str = data.get_unique_consumer_key()
        self.pose_meshes: PoseMeshes = pose_meshes
        self.fbo: Fbo = Fbo()
        self.image: Image = Image()
        self.cam_id: int = cam_id
        text_init()

        hot_reload = HotReloadMethods(self.__class__, True, True)

    def allocate(self, width: int, height: int, internal_format: int) -> None:
        self.fbo.allocate(width, height, internal_format)

    def deallocate(self) -> None:
        self.fbo.deallocate()
        self.image.deallocate()

    def draw(self, rect: Rect) -> None:
        self.fbo.draw(rect.x, rect.y, rect.width, rect.height)

    def update(self) -> None:
        frame: np.ndarray | None = self.data.get_cam_image(self.cam_id, True, self.data_consumer_key)
        if frame is not None:
            self.image.set_image(frame)
            self.image.update()

        depth_tracklets: list[DepthTracklet] | None = self.data.get_depth_tracklets(self.cam_id, False, self.data_consumer_key)
        poses: list[Pose] = self.data.get_poses_for_cam(self.cam_id)
        meshes: dict[int, Mesh] = self.pose_meshes.meshes

        LayerBase.setView(self.fbo.width, self.fbo.height)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

        self.fbo.begin()
        glClearColor(0.0, 0.0, 0.0, 1.0)
        self.image.draw(0, 0, self.fbo.width, self.fbo.height)
        glLineWidth(3.0)
        CamTrackPoseLayer.draw_camera_overlay(depth_tracklets, poses, meshes, 0, 0, self.fbo.width, self.fbo.height)
        self.fbo.end()

    @staticmethod
    def draw_camera_overlay(depth_tracklets: list[DepthTracklet], poses: list[Pose], pose_meshes: dict[int, Mesh], x: float, y: float, width: float, height: float) -> None:
        for pose in poses:
            tracklet: Tracklet | None = pose.tracklet
            if tracklet is None or tracklet.is_removed or tracklet.is_lost:
                continue
            roi: Rect | None = pose.crop_rect
            mesh: Mesh = pose_meshes[pose.tracklet.id]
            if roi is None or not mesh.isInitialized():
                continue
            roi_x, roi_y, roi_w, roi_h = roi.x, roi.y, roi.width, roi.height
            roi_x: float = x + roi_x * width
            roi_y: float = y + roi_y * height
            roi_w: float = roi_w * width
            roi_h: float = roi_h * height
            CamTrackPoseLayer.draw_tracklet(tracklet, mesh, roi_x, roi_y, roi_w, roi_h)

        for depth_tracklet in depth_tracklets:
            CamTrackPoseLayer.draw_depth_tracklet(depth_tracklet, 0, 0, width, height)

    @staticmethod
    def draw_depth_tracklet(tracklet: DepthTracklet, x: float, y: float, width: float, height: float) -> None:
        if tracklet.status == DepthTracklet.TrackingStatus.REMOVED:
            return

        t_x: float = x + tracklet.roi.x * width
        t_y: float = y + tracklet.roi.y * height
        t_w: float = tracklet.roi.width * width
        t_h: float = tracklet.roi.height* height

        r: float = 1.0
        g: float = 1.0
        b: float = 1.0
        a: float = min(tracklet.age / 100.0, 0.33)
        if tracklet.status == DepthTracklet.TrackingStatus.NEW:
            r, g, b, a = (1.0, 1.0, 1.0, 1.0)
        if tracklet.status == DepthTracklet.TrackingStatus.TRACKED:
            r, g, b, a = (0.0, 1.0, 0.0, a)
        if tracklet.status == DepthTracklet.TrackingStatus.LOST:
            r, g, b, a = (1.0, 0.0, 0.0, a)
        if tracklet.status == DepthTracklet.TrackingStatus.REMOVED:
            r, g, b, a = (1.0, 0.0, 0.0, 1.0)

        glColor4f(r, g, b, a)   # Set color
        glBegin(GL_QUADS)       # Start drawing a quad
        glVertex2f(t_x, t_y)        # Bottom left
        glVertex2f(t_x, t_y + t_h)    # Bottom right
        glVertex2f(t_x + t_w, t_y + t_h)# Top right
        glVertex2f(t_x + t_w, t_y)    # Top left
        glEnd()                 # End drawing
        glColor4f(1.0, 1.0, 1.0, 1.0)  # Reset color

        string: str
        t_x += t_w -6
        if t_x + 66 > width:
            t_x: float = width - 66
        t_y += 22
        string = f'ID: {tracklet.id}'
        draw_box_string(t_x, t_y, string)
        t_y += 22
        string = f'Age: {tracklet.age}'
        draw_box_string(t_x, t_y, string)

    @staticmethod
    def draw_tracklet(tracklet: Tracklet, pose_mesh: Mesh, x: float, y: float, width: float, height: float) -> None:
        glColor4f(0.0, 0.0, 0.0, 0.1)
        glBegin(GL_QUADS)
        glVertex2f(x, y)        # Bottom left
        glVertex2f(x, y + height)    # Bottom right
        glVertex2f(x + width, y + height)# Top right
        glVertex2f(x + width, y)    # Top left
        glEnd()                 # End drawing
        glColor4f(1.0, 1.0, 1.0, 1.0)  # Reset color

        if pose_mesh.isInitialized():
            pose_mesh.draw(x, y, width, height)


