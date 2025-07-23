# Standard library imports
import numpy as np

# Third-party imports
from OpenGL.GL import * # type: ignore

# Local application imports
from modules.gl.Image import Image
from modules.gl.Fbo import Fbo
from modules.gl.Text import draw_box_string, text_init

from modules.tracker.TrackerBase import TrackerType, TrackerMetadata
from modules.tracker.Tracklet import Tracklet, TrackletIdColor, TrackingStatus

from modules.render.DataManager import DataManager
from modules.pose.PoseDefinitions import Pose, PoseAngleNames
from modules.render.meshes.PoseMeshes import PoseMeshes
from modules.render.renders.BaseRender import BaseRender, Rect

from modules.utils.HotReloadMethods import HotReloadMethods

from modules.gl.shaders.NoiseSimplex import NoiseSimplex

class OnePerCamTrackerRender(BaseRender):
    def __init__(self, data: DataManager, pose_meshes: PoseMeshes, cam_id: int) -> None:
        self.data: DataManager = data
        self.pose_meshes: PoseMeshes = pose_meshes
        self.cam_id: int = cam_id
        self.fbo: Fbo = Fbo()
        self.cam_image: Image = Image()
        text_init()
        self.noise_shader: NoiseSimplex = NoiseSimplex()

        hot_reload = HotReloadMethods(self.__class__, True, True)

    def allocate(self, width: int, height: int, internal_format: int) -> None:
        self.fbo.allocate(width, height, internal_format)

    def deallocate(self) -> None:
        self.fbo.deallocate()

    def draw(self, rect: Rect) -> None:
        self.fbo.draw(rect.x, rect.y, rect.width, rect.height)

    def update(self) -> None:
        key: int = self.cam_id


        self.noise_shader: NoiseSimplex = NoiseSimplex()
        if not self.noise_shader.allocated:
            self.noise_shader.allocate()

        tracklets: list[Tracklet] = self.data.get_tracklets_for_cam(self.cam_id)
        if not tracklets:
            self.clear_fbo()
            return
        tracklet: Tracklet = tracklets[0]
        if tracklet.is_removed:
            self.clear_fbo()
            return

        tracklet_metadata: TrackerMetadata | None = tracklet.metadata
        if tracklet_metadata is None or tracklet_metadata.tracker_type != TrackerType.ONEPERCAM:
            self.clear_fbo()
            return
        cam_image_roi: Rect = getattr(tracklet.metadata, "smooth_rect", Rect(0.0, 0.0, 1.0, 1.0))
        # print(draw_rect)

        pose: Pose | None = self.data.get_pose(key, True, self.key())
        cam_image_np: np.ndarray | None = self.data.get_cam_image(key, True, self.key())

        if cam_image_np is not None:
            self.cam_image.set_image(cam_image_np)
            self.cam_image.update()

        cam_image_aspect_ratio: float = self.cam_image.width / self.cam_image.height
        width: float = cam_image_roi.width / cam_image_aspect_ratio
        x: float = cam_image_roi.x + (cam_image_roi.width - width) / 2.0

        BaseRender.setView(self.fbo.width, self.fbo.height)
        self.fbo.begin()
        glColor4f(1.0, 1.0, 1.0, 0.1)

        self.cam_image.draw_roi(0, 0, self.fbo.width, self.fbo.height,
                                x, cam_image_roi.y, width, cam_image_roi.height)


        glColor4f(1.0, 1.0, 1.0, 1.0)
        self.fbo.end()


    def clear_fbo(self) -> None:
        """Clear the render"""
        BaseRender.setView(self.fbo.width, self.fbo.height)

        glColor4f(0.0, 0.0, 0.0, 0.2)
        # self.noise_shader.use(self.fbo.fbo_id, 0.001, self.fbo.width, self.fbo.height)

        self.fbo.begin()


        glColor4f(0.0, 0.0, 0.0, 0.05)
        # draw rectangle covering the whole FBO
        glBegin(GL_QUADS)
        glVertex2f(0.0, 0.0)
        glVertex2f(0.0, self.fbo.height)
        glVertex2f(self.fbo.width, self.fbo.height)
        glVertex2f(self.fbo.width, 0.0)
        glEnd()

        glColor4f(1.0, 1.0, 1.0, 1.0)
        self.fbo.end()