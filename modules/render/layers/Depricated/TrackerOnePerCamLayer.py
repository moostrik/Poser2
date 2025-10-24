# Standard library imports
import numpy as np

# Third-party imports
from OpenGL.GL import * # type: ignore

# Local application imports
from modules.gl.Image import Image
from modules.gl.Fbo import Fbo, SwapFbo
from modules.gl.Text import draw_box_string, text_init

from modules.tracker.TrackerBase import TrackerType, TrackerMetadata
from modules.tracker.Tracklet import Tracklet, TrackletIdColor, TrackingStatus

from modules.data.CaptureDataHub import DataManager
from modules.pose.Pose import Pose
from modules.render.meshes.PoseMeshes import PoseMeshes
from modules.gl.LayerBase import LayerBase, Rect

from modules.utils.HotReloadMethods import HotReloadMethods

from modules.gl.shaders.Exposure import Exposure

class TrackerOnePerCamLayer(LayerBase):
    exposure_shader = Exposure()
    def __init__(self, data: DataManager, pose_meshes: PoseMeshes, cam_id: int) -> None:
        self.data: DataManager = data
        self.data_consumer_key: str = data.get_unique_consumer_key()
        self.pose_meshes: PoseMeshes = pose_meshes
        self.cam_id: int = cam_id
        self.cam_fbo: Fbo = Fbo()
        self.exp_fbo: Fbo = Fbo()
        self.cam_image: Image = Image()
        text_init()
        hot_reload = HotReloadMethods(self.__class__, True, True)

    def allocate(self, width: int, height: int, internal_format: int) -> None:
        self.cam_fbo.allocate(width, height, internal_format)
        self.exp_fbo.allocate(width, height, internal_format)
        if not TrackerOnePerCamLayer.exposure_shader.allocated:
            TrackerOnePerCamLayer.exposure_shader.allocate()

    def deallocate(self) -> None:
        self.cam_fbo.deallocate()
        self.exp_fbo.deallocate()
        if TrackerOnePerCamLayer.exposure_shader.allocated:
            TrackerOnePerCamLayer.exposure_shader.deallocate()

    def draw(self, rect: Rect) -> None:
        self.exp_fbo.draw(rect.x, rect.y, rect.width, rect.height)

    def update(self) -> None:
        key: int = self.cam_id


        self.shader: Exposure = Exposure()
        if not self.shader.allocated:
            self.shader.allocate()

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

        pose: Pose | None = self.data.get_pose(key, True, self.data_consumer_key)
        cam_image_np: np.ndarray | None = self.data.get_cam_image(key, True, self.data_consumer_key)

        if cam_image_np is not None:
            self.cam_image.set_image(cam_image_np)
            self.cam_image.update()

        cam_image_aspect_ratio: float = self.cam_image.width / self.cam_image.height
        width: float = cam_image_roi.width / cam_image_aspect_ratio
        x: float = cam_image_roi.x + (cam_image_roi.width - width) / 2.0


        LayerBase.setView(self.cam_fbo.width, self.cam_fbo.height)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

        self.cam_fbo.begin()
        glColor4f(1.0, 1.0, 0.5, 0.1)

        self.cam_image.draw_roi(0, 0, self.cam_fbo.width, self.cam_fbo.height,
                                x, cam_image_roi.y, width, cam_image_roi.height)


        glColor4f(1.0, 1.0, 1.0, 1.0)
        self.cam_fbo.end()

        if not TrackerOnePerCamLayer.exposure_shader.allocated:
            TrackerOnePerCamLayer.exposure_shader.allocate()
        TrackerOnePerCamLayer.exposure_shader.use(self.exp_fbo.fbo_id, self.cam_fbo.tex_id, 1.5, 0.1, 0.2)

        # self.exp_fbo.begin()
        # glClearColor(0.0, 0.0, 0.0, 1.0)
        # glClear(GL_COLOR_BUFFER_BIT)
        # self.exp_fbo.end()


    def clear_fbo(self) -> None:
        LayerBase.setView(self.cam_fbo.width, self.cam_fbo.height)

        glColor4f(0.0, 0.0, 0.0, 0.05)
        self.exp_fbo.begin()
        glBegin(GL_QUADS)
        glVertex2f(0.0, 0.0)
        glVertex2f(0.0, self.cam_fbo.height)
        glVertex2f(self.cam_fbo.width, self.cam_fbo.height)
        glVertex2f(self.cam_fbo.width, 0.0)
        glEnd()
        self.exp_fbo.end()
        glColor4f(1.0, 1.0, 1.0, 1.0)