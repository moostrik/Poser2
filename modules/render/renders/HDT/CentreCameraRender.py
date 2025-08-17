# Standard library imports
import numpy as np

# Third-party imports
from OpenGL.GL import * # type: ignore

# Local application imports
from modules.gl.Image import Image
from modules.gl.Fbo import Fbo, SwapFbo
from modules.gl.Text import draw_box_string, text_init

from modules.pose.PoseDefinitions import Pose, PoseAngleNames
from modules.tracker.TrackerBase import TrackerType, TrackerMetadata
from modules.tracker.Tracklet import Tracklet, TrackletIdColor, TrackingStatus

from modules.render.DataManager import DataManager
from modules.render.renders.BaseRender import BaseRender, Rect

from modules.utils.HotReloadMethods import HotReloadMethods

class CentreCameraRender(BaseRender):
    def __init__(self, data: DataManager, cam_id: int) -> None:
        self.data: DataManager = data
        self.cam_id: int = cam_id
        self.cam_fbo: Fbo = Fbo()
        self.cam_image: Image = Image()

        self.last_Rect: Rect | None = None
        # self.last_tracklet: Tracklet | None = None
        text_init()
        hot_reload = HotReloadMethods(self.__class__, True, True)

    def allocate(self, width: int, height: int, internal_format: int) -> None:
        self.cam_fbo.allocate(width, height, internal_format)

    def deallocate(self) -> None:
        self.cam_fbo.deallocate()

    def draw(self, rect: Rect) -> None:
        self.cam_fbo.draw(rect.x, rect.y, rect.width, rect.height)

    def update(self) -> None:
        key: int = self.cam_id

        tracklets: list[Tracklet] = self.data.get_tracklets_for_cam(self.cam_id)
        if not tracklets:
            self.clear_fbo()
            return
        # cam_image_roi: Rect = getattr(self.last_tracklet.metadata, "smooth_rect", Rect(0.0, 0.0, 1.0, 1.0))

        pose: Pose | None = self.data.get_pose(key, True, self.key())
        if pose is not None:
            if pose.smooth_rect is not None:
                self.last_Rect = pose.smooth_rect

        if self.last_Rect is None:
            self.last_Rect = Rect(0.0, 0.0, 1.0, 1.0)

        cam_image_np: np.ndarray | None = self.data.get_cam_image(key, True, self.key())

        if cam_image_np is not None:
            self.cam_image.set_image(cam_image_np)
            self.cam_image.update()



        # print (self.cam_image.width,self.cam_image.height)
        cam_image_aspect_ratio: float = self.cam_image.width / self.cam_image.height

        width: float =  self.last_Rect.width / cam_image_aspect_ratio

        x: float =  self.last_Rect.x + ( self.last_Rect.width - width) / 2.0


        BaseRender.setView(self.cam_fbo.width, self.cam_fbo.height)
        self.cam_fbo.begin()

        self.cam_image.draw_roi(0, 0, self.cam_fbo.width, self.cam_fbo.height,
                                x,  self.last_Rect.y, width,  self.last_Rect.height)

        glColor4f(1.0, 1.0, 1.0, 1.0)
        self.cam_fbo.end()


    def clear_fbo(self) -> None:
        BaseRender.setView(self.cam_fbo.width, self.cam_fbo.height)
        self.cam_fbo.begin()
        glClearColor(0.0, 0.0, 0.0, 1.0)
        glClear(GL_COLOR_BUFFER_BIT)
        self.cam_fbo.end()

    def get_fbo(self) -> Fbo:
        return self.cam_fbo