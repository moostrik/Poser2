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

from modules.render.DataManager import DataManager
from modules.render.renders.BaseRender import BaseRender, Rect

from modules.utils.HotReloadMethods import HotReloadMethods

class CentreCameraRender(BaseRender):
    def __init__(self, data: DataManager, cam_id: int) -> None:
        self.data: DataManager = data
        self.cam_id: int = cam_id
        self.cam_fbo: Fbo = Fbo()
        self.cam_image: Image = Image()

        self.last_tracklet: Tracklet | None = None
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
        if tracklets:
            self.last_tracklet = tracklets[0]

        if self.last_tracklet is None:
            self.clear_fbo()
            return

        tracklet_metadata: TrackerMetadata | None = self.last_tracklet.metadata
        if tracklet_metadata is None or tracklet_metadata.tracker_type != TrackerType.ONEPERCAM:
            self.clear_fbo()
            return
        cam_image_roi: Rect = getattr(self.last_tracklet.metadata, "smooth_rect", Rect(0.0, 0.0, 1.0, 1.0))

        cam_image_np: np.ndarray | None = self.data.get_cam_image(key, True, self.key())

        if cam_image_np is not None:
            self.cam_image.set_image(cam_image_np)
            self.cam_image.update()

        cam_image_aspect_ratio: float = self.cam_image.width / self.cam_image.height
        width: float = cam_image_roi.width / cam_image_aspect_ratio
        x: float = cam_image_roi.x + (cam_image_roi.width - width) / 2.0


        BaseRender.setView(self.cam_fbo.width, self.cam_fbo.height)
        self.cam_fbo.begin()

        self.cam_image.draw_roi(0, 0, self.cam_fbo.width, self.cam_fbo.height,
                                x, cam_image_roi.y, width, cam_image_roi.height)

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