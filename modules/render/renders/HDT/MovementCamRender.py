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
from modules.pose.PoseDefinitions import Pose, PoseAngleNames
from modules.render.meshes.PoseMeshes import PoseMeshes
from modules.render.renders.BaseRender import BaseRender, Rect

from modules.utils.HotReloadMethods import HotReloadMethods

from modules.gl.shaders.Exposure import Exposure

class MovementCamRender(BaseRender):
    exposure_shader = Exposure()
    def __init__(self, data: DataManager, cam_id: int) -> None:
        self.data: DataManager = data
        self.cam_id: int = cam_id
        self.color_fbo: Fbo = Fbo()
        self.exp_fbo: Fbo = Fbo()
        self.cam_image: Image = Image()
        text_init()
        hot_reload = HotReloadMethods(self.__class__, True, True)

    def allocate(self, width: int, height: int, internal_format: int) -> None:
        self.color_fbo.allocate(width, height, internal_format)
        self.exp_fbo.allocate(width, height, internal_format)
        if not MovementCamRender.exposure_shader.allocated:
            MovementCamRender.exposure_shader.allocate()

    def deallocate(self) -> None:
        self.color_fbo.deallocate()
        self.exp_fbo.deallocate()
        if MovementCamRender.exposure_shader.allocated:
            MovementCamRender.exposure_shader.deallocate()

    def draw(self, rect: Rect) -> None:
        self.exp_fbo.draw(rect.x, rect.y, rect.width, rect.height)

    def update(self, cam_fbo: Fbo) -> None:
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


        alpha: float = 0.1
        if self.cam_id == 0: # Red
            glColor4f(1.0, 0.5, 0.5, alpha) # Red
        elif self.cam_id == 1: # Yellow
            glColor4f(1.0, 1.0, 0.5, alpha)
        elif self.cam_id == 2: # Cyan
            glColor4f(0.5, 1.0, 1.0, alpha)


        BaseRender.setView(self.color_fbo.width, self.color_fbo.height)
        self.color_fbo.begin()
        cam_fbo.draw(0, 0, self.color_fbo.width, self.color_fbo.height)
        glColor4f(1.0, 1.0, 1.0, 1.0)

        self.color_fbo.end()


        BaseRender.setView(self.exp_fbo.width, self.exp_fbo.height)

        if not MovementCamRender.exposure_shader.allocated:
            MovementCamRender.exposure_shader.allocate()

        exposure = 1.5
        gamma = 0.1
        brightness = 0.2

        MovementCamRender.exposure_shader.use(self.exp_fbo.fbo_id, self.color_fbo.tex_id, exposure, gamma, brightness)

        # self.exp_fbo.begin()
        # glClearColor(0.0, 0.0, 0.0, 1.0)
        # glClear(GL_COLOR_BUFFER_BIT)
        # self.exp_fbo.end()





    def clear_fbo(self) -> None:
        BaseRender.setView(self.exp_fbo.width, self.exp_fbo.height)

        glColor4f(0.0, 0.0, 0.0, 0.05)
        self.exp_fbo.begin()
        glBegin(GL_QUADS)
        glVertex2f(0.0, 0.0)
        glVertex2f(0.0, self.exp_fbo.height)
        glVertex2f(self.exp_fbo.width, self.exp_fbo.height)
        glVertex2f(self.exp_fbo.width, 0.0)
        glEnd()
        self.exp_fbo.end()
        glColor4f(1.0, 1.0, 1.0, 1.0)

    def get_fbo(self) -> Fbo:
        return self.exp_fbo