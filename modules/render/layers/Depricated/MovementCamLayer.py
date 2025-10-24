# Standard library imports
import numpy as np

# Third-party imports
from OpenGL.GL import * # type: ignore

# Local application imports
from modules.gl.Image import Image
from modules.gl.Fbo import Fbo, SwapFbo
from modules.gl.Text import draw_box_string, text_init

from modules.pose.PoseStream import PoseStreamData
from modules.tracker.Tracklet import Tracklet

from modules.data.CaptureDataHub import DataManager
from modules.gl.LayerBase import LayerBase, Rect

from modules.utils.HotReloadMethods import HotReloadMethods

from modules.gl.shaders.Exposure import Exposure
from modules.gl.shaders.Contrast import Contrast

class MovementCamLayer(LayerBase):
    exposure_shader = Exposure()
    contrast_shader = Contrast()

    def __init__(self, data: DataManager, cam_id: int) -> None:
        self.data: DataManager = data
        self.data_consumer_key: str = data.get_unique_consumer_key()
        self.cam_id: int = cam_id
        self.color_fbo: Fbo = Fbo()
        self.exp_fbo: Fbo = Fbo()
        self.con_fbo: Fbo = Fbo()
        self.cam_image: Image = Image()

        self.movement: float = 0.0
        self.movement_for_synchrony: float = 0.0
        text_init()
        hot_reload = HotReloadMethods(self.__class__, True, True)

    def allocate(self, width: int, height: int, internal_format: int) -> None:
        self.color_fbo.allocate(width, height, internal_format)
        self.exp_fbo.allocate(width, height, internal_format)
        self.con_fbo.allocate(width, height, internal_format)
        if not MovementCamLayer.exposure_shader.allocated:
            MovementCamLayer.exposure_shader.allocate()
        if not MovementCamLayer.contrast_shader.allocated:
            MovementCamLayer.contrast_shader.allocate()

    def deallocate(self) -> None:
        self.color_fbo.deallocate()
        self.exp_fbo.deallocate()
        self.con_fbo.deallocate()
        if MovementCamLayer.exposure_shader.allocated:
            MovementCamLayer.exposure_shader.deallocate()
        if MovementCamLayer.contrast_shader.allocated:
            MovementCamLayer.contrast_shader.deallocate()

    def draw(self, rect: Rect) -> None:
        self.con_fbo.draw(rect.x, rect.y, rect.width, rect.height)

    def update(self, cam_fbo: Fbo) -> None:
        key: int = self.cam_id



        tracklets: list[Tracklet] = self.data.get_tracklets_for_cam(self.cam_id)
        if not tracklets:
            self.clear_fbo()
            return
        tracklet: Tracklet = tracklets[0]
        if tracklet.is_removed:
            self.clear_fbo()
            return

        glColor4f(0.0, 0., 0., 0.01)
        pose_stream: PoseStreamData | None = self.data.get_pose_stream(key, True, self.data_consumer_key)
        if pose_stream is not None:
            self.movement_for_synchrony = pose_stream.mean_movement
            if pose_stream.mean_movement > 0.009:
                self.movement = 0.03

                alpha: float = 0.25
                if self.cam_id == 0: # Red
                    glColor4f(1.0, 0., 0., alpha) # Red
                elif self.cam_id == 1: # Yellow
                    glColor4f(0.84, 0.76, 0., alpha)
                elif self.cam_id == 2: # Cyan
                    glColor4f(0.0, 0.9, 1.0, alpha)
            else:
                self.movement = 0.001
            # if key == 0:
            #     print(f'pose_stream.mean_movement: {pose_stream.mean_movement: .3f}')




        LayerBase.setView(self.color_fbo.width, self.color_fbo.height)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

        self.color_fbo.begin()
        # glClearColor(0.0, 0.0, 0.0, 1.0)
        # glClear(GL_COLOR_BUFFER_BIT)
        cam_fbo.draw(0, 0, self.color_fbo.width, self.color_fbo.height)
        glColor4f(1.0, 1.0, 1.0, 1.0)

        self.color_fbo.end()


        LayerBase.setView(self.exp_fbo.width, self.exp_fbo.height)

        if not MovementCamLayer.exposure_shader.allocated:
            MovementCamLayer.exposure_shader.allocate()
        if not MovementCamLayer.contrast_shader.allocated:
            MovementCamLayer.contrast_shader.allocate()

        brightness: float = 0.5
        contrast: float = 1.9

        exposure: float = 1.1
        offset: float = 0.0
        gamma: float = 0.2

        MovementCamLayer.exposure_shader.use(self.exp_fbo.fbo_id, self.color_fbo.tex_id, exposure, offset, gamma)

        MovementCamLayer.contrast_shader.use(self.con_fbo.fbo_id, self.color_fbo.tex_id, brightness, contrast)

        # self.exp_fbo.begin()
        # glClearColor(0.0, 0.0, 0.0, 0.00000005)
        # glClear(GL_COLOR_BUFFER_BIT)
        # self.exp_fbo.end()

        self.clear_fbo()

    def clear_fbo(self) -> None:
        LayerBase.setView(self.exp_fbo.width, self.exp_fbo.height)

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
        return self.con_fbo
