
# Standard library imports
import traceback
from dataclasses import dataclass
import numpy as np
from time import time

# Third-party imports
import pytweening
from OpenGL.GL import *  # type: ignore

# Local imports
from modules.gl.Fbo import Fbo, SwapFbo
from modules.gl.Image import Image
from modules.gl.shaders.HDT_Lines import HDT_Lines
from modules.pose.smooth.PoseSmoothDataManager import PoseJoint, PoseSmoothDataManager, SymmetricJointType
from modules.gl.LayerBase import LayerBase, Rect
from modules.utils.HotReloadMethods import HotReloadMethods

PI: float = np.pi
TWOPI: float = 2 * np.pi
HALFPI: float = np.pi / 2

@dataclass
class LineFieldsSettings():
    line_sharpness: float = 1.5         # higher is sharper
    line_speed: float = 1.5             # higher is faster
    line_width: float = 0.1             # in normalized world width (0..1)
    line_amount: float = 20.0           # number of lines

class LF(LayerBase):
    line_shader = HDT_Lines()

    def __init__(self, smooth_data: PoseSmoothDataManager, cam_id: int) -> None:
        self.smooth_data: PoseSmoothDataManager = smooth_data
        self.cam_id: int = cam_id
        self.fbo: Fbo = Fbo()
        self.left_fbo: Fbo = Fbo()
        self.rigt_fbo: Fbo = Fbo()
        self.interval: float = 1.0 / 60.0
        self.pattern_time: float = 0.0

        self.hot_reloader = HotReloadMethods(self.__class__, True)

    def allocate(self, width: int, height: int, internal_format: int) -> None:
        self.fbo.allocate(width, height, internal_format)
        self.left_fbo.allocate(1, height, GL_RGBA32F)
        self.rigt_fbo.allocate(1, height, GL_RGBA32F)
        if not LF.line_shader.allocated:
            LF.line_shader.allocate(monitor_file=True)

    def deallocate(self) -> None:
        self.fbo.deallocate()
        self.left_fbo.deallocate()
        self.rigt_fbo.deallocate()
        if LF.line_shader.allocated:
            LF.line_shader.deallocate

    def draw(self, rect: Rect) -> None:
        self.fbo.draw(rect.x, rect.y, rect.width, rect.height)

    def update(self) -> None:
        if not LF.line_shader.allocated:
            LF.line_shader.allocate(monitor_file=True)

        # if not self.smooth_data.get_is_active(self.cam_id):
        #     self._clear()
        #     return

        # if self.cam_id != 0:
        #     return

        self.smooth_data.OneEuro_settings.min_cutoff = 0.1
        self.smooth_data.OneEuro_settings.beta = 0.3

        self.smooth_data.rect_settings.centre_dest_y = 0.25
        self.smooth_data.rect_settings.height_dest = 0.8

        P: LineFieldsSettings = LineFieldsSettings()
        P.line_sharpness = 4
        P.line_speed = 0.2
        P.line_width = 1.1
        P.line_amount = 5.0

        elbow_L: float  = self.smooth_data.get_angle(self.cam_id, PoseJoint.left_elbow)
        shldr_L: float  = self.smooth_data.get_angle(self.cam_id, PoseJoint.left_shoulder)
        elbow_R: float  = self.smooth_data.get_angle(self.cam_id, PoseJoint.right_elbow)
        shldr_R: float  = self.smooth_data.get_angle(self.cam_id, PoseJoint.right_shoulder)
        head: float     = self.smooth_data.get_head(self.cam_id)
        motion: float   = self.smooth_data.get_motion(self.cam_id)
        age: float      = self.smooth_data.get_age(self.cam_id)
        anchor: float   = 1.0 - self.smooth_data.rect_settings.centre_dest_y
        synchrony: float= self.smooth_data.get_mean_synchrony(self.cam_id)

        # print(f"motion={motion:.2f}, age={age:.2f}")

        if not self.smooth_data.get_is_active(self.cam_id):
        # if True:
            m = 0.1
            elbow_L = m*PI# * 0.5 #np.sin(age) * PI
            shldr_L = m*PI #* 0.5
            elbow_R = m*-PI # * 0.5
            shldr_R = m*-PI #* 0.5
            # motion = self.pattern_time

        left_count: float = 5 + P.line_amount   * LF.n_cos_inv(shldr_L)
        rigt_count: float = 5 + P.line_amount   * LF.n_cos_inv(shldr_R)
        left_width: float = P.line_width        * pytweening.easeInExpo(LF.n_cos(elbow_L) * LF.n_cos(shldr_L)) * 0.9 + 0.3
        rigt_width: float = P.line_width        * pytweening.easeInExpo(LF.n_cos(elbow_R) * LF.n_cos(shldr_R)) * 0.9 + 0.3
        left_sharp: float = P.line_sharpness    * LF.n_abs(elbow_L)
        rigt_sharp: float = P.line_sharpness    * LF.n_abs(elbow_R)
        left_speed: float = P.line_speed        * LF.n_cos_inv(elbow_L) + LF.n_cos_inv(shldr_L)
        rigt_speed: float = P.line_speed        * LF.n_cos_inv(elbow_R) + LF.n_cos_inv(shldr_R)

        self.pattern_time  += self.interval * left_speed * rigt_speed
        line_time: float = motion * 0.1 + self.pattern_time * 0.1
        left_strth: float = pytweening.easeInOutQuad(LF.n_cos(shldr_L))
        rigt_strth: float = pytweening.easeInOutQuad(LF.n_cos(shldr_R))
        mess: float = 0.0# (1.0 - synchrony) * 1
        p01: float = np.sin(motion * 0.1) * 0.5 + 1.0


        # print(self.smooth_data.get_angle(self.cam_id, PoseJoint.left_elbow), self.smooth_data.get_synchrony(self.cam_id, SymmetricJointType.elbow))
        # print(LF.n_cos(PI), pytweening.easeInExpo(LF.n_cos(PI) * LF.n_cos(PI)) * 0.6 + 0.4)

        LayerBase.setView(self.fbo.width, self.fbo.height)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

        LF.line_shader.use(self.left_fbo.fbo_id,
                           time=line_time,
                           phase=0.0 + (1.0 - synchrony),
                           anchor=anchor,
                           amount=left_count,
                           thickness=left_width,
                           sharpness=left_sharp,
                           stretch=left_strth,
                           mess=mess,
                           param01=p01)
        LF.line_shader.use(self.rigt_fbo.fbo_id,
                           time=line_time,
                           phase=0.5 + (1.0 - synchrony),
                           anchor=anchor,
                           amount=rigt_count,
                           thickness=rigt_width,
                           sharpness=rigt_sharp,
                           stretch=rigt_strth,
                           mess=mess,
                           param01=p01)

        self._render()

    def _render(self) -> None:
        LayerBase.setView(self.fbo.width, self.fbo.height)
        glBlendFunc(GL_ONE, GL_ONE)
        self.fbo.begin()
        glClearColor(0.0, 0.0, 0.0, 1.0)
        glClear(GL_COLOR_BUFFER_BIT)
        glColor3f(1.0, 0.0, 0.0)
        self.left_fbo.draw(0,0,self.fbo.width, self.fbo.height)
        glColor3f(0.0, 1.0, 1.0)
        self.rigt_fbo.draw(0,0,self.fbo.width, self.fbo.height)
        self.fbo.end()
        glColor3f(1.0, 1.0, 1.0)

    def _clear(self) -> None:
        LayerBase.setView(self.fbo.width, self.fbo.height)
        self.fbo.begin()
        glClearColor(0.0, 0.0, 0.0, 1.0)
        glClear(GL_COLOR_BUFFER_BIT)
        self.fbo.end()

    @staticmethod
    def n_cos(angle) -> float:
        return (np.cos(angle) + 1.0) / 2.0

    @staticmethod
    def n_sin(angle) -> float:
        return (np.sin(angle) + 1.0) / 2.0

    @staticmethod
    def n_abs(angle) -> float:
        return abs(angle) / PI

    @staticmethod
    def n_cos_inv(angle) -> float:
        return 1.0 - (np.cos(angle) + 1.0) / 2.0

    @staticmethod
    def n_sin_inv(angle) -> float:
        return 1.0 - (np.sin(angle) + 1.0) / 2.0

    @staticmethod
    def n_abs_inv(angle) -> float:
        return 1.0 - abs(angle) / PI