# TODO
# fix length calculation
# external osc
# monolithic shader / cleanup this class / better alpha
# start and stop event (from external rec event)

# nice to haves
# pose snapshots
# pose deviation form neutral (snapshot)
# pose deviation form other (snapshots)
# smoot filters numpy based
# lut color system

# Standard library imports
import traceback
from dataclasses import dataclass
import numpy as np
from time import time
import math

# Third-party imports
from pytweening import *    # type: ignore
from OpenGL.GL import *     # type: ignore

# Local imports
from modules.gl.Fbo import Fbo, SwapFbo
from modules.gl.Image import Image
from modules.gl.shaders.HDT_Lines import HDT_Lines
from modules.gl.shaders.HDT_LineBlend import HDT_LineBlend
from modules.data.depricated.RenderDataHub_old import RenderDataHub_Old
from modules.pose.features.Symmetry import Symmetry
from modules.pose.features.Angles import AngleLandmark
from modules.gl.LayerBase import LayerBase, Rect
from modules.utils.Smoothing import OneEuroFilterAngular
from modules.utils.HotReloadMethods import HotReloadMethods

PI: float = math.pi
TWOPI: float = 2 * math.pi
HALFPI: float = math.pi / 2


CAM_COLOR = np.array([
    (1.0, 0.0, 0.0, 1.0),  # 0
    (1.0, 1.0, 0.0, 1.0),  # 1
    (0.0, 1.0, 1.0, 1.0),  # 2
])

@dataclass
class LineFieldsSettings():
    line_sharpness: float = 1.5         # higher is sharper
    line_speed: float = 1.5             # higher is faster
    line_width: float = 0.1             # in normalized world width (0..1)
    line_amount: float = 20.0           # number of lines

class LF(LayerBase):
    line_shader = HDT_Lines()
    line_blend_shader = HDT_LineBlend()

    def __init__(self, smooth_data: RenderDataHub_Old, cam_fbos: dict[int, Fbo], cam_id: int) -> None:
        self.smooth_data: RenderDataHub_Old = smooth_data
        self.cam_fbos: dict[int, Fbo] = cam_fbos
        self.cam_id: int = cam_id
        self.fbo: Fbo = Fbo()
        self.white_line_fbo: Fbo = Fbo()
        self.left_line_fbo: Fbo = Fbo()
        self.rigt_line_fbo: Fbo = Fbo()
        self.this_cam_fbo: Fbo = Fbo()
        self.left_cam_fbo: Fbo = Fbo()
        self.rigt_cam_fbo: Fbo = Fbo()
        self.other_cam_fbo: Fbo = Fbo()
        self.interval: float = 1.0 / 60.0
        self.pattern_time: float = 0.0

        self.left_width_OneEuro = OneEuroFilterAngular(60, 1.1, 0.2)
        self.rigt_width_OneEuro = OneEuroFilterAngular(60, 1.1, 0.2)
        self.left_sharp_OneEuro = OneEuroFilterAngular(60, 1.1, 0.2)
        self.rigt_sharp_OneEuro = OneEuroFilterAngular(60, 1.1, 0.2)

        self.hot_reloader = HotReloadMethods(self.__class__, True)

    def allocate(self, width: int, height: int, internal_format: int) -> None:
        self.fbo.allocate(width, height, internal_format)
        self.white_line_fbo.allocate(1, height, GL_RGBA32F)
        self.white_line_fbo.begin()
        glClearColor(1.0, 1.0, 1.0, 1.0)
        glClear(GL_COLOR_BUFFER_BIT)
        self.white_line_fbo.end()

        self.left_line_fbo.allocate(1, height, GL_RGBA32F)
        self.rigt_line_fbo.allocate(1, height, GL_RGBA32F)
        self.this_cam_fbo.allocate(width, height, GL_RGBA32F)
        self.left_cam_fbo.allocate(width, height, GL_RGBA32F)
        self.rigt_cam_fbo.allocate(width, height, GL_RGBA32F)
        self.other_cam_fbo.allocate(width, height, GL_RGBA32F)
        if not LF.line_shader.allocated:
            LF.line_shader.allocate(monitor_file=True)
        if not LF.line_blend_shader.allocated:
            LF.line_blend_shader.allocate(monitor_file=True)

    def deallocate(self) -> None:
        self.fbo.deallocate()
        self.white_line_fbo.deallocate()
        self.left_line_fbo.deallocate()
        self.rigt_line_fbo.deallocate()
        self.this_cam_fbo.deallocate()
        self.left_cam_fbo.deallocate()
        self.rigt_cam_fbo.deallocate()
        self.other_cam_fbo.deallocate()
        if LF.line_shader.allocated:
            LF.line_shader.deallocate()
        if LF.line_blend_shader.allocated:
            LF.line_blend_shader.deallocate()

    def draw(self, rect: Rect) -> None:
        self.fbo.draw(rect.x, rect.y, rect.width, rect.height)

    def update(self) -> None:
        if not LF.line_shader.allocated:
            LF.line_shader.allocate(monitor_file=True)
        if not LF.line_blend_shader.allocated:
            LF.line_blend_shader.allocate(monitor_file=True)

        if not self.smooth_data.get_is_active(self.cam_id):
            self._clear(self.fbo, CAM_COLOR[self.cam_id])
            return

        # if self.cam_id != 1:
        #     return

        self.smooth_data.one_euro_settings.min_cutoff = 0.1
        self.smooth_data.one_euro_settings.beta = 1.0

        self.smooth_data.viewport_settings.centre_dest_y = 0.25
        self.smooth_data.viewport_settings.height_dest = 0.8

        self.smooth_data.angle_settings.motion_threshold = 0.003

        P: LineFieldsSettings = LineFieldsSettings()
        P.line_sharpness = 1
        P.line_speed = 0.2
        P.line_width = 0.8
        P.line_amount = 5.0

        elbow_L: float =    self.smooth_data.get_angles(self.cam_id).get(AngleLandmark.left_elbow)
        elbow_L_Vel: float= self.smooth_data.get_velocities(self.cam_id).get(AngleLandmark.left_elbow)
        shldr_L: float =    self.smooth_data.get_angles(self.cam_id).get(AngleLandmark.left_shoulder)
        shldr_L_Vel: float= self.smooth_data.get_velocities(self.cam_id).get(AngleLandmark.left_shoulder)
        elbow_R: float =    self.smooth_data.get_angles(self.cam_id).get(AngleLandmark.right_elbow)
        elbow_R_Vel: float= self.smooth_data.get_velocities(self.cam_id).get(AngleLandmark.right_elbow)
        shldr_R: float =    self.smooth_data.get_angles(self.cam_id).get(AngleLandmark.right_shoulder)
        shldr_R_Vel: float= self.smooth_data.get_velocities(self.cam_id).get(AngleLandmark.right_shoulder)
        head: float =       self.smooth_data.get_angles(self.cam_id).get(AngleLandmark.head)
        motion: float =     self.smooth_data.get_cumulative_motion(self.cam_id)
        age: float =        self.smooth_data.get_age(self.cam_id)
        anchor: float =     1.0 - self.smooth_data.viewport_settings.centre_dest_y
        symmetry: float =   self.smooth_data.get_symmetries(self.cam_id).geometric_mean()



        # print(f"motion={motion:.2f}, age={age:.2f}")

        if not self.smooth_data.get_is_active(self.cam_id):
        # if True:
            m = 1.0
            elbow_L = m*PI# * 0.5 #np.sin(age) * PI
            shldr_L = m*PI #* 0.5
            elbow_R = m*-PI # * 0.5
            shldr_R = m*-PI #* 0.5
            # motion = self.pattern_time
            age = 10.0

        left_count: float = 5 + P.line_amount   * LF.n_cos_inv(shldr_L)
        rigt_count: float = 5 + P.line_amount   * LF.n_cos_inv(shldr_R)

        self.rigt_width_OneEuro.setMinCutoff(1)
        self.rigt_width_OneEuro.setBeta(0.2)
        self.left_width_OneEuro(abs(shldr_L_Vel))
        self.rigt_width_OneEuro(abs(shldr_R_Vel))

        self.left_sharp_OneEuro.setMinCutoff(0.75)
        self.left_sharp_OneEuro.setBeta(0.0)
        self.left_sharp_OneEuro.filter((abs(shldr_L_Vel) + abs(elbow_L_Vel)) / 2)
        self.rigt_sharp_OneEuro.filter((abs(shldr_L_Vel) + abs(elbow_R_Vel)) / 2)

        left_width: float = 0.5
        rigt_width: float = 0.5
        left_width = P.line_width * easeInQuad(1.0 - LF.limb_up(elbow_L, shldr_L)) * 0.7 + 0.3 * P.line_width
        rigt_width = P.line_width * easeInQuad(1.0 - LF.limb_up(elbow_R, shldr_R)) * 0.7 + 0.3 * P.line_width
        left_width *= 1.0 - easeInSine(min(self.left_width_OneEuro.value * TWOPI, 1.0)) * 0.9
        rigt_width *= 1.0 - easeInSine(min(self.rigt_width_OneEuro.value * TWOPI, 1.0)) * 0.9

        left_sharp: float = 1.0
        rigt_sharp: float = 1.0
        left_sharp = easeOutQuart(LF.n_cos_inv(shldr_L)) * 1.5
        rigt_sharp = easeOutQuart(LF.n_cos_inv(shldr_R)) * 1.5
        left_sharp -= (left_sharp) * easeOutQuart(min(self.left_sharp_OneEuro.value * 12, 1.0))
        rigt_sharp -= (rigt_sharp) * easeOutQuart(min(self.rigt_sharp_OneEuro.value * 12, 1.0))

        # left_sharp = 1.0 -  pytweening.easeOutQuad(min(self.left_speed_OneEuro.smooth_value * TWOPI, 1.0))
        # rigt_sharp = 1.0 -  pytweening.easeOutQuad(min(self.rigt_speed_OneEuro.smooth_value * TWOPI, 1.0))

        # if self.cam_id == 2:
        #     print(left_sharp)
        left_speed: float = P.line_speed        * LF.n_cos_inv(elbow_L) + LF.n_cos_inv(shldr_L)
        rigt_speed: float = P.line_speed        * LF.n_cos_inv(elbow_R) + LF.n_cos_inv(shldr_R)

        self.pattern_time  += self.interval * left_speed * rigt_speed
        line_time: float = motion * 0.1 + self.pattern_time * 0.1
        left_strth: float = easeInOutQuad(LF.n_cos(shldr_L))
        rigt_strth: float = easeInOutQuad(LF.n_cos(shldr_R))
        mess: float = (1.0 - symmetry) * 1
        p01: float = math.sin(motion * 0.1) * 0.5 + 1.0

        # print(self.smooth_data.get_angle(self.cam_id, PoseJoint.left_elbow), self.smooth_data.get_synchrony(self.cam_id, SymmetricJointType.elbow))
        # print(LF.n_cos(PI), pytweening.easeInExpo(LF.n_cos(PI) * LF.n_cos(PI)) * 0.6 + 0.4)

        self._clear(self.left_line_fbo)
        self._clear(self.rigt_line_fbo)
        self._clear(self.this_cam_fbo)
        self._clear(self.left_cam_fbo)
        self._clear(self.rigt_cam_fbo)
        self._clear(self.other_cam_fbo)


        LayerBase.setView(self.fbo.width, self.fbo.height)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)


        LF.line_shader.use(self.left_line_fbo.fbo_id,
                           time=line_time,
                           phase=0.0,# + (1.0 - synchrony),
                           anchor=anchor,
                           amount=left_count,
                           thickness=left_width,
                           sharpness=left_sharp,
                           stretch=left_strth,
                           mess=mess,
                           param01=p01,
                           param05=1.0)

        LF.line_shader.use(self.rigt_line_fbo.fbo_id,
                           time=line_time,
                           phase=0.5,# + (1.0 - synchrony),
                           anchor=anchor,
                           amount=rigt_count,
                           thickness=rigt_width,
                           sharpness=rigt_sharp,
                           stretch=rigt_strth,
                           mess=mess,
                           param01=p01,
                           param05=1.0)

        # RENDER

        # if self.cam_id == 1:
        #     synchrony = self.smooth_data.get_motion_correlation(1, 0)
        #     if synchrony > 0.0:
        #         print (synchrony)

        fade_in_cam_time: float = 1.1
        fade_in_color_time: float = 5
        fade_cam = easeOutQuad(min(age, fade_in_cam_time) / fade_in_cam_time)
        fade_color = easeInOutQuad(min(max(age - 0., 0.0), fade_in_color_time) / fade_in_color_time)

        this_cam_id = self.cam_id
        other_cam_id_0 = (self.cam_id + 1) % 3
        other_cam_id_1 = (self.cam_id + 2) % 3


        cam_color = CAM_COLOR[this_cam_id]
        other_cam_color_0 = CAM_COLOR[other_cam_id_0]
        other_cam_color_1 = CAM_COLOR[other_cam_id_1]

        sync_0 = self.smooth_data.get_pose_correlation(this_cam_id, other_cam_id_0)
        sync_1 = self.smooth_data.get_pose_correlation(this_cam_id, other_cam_id_1)

        sync_0 = (min(max((sync_0 - 0.5) / 0.5, 0.0), 1.0))
        sync_1 = (min(max((sync_1 - 0.5) / 0.5, 0.0), 1.0))

        # if self.cam_id == 0:
        #     print(f"sync0={sync_0:.2f}, sync1={sync_1:.2f}")

        LF.line_blend_shader.use(self.this_cam_fbo.fbo_id,
                                 self.cam_fbos[this_cam_id].tex_id,
                                 self.white_line_fbo.tex_id,
                                 color=cam_color,
                                 visibility=fade_cam,
                                 param0=0.0,
                                 param1=1.0)

        LF.line_blend_shader.use(self.left_cam_fbo.fbo_id,
                                 self.cam_fbos[other_cam_id_0].tex_id,
                                 self.left_line_fbo.tex_id,
                                 color=other_cam_color_0,
                                 visibility=sync_0,
                                 param0=0.0,
                                 param1=fade_color)

        LF.line_blend_shader.use(self.rigt_cam_fbo.fbo_id,
                                 self.cam_fbos[other_cam_id_1].tex_id,
                                 self.rigt_line_fbo.tex_id,
                                 color=other_cam_color_1,
                                 visibility=sync_1,
                                 param0=0.0,
                                 param1=fade_color)

        LayerBase.setView(self.fbo.width, self.fbo.height)
        self.other_cam_fbo.begin()
        glClearColor(0.0, 0.0, 0.0, 0.0)
        glClear(GL_COLOR_BUFFER_BIT)

        glBlendFunc(GL_ONE, GL_ONE)
        glColor4f(0.75, 0.75, 0.75, 0.5)
        self.left_cam_fbo.draw(0,0,self.fbo.width, self.fbo.height)
        self.rigt_cam_fbo.draw(0,0,self.fbo.width, self.fbo.height)
        glColor4f(1.0, 1.0, 1.0, 1.0)
        self.other_cam_fbo.end()


        glEnable(GL_BLEND)
        # glBlendFunc(GL_ONE, GL_ONE)
        self.fbo.begin()

        glClearColor(*CAM_COLOR[this_cam_id])
        glClear(GL_COLOR_BUFFER_BIT)


        # glColor4f(1.0, 1.0, 1.0, 0.75)
        # glBlendFunc(GL_ONE, GL_ONE)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

        self.this_cam_fbo.draw(0,0,self.fbo.width, self.fbo.height)
        self.other_cam_fbo.draw(0,0,self.fbo.width, self.fbo.height)
        # glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        # self.left_cam_fbo.draw(0,0,self.fbo.width, self.fbo.height)
        # self.rigt_cam_fbo.draw(0,0,self.fbo.width, self.fbo.height)
        self.fbo.end()

    def _clear(self, fbo:Fbo, color: tuple[float, float, float, float] = (0.0,0.0,0.0,0.0)) -> None:
        LayerBase.setView(fbo.width, fbo.height)
        fbo.begin()
        glClearColor(*color)
        glClear(GL_COLOR_BUFFER_BIT)
        fbo.end()

    @staticmethod
    def n_cos(angle) -> float:
        return (math.cos(angle) + 1.0) / 2.0

    @staticmethod
    def n_sin(angle) -> float:
        return (math.sin(angle) + 1.0) / 2.0

    @staticmethod
    def n_abs(angle) -> float:
        return abs(angle) / PI

    @staticmethod
    def n_cos_inv(angle) -> float:
        return 1.0 - (math.cos(angle) + 1.0) / 2.0

    @staticmethod
    def n_sin_inv(angle) -> float:
        return 1.0 - (math.sin(angle) + 1.0) / 2.0

    @staticmethod
    def n_abs_inv(angle) -> float:
        return 1.0 - abs(angle) / PI

    @staticmethod
    def limb_up(angle0: float, angle1: float) -> float:
        return LF.n_cos(angle0) * LF.n_cos(angle1)