
# Standard library imports
import numpy as np
import traceback
from dataclasses import dataclass
from enum import Enum
from time import time

# Third-party imports
import pytweening
from OpenGL.GL import * # type: ignore

# Local imports
from modules.gl.Fbo import Fbo, SwapFbo
from modules.gl.Image import Image
from modules.gl.shaders.HDT_Lines import HDT_Lines
from modules.pose.smooth.PoseSmoothData import PoseSmoothData, PoseJoint
from modules.render.renders.BaseRender import BaseRender, Rect

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

class LF(BaseRender):
    line_shader = HDT_Lines()

    def __init__(self, smooth_data: PoseSmoothData, cam_id: int) -> None:
        self.smooth_data: PoseSmoothData = smooth_data
        self.cam_id: int = cam_id
        self.fbo: Fbo = Fbo()
        self.left_fbo: Fbo = Fbo()
        self.rigt_fbo: Fbo = Fbo()


        self.pattern_time: float = 0.0

        self.interval: float = 1.0 / 60.0
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

    def clear(self) -> None:
        BaseRender.setView(self.fbo.width, self.fbo.height)
        self.fbo.begin()
        glClearColor(0.0, 0.0, 0.0, 1.0)
        glClear(GL_COLOR_BUFFER_BIT)
        self.fbo.end()

    def update(self) -> None:
        if not LF.line_shader.allocated:
            LF.line_shader.allocate(monitor_file=True)

        self.clear()

        # if not self.smooth_data.get_is_active(self.cam_id):
        #     return

        if self.cam_id != 0:
            return

        P: LineFieldsSettings = LineFieldsSettings()
        P.line_sharpness = 4
        P.line_speed = 0.2
        P.line_width = 1.1
        P.line_amount = 2.0

        left_elbow: float =     self.smooth_data.get_angle(self.cam_id, PoseJoint.left_elbow)
        left_shoulder: float =  self.smooth_data.get_angle(self.cam_id, PoseJoint.left_shoulder)
        rigt_elbow: float =     self.smooth_data.get_angle(self.cam_id, PoseJoint.right_elbow)
        rigt_shoulder: float =  self.smooth_data.get_angle(self.cam_id, PoseJoint.right_shoulder)
        head: float =           self.smooth_data.get_head_orientation(self.cam_id)
        motion: float =         self.smooth_data.get_motion(self.cam_id)
        age: float =            self.smooth_data.get_age(self.cam_id)
        anchor: float =         1.0 - self.smooth_data.rect_settings.centre_dest_y

        self.smooth_data.angle_settings.motion_threshold = 0.002
        # print(f"motion={motion:.2f}, age={age:.2f}")

        # return
        if not self.smooth_data.get_is_active(self.cam_id):
            left_elbow =    PI #np.sin(age) * PI
            left_shoulder = PI * 0.5
            rigt_elbow =    PI * 0.5
            rigt_shoulder = PI * 0.5
        else:
            # print(self.smooth_data.get_angular_motion(self.cam_id))
            pass

        if left_elbow is None:
            print("No left_elbow for cam", self.cam_id)
            left_elbow = 0.0
        if left_shoulder is None:
            print("No left_shoulder for cam", self.cam_id)
            left_shoulder = 0.0
        if rigt_elbow is None:
            print("No rigt_elbow for cam", self.cam_id)
            rigt_elbow = 0.0
        if rigt_shoulder is None:
            print("No rigt_shoulder for cam", self.cam_id)
            rigt_shoulder = 0.0

        try:
            left_count: float = 5 + P.line_amount   * LF.n_cos_inv(left_shoulder)
            rigt_count: float = 5 + P.line_amount   * LF.n_cos_inv(rigt_shoulder)
            left_width: float = P.line_width        * LF.n_cos(left_elbow) * LF.n_cos(left_shoulder) * 0.6 + 0.4
            rigt_width: float = P.line_width        * LF.n_cos(rigt_elbow) * LF.n_cos(rigt_shoulder) * 0.6 + 0.4
            left_sharp: float = P.line_sharpness    * LF.n_abs(left_elbow)
            rigt_sharp: float = P.line_sharpness    * LF.n_abs(rigt_elbow)
            left_speed: float = P.line_speed        * LF.n_cos_inv(left_elbow) + LF.n_cos_inv(left_shoulder)
            rigt_speed: float = P.line_speed        * LF.n_cos_inv(rigt_elbow) + LF.n_cos_inv(rigt_shoulder)
        except Exception as e:
            print(e)
            print(self.cam_id, self.smooth_data.get_is_active(self.cam_id), left_elbow, left_shoulder, rigt_elbow, rigt_shoulder)

        self.pattern_time  += self.interval * left_speed * rigt_speed
        motion_time: float = motion * 0.1 #+ self.pattern_time

        LF.line_shader.use(self.left_fbo.fbo_id,
                           ex_time=motion_time ,
                           phase=0.0,
                           anchor=anchor,
                           amount=left_count,
                           thickness=left_width,
                           sharpness=left_sharp)
        LF.line_shader.use(self.rigt_fbo.fbo_id,
                           ex_time=motion_time,
                           phase=0.5,
                           anchor=anchor,
                           amount=rigt_count,
                           thickness=rigt_width,
                           sharpness=rigt_sharp)

        BaseRender.setView(self.fbo.width, self.fbo.height)
        glEnable(GL_BLEND)
        glBlendFunc(GL_ONE, GL_ONE)
        self.fbo.begin()
        glColor3f(1.0, 0.0, 0.0)
        self.left_fbo.draw(0,0,self.fbo.width, self.fbo.height)
        glColor3f(0.0, 1.0, 1.0)
        self.rigt_fbo.draw(0,0,self.fbo.width, self.fbo.height)
        self.fbo.end()
        glColor3f(1.0, 1.0, 1.0)
        glDisable(GL_BLEND)

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