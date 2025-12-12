# Standard library imports
import numpy as np
import time

# Third-party imports
from OpenGL.GL import * # type: ignore
from pytweening import *    # type: ignore

# Local application imports
from modules.gl.Fbo import Fbo

from modules.pose.features import AggregationMethod
from modules.pose.Frame import Frame
from modules.gl.LayerBase import LayerBase, Rect
from modules.DataHub import DataHub, DataHubType, PoseDataHubTypes

from modules.utils.Smoothing import EMAFilterAttackRelease
from modules.utils.HotReloadMethods import HotReloadMethods

from modules.render.layers.generic.CentreCamLayer import CentreCamLayer
from modules.gl.Texture import Texture


# Shaders
from modules.gl.shaders.ApplyMask import ApplyMask as shader


def smoothstep(edge0: float, edge1: float, x: float) -> float:
    """Smooth Hermite interpolation between 0 and 1"""
    # Clamp x to range [edge0, edge1]
    t = max(0.0, min(1.0, (x - edge0) / (edge1 - edge0)))
    # Evaluate polynomial
    return t * t * (3.0 - 2.0 * t)

class MotionMultiply(LayerBase):
    _shader: shader = shader()

    def __init__(self, cam_id: int, data_hub: DataHub, data_type: PoseDataHubTypes, centre_cam: CentreCamLayer) -> None:
        self._cam_id: int = cam_id
        self._data_hub: DataHub = data_hub
        self._centre_cam: CentreCamLayer = centre_cam
        self._fbo: Fbo = Fbo()
        self._cam_fbo: Fbo = Fbo()
        self._mask_fbo: Fbo = Fbo()
        self._p_pose: Frame | None = None

        self.data_type: PoseDataHubTypes = data_type

        self.low_threshold: float = 0.0   # Turn off below this
        self.high_threshold: float = 0.1  # Turn on above this

        # Time-based fade parameters
        self.fade_in_duration: float = 1.0   # Time to fade in (seconds)
        self.fade_out_duration: float = 1.0  # Time to fade out (seconds)
        self._current_alpha: float = 0.0
        self._fade_progress: float = 0.0  # Progress through current fade (0-1)
        self._target_alpha: float = 0.0
        self._last_update_time: float | None = None

        self.filter = EMAFilterAttackRelease(freq=60, attack=0.1, release=0.5)

        # hot reloader
        self.hot_reloader = HotReloadMethods(self.__class__, True, True)

    @property
    def texture(self) -> Texture:
        return self._fbo

    @property
    def cam_texture(self) -> Texture:
        return self._cam_fbo

    @property
    def mask_texture(self) -> Texture:
        return self._mask_fbo

    def allocate(self, width: int, height: int, internal_format: int) -> None:
        self._fbo.allocate(width, height, internal_format)
        self._cam_fbo.allocate(width, height, internal_format)
        self._mask_fbo.allocate(width, height, GL_R32F)
        if not MotionMultiply._shader.allocated:
            MotionMultiply._shader.allocate(monitor_file=True)

    def deallocate(self) -> None:
        self._fbo.deallocate()
        self._cam_fbo.deallocate()
        self._mask_fbo.deallocate()
        if MotionMultiply._shader.allocated:
            MotionMultiply._shader.deallocate()

    def draw(self, rect: Rect) -> None:
        self._fbo.draw(rect.x, rect.y, rect.width, rect.height)
        # self._cam_fbo.draw(rect.x, rect.y, rect.width, rect.height)
        # self._mask_fbo.draw(rect.x, rect.y, rect.width, rect.height)

    def update(self) -> None:
        # reallocate shader if needed if hot-reloaded
        if not MotionMultiply._shader.allocated:
            MotionMultiply._shader.allocate(monitor_file=True)

        # Calculate delta time
        current_time = time.time()
        dt = current_time - self._last_update_time if self._last_update_time is not None else 0.0
        self._last_update_time = current_time

        pose: Frame | None = self._data_hub.get_item(DataHubType(self.data_type), self._cam_id)

        if pose is self._p_pose:
            return # no update needed
        self._p_pose = pose

        LayerBase.setView(self._cam_fbo.width, self._cam_fbo.height)
        glDisable(GL_BLEND)
        # glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        self._fbo.clear(0.0, 0.0, 0.0, 0.0)
        self._cam_fbo.clear(0.0, 0.0, 0.0, 0.0)
        self._mask_fbo.clear(0.0, 0.0, 0.0, 0.0)

        if pose is None:
            return

        cam = self._centre_cam.cam_texture
        mask = self._centre_cam.mask_texture

        motion = pose.angle_motion.aggregate(AggregationMethod.MAX)

        t = 0.05
        self.filter.setAttack(t)
        self.filter.setRelease(t / 2.0)

        m = motion * 1.5
        m = min(1.0, m)
        alpha = m # self.filter(m)
        alpha = easeInOutQuad(alpha)


        self._current_alpha = alpha

        self._cam_fbo.begin()
        glColor4f(1.0, 1.0, 1.0, self._current_alpha)
        cam.draw(0, 0, self._cam_fbo.width, self._cam_fbo.height)
        self._cam_fbo.end()

        self._mask_fbo.begin()
        glColor4f(self._current_alpha, 0.0, 0.0, 0.0)
        mask.draw(0, 0, self._mask_fbo.width, self._mask_fbo.height)
        self._mask_fbo.end()

        MotionMultiply._shader.use(self._fbo.fbo_id, cam.tex_id, mask.tex_id, self._current_alpha)

        glEnable(GL_BLEND)
        glColor4f(1.0, 1.0, 1.0, 1.0)
