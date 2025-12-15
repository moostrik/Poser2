# Standard library imports
import numpy as np
import time

# Third-party imports
from OpenGL.GL import * # type: ignore
from pytweening import *    # type: ignore

# Local application imports
from modules.DataHub import DataHub, DataHubType, PoseDataHubTypes

from modules.gl.Fbo import Fbo
from modules.gl.LayerBase import LayerBase, Rect
from modules.gl.Texture import Texture

from modules.pose.features import AggregationMethod
from modules.pose.Frame import Frame

from modules.render.layers.generic.CentreCamLayer import CentreCamLayer

from modules.utils.HotReloadMethods import HotReloadMethods



# Shaders
from modules.gl.shaders.ApplyMask import ApplyMask as shader


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
        self._motion: float = 0.0

        self.data_type: PoseDataHubTypes = data_type

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

    @property
    def motion(self) -> float:
        return self._motion

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
        # self._fbo.draw(rect.x, rect.y, rect.width, rect.height)
        # self._cam_fbo.draw(rect.x, rect.y, rect.width, rect.height)
        self._mask_fbo.draw(rect.x, rect.y, rect.width, rect.height)

    def update(self) -> None:
        # reallocate shader if needed if hot-reloaded
        if not MotionMultiply._shader.allocated:
            MotionMultiply._shader.allocate(monitor_file=True)

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
        self._motion = 0.0

        if pose is None:
            return

        cam = self._centre_cam.cam_texture
        mask = self._centre_cam.mask_texture



        motion: float = pose.angle_motion.aggregate(AggregationMethod.MAX)
        # motion = max(0.0, motion - 0.25)
        motion = min(1.0, motion * 1.5)
        motion = easeInOutSine(motion)
        self._motion = motion

        self._cam_fbo.begin()
        glColor4f(1.0, 1.0, 1.0, motion)
        cam.draw(0, 0, self._cam_fbo.width, self._cam_fbo.height)
        self._cam_fbo.end()

        self._mask_fbo.begin()
        glColor4f(motion, 0.0, 0.0, 0.0)
        mask.draw(0, 0, self._mask_fbo.width, self._mask_fbo.height)
        self._mask_fbo.end()

        MotionMultiply._shader.use(self._fbo.fbo_id, cam.tex_id, mask.tex_id, motion)
        glEnable(GL_BLEND)
        glColor4f(1.0, 1.0, 1.0, 1.0)
