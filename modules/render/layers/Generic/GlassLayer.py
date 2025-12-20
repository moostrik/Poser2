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

from modules.pose.features import AggregationMethod, AngleMotion, AngleLandmark
from modules.pose.Frame import Frame

from modules.render.layers.generic.SimilarityBlend import SimilarityBlend

from modules.utils.HotReloadMethods import HotReloadMethods



# Shaders
from modules.gl.shaders.Glass import Glass as shader


class GlassLayer(LayerBase):
    _shader: shader = shader()

    def __init__(self, cam_id: int, data_hub: DataHub, data_type: PoseDataHubTypes, sim_blend: SimilarityBlend) -> None:
        self._cam_id: int = cam_id
        self._data_hub: DataHub = data_hub
        self._sim_blend: SimilarityBlend = sim_blend
        self._fbo: Fbo = Fbo()
        self._p_pose: Frame | None = None

        self.data_type: PoseDataHubTypes = data_type

        # hot reloader
        self.hot_reloader = HotReloadMethods(self.__class__, True, True)

    @property
    def texture(self) -> Texture:
        return self._fbo

    def allocate(self, width: int, height: int, internal_format: int) -> None:
        self._fbo.allocate(width, height, internal_format)
        if not GlassLayer._shader.allocated:
            GlassLayer._shader.allocate(monitor_file=True)

    def deallocate(self) -> None:
        self._fbo.deallocate()
        if GlassLayer._shader.allocated:
            GlassLayer._shader.deallocate()

    def draw(self, rect: Rect) -> None:
        self._fbo.draw(rect.x, rect.y, rect.width, rect.height)

    def update(self) -> None:
        # reallocate shader if needed if hot-reloaded
        if not GlassLayer._shader.allocated:
            GlassLayer._shader.allocate(monitor_file=True)

        pose: Frame | None = self._data_hub.get_item(DataHubType(self.data_type), self._cam_id)

        if pose is self._p_pose:
            return # no update needed
        self._p_pose = pose

        LayerBase.setView(self._fbo.width, self._fbo.height)
        glDisable(GL_BLEND)
        # glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        self._fbo.clear(0.0, 0.0, 0.0, 0.0)

        if pose is None:
            return

        GlassLayer._shader.use(self._fbo.fbo_id, self._sim_blend.tex_id)
        glEnable(GL_BLEND)
        glColor4f(1.0, 1.0, 1.0, 1.0)
