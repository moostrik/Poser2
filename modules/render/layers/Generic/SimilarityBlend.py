# Standard library imports
import numpy as np

# Third-party imports
from OpenGL.GL import * # type: ignore

# Local application imports
from modules.gl.Fbo import Fbo
from modules.gl.Image import Image
from modules.gl.Text import draw_box_string, text_init

from modules.pose.Frame import Frame
from modules.gl.LayerBase import LayerBase, Rect
from modules.DataHub import DataHub, DataHubType, PoseDataHubTypes

from modules.utils.HotReloadMethods import HotReloadMethods

from modules.render.layers.Generic.CentreCamLayer import CentreCamLayer

# Shaders
from modules.gl.shaders.TripleBlend import TripleBlend as shader


class SimilarityBlend(LayerBase):
    _shader = shader()

    def __init__(self, cam_id: int, data_hub: DataHub, data_type: PoseDataHubTypes, layers: dict[int, CentreCamLayer]) -> None:
        self._cam_id: int = cam_id
        self._data_hub: DataHub = data_hub
        self._layers: dict[int, CentreCamLayer] = layers
        self._fbo: Fbo = Fbo()
        self._p_pose: Frame | None = None

        self.data_type: PoseDataHubTypes = data_type

        text_init()

        # hot reloader
        self.hot_reloader = HotReloadMethods(self.__class__, True, True)

    def allocate(self, width: int, height: int, internal_format: int) -> None:
        self._fbo.allocate(width, height, internal_format)
        if not SimilarityBlend._shader.allocated:
            SimilarityBlend._shader.allocate(monitor_file=True)

    def deallocate(self) -> None:
        self._fbo.deallocate()
        if SimilarityBlend._shader.allocated:
            SimilarityBlend._shader.deallocate()

    def draw(self, rect: Rect) -> None:
        self._fbo.draw(rect.x, rect.y, rect.width, rect.height)

    def update(self) -> None:
        # reallocate shader if needed if hot-reloaded
        if not SimilarityBlend._shader.allocated:
            SimilarityBlend._shader.allocate(monitor_file=True)

        pose: Frame = self._data_hub.get_item(DataHubType(self.data_type), self._cam_id)

        if pose is self._p_pose:
            return # no update needed
        self._p_pose = pose

        LayerBase.setView(self._fbo.width, self._fbo.height)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        self._fbo.clear(0.0, 0.0, 0.0, 0.0)

        if pose is None:
            return

        other_1_index = (self._cam_id + 1) % 3
        other_2_index = (self._cam_id + 2) % 3

        blend_1 = pose.similarity.get_value(other_1_index, fill=0.0)
        blend_2 = pose.similarity.get_value(other_2_index, fill=0.0)

        # print(f"SimilarityBlend Layer {self._cam_id}: blend_1={blend_1}, blend_2={blend_2}")

        tex_0 = self._layers[self._cam_id]._fbo.tex_id
        tex_1 = self._layers[other_1_index]._fbo.tex_id
        tex_2 = self._layers[other_2_index]._fbo.tex_id

        SimilarityBlend._shader.use(self._fbo.fbo_id, tex_0, tex_1, tex_2, blend_1, blend_2)
