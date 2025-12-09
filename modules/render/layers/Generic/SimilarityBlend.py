# Standard library imports
import numpy as np

# Third-party imports
from OpenGL.GL import * # type: ignore

# Local application imports
from modules.gl.Fbo import Fbo

from modules.pose.features import AggregationMethod
from modules.pose.Frame import Frame
from modules.gl.LayerBase import LayerBase, Rect
from modules.DataHub import DataHub, DataHubType, PoseDataHubTypes

from modules.utils.HotReloadMethods import HotReloadMethods

from modules.render.layers.generic.MotionMultiply import MotionMultiply


# Shaders
from modules.gl.shaders.ApplyMask import ApplyMask
from modules.gl.shaders.MaskMultiply import MaskMultiply
from modules.gl.shaders.TripleBlend import TripleBlend as shader


class SimilarityBlend(LayerBase):
    _mask_shader: ApplyMask = ApplyMask()
    _mask_multiply_shader: MaskMultiply = MaskMultiply()
    _shader = shader()

    def __init__(self, cam_id: int, data_hub: DataHub, data_type: PoseDataHubTypes, layers: dict[int, MotionMultiply]) -> None:
        self._cam_id: int = cam_id
        self._data_hub: DataHub = data_hub
        self._layers: dict[int, MotionMultiply] = layers
        self._fbo: Fbo = Fbo()

        self._cam_other_1_fbo: Fbo = Fbo()
        self._cam_other_2_fbo: Fbo = Fbo()

        self._mask_fbo: Fbo = Fbo()
        self._mask_other_1_fbo: Fbo = Fbo()
        self._mask_other_2_fbo: Fbo = Fbo()

        self._p_pose: Frame | None = None

        self.data_type: PoseDataHubTypes = data_type

        # Hysteresis for motion visibility
        self._is_visible: bool = False
        self._motion_low_threshold: float = 0.1   # Turn off below this
        self._motion_high_threshold: float = 0.7  # Turn on above this
        self._smoothstep_margin: float = 0.15     # Smoothstep range around thresholds

        # hot reloader
        self.hot_reloader = HotReloadMethods(self.__class__, True, True)

    def allocate(self, width: int, height: int, internal_format: int) -> None:
        self._fbo.allocate(width, height, internal_format)

        self._cam_other_1_fbo.allocate(width, height, internal_format)
        self._cam_other_2_fbo.allocate(width, height, internal_format)

        self._mask_fbo.allocate(width, height, GL_R32F)
        self._mask_other_1_fbo.allocate(width, height, GL_R32F)
        self._mask_other_2_fbo.allocate(width, height, GL_R32F)

        if not SimilarityBlend._shader.allocated:
            SimilarityBlend._shader.allocate(monitor_file=True)

    def deallocate(self) -> None:
        self._fbo.deallocate()

        self._cam_other_1_fbo.deallocate()
        self._cam_other_2_fbo.deallocate()
        self._mask_other_1_fbo.deallocate()
        self._mask_other_2_fbo.deallocate()

        if SimilarityBlend._shader.allocated:
            SimilarityBlend._shader.deallocate()

    def draw(self, rect: Rect) -> None:
        self._mask_fbo.draw(rect.x, rect.y, rect.width, rect.height)

    def update(self) -> None:
        # reallocate shader if needed if hot-reloaded
        if not SimilarityBlend._shader.allocated:
            SimilarityBlend._shader.allocate(monitor_file=True)
        if not SimilarityBlend._mask_shader.allocated:
            SimilarityBlend._mask_shader.allocate(monitor_file=True)
        if not SimilarityBlend._mask_multiply_shader.allocated:
            SimilarityBlend._mask_multiply_shader.allocate(monitor_file=True)

        pose: Frame | None = self._data_hub.get_item(DataHubType(self.data_type), self._cam_id)

        if pose is self._p_pose:
            return # no update needed
        self._p_pose = pose

        LayerBase.setView(self._fbo.width, self._fbo.height)
        glDisable(GL_BLEND)
        self._fbo.clear(0.0, 0.0, 0.0, 0.0)

        if pose is None:
            return

        other_1_index = (self._cam_id + 1) % 3
        other_2_index = (self._cam_id + 2) % 3

        cam = self._layers[self._cam_id].cam_texture
        cam_other_1 = self._layers[other_1_index].cam_texture
        cam_other_2 = self._layers[other_2_index].cam_texture

        mask = self._layers[self._cam_id].mask_texture
        mask_other_1 = self._layers[other_1_index].mask_texture
        mask_other_2 = self._layers[other_2_index].mask_texture

        similarity = np.nan_to_num(pose.similarity.values)
        threshold = 0.33
        similarity = np.clip((similarity - threshold) / (1.0 - threshold), 0.0, 1.0)
        exponent = 1.5
        similarity = np.power(similarity, exponent)

        alpha_1 = similarity[other_1_index]
        alpha_2 = similarity[other_2_index]

        self._mask_other_1_fbo.begin()
        glColor4f(alpha_1, 0.0, 0.0, 0.0)
        mask_other_1.draw(0, 0, self._fbo.width, self._fbo.height)
        self._mask_other_1_fbo.end()

        SimilarityBlend._mask_multiply_shader.use(
            self._mask_fbo.fbo_id, mask_other_1.tex_id, mask.tex_id)



