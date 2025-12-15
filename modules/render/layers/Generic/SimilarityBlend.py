# Standard library imports
import numpy as np

# Third-party imports
from OpenGL.GL import * # type: ignore
from pytweening import *    # type: ignore

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
from modules.gl.shaders.TripleBlendColor import TripleBlendColor as shader


class SimilarityBlend(LayerBase):
    _mask_shader: ApplyMask = ApplyMask()
    _mask_multiply_shader: MaskMultiply = MaskMultiply()
    _blend_shader: shader = shader()

    def __init__(self, cam_id: int, data_hub: DataHub, data_type: PoseDataHubTypes, layers: dict[int, MotionMultiply]) -> None:
        self._cam_id: int = cam_id
        self._data_hub: DataHub = data_hub
        self._layers: dict[int, MotionMultiply] = layers
        self._fbo: Fbo = Fbo()

        self._cam_fbo: Fbo = Fbo()
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

        self._cam_fbo.allocate(width, height, internal_format)
        self._cam_other_1_fbo.allocate(width, height, internal_format)
        self._cam_other_2_fbo.allocate(width, height, internal_format)

        self._mask_fbo.allocate(width, height, GL_R32F)
        self._mask_other_1_fbo.allocate(width, height, GL_R32F)
        self._mask_other_2_fbo.allocate(width, height, GL_R32F)

        if not SimilarityBlend._blend_shader.allocated:
            SimilarityBlend._blend_shader.allocate(monitor_file=True)

    def deallocate(self) -> None:
        self._fbo.deallocate()

        self._cam_other_1_fbo.deallocate()
        self._cam_other_2_fbo.deallocate()
        self._mask_other_1_fbo.deallocate()
        self._mask_other_2_fbo.deallocate()

        if SimilarityBlend._blend_shader.allocated:
            SimilarityBlend._blend_shader.deallocate()

    def draw(self, rect: Rect) -> None:
        self._fbo.draw(rect.x, rect.y, rect.width, rect.height)

    def update(self) -> None:
        # reallocate shader if needed if hot-reloaded
        if not SimilarityBlend._blend_shader.allocated:
            SimilarityBlend._blend_shader.allocate(monitor_file=True)
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

        colors: list[tuple[float, float, float, float]] = [
            (1.0, 0.0, 0.0, 1.0),
            (0.0, 0.0, 1.0, 1.0),
            (0.0, 1.0, 0.0, 1.0)
        ]

        index = self._cam_id
        other_indices = [i for i in range(3) if i != self._cam_id]
        other_1_index, other_2_index = other_indices[0], other_indices[1]

        # print(f"{self._layers[index].motion:.2f}, {self._layers[other_1_index].motion:.2f}, {self._layers[other_2_index].motion:.2f}")


        similarities = np.nan_to_num(pose.similarity.values)
        threshold = 0.33
        similarities = np.clip((similarities - threshold) / (1.0 - threshold), 0.0, 1.0)
        exponent = 1.5
        similarities = np.power(similarities, exponent)



        cam = self._layers[index].cam_texture
        cam_other_1 = self._layers[other_1_index].cam_texture
        cam_other_2 = self._layers[other_2_index].cam_texture

        self._cam_fbo.clear(*colors[index])
        self._cam_other_1_fbo.clear(*colors[other_1_index])
        self._cam_other_2_fbo.clear(*colors[other_2_index])

        mask = self._layers[index].mask_texture
        mask_other_1 = self._layers[other_1_index].mask_texture
        mask_other_2 = self._layers[other_2_index].mask_texture



        alpha: float =      self._layers[index].motion
        alpha_1: float =    float(similarities[other_1_index])
        alpha_2: float =    float(similarities[other_2_index])

        SimilarityBlend._blend_shader.use(self._fbo.fbo_id,
                                          self._cam_fbo.tex_id,     self._cam_other_1_fbo.tex_id,   self._cam_other_2_fbo.tex_id,
                                          mask.tex_id,              mask_other_1.tex_id,            mask_other_2.tex_id,
                                          alpha,                    alpha_1,                        alpha_2,
                                          colors[self._cam_id],     colors[other_1_index],          colors[other_2_index]
                                          )


        glEnable(GL_BLEND)
        return

        glEnable(GL_BLEND)
        # set to add
        glBlendFunc(GL_ONE, GL_ONE)

        self._fbo.begin()
        glColor4f(*colors[self._cam_id], 1.0)
        mask.draw(0, 0, self._fbo.width, self._fbo.height)
        glColor4f(*colors[other_1_index], alpha_1)
        mask_other_1.draw(0, 0, self._fbo.width, self._fbo.height)
        glColor4f(*colors[other_2_index], alpha_2)
        mask_other_2.draw(0, 0, self._fbo.width, self._fbo.height)
        self._fbo.end()

        return


        self._mask_other_1_fbo.begin()
        glColor4f(alpha_1, 0.0, 0.0, 0.0)
        mask_other_1.draw(0, 0, self._fbo.width, self._fbo.height)
        self._mask_other_1_fbo.end()

        SimilarityBlend._mask_multiply_shader.use(
            self._mask_fbo.fbo_id, mask_other_1.tex_id, mask.tex_id)
