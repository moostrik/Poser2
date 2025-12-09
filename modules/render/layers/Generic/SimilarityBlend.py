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

from modules.render.layers.generic.CentreCamLayer import CentreCamLayer


# Shaders
from modules.gl.shaders.ApplyMask import ApplyMask
from modules.gl.shaders.TripleBlend import TripleBlend as shader


def smoothstep(edge0: float, edge1: float, x: float) -> float:
    """Smooth Hermite interpolation between 0 and 1"""
    # Clamp x to range [edge0, edge1]
    t = max(0.0, min(1.0, (x - edge0) / (edge1 - edge0)))
    # Evaluate polynomial
    return t * t * (3.0 - 2.0 * t)

class SimilarityBlend(LayerBase):
    _mask_shader: ApplyMask = ApplyMask()
    _shader = shader()

    def __init__(self, cam_id: int, data_hub: DataHub, data_type: PoseDataHubTypes, layers: dict[int, CentreCamLayer]) -> None:
        self._cam_id: int = cam_id
        self._data_hub: DataHub = data_hub
        self._layers: dict[int, CentreCamLayer] = layers
        self._fbo: Fbo = Fbo()
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
        if not SimilarityBlend._mask_shader.allocated:
            SimilarityBlend._mask_shader.allocate(monitor_file=True)

        pose: Frame | None = self._data_hub.get_item(DataHubType(self.data_type), self._cam_id)

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

        cam = self._layers[self._cam_id].cam_texture
        mask = self._layers[self._cam_id].mask_texture

        other_1 = self._layers[other_1_index].cam_texture
        other_2 = self._layers[other_2_index].cam_texture

        raw_motion = pose.angle_motion.aggregate(AggregationMethod.MAX)
        # motion *= 1.3
        # motion_exponent = 0.5
        # motion = pow(motion, motion_exponent)
        # motion = np.clip(motion, 0.0, 1.0)

        # Hysteresis: different thresholds for turning on vs off
        if raw_motion > self._motion_high_threshold:
            self._is_visible = True
        elif raw_motion < self._motion_low_threshold:
            self._is_visible = False
        # else: maintain current state (prevents flickering)

        # Apply smoothstep transition based on current state
        if self._is_visible:
            # Visible: smoothstep around LOW threshold (so it fades out smoothly when going down)
            motion = smoothstep(
                self._motion_low_threshold - self._smoothstep_margin,
                self._motion_low_threshold + self._smoothstep_margin,
                raw_motion
            )
        else:
            # Hidden: smoothstep around HIGH threshold (so it fades in smoothly when going up)
            motion = smoothstep(
                self._motion_high_threshold - self._smoothstep_margin,
                self._motion_high_threshold + self._smoothstep_margin,
                raw_motion
            )

        # SimilarityBlend._mask_shader.use(self._fbo.fbo_id, cam.tex_id, mask.tex_id, motion)

        self._fbo.begin()
        glColor4f(1.0, 1.0, 1.0, motion)
        cam.draw(0, 0, self._fbo.width, self._fbo.height)
        self._fbo.end()

        glColor4f(1.0, 1.0, 1.0, 1.0)

        # return

        similarity = np.nan_to_num(pose.similarity.values)
        threshold = 0.33
        similarity = np.clip((similarity - threshold) / (1.0 - threshold), 0.0, 1.0)
        exponent =1.5
        similarity = np.power(similarity, exponent)
        # similarity *= motion

        blend_1 = similarity[other_1_index]
        blend_2 = similarity[other_2_index]

        # blend_1 = pow(pose.similarity.get_value(other_1_index, fill=0.0), 2.0) * pow(motion, 1.0)
        # blend_2 = pow(pose.similarity.get_value(other_2_index, fill=0.0), 2.0) * pow(motion, 1.0)

        # print(f"SimilarityBlend Layer {self._cam_id}: blend_1={blend_1}, blend_2={blend_2}")


        # SimilarityBlend._shader.use(self._fbo.fbo_id, cam.tex_id, other_1.tex_id, other_2.tex_id, blend_1, blend_2)

        # self._fbo.begin()
        # glColor4f(1.0, 1.0, 1.0, motion)
        # self._fbo.draw(0, 0, self._fbo.width, self._fbo.height)
        # self._fbo.end()
        # glColor4f(1.0, 1.0, 1.0, 1.0)


