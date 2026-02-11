""" Draws a motion multiplied view based on the centred camera view."""

# Standard library imports

# Third-party imports
from OpenGL.GL import * # type: ignore
from pytweening import *    # type: ignore

# Local application imports
from modules.DataHub import DataHub, DataHubType, PoseDataHubTypes

from modules.gl import Fbo, Texture, Style
from modules.render.layers.LayerBase import LayerBase, DataCache
from modules.render.shaders import Tint as shader

from modules.pose.Frame import Frame

from modules.utils.HotReloadMethods import HotReloadMethods


class MotionMultiply(LayerBase):

    def __init__(self, cam_id: int, data_hub: DataHub, data_type: PoseDataHubTypes, centre_mask: Texture) -> None:
        self._cam_id: int = cam_id
        self._data_hub: DataHub = data_hub
        self._centre_mask: Texture = centre_mask
        self._fbo: Fbo = Fbo()
        self._data_cache: DataCache[Frame]= DataCache[Frame]()

        self._shader: shader = shader()

        self.data_type: PoseDataHubTypes = data_type

        # hot reloader
        self.hot_reloader = HotReloadMethods(self.__class__, True, True)

    @property
    def texture(self) -> Texture:
        return self._fbo

    def allocate(self, width: int, height: int, internal_format: int) -> None:
        self._fbo.allocate(width, height, GL_R32F)
        self._shader.allocate()

    def deallocate(self) -> None:
        self._fbo.deallocate()
        self._shader.deallocate()

    def update(self) -> None:
        pose: Frame | None = self._data_hub.get_item(DataHubType(self.data_type), self._cam_id)
        self._data_cache.update(pose)

        if self._data_cache.lost:
            self._fbo.clear()
            self._motion = 0.0

        if self._data_cache.idle or pose is None:
            return

        Style.push_style()
        Style.set_blend_mode(Style.BlendMode.DISABLED)

        mask = self._centre_mask  # MotionMultiply currently doesn't use mask separately

        # Motion value is already normalized [0,1] and eased by pipeline
        motion: float = easeInOutSine(pose.angle_motion.value)

        self._fbo.begin()
        self._shader.use(mask, 1.0, 1.0, 1.0,motion)
        self._fbo.end()

        Style.pop_style()
