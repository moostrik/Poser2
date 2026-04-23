""" Draws a motion multiplied view based on the centred camera view."""

# Third-party imports
from OpenGL.GL import * # type: ignore
from pytweening import *    # type: ignore

# Local application imports
from modules.board import HasFrames

from modules.gl import Fbo, Texture, Style
from ..LayerBase import LayerBase, DataCache
from ...shaders import Tint as shader

from modules.pose.frame import Frame
from modules.pose.features import AngleMotion
from ...color_settings import ColorSettings

from modules.utils import HotReloadMethods


class MotionLayer(LayerBase):

    def __init__(self, cam_id: int, board: HasFrames, centre_mask: Texture, color_settings: ColorSettings, stage: int = 3) -> None:
        self._cam_id: int = cam_id
        self._board: HasFrames = board
        self._stage: int = stage
        self._centre_mask: Texture = centre_mask
        self._color_settings: ColorSettings = color_settings
        self._fbo: Fbo = Fbo()
        self._data_cache: DataCache[Frame]= DataCache[Frame]()

        self._shader: shader = shader()

        # hot reloader
        self.hot_reloader = HotReloadMethods(self.__class__, True, True)

    @property
    def texture(self) -> Texture:
        return self._fbo

    def allocate(self, width: int, height: int, internal_format: int) -> None:
        self._fbo.allocate(width, height, internal_format)
        self._shader.allocate()

    def deallocate(self) -> None:
        self._fbo.deallocate()
        self._shader.deallocate()

    def update(self) -> None:
        pose: Frame | None = self._board.get_frame(self._stage, self._cam_id)
        self._data_cache.update(pose)

        if self._data_cache.lost:
            self._fbo.clear()
            self._motion = 0.0

        if self._data_cache.idle or pose is None:
            return

        Style.push_style()
        Style.set_blend_mode(Style.BlendMode.DISABLED)

        mask = self._centre_mask  # MotionMultiply currently doesn't use mask separately
        # print("Motion value:", pose[AngleMotion].value)

        # Motion value is already normalized [0,1] and eased by pipeline
        motion: float = easeInOutSine(pose[AngleMotion].value)

        color = self._color_settings.track_colors[self._cam_id % len(self._color_settings.track_colors)]
        self._fbo.begin()
        self._shader.use(mask, color.r, color.g, color.b, motion)
        self._fbo.end()

        Style.pop_style()
