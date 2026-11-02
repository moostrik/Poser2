# Third-party imports
from OpenGL.GL import GL_R16F

# Local application imports
from modules.gl import Fbo, Style, Texture
from modules.settings import BaseSettings, Field
from ..layer_base import LayerBase
from .depth_source_layer import DepthSourceLayer
from ...shaders import DepthMask


class DepthMaskLayerSettings(BaseSettings):
    """The band of depth that is the player."""
    near: Field[float] = Field(0.5, min=0.0, max=12.0, step=0.05, description="Nearest depth kept as the player (m)")
    far:  Field[float] = Field(4.0, min=0.1, max=12.0, step=0.05, description="Farthest depth kept as the player (m)")
    edge: Field[float] = Field(0.1, min=0.0, max=1.0, step=0.01, description="Soft falloff beyond near and far (m)")


class DepthMaskLayer(LayerBase):
    """The player cut from the capture's depth: 1 where the depth lies from `near` to `far`, soft over
    `edge` beyond, 0 without depth. On the view of the camera image, at the depth's size, read
    linearly, so the depth's few rows soften instead of stepping. New with each depth frame."""

    def __init__(self, depth: DepthSourceLayer, settings: DepthMaskLayerSettings) -> None:
        self._depth: DepthSourceLayer = depth
        self.settings: DepthMaskLayerSettings = settings
        self._fbo: Fbo = Fbo()
        self._shader: DepthMask = DepthMask()
        self._dirty: bool = False

    @property
    def texture(self) -> Texture:
        return self._fbo

    @property
    def dirty(self) -> bool:
        return self._dirty

    def allocate(self, width: int | None = None, height: int | None = None, internal_format: int | None = None) -> None:
        self._shader.allocate()     # the FBO on the first depth, at its size

    def deallocate(self) -> None:
        self._fbo.deallocate()
        self._shader.deallocate()

    def update(self) -> None:
        self._dirty = False
        if not self._depth.dirty:
            return
        depth: Texture = self._depth.depth
        if not self._fbo.allocated:
            self._fbo.allocate(depth.width, depth.height, GL_R16F)

        Style.push_style()
        Style.set_blend_mode(Style.BlendMode.DISABLED)
        self._fbo.begin()
        self._shader.use(depth, self.settings.near, self.settings.far, self.settings.edge)
        self._fbo.end()
        Style.pop_style()
        self._dirty = True
