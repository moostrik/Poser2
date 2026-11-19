"""The inference frame's monitor: the frame's color image."""

from OpenGL.GL import GL_RGBA16F

from modules import render
from modules.gl import Fbo, Style, Texture, clear_color


class InferenceFrameMonitorLayer(render.LayerBase):
    """The frame's `image`, drawn again with each new frame."""

    def __init__(self, image: render.InferenceFrameImageLayer) -> None:
        self._image: render.InferenceFrameImageLayer = image
        self._fbo: Fbo = Fbo()
        self._drawn: float | None = None

    @property
    def texture(self) -> Texture:
        return self._fbo

    def allocate(self, width: int | None = None, height: int | None = None, internal_format: int | None = None) -> None:
        assert width is not None and height is not None
        self._fbo.allocate(width, height, GL_RGBA16F)
        self._drawn = None

    def deallocate(self) -> None:
        self._fbo.deallocate()

    def update(self) -> None:
        timestamp: float | None = self._image.timestamp
        if timestamp == self._drawn or not self._fbo.allocated or not self._image.texture.allocated:
            return
        self._drawn = timestamp
        Style.push_style()
        Style.set_blend_mode(Style.BlendMode.DISABLED)
        self._fbo.begin()
        clear_color()
        self._image.texture.draw()
        self._fbo.end()
        Style.pop_style()
