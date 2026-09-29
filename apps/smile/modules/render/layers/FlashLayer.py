from modules.gl import Fbo, Image, Texture
from modules.gl.shaders.convenience.FlashIn import FlashIn

from .LayerBase import LayerBase


class FlashLayer(LayerBase):
    """The flash: the camera brightens, then fades to the logo (white)."""

    def __init__(self, camera: LayerBase, logo: Image) -> None:
        self._camera: LayerBase = camera
        self._logo: Image = logo
        self._fbo: Fbo = Fbo()
        self._flash_in: FlashIn = FlashIn()

    def allocate(self, width: int | None = None, height: int | None = None, internal_format: int | None = None) -> None:
        assert width and height and internal_format
        self._fbo.allocate(width, height, internal_format)
        self._flash_in.allocate()

    def deallocate(self) -> None:
        self._fbo.deallocate()
        self._flash_in.deallocate()

    def update(self) -> None:
        self._flash_in.reload()

    @property
    def texture(self) -> Texture:
        return self._fbo

    def render(self, flash: float) -> None:
        if not self._logo.allocated or not self._camera.texture.allocated:
            return
        self._fbo.begin()
        self._flash_in.use(self._camera.texture, self._logo, flash)
        self._fbo.end()
