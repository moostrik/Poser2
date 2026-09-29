from modules.gl import Fbo, Image, Texture
from modules.gl.shaders import Hsv

from .LayerBase import LayerBase
from .passes import copy


class CameraLayer(LayerBase):
    """The live camera. Keeps its last frame until render() is called again (the capture freeze)."""

    def __init__(self, camera: Image) -> None:
        self._camera: Image = camera
        self._fbo: Fbo = Fbo()
        self._hsv: Hsv = Hsv()

    def allocate(self, width: int | None = None, height: int | None = None, internal_format: int | None = None) -> None:
        assert width and height and internal_format
        self._fbo.allocate(width, height, internal_format)
        self._hsv.allocate()

    def deallocate(self) -> None:
        self._fbo.deallocate()
        self._hsv.deallocate()

    def update(self) -> None:
        self._hsv.reload()

    @property
    def texture(self) -> Texture:
        return self._fbo

    def render(self, saturation: float | None = None) -> None:
        """Draw the newest camera frame: in colour, or through the saturation (0 is black and white)."""
        if not self._camera.allocated:
            return
        if saturation is None:
            copy(self._camera, self._fbo)
            return
        self._fbo.begin()
        self._hsv.use(self._camera, 1.0, saturation, 1.0)
        self._fbo.end()
