# Standard library imports
from enum import IntEnum

# Third-party imports
import numpy as np
from OpenGL.GL import GL_NEAREST, GL_R32F, GL_RGBA16F

# Local application imports
from modules.board import HasCameraDepth
from modules.camera import CameraFrame
from modules.gl import Fbo, Image, Texture
from modules.settings import BaseSettings, Field
from ..layer_base import LayerBase, DataCache
from ...shaders import DepthView, DepthWarp


class DepthDrawMode(IntEnum):
    GREY = 0            # near white, far black
    COLOR = 1           # the OAK's color map
    OVER_IMAGE = 2      # the color map over the camera image


class DepthSourceLayerSettings(BaseSettings):
    """How depth is shown; the range and the color map are depthai's `colorizeDepthFrame` defaults."""
    draw_mode: Field[DepthDrawMode] = Field(DepthDrawMode.COLOR, description="How depth is drawn")
    near:      Field[float]         = Field(0.3, min=0.1, max=12.0, step=0.1, description="Start of the color map (m)")
    far:       Field[float]         = Field(12.0, min=0.5, max=20.0, step=0.5, description="End of the color map (m)")


def depth_metres(depth: np.ndarray) -> np.ndarray:
    """uint16 millimetres (H, W) as float32 metres; 0 stays 0, no depth."""
    return depth.astype(np.float32) / 1000.0


class DepthSourceLayer(LayerBase):
    """The camera's depth on the view of the camera image, straightened and mirrored as that image
    is: `depth` in metres, `texture` as a picture (`DepthDrawMode`). Stays empty on a source without
    depth."""

    def __init__(self, cam_id: int, board: HasCameraDepth, cam_texture: Texture, settings: DepthSourceLayerSettings) -> None:
        self._cam_id: int = cam_id
        self._board: HasCameraDepth = board
        self._cam_texture: Texture = cam_texture
        self.settings: DepthSourceLayerSettings = settings
        self._source: Image = Image(interpolation=GL_NEAREST)   # an edge never reads as a distance in between
        self._depth: Fbo = Fbo(interpolation=GL_NEAREST)
        self._view: Fbo = Fbo()
        self._warp_shader: DepthWarp = DepthWarp()
        self._shader: DepthView = DepthView()
        self._data_cache: DataCache[CameraFrame] = DataCache[CameraFrame]()
        self._dirty: bool = False

    @property
    def texture(self) -> Texture:
        return self._view

    @property
    def depth(self) -> Texture:
        """Metres, 0 where there is no depth."""
        return self._depth

    @property
    def dirty(self) -> bool:
        return self._dirty

    def allocate(self, width: int | None = None, height: int | None = None, internal_format: int | None = None) -> None:
        self._warp_shader.allocate()
        self._shader.allocate()     # the textures on the first frame, at its size

    def deallocate(self) -> None:
        self._source.deallocate()
        self._depth.deallocate()
        self._view.deallocate()
        self._warp_shader.deallocate()
        self._shader.deallocate()

    def update(self) -> None:
        self._dirty = False
        frame: CameraFrame | None = self._board.get_camera_depth(self._cam_id)
        self._data_cache.set(frame)

        if self._data_cache.lost:
            self._view.clear()

        if self._data_cache.idle or frame is None:
            return

        self._source.set_image(depth_metres(frame.image))
        self._source.update()
        if not self._view.allocated:
            self._depth.allocate(frame.image.shape[1], frame.image.shape[0], GL_R32F)
            self._view.allocate(frame.image.shape[1], frame.image.shape[0], GL_RGBA16F)

        self._depth.begin()
        self._warp_shader.use(self._source, frame.homography, frame.flip_h)
        self._depth.end()

        self._view.clear()          # no depth is transparent, so it must not leave the last frame showing
        self._view.begin()
        self._shader.use(self._depth, self._cam_texture, self.settings.near, self.settings.far, self.settings.draw_mode)
        self._view.end()
        self._dirty = True
