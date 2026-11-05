# Standard library imports
from enum import IntEnum
from typing import TYPE_CHECKING

# Third-party imports
import numpy as np
from OpenGL.GL import GL_CLAMP_TO_BORDER, GL_NEAREST, GL_R32F, GL_R32UI, GL_RGBA16F

# Local application imports
from modules.board import HasCaptures
from modules.camera import CameraFrame, DepthAlignment, aligned_depth_size, depth_projection
from modules.gl import Fbo, Image, Texture
from modules.settings import BaseSettings, Field
from ..layer_base import LayerBase, DataCache
from ...shaders import DepthAlign, DepthView

if TYPE_CHECKING:
    from modules.inference import Capture


class DepthDrawMode(IntEnum):
    GREY = 0            # near white, far black
    COLOR = 1           # the OAK's color map
    OVER_IMAGE = 2      # the color map over the camera image


class DepthSourceLayerSettings(BaseSettings):
    """The player's band of depth, which the mask cuts and the picture is colored over; the color map is depthai's `colorizeDepthFrame`."""
    draw_mode: Field[DepthDrawMode] = Field(DepthDrawMode.COLOR, description="How depth is drawn")
    near:      Field[float]         = Field(0.5, min=0.1, max=12.0, step=0.05, description="Nearest depth that is the player (m)")
    far:       Field[float]         = Field(4.0, min=0.1, max=12.0, step=0.05, description="Farthest depth that is the player (m)")


def depth_metres(depth: np.ndarray) -> np.ndarray:
    """uint16 millimetres (H, W) as float32 metres; 0 stays 0, no depth."""
    return depth.astype(np.float32) / 1000.0


class DepthSourceLayer(LayerBase):
    """The camera's depth on the view of the camera image, aligned to it (`DepthAlign`), straightened
    and mirrored as that image is: `depth` in metres, `texture` as a picture (`DepthDrawMode`). The
    depth of the newest capture, so it belongs with that capture's image and poses. Stays empty on a
    source without depth."""

    def __init__(self, cam_id: int, board: HasCaptures, cam_texture: Texture, settings: DepthSourceLayerSettings) -> None:
        self._cam_id: int = cam_id
        self._board: HasCaptures = board
        self._cam_texture: Texture = cam_texture
        self.settings: DepthSourceLayerSettings = settings
        self._source: Image = Image(interpolation=GL_NEAREST)   # an edge never reads as a distance in between
        self._nearest: Texture = Texture(interpolation=GL_NEAREST)
        self._depth: Texture = Texture(interpolation=GL_NEAREST, wrap=GL_CLAMP_TO_BORDER, border_color=(0.0, 0.0, 0.0, 0.0))     # beyond the image: no depth
        self._view: Fbo = Fbo()
        self._align_shader: DepthAlign = DepthAlign()
        self._shader: DepthView = DepthView()
        self._projection: tuple[DepthAlignment | None, np.ndarray, np.ndarray] | None = None   # an alignment and what it makes
        self._data_cache: DataCache[CameraFrame] = DataCache[CameraFrame]()
        self._dirty: bool = False
        self._timestamp: float | None = None     # the capture time of the depth frame it holds

    @property
    def texture(self) -> Texture:
        return self._view

    @property
    def timestamp(self) -> float | None:
        """The capture time of the depth frame it holds (`CameraFrame.timestamp`), None before the first."""
        return self._timestamp

    @property
    def depth(self) -> Texture:
        """Metres, 0 where there is no depth."""
        return self._depth

    @property
    def sample_size(self) -> tuple[float, float]:
        """The uv one depth sample covers on the camera image; (0, 0) before the first frame."""
        if not self._depth.allocated or self._depth.width <= 0 or self._depth.height <= 0:
            return (0.0, 0.0)
        return (1.0 / self._depth.width, 1.0 / self._depth.height)

    @property
    def dirty(self) -> bool:
        return self._dirty

    def allocate(self, width: int | None = None, height: int | None = None, internal_format: int | None = None) -> None:
        self._align_shader.allocate()
        self._shader.allocate()     # the textures on the first frame, at its size

    def deallocate(self) -> None:
        self._source.deallocate()
        self._nearest.deallocate()
        self._depth.deallocate()
        self._view.deallocate()
        self._align_shader.deallocate()
        self._shader.deallocate()

    def update(self) -> None:
        self._dirty = False
        capture: "Capture | None" = self._board.get_capture(self._cam_id)
        frame: CameraFrame | None = capture.depth if capture is not None else None
        self._data_cache.set(frame)

        if self._data_cache.lost:
            self._view.clear()
            self._timestamp = None

        if self._data_cache.idle or frame is None:
            return

        self._timestamp = frame.timestamp
        self._source.set_image(depth_metres(frame.image))
        self._source.update()
        size: tuple[int, int] = (frame.image.shape[1], frame.image.shape[0])
        if not self._view.allocated:
            width, height = aligned_depth_size(size, frame.alignment)
            self._nearest.allocate(width, height, GL_R32UI)
            DepthAlign.clear(self._nearest)
            self._depth.allocate(width, height, GL_R32F)
            self._view.allocate(width, height, GL_RGBA16F)

        if self._projection is None or self._projection[0] is not frame.alignment:
            self._projection = (frame.alignment, *depth_projection(frame.alignment, size))
        _, to_ray, to_uv = self._projection
        self._align_shader.use(self._source, self._nearest, self._depth, to_ray, to_uv, frame.homography, frame.flip_h)

        self._view.clear()          # no depth is transparent, so it must not leave the last frame showing
        self._view.begin()
        self._shader.use(self._depth, self._cam_texture, self.settings.near, self.settings.far, self.settings.draw_mode)
        self._view.end()
        self._dirty = True
