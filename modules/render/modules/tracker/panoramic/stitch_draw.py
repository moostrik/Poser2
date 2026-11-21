# Local application imports
from modules.gl import Texture
from modules.tracker import PanoramicTrackerSettings

from ....layer_base import LayerBase
from .shaders import PanoramicStitch
from .panorama_compositor_settings import PanoramaCompositorSettings


class StitchDraw(LayerBase):
    """The cameras' images unwrapped into one 360-degree strip, x azimuth from 0 at the left edge, each camera drawn at the
    azimuth it claims and stitched for one assumed depth, `focus_radius`. What it shows about `fov` and `tilt`, and why
    one depth: apps/white_space/docs/CALIBRATION.md, *Reading the panorama*."""

    def __init__(self, cam_textures: list[Texture], tracker: PanoramicTrackerSettings,
                 settings: PanoramaCompositorSettings) -> None:
        self._cam_textures: list[Texture] = cam_textures
        self._tracker: PanoramicTrackerSettings = tracker
        self._settings: PanoramaCompositorSettings = settings
        self._stitch: PanoramicStitch = PanoramicStitch()

        # Handed down by the compositor, which owns the one copy of the strip's geometry.
        self._target_fov: float = 90.0
        self._row_model: tuple[float, float] = (0.5, 0.72)      # (horizon_row, focal_rows)
        self._elevation_window: tuple[float, float] = (0.0, 0.0)
        self._populated_band: tuple[float, float] = (0.0, 0.0)

    def set_geometry(self, target_fov: float, row_model: tuple[float, float],
                     elevation_window: tuple[float, float],
                     populated_band: tuple[float, float]) -> None:
        self._target_fov = target_fov
        self._row_model = row_model
        self._elevation_window = elevation_window
        self._populated_band = populated_band

    def allocate(self, width: int, height: int, internal_format: int) -> None:
        self._stitch.allocate()

    def deallocate(self) -> None:
        self._stitch.deallocate()

    def update(self) -> None:
        pass

    def draw(self) -> None:
        self._stitch.use(
            self._cam_textures,
            self._tracker.fov,
            self._row_model,
            self._target_fov,
            max(0.0, self._tracker.rig.camera_radius),
            self._settings.focus_radius,
            self._elevation_window,
            self._populated_band,
            int(self._settings.blend),
        )
