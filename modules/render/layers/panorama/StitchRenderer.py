# Third-party imports
from OpenGL.GL import * # type: ignore

# Local application imports
from modules.gl import Texture
from modules.tracker import PanoramicTrackerSettings

from ...shaders import PanoramicStitch
from ..LayerBase import LayerBase
from .PanoramaLayerSettings import PanoramaLayerSettings


class StitchRenderer(LayerBase):
    """The cameras' images unwrapped into one 360-degree strip: x is azimuth, 0 at the left edge.

    **What it is for.** `fov` and `tilt` define the azimuth frame every other number in the
    installation is expressed in, and nothing else in the app shows whether they are right — the
    tracker fuses both cameras' views of a seam person into one `Azimuth` before anything draws it,
    destroying the disagreement that would reveal the error. Here the two cameras' pixels are drawn
    on top of each other at the azimuth each one claims. Right geometry: content in an overlap
    coincides. Wrong geometry: it ghosts, and *what* ghosts says *which* number is wrong.

    **Depth.** The image is stitched for one assumed depth, `focus_diameter`, because a camera
    0.36 m off centre genuinely sees a different bearing than its neighbour and only the distance
    to the subject resolves it. Nothing about a person feeds this — no box, no pose, no estimate —
    so nothing can fool it: the ghost at other depths is a known, bounded residual rather than a
    symptom.

    Owns no FBO: the compositor's is bound when `draw()` is called.
    """

    def __init__(self, cam_textures: list[Texture], tracker: PanoramicTrackerSettings,
                 settings: PanoramaLayerSettings) -> None:
        self._cam_textures: list[Texture] = cam_textures
        self._tracker: PanoramicTrackerSettings = tracker
        self._settings: PanoramaLayerSettings = settings
        self._stitch: PanoramicStitch = PanoramicStitch()

        # Handed down by the compositor, which owns the one copy of the strip's geometry.
        self._target_fov: float = 90.0
        self._vfov: float = 79.4
        self._elevation_window: tuple[float, float] = (0.0, 0.0)
        self._populated_band: tuple[float, float] = (0.0, 0.0)

    def set_geometry(self, target_fov: float, vfov: float,
                     elevation_window: tuple[float, float],
                     populated_band: tuple[float, float]) -> None:
        self._target_fov = target_fov
        self._vfov = vfov
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
            self._vfov,
            self._target_fov,
            self._tracker.parallax.ring_radius,
            self._settings.focus_diameter,
            self._elevation_window,
            self._populated_band,
            int(self._settings.blend),
        )
