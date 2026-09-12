# Third-party imports
from OpenGL.GL import * # type: ignore

# Local application imports
from modules.tracker import PanoramicTrackerSettings, camera_azimuth

from ...shaders import DrawColoredRectangle
from ..LayerBase import LayerBase
from .PanoramaLayerSettings import PanoramaLayerSettings, \
    AXIS_COLOR, BAND_BAR_COLOR, BAND_COLOR, SEAM_COLOR


class SeamRenderer(LayerBase):
    """Everything about where one camera ends and the next begins.

    The lines — each sector boundary in orange, each camera's optical axis in blue — and the two
    zones the tracker's fusion rules define, read straight off `camera.tracker.seam.angles`, which
    the tracker keeps up to date as READ degrees:

    - **reach** (`angles.reach`) — within this of a camera's field edge, an observation may be
      matched with a neighbouring camera's and fused into one person. It is *wider* than the
      overlap (`reach` is a ratio ≥ 1), so it crosses the seam.
    - **reject** (`angles.reject`) — the dead zone inside it, where no *new* person may be born.
      A person already tracked is still refreshed here; only arrivals are refused, so that nobody
      is created twice on a seam.

    Both are measured inward from a camera's own field edge, which is `fov_overlap` outside the
    seam, so the pair of bands a seam carries sits asymmetrically about it — that is the geometry,
    not a drawing error.

    Drawn beneath the grid so the lattice stays legible over a band, and faintly, so the image
    underneath still reads.
    """

    def __init__(self, num_cams: int, tracker: PanoramicTrackerSettings,
                 settings: PanoramaLayerSettings) -> None:
        self._num_cams: int = max(1, num_cams)
        self._tracker: PanoramicTrackerSettings = tracker
        self._settings: PanoramaLayerSettings = settings
        self._rect: DrawColoredRectangle = DrawColoredRectangle()
        self._height: int = 1
        self._width: int = 1

    @property
    def _target_fov(self) -> float:
        return 360.0 / self._num_cams

    def allocate(self, width: int, height: int, internal_format: int) -> None:
        self._rect.allocate()
        self._width = max(1, width)
        self._height = max(1, height)

    def deallocate(self) -> None:
        self._rect.deallocate()

    def update(self) -> None:
        pass

    def draw(self) -> None:
        angles = self._tracker.seam.angles
        # Widest first: reach is drawn under reject, so the dead zone reads as the darker core.
        self._bands(angles.reach)
        self._bands(angles.reject)

        px_x: float = 1.0 / self._width
        for cam_id in range(self._num_cams):
            self._vertical(camera_azimuth(cam_id, self._target_fov), 2.0 * px_x, AXIS_COLOR)
            self._vertical(self._target_fov * cam_id, 2.0 * px_x, SEAM_COLOR)

    def _bands(self, width_degrees: float) -> None:
        """One band inward from each end of every camera's field."""
        if width_degrees <= 0.0:
            return
        overlap: float = self._tracker.seam.angles.overlap
        for cam_id in range(self._num_cams):
            # A camera's field runs from `target_fov * cam_id - overlap` for `fov` degrees:
            # `Geometry._calc_world_angle` with local 0 and local `cam_fov`.
            start: float = self._target_fov * cam_id - overlap
            end: float = start + self._tracker.fov
            self._band(start, width_degrees)
            self._band(end - width_degrees, width_degrees)

    def _band(self, azimuth: float, width_degrees: float) -> None:
        """A faint full-height fill plus a solid bar along the bottom edge, wrapped at 360."""
        px_y: float = 1.0 / self._height
        bar_h: float = 3.0 * px_y
        left: float = azimuth % 360.0
        width: float = width_degrees / 360.0
        # Wrapping past the right edge: draw the remainder at the left, so a band on the azimuth-0
        # seam is not silently clipped.
        spans: list[tuple[float, float]] = [(left / 360.0, width)]
        if left / 360.0 + width > 1.0:
            spans = [(left / 360.0, 1.0 - left / 360.0),
                     (0.0, width - (1.0 - left / 360.0))]
        for x, w in spans:
            self._rect.use(x, 0.0, w, 1.0, *BAND_COLOR)
            self._rect.use(x, 1.0 - bar_h, w, bar_h, *BAND_BAR_COLOR)

    def _vertical(self, azimuth: float, width: float,
                  color: tuple[float, float, float, float]) -> None:
        self._rect.use((azimuth % 360.0) / 360.0, 0.0, width, 1.0, *color)
