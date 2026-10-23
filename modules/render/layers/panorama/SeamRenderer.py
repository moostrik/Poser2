# Third-party imports
from OpenGL.GL import * # type: ignore

# Local application imports
from modules.tracker import PanoramicTrackerSettings, camera_local_to_azimuth, wrap180

from ...shaders import DrawColoredRectangle
from ..LayerBase import LayerBase
from .settings import PanoramaLayerSettings, DEAD_ZONE_COLOR
from .strip import strip_spans


class SeamRenderer(LayerBase):
    """The seam rules defined on a camera's own frame, read against the picture rather than the grid
    (an image column has no single azimuth; `GridRenderer` holds what does).

    Today that is the **dead zone**: `seam.dead_zone` degrees in from each camera's field edges, where
    that camera refuses to start a new person. The band goes through the same map at the same depth
    as the stitch, so band and pixels agree at every depth: a person whose pixels fall in the red is
    one that camera will not start. Where two bands overlap on a seam (close to the fixture) nobody
    can be born until they move. Each band's outer edge is its camera's field edge, so the pair also
    shows where the two pictures reach.

    Drawn faintly, beneath the grid.
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
        self._dead_zone_bands()

    def _dead_zone_bands(self) -> None:
        """`seam.dead_zone` in from both ends of every camera's own field — its own rule, so one
        band per edge rather than one per seam."""
        fov: float = self._tracker.fov
        width: float = min(self._tracker.seam.dead_zone, fov / 2.0)
        if width <= 0.0:
            return
        for cam_id in range(self._num_cams):
            self._band(self._azimuth(cam_id, 0.0), self._azimuth(cam_id, width), DEAD_ZONE_COLOR)
            self._band(self._azimuth(cam_id, fov - width), self._azimuth(cam_id, fov),
                       DEAD_ZONE_COLOR)

    def _band(self, left: float, right: float, fill: tuple[float, float, float, float]) -> None:
        """An azimuth range as a faint full-height fill.

        The width is folded into ±180 rather than taken modulo 360 so that a degenerate range draws
        nothing instead of a band spanning almost the whole turn. `strip_spans` then splits the band
        that straddles the strip's 0/360 join, so the one on the azimuth-0 seam is drawn whole
        rather than clipped.
        """
        span: float = wrap180(right - left)
        if span <= 0.0:
            return
        for x, w in strip_spans(left / 360.0, span / 360.0):
            self._rect.use(x, 0.0, w, 1.0, *fill)

    def _azimuth(self, cam_id: int, local: float) -> float:
        """A camera's own local angle to the strip x its pixels land on, at the focus depth — the
        stitch's own chain, which is what keeps a band on the columns it describes."""
        return camera_local_to_azimuth(local, cam_id, self._tracker.fov, self._target_fov,
                                       max(0.0, self._tracker.rig.camera_radius),
                                       self._settings.focus_radius)
