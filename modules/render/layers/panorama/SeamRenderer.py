# Third-party imports
from OpenGL.GL import * # type: ignore

# Local application imports
from modules.tracker import PanoramicTrackerSettings, camera_local_to_azimuth, strip_spans, wrap180

from ...shaders import DrawColoredRectangle
from ..LayerBase import LayerBase
from .PanoramaLayerSettings import PanoramaLayerSettings, DEAD_ZONE_COLOR


class SeamRenderer(LayerBase):
    """The seam rules that live in **image space** — what has to be read against the picture.

    The split from `GridRenderer` is by coordinate system, not by subject. Anything expressible in
    the strip's own two axes — centre azimuth and centre elevation — is a lattice mark and lives
    there, where the degree labels can measure it: the sector boundaries, the camera axes, the
    overlap. What lands here instead is anything defined on a camera's **own** frame, because one
    image column has no single azimuth — it maps to a different bearing at every depth, so there is
    nothing on the grid to read it against.

    Today that is the **dead zone**: `seam.dead_zone` degrees in from each camera's field edges, two
    bands per camera, where *that* camera refuses to start a new person. Someone already tracked is
    still refreshed there and a re-acquisition is still allowed; only *arrivals* are refused, so
    that nobody is created twice on a seam.

    **Why a band here works where a line on the grid would not.** The rule reads the raw local angle
    — the bearing within that camera's frame, before the parallax re-projection — and the stitch
    places the picture through the same map at the same depth. So **band and pixels agree by
    construction at every depth**: a person whose pixels fall inside the red is a person that camera
    will not start, wherever they are standing. That is what makes it checkable against the image
    anywhere rather than only at the focus radius.

    **Its cost is the region where two bands overlap.** A person is born as long as **one** camera
    accepts them, so nobody can be born only where both refuse: at R 1.35 the two bands do overlap on
    the seam, and a person arriving exactly there is not picked up until they move. From about R 1.5
    they no longer do. Each band's *outer* edge sits exactly on its camera's field edge, so the pair
    also delimits where the two pictures reach — no separate overlap fill is needed to show that.

    How close two views must be to count as one person is `seam.link_angle`, a property of a *pair*
    rather than of a place, so it is drawn as a field around each observation instead.

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
        nothing instead of a band spanning almost the whole ring. `strip_spans` then splits the band
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
