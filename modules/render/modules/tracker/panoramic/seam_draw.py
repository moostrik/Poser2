# Local application imports
from modules.tracker import PanoramicTrackerSettings, camera_local_to_azimuth, wrap180

from modules.gl import shaders
from modules.utils import Color, Rect
from ....layer_base import LayerBase
from .panorama_compositor_settings import PanoramaCompositorSettings, DEAD_ZONE_COLOR
from .strip import strip_spans


class SeamDraw(LayerBase):
    """The seam rules defined on a camera's own frame, drawn against the picture: the dead zone, `seam.dead_zone` degrees in
    from each camera's field edges, as a faint band through the same map and depth as the stitch, beneath the grid. How to
    read it: apps/white_space/docs/CALIBRATION.md, *Two axes, and the depth that varies*."""

    def __init__(self, num_cams: int, target_fov: float, tracker: PanoramicTrackerSettings,
                 settings: PanoramaCompositorSettings) -> None:
        self._num_cams: int = max(1, num_cams)
        self._target_fov: float = target_fov        # the sector one camera owns
        self._tracker: PanoramicTrackerSettings = tracker
        self._settings: PanoramaCompositorSettings = settings
        self._rect: shaders.FillRect = shaders.FillRect()
        self._height: int = 1
        self._width: int = 1

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

    def _band(self, left: float, right: float, fill: Color) -> None:
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
            self._rect.use(Rect(x, 0.0, w, 1.0), fill)

    def _azimuth(self, cam_id: int, local: float) -> float:
        """A camera's own local angle to the strip x its pixels land on, at the focus depth — the
        stitch's own chain, which is what keeps a band on the columns it describes."""
        return camera_local_to_azimuth(local, cam_id, self._tracker.fov, self._target_fov,
                                       max(0.0, self._tracker.rig.camera_radius),
                                       self._settings.focus_radius)
