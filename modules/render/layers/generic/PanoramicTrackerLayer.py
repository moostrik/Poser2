# Standard library imports
from typing import Protocol

# Third-party imports
from OpenGL.GL import * # type: ignore

# Local application imports
from modules.gl import Fbo, Texture, Text
from ...shaders import DrawColoredRectangle
from ...color_settings import ColorSettings

from modules.tracker import Tracklet, TrackingStatus, PanoramicAnnotation, PanoramicTrackerSettings

from modules.board import HasObservations, HasTracklets
from ..LayerBase import LayerBase
from .PanoramicCameraLayer import PanoramaLayerSettings

from modules.utils import HotReloadMethods


class PanoramicTrackerBoard(HasTracklets, HasObservations, Protocol):
    """The slice of the board this layer needs: the fused primaries and the raw observations."""


class PanoramicTrackerLayer(LayerBase):
    """The tracker's world, drawn as a 360-degree strip: x is azimuth, 0 at the left edge.

    Draws **every observation**, not one per person, which is the point. A person standing on a
    seam is seen by two cameras, and each camera has its own opinion of their azimuth. The
    tracker fuses those into one `Azimuth` before anything else sees it, so a disagreement — the
    symptom of a wrong `fov`, `tilt` or `ring_radius` — is invisible everywhere else in the app.
    Here the two boxes sit side by side in the same world-id colour, and the gap between them
    *is* the error. The one the tracker picked is outlined.

    Same x mapping as the stitched camera panorama in the row above, so the two line up.
    """

    def __init__(self, board: PanoramicTrackerBoard, num_cams: int, color_settings: ColorSettings,
                 tracker: PanoramicTrackerSettings, settings: PanoramaLayerSettings) -> None:
        self.board: PanoramicTrackerBoard = board
        self.num_cams: int = num_cams
        self._color_settings: ColorSettings = color_settings
        self._tracker: PanoramicTrackerSettings = tracker
        # Shared with the stitched image above: the two are one display.
        self._settings: PanoramaLayerSettings = settings
        self.fbo: Fbo = Fbo()
        self._text: Text = Text()
        self._rect_shader: DrawColoredRectangle = DrawColoredRectangle()

        self.hot_reloader = HotReloadMethods(self.__class__, True, True)

    @property
    def texture(self) -> Texture:
        return self.fbo

    def allocate(self, width: int, height: int, internal_format: int) -> None:
        self.fbo.allocate(width, height, internal_format)
        self._text.allocate()
        self._rect_shader.allocate()

    def deallocate(self) -> None:
        self.fbo.deallocate()
        self._text.deallocate()
        self._rect_shader.deallocate()

    def update(self) -> None:
        self.fbo.begin()
        glClearColor(0.0, 0.0, 0.0, 1.0)
        glClear(GL_COLOR_BUFFER_BIT)

        # Which observation the tracker chose per world, so the winner can be marked. Keyed by
        # the host-owned observation id, the only thing that identifies one uniquely.
        chosen: list[Tracklet] = [t for t in self.board.get_tracklets().values() if t is not None]
        primaries: set[int] = {t.obs_id for t in chosen}
        # Every camera's own opinion, or only the fused one. The disagreement is the whole point,
        # so all of them by default; primaries-only is the fallback for reading a busy room.
        observations: list[Tracklet] = \
            self.board.get_observations() if self._settings.show_all_observations else chosen

        colors = self._color_settings.track_color_tuples

        for tracklet in observations:
            if tracklet is None or tracklet.is_removed:
                continue
            if not isinstance(tracklet.annotation, PanoramicAnnotation):
                continue

            world_angle: float = tracklet.annotation.world_angle
            local_angle: float = tracklet.annotation.local_angle
            distance: float = tracklet.annotation.distance
            is_primary: bool = tracklet.obs_id in primaries
            lost: bool = tracklet.status == TrackingStatus.LOST

            # A box of normalized width w spans w * fov degrees of the world, so as a fraction of
            # the 360-degree strip it is w * fov / 360. It is NOT w / num_cams: that would only
            # be right with no overlap between the cameras, and at fov 127 with four cameras it
            # draws every box 29 per cent too narrow.
            roi_width: float = tracklet.roi.width * self._tracker.fov / 360.0
            roi_height: float = tracklet.roi.height
            roi_x: float = world_angle / 360.0 - roi_width / 2.0
            roi_y: float = tracklet.roi.y

            r, g, b, a = colors[tracklet.id % len(colors)]
            # The world colour says who; the alpha says how much to trust it. A lost observation
            # is still anchoring, so it is drawn, but faintly; a loser at a seam is dimmed so the
            # primary reads as the one in charge.
            if lost:
                a *= 0.25
            elif not is_primary:
                a *= 0.5
            self._rect_shader.use(roi_x, roi_y, roi_width, roi_height, r, g, b, a)

            # The primary gets a thin outline rather than a fill change, so comparing the two
            # boxes' positions is not confused by comparing their brightness.
            if is_primary:
                edge: float = 2.0 / max(1, self.fbo.height)
                self._rect_shader.use(roi_x, roi_y, roi_width, edge, r, g, b, 1.0)
                self._rect_shader.use(roi_x, roi_y + roi_height - edge, roi_width, edge, r, g, b, 1.0)

            # A camera tick at the top of the box: which side this opinion came from.
            tick_w: float = roi_width / max(1, self.num_cams)
            tick_x: float = roi_x + tick_w * tracklet.cam_id
            self._rect_shader.use(tick_x, roi_y, tick_w, 4.0 / max(1, self.fbo.height), r, g, b, 1.0)

            fg = (1.0, 1.0, 1.0, 1.0)
            bg = (0.0, 0.0, 0.0, 0.6)
            text_x: float = (roi_x * self.fbo.width) + 9
            text_y: float = (roi_y * self.fbo.height) + 22
            # One value per line, each labelled. `az` is the room azimuth this box is drawn at,
            # `loc` the angle within this camera's own field, `dis` the estimated distance from
            # that camera. Two boxes of one person share `#id` and differ in `cam`.
            for line in (f'#{tracklet.id} cam{tracklet.cam_id}',
                         f'az {world_angle:.0f}',
                         f'loc {local_angle:.0f}',
                         f'dis {distance:.1f}m'):
                self._text.draw_box_text(text_x, text_y, line, fg, bg, self.fbo.width, self.fbo.height)
                text_y += 22
        self.fbo.end()
