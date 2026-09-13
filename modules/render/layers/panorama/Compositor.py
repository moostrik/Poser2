# Standard library imports
import time
from typing import Protocol

# Third-party imports
from OpenGL.GL import * # type: ignore

# Local application imports
from modules.board import HasObservations, HasTracklets
from modules.gl import Fbo, Texture, clear_color
from modules.tracker import PanoramicTrackerSettings, Tracklet, row_model
from modules.utils import HotReloadMethods

from ..LayerBase import LayerBase
from ...color_settings import ColorSettings
from .GridRenderer import GridRenderer
from .LabelRenderer import LabelRenderer
from .MarkRenderer import MarkRenderer
from .SeamRenderer import SeamRenderer
from .StitchRenderer import StitchRenderer
from .marks import Mark, MarkContext, build_marks
from .settings import PanoramaLayerSettings, Part, REJECTED_COLOR
from .strip import elevation_window, strip_aspect_ratio


class PanoramaBoard(HasTracklets, HasObservations, Protocol):
    """The slice of the board this needs: the fused primaries and the raw observations."""


class Compositor(LayerBase):
    """The 360-degree calibration display: one strip, one FBO, five renderers in a fixed order.

    x is centre azimuth, linear in degrees, 0 at the left edge; y is the tangent of centre elevation
    (`strip.strip_y`), a photograph's vertical like the frames'. The image and the tracker's data are
    drawn on the same axes so a person's pixels and numbers are read in the same place — and which
    is wrong says whether to reach for `fov`/`tilt` or the `rig` metres.

    **This class owns the strip's geometry** and hands it down, so nothing can drift. `aspect_ratio`
    is read by the render's row layout.

    **What varies between the renderers is the depth, not the space.** Two things on the strip are
    comparable only if they also share a depth:

    | drawn thing                        | x              | y                  | depth |
    |------------------------------------|----------------|--------------------|-------|
    | image (`StitchRenderer`)           | centre azimuth | centre elevation   | `focus_radius` |
    | dead zone (`SeamRenderer`)         | centre azimuth | — full height      | `focus_radius` |
    | lattice, seam + axis lines (`Grid`)| centre azimuth | —                  | **exact** |
    | overlap verticals (`Grid`)         | centre azimuth | —                  | `parallax_radius` |
    | horizon, zone band (`Grid`)        | —              | centre elevation   | **exact** |
    | mark: line, tick, field (`Mark`)   | centre azimuth | centre elevation   | x: `parallax_radius`, y: **the person's own distance** |
    | label (`LabelRenderer`)            | centre azimuth | pixel lane by id   | `parallax_radius` |

    The picture's depth so a band lands on its pixels; the tracker's depth so marks and the overlap
    verticals agree with the azimuth it emits; the person's own distance for a mark's rows (`marks`).
    Exact means no depth enters: a seam is a bearing, a floor circle a fixed depression.

    Renderers own no FBO: each draws into this one between `begin()` and `end()`, in order, and an
    unticked `Part` is skipped entirely.
    """

    def __init__(self, board: PanoramaBoard, cam_textures: list[Texture], num_cams: int,
                 color_settings: ColorSettings, tracker: PanoramicTrackerSettings,
                 settings: PanoramaLayerSettings) -> None:
        self._board: PanoramaBoard = board
        self.num_cams: int = max(1, num_cams)
        self._color_settings: ColorSettings = color_settings
        self._tracker: PanoramicTrackerSettings = tracker
        self._settings: PanoramaLayerSettings = settings

        self._fbo: Fbo = Fbo()

        self._stitch: StitchRenderer = StitchRenderer(cam_textures, tracker, settings)
        self._seams: SeamRenderer = SeamRenderer(self.num_cams, tracker, settings)
        self._grid: GridRenderer = GridRenderer(self.num_cams, tracker, settings)
        self._marks: MarkRenderer = MarkRenderer()
        self._labels: LabelRenderer = LabelRenderer()

        # Bottom to top. The image first because everything else is read against it; the seams
        # under the grid so the lattice stays legible over a band; the labels last so nothing
        # covers them.
        self._order: list[tuple[Part, LayerBase]] = [
            (Part.image,  self._stitch),
            (Part.seams,  self._seams),
            (Part.grid,   self._grid),
            (Part.marks,  self._marks),
            (Part.labels, self._labels),
        ]

        self.hot_reloader = HotReloadMethods(self.__class__, True, True)

    # -- geometry ------------------------------------------------------------

    @property
    def texture(self) -> Texture:
        return self._fbo.texture

    @property
    def target_fov(self) -> float:
        """The sector one camera owns — the tracker's own `360 / num_cameras`."""
        return 360.0 / self.num_cams

    @property
    def camera_radius(self) -> float:
        """`rig.camera_radius`, clamped. Every setting, every triangle below and the shader's
        `cameraRadius` are all radii from the fixture axis, so nothing is converted here."""
        return max(0.0, self._tracker.rig.camera_radius)

    @property
    def row_model(self) -> tuple[float, float]:
        """(horizon_row, focal_rows): the delivered frame's rows, rebuilt from the two edge angles
        the tracker publishes. Exact — the angles carry the whole row model — so the stitch and the
        marks convert rows while the panel only ever shows degrees. Rows are tangents of elevation:
        see `projection.row_from_elevation`."""
        horizon_row, focal_rows = row_model(*self.populated_band)
        return (horizon_row, max(1e-6, focal_rows))

    @property
    def populated_band(self) -> tuple[float, float]:
        """The angles the delivered frames carry, measured at the camera: the window's bottom and
        top rows, as the tracker published them."""
        p = self._tracker.rig
        return (p.angle_bottom, p.angle_top)

    @property
    def elevation_window(self) -> tuple[float, float]:
        """(top, bottom) elevation of the strip, measured at the rig centre."""
        return elevation_window(self.populated_band, self.camera_radius,
                                max(1e-6, self._settings.focus_radius))

    @property
    def aspect_ratio(self) -> float:
        """Width:height of the strip, for the row that holds it.

        360 degrees of azimuth at the same focal as the tangent rows: a degree at the horizon is
        the same size either way, and the window's tangent span is the height. Nothing here is a
        preference — `tilt`, `fov`, `frame_height` and `focus_radius` all move it.
        """
        return strip_aspect_ratio(self.elevation_window)

    # -- lifecycle -----------------------------------------------------------

    def allocate(self, width: int, height: int, internal_format: int) -> None:
        self._fbo.allocate(width, height, internal_format)
        for _, renderer in self._order:
            renderer.allocate(width, height, internal_format)

    def deallocate(self) -> None:
        self._fbo.deallocate()
        for _, renderer in self._order:
            renderer.deallocate()

    def update(self) -> None:
        if not self._fbo.allocated:
            return

        parts: list[Part] = self._settings.parts
        window: tuple[float, float] = self.elevation_window

        self._stitch.set_geometry(self.target_fov, self.row_model, window, self.populated_band)
        self._grid.set_geometry(window)

        if Part.marks in parts or Part.labels in parts:
            self._set_marks(window)

        self._fbo.begin()
        clear_color(0.0, 0.0, 0.0, 1.0)
        for part, renderer in self._order:
            if part not in parts:
                continue
            renderer.update()
            renderer.draw()
        self._fbo.end()

    def _set_marks(self, window: tuple[float, float]) -> None:
        """Build the marks once, for the mark and label renderers."""
        chosen: list[Tracklet] = [t for t in self._board.get_tracklets().values() if t is not None]
        primaries: set[int] = {t.obs_id for t in chosen}
        # Every camera's own view, or only the primary. The disagreement is the whole point, so all
        # of them by default; primaries-only is the fallback for reading a busy room.
        observations: list[Tracklet] = \
            self._board.get_observations() if self._settings.show_all_observations else chosen

        context: MarkContext = MarkContext(
            cam_fov=self._tracker.fov,
            target_fov=self.target_fov,
            camera_radius=self.camera_radius,
            parallax_radius=self._tracker.rig.parallax_radius,
            camera_height=self._tracker.rig.camera_height,
            row_model=self.row_model,
            elevation_window=window,
            link_angle=self._tracker.seam.link_angle,
            reacquire_angle=self._tracker.reacquire_angle,
            zone_max_radius=self._tracker.rig.zone_max_radius,
            lost_timeout=self._tracker.lost_timeout,
            now=time.time(),
        )
        marks: list[Mark] = build_marks(
            observations, primaries, self._color_settings.track_color_tuples, REJECTED_COLOR, context)
        self._marks.set_marks(marks)
        self._labels.set_marks(marks)
