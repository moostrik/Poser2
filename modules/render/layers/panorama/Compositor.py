# Standard library imports
from typing import Protocol

# Third-party imports
from OpenGL.GL import * # type: ignore

# Local application imports
from modules.board import HasObservations, HasTracklets
from modules.gl import Fbo, Texture, clear_color
from modules.tracker import PanoramicTrackerSettings, Tracklet, elevation_window, populated_band
from modules.utils import HotReloadMethods

from ..LayerBase import LayerBase
from ...color_settings import ColorSettings
from .GridRenderer import GridRenderer
from .LabelRenderer import LabelRenderer
from .ObservationRenderer import ObservationRenderer
from .PanoramaLayerSettings import PanoramaLayerSettings, Part
from .SeamRenderer import SeamRenderer
from .StitchRenderer import StitchRenderer
from .marks import Mark, build_marks


class PanoramaBoard(HasTracklets, HasObservations, Protocol):
    """The slice of the board this needs: the fused primaries and the raw observations."""


class Compositor(LayerBase):
    """The 360-degree calibration display: one strip, one FBO, five renderers in a fixed order.

    x is azimuth, 0 at the left edge; y is elevation, both measured at the rig centre and both
    linear, so the grid is square and a degree is a degree either way.

    **Why the image and the data are one display.** The camera constants define the azimuth frame
    every other number in the installation is expressed in, and the distance model rides on top of
    them. Drawn apart, on two vertical scales, the two could not be compared; drawn together, a
    person's pixels and a person's numbers are read in the same place — and which of the two is
    wrong tells you whether to reach for `fov`/`tilt` or for `ring_radius`/`camera_height`.

    **This class owns the strip's geometry**, and hands it down: the elevation window is derived
    once here and passed to whichever renderer needs it, so nothing can drift. `aspect_ratio` is
    read by the render's row layout — change `tilt`, `fov` or `focus_diameter` and the row follows.

    Renderers own no FBO of their own (as in `cam/`): each draws into this one between `begin()` and
    `end()`, in the order below, and an unticked `Part` is skipped entirely rather than drawn and
    hidden.
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
        self._grid: GridRenderer = GridRenderer(tracker, settings)
        self._observations: ObservationRenderer = ObservationRenderer()
        self._labels: LabelRenderer = LabelRenderer()

        # Bottom to top. The image first because everything else is read against it; the seams under
        # the grid so the lattice stays legible over a band; the text last so nothing covers it.
        self._order: list[tuple[Part, LayerBase]] = [
            (Part.image,        self._stitch),
            (Part.seams,        self._seams),
            (Part.grid,         self._grid),
            (Part.observations, self._observations),
            (Part.labels,       self._labels),
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
    def vfov(self) -> float:
        """One camera's vertical field (degrees) — derived by the tracker from `fov`."""
        return max(1.0, self._tracker.parallax.vfov)

    @property
    def populated_band(self) -> tuple[float, float]:
        """The elevations the delivered frames actually carry, measured at the camera."""
        return populated_band(self.vfov, self._settings.tilt)

    @property
    def elevation_window(self) -> tuple[float, float]:
        """(top, bottom) elevation of the strip, measured at the rig centre."""
        return elevation_window(self.populated_band,
                                self._tracker.parallax.ring_radius,
                                max(1e-6, self._settings.focus_diameter / 2.0))

    @property
    def aspect_ratio(self) -> float:
        """Width:height of the strip, for the row that holds it.

        360 degrees of azimuth over however many degrees of elevation the window spans — square
        degrees, which is the right answer because both axes are re-projected to the rig centre.
        Nothing here is a preference.
        """
        top, bottom = self.elevation_window
        return 360.0 / max(1.0, top - bottom)

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

        self._stitch.set_geometry(self.target_fov, self.vfov, window, self.populated_band)
        self._grid.set_geometry(window)

        if Part.observations in parts or Part.labels in parts:
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
        """Build the marks once, for the two renderers that draw the same people."""
        chosen: list[Tracklet] = [t for t in self._board.get_tracklets().values() if t is not None]
        primaries: set[int] = {t.obs_id for t in chosen}
        # Every camera's own opinion, or only the fused one. The disagreement is the whole point, so
        # all of them by default; primaries-only is the fallback for reading a busy room.
        observations: list[Tracklet] = \
            self._board.get_observations() if self._settings.show_all_observations else chosen

        marks: list[Mark] = build_marks(
            observations, primaries,
            self._color_settings.track_color_tuples,
            self._tracker.fov, self.vfov,
            self._tracker.parallax.ring_radius,
            window,
        )
        self._observations.set_marks(marks)
        self._labels.set_marks(marks)
