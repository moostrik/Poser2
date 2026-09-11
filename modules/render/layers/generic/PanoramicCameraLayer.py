# Standard library imports
from enum import IntEnum, auto

# Third-party imports
from OpenGL.GL import * # type: ignore

# Local application imports
from modules.gl import Fbo, Texture, Text
from modules.settings import BaseSettings, Field, Widget
from modules.tracker import PanoramicTrackerSettings, camera_azimuth
from modules.utils import HotReloadMethods

from ...shaders import DrawColoredRectangle, PanoramicStitch
from ..LayerBase import LayerBase


class PanoramaBlend(IntEnum):
    """How the two cameras' pixels combine where their fields overlap."""
    MAX     = 0   # the brighter of the two — a ghost reads as a doubled bright edge
    AVERAGE = auto()  # both at half weight — a ghost reads as a soft double image


class PanoramaLayerSettings(BaseSettings):
    """The stitched 360-degree calibration view, and the observation strip under it.

    Both halves of one display, so they share one settings group: the image above says whether
    the camera constants (`fov`, `tilt`) are right, the boxes below say whether the distance
    model is.
    """
    enabled: Field[bool] = Field(False, widget=Widget.switch,
                                 description="Replace the per-camera row with the stitched 360° panorama. A calibration view, not a show view — the window layout reflows when it is switched.")
    blend: Field[PanoramaBlend] = Field(PanoramaBlend.MAX,
                                        description="How overlapping cameras combine. MAX keeps the brighter pixel, so a misalignment shows as a doubled bright edge; AVERAGE shows it as a soft double image.")
    focus_diameter: Field[float] = Field(4.5, min=1.0, max=12.0, step=0.5,
                                         description="Play-zone diameter (m) the image is aligned for. Cameras 0.36 m apart genuinely disagree about where things are, by more the nearer they are, so an image can only be stitched for one depth. The middle of the play zone: exact there, ±4° at Ø3 and Ø7.")
    grid: Field[bool] = Field(True, widget=Widget.switch,
                              description="Draw the azimuth and elevation grid, with the sector seams and camera axes picked out.")
    grid_degrees: Field[float] = Field(10.0, min=1.0, max=90.0, step=1.0,
                                       description="Grid spacing (°), the same on both axes. Horizontal lines are what let the overlap be compared at head height against knee height — the tilt signature.")
    strip_aspect: Field[float] = Field(6.5, min=3.0, max=16.0, step=0.5,
                                       description="Width:height of the stitched row. 4.5 is square degrees — the whole 800-row frame at its true scale — but a padded clip wastes part of that height and the row is then taller than it needs to be. Higher squashes it vertically, which costs nothing here: the check is whether the overlap lines up horizontally at two different heights.")
    show_all_observations: Field[bool] = Field(True, widget=Widget.switch,
                                              description="Draw every camera's own opinion in the strip below, not just the one the tracker picked. Off falls back to one box per person.")


# Grid colours. Not in ColorSettings: those are per-player track colours, and these are fixed
# meanings that never want tuning — the reader has to be able to tell a seam from a tick.
_GRID_COLOR:   tuple[float, float, float, float] = (1.0, 1.0, 1.0, 0.18)
_HORIZON_COLOR: tuple[float, float, float, float] = (1.0, 1.0, 1.0, 0.45)
_SEAM_COLOR:   tuple[float, float, float, float] = (1.0, 0.35, 0.0, 0.75)   # sector boundary
_AXIS_COLOR:   tuple[float, float, float, float] = (0.0, 0.7, 1.0, 0.6)     # camera optical axis
_LABEL_FG:     tuple[float, float, float, float] = (1.0, 1.0, 1.0, 1.0)
_LABEL_BG:     tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.6)


class PanoramicCameraLayer(LayerBase):
    """The cameras' images unwrapped into one 360-degree strip: x is azimuth, 0 at the left edge.

    **What it is for.** `fov` and `tilt` define the azimuth frame every other number in the
    installation is expressed in, and nothing in the app shows whether they are right — the
    tracker fuses both cameras' views of a seam person into one `Azimuth` before anything draws
    it, destroying the disagreement that would reveal the error. Here the two cameras' pixels are
    drawn on top of each other at the azimuth each one claims. Right geometry: content in an
    overlap coincides. Wrong geometry: it ghosts, and *what* ghosts says *which* number is wrong.

    **Reading it.** Aligns at one height but not another → `tilt`. A constant sideways offset
    across the whole overlap → `fov`. A residual growing toward the frame edges would mean the
    lens is not the equidistant one its spec describes, for which there is no knob. And if the
    *image* coincides while a person's two *boxes* in the strip below do not, the camera geometry
    is right and the distance model is what is off — `ring_radius` and `camera_height`, both
    measured with a tape, not tuned.

    **Depth.** The image is stitched for one assumed depth, `focus_diameter`, because a camera
    0.36 m off centre genuinely sees a different bearing than its neighbour and only the distance
    to the subject resolves it. Nothing about a person feeds this — no box, no pose, no estimate —
    so nothing can fool it: the ghost at other depths is a known, bounded residual rather than a
    symptom.

    Same x mapping and same full-width row as `PanoramicTrackerLayer` below it, so the two line up
    column for column.
    """

    def __init__(self, cam_textures: list[Texture], num_cams: int,
                 tracker: PanoramicTrackerSettings, settings: PanoramaLayerSettings) -> None:
        self._cam_textures: list[Texture] = cam_textures
        self.num_cams: int = max(1, num_cams)
        self._tracker: PanoramicTrackerSettings = tracker
        self._settings: PanoramaLayerSettings = settings

        self.fbo: Fbo = Fbo()
        self._text: Text = Text()
        self._stitch: PanoramicStitch = PanoramicStitch()
        self._rect_shader: DrawColoredRectangle = DrawColoredRectangle()

        self.hot_reloader = HotReloadMethods(self.__class__, True, True)

    @property
    def texture(self) -> Texture:
        return self.fbo

    @property
    def target_fov(self) -> float:
        """The sector one camera owns — the tracker's own `360 / num_cameras`."""
        return 360.0 / self.num_cams

    def allocate(self, width: int, height: int, internal_format: int) -> None:
        self.fbo.allocate(width, height, internal_format)
        self._text.allocate()
        self._stitch.allocate()
        self._rect_shader.allocate()

    def deallocate(self) -> None:
        self.fbo.deallocate()
        self._text.deallocate()
        self._stitch.deallocate()
        self._rect_shader.deallocate()

    def update(self) -> None:
        if not self.fbo.allocated:
            return

        self.fbo.begin()
        glClearColor(0.0, 0.0, 0.0, 1.0)
        glClear(GL_COLOR_BUFFER_BIT)

        self._stitch.use(
            self._cam_textures,
            self._tracker.fov,
            self.target_fov,
            self._tracker.parallax.ring_radius,
            self._settings.focus_diameter,
            self._settings.blend == PanoramaBlend.AVERAGE,
        )

        if self._settings.grid:
            self._draw_grid()

        self.fbo.end()

    def _draw_grid(self) -> None:
        """Azimuth and elevation lines, with the seams and the camera axes picked out.

        The grid is drawn from Python with one quad per line rather than in the stitch shader:
        a few dozen quads a frame costs nothing, and it keeps the line widths in pixels and the
        labels in the same place as the numbers they label.
        """
        spacing: float = max(1.0, self._settings.grid_degrees)
        px_x: float = 1.0 / max(1, self.fbo.width)
        px_y: float = 1.0 / max(1, self.fbo.height)

        seams: set[int] = {round(self.target_fov * i) % 360 for i in range(self.num_cams)}
        axes: set[int] = {round(camera_azimuth(i, self.target_fov)) % 360 for i in range(self.num_cams)}

        # Vertical lines: one azimuth each. The minor grid skips the azimuths that already carry
        # a seam or an axis line, so a coloured line is never overdrawn by a grey one.
        azimuth: float = 0.0
        while azimuth < 360.0:
            if round(azimuth) % 360 not in seams and round(azimuth) % 360 not in axes:
                self._vertical(azimuth, px_x, _GRID_COLOR)
            azimuth += spacing

        for cam_id in range(self.num_cams):
            self._vertical(camera_azimuth(cam_id, self.target_fov), 2.0 * px_x, _AXIS_COLOR)
            self._vertical(self.target_fov * cam_id, 2.0 * px_x, _SEAM_COLOR)

        # Horizontal lines: one elevation each, from the horizon out. The frame is
        # equirectangular, so a row *is* an elevation and this is a linear scale — the same
        # degrees-per-pixel as the azimuth axis.
        vfov: float = max(1.0, self._tracker.parallax.vfov)
        self._horizontal(0.0, vfov, 2.0 * px_y, _HORIZON_COLOR)
        elevation: float = spacing
        while elevation < vfov / 2.0:
            self._horizontal(elevation, vfov, px_y, _GRID_COLOR)
            self._horizontal(-elevation, vfov, px_y, _GRID_COLOR)
            elevation += spacing

        self._draw_labels(spacing)

    def _vertical(self, azimuth: float, width: float, color: tuple[float, float, float, float]) -> None:
        self._rect_shader.use((azimuth % 360.0) / 360.0, 0.0, width, 1.0, *color)

    def _horizontal(self, elevation: float, vfov: float, height: float,
                    color: tuple[float, float, float, float]) -> None:
        # Row 0 is the top of the frame and elevation grows upward, so the horizon is the centre
        # row and positive elevations sit above it.
        y: float = 0.5 - elevation / vfov
        self._rect_shader.use(0.0, y, 1.0, height, *color)

    def _draw_labels(self, spacing: float) -> None:
        """Degree labels along the top, and the focus depth in the corner.

        Labelled every other grid line when the spacing is fine, so they never collide.
        """
        stride: float = spacing if spacing >= 15.0 else spacing * 2.0
        azimuth: float = 0.0
        while azimuth < 360.0:
            x: float = (azimuth / 360.0) * self.fbo.width + 3
            self._text.draw_box_text(x, 3, f'{azimuth:.0f}', _LABEL_FG, _LABEL_BG,
                                     self.fbo.width, self.fbo.height)
            azimuth += stride

        blend: str = 'avg' if self._settings.blend == PanoramaBlend.AVERAGE else 'max'
        footer: str = f'Ø{self._settings.focus_diameter:.1f}m  fov {self._tracker.fov:.0f}  {blend}'
        self._text.draw_box_text(3, self.fbo.height - 22, footer, _LABEL_FG, _LABEL_BG,
                                 self.fbo.width, self.fbo.height)
