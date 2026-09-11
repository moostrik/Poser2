# Standard library imports
import math
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
                                 description="Replace the per-camera row with the stitched 360° panorama")
    blend: Field[PanoramaBlend] = Field(PanoramaBlend.MAX,
                                        description="How overlapping cameras combine")
    focus_diameter: Field[float] = Field(4.5, min=1.0, max=12.0, step=0.5,
                                         description="Play-zone diameter (m) the image is stitched for — exact there, ghosts elsewhere")
    grid: Field[bool] = Field(True, widget=Widget.switch,
                              description="Draw the azimuth and elevation grid")
    grid_degrees: Field[float] = Field(10.0, min=1.0, max=90.0, step=1.0,
                                       description="Grid spacing (°), the same on both axes")
    tilt: Field[float] = Field(0.0, access=Field.INIT,
                               description="Camera up-tilt (°), shared — says which rows the sensor never imaged")
    show_all_observations: Field[bool] = Field(True, widget=Widget.switch,
                                              description="Draw every camera's own opinion, not just the one the tracker picked")


# Grid colours. Not in ColorSettings: those are per-player track colours, and these are fixed
# meanings that never want tuning — the reader has to be able to tell a seam from a tick.
_GRID_COLOR:   tuple[float, float, float, float] = (1.0, 1.0, 1.0, 0.18)
_HORIZON_COLOR: tuple[float, float, float, float] = (0.3, 1.0, 0.3, 1.0)    # its own hue, opaque
_HORIZON_PX:   float = 1.0                                                 # the colour sets it apart, not the width
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

    @property
    def vfov(self) -> float:
        """One camera's vertical field (degrees) — derived by the tracker from `fov`."""
        return max(1.0, self._tracker.parallax.vfov)

    @property
    def populated_band(self) -> tuple[float, float]:
        """The elevations the delivered frames actually carry, measured at the camera.

        The warp hands back a *levelled* frame spanning +/- vfov/2 whatever the mount does, but a
        camera aimed up by `tilt` never imaged the bottom of that: it saw
        `[tilt - vfov/2, tilt + vfov/2]`, and the rows outside the intersection are empty. At
        tilt 15 that is the bottom 18.5% of the frame — the black band.
        """
        half: float = self.vfov / 2.0
        tilt: float = self._settings.tilt
        return (max(-half, tilt - half), min(half, tilt + half))

    @property
    def elevation_window(self) -> tuple[float, float]:
        """(top, bottom) elevation of the strip, measured at the RIG CENTRE.

        The populated band converted to the centre's point of view, at the bearing where the
        conversion is tightest. `tan(e_centre) = tan(e_cam) * d / focus_radius`, and `d` is
        smallest straight ahead (`focus_radius - ring_radius`), so taking the window there
        guarantees every column of the strip is filled rather than fading to black near the
        camera axes.
        """
        focus_radius: float = max(1e-6, self._settings.focus_diameter / 2.0)
        ratio: float = max(0.0, focus_radius - self._tracker.parallax.ring_radius) / focus_radius
        low, high = self.populated_band
        return (math.degrees(math.atan(math.tan(math.radians(high)) * ratio)),
                math.degrees(math.atan(math.tan(math.radians(low)) * ratio)))

    @property
    def aspect_ratio(self) -> float:
        """Width:height of the strip, for the row that holds it.

        360 degrees of azimuth over however many degrees of elevation the window spans — square
        degrees, which is now the *right* answer because both axes are re-projected to the rig
        centre. Nothing here is a preference: change `tilt`, `fov` or `focus_diameter` and the
        row follows.
        """
        top, bottom = self.elevation_window
        return 360.0 / max(1.0, top - bottom)

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
            self.vfov,
            self.target_fov,
            self._tracker.parallax.ring_radius,
            self._settings.focus_diameter,
            self.elevation_window,
            self.populated_band,
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

        # Horizontal lines: one elevation each, from the horizon out. Both axes of the strip are
        # centre-referenced and linear, so this is the same degrees-per-pixel as the azimuth
        # axis — which is what makes the grid square and the head-versus-knee comparison fair.
        top, bottom = self.elevation_window
        elevation: float = spacing
        while elevation < max(abs(top), abs(bottom)):
            if elevation < top:
                self._horizontal(elevation, px_y, _GRID_COLOR)
            if -elevation > bottom:
                self._horizontal(-elevation, px_y, _GRID_COLOR)
            elevation += spacing

        # The horizon last, so no grid, seam or axis line is drawn over it. It is the reference the
        # levelling check is read against (tape at lens height must sit on it), so it has to be
        # unmistakable rather than one more line.
        self._horizontal(0.0, _HORIZON_PX * px_y, _HORIZON_COLOR)

        self._draw_labels(spacing)

    def _vertical(self, azimuth: float, width: float, color: tuple[float, float, float, float]) -> None:
        self._rect_shader.use((azimuth % 360.0) / 360.0, 0.0, width, 1.0, *color)

    def _horizontal(self, elevation: float, height: float,
                    color: tuple[float, float, float, float]) -> None:
        # Rect y is top-down, and the window's top elevation is the strip's top row. The horizon
        # is only at mid-height when the window is symmetric, which a tilted camera's is not.
        # Centred on the elevation, so a thicker line does not drift below the row it marks.
        self._rect_shader.use(0.0, self._elevation_y(elevation) - height / 2.0, 1.0, height, *color)

    def _elevation_y(self, elevation: float) -> float:
        """Normalised, top-down y of an elevation in the strip."""
        top, bottom = self.elevation_window
        return (top - elevation) / max(1e-6, top - bottom)

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

        # Name the horizon at the left end, just above the line, in the line's own colour.
        horizon_px: float = self._elevation_y(0.0) * self.fbo.height
        self._text.draw_box_text(3, max(3.0, horizon_px - 24), 'horizon', _HORIZON_COLOR, _LABEL_BG,
                                 self.fbo.width, self.fbo.height)

        blend: str = 'avg' if self._settings.blend == PanoramaBlend.AVERAGE else 'max'
        top, bottom = self.elevation_window
        footer: str = (f'Ø{self._settings.focus_diameter:.1f}m  fov {self._tracker.fov:.0f}  '
                       f'tilt {self._settings.tilt:.0f}  elev {bottom:.0f}..{top:.0f}  {blend}')
        self._text.draw_box_text(3, self.fbo.height - 22, footer, _LABEL_FG, _LABEL_BG,
                                 self.fbo.width, self.fbo.height)
