# Third-party imports
from OpenGL.GL import * # type: ignore

# Local application imports
from modules.gl import Text
from modules.tracker import PanoramicTrackerSettings, strip_y

from ...shaders import DrawColoredRectangle
from ..LayerBase import LayerBase
from .PanoramaLayerSettings import PanoramaLayerSettings, PanoramaBlend, \
    GRID_COLOR, HORIZON_COLOR, HORIZON_PX, LABEL_BG, LABEL_FG

# Short names for the footer, in enum order.
_BLEND_NAMES: dict[PanoramaBlend, str] = {
    PanoramaBlend.MAX:        'max',
    PanoramaBlend.AVERAGE:    'avg',
    PanoramaBlend.MIN:        'min',
    PanoramaBlend.DIFFERENCE: 'diff',
    PanoramaBlend.SPLIT:      'rg',
    PanoramaBlend.STRIPE:     'stripe',
}


class GridRenderer(LayerBase):
    """The measuring lattice: degree lines both ways, the horizon, the labels and the footer.

    Both axes are centre-referenced. Azimuth is linear, so the vertical lines are evenly spaced;
    the rows are tangents of elevation (`strip_y`), so the horizontal lines spread toward the top
    exactly as the camera frames' rows do. The horizon stays one straight row.

    The seam and axis lines are **not** here; they belong to `SeamRenderer`, so that turning the
    seams off leaves nothing of them behind and this stays purely a lattice.

    Drawn from Python, one quad per line, rather than in the stitch shader: a few dozen quads a
    frame costs nothing, and it keeps the widths in pixels and the labels beside the numbers they
    label.
    """

    def __init__(self, tracker: PanoramicTrackerSettings, settings: PanoramaLayerSettings) -> None:
        self._tracker: PanoramicTrackerSettings = tracker
        self._settings: PanoramaLayerSettings = settings
        self._rect: DrawColoredRectangle = DrawColoredRectangle()
        self._text: Text = Text()

        self._width: int = 1
        self._height: int = 1
        self._elevation_window: tuple[float, float] = (0.0, 0.0)

    def set_geometry(self, elevation_window: tuple[float, float]) -> None:
        self._elevation_window = elevation_window

    def allocate(self, width: int, height: int, internal_format: int) -> None:
        self._rect.allocate()
        self._text.allocate()
        self._width = max(1, width)
        self._height = max(1, height)

    def deallocate(self) -> None:
        self._rect.deallocate()
        self._text.deallocate()

    def update(self) -> None:
        pass

    def draw(self) -> None:
        spacing: float = max(1.0, self._settings.grid_degrees)
        px_x: float = 1.0 / self._width
        px_y: float = 1.0 / self._height

        azimuth: float = 0.0
        while azimuth < 360.0:
            self._vertical(azimuth, px_x, GRID_COLOR)
            azimuth += spacing

        top, bottom = self._elevation_window
        elevation: float = spacing
        while elevation < max(abs(top), abs(bottom)):
            if elevation < top:
                self._horizontal(elevation, px_y, GRID_COLOR)
            if -elevation > bottom:
                self._horizontal(-elevation, px_y, GRID_COLOR)
            elevation += spacing

        # The horizon last, so no grid line is drawn over it. It is the reference the levelling
        # check is read against (tape at lens height must sit on it), so it has to be unmistakable
        # rather than one more line.
        self._horizontal(0.0, HORIZON_PX * px_y, HORIZON_COLOR)

        self._draw_labels(spacing)

    def _vertical(self, azimuth: float, width: float,
                  color: tuple[float, float, float, float]) -> None:
        self._rect.use((azimuth % 360.0) / 360.0, 0.0, width, 1.0, *color)

    def _horizontal(self, elevation: float, height: float,
                    color: tuple[float, float, float, float]) -> None:
        # Rect y is top-down, and the window's top elevation is the strip's top row. The horizon is
        # only at mid-height when the window is symmetric, which a tilted camera's is not. Centred
        # on the elevation, so a thicker line does not drift below the row it marks.
        self._rect.use(0.0, self._elevation_y(elevation) - height / 2.0, 1.0, height, *color)

    def _elevation_y(self, elevation: float) -> float:
        return strip_y(elevation, self._elevation_window)

    def _draw_labels(self, spacing: float) -> None:
        """Degree labels along the top, the horizon named, and the geometry in the corner.

        Labelled every other grid line when the spacing is fine, so they never collide.
        """
        stride: float = spacing if spacing >= 15.0 else spacing * 2.0
        azimuth: float = 0.0
        while azimuth < 360.0:
            x: float = (azimuth / 360.0) * self._width + 3
            self._text.draw_box_text(x, 3, f'{azimuth:.0f}', LABEL_FG, LABEL_BG,
                                     self._width, self._height)
            azimuth += stride

        horizon_px: float = self._elevation_y(0.0) * self._height
        self._text.draw_box_text(3, max(3.0, horizon_px - 24), 'horizon', HORIZON_COLOR, LABEL_BG,
                                 self._width, self._height)

        top, bottom = self._elevation_window
        blend: str = _BLEND_NAMES.get(self._settings.blend, 'max')
        footer: str = (f'Ø{self._settings.focus_diameter:.1f}m  fov {self._tracker.fov:.0f}  '
                       f'tilt {self._settings.tilt:.0f}  elev {bottom:.0f}..{top:.0f}  {blend}')
        self._text.draw_box_text(3, self._height - 22, footer, LABEL_FG, LABEL_BG,
                                 self._width, self._height)
