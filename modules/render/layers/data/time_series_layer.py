"""Time Series Layer - values over time from the board, as labelled lanes."""

# Standard library imports
from enum import IntEnum

# Third-party imports
import numpy as np
from OpenGL.GL import GL_RGBA8, GL_VIEWPORT, glGetIntegerv

# Local application imports
from modules.board import HasTimeSeries
from modules.gl import Blit, Fbo, Image, Style, Text, Texture, clear_color
from modules.settings import BaseSettings, Field
from modules.utils import bin_samples, hold_gaps
from ..layer_base import LayerBase
from ...shaders import TimeSeriesShader

COLUMNS: int = 600      # time slices across the layer

LANE_COLORS: list[tuple[float, float, float, float]] = [
    (1.0, 1.0, 1.0, 1.0), (1.0, 0.45, 0.1, 1.0), (0.1, 0.5, 1.0, 1.0), (1.0, 0.85, 0.3, 1.0),
    (0.4, 1.0, 0.5, 1.0), (1.0, 0.3, 0.6, 1.0), (0.6, 0.6, 1.0, 1.0), (0.3, 0.9, 0.9, 1.0),
]


class TimeSeriesLayerSettings(BaseSettings):
    """Configuration for TimeSeriesLayer."""
    window: Field[float] =          Field(10.0, min=1.0, max=120.0, description="Time shown across the layer (s)")
    max_values: Field[list[float]] = Field([0.0], description="Top of each lane in series order; 0 or a missing entry scales to the data")
    line_width: Field[float] =      Field(2.0, min=1.0, max=10.0, description="Line thickness (pixels)")


class TimeSeriesLayer(LayerBase):
    """Every series on the board in its own lane, newest on the right.

    Lanes follow the series' IntEnum order, each named after its member and labelled with its last
    value and the top of its scale. Updated only while drawn: it reads the board, it feeds nothing.
    """

    def __init__(self, board: HasTimeSeries, settings: TimeSeriesLayerSettings) -> None:
        self._board: HasTimeSeries = board
        self.settings: TimeSeriesLayerSettings = settings

        self._fbo: Fbo = Fbo()
        self._image: Image = Image()
        self._shader: TimeSeriesShader = TimeSeriesShader()
        self._text: Text = Text()
        self._labels: list[str] = []

    @property
    def texture(self) -> Texture:
        return self._fbo

    def allocate(self, width: int, height: int, internal_format: int | None = None) -> None:
        self._fbo.allocate(width, height, GL_RGBA8)
        self._shader.allocate()
        self._text.allocate(font_size=22)

    def deallocate(self) -> None:
        self._fbo.deallocate()
        self._image.deallocate()
        self._shader.deallocate()
        self._text.deallocate()

    def update(self) -> None:
        window: float = self.settings.window
        end, series = self._board.get_time_series(window)
        if not series:
            return

        lanes: list[IntEnum] = sorted(series)
        max_values: list[float] = self.settings.max_values
        ranges: np.ndarray = np.full((len(lanes), COLUMNS, 2), -1.0, np.float32)
        self._labels = []
        for index, lane in enumerate(lanes):
            timestamps, values = series[lane]
            low, high = hold_gaps(*bin_samples(timestamps, values, end, window, COLUMNS))
            fixed: float = max_values[index] if index < len(max_values) else 0.0
            top: float = fixed if fixed > 0.0 else float(np.nanmax(high)) * 1.1 if np.isfinite(high).any() else 1.0
            top = max(top, 1e-6)
            sampled: np.ndarray = np.isfinite(low)
            ranges[index, sampled, 0] = np.clip(low[sampled] / top, 0.0, 1.0)
            ranges[index, sampled, 1] = np.clip(high[sampled] / top, 0.0, 1.0)
            last: str = f"{values[-1]:.1f}" if len(values) else "-"
            self._labels.append(f"{lane.name.lower().replace('_', ' ')}  {last}  (top {top:.1f})")

        self._image.set_image(ranges)
        self._image.update()

        self._shader.reload()
        self._fbo.begin()
        clear_color()
        self._shader.use(self._image.texture, len(lanes), COLUMNS,
                         self.settings.line_width * len(lanes) / max(self._fbo.height, 1), LANE_COLORS)
        self._fbo.end()

    def draw(self) -> None:
        Style.push_style()
        Style.set_blend_mode(Style.BlendMode.ALPHA)
        if self._fbo.allocated:
            Blit.use(self._fbo)
        if self._labels and self._text.allocated:
            self._draw_labels()
        Style.pop_style()

    def _draw_labels(self) -> None:
        """Each lane's name, last value and scale, at its top left."""
        viewport = glGetIntegerv(GL_VIEWPORT)
        width, height = int(viewport[2]), int(viewport[3])
        if width <= 0 or height <= 0:
            return
        step: float = height / len(self._labels)
        for index, label in enumerate(self._labels):
            color: tuple[float, float, float, float] = LANE_COLORS[index % len(LANE_COLORS)]
            self._text.draw_box_text(8, int(index * step) + 6, label, color, (0.0, 0.0, 0.0, 0.66), width, height)
