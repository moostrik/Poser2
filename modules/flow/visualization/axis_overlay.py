"""The legend: the strip's and the histogram's lane names, log lines and labels, and the
colormap's key, drawn into a small FBO only when what they show changes, and blitted over the
plot every draw. Text builds its glyphs in Python per string, so drawing some twenty strings per
panel each tick is what it saves."""

from enum import IntEnum, auto

from OpenGL.GL import GL_RGBA8

from modules.gl import Fbo, Style, Text, Texture
from .log_axis import histogram_lines, strip_lines, value_limit, AxisLine
from .shaders import LogGrid, ColormapKey
from .strip import ALL_CHANNELS, LANES, STRIP_HEIGHT, STRIP_WIDTH

TEXT: tuple[float, float, float, float] = (1.0, 1.0, 1.0, 1.0)
BOX: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.6)
MARGIN: int = 8
KEY_BAR: int = 18               # a key bar's height (pixels)
KEY_GAP: int = 8                # between key bars (pixels)
KEY_NAME: int = 28              # the column of a key bar's channel name (pixels)


class Axis(IntEnum):
    """The plot a legend is drawn for."""
    STRIP = 0
    HISTOGRAM = auto()
    COLORMAP = auto()


class AxisOverlay:
    """Lane names, log lines and their labels, or the color key, on transparent at the strip's size.
    Render thread only."""

    def __init__(self) -> None:
        self._fbo: Fbo = Fbo()
        self._text: Text = Text()
        self._grid: LogGrid = LogGrid()
        self._key: ColormapKey = ColormapKey()
        self._drawn: tuple[Axis, int, float, int] | None = None     # what the FBO shows

    @property
    def texture(self) -> Texture:
        return self._fbo

    def allocate(self) -> None:
        self._fbo.allocate(STRIP_WIDTH, STRIP_HEIGHT, GL_RGBA8)
        self._text.allocate(font_size=18)
        self._grid.allocate()
        self._key.allocate()
        self._drawn = None

    def deallocate(self) -> None:
        self._fbo.deallocate()
        self._text.deallocate()
        self._grid.deallocate()
        self._key.deallocate()
        self._drawn = None

    def update(self, axis: Axis, channel: int, scale: float, samples: int) -> None:
        """Draw the legend for this plot, unless it shows it already; `samples` is the histogram's
        total of samples counted."""
        shown: tuple[Axis, int, float, int] = (axis, channel, scale, samples)
        if shown == self._drawn or not self._fbo.allocated:
            return
        self._drawn = shown

        self._fbo.clear(0.0, 0.0, 0.0, 0.0)
        self._fbo.begin()
        Style.push_style()
        Style.set_blend_mode(Style.BlendMode.DISABLED)     # the shapes' alpha stored as is, for the blit to apply once
        if axis == Axis.COLORMAP:
            self._draw_key(channel, scale)
        else:
            self._draw_axis(axis, channel, scale, samples)
        Style.pop_style()
        self._fbo.end()

    def _draw_axis(self, axis: Axis, channel: int, scale: float, samples: int) -> None:
        """The lane names, a line at each decade with its label, and the histogram's 0 and limits."""
        names: tuple[str, ...] = LANES if channel == ALL_CHANNELS else (LANES[channel],)
        lanes: int = len(names)
        lines: list[AxisLine] = strip_lines(scale) if axis == Axis.STRIP else histogram_lines(samples)
        w, h = STRIP_WIDTH, STRIP_HEIGHT

        self._grid.reload()
        self._grid.use(lanes, [line.fraction for line in lines], axis == Axis.HISTOGRAM)
        if not self._text.allocated:
            return
        Style.set_blend_mode(Style.BlendMode.ALPHA)
        for lane in range(lanes):
            top: float = lane * h / lanes
            bottom: float = (lane + 1) * h / lanes
            self._text.draw_box_text(MARGIN, top + MARGIN, names[lane], TEXT, BOX, w, h)
            for line in lines:
                text_w, text_h = self._text.measure_text(line.label)
                line_y: float = bottom - line.fraction * (bottom - top)
                y: float = line_y - text_h - 4
                if y < top + 4:
                    y = line_y + 4                  # under the line where it does not fit above
                self._text.draw_box_text(w - text_w - MARGIN, y, line.label, TEXT, BOX, w, h)
        if axis == Axis.HISTOGRAM:
            self._draw_limits(scale, MARGIN, w - MARGIN, h - MARGIN)

    def _draw_key(self, channel: int, scale: float) -> None:
        """A color bar per channel shown, with its name, and -1, 0 and 1 after `scale` under them."""
        w, h = STRIP_WIDTH, STRIP_HEIGHT
        names: tuple[str, ...] = LANES[:3] if channel == ALL_CHANNELS else (LANES[channel],)
        bars: int = len(names)
        text_h: int = self._text.measure_text('0')[1] if self._text.allocated else 0
        x0: float = MARGIN + KEY_NAME
        x1: float = w - MARGIN
        bottom: float = MARGIN + text_h + 8                         # from the bottom, room for the values
        top: float = bottom + bars * KEY_BAR + (bars - 1) * KEY_GAP
        self._key.reload()
        self._key.use((x0, bottom, x1, top), channel, KEY_BAR, KEY_GAP)
        if not self._text.allocated:
            return
        Style.set_blend_mode(Style.BlendMode.ALPHA)
        for bar, name in enumerate(names):
            bar_top: float = h - top + bar * (KEY_BAR + KEY_GAP)    # in text's pixels, from the top
            self._text.draw_box_text(MARGIN, bar_top, name, TEXT, BOX, w, h)
        self._draw_limits(scale, x0, x1, h - MARGIN)

    def _draw_limits(self, scale: float, left: float, right: float, bottom: float) -> None:
        """-1, 0 and 1 after `scale`, at the left, the middle and the right, ending at `bottom` (from the top)."""
        w, h = STRIP_WIDTH, STRIP_HEIGHT
        limit: str = value_limit(scale)
        for label, anchor in (('-' + limit, 0.0), ('0', 0.5), (limit, 1.0)):
            text_w, text_h = self._text.measure_text(label)
            x: float = left + anchor * (right - left) - anchor * text_w
            self._text.draw_box_text(x, bottom - text_h, label, TEXT, BOX, w, h)
