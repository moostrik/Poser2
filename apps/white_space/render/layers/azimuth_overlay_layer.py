"""AzimuthOverlayLayer — each person's eye and bbox-centre azimuth over the light strip.

The light places people at their eyes (``EyeAzimuthExtractor``); this shows that position beside
the tracker's bbox-centre azimuth it was shifted from, so the correction can be judged live against
the light it drives. Drawn over both the ring and the beam simulation, which share one 360° x axis.
"""

import math
from typing import Protocol

from modules.board import HasFrames
from modules.render import ColorSettings
from modules.render.layers import LayerBase, strip_spans
from modules.render.shaders import DrawColoredRectangle

from .azimuth_marks import AzimuthMark, build_azimuth_marks, signed_strip_gap
from ...settings import Stage

_EYE_PX: float = 2.0
_BBOX_PX: float = 1.0
_BBOX_ALPHA: float = 0.5
_CONNECTOR_PX: float = 2.0


class AzimuthOverlayBoard(HasFrames, Protocol):
    ...


class AzimuthOverlayLayer(LayerBase):
    """Per person, in their track colour: a solid line at the eye azimuth, a faint line at the
    bbox-centre azimuth, and a bar joining them at mid-height so a small offset still reads.

    Owns no FBO: it draws into whatever viewport is current when `draw()` is called, which must be
    the light strip's, sized by the last `allocate`.
    """

    def __init__(self, board: AzimuthOverlayBoard, colors: ColorSettings) -> None:
        self._board: AzimuthOverlayBoard = board
        self._colors: ColorSettings = colors
        self._rect: DrawColoredRectangle = DrawColoredRectangle()
        self._width: int = 1
        self._height: int = 1

    def allocate(self, width: int, height: int, internal_format: int) -> None:
        # Reallocated with the strip's size whenever the layout changes; the shader is ref-counted,
        # so allocate it once or a single deallocate would never free it.
        if not self._rect.allocated:
            self._rect.allocate()
        self._width = max(1, width)
        self._height = max(1, height)

    def deallocate(self) -> None:
        if self._rect.allocated:
            self._rect.deallocate()

    def update(self) -> None:
        pass

    def draw(self) -> None:
        marks: list[AzimuthMark] = build_azimuth_marks(
            self._board.get_frames(int(Stage.LERP)),
            self._board.get_frames(int(Stage.PREDICT)),
        )
        if not marks:
            return

        colors = self._colors.track_color_tuples
        px_x: float = 1.0 / self._width
        px_y: float = 1.0 / self._height

        for mark in marks:
            r, g, b, a = colors[mark.track_id % len(colors)]
            if not math.isnan(mark.bbox_x):
                self._line(mark.bbox_x, _BBOX_PX * px_x, (r, g, b, a * _BBOX_ALPHA))
            if not math.isnan(mark.eye_x):
                self._line(mark.eye_x, _EYE_PX * px_x, (r, g, b, a))
            if not math.isnan(mark.eye_x) and not math.isnan(mark.bbox_x):
                gap: float = signed_strip_gap(mark.bbox_x, mark.eye_x)
                height: float = _CONNECTOR_PX * px_y
                self._spans(min(mark.bbox_x, mark.bbox_x + gap), abs(gap), 0.5 - height / 2.0, height,
                            (r, g, b, a))

    def _line(self, x: float, width: float, color: tuple[float, float, float, float]) -> None:
        self._spans(x - width / 2.0, width, 0.0, 1.0, color)

    def _spans(self, x: float, width: float, y: float, height: float,
               color: tuple[float, float, float, float]) -> None:
        """One quad, or two when it runs off the strip's 0/360 join (`strip_spans`)."""
        for span_x, span_w in strip_spans(x, width):
            self._rect.use(span_x, y, span_w, height, *color)
