# Third-party imports
from OpenGL.GL import * # type: ignore

# Local application imports
from modules.utils import HotReloadMethods

from ...shaders import DrawColoredRectangle
from ..LayerBase import LayerBase
from .marks import Mark
from .strip import strip_spans

_PRIMARY_PX: float = 2.0
_PASSIVE_PX: float = 1.0

# The foot tick: wide enough to read against a zone edge by eye, 3 px so it has a definite row. The
# same for every view, since a reading's precision does not depend on which camera was picked.
_TICK_PX: float = 17.0
_TICK_HEIGHT_PX: float = 3.0

# The field's opacity, filled and outlined, both scaled by `field_color`'s alpha (a LOST fade). The
# outline stays below a passive line's alpha (`marks._PASSIVE_ALPHA`) so it does not read as a line.
_FIELD_ALPHA: float = 0.2
_OUTLINE_ALPHA: float = 0.6


class MarkRenderer(LayerBase):
    """Every observation's mark, not one per person: each camera's own view of a seam person stands
    beside the other's in the same world colour, which the fused `world_angle` hides everywhere else.
    What a mark's position means is in `marks`; its label is `LabelRenderer`'s.

    Per mark:

    - **The line**, head to foot elevation; 2 px for the primary view, 1 px for the passive views.
    - **The foot tick** at the foot row: the instrument. Stand on a taped R 1.5 or R 3.5 circle and
      the tick must land on that edge of the grid's zone band.
    - **The field**, the join range as a width (`marks._field`): two fields of one colour that
      overlap will be joined; two colours that overlap are two people it might confuse. Filled for
      the primary view, outlined for passive views and for LOST marks.

    A rejected detection is its grey line and tick, no field. Owns no FBO: the compositor's is bound
    when `draw()` is called.
    """

    def __init__(self) -> None:
        self._rect: DrawColoredRectangle = DrawColoredRectangle()
        self._marks: list[Mark] = []
        self._width: int = 1
        self._height: int = 1

        self.hot_reloader = HotReloadMethods(self.__class__, True, True)

    def set_marks(self, marks: list[Mark]) -> None:
        self._marks = marks

    def allocate(self, width: int, height: int, internal_format: int) -> None:
        self._rect.allocate()
        self._width = max(1, width)
        self._height = max(1, height)

    def deallocate(self) -> None:
        self._rect.deallocate()

    def update(self) -> None:
        pass

    def draw(self) -> None:
        px_x: float = 1.0 / self._width
        px_y: float = 1.0 / self._height

        # Two passes, not one per mark: `build_marks` sorts primaries last, so drawing each mark
        # whole would let a primary's translucent field tint a passive line. Every field first
        # means no field ever covers a line.
        for mark in self._marks:
            if mark.rejected:
                continue                                  # no rule can join it: no field
            top, bottom = self._rows(mark, px_y)
            r, g, b, visible = mark.field_color
            if visible <= 0.0:
                continue
            if mark.field_outline:
                self._outline(mark.field_x, mark.field_w, top, bottom, px_x, px_y,
                              (r, g, b, _OUTLINE_ALPHA * visible))
            else:
                self._spans(mark.field_x, mark.field_w, top, bottom - top,
                            (r, g, b, _FIELD_ALPHA * visible))

        for mark in self._marks:
            top, bottom = self._rows(mark, px_y)
            width: float = (_PRIMARY_PX if mark.is_primary else _PASSIVE_PX) * px_x
            self._spans(mark.x - width / 2.0, width, top, bottom - top, mark.color)
            self._foot_tick(mark, px_x, px_y)

    def _foot_tick(self, mark: Mark, px_x: float, px_y: float) -> None:
        """The foot row, marked so it can be read against the zone band.

        Guarded on the **raw** row rather than the clipped one: a tick pinned to the strip's edge
        would claim a reading that was not made, and a person whose feet fall outside the window is
        exactly the case where that matters. Centred on the row, so its thickness does not move the
        reading, and centred on the line, so it reads as that person's. Nothing is drawn for feet
        that are not on the floor at all (`has_foot`).
        """
        if not mark.has_foot or not 0.0 < mark.bottom_y < 1.0:
            return
        height: float = _TICK_HEIGHT_PX * px_y
        self._spans(mark.x - _TICK_PX * px_x / 2.0, _TICK_PX * px_x,
                    mark.bottom_y - height / 2.0, height, mark.color)

    def _outline(self, x: float, width: float, top: float, bottom: float, px_x: float, px_y: float,
                 color: tuple[float, float, float, float]) -> None:
        """A 1 px frame around a field, drawn inside its bounds so it covers the same pixels a fill
        would, and its overlap with another field still reads as the pair test. Edges only, so the
        sides do not double the corners' alpha."""
        width = max(width, 2.0 * px_x)
        height: float = bottom - top
        self._spans(x, width, top, px_y, color)                                  # top
        if height > 2.0 * px_y:
            self._spans(x, width, bottom - px_y, px_y, color)                    # bottom
            self._spans(x, px_x, top + px_y, height - 2.0 * px_y, color)         # left
            self._spans(x + width - px_x, px_x, top + px_y, height - 2.0 * px_y, color)  # right

    def _rows(self, mark: Mark, px_y: float) -> tuple[float, float]:
        """The mark's (top, bottom) as the strip can draw them.

        A partly visible person's box legitimately runs past the frame edge, so clip to the strip
        rather than trusting the rows, and keep at least a pixel of height so a degenerate box
        still shows something.
        """
        top: float = min(max(mark.top_y, 0.0), 1.0)
        bottom: float = min(max(mark.bottom_y, 0.0), 1.0)
        if bottom < top:
            top, bottom = bottom, top
        return top, max(top + px_y, bottom)

    def _spans(self, x: float, width: float, y: float, height: float,
               color: tuple[float, float, float, float]) -> None:
        """One quad, or two when it runs off the strip's 0/360 join (`strip_spans`)."""
        for span_x, span_w in strip_spans(x, width):
            self._rect.use(span_x, y, span_w, height, *color)
