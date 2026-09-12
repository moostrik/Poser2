# Third-party imports
from OpenGL.GL import * # type: ignore

# Local application imports
from modules.tracker import strip_spans
from modules.utils import HotReloadMethods

from ...shaders import DrawColoredRectangle
from ..LayerBase import LayerBase
from .marks import Mark

# A mark is a line, not a box: the box's width said nothing the azimuth does not, and two boxes at
# a seam overlapped into a shape neither camera claimed.
_PRIMARY_PX: float = 2.0
_CANDIDATE_PX: float = 1.0

# The tolerance field's opacity, and it is NOT scaled by the mark's own confidence alpha. The line
# already says LOST / candidate / primary, and the re-acquisition window matters most on a LOST
# mark — dimming that one to a twentieth would hide the single case it exists for.
_FIELD_ALPHA: float = 0.2


class ObservationRenderer(LayerBase):
    """A line per observation at the azimuth the tracker gives it, inside the rule that governs it.

    **Every observation, not one per person**, which is the point. A person on a seam is seen by
    two cameras and each has its own opinion of their azimuth; the tracker fuses those into one
    `world_angle` before anything else in the app sees it, so a disagreement — the symptom of a
    wrong `fov`, `tilt` or `ring_radius` — is invisible everywhere else. Here the two lines stand
    side by side in the same world colour and the gap between them *is* the error. The primary is
    the opaque one.

    A mark is the tracker's belief, not the picture: it is placed at `world_angle` and at the
    person's own estimated distance, while the image under it is stitched for `focus_diameter`. So
    a line generally sits beside its own pixels, by an amount that is the difference between those
    two depths and is not a measurement of anything — `R` on the label is the honest reading.

    Two things per observation, and no horizontal marks at all: those used to crowd the two rows —
    the feet and the horizon — that everything else on the strip is read against.

    - **The line**, head elevation to foot elevation, so it says how high in the room they are as
      well as where. Its bottom end is the row the distance was read from, so what feeds `R` is
      still visible without a tick to point at it.
    - **The field** around it, same height, translucent, as wide as the tolerance that decides what
      this observation may be joined to: `seam.link_angle` where a second camera also sees it,
      `reacquire_angle` where none does (`marks._tolerance`). Read it as a **pair test** — two
      fields of one colour that overlap are two observations the tracker will join, and two colours
      that overlap are two people it might confuse. A field is drawn the tolerance wide rather than
      either side of the line precisely so that overlapping *is* the gate.

    Owns no FBO: the compositor's is bound when `draw()` is called.
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
        # whole would let a primary's translucent field tint a candidate's line. Every field first
        # means no field ever covers a line.
        for mark in self._marks:
            top, bottom = self._rows(mark, px_y)
            r, g, b, _a = mark.color
            self._spans(mark.tolerance_x, mark.tolerance_w, top, bottom - top,
                        (r, g, b, _FIELD_ALPHA))

        for mark in self._marks:
            top, bottom = self._rows(mark, px_y)
            width: float = (_PRIMARY_PX if mark.is_primary else _CANDIDATE_PX) * px_x
            self._spans(mark.x - width / 2.0, width, top, bottom - top, mark.color)

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
