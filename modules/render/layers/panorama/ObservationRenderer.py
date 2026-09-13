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

# The foot tick: wide enough to read against a zone edge by eye, and 3 px tall so it has a
# definite row. Deliberately larger than the line — the tick is the measurement, the line is
# context — and the same size for every observation, since a reading's precision does not
# depend on whether the tracker picked that camera.
_TICK_PX: float = 17.0
_TICK_HEIGHT_PX: float = 3.0

# The tolerance field's opacity, and it is NOT scaled by the line's confidence alpha: the line
# already says candidate / primary. Only a LOST mark's field dims, by its own `field_color` alpha,
# fading out as the identity runs toward `lost_timeout`.
_FIELD_ALPHA: float = 0.2


class ObservationRenderer(LayerBase):
    """A line per observation at the azimuth the tracker gives it, inside the rule that governs it.

    **Every observation, not one per person**, which is the point. A person on a seam is seen by
    two cameras and each has its own opinion of their azimuth; the tracker fuses those into one
    `world_angle` before anything else in the app sees it, so a disagreement — the symptom of a
    wrong `fov`, `tilt` or `ring_radius` — is invisible everywhere else. Here the two lines stand
    side by side in the same world colour and the gap between them *is* the error. The primary is
    the opaque one.

    A mark is the tracker's belief, not the picture: its x is `world_angle`, at the fixed
    `parallax_radius`, while the image under it is stitched for `focus_radius`. So a line
    generally sits beside its own pixels, by a constant that is the difference between those two
    depths and is not a measurement of anything. Its **rows** are the person's own distance, which
    is what the tick below depends on.

    Three things per observation:

    - **The line**, head elevation to foot elevation, so it says how high in the room they are as
      well as where.
    - **The foot tick**, a short horizontal at the foot row — and this one is an **instrument, not
      a decoration**. Because a mark's rows go through the person's own distance, the foot row is
      exactly `atan(camera_height / R)` below the horizon, which is the formula the grid's zone
      field is drawn from. So the tick and the yellow zone edges are directly comparable: **tape
      R 1.5 and R 3.5 on the floor, stand on one, and the tick must land on that edge.** That is the
      strip's one precise distance check, and it is why the rows are not on the parallax cylinder
      with the x — there the tick would be 20 px out at R 1.5 and check nothing.
    - **The field** around the line, same height, translucent, as wide as the tolerance that decides
      what this observation may be joined to: `seam.link_angle` where a second camera also sees it,
      `reacquire_angle` where none does (`marks._tolerance`). Read it as a **pair test** — two
      fields of one colour that overlap are two observations the tracker will join, and two colours
      that overlap are two people it might confuse. A field is drawn the tolerance wide rather than
      either side of the line precisely so that overlapping *is* the gate.

    **A LOST mark** keeps its line and field, and fades: the line to grey, the field out.

    **A detection the tracker dropped** is drawn in grey as its line and its foot tick, with no field
    — no rule can join it to anything. Its tag names the filter (`LabelRenderer`). So a person never
    leaves the strip without a reason on screen.

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
            if mark.rejected:
                continue                                  # no rule can join it: no field
            top, bottom = self._rows(mark, px_y)
            r, g, b, visible = mark.field_color
            if visible <= 0.0:
                continue
            self._spans(mark.tolerance_x, mark.tolerance_w, top, bottom - top,
                        (r, g, b, _FIELD_ALPHA * visible))

        for mark in self._marks:
            top, bottom = self._rows(mark, px_y)
            width: float = (_PRIMARY_PX if mark.is_primary else _CANDIDATE_PX) * px_x
            self._spans(mark.x - width / 2.0, width, top, bottom - top, mark.color)
            self._foot_tick(mark, px_x, px_y)

    def _foot_tick(self, mark: Mark, px_x: float, px_y: float) -> None:
        """The foot row, marked so it can be read against the zone field.

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
