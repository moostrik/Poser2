# Third-party imports
from OpenGL.GL import * # type: ignore

# Local application imports
from modules.utils import HotReloadMethods

from ...shaders import DrawColoredRectangle
from ..LayerBase import LayerBase
from .marks import Mark

# A mark is a line, not a box: the box's width said nothing the azimuth does not, and two boxes at
# a seam overlapped into a shape neither camera claimed.
_PRIMARY_PX: float = 2.0
_CANDIDATE_PX: float = 1.0
_TICK_PX: float = 9.0     # half-width of the foot tick
_TICK_H_PX: float = 2.0


class ObservationRenderer(LayerBase):
    """A line and a foot tick per observation, at the azimuth the tracker gives it.

    **Every observation, not one per person**, which is the point. A person on a seam is seen by two
    cameras and each has its own opinion of their azimuth; the tracker fuses those into one
    `Azimuth` before anything else in the app sees it, so a disagreement — the symptom of a wrong
    `fov`, `tilt` or `ring_radius` — is invisible everywhere else. Here the two lines stand side by
    side in the same world colour and the gap between them *is* the error. The primary is the opaque
    one.

    The line spans the person's own extent, head elevation to foot elevation, so it says how high in
    the room they are as well as where; the tick marks the row the distance estimate was read from,
    which makes the input to `R` visible rather than implied.

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

        for mark in self._marks:
            r, g, b, a = mark.color
            width: float = (_PRIMARY_PX if mark.is_primary else _CANDIDATE_PX) * px_x

            # A partly visible person's box legitimately runs past the frame edge, so clip to the
            # strip rather than trusting the rows.
            top: float = min(max(mark.top_y, 0.0), 1.0)
            bottom: float = min(max(mark.bottom_y, 0.0), 1.0)
            if bottom < top:
                top, bottom = bottom, top

            self._rect.use(mark.x - width / 2.0, top, width, max(px_y, bottom - top), r, g, b, a)

            # The foot tick, only where the feet are actually in the picture: a person closer than
            # the floor-plane model can see has their feet in the empty band below the frame, and a
            # tick pinned to the strip's bottom edge would claim a reading that was never made.
            if 0.0 < mark.bottom_y < 1.0:
                self._rect.use(mark.x - _TICK_PX * px_x, bottom - _TICK_H_PX * px_y / 2.0,
                               2.0 * _TICK_PX * px_x, _TICK_H_PX * px_y, r, g, b, a)
