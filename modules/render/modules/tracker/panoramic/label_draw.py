# Local application imports
from modules.gl import Text
from modules.utils import Color

from ....layer_base import LayerBase
from .marks import Mark
from .panorama_compositor_settings import LABEL_BG

_LANE_TOP_PX: float = 25.0   # clear of the grid's azimuth labels along the top edge
_LANE_GAP_PX: float = 2.0
_MARK_GAP_PX: float = 6.0    # between a label and the line it belongs to


class LabelDraw(LayerBase):
    """Each mark's label: on a lane picked by its player slot, anchored on its mark's x, its text running away from the
    other observation of the same person; marks outside the lanes (rejected, waiting) are labeled at the top of their
    own line (`_draw_line_label`). Drawn last. The layout's reasons: modules/render/RENDER.md, *The panorama*."""

    def __init__(self) -> None:
        self._text: Text = Text()
        self._marks: list[Mark] = []
        self._width: int = 1
        self._height: int = 1

    def set_marks(self, marks: list[Mark]) -> None:
        self._marks = marks

    def allocate(self, width: int, height: int, internal_format: int) -> None:
        self._text.allocate()
        self._width = max(1, width)
        self._height = max(1, height)

    def deallocate(self) -> None:
        self._text.deallocate()

    def update(self) -> None:
        pass

    def draw(self) -> None:
        for mark in self._marks:
            if mark.rejected or mark.player is None:
                self._draw_line_label(mark)

        for group in self._by_world().values():
            count: int = len(group)
            for index, mark in enumerate(group):
                # Only the outermost two of a world are labeled; a third view (which this rig's
                # fields cannot produce) keeps its line and loses its text.
                if 0 < index < count - 1:
                    continue

                width, height = self._text.measure_text(mark.label)
                mark_px: float = mark.x * self._width

                # A lone mark reads left to right from its line. A pair diverges: the left one's
                # text runs left, the right one's runs right, whichever of them is the primary.
                if count == 1:
                    to_the_left: bool = mark_px + _MARK_GAP_PX + width > self._width
                else:
                    to_the_left = index == 0

                # The strip's edge overrules the preference either way, because nothing in the text
                # path clamps: text past an edge is cut off, not pushed back in.
                if to_the_left and mark_px - _MARK_GAP_PX - width < 0.0:
                    to_the_left = False
                elif not to_the_left and mark_px + _MARK_GAP_PX + width > self._width:
                    to_the_left = True

                x: float = mark_px - _MARK_GAP_PX - width if to_the_left else mark_px + _MARK_GAP_PX
                x = min(max(x, 0.0), max(0.0, self._width - width))

                assert mark.player is not None    # _by_world leaves out the marks without a player
                y: float = _LANE_TOP_PX + mark.player * (height + _LANE_GAP_PX)
                if y + height > self._height:
                    continue

                self._text.draw_box_text(x, y, mark.label, Color(*mark.color), LABEL_BG,
                                         self._width, self._height)

    def _draw_line_label(self, mark: Mark) -> None:
        """A label at the top of its own line, for marks outside the lanes: a rejected detection's
        rejection, a waiting person's readout.

        Not in a lane: lanes are player slots, and neither of them holds one. Beside the top of the
        line, reading right from it; flipped to the left at the strip's right edge, for the same
        reason lane labels flip. Two such views of one person at a seam can overlap their labels —
        rare, and left so.
        """
        width, height = self._text.measure_text(mark.label)
        mark_px: float = mark.x * self._width
        x: float = mark_px + _MARK_GAP_PX
        if x + width > self._width:
            x = mark_px - _MARK_GAP_PX - width
        x = min(max(x, 0.0), max(0.0, self._width - width))
        y: float = max(0.0, min(mark.top_y * self._height, self._height - height))
        self._text.draw_box_text(x, y, mark.label, Color(*mark.color), LABEL_BG, self._width, self._height)

    def _by_world(self) -> dict[int, list[Mark]]:
        """One slotted person's marks together, ordered left to right — the order the side rule
        needs. The group key is the world id — all waiting marks share `player` None, so the slot
        cannot group them — but the lane drawn is the player's. Rejected and waiting marks are
        labeled at their line instead (`_draw_line_label`)."""
        grouped: dict[int, list[Mark]] = {}
        for mark in self._marks:
            if mark.rejected or mark.player is None:
                continue
            grouped.setdefault(mark.world_id, []).append(mark)
        for group in grouped.values():
            group.sort(key=lambda m: m.x)
        return grouped
