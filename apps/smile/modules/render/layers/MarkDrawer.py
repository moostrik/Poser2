from modules.render.shaders import DrawColoredRectangle, DrawRectangleOutline

from .face_marks import Mark

KEYPOINT_PX: float = 3.0    # half size of a keypoint square
BAR_PX: float = 6.0         # width of a progress bar
BAR_PAD_PX: float = 6.0     # between a bar and the box's side, top and bottom edges
BAR_TRACK_ALPHA: float = 0.25


class MarkDrawer:
    """Draws a mark's box outline, keypoint dots and progress bars into the bound framebuffer
    (alpha blending on). `alpha` fades them as a whole."""

    def __init__(self) -> None:
        self._outline: DrawRectangleOutline = DrawRectangleOutline()
        self._rect: DrawColoredRectangle = DrawColoredRectangle()

    def allocate(self) -> None:
        self._outline.allocate()
        self._rect.allocate()

    def deallocate(self) -> None:
        self._outline.deallocate()
        self._rect.deallocate()

    def reload(self) -> None:
        self._outline.reload()
        self._rect.reload()

    def draw(self, mark: Mark, width: int, height: int, alpha: float = 1.0) -> None:
        w, h = float(width), float(height)
        r, g, b, a = mark.color
        a *= alpha
        self._outline.use(mark.x, mark.y, mark.w, mark.h, r, g, b, a, mark.line / w / max(1e-6, mark.w), mark.line / h / max(1e-6, mark.h))
        for kx, ky in mark.kps:
            self._rect.use(kx - KEYPOINT_PX / w, ky - KEYPOINT_PX / h, 2.0 * KEYPOINT_PX / w, 2.0 * KEYPOINT_PX / h, r, g, b, a)

    def draw_bar(self, mark: Mark, progress: float, width: int, height: int, alpha: float = 1.0) -> None:
        """A vertical bar inside the box at its left and its right edge, filled from the bottom up to
        progress (0..1)."""
        w, h = float(width), float(height)
        bar_w: float = BAR_PX / w
        pad_w: float = BAR_PAD_PX / w
        bar_y: float = mark.y + BAR_PAD_PX / h
        bar_h: float = mark.h - 2.0 * BAR_PAD_PX / h
        if bar_h <= 0.0 or mark.w <= 2.0 * (pad_w + bar_w):
            return
        r, g, b, a = mark.color
        a *= alpha
        fill: float = bar_h * min(max(progress, 0.0), 1.0)
        for bar_x in (mark.x + pad_w, mark.x + mark.w - pad_w - bar_w):
            self._rect.use(bar_x, bar_y, bar_w, bar_h, r, g, b, a * BAR_TRACK_ALPHA)
            if fill > 0.0:
                self._rect.use(bar_x, bar_y + bar_h - fill, bar_w, fill, r, g, b, a)
