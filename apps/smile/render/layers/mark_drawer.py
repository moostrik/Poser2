from modules.gl import shaders
from modules.utils import Color, Rect

from .face_marks import Mark

KEYPOINT_PX: float = 3.0    # half size of a keypoint square


class MarkDrawer:
    """Draws a mark's box outline and keypoint dots into the bound framebuffer (alpha blending on).
    `alpha` fades them as a whole; without `outline` only the dots."""

    def __init__(self) -> None:
        self._outline: shaders.DrawRectangleOutline = shaders.DrawRectangleOutline()
        self._rect: shaders.DrawColoredRectangle = shaders.DrawColoredRectangle()

    def allocate(self) -> None:
        self._outline.allocate()
        self._rect.allocate()

    def deallocate(self) -> None:
        self._outline.deallocate()
        self._rect.deallocate()

    def reload(self) -> None:
        self._outline.reload()
        self._rect.reload()

    def draw(self, mark: Mark, width: int, height: int, alpha: float = 1.0, outline: bool = True) -> None:
        w, h = float(width), float(height)
        r, g, b, a = mark.color
        a *= alpha
        if outline:
            self._outline.use(Rect(mark.x, mark.y, mark.w, mark.h), Color(r, g, b, a),
                              mark.line / w / max(1e-6, mark.w), mark.line / h / max(1e-6, mark.h))
        if mark.dots is not None:
            r, g, b, a = mark.dots
            a *= alpha
        dot: Color = Color(r, g, b, a)
        for kx, ky in mark.kps:
            self._rect.use(Rect(kx - KEYPOINT_PX / w, ky - KEYPOINT_PX / h, 2.0 * KEYPOINT_PX / w, 2.0 * KEYPOINT_PX / h), dot)
