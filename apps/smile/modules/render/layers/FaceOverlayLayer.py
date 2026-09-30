from modules.gl import Blit, Fbo, Image, Style, Text, Texture
from modules.render.shaders import DrawColoredRectangle, DrawRectangleOutline

from .LayerBase import LayerBase
from .face_marks import LABEL_BG, Mark

FONT_SIZE: int = 18
KEYPOINT_PX: float = 3.0    # half size of a keypoint square
LABEL_PAD_PX: float = 4.0


class FaceOverlayLayer(LayerBase):
    """The operator's view: the raw camera with a box, keypoints and label per detected face."""

    def __init__(self, camera: Image) -> None:
        self._camera: Image = camera
        self._fbo: Fbo = Fbo()
        self._outline: DrawRectangleOutline = DrawRectangleOutline()
        self._rect: DrawColoredRectangle = DrawColoredRectangle()
        self._text: Text = Text()
        self._marks: list[Mark] = []
        self._width: int = 1
        self._height: int = 1

    def allocate(self, width: int | None = None, height: int | None = None, internal_format: int | None = None) -> None:
        assert width and height and internal_format
        self._width, self._height = width, height
        self._fbo.allocate(width, height, internal_format)
        self._outline.allocate()
        self._rect.allocate()
        self._text.allocate(font_size=FONT_SIZE)

    def deallocate(self) -> None:
        self._fbo.deallocate()
        self._outline.deallocate()
        self._rect.deallocate()
        self._text.deallocate()

    def update(self) -> None:
        self._outline.reload()
        self._rect.reload()

    @property
    def texture(self) -> Texture:
        return self._fbo

    def set_marks(self, marks: list[Mark]) -> None:
        self._marks = marks

    def render(self) -> None:
        if not self._camera.allocated:
            return
        self._fbo.begin()
        Blit.use(self._camera)
        Style.set_blend_mode(Style.BlendMode.ALPHA)
        for mark in self._marks:
            self._draw_box(mark)
        for mark in self._marks:
            self._draw_label(mark)
        Style.set_blend_mode(Style.BlendMode.DISABLED)
        self._fbo.end()

    def _draw_box(self, mark: Mark) -> None:
        w, h = float(self._width), float(self._height)
        self._outline.use(mark.x, mark.y, mark.w, mark.h, *mark.color, mark.line / w / max(1e-6, mark.w), mark.line / h / max(1e-6, mark.h))
        for kx, ky in mark.kps:
            self._rect.use(kx - KEYPOINT_PX / w, ky - KEYPOINT_PX / h, 2.0 * KEYPOINT_PX / w, 2.0 * KEYPOINT_PX / h, *mark.color)

    def _draw_label(self, mark: Mark) -> None:
        """Above the box, inside the frame; below its top edge when there is no room above."""
        text_width, text_height = self._text.measure_text(mark.label)
        x: float = mark.x * self._width
        y: float = mark.y * self._height - text_height - LABEL_PAD_PX * 2
        if y < 0.0:
            y = mark.y * self._height + LABEL_PAD_PX
        x = max(LABEL_PAD_PX, min(x, self._width - text_width - LABEL_PAD_PX))
        y = max(LABEL_PAD_PX, min(y, self._height - text_height - LABEL_PAD_PX))
        self._text.draw_box_text(x, y, mark.label, mark.color, LABEL_BG, self._width, self._height)
