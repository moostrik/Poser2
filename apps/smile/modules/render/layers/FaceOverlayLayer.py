from modules.gl import Blit, Fbo, Style, Text, Texture

from .LayerBase import LayerBase
from .MarkDrawer import MarkDrawer
from .face_marks import LABEL_BG, Mark

FONT_SIZE: int = 18
LABEL_PAD_PX: float = 4.0


class FaceOverlayLayer(LayerBase):
    """The operator's view: the show's image with a box and keypoints per detected face, and
    the reject reason of a rejected one."""

    def __init__(self) -> None:
        self._fbo: Fbo = Fbo()
        self._drawer: MarkDrawer = MarkDrawer()
        self._text: Text = Text()
        self._marks: list[Mark] = []
        self._width: int = 1
        self._height: int = 1

    def allocate(self, width: int | None = None, height: int | None = None, internal_format: int | None = None) -> None:
        assert width and height and internal_format
        self._width, self._height = width, height
        self._fbo.allocate(width, height, internal_format)
        self._drawer.allocate()
        self._text.allocate(font_size=FONT_SIZE)

    def deallocate(self) -> None:
        self._fbo.deallocate()
        self._drawer.deallocate()
        self._text.deallocate()

    def update(self) -> None:
        self._drawer.reload()

    @property
    def texture(self) -> Texture:
        return self._fbo

    def set_marks(self, marks: list[Mark]) -> None:
        self._marks = marks

    def render(self, source: Texture) -> None:
        """Draw the marks over source, the stage's output of the same camera frame."""
        if not source.allocated:
            return
        self._fbo.begin()
        Blit.use(source)
        Style.set_blend_mode(Style.BlendMode.ALPHA)
        for mark in self._marks:
            self._drawer.draw(mark, self._width, self._height)
        for mark in self._marks:
            if mark.label:
                self._draw_label(mark)
        Style.set_blend_mode(Style.BlendMode.DISABLED)
        self._fbo.end()

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
