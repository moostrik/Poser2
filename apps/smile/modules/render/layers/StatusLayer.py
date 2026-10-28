from modules.gl import Blit, Fbo, Style, Text, Texture

from .LayerBase import LayerBase

FONT_SIZE: int = 18
MARGIN_PX: float = 16.0
TEXT_COLOR: tuple[float, float, float, float] = (1.0, 1.0, 1.0, 1.0)
BOX_COLOR: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.6)


class StatusLayer(LayerBase):
    """The operator's status line: the show's image with one line of text in the bottom left corner."""

    def __init__(self) -> None:
        self._fbo: Fbo = Fbo()
        self._text: Text = Text()
        self._width: int = 1
        self._height: int = 1

    def allocate(self, width: int | None = None, height: int | None = None, internal_format: int | None = None) -> None:
        assert width and height and internal_format
        self._width, self._height = width, height
        self._fbo.allocate(width, height, internal_format)
        self._text.allocate(font_size=FONT_SIZE)

    def deallocate(self) -> None:
        self._fbo.deallocate()
        self._text.deallocate()

    def update(self) -> None:
        pass

    @property
    def texture(self) -> Texture:
        return self._fbo

    def render(self, source: Texture, text: str) -> None:
        """Draw the line over source, the image that would be shown without it."""
        if not source.allocated:
            return
        _, text_height = self._text.measure_text(text)
        self._fbo.begin()
        Blit.use(source)
        Style.set_blend_mode(Style.BlendMode.ALPHA)
        self._text.draw_box_text(MARGIN_PX, self._height - text_height - MARGIN_PX, text, TEXT_COLOR, BOX_COLOR,
                                 self._width, self._height)
        Style.set_blend_mode(Style.BlendMode.DISABLED)
        self._fbo.end()
