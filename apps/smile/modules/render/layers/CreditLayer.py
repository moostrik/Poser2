from OpenGL.GL import glViewport  # type: ignore

from modules.credit import FONT, qr_image, wrap_balanced
from modules.gl import Blit, Fbo, Image, Style, Text, Texture
from modules.render.Sequencer import SequenceType
from modules.render.settings import RenderSettings
from modules.render.shaders import Blend

from .LayerBase import LayerBase

FONT_SIZE: int = 96
MARGIN_PX: int = 40             # from the image's edges
LINE_SPACING: float = 1.0       # of the text's height
OUTLINE_PX: int = 5             # the black edge round the letters, as on a meme
TEXT_COLOR: tuple[float, float, float, float] = (1.0, 1.0, 1.0, 1.0)
OUTLINE_COLOR: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 1.0)
OUTLINE: list[tuple[int, int]] = [(x, y) for x in (-OUTLINE_PX, 0, OUTLINE_PX) for y in (-OUTLINE_PX, 0, OUTLINE_PX) if x or y]
FADE_S: float = 0.5             # the credit fades in and out over this time

# the credit is shown with the visitor's portrait: while the grin plays and while the result holds after it
SHOWN: frozenset[SequenceType] = frozenset({SequenceType.DIFF_LOOP, SequenceType.TIMEOUT})


def fade_step(alpha: float, shown: bool, dt: float) -> float:
    """The credit's opacity one frame of `dt` seconds later: towards 1 while shown, towards 0 after."""
    step: float = dt / FADE_S
    return min(1.0, alpha + step) if shown else max(0.0, alpha - step)


class CreditLayer(LayerBase):
    """The artist's line as a meme caption over the top of the image (white with a black edge, centred,
    in even lines) and a QR code to the project page in the bottom right corner. People film the
    result, so the credit is in their videos."""

    def __init__(self, settings: RenderSettings) -> None:
        self.settings: RenderSettings = settings
        self._card: Fbo = Fbo()             # the image with the credit drawn over it
        self._fbo: Fbo = Fbo()              # the image and the card mixed by the credit's opacity
        self._blend: Blend = Blend()
        self._text: Text = Text()
        self._qr: Image = Image('BGR')
        self._qr_key: tuple[str, int] | None = None
        self._width: int = 1
        self._height: int = 1

    def allocate(self, width: int | None = None, height: int | None = None, internal_format: int | None = None) -> None:
        assert width and height and internal_format
        self._width, self._height = width, height
        self._card.allocate(width, height, internal_format)
        self._fbo.allocate(width, height, internal_format)
        self._blend.allocate()
        self._text.allocate(FONT, FONT_SIZE)

    def deallocate(self) -> None:
        self._card.deallocate()
        self._fbo.deallocate()
        self._blend.deallocate()
        self._text.deallocate()
        self._qr.deallocate()

    def update(self) -> None:
        self._blend.reload()
        key: tuple[str, int] = (self.settings.credit_url, self.settings.credit_size)
        if key != self._qr_key:
            self._qr_key = key
            self._qr.set_image(qr_image(*key))     # uploaded below, at its own size
        self._qr.update()

    @property
    def texture(self) -> Texture:
        return self._fbo

    def render(self, source: Texture, alpha: float) -> None:
        """The credit over source at opacity `alpha`."""
        if not source.allocated:
            return
        self._card.begin()
        Blit.use(source)
        self._draw_credit()
        self._card.end()
        self._fbo.begin()
        self._blend.use(source, self._card, alpha)
        self._fbo.end()

    def _draw_credit(self) -> None:
        qr: int = self._qr.width if self._qr.allocated else 0
        if qr:
            # the viewport's origin is the bottom left
            glViewport(self._width - MARGIN_PX - qr, MARGIN_PX, qr, qr)
            Blit.use(self._qr)
            glViewport(0, 0, self._width, self._height)

        measure = lambda line: self._text.measure_text(line)[0]
        lines: list[str] = wrap_balanced(self.settings.credit_text, measure, self._width - 2 * MARGIN_PX)
        _, text_height = self._text.measure_text("Mg")
        Style.set_blend_mode(Style.BlendMode.ALPHA)
        for i, line in enumerate(lines):
            x: float = (self._width - measure(line)) / 2
            y: float = MARGIN_PX + i * text_height * LINE_SPACING
            for dx, dy in OUTLINE:
                self._text.draw_text(x + dx, y + dy, line, OUTLINE_COLOR, self._width, self._height)
            self._text.draw_text(x, y, line, TEXT_COLOR, self._width, self._height)
        Style.set_blend_mode(Style.BlendMode.DISABLED)
