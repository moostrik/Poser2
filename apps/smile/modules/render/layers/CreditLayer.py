from OpenGL.GL import glViewport  # type: ignore

from modules.credit import FONT, CreditLines, qr_image, wrap_balanced
from modules.gl import Blit, Fbo, Image, Style, Text, Texture
from modules.render.Sequencer import SequenceType
from modules.render.settings import RenderSettings
from modules.render.shaders import Blend

from .LayerBase import LayerBase
from .passes import copy

FONT_SIZE: int = 96
MARGIN_PX: int = 40             # from the image's edges
LINE_SPACING: float = 1.0       # of the text's height
OUTLINE_PX: int = 5             # the black edge round the letters, as on a meme
TEXT_COLOR: tuple[float, float, float, float] = (1.0, 1.0, 1.0, 1.0)
OUTLINE_COLOR: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 1.0)
OUTLINE: list[tuple[int, int]] = [(x, y) for x in (-OUTLINE_PX, 0, OUTLINE_PX) for y in (-OUTLINE_PX, 0, OUTLINE_PX) if x or y]
FADE_S: float = 0.5             # the caption fades in and out over this time

# the caption is shown with the visitor's portrait: while the grin plays and while the result holds after it
SHOWN: frozenset[SequenceType] = frozenset({SequenceType.DIFF_LOOP, SequenceType.TIMEOUT})


def fade_step(alpha: float, shown: bool, dt: float) -> float:
    """The caption's opacity one frame of `dt` seconds later: towards 1 while shown, towards 0 after."""
    step: float = dt / FADE_S
    return min(1.0, alpha + step) if shown else max(0.0, alpha - step)


class CreditLayer(LayerBase):
    """A QR code to the project page in the bottom right corner, always, and the artist's credit as a meme
    caption over the top of the image (white with a black edge, centred): the name, and under it the
    visitor's line, drawn from the lines file at every capture (`new_visitor`), at the opacity `render`
    gets. People film the result, so the credit is in their videos."""

    def __init__(self, settings: RenderSettings) -> None:
        self.settings: RenderSettings = settings
        self._lines: CreditLines = CreditLines(settings.credit_lines)
        self._base: Fbo = Fbo()             # the image with the QR code
        self._card: Fbo = Fbo()             # and with the caption
        self._fbo: Fbo = Fbo()              # the two mixed by the caption's opacity
        self._blend: Blend = Blend()
        self._text: Text = Text()
        self._qr: Image = Image('BGR')
        self._qr_key: tuple[str, int] | None = None
        self._width: int = 1
        self._height: int = 1

    def allocate(self, width: int | None = None, height: int | None = None, internal_format: int | None = None) -> None:
        assert width and height and internal_format
        self._width, self._height = width, height
        self._base.allocate(width, height, internal_format)
        self._card.allocate(width, height, internal_format)
        self._fbo.allocate(width, height, internal_format)
        self._blend.allocate()
        self._text.allocate(FONT, FONT_SIZE)

    def deallocate(self) -> None:
        self._base.deallocate()
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
        """The QR code over source, and the caption at opacity `alpha`."""
        if not source.allocated:
            return
        self._base.begin()
        Blit.use(source)
        qr: int = self._qr.width if self._qr.allocated else 0
        if qr:
            # the viewport's origin is the bottom left
            glViewport(self._width - MARGIN_PX - qr, MARGIN_PX, qr, qr)
            Blit.use(self._qr)
            glViewport(0, 0, self._width, self._height)
        self._base.end()
        if alpha <= 0.0:
            copy(self._base, self._fbo)
            return
        self._card.begin()
        Blit.use(self._base)
        self._draw_caption()
        self._card.end()
        self._fbo.begin()
        self._blend.use(self._base, self._card, alpha)
        self._fbo.end()

    def new_visitor(self) -> None:
        """Draw the line under the name for the visitor that is captured now."""
        self.settings.credit_line = self._lines.pick()

    def _draw_caption(self) -> None:
        measure = lambda line: self._text.measure_text(line)[0]
        width: int = self._width - 2 * MARGIN_PX
        # the name and the line each on a line of their own; one that is too wide for the image wraps
        lines: list[str] = wrap_balanced(self.settings.credit_name, measure, width) + wrap_balanced(self.settings.credit_line, measure, width)
        _, text_height = self._text.measure_text("Mg")
        Style.set_blend_mode(Style.BlendMode.ALPHA)
        for i, line in enumerate(lines):
            x: float = (self._width - measure(line)) / 2
            y: float = MARGIN_PX + i * text_height * LINE_SPACING
            for dx, dy in OUTLINE:
                self._text.draw_text(x + dx, y + dy, line, OUTLINE_COLOR, self._width, self._height)
            self._text.draw_text(x, y, line, TEXT_COLOR, self._width, self._height)
        Style.set_blend_mode(Style.BlendMode.DISABLED)
