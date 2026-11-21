from OpenGL.GL import glViewport

from ...credit import FONT, TITLE_FONT, title_image, wrap_balanced
from modules.gl import BlendState, ChannelOrder, Image, Text
from modules.utils import Color

TEXT_COLOR: Color = Color(1.0, 1.0, 1.0)
OUTLINE_COLOR: Color = Color(0.0, 0.0, 0.0)


class CaptionRenderer:
    """The machine's joke as a meme caption, into the target that is bound: the title SMILE(ai) centered over the
    top and the line in even lines (`wrap_balanced`), centered, white with a black edge; the line under the title
    (the main screen) or along the bottom (a clip's cell). Sizes in the target's pixels."""

    def __init__(self, font_px: int, outline_px: int, margin_px: int) -> None:
        self.font_px: int = font_px
        self.margin_px: int = margin_px             # from the target's edges
        self._title_px: int = round(font_px / TITLE_FONT)      # the title's capitals as tall as the line's
        self._outline_px: int = outline_px
        self._outline: list[tuple[int, int]] = [(x, y) for x in (-outline_px, 0, outline_px)
                                                for y in (-outline_px, 0, outline_px) if x or y]
        self._text: Text = Text()
        self._title: Image = Image(ChannelOrder.BGR)

    @property
    def allocated(self) -> bool:
        return self._text.allocated

    def allocate(self) -> bool:
        """The font and the title; False without the font."""
        if not self._text.allocate(FONT, self.font_px):
            return False
        self._title.set_image(title_image('', self._title_px, outline_ratio=self._outline_px / self._title_px))
        return True

    def deallocate(self) -> None:
        self._text.deallocate()
        self._title.deallocate()

    def update(self) -> None:
        self._title.update()

    def draw(self, line: str, width: int, height: int, title: bool = True, bottom: bool = False) -> None:
        """The caption over a target of `width` x `height`: the title when `title`, and `line` under it, or along
        the bottom with `bottom`; a line too wide for the target wraps."""
        text: Text = self._text
        measure = lambda part: text.measure_text(part)[0]
        _, line_height = text.measure_text("Mg") if text.allocated else (0, 0)
        lines: list[str] = wrap_balanced(line, measure, width - 2 * self.margin_px) if text.allocated else []
        top: int = self.margin_px
        BlendState.set_blend_mode(BlendState.BlendMode.ALPHA)
        if title and self._title.allocated:
            # the viewport's origin is the bottom left
            glViewport((width - self._title.width) // 2, height - top - self._title.height, self._title.width, self._title.height)
            self._title.draw()
            glViewport(0, 0, width, height)
            top += self._title.height
        y: float = height - self.margin_px - len(lines) * line_height if bottom else top
        for i, part in enumerate(lines):
            x: float = (width - measure(part)) / 2
            for dx, dy in self._outline:
                text.draw_text(x + dx, y + i * line_height + dy, part, OUTLINE_COLOR, width, height)
            text.draw_text(x, y + i * line_height, part, TEXT_COLOR, width, height)
        BlendState.set_blend_mode(BlendState.BlendMode.DISABLED)
