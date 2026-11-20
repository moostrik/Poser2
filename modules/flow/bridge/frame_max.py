"""The larger of several frames' outputs, per pixel, when more than one frame comes in one render tick.

The bridge steps once per frame; the fluid steps once per tick. With two frames in one tick the fluid takes,
for a density and a temperature, per pixel the output of the frame whose amount there is larger, with its own
color or warmth, so a frame's dye is not lost to the next one's; a velocity is the newest's alone. The density's
color goes with its amount: both are picked by the amount.
"""

from OpenGL.GL import Constant

from modules.gl import Style, SwapFbo, Texture
from .shaders import LargerAmount


class FrameMax:
    """`hold` keeps the first frame's output, `take` keeps per pixel the larger of what it holds and the next
    frame's; `texture` is the result. The amount is the size of the output's red, or of the amounts `take` is
    given, which the color of an amount is picked by."""

    def __init__(self) -> None:
        self._held: SwapFbo = SwapFbo()
        self._larger_shader: LargerAmount = LargerAmount()

    @property
    def texture(self) -> Texture:
        return self._held

    def allocate(self, width: int, height: int, internal_format: Constant) -> None:
        self._held.allocate(width, height, internal_format)
        self._larger_shader.allocate()

    def deallocate(self) -> None:
        self._held.deallocate()
        self._larger_shader.deallocate()

    def follow(self, source: Texture) -> None:
        """Allocate at `source`'s size and format, and again when they change."""
        held: SwapFbo = self._held
        if source.allocated and (not held.allocated or (held.width, held.height, held.internal_format) !=
                                 (source.width, source.height, source.internal_format)):
            self.deallocate()
            self.allocate(source.width, source.height, source.internal_format)

    def hold(self, source: Texture) -> None:
        """Keep `source`, the first frame's output."""
        self.follow(source)
        Style.push_style()
        Style.set_blend_mode(Style.BlendMode.DISABLED)
        self._held.begin()
        source.draw()
        self._held.end()
        Style.pop_style()

    def take(self, source: Texture, amounts: tuple[Texture, Texture] | None = None) -> None:
        """Keep, per pixel, the larger of what it holds and `source`: by `amounts`, the held one's and the
        source's, or by the textures themselves without them."""
        Style.push_style()
        Style.set_blend_mode(Style.BlendMode.DISABLED)
        self._held.swap()
        held_amount, source_amount = amounts if amounts is not None else (self._held.back, source)
        self._held.begin()
        self._larger_shader.use(self._held.back, source, held_amount, source_amount)
        self._held.end()
        Style.pop_style()
