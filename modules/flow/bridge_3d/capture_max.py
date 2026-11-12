"""The larger of several captures' outputs, per pixel, when more than one capture comes in one render tick.

The bridge steps once per capture; the fluid steps once per tick. With two captures in one tick the fluid takes,
for a density and a temperature, per pixel the output of the capture whose amount there is larger, with its own
color or warmth, so a capture's dye is not lost to the next one's; a velocity is the newest's alone.
"""

from modules.gl import Style, SwapFbo, Texture
from ..shaders import Blit
from .shaders import LargerAmount


class CaptureMax:
    """`hold` keeps the first capture's output, `take` keeps per pixel the larger of what it holds and the next
    capture's; `texture` is the result. `by_alpha`: the amount is a density's alpha, else a temperature's size."""

    def __init__(self, by_alpha: bool) -> None:
        self._by_alpha: bool = by_alpha
        self._held: SwapFbo = SwapFbo()
        self._copy_shader: Blit = Blit()
        self._larger_shader: LargerAmount = LargerAmount()

    @property
    def texture(self) -> Texture:
        return self._held

    def allocate(self, width: int, height: int, internal_format: int) -> None:
        self._held.allocate(width, height, internal_format)
        self._copy_shader.allocate()
        self._larger_shader.allocate()

    def deallocate(self) -> None:
        self._held.deallocate()
        self._copy_shader.deallocate()
        self._larger_shader.deallocate()

    def follow(self, source: Texture) -> None:
        """Allocate at `source`'s size and format, and again when they change."""
        held: SwapFbo = self._held
        if source.allocated and (not held.allocated or (held.width, held.height, held.internal_format) !=
                                 (source.width, source.height, source.internal_format)):
            self.deallocate()
            self.allocate(source.width, source.height, source.internal_format)

    def hold(self, source: Texture) -> None:
        """Keep `source`, the first capture's output."""
        self.follow(source)
        Style.push_style()
        Style.set_blend_mode(Style.BlendMode.DISABLED)
        self._held.begin()
        self._copy_shader.use(source)
        self._held.end()
        Style.pop_style()

    def take(self, source: Texture) -> None:
        """Keep, per pixel, the larger of what it holds and `source`."""
        Style.push_style()
        Style.set_blend_mode(Style.BlendMode.DISABLED)
        self._held.swap()
        self._held.begin()
        self._larger_shader.use(self._held.back, source, self._by_alpha)
        self._held.end()
        Style.pop_style()
