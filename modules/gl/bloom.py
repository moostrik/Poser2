"""Bloom: the glow of an image's bright parts, over a dual-filter chain of half-size targets."""

from OpenGL.GL import GL_RGBA16F

from .fbo import Fbo
from .shaders import DualFilter
from .texture import Texture
from . import style as Style

LEVELS: int = 5     # half sizes down: the glow reaches about 2^LEVELS pixels


class Bloom:
    """`process` draws the glow of `src`'s parts brighter than `threshold` into `texture`, at `src`'s size: down a
    chain of half sizes and up again, each level's blur added on the way up. Draw `texture` over the image with
    `Style.BlendMode.ADD`."""

    def __init__(self, levels: int = LEVELS) -> None:
        self._levels: int = levels
        self._down: list[Fbo] = [Fbo() for _ in range(levels)]
        self._up: list[Fbo] = [Fbo() for _ in range(levels)]     # _up[i] is _down[i - 1]'s size, _up[0] src's
        self._shader: DualFilter = DualFilter()

    @property
    def texture(self) -> Texture:
        return self._up[0]

    @property
    def allocated(self) -> bool:
        return self._up[0].allocated

    def allocate(self, width: int, height: int) -> None:
        sizes: list[tuple[int, int]] = [(width, height)]
        for _ in range(self._levels):
            w, h = sizes[-1]
            sizes.append((max(w // 2, 1), max(h // 2, 1)))
        for i in range(self._levels):
            self._down[i].allocate(*sizes[i + 1], GL_RGBA16F)
            self._up[i].allocate(*sizes[i], GL_RGBA16F)
        self._shader.allocate()

    def deallocate(self) -> None:
        for fbo in self._down + self._up:
            fbo.deallocate()
        self._shader.deallocate()

    def process(self, src: Texture, threshold: float = 0.8) -> None:
        if not self.allocated or not src.allocated:
            return
        Style.push_style()
        Style.set_blend_mode(Style.BlendMode.DISABLED)
        source: Texture = src
        for i, target in enumerate(self._down):
            target.begin()
            self._shader.down(source, threshold if i == 0 else -1.0)
            target.end()
            source = target
        for i in range(self._levels - 1, -1, -1):
            self._up[i].begin()
            Style.set_blend_mode(Style.BlendMode.DISABLED)
            self._shader.up(source)
            if i > 0:
                Style.set_blend_mode(Style.BlendMode.SUM)       # the level's own blur added to what came up
                self._down[i - 1].draw()
            self._up[i].end()
            source = self._up[i]
        Style.pop_style()
