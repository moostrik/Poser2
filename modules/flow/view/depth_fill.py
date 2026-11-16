"""DepthFill: a depth's holes filled where a mask is, by push-pull over a pyramid of its own textures."""

# Third-party imports
from OpenGL.GL import GL_NEAREST, GL_RG32F

# Local application imports
from modules.gl import Fbo, Texture
from .depth_read import pyramid_sizes
from .shaders import PushPull


class DepthFill:
    """`use` writes the read filled where the mask is above 0 into a target: a pixel with a depth keeps it, one
    without takes the depth around it (`depth_read.push_pull`). The pyramid, every level half the one above down to
    1 × 1, is made with its owner's textures."""

    def __init__(self) -> None:
        self._pushed: list[Fbo] = []                    # the levels below the read, each (depth × weight, weight)
        self._pulled: list[Fbo] = []                    # the same levels pulled, from the coarsest up
        self._shader: PushPull = PushPull()

    def allocate(self, width: int, height: int) -> None:
        self.deallocate()
        sizes: list[tuple[int, int]] = pyramid_sizes(height, width)
        for index, (h, w) in enumerate(sizes):
            # the coarsest level is pulled as it is pushed
            for levels in (self._pushed, self._pulled) if index < len(sizes) - 1 else (self._pushed,):
                fbo: Fbo = Fbo(interpolation=GL_NEAREST)
                fbo.allocate(w, h, GL_RG32F)
                levels.append(fbo)
        self._shader.allocate()

    def deallocate(self) -> None:
        for fbo in (*self._pushed, *self._pulled):
            fbo.deallocate()
        self._pushed.clear()
        self._pulled.clear()
        self._shader.deallocate()

    def use(self, read: Texture, mask: Texture, target: Fbo) -> None:
        """`read` (metres, 0 for none) filled where `mask` is above 0, into `target`, the read's size."""
        if not self._pushed:
            return
        finer: Texture = read
        for level in self._pushed:
            level.begin()
            self._shader.push(finer, first=finer is read)
            level.end()
            finer = level
        coarser: Texture = self._pushed[-1]
        for level, pulled in zip(reversed(self._pushed[:-1]), reversed(self._pulled)):
            pulled.begin()
            self._shader.pull(level, coarser)
            pulled.end()
            coarser = pulled
        target.begin()
        self._shader.last(read, coarser, mask)
        target.end()
