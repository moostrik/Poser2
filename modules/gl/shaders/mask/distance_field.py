"""DistanceField: each pixel's distance to a mask, by jump flooding, for outlines and glows of any width."""

from OpenGL.GL import GL_R32F, GL_RG32F

from ... import Fbo, Style, SwapFbo, Texture
from .jump_flood import JumpFlood


class DistanceField:
    """`process` writes into `texture` (R32F) each pixel's distance in pixels to the nearest pixel of `mask` whose red
    reaches `threshold`, 0 on the mask, -1 with no mask at all; log2 of the larger side in passes, plus one."""

    def __init__(self) -> None:
        self._map: SwapFbo = SwapFbo()
        self._distance: Fbo = Fbo()
        self._shader: JumpFlood = JumpFlood()

    @property
    def texture(self) -> Texture:
        return self._distance

    @property
    def allocated(self) -> bool:
        return self._distance.allocated

    def allocate(self, width: int, height: int) -> None:
        self._map.allocate(width, height, GL_RG32F)
        self._distance.allocate(width, height, GL_R32F)
        self._shader.allocate()

    def deallocate(self) -> None:
        self._map.deallocate()
        self._distance.deallocate()
        self._shader.deallocate()

    def process(self, mask: Texture, threshold: float = 0.5) -> None:
        if not self.allocated or not mask.allocated:
            return
        Style.push_style()
        Style.set_blend_mode(Style.BlendMode.DISABLED)
        self._map.begin()
        self._shader.seed(mask, threshold)
        self._map.end()
        jump: int = 1 << max(max(self._distance.width, self._distance.height) - 1, 1).bit_length() - 1
        while jump >= 1:
            self._map.swap()
            self._map.begin()
            self._shader.step(self._map.back, jump)
            self._map.end()
            jump //= 2
        self._map.swap()            # the extra pass at jump 1 (JFA+1) mends most of the algorithm's misses
        self._map.begin()
        self._shader.step(self._map.back, 1)
        self._map.end()
        self._distance.begin()
        self._shader.distance(self._map)
        self._distance.end()
        Style.pop_style()
