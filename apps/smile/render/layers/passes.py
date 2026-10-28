"""Render passes shared by the layers."""

from modules.gl import Blit, Fbo, SwapFbo, Texture
from modules.gl.shaders.convenience.BlurH import BlurH
from modules.gl.shaders.convenience.BlurV import BlurV


def copy(src: Texture, dst: Fbo) -> None:
    """Draw src stretched over dst (the current buffer of a SwapFbo)."""
    dst.begin()
    Blit.use(src)
    dst.end()


def blur(swap: SwapFbo, blur_h: BlurH, blur_v: BlurV, radius: float) -> None:
    """Separable blur of the current buffer, in place."""
    swap.swap()
    swap.begin()
    blur_h.use(swap.back_texture, radius)
    swap.end()
    swap.swap()
    swap.begin()
    blur_v.use(swap.back_texture, radius)
    swap.end()
