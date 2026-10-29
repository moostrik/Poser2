"""Render passes shared by the layers."""

from modules.gl import Blit, Fbo, SwapFbo, Texture, shaders


def copy(src: Texture, dst: Fbo) -> None:
    """Draw src stretched over dst (the current buffer of a SwapFbo)."""
    dst.begin()
    Blit.use(src)
    dst.end()


def blur(swap: SwapFbo, blur_h: shaders.BlurH, blur_v: shaders.BlurV, radius: float) -> None:
    """Separable blur of the current buffer, in place."""
    swap.swap()
    swap.begin()
    blur_h.use(swap.back_texture, radius)
    swap.end()
    swap.swap()
    swap.begin()
    blur_v.use(swap.back_texture, radius)
    swap.end()
