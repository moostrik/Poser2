"""Render passes shared by the layers."""

from modules.gl import Fbo, SwapFbo, Texture, shaders


def copy(src: Texture, dst: Fbo) -> None:
    """Draw src stretched over dst (the current buffer of a SwapFbo)."""
    dst.begin()
    src.draw()
    dst.end()


def blur(swap: SwapFbo, shader: shaders.Blur, sigma: float) -> None:
    """Gaussian blur of the current buffer, in place: `sigma` in pixels."""
    swap.swap()
    swap.begin()
    shader.use(swap.back, True, sigma)
    swap.end()
    swap.swap()
    swap.begin()
    shader.use(swap.back, False, sigma)
    swap.end()
