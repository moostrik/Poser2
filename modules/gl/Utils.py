from time import time
import math

from OpenGL.GL import glDrawArrays, GL_TRIANGLE_FAN, glClearColor, glClear, GL_COLOR_BUFFER_BIT, glClearDepth, GL_DEPTH_BUFFER_BIT  # type: ignore

# -----------------------------------------------------------------------------
# Quad drawing (VAO must be bound by WindowManager before calling)
# -----------------------------------------------------------------------------

def draw_quad() -> None:
    """Draw fullscreen quad. Assumes VAO already bound by WindowManager."""
    glDrawArrays(GL_TRIANGLE_FAN, 0, 4)

def clear_color(r: float = 0, g: float = 0, b: float = 0, a: float = 0.0) -> None:
    """Clear framebuffer. Must be called between begin() and end()."""
    glClearColor(r, g, b, a)
    glClear(GL_COLOR_BUFFER_BIT)

def clear_depth(depth: float = 1.0) -> None:
    """Clear depth buffer. Must be called between begin() and end()."""
    glClearDepth(depth)
    glClear(GL_DEPTH_BUFFER_BIT)

# -----------------------------------------------------------------------------
# FPS utilities
# -----------------------------------------------------------------------------


class FpsCounter:
    def __init__(self, numSamples = 120) -> None:
        self._times: list[float] = []
        self.numSamples: int = numSamples

    def tick(self) -> None:
        now: float = time()
        self._times.append(now)
        if len(self._times) > self.numSamples:
            self._times.pop(0)

    def get_fps(self) -> int:
        if len(self._times) < 2:
            return 0
        diff: float = self._times[-1] - self._times[0]
        if diff == 0:
            return 0
        return int(math.floor(len(self._times) / diff))

    def get_min_fps(self) -> int:
        if len(self._times) < 2:
            return 0
        max_diff: float = max(self._times[i+1] - self._times[i] for i in range(len(self._times) - 1))
        if max_diff == 0:
            return 0
        return int(math.floor(1.0 / max_diff))

def lfo(frequency, phase=0) -> float:
    elapsed_time: float = time()
    lfo_value: float = 0.5 + 0.5 * math.sin(2 * math.pi * frequency * elapsed_time + phase)
    return lfo_value

def fit(src_width: int | float, src_height: int | float, dst_width: int | float, dst_height: int | float) -> tuple[float, float, float, float]:
    src_ratio: float = float(src_width) / float(src_height)
    dst_ratio: float = float(dst_width) / float(dst_height)

    x: float
    y: float
    width: float
    height: float

    if dst_ratio > src_ratio:
        height = dst_height
        width = height * src_ratio
    else:
        width = dst_width
        height = width / src_ratio

    x = (dst_width - width) / 2.0
    y = (dst_height - height) / 2.0

    return (x, y, width, height)

def fill(src_width: int | float, src_height: int | float, dst_width: int | float, dst_height: int | float) -> tuple[float, float, float, float]:
    src_ratio: float = float(src_width) / float(src_height)
    dst_ratio: float = float(dst_width) / float(dst_height)

    x: float
    y: float
    width: float
    height: float

    if dst_ratio < src_ratio:
        height = dst_height
        width = height * src_ratio
    else:
        width = dst_width
        height = width / src_ratio

    x = (dst_width - width) / 2.0
    y = (dst_height - height) / 2.0

    return (x, y, width, height)

class Blit:
    """Singleton lazy-loaded Blit shader for drawing textures fullscreen."""
    _shader = None

    @staticmethod
    def use(texture) -> None:
        """Draw texture fullscreen to current viewport/FBO."""
        if Blit._shader is None:
            from modules.gl.shaders.Blit import Blit as BlitShader
            Blit._shader = BlitShader()
            Blit._shader.allocate()
        Blit._shader.use(texture)

class BlitRect:
    """Singleton lazy-loaded BlitRect shader."""
    _shader = None

    @staticmethod
    def use(texture, rect_x: float, rect_y: float, rect_w: float, rect_h: float) -> None:
        if BlitRect._shader is None:
            from modules.gl.shaders.BlitRect import BlitRect as BlitRectShader
            BlitRect._shader = BlitRectShader()
            BlitRect._shader.allocate()
        BlitRect._shader.use(texture, rect_x, rect_y, rect_w, rect_h)

class BlitRegion:
    """Singleton lazy-loaded BlitRegion shader."""
    _shader = None

    @staticmethod
    def use(texture, x: float, y: float, w: float, h: float) -> None:
        if BlitRegion._shader is None:
            from modules.gl.shaders.BlitRegion import BlitRegion as BlitRegionShader
            BlitRegion._shader = BlitRegionShader()
            BlitRegion._shader.allocate()
        BlitRegion._shader.use(texture, x, y, w, h)

class BlitFlip:
    """Singleton lazy-loaded BlitFlip shader for flipped texture blitting."""
    _shader = None

    @staticmethod
    def use(texture, flip_x: bool = False, flip_y: bool = False) -> None:
        """Draw texture fullscreen with optional X/Y flip."""
        if BlitFlip._shader is None:
            from modules.gl.shaders.BlitFlip import BlitFlip as BlitFlipShader
            BlitFlip._shader = BlitFlipShader()
            BlitFlip._shader.allocate()
        BlitFlip._shader.use(texture, flip_x, flip_y)
