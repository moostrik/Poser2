from time import perf_counter
import math

from OpenGL.GL import glDrawArrays, GL_TRIANGLE_FAN, glClearColor, glClear, GL_COLOR_BUFFER_BIT, glClearDepth, GL_DEPTH_BUFFER_BIT


def clear_color(r: float = 0, g: float = 0, b: float = 0, a: float = 0.0) -> None:
    """Clear framebuffer. Must be called between begin() and end()."""
    glClearColor(r, g, b, a)
    glClear(GL_COLOR_BUFFER_BIT)

def clear_depth(depth: float = 1.0) -> None:
    """Clear depth buffer. Must be called between begin() and end()."""
    glClearDepth(depth)
    glClear(GL_DEPTH_BUFFER_BIT)

def lfo(frequency, phase=0) -> float:
    elapsed_time: float = perf_counter()
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
