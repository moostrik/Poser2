"""The layout of a string: its glyph quads, placed in pixels and handed to the shader in 0..1 of the area."""

from collections.abc import Callable

import numpy as np

from .font_atlas import GlyphMetrics


def text_layout(text: str, x: float, y: float, ascent: float, glyph: Callable[[str], GlyphMetrics | None],
                width: float, height: float) -> np.ndarray:
    """The string's glyph quads, two triangles each, as interleaved (x, y, u, v) vertices of shape (glyphs * 6, 4).

    `x`, `y` are the text's left and top edge in pixels of an area `width` × `height`; the positions come out in
    0..1 of that area, origin top-left. A glyph sits `ascent - bearing_y` below the top, so a line places the same
    whatever characters it holds; a character the atlas has no glyph for is skipped. Empty when none is drawn.
    """
    quads: list[tuple[float, float, float, float, float, float, float, float]] = []
    cursor_x: float = x
    for char in text:
        metrics: GlyphMetrics | None = glyph(char)
        if metrics is None:
            continue
        quads.append((cursor_x + metrics.bearing_x, y + (ascent - metrics.bearing_y),
                      float(metrics.width), float(metrics.height),
                      metrics.u0, metrics.v0, metrics.u1, metrics.v1))
        cursor_x += metrics.advance
    if not quads:
        return np.empty((0, 4), dtype=np.float32)

    g = np.array(quads, dtype=np.float32)
    x0, y0 = g[:, 0] / width, g[:, 1] / height
    x1, y1 = (g[:, 0] + g[:, 2]) / width, (g[:, 1] + g[:, 3]) / height
    u0, v0, u1, v1 = g[:, 4], g[:, 5], g[:, 6], g[:, 7]
    vertices = np.empty((len(g), 6, 4), dtype=np.float32)
    vertices[:, 0] = np.stack([x0, y0, u0, v0], axis=1)
    vertices[:, 1] = np.stack([x1, y0, u1, v0], axis=1)
    vertices[:, 2] = np.stack([x1, y1, u1, v1], axis=1)
    vertices[:, 3] = np.stack([x0, y0, u0, v0], axis=1)
    vertices[:, 4] = np.stack([x1, y1, u1, v1], axis=1)
    vertices[:, 5] = np.stack([x0, y1, u0, v1], axis=1)
    return vertices.reshape(-1, 4)
