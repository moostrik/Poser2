"""GPU-based text rendering using VAO and font texture atlas.

Replaces the deprecated GLUT-based text rendering with modern OpenGL.
Uses freetype-py to generate a font atlas at allocation time.
"""

import ctypes
from pathlib import Path
from typing import Tuple

import numpy as np
from OpenGL.GL import *  # type: ignore

from .FontAtlas import FontAtlas
from .shaders.TextShader import TextShader
from .shaders.BoxShader import BoxShader


# Default font path
DEFAULT_FONT = "data/RobotoMono-Regular.ttf"


class Text:
    """GPU-based text renderer using texture atlas.

    Renders text strings using a pre-built font texture atlas.
    Supports colored text with optional background boxes.

    Example:
        renderer = TextRenderer()
        renderer.allocate("path/to/font.ttf", 16)
        renderer.draw_box_text(10, 10, "Hello World",
                               (1, 1, 1, 1), (0, 0, 0, 0.6),
                               screen_width, screen_height)
        renderer.deallocate()
    """

    # Default padding around text for background box (in pixels)
    BOX_PADDING = 3

    def __init__(self) -> None:
        self._atlas: FontAtlas = FontAtlas()
        self._text_shader: TextShader = TextShader()
        self._box_shader: BoxShader = BoxShader()
        self._allocated: bool = False
        self._vao: int = 0      # the string batch: interleaved (x, y, u, v), 6 vertices per glyph
        self._vbo: int = 0

    @property
    def allocated(self) -> bool:
        return self._allocated

    def allocate(self, font_path: str | Path | None = None, font_size: int = 16) -> bool:
        """Initialize text rendering resources.

        Args:
            font_path: Path to TTF font file (defaults to RobotoMono-Regular.ttf)
            font_size: Font size in pixels

        Returns:
            True if successful, False otherwise
        """
        if self._allocated:
            return True

        # Default to RobotoMono-Regular if no font specified
        if font_path is None:
            font_path = DEFAULT_FONT

        # Build font atlas
        if not self._atlas.allocate(font_path, font_size):
            return False

        # Allocate shaders
        self._text_shader.allocate()
        self._box_shader.allocate()

        if not self._text_shader.allocated or not self._box_shader.allocated:
            self.deallocate()
            return False

        # The batch buffer: one VAO + dynamic VBO holding a whole string's glyph quads.
        previous_vao = glGetIntegerv(GL_VERTEX_ARRAY_BINDING)
        self._vao = glGenVertexArrays(1)
        self._vbo = glGenBuffers(1)
        glBindVertexArray(self._vao)
        glBindBuffer(GL_ARRAY_BUFFER, self._vbo)
        stride = 4 * 4                                   # x, y, u, v float32
        glEnableVertexAttribArray(0)
        glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, stride, ctypes.c_void_p(0))
        glEnableVertexAttribArray(1)
        glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, stride, ctypes.c_void_p(2 * 4))
        glBindBuffer(GL_ARRAY_BUFFER, 0)
        glBindVertexArray(previous_vao)

        self._allocated = True
        return True

    def deallocate(self) -> None:
        """Release all GPU resources."""
        if self._vao:
            glDeleteVertexArrays(1, [self._vao])
            self._vao = 0
        if self._vbo:
            glDeleteBuffers(1, [self._vbo])
            self._vbo = 0
        self._atlas.deallocate()
        self._text_shader.deallocate()
        self._box_shader.deallocate()
        self._allocated = False

    def measure_text(self, text: str) -> Tuple[int, int]:
        """Measure text dimensions in pixels.

        Args:
            text: Text string to measure

        Returns:
            Tuple of (width, height) in pixels
        """
        return self._atlas.measure_text(text)

    def draw_text(self, x: float, y: float, text: str,
                  color: Tuple[float, float, float, float],
                  screen_width: int, screen_height: int) -> None:
        """Render text at screen position.

        Args:
            x: X position in pixels (left edge)
            y: Y position in pixels (top edge of text box)
            text: Text string to render
            color: (r, g, b, a) text color
            screen_width: Width of render target in pixels
            screen_height: Height of render target in pixels
        """
        if not self._allocated or not text:
            return

        # Collect the glyph quads (no GL yet): x, y, width, height, u0, v0, u1, v1 per glyph.
        # Glyphs are positioned from the top edge (y) down by (ascent - bearing_y), so text
        # places consistently regardless of which characters are used.
        quads: list[tuple[float, float, float, float, float, float, float, float]] = []
        cursor_x = x
        ascent = self._atlas.ascent
        for char in text:
            glyph = self._atlas.get_glyph(char)
            if glyph is None:
                continue
            quads.append((cursor_x + glyph.bearing_x, y + (ascent - glyph.bearing_y),
                          float(glyph.width), float(glyph.height),
                          glyph.u0, glyph.v0, glyph.u1, glyph.v1))
            cursor_x += glyph.advance
        if not quads:
            return

        # Two triangles per glyph, interleaved (x, y, u, v) — the whole string in one upload
        # and one draw, so the GL call count stays flat in the text length.
        g = np.array(quads, dtype=np.float32)
        x0, y0 = g[:, 0], g[:, 1]
        x1, y1 = x0 + g[:, 2], y0 + g[:, 3]
        u0, v0, u1, v1 = g[:, 4], g[:, 5], g[:, 6], g[:, 7]
        vertices = np.empty((len(g), 6, 4), dtype=np.float32)
        vertices[:, 0] = np.stack([x0, y0, u0, v0], axis=1)
        vertices[:, 1] = np.stack([x1, y0, u1, v0], axis=1)
        vertices[:, 2] = np.stack([x1, y1, u1, v1], axis=1)
        vertices[:, 3] = np.stack([x0, y0, u0, v0], axis=1)
        vertices[:, 4] = np.stack([x1, y1, u1, v1], axis=1)
        vertices[:, 5] = np.stack([x0, y1, u0, v1], axis=1)

        # Enable blending for text transparency
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

        previous_vao = glGetIntegerv(GL_VERTEX_ARRAY_BINDING)
        glBindVertexArray(self._vao)
        glBindBuffer(GL_ARRAY_BUFFER, self._vbo)
        glBufferData(GL_ARRAY_BUFFER, vertices.nbytes, vertices, GL_DYNAMIC_DRAW)
        glBindBuffer(GL_ARRAY_BUFFER, 0)
        self._text_shader.use(len(g) * 6, color, (float(screen_width), float(screen_height)),
                              self._atlas.texture_id)
        glBindVertexArray(previous_vao)

    def draw_box_text(self, x: float, y: float, text: str,
                      color: Tuple[float, float, float, float],
                      bg_color: Tuple[float, float, float, float],
                      screen_width: int, screen_height: int) -> None:
        """Render text with a background box.

        Args:
            x: X position in pixels (left edge)
            y: Y position in pixels (top edge)
            text: Text string to render
            color: (r, g, b, a) text color
            bg_color: (r, g, b, a) background box color
            screen_width: Width of render target in pixels
            screen_height: Height of render target in pixels
        """
        if not self._allocated or not text:
            return

        # Measure text for background box
        text_width, text_height = self.measure_text(text)

        # Calculate box dimensions with padding
        pad = self.BOX_PADDING
        box_x = x - pad
        box_y = y - pad
        box_width = text_width + pad * 2
        box_height = text_height + pad * 2

        # Enable blending
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

        screen_size = (float(screen_width), float(screen_height))

        # Draw background box first
        self._box_shader.use(
            (box_x, box_y, box_width, box_height),
            bg_color, screen_size
        )

        # Draw text with slight vertical offset for better centering
        text_y = y + pad * 0.5
        self.draw_text(x, text_y, text, color, screen_width, screen_height)
