"""GPU-based text rendering using VAO and font texture atlas.

Uses freetype-py to generate a font atlas at allocation time.
"""

import ctypes
from pathlib import Path

import numpy as np
from modules.utils import Color, Rect
from OpenGL.GL import GL_ARRAY_BUFFER, GL_DYNAMIC_DRAW, GL_FALSE, GL_FLOAT, glBindBuffer, glBufferData, glDeleteBuffers, glDeleteVertexArrays, glEnableVertexAttribArray, glGenBuffers, glGenVertexArrays, glVertexAttribPointer

from ..state import context, style
from ..shaders.draw import FillRect, Glyphs
from .font_atlas import FontAtlas
from .text_layout import text_layout


# Default font path
DEFAULT_FONT = "data/RobotoMono-Regular.ttf"


class Text:
    """GPU-based text renderer using texture atlas.

    Renders text strings using a pre-built font texture atlas.
    Supports colored text with optional background boxes.

    Example:
        renderer = Text()
        renderer.allocate("path/to/font.ttf", 16)
        renderer.draw_box_text(10, 10, "Hello World",
                               Color(1, 1, 1), Color(0, 0, 0, 0.6),
                               screen_width, screen_height)
        renderer.deallocate()
    """

    # Default padding around text for background box (in pixels)
    BOX_PADDING = 3

    def __init__(self) -> None:
        self._atlas: FontAtlas = FontAtlas()
        self._glyphs: Glyphs = Glyphs()
        self._box: FillRect = FillRect()
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
        self._glyphs.allocate()
        self._box.allocate()

        if not self._glyphs.allocated or not self._box.allocated:
            self.deallocate()
            return False

        # The batch buffer: one VAO + dynamic VBO holding a whole string's glyph quads.
        self._vao = glGenVertexArrays(1)
        self._vbo = glGenBuffers(1)
        previous_vao: int = context().bind_vertex_array(self._vao)
        glBindBuffer(GL_ARRAY_BUFFER, self._vbo)
        stride = 4 * 4                                   # x, y, u, v float32
        glEnableVertexAttribArray(0)
        glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, stride, ctypes.c_void_p(0))
        glEnableVertexAttribArray(1)
        glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, stride, ctypes.c_void_p(2 * 4))
        glBindBuffer(GL_ARRAY_BUFFER, 0)
        context().bind_vertex_array(previous_vao)

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
        self._glyphs.deallocate()
        self._box.deallocate()
        self._allocated = False

    def measure_text(self, text: str) -> tuple[int, int]:
        """Measure text dimensions in pixels.

        Args:
            text: Text string to measure

        Returns:
            Tuple of (width, height) in pixels
        """
        return self._atlas.measure_text(text)

    def draw_text(self, x: float, y: float, text: str,
                  color: Color,
                  screen_width: int, screen_height: int) -> None:
        """Render text at screen position.

        Args:
            x: X position in pixels (left edge)
            y: Y position in pixels (top edge of text box)
            text: Text string to render
            color: text color
            screen_width: Width of render target in pixels
            screen_height: Height of render target in pixels
        """
        if not self._allocated or not text:
            return

        # The whole string in one upload and one draw, so the GL call count stays flat in the text length
        vertices: np.ndarray = text_layout(text, x, y, self._atlas.ascent, self._atlas.get_glyph,
                                           float(screen_width), float(screen_height))
        if not len(vertices):
            return

        style.set_blend_mode(style.BlendMode.ALPHA)

        previous_vao: int = context().bind_vertex_array(self._vao)
        glBindBuffer(GL_ARRAY_BUFFER, self._vbo)
        glBufferData(GL_ARRAY_BUFFER, vertices.nbytes, vertices, GL_DYNAMIC_DRAW)
        glBindBuffer(GL_ARRAY_BUFFER, 0)
        self._glyphs.use(len(vertices), color, self._atlas.texture_id)
        context().bind_vertex_array(previous_vao)

    def draw_box_text(self, x: float, y: float, text: str,
                      color: Color,
                      bg_color: Color,
                      screen_width: int, screen_height: int) -> None:
        """Render text with a background box.

        Args:
            x: X position in pixels (left edge)
            y: Y position in pixels (top edge)
            text: Text string to render
            color: text color
            bg_color: background box color
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

        style.set_blend_mode(style.BlendMode.ALPHA)

        # Draw background box first, in 0..1 of the area
        self._box.use(Rect(box_x / screen_width, box_y / screen_height, box_width / screen_width, box_height / screen_height),
                      bg_color)

        # Draw text with slight vertical offset for better centering
        text_y = y + pad * 0.5
        self.draw_text(x, text_y, text, color, screen_width, screen_height)
