"""GPU-based text rendering using VAO and font texture atlas.

Replaces the deprecated GLUT-based text rendering with modern OpenGL.
Uses freetype-py to generate a font atlas at allocation time.
"""

from pathlib import Path
from typing import Tuple

from OpenGL.GL import *  # type: ignore

from modules.gl.FontAtlas import FontAtlas
from modules.gl.shaders.TextShader import TextShader
from modules.gl.shaders.BoxShader import BoxShader


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

    @property
    def allocated(self) -> bool:
        return self._allocated

    def allocate(self, font_path: str | Path, font_size: int = 16) -> bool:
        """Initialize text rendering resources.

        Args:
            font_path: Path to TTF font file
            font_size: Font size in pixels

        Returns:
            True if successful, False otherwise
        """
        if self._allocated:
            return True

        # Build font atlas
        if not self._atlas.allocate(font_path, font_size):
            return False

        # Allocate shaders
        self._text_shader.allocate()
        self._box_shader.allocate()

        if not self._text_shader.allocated or not self._box_shader.allocated:
            self.deallocate()
            return False

        self._allocated = True
        return True

    def deallocate(self) -> None:
        """Release all GPU resources."""
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
            y: Y position in pixels (top edge / baseline area)
            text: Text string to render
            color: (r, g, b, a) text color
            screen_width: Width of render target in pixels
            screen_height: Height of render target in pixels
        """
        if not self._allocated or not text:
            return

        # Enable blending for text transparency
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

        screen_size = (float(screen_width), float(screen_height))
        cursor_x = x

        for char in text:
            glyph = self._atlas.get_glyph(char)
            if glyph is None:
                continue

            # Calculate glyph position
            # bearing_y is distance from baseline to top of glyph
            glyph_x = cursor_x + glyph.bearing_x
            glyph_y = y + (self._atlas.line_height - glyph.bearing_y)

            glyph_rect = (glyph_x, glyph_y, float(glyph.width), float(glyph.height))
            uv_rect = (glyph.u0, glyph.v0, glyph.u1, glyph.v1)

            self._text_shader.use(
                glyph_rect, uv_rect, color, screen_size,
                self._atlas.texture_id
            )

            cursor_x += glyph.advance

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

        # Draw text on top
        self.draw_text(x, y, text, color, screen_width, screen_height)


# Legacy compatibility stubs - use TextRenderer class directly instead

def draw_box_string(x: float, y: float, string: str,
                    color: Tuple[float, float, float, float] = (1.0, 1.0, 1.0, 1.0),
                    box_color: Tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.6),
                    big: bool = False) -> None:
    """Legacy compatibility stub - does nothing."""
    pass


def draw_string(x: float, y: float, string: str,
                color: Tuple[float, float, float, float] = (1.0, 1.0, 1.0, 1.0),
                big: bool = False) -> None:
    """Legacy compatibility stub - does nothing."""
    pass


def text_init() -> None:
    """Legacy compatibility stub - does nothing."""
    pass

