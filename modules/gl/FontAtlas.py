"""Font texture atlas using freetype-py.

Loads a TTF font and generates a texture atlas containing all printable ASCII
characters. Glyph metrics are stored for text rendering.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict

import freetype
import numpy as np
from OpenGL.GL import *  # type: ignore


@dataclass
class GlyphMetrics:
    """Metrics for a single glyph in the atlas."""
    u0: float  # UV left
    v0: float  # UV top
    u1: float  # UV right
    v1: float  # UV bottom
    width: int  # Glyph bitmap width in pixels
    height: int  # Glyph bitmap height in pixels
    bearing_x: int  # Horizontal bearing (offset from cursor)
    bearing_y: int  # Vertical bearing (offset from baseline)
    advance: int  # Horizontal advance to next character


class FontAtlas:
    """Texture atlas containing font glyphs.

    Generates a single texture containing all printable ASCII characters
    (32-126) from a TTF font file. Glyph metrics are stored for positioning
    during text rendering.
    """

    def __init__(self) -> None:
        self.texture_id: int = 0
        self.atlas_width: int = 0
        self.atlas_height: int = 0
        self.glyphs: Dict[str, GlyphMetrics] = {}
        self.line_height: int = 0
        self.ascent: int = 0  # Maximum bearing_y (distance from baseline to top)
        self.allocated: bool = False

    def allocate(self, font_path: str | Path, font_size: int = 16) -> bool:
        """Load font and generate texture atlas.

        Args:
            font_path: Path to TTF font file
            font_size: Font size in pixels

        Returns:
            True if successful, False otherwise
        """
        font_path = Path(font_path)
        if not font_path.exists():
            print(f"FontAtlas: Font file not found: {font_path}")
            return False

        try:
            face = freetype.Face(str(font_path))
            face.set_pixel_sizes(0, font_size)
        except Exception as e:
            print(f"FontAtlas: Failed to load font: {e}")
            return False

        # Printable ASCII characters
        chars = [chr(i) for i in range(32, 127)]

        # First pass: measure all glyphs to determine atlas size
        max_height = 0
        max_bearing_y = 0
        total_width = 0
        glyph_data: list = []

        for char in chars:
            face.load_char(char, freetype.FT_LOAD_RENDER)  # type: ignore
            bitmap = face.glyph.bitmap

            width = bitmap.width
            height = bitmap.rows
            bearing_x = face.glyph.bitmap_left
            bearing_y = face.glyph.bitmap_top
            advance = face.glyph.advance.x >> 6  # Convert from 26.6 fixed point

            # Copy bitmap data
            if width > 0 and height > 0:
                buffer = np.array(bitmap.buffer, dtype=np.uint8).reshape(height, width)
            else:
                buffer = np.zeros((1, 1), dtype=np.uint8)
                width, height = 1, 1

            glyph_data.append((char, buffer, width, height, bearing_x, bearing_y, advance))
            total_width += width + 2  # 2px padding
            max_height = max(max_height, height)
            max_bearing_y = max(max_bearing_y, bearing_y)

        # Calculate atlas dimensions (simple row packing)
        self.atlas_width = min(1024, total_width)
        rows_needed = (total_width // self.atlas_width) + 1
        self.atlas_height = rows_needed * (max_height + 2)

        # Round up to power of 2 for better GPU compatibility
        self.atlas_height = 1 << (self.atlas_height - 1).bit_length()
        self.atlas_width = 1 << (self.atlas_width - 1).bit_length()

        # Create atlas image
        atlas = np.zeros((self.atlas_height, self.atlas_width), dtype=np.uint8)

        # Second pass: pack glyphs into atlas
        x, y = 0, 0
        row_height = 0
        
        # Store ascent for text positioning
        self.ascent = max_bearing_y

        for char, buffer, width, height, bearing_x, bearing_y, advance in glyph_data:
            # Move to next row if needed
            if x + width + 2 > self.atlas_width:
                x = 0
                y += row_height + 2
                row_height = 0

            # Copy glyph bitmap to atlas
            atlas[y:y + height, x:x + width] = buffer

            # Store glyph metrics with UV coordinates
            self.glyphs[char] = GlyphMetrics(
                u0=x / self.atlas_width,
                v0=y / self.atlas_height,
                u1=(x + width) / self.atlas_width,
                v1=(y + height) / self.atlas_height,
                width=width,
                height=height,
                bearing_x=bearing_x,
                bearing_y=bearing_y,
                advance=advance
            )

            x += width + 2
            row_height = max(row_height, height)

        # Store line height for multi-line text
        self.line_height = max_height

        # Upload to GPU
        self.texture_id = glGenTextures(1)
        glBindTexture(GL_TEXTURE_2D, self.texture_id)
        glTexImage2D(
            GL_TEXTURE_2D, 0, GL_R8,
            self.atlas_width, self.atlas_height, 0,
            GL_RED, GL_UNSIGNED_BYTE, atlas
        )
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
        glBindTexture(GL_TEXTURE_2D, 0)

        self.allocated = True
        return True

    def deallocate(self) -> None:
        """Release GPU resources."""
        if self.texture_id:
            glDeleteTextures([self.texture_id])
            self.texture_id = 0
        self.glyphs.clear()
        self.allocated = False

    def get_glyph(self, char: str) -> GlyphMetrics | None:
        """Get metrics for a character."""
        return self.glyphs.get(char)

    def measure_text(self, text: str) -> tuple[int, int]:
        """Measure text dimensions in pixels.

        Args:
            text: Text string to measure

        Returns:
            Tuple of (width, height) in pixels
        """
        if not text:
            return (0, 0)

        width = 0
        for char in text:
            glyph = self.glyphs.get(char)
            if glyph:
                width += glyph.advance

        return (width, self.line_height)
