from OpenGL.GL import * # type: ignore
import numpy as np
from .View import draw_quad


def get_numpy_dtype(data_type: Constant, internal_format: Constant) -> type | None:
    """Convert OpenGL data type to NumPy dtype.

    Args:
        data_type: OpenGL data type (e.g., GL_UNSIGNED_BYTE, GL_FLOAT)
        internal_format: OpenGL internal format to distinguish float16 vs float32

    Returns:
        NumPy dtype (np.uint8, np.float16, np.float32) or None if unsupported
    """
    if data_type == GL_UNSIGNED_BYTE:
        return np.uint8
    elif data_type == GL_FLOAT:
        # Distinguish float16 vs float32 based on internal format
        if internal_format in (GL_RGB16F, GL_RGBA16F, GL_R16F, GL_RG16F):
            return np.float16
        else:
            return np.float32
    return None


def get_channel_count(format: Constant) -> int | None:
    """Get number of channels from OpenGL format.

    Args:
        format: OpenGL format (e.g., GL_RGB, GL_RGBA, GL_RED)

    Returns:
        Number of channels (1-4) or None if unsupported
    """
    if format in (GL_BGR, GL_RGB):
        return 3
    elif format in (GL_BGRA, GL_RGBA):
        return 4
    elif format == GL_RED:
        return 1
    elif format == GL_RG:
        return 2
    return None


def get_format(internal_format) -> Constant:
    """Get the default pixel format for glTexImage2D (always RGB order).

    Args:
        internal_format: OpenGL internal format (e.g., GL_RGB8, GL_RGBA8)

    Returns:
        OpenGL format constant (GL_RGB, GL_RGBA, GL_RED, GL_RG)
    """
    if internal_format == GL_RGB: return GL_RGB
    if internal_format == GL_RGB8: return GL_RGB
    if internal_format == GL_RGB16F: return GL_RGB
    if internal_format == GL_RGB32F: return GL_RGB
    if internal_format == GL_RGBA: return GL_RGBA
    if internal_format == GL_RGBA8: return GL_RGBA
    if internal_format == GL_RGBA16F: return GL_RGBA
    if internal_format == GL_RGBA32F: return GL_RGBA
    if internal_format == GL_R8: return GL_RED
    if internal_format == GL_R16F: return GL_RED
    if internal_format == GL_R32F: return GL_RED
    if internal_format == GL_RG8: return GL_RG
    if internal_format == GL_RG16F: return GL_RG
    if internal_format == GL_RG32F: return GL_RG
    print('GL_texture', 'internal format not supported')
    return GL_RGB  # fallback


def get_data_type(internal_format) -> Constant:
    if internal_format == GL_RGB: return GL_UNSIGNED_BYTE
    if internal_format == GL_RGB8: return GL_UNSIGNED_BYTE
    if internal_format == GL_RGB16F: return GL_FLOAT
    if internal_format == GL_RGB32F: return GL_FLOAT
    if internal_format == GL_RGBA: return GL_UNSIGNED_BYTE
    if internal_format == GL_RGBA8: return GL_UNSIGNED_BYTE
    if internal_format == GL_RGBA16F: return GL_FLOAT
    if internal_format == GL_RGBA32F: return GL_FLOAT
    if internal_format == GL_R8: return GL_UNSIGNED_BYTE
    if internal_format == GL_R16F: return GL_FLOAT
    if internal_format == GL_R32F: return GL_FLOAT
    if internal_format == GL_RG8: return GL_UNSIGNED_BYTE
    if internal_format == GL_RG16F: return GL_FLOAT
    if internal_format == GL_RG32F: return GL_FLOAT
    print('GL_texture', 'internal format not supported')
    return GL_NONE




class Texture():
    def __init__(self) -> None :
        self.allocated = False
        self.width: int = 0
        self.height: int = 0
        self.internal_format: Constant = GL_NONE
        self.format: Constant = GL_NONE
        self.data_type: Constant = GL_NONE
        self.tex_id = 0

    @property
    def texture(self) -> 'Texture':
        """Convenience property for interface compatibility."""
        return self

    def allocate(self, width: int, height: int, internal_format,
                 wrap_s: int = GL_CLAMP_TO_EDGE,
                 wrap_t: int = GL_CLAMP_TO_EDGE,
                 min_filter: int = GL_LINEAR,
                 mag_filter: int = GL_LINEAR) -> None :
        """Allocate OpenGL texture with specified dimensions and format.

        Args:
            width: Texture width in pixels
            height: Texture height in pixels
            internal_format: OpenGL internal format (e.g., GL_RGB8, GL_RGBA8)
            wrap_s: Horizontal wrap mode (default: GL_CLAMP_TO_EDGE)
            wrap_t: Vertical wrap mode (default: GL_CLAMP_TO_EDGE)
            min_filter: Minification filter (default: GL_LINEAR)
            mag_filter: Magnification filter (default: GL_LINEAR)
        """
        data_type: Constant = get_data_type(internal_format)
        if data_type == GL_NONE: return

        self.width = width
        self.height = height
        self.internal_format = internal_format
        self.format = get_format(internal_format)
        self.data_type = data_type
        self.tex_id: int = glGenTextures(1)

        glBindTexture(GL_TEXTURE_2D, self.tex_id)
        glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, wrap_s)
        glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, wrap_t)
        glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, mag_filter)
        glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, min_filter)

        # Set the swizzle mask for the texture to draw it as grayscale
        if self.format == GL_RED:
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_SWIZZLE_R, GL_RED)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_SWIZZLE_G, GL_RED)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_SWIZZLE_B, GL_RED)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_SWIZZLE_A, GL_ONE)

        glTexImage2D(GL_TEXTURE_2D, 0, self.internal_format, self.width, self.height, 0, self.format, self.data_type, None)
        glBindTexture(GL_TEXTURE_2D, 0)

        self.allocated = True

    def deallocate(self) -> None :
        if not self.allocated: return
        self.allocated = False
        self.width = 0
        self.height = 0
        self.internal_format = GL_NONE
        self.format = GL_NONE
        self.data_type = GL_NONE
        glDeleteTextures(1, [self.tex_id])
        self.tex_id = 0

    def bind(self) -> None :
        glBindTexture(GL_TEXTURE_2D, self.tex_id)

    def unbind(self) -> None :
        glBindTexture(GL_TEXTURE_2D, 0)

    def draw(self, x, y, w, h) -> None :
        self.bind()
        draw_quad(x, y, w, h)
        self.unbind()

    def read_to_numpy(self, flip: bool = True) -> np.ndarray | None:
        """Read texture data back to CPU as NumPy array.

        Useful for debugging, saving captures, or analyzing texture content.

        Returns:
            NumPy array with texture data, or None if not allocated
        """
        if not self.allocated:
            return None

        # Get NumPy dtype from GL types
        dtype = get_numpy_dtype(self.data_type, self.internal_format)
        if dtype is None:
            print(f"Texture.read_to_numpy: Unsupported data type {self.data_type}")
            return None

        # Get channel count from format
        channels = get_channel_count(self.format)
        if channels is None:
            print(f"Texture.read_to_numpy: Unsupported format {self.format}")
            return None

        # Allocate output array
        if channels == 1:
            pixels = np.zeros((self.height, self.width), dtype=dtype)
        else:
            pixels = np.zeros((self.height, self.width, channels), dtype=dtype)

        # Read texture data from GPU
        self.bind()
        glGetTexImage(GL_TEXTURE_2D, 0, self.format, self.data_type, pixels)
        self.unbind()

        if flip:
            pixels = np.flipud(pixels)
        return pixels
