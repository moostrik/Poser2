from OpenGL.GL import * # type: ignore
import numpy as np


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
    def __init__(self,
                 interpolation: int = GL_LINEAR,
                 wrap: int = GL_CLAMP_TO_EDGE,
                 border_color: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 1.0)) -> None:
        """Initialize texture with configuration.

        Args:
            interpolation: Filter mode for min/mag (GL_LINEAR or GL_NEAREST)
            wrap: Wrap mode for both axes (GL_CLAMP_TO_EDGE, GL_REPEAT, GL_MIRRORED_REPEAT, GL_CLAMP_TO_BORDER)
            border_color: RGBA color when wrap is GL_CLAMP_TO_BORDER
        """
        self.allocated = False
        self.width: int = 0
        self.height: int = 0
        self.internal_format: Constant = GL_NONE
        self.format: Constant = GL_NONE
        self.data_type: Constant = GL_NONE
        self.tex_id = 0
        # Texture configuration
        self._interpolation: int = interpolation
        self._wrap: int = wrap
        self._border_color: tuple[float, float, float, float] = border_color

    @property
    def texture(self) -> 'Texture':
        """Convenience property for interface compatibility."""
        return self

    def allocate(self, width: int, height: int, internal_format) -> None:
        """Allocate OpenGL texture with specified dimensions and format.

        Args:
            width: Texture width in pixels
            height: Texture height in pixels
            internal_format: OpenGL internal format (e.g., GL_RGB8, GL_RGBA8)
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
        glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, self._wrap)
        glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, self._wrap)
        glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, self._interpolation)
        glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, self._interpolation)
        glTexParameterfv(GL_TEXTURE_2D, GL_TEXTURE_BORDER_COLOR, self._border_color)

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

    def clear(self, r: float = 0.0, g: float = 0.0, b: float = 0.0, a: float = 0.0) -> None:
        """Clear texture to specified color using glClearTexImage (OpenGL 4.4+).

        Args:
            r: Red component [0.0, 1.0]
            g: Green component [0.0, 1.0]
            b: Blue component [0.0, 1.0]
            a: Alpha component [0.0, 1.0]
        """
        if not self.allocated:
            return

        # Get channel count to know how many components to pass
        channels = get_channel_count(self.format)
        if channels is None:
            return

        # Prepare clear color data based on data_type
        if self.data_type == GL_UNSIGNED_BYTE:
            # Convert [0.0, 1.0] to [0, 255]
            clear_data = np.array([r * 255, g * 255, b * 255, a * 255][:channels], dtype=np.uint8)
        elif self.data_type == GL_HALF_FLOAT:
            clear_data = np.array([r, g, b, a][:channels], dtype=np.float16)
        else:  # GL_FLOAT
            clear_data = np.array([r, g, b, a][:channels], dtype=np.float32)

        # Clear texture directly (OpenGL 4.4+) - use self.format directly
        glClearTexImage(self.tex_id, 0, self.format, self.data_type, clear_data)