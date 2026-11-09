from OpenGL.GL import Constant, GL_BGR, GL_BGRA, GL_CLAMP_TO_EDGE, GL_FLOAT, GL_HALF_FLOAT, GL_LINEAR, GL_NONE, GL_ONE, GL_R16F, GL_R32F, GL_R32UI, GL_R8, GL_RED, GL_RED_INTEGER, GL_UNSIGNED_INT, GL_RG, GL_RG16F, GL_RG32F, GL_RG8, GL_RGB, GL_RGB16F, GL_RGB32F, GL_RGB8, GL_RGBA, GL_RGBA16F, GL_RGBA32F, GL_RGBA8, GL_TEXTURE_2D, GL_TEXTURE_BORDER_COLOR, GL_TEXTURE_MAG_FILTER, GL_TEXTURE_MIN_FILTER, GL_TEXTURE_SWIZZLE_A, GL_TEXTURE_SWIZZLE_B, GL_TEXTURE_SWIZZLE_G, GL_TEXTURE_SWIZZLE_R, GL_TEXTURE_WRAP_S, GL_TEXTURE_WRAP_T, GL_UNSIGNED_BYTE, glBindTexture, glClearTexImage, glDeleteTextures, glGenTextures, glGetTexImage, glTexImage2D, glTexParameterf, glTexParameterfv, glTexParameteri
import logging
import numpy as np

logger = logging.getLogger(__name__)


def get_numpy_dtype(data_type: Constant) -> type | None:
    """Convert OpenGL data type to NumPy dtype.

    A 16-bit float texture is read and written as GL_FLOAT, which GL converts to and from half
    floats, so its array is float32 too; only GL_HALF_FLOAT data is float16.

    Args:
        data_type: OpenGL data type (e.g., GL_UNSIGNED_BYTE, GL_FLOAT)

    Returns:
        NumPy dtype (np.uint8, np.float16, np.float32) or None if unsupported
    """
    if data_type == GL_UNSIGNED_BYTE:
        return np.uint8
    if data_type == GL_FLOAT:
        return np.float32
    if data_type == GL_HALF_FLOAT:
        return np.float16
    return None


# The clear colors handed to glClearTexImage, kept for the life of the process: the driver can read a clear
# color after the call returns, and a temporary array's memory is then already reused (measured: a clear
# to 0 left 1.75, 3.75 and the bit pattern of 100 at startup). Zero passes no memory at all.
_CLEAR_VALUES: dict[tuple, np.ndarray] = {}


def clear_value(data_type: Constant, channels: int, color: tuple[float, float, float, float]) -> np.ndarray | None:
    """The clear color for glClearTexImage: None for zero (GL fills zeros without reading memory), else
    an array that is never freed, one per value."""
    values = color[:channels]
    if all(v == 0.0 for v in values):
        return None
    key = (int(data_type), values)
    if key not in _CLEAR_VALUES:
        if data_type == GL_UNSIGNED_BYTE:
            _CLEAR_VALUES[key] = np.array([v * 255 for v in values], dtype=np.uint8)
        elif data_type == GL_HALF_FLOAT:
            _CLEAR_VALUES[key] = np.array(values, dtype=np.float16)
        else:
            _CLEAR_VALUES[key] = np.array(values, dtype=np.float32)
    return _CLEAR_VALUES[key]


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
    if internal_format == GL_R32UI: return GL_RED_INTEGER
    if internal_format == GL_RG8: return GL_RG
    if internal_format == GL_RG16F: return GL_RG
    if internal_format == GL_RG32F: return GL_RG
    logger.warning("internal format not supported")
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
    if internal_format == GL_R32UI: return GL_UNSIGNED_INT
    if internal_format == GL_RG8: return GL_UNSIGNED_BYTE
    if internal_format == GL_RG16F: return GL_FLOAT
    if internal_format == GL_RG32F: return GL_FLOAT
    logger.warning("internal format not supported")
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

    def set_wrap(self, wrap: int, border_color: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0)) -> None:
        """Change wrap mode at runtime (GL thread only)."""
        if not self.allocated:
            return
        self._wrap = wrap
        self._border_color = border_color
        glBindTexture(GL_TEXTURE_2D, self.tex_id)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, wrap)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, wrap)
        glTexParameterfv(GL_TEXTURE_2D, GL_TEXTURE_BORDER_COLOR, border_color)
        glBindTexture(GL_TEXTURE_2D, 0)

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
        dtype = get_numpy_dtype(self.data_type)
        if dtype is None:
            logger.warning("Unsupported data type %s", self.data_type)
            return None

        # Get channel count from format
        channels = get_channel_count(self.format)
        if channels is None:
            logger.warning("Unsupported format %s", self.format)
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

        # The clear color, kept alive (clear_value)
        clear_data = clear_value(self.data_type, channels, (r, g, b, a))

        # Clear texture directly (OpenGL 4.4+) - use self.format directly
        glClearTexImage(self.tex_id, 0, self.format, self.data_type, clear_data)