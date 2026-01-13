from OpenGL.GL import * # type: ignore
from modules.gl.Texture import Texture, draw_quad, get_data_type
import numpy as np
from threading import Lock
from typing import Literal


def get_format(internal_format, channel_order: Literal['BGR', 'RGB']) -> Constant:
    """Get pixel format for glTexImage2D with channel order support.

    Args:
        internal_format: OpenGL internal format (e.g., GL_RGB8, GL_RGBA8)
        channel_order: Channel order of source image - 'BGR' (OpenCV) or 'RGB' (standard)

    Returns:
        OpenGL format constant (GL_BGR, GL_RGB, GL_BGRA, GL_RGBA, GL_RED, GL_RG)
    """
    if channel_order == 'RGB':
        if internal_format == GL_RGB: return GL_RGB
        if internal_format == GL_RGB8: return GL_RGB
        if internal_format == GL_RGB16F: return GL_RGB
        if internal_format == GL_RGB32F: return GL_RGB
        if internal_format == GL_RGBA: return GL_RGBA
        if internal_format == GL_RGBA8: return GL_RGBA
        if internal_format == GL_RGBA16F: return GL_RGBA
        if internal_format == GL_RGBA32F: return GL_RGBA
    else:  # BGR (for OpenCV compatibility)
        if internal_format == GL_RGB: return GL_BGR
        if internal_format == GL_RGB8: return GL_BGR
        if internal_format == GL_RGB16F: return GL_BGR
        if internal_format == GL_RGB32F: return GL_BGR
        if internal_format == GL_RGBA: return GL_BGRA
        if internal_format == GL_RGBA8: return GL_BGRA
        if internal_format == GL_RGBA16F: return GL_BGRA
        if internal_format == GL_RGBA32F: return GL_BGRA

    # Single/dual channel formats are unaffected by RGB/BGR
    if internal_format == GL_R8: return GL_RED
    if internal_format == GL_R16F: return GL_RED
    if internal_format == GL_R32F: return GL_RED
    if internal_format == GL_RG32F: return GL_RG
    print('GL_Image', 'internal format not supported')
    return GL_NONE

def get_internal_format(image: np.ndarray) -> Constant:
    """Determine OpenGL internal format from NumPy array properties."""
    if image.dtype == np.uint8:
        if len(image.shape) == 2:  # Grayscale image
            return GL_R8
        elif len(image.shape) == 3:
            if image.shape[2] == 3:  # RGB image
                return GL_RGB8
            elif image.shape[2] == 4:  # RGBA image
                return GL_RGBA8
    elif image.dtype == np.float16:
        if len(image.shape) == 2:  # Grayscale image
            return GL_R16F
        elif len(image.shape) == 3:
            if image.shape[2] == 3:  # RGB image
                return GL_RGB16F
            elif image.shape[2] == 4:  # RGBA image
                return GL_RGBA16F
    elif image.dtype == np.float32:
        if len(image.shape) == 2:  # Grayscale image
            return GL_R32F
        elif len(image.shape) == 3:
            if image.shape[2] == 3:  # RGB image
                return GL_RGB32F
            elif image.shape[2] == 4:  # RGBA image
                return GL_RGBA32F

    print('GL_texture', 'image format not supported')
    return GL_NONE


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


class Image(Texture):
    def __init__(self, channel_order: Literal['BGR', 'RGB'] = 'RGB') -> None:
        """Initialize Image texture with specified channel order.

        Args:
            channel_order: Channel order for all images - 'BGR' (OpenCV default) or 'RGB' (standard)
        """
        super().__init__()
        self._image: np.ndarray | None = None
        self._needs_update: bool = False
        self._mutex: Lock = Lock()
        self._channel_order: Literal['BGR', 'RGB'] = channel_order

    def set_image(self, image: np.ndarray) -> None:
        """Set image to be uploaded to texture.

        Args:
            image: NumPy array containing the image data
        """
        with self._mutex:
            self._image = image
            self._needs_update = True

    def update(self) -> None:
        image: None | np.ndarray = None
        needs_update: bool = False
        with self._mutex:
            image = self._image
            needs_update = self._needs_update
            self._needs_update = False

        if needs_update and image is not None:
            self.set_from_image(image)

    def allocate(self, width: int, height: int, internal_format,
                 wrap_s: int = GL_CLAMP_TO_EDGE,
                 wrap_t: int = GL_CLAMP_TO_EDGE,
                 min_filter: int = GL_LINEAR,
                 mag_filter: int = GL_LINEAR) -> None:
        """Allocate OpenGL texture with channel order support.

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
        self.format = get_format(internal_format, self._channel_order)
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

    def set_from_image(self, image: np.ndarray) -> None:
        """Upload a NumPy array to the texture.

        Args:
            image: NumPy array containing the image data
        """
        internal_format: Constant = get_internal_format(image)
        if internal_format == GL_NONE: return
        height: int = image.shape[0]
        width: int = image.shape[1]

        # Reallocate if dimensions or format changed
        if internal_format != self.internal_format or width != self.width or height != self.height:
            if self.allocated: self.deallocate()
            self.allocate(width, height, internal_format)

        if not self.allocated: return

        # Upload image data
        self.bind()
        glTexImage2D(GL_TEXTURE_2D, 0, self.internal_format, width, height, 0, self.format, self.data_type, image)
        self.unbind()

    def read_to_numpy(self) -> np.ndarray | None:
        """Read texture data back to CPU as NumPy array (OpenFrameworks-inspired).

        Useful for debugging, saving captures, or analyzing texture content.
        Respects channel_order to return data in correct BGR/RGB format.

        Returns:
            NumPy array with texture data in the configured channel order, or None if not allocated
        """
        if not self.allocated:
            return None

        # Get NumPy dtype from GL types
        dtype = get_numpy_dtype(self.data_type, self.internal_format)
        if dtype is None:
            print(f"Image.read_to_numpy: Unsupported data type {self.data_type}")
            return None

        # Get channel count from format
        channels = get_channel_count(self.format)
        if channels is None:
            print(f"Image.read_to_numpy: Unsupported format {self.format}")
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

        return pixels

    def draw(self, x, y, w, h) -> None : #override
        self.bind()
        draw_quad(x, y, w, h, True)
        self.unbind()
