from OpenGL.GL import * # type: ignore
from modules.gl.Fbo import Fbo
from modules.gl.Texture import Texture, get_data_type
from modules.gl.Utils import BlitFlip
import numpy as np
from threading import Lock
from typing import Literal


def _get_format_bgr(internal_format, channel_order: Literal['BGR', 'RGB']) -> Constant:
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

def _get_internal_format(image: np.ndarray) -> Constant:
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


class Image(Fbo):
    def __init__(self, channel_order: Literal['BGR', 'RGB'] = 'RGB') -> None:
        """Initialize Image texture with specified channel order.

        Args:
            channel_order: Channel order for input images - 'BGR' (OpenCV default) or 'RGB' (standard)
            The output FBO texture is always in RGB format.
        """
        super().__init__()
        self._image: np.ndarray | None = None
        self._needs_update: bool = False
        self._mutex: Lock = Lock()
        self._channel_order: Literal['BGR', 'RGB'] = channel_order
        self._source: Texture = Texture()  # Private texture for numpy upload

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
        """Allocate source texture and FBO for flipped output.

        Args:
            width: Texture width in pixels
            height: Texture height in pixels
            internal_format: OpenGL internal format (e.g., GL_RGB8, GL_RGBA8)
            wrap_s: Horizontal wrap mode (default: GL_CLAMP_TO_EDGE)
            wrap_t: Vertical wrap mode (default: GL_CLAMP_TO_EDGE)
            min_filter: Minification filter (default: GL_LINEAR)
            mag_filter: Magnification filter (default: GL_LINEAR)
        """
        # Allocate FBO (output texture in RGB format)
        super().allocate(width, height, internal_format, wrap_s, wrap_t, min_filter, mag_filter)
        if not self.allocated:
            return

        # Allocate source texture for numpy upload (with BGR support)
        self._allocate_source(width, height, internal_format, wrap_s, wrap_t, min_filter, mag_filter)

    def _allocate_source(self, width: int, height: int, internal_format,
                         wrap_s: int, wrap_t: int, min_filter: int, mag_filter: int) -> None:
        """Allocate the internal source texture with BGR channel order support."""
        data_type: Constant = get_data_type(internal_format)
        if data_type == GL_NONE:
            return

        self._source.width = width
        self._source.height = height
        self._source.internal_format = internal_format
        self._source.format = _get_format_bgr(internal_format, self._channel_order)
        self._source.data_type = data_type
        self._source.tex_id = glGenTextures(1)

        glBindTexture(GL_TEXTURE_2D, self._source.tex_id)
        glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, wrap_s)
        glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, wrap_t)
        glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, mag_filter)
        glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, min_filter)

        # Set the swizzle mask for grayscale textures
        if self._source.format == GL_RED:
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_SWIZZLE_R, GL_RED)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_SWIZZLE_G, GL_RED)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_SWIZZLE_B, GL_RED)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_SWIZZLE_A, GL_ONE)

        glTexImage2D(GL_TEXTURE_2D, 0, self._source.internal_format, width, height, 0,
                     self._source.format, self._source.data_type, None)
        glBindTexture(GL_TEXTURE_2D, 0)

        self._source.allocated = True

    def deallocate(self) -> None:
        """Deallocate source texture and FBO."""
        self._source.deallocate()
        super().deallocate()

    def set_from_image(self, image: np.ndarray) -> None:
        """Upload a NumPy array to the texture and flip vertically via FBO.

        Args:
            image: NumPy array containing the image data
        """
        internal_format: Constant = _get_internal_format(image)
        if internal_format == GL_NONE:
            return
        height: int = image.shape[0]
        width: int = image.shape[1]

        # Reallocate if dimensions or format changed
        if internal_format != self.internal_format or width != self.width or height != self.height:
            if self.allocated:
                self.deallocate()
            self.allocate(width, height, internal_format)

        if not self.allocated:
            return

        # Upload image data to source texture
        self._source.bind()
        glTexImage2D(GL_TEXTURE_2D, 0, self._source.internal_format, width, height, 0,
                     self._source.format, self._source.data_type, image)
        self._source.unbind()

        # Render flipped to FBO using modern shader
        self.begin()
        BlitFlip.use(self._source, flip_y=True)
        self.end()


