from OpenGL.GL import * # type: ignore
from modules.gl.Fbo import Fbo
from modules.gl.Texture import Texture, get_data_type
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


class Image(Fbo):
    """GPU texture for CPU-uploaded images with automatic GPU flip for uniform orientation.

    Extends Fbo to enable GPU-side vertical flip during upload, ensuring all textures
    have consistent V orientation (top at V=1) regardless of source format.
    """

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
        # Staging texture for CPU upload before GPU flip
        self._staging: Texture = Texture()

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
        """Allocate FBO and staging texture for GPU-flipped image upload.

        Args:
            width: Texture width in pixels
            height: Texture height in pixels
            internal_format: OpenGL internal format (e.g., GL_RGB8, GL_RGBA8)
            wrap_s: Horizontal wrap mode (default: GL_CLAMP_TO_EDGE)
            wrap_t: Vertical wrap mode (default: GL_CLAMP_TO_EDGE)
            min_filter: Minification filter (default: GL_LINEAR)
            mag_filter: Magnification filter (default: GL_LINEAR)
        """
        # Allocate staging texture for CPU upload (uses channel order format)
        self._staging.allocate(width, height, internal_format, wrap_s, wrap_t, min_filter, mag_filter)
        # Override staging format for BGR support
        self._staging.format = get_format(internal_format, self._channel_order)

        # Allocate FBO (self) for flipped result - uses parent Fbo.allocate
        super().allocate(width, height, internal_format, wrap_s, wrap_t, min_filter, mag_filter)

    def deallocate(self) -> None:
        """Deallocate FBO and staging texture."""
        self._staging.deallocate()
        super().deallocate()

    def set_from_image(self, image: np.ndarray) -> None:
        """Upload a NumPy array to the texture with GPU flip for uniform orientation.

        The image is first uploaded to a staging texture, then blitted with V flip
        to this FBO, ensuring top of image is at V=1 (matching FBO-rendered content).

        Args:
            image: NumPy array containing the image data
        """
        from modules.gl.shaders.FboBlit import FboBlit

        internal_format: Constant = get_internal_format(image)
        if internal_format == GL_NONE: return
        height: int = image.shape[0]
        width: int = image.shape[1]

        # Reallocate if dimensions or format changed
        if internal_format != self.internal_format or width != self.width or height != self.height:
            if self.allocated: self.deallocate()
            self.allocate(width, height, internal_format)

        if not self.allocated: return

        # Step 1: Upload to staging texture (CPU -> GPU, no flip)
        self._staging.bind()
        glTexImage2D(GL_TEXTURE_2D, 0, self._staging.internal_format, width, height, 0,
                     self._staging.format, self._staging.data_type, image)
        self._staging.unbind()

        # Step 2: Blit with V flip to self (FBO) - normalizes orientation on GPU
        shader = FboBlit()
        if not shader.allocated:
            shader.allocate()
        shader.use(self, self._staging, flip_v=True)