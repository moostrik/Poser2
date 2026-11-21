import logging
from enum import IntEnum
from threading import Lock

import numpy as np
from OpenGL.GL import Constant, GL_CLAMP_TO_EDGE, GL_LINEAR, GL_NONE

from modules.utils import Point2f, Quad, Rect
from .fbo import Fbo
from .texture import Texture
from .texture_format import array_format

logger = logging.getLogger(__name__)

_FLIPPED: Rect = Rect(0.0, 1.0, 1.0, -1.0)     # the whole texture, its rows drawn bottom to top


def _turned(src: Quad) -> bool:
    """Whether `src` takes the image a quarter turned: its top edge runs down the image, not across."""
    return abs(src.top_right.y - src.top_left.y) > abs(src.top_right.x - src.top_left.x)


def _get_internal_format(image: np.ndarray) -> Constant:
    """The texture format of an image, (height, width) or (height, width, channels); GL_NONE, with a warning, for one
    the format table does not hold."""
    channels: int = image.shape[2] if image.ndim == 3 else 1 if image.ndim == 2 else 0
    internal_format: Constant = array_format(image.dtype, channels)
    if internal_format == GL_NONE:
        logger.warning("image format not supported: %s, shape %s", image.dtype, image.shape)
    return internal_format


class ChannelOrder(IntEnum):
    """The order of a CPU image's color channels: RGB, or BGR as OpenCV keeps them."""
    RGB = 0
    BGR = 1


class Image(Fbo):
    def __init__(self,
                 channel_order: ChannelOrder = ChannelOrder.RGB,
                 interpolation: int = GL_LINEAR,
                 wrap: int = GL_CLAMP_TO_EDGE,
                 border_color: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 1.0)) -> None:
        """Initialize Image texture with specified channel order and texture settings.

        Args:
            channel_order: Channel order of the input images: BGR (OpenCV's) or RGB
            interpolation: Filter mode for min/mag (GL_LINEAR or GL_NEAREST)
            wrap: Wrap mode for both axes (GL_CLAMP_TO_EDGE, GL_REPEAT, GL_MIRRORED_REPEAT, GL_CLAMP_TO_BORDER)
            border_color: RGBA color when wrap is GL_CLAMP_TO_BORDER
        """
        super().__init__(interpolation, wrap, border_color)
        self._image: np.ndarray | None = None
        self._src: Quad | None = None
        self._needs_update: bool = False
        self._mutex: Lock = Lock()
        self._channel_order: ChannelOrder = channel_order
        self._source: Texture = Texture(interpolation, wrap, border_color,     # the upload's, in the image's order
                                        bgr=channel_order == ChannelOrder.BGR)

    def set_image(self, image: np.ndarray, src: Quad | None = None) -> None:
        """Set image to be uploaded to texture.

        Args:
            image: NumPy array containing the image data
            src: where in the image (0..1, the origin top-left) the texture's top left, top right, bottom right and
                bottom left come from, to turn or flip it in the upload; None for the image as it is
        """
        with self._mutex:
            self._image = image
            self._src = src
            self._needs_update = True

    def update(self) -> None:
        image: None | np.ndarray = None
        needs_update: bool = False
        with self._mutex:
            image = self._image
            src: Quad | None = self._src
            needs_update = self._needs_update
            self._needs_update = False

        if needs_update and image is not None:
            self.set_from_image(image, src)

    def allocate(self, width: int, height: int, internal_format: Constant | int, turned: bool = False) -> None:
        """The texture at `width` × `height` in `internal_format`, and the source the image is uploaded into, in its
        channel order: the same size, or `height` × `width` when the image is drawn a quarter turned."""
        super().allocate(width, height, internal_format)
        if self.allocated:
            self._source.allocate(height if turned else width, width if turned else height, internal_format)

    def deallocate(self) -> None:
        """Deallocate source texture and FBO."""
        self._source.deallocate()
        super().deallocate()

    def set_from_image(self, image: np.ndarray, src: Quad | None = None) -> None:
        """Upload a NumPy array to the texture and flip vertically via FBO; with `src`, turned and flipped in the
        same draw (`set_image`), a quarter turn swapping the texture's width and height.

        Args:
            image: NumPy array containing the image data
            src: where in the image each corner of the texture comes from; None for the image as it is
        """
        height: int = image.shape[0]
        width: int = image.shape[1]
        turned: bool = src is not None and _turned(src)
        out_width, out_height = (height, width) if turned else (width, height)

        internal_format: Constant = _get_internal_format(image)
        if internal_format == GL_NONE:
            return
        if not self.allocated or internal_format != self.internal_format \
                or (out_width, out_height) != (self.width, self.height) \
                or (width, height) != (self._source.width, self._source.height):
            if self.allocated:
                self.deallocate()
            self.allocate(out_width, out_height, internal_format, turned)

        if not self.allocated:
            return

        self._source.upload(image)

        # Render flipped to FBO using modern shader; the source's rows are the image's bottom to top
        self.begin()
        self._source.draw(src=_FLIPPED if src is None else Quad(*(Point2f(p.x, 1.0 - p.y) for p in
                                                                  (src.top_left, src.top_right, src.bottom_right, src.bottom_left))))
        self.end()


