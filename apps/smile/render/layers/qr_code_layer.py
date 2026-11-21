import numpy as np

from ...credit import qr_image, quiet_px
from modules.gl import ChannelOrder, Image, Texture


class QrCodeLayer:
    """The QR code of a url as a texture (`qr_image`: black on white with its quiet zone, every module whole
    pixels), made again only when the url or the size changes."""

    def __init__(self) -> None:
        self._image: Image = Image(ChannelOrder.BGR)
        self._key: tuple[str, int] | None = None
        self.quiet: int = 0         # px of white border round the code (`quiet_px`)

    @property
    def texture(self) -> Texture:
        return self._image

    @property
    def size(self) -> int:
        """The code's width and height with its border, px; 0 before the first `set`."""
        return self._image.width if self._image.allocated else 0

    def set(self, url: str, size: int) -> None:
        """The code of `url` at most `size` px square, uploaded now (main thread)."""
        key: tuple[str, int] = (url, size)
        if key != self._key:
            self._key = key
            image: np.ndarray = qr_image(url, size)
            self.quiet = quiet_px(image)
            self._image.set_image(image)
        self._image.update()

    def deallocate(self) -> None:
        self._image.deallocate()
        self._key = None
