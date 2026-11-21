from typing import Callable, TypeAlias

import torch
from OpenGL.GL import GL_CLAMP_TO_EDGE

from modules.gl import Tensor, Texture

from ...layer_base import LayerBase
from ...data_cache import DataCache

# The board read a layer shows: the image's tensor and the capture time of the frame it was made from (None where the
# read has none), or None while the board holds no image
BoardImageRead: TypeAlias = Callable[[], tuple[torch.Tensor, float | None] | None]


class BoardImageLayer(LayerBase):
    """The image a board read gives, as a texture: uploaded when the read gives a new tensor, cleared when it gives
    none, with the capture time of the frame it was made from."""

    def __init__(self, read: BoardImageRead, wrap: int = GL_CLAMP_TO_EDGE,
                 border_color: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 1.0)) -> None:
        self._read: BoardImageRead = read
        self._cuda_image: Tensor = Tensor(wrap=wrap, border_color=border_color)
        self._data_cache: DataCache[torch.Tensor] = DataCache[torch.Tensor]()
        self._timestamp: float | None = None

    @property
    def texture(self) -> Texture:
        return self._cuda_image

    @property
    def timestamp(self) -> float | None:
        """The capture time of the frame the image it holds was made from; None before the first, or where the read
        has none."""
        return self._timestamp

    def deallocate(self) -> None:
        if self._cuda_image.allocated:
            self._cuda_image.deallocate()

    def update(self) -> None:
        read = self._read()
        image: torch.Tensor | None = read[0] if read else None
        self._data_cache.set(image)

        if self._data_cache.lost:
            self._cuda_image.clear()
            self._timestamp = None

        if self._data_cache.idle or image is None or read is None:
            return

        # Handed on written; its memory is kept from being reused until this stream's copy is done
        image.record_stream(torch.cuda.current_stream())
        self._cuda_image.set_tensor(image)
        self._cuda_image.update()
        self._timestamp = read[1]
