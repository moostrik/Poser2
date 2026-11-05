# Standard library imports
import torch
from typing import TYPE_CHECKING

# Third-party imports
from OpenGL.GL import GL_CLAMP_TO_BORDER

# Local application imports
from modules.board import HasSegmentationImages
from modules.gl import Tensor, Texture
from ..layer_base import LayerBase, DataCache

if TYPE_CHECKING:
    from modules.inference import segmentation


class ForegroundSourceLayer(LayerBase):

    def __init__(self, track_id: int, board: HasSegmentationImages) -> None:
        self._track_id: int = track_id
        self._board: HasSegmentationImages = board
        self._cuda_image: Tensor = Tensor(wrap=GL_CLAMP_TO_BORDER, border_color=(0.0, 0.0, 0.0, 0.0))
        self._data_cache: DataCache[torch.Tensor]= DataCache[torch.Tensor]()
        self._dirty: bool = False
        self._timestamp: float | None = None     # the capture time of the frame the foreground was made from

    @property
    def texture(self) -> Texture:
        return self._cuda_image

    @property
    def timestamp(self) -> float | None:
        """The capture time of the camera frame the foreground it holds was made from (`segmentation.Image.timestamp`), None before the first."""
        return self._timestamp

    @property
    def dirty(self) -> bool:
        return self._dirty

    def deallocate(self) -> None:
        self._cuda_image.deallocate()

    def update(self) -> None:
        self._dirty = False
        seg_image: segmentation.Image | None = self._board.get_segmentation_image(self._track_id)
        foreground: torch.Tensor | None = seg_image.foreground if seg_image else None
        self._data_cache.set(foreground)

        if self._data_cache.lost:
            self._cuda_image.clear()
            self._timestamp = None

        if self._data_cache.idle or foreground is None or seg_image is None:
            return

        self._cuda_image.set_tensor(foreground)
        self._cuda_image.update()
        self._timestamp = seg_image.timestamp
        self._dirty = True