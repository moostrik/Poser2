# Standard library imports
import torch

# Third-party imports
from OpenGL.GL import * # type: ignore

# Local application imports
from modules.DataHub import DataHub, DataHubType
from modules.gl import Tensor, Texture
from modules.render.layers.LayerBase import LayerBase, DataCache
from modules.pose.batch.ImageFrame import ImageFrame


class CropSourceLayer(LayerBase):
    """Renders the cropped region from GPUFrame.

    Displays the 384x512 (or configured size) crop that will be sent to TRT models.
    """

    def __init__(self, track_id: int, data_hub: DataHub) -> None:
        self._track_id: int = track_id
        self._data_hub: DataHub = data_hub
        self._cuda_image: Tensor = Tensor()
        self._data_cache: DataCache[ImageFrame] = DataCache[ImageFrame]()
        self._dirty: bool = False

    @property
    def texture(self) -> Texture:
        return self._cuda_image

    @property
    def dirty(self) -> bool:
        return self._dirty

    def allocate(self, width: int | None = None, height: int | None = None, internal_format: int | None = None) -> None:
        pass  # Lazy allocation on first update

    def deallocate(self) -> None:
        self._cuda_image.deallocate()

    def update(self) -> None:
        self._dirty = False
        gpu_frame: ImageFrame | None = self._data_hub.get_item(DataHubType.gpu_frames, self._track_id)
        self._data_cache.update(gpu_frame)

        if self._data_cache.lost:
            self._cuda_image.clear()

        if self._data_cache.idle or gpu_frame is None or gpu_frame.crop is None:
            return

        self._cuda_image.set_tensor(gpu_frame.crop)
        self._cuda_image.update()
        self._dirty = True
