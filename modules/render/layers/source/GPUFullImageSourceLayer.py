# Standard library imports
import torch
import cupy as cp

# Third-party imports
from OpenGL.GL import * # type: ignore

# Local application imports
from modules.DataHub import DataHub, DataHubType
from modules.gl import Tensor, Texture
from modules.render.layers.LayerBase import LayerBase, DataCache
from modules.pose.batch.GPUFrame import GPUFrame

from modules.utils.HotReloadMethods import HotReloadMethods


class GPUFullImageSourceLayer(LayerBase):
    """Renders the full source image from GPUFrame.

    Displays the complete camera frame that was uploaded to GPU.
    """

    def __init__(self, track_id: int, data_hub: DataHub) -> None:
        self._track_id: int = track_id
        self._data_hub: DataHub = data_hub
        self._cuda_image: Tensor = Tensor()
        self._data_cache: DataCache[GPUFrame] = DataCache[GPUFrame]()

        # hot reloader
        self.hot_reloader = HotReloadMethods(self.__class__, True, True)

    @property
    def texture(self) -> Texture:
        return self._cuda_image

    def allocate(self, width: int | None = None, height: int | None = None, internal_format: int | None = None) -> None:
        pass  # Lazy allocation on first update

    def deallocate(self) -> None:
        self._cuda_image.deallocate()

    def update(self) -> None:
        gpu_frame: GPUFrame | None = self._data_hub.get_item(DataHubType.gpu_frames, self._track_id)
        self._data_cache.update(gpu_frame)

        if self._data_cache.lost:
            self._cuda_image.deallocate()

        if self._data_cache.idle or gpu_frame is None:
            return

        # Convert CuPy full image to PyTorch tensor (zero-copy via DLPack)
        full_image_tensor = torch.as_tensor(gpu_frame.full_image, device='cuda')  # (H, W, 3) uint8

        self._cuda_image.set_tensor(full_image_tensor)
        self._cuda_image.update()
