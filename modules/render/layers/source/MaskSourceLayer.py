# Standard library imports
import torch

# Third-party imports
from OpenGL.GL import * # type: ignore

# Local application imports
from modules.DataHub import DataHub, DataHubType
from modules.gl import Tensor, Texture
from modules.pose.batch.ImageFrame import ImageFrame
from modules.render.layers.LayerBase import LayerBase, DataCache


class MaskSourceLayer(LayerBase):
    """Pure source layer for mask retrieval from DataHub.

    Retrieves mask tensor from GPUFrame and uploads to GPU texture.
    No processing - dilation now handled by CentreMaskLayer.
    """

    def __init__(self, track_id: int, data_hub: DataHub) -> None:
        self._track_id: int = track_id
        self._data_hub: DataHub = data_hub
        self._cuda_image: Tensor = Tensor(wrap=GL_CLAMP_TO_BORDER)
        self._data_cache: DataCache[torch.Tensor] = DataCache[torch.Tensor]()
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
        mask_tensor: torch.Tensor | None = gpu_frame.mask if gpu_frame else None
        self._data_cache.update(mask_tensor)

        if self._data_cache.lost:
            self._cuda_image.clear()

        if self._data_cache.idle or mask_tensor is None:
            return

        self._cuda_image.set_tensor(mask_tensor)
        self._cuda_image.update()
        self._dirty = True