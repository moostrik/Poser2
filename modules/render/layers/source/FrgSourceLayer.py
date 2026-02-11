# Standard library imports
import torch

# Third-party imports
from OpenGL.GL import * # type: ignore

# Local application imports
from modules.DataHub import DataHub, DataHubType
from modules.gl import Tensor, Texture
from modules.pose.batch.ImageFrame import ImageFrame
from modules.render.layers.LayerBase import LayerBase, DataCache


class FrgSourceLayer(LayerBase):

    def __init__(self, track_id: int, data_hub: DataHub) -> None:
        self._track_id: int = track_id
        self._data_hub: DataHub = data_hub
        self._cuda_image: Tensor = Tensor()
        self._data_cache: DataCache[torch.Tensor]= DataCache[torch.Tensor]()

    @property
    def texture(self) -> Texture:
        return self._cuda_image.texture

    def deallocate(self) -> None:
        self._cuda_image.deallocate()

    def update(self) -> None:
        gpu_frame: ImageFrame | None = self._data_hub.get_item(DataHubType.gpu_frames, self._track_id)
        foreground: torch.Tensor | None = gpu_frame.foreground if gpu_frame else None
        self._data_cache.update(foreground)

        if self._data_cache.lost:
            self._cuda_image.clear()

        if self._data_cache.idle or foreground is None:
            return

        self._cuda_image.set_tensor(foreground)
        self._cuda_image.update()