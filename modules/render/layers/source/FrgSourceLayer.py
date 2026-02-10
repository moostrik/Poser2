# Standard library imports
import torch

# Third-party imports
from OpenGL.GL import * # type: ignore

# Local application imports
from modules.DataHub import DataHub, DataHubType
from modules.gl import Tensor, SwapFbo, Texture, Blit
from modules.pose.batch.GPUFrame import GPUFrame
from modules.render.layers.LayerBase import LayerBase, DataCache, Rect
from modules.render.shaders import MaskDilate

from modules.utils.HotReloadMethods import HotReloadMethods


class FrgSourceLayer(LayerBase):

    def __init__(self, track_id: int, data_hub: DataHub) -> None:
        self._track_id: int = track_id
        self._data_hub: DataHub = data_hub
        self._cuda_image: Tensor = Tensor()
        self._data_cache: DataCache[torch.Tensor]= DataCache[torch.Tensor]()

        # hot reloader
        self.hot_reloader = HotReloadMethods(self.__class__, True, True)

    @property
    def texture(self) -> Texture:
        return self._cuda_image.texture

    def deallocate(self) -> None:
        self._cuda_image.deallocate()

    def update(self) -> None:
        gpu_frame: GPUFrame | None = self._data_hub.get_item(DataHubType.gpu_frames, self._track_id)
        foreground: torch.Tensor | None = gpu_frame.foreground if gpu_frame else None
        self._data_cache.update(foreground)

        if self._data_cache.lost:
            self._cuda_image.clear()

        if self._data_cache.idle or foreground is None:
            return

        self._cuda_image.set_tensor(foreground)
        self._cuda_image.update()