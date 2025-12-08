# Standard library imports
import torch

# Third-party imports

# Local application imports
from modules.DataHub import DataHub, DataHubType
from modules.gl.LayerBase import LayerBase, Rect
from modules.gl.TensorTexture import TensorTexture

from modules.utils.HotReloadMethods import HotReloadMethods


class MaskRenderer(LayerBase):

    def __init__(self, track_id: int, data_hub: DataHub) -> None:
        self._track_id: int = track_id
        self._data_hub: DataHub = data_hub
        self._cuda_image: TensorTexture = TensorTexture()
        self._prev_tensor: torch.Tensor | None = None

        # hot reloader
        self.hot_reloader = HotReloadMethods(self.__class__, True, True)

    @property
    def width(self) -> int:
        return self._cuda_image.width
    @property
    def height(self) -> int:
        return self._cuda_image.height
    @property
    def internal_format(self):
        return self._cuda_image.internal_format
    @property
    def tex_id(self) -> int:
        return self._cuda_image.tex_id

    def deallocate(self) -> None:
        self._cuda_image.deallocate()

    def draw(self, rect: Rect) -> None:
        self._cuda_image.draw(rect.x, rect.y, rect.width, rect.height)


    def update(self) -> None:
        mask_tensor: torch.Tensor | None = self._data_hub.get_item(DataHubType.mask_tensor, self._track_id)

        # Only update if tensor changed
        if mask_tensor is self._prev_tensor:
            return
        self._prev_tensor = mask_tensor

        if mask_tensor is None:
            self._cuda_image.clear()
            return

        self._cuda_image.set_tensor(mask_tensor)
        self._cuda_image.update()
