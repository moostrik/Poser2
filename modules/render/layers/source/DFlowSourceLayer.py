# Standard library imports
import torch

# Third-party imports
from OpenGL.GL import * # type: ignore

# Local application imports
from modules.DataHub import DataHub, DataHubType
from modules.gui.PyReallySimpleGui import Frame
from modules.render.layers.LayerBase import LayerBase, DataCache
from modules.gl import Tensor, SwapFbo, Texture, Blit

from modules.render.shaders import DenseFlowFilter as shader


class DFlowSourceLayer(LayerBase):
    """Renderer for optical flow visualization.

    Retrieves flow tensors from DataHub and converts them to OpenGL textures
    for visualization. Flow data is 2-channel (x, y displacement).
    """

    def __init__(self, track_id: int, data_hub: DataHub) -> None:
        self._track_id: int = track_id
        self._data_hub: DataHub = data_hub
        self._flow_texture: Tensor = Tensor()
        self._data_cache: DataCache[torch.Tensor]= DataCache[torch.Tensor]()
        self._fbo: SwapFbo = SwapFbo()

        self.process_scale: float = 2.0
        self.flow_scale: float = 10.0
        self.flow_gamma: float = 0.5
        self.noise_threshold: float = 0.2

        self._shader: shader = shader()
        self._dirty: bool = False

    @property
    def texture(self) -> Texture:
        return self._fbo.texture

    @property
    def dirty(self) -> bool:
        return self._dirty

    def allocate(self, width: int | None = None, height: int | None = None, internal_format: int | None = None) -> None:
        """Initialize renderer resources."""
        self._shader.allocate()

    def deallocate(self) -> None:
        """Release all GPU resources."""
        self._flow_texture.deallocate()
        self._fbo.deallocate()
        self._shader.deallocate()

    def draw(self) -> None:
        """Draw the flow visualization."""
        if self._fbo.allocated:
            Blit.use(self._fbo.texture)

    def update(self) -> None:
        """Update flow texture from DataHub."""
        self._dirty = False
        flow_tensor: torch.Tensor | None = self._data_hub.get_item(DataHubType.flow_tensor, self._track_id)

        self._data_cache.update(flow_tensor)

        if self._data_cache.lost:
            self._fbo.clear()

        if self._data_cache.idle or flow_tensor is None:
            return

        self._flow_texture.set_tensor(flow_tensor)
        self._flow_texture.update()

        self.process_scale: float = 1.0
        self.flow_scale: float = 1.0
        self.flow_gamma: float = 1.0
        self.noise_threshold: float = 0.1

        # return
        if self._flow_texture.allocated:
            w = int(self._flow_texture.width * self.process_scale)
            h = int(self._flow_texture.height * self.process_scale)
            if not self._fbo.allocated or self._fbo.width != w or self._fbo.height != h:
                self._fbo.allocate(w, h, self._flow_texture.internal_format)

            self._fbo.begin()
            Blit.use(self._flow_texture)
            self._fbo.end()

            # Apply flow visualization shader with noise filtering
            self._fbo.swap()
            self._fbo.begin()
            self._shader.use(self._fbo.back_texture, self.flow_scale, self.flow_gamma, self.noise_threshold)
            self._fbo.end()
        self._dirty = True