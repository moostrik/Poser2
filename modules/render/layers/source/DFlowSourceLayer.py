# Standard library imports
import torch

# Third-party imports
from OpenGL.GL import * # type: ignore

# Local application imports
from modules.DataHub import DataHub, DataHubType
from modules.render.layers.LayerBase import LayerBase, Rect
from modules.gl import Tensor, SwapFbo, Texture
from modules.gl.Texture import draw_quad
from modules.render.shaders import DenseFlowFilter as shader

from modules.utils.HotReloadMethods import HotReloadMethods


class DFlowSourceLayer(LayerBase):
    """Renderer for optical flow visualization.

    Retrieves flow tensors from DataHub and converts them to OpenGL textures
    for visualization. Flow data is 2-channel (x, y displacement).
    """

    def __init__(self, track_id: int, data_hub: DataHub) -> None:
        self._track_id: int = track_id
        self._data_hub: DataHub = data_hub
        self._flow_texture: Tensor = Tensor()
        self._prev_tensor: torch.Tensor | None = None
        self._fbo: SwapFbo = SwapFbo()

        self.process_scale: float = 2.0
        self.flow_scale: float = 10.0
        self.flow_gamma: float = 0.5
        self.noise_threshold: float = 0.2

        self._shader: shader = shader()

        # hot reloader
        self.hot_reloader = HotReloadMethods(self.__class__, True, True)

    @property
    def texture(self) -> Texture:
        return self._fbo.texture

    def allocate(self, width: int | None = None, height: int | None = None, internal_format: int | None = None) -> None:
        """Initialize renderer resources."""
        self._shader.allocate()

    def deallocate(self) -> None:
        """Release all GPU resources."""
        self._flow_texture.deallocate()
        self._fbo.deallocate()
        self._shader.deallocate()

    def draw(self, rect: Rect) -> None:
        """Draw the flow visualization."""
        if self._fbo.allocated:
            self._fbo.draw(rect.x, rect.y, rect.width, rect.height)

    def update(self) -> None:
        """Update flow texture from DataHub."""

        flow_tensor: torch.Tensor | None = self._data_hub.get_item(DataHubType.flow_tensor, self._track_id)

        # Only update if tensor changed
        if flow_tensor is self._prev_tensor:
            return
        self._prev_tensor = flow_tensor

        if self._fbo.allocated:
            self._fbo.clear()

        if flow_tensor is None:
            # self._flow_texture.clear()
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

            self._fbo.clear()
            glColor4f(1.0, 1.0, 1.0, 1.0)

            self._fbo.begin()
            self._flow_texture.bind()
            draw_quad(0, 0, w, h, flipV=True)
            self._flow_texture.unbind()
            self._fbo.end()

            # Apply flow visualization shader with noise filtering
            self._fbo.swap()
            self._shader.use(
                self._fbo.fbo_id,
                self._fbo.back_tex_id,
                scale=self.flow_scale,
                gamma=self.flow_gamma,
                noise_threshold=self.noise_threshold
            )