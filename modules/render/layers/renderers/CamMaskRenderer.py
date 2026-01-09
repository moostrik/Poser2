# Standard library imports
import torch

# Third-party imports
from OpenGL.GL import * # type: ignore

# Local application imports
from modules.DataHub import DataHub, DataHubType
from modules.gl import Tensor, SwapFbo, Texture
from modules.render.layers.LayerBase import LayerBase, Rect
from modules.render.shaders import MaskDilate

from modules.utils.HotReloadMethods import HotReloadMethods


class CamMaskRenderer(LayerBase):

    def __init__(self, track_id: int, data_hub: DataHub) -> None:
        self._track_id: int = track_id
        self._data_hub: DataHub = data_hub
        self._cuda_image: Tensor = Tensor()
        self._prev_tensor: torch.Tensor | None = None
        self._fbo: SwapFbo = SwapFbo()
        self._dilate_shader: MaskDilate = MaskDilate()

        self.process_scale: float = 2.0

        # hot reloader
        self.hot_reloader = HotReloadMethods(self.__class__, True, True)

    @property
    def texture(self) -> Texture:
        return self._fbo.texture

    def allocate(self, width: int | None = None, height: int | None = None, internal_format: int | None = None) -> None:
        self._dilate_shader.allocate()

    def deallocate(self) -> None:
        self._cuda_image.deallocate()
        self._fbo.deallocate()
        self._dilate_shader.deallocate()

    def update(self) -> None:
        mask_tensor: torch.Tensor | None = self._data_hub.get_item(DataHubType.mask_tensor, self._track_id)

        # Only update if tensor changed
        if mask_tensor is self._prev_tensor:
            return
        self._prev_tensor = mask_tensor

        if self._fbo.allocated:
            self._fbo.clear()

        if mask_tensor is None:
            # self._cuda_image.clear()
            return

        self._cuda_image.set_tensor(mask_tensor)
        self._cuda_image.update()


        self.process_scale: float = 2.0

        if self._cuda_image.allocated:
            w = int(self._cuda_image.width * self.process_scale)
            h = int(self._cuda_image.height * self.process_scale)
            if not self._fbo.allocated or self._fbo.width != w or self._fbo.height != h:
                self._fbo.allocate(w, h, self._cuda_image.internal_format)

            self._fbo.clear()
            # glDisable(GL_BLEND)
            glColor4f(1.0, 1.0, 1.0, 1.0)

            self._fbo.begin()
            self._cuda_image.draw(0, 0, w, h)
            self._fbo.end()

            for i in range(1):
                self._fbo.swap()
                self._dilate_shader.use(self._fbo.fbo_id, self._fbo.back_tex_id, 1.0)




