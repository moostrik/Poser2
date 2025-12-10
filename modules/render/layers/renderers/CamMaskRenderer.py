# Standard library imports
import torch

# Third-party imports
from OpenGL.GL import * # type: ignore

# Local application imports
from modules.DataHub import DataHub, DataHubType
from modules.gl.LayerBase import LayerBase, Rect
from modules.gl.TensorTexture import TensorTexture
from modules.gl.Fbo import Fbo, SwapFbo

from modules.gl.shaders.MaskDilate import MaskDilate

from modules.utils.HotReloadMethods import HotReloadMethods


class CamMaskRenderer(LayerBase):
    _dilate_shader: MaskDilate = MaskDilate()

    def __init__(self, track_id: int, data_hub: DataHub) -> None:
        self._track_id: int = track_id
        self._data_hub: DataHub = data_hub
        self._cuda_image: TensorTexture = TensorTexture()
        self._prev_tensor: torch.Tensor | None = None
        self._fbo: SwapFbo = SwapFbo()

        self.process_scale: float = 2.0

        # hot reloader
        self.hot_reloader = HotReloadMethods(self.__class__, True, True)

    @property
    def width(self) -> int:
        return self._fbo.width
    @property
    def height(self) -> int:
        return self._fbo.height
    @property
    def internal_format(self):
        return self._fbo.internal_format
    @property
    def tex_id(self) -> int:
        return self._fbo.tex_id

    def allocate(self, width: int | None = None, height: int | None = None, internal_format: int | None = None) -> None:
        if not CamMaskRenderer._dilate_shader.allocated:
            CamMaskRenderer._dilate_shader.allocate(monitor_file=True)

    def deallocate(self) -> None:
        self._cuda_image.deallocate()
        self._fbo.deallocate()

    def draw(self, rect: Rect) -> None:
        if self._fbo.allocated:
            self._fbo.draw(rect.x, rect.y, rect.width, rect.height)
        # self._cuda_image.draw(rect.x, rect.y, rect.width, rect.height)

    def update(self) -> None:
        if not CamMaskRenderer._dilate_shader.allocated:
            CamMaskRenderer._dilate_shader.allocate(monitor_file=True)

        mask_tensor: torch.Tensor | None = self._data_hub.get_item(DataHubType.mask_tensor, self._track_id)

        # Only update if tensor changed
        if mask_tensor is self._prev_tensor:
            return
        self._prev_tensor = mask_tensor

        self._fbo.clear()

        if mask_tensor is None:
            self._cuda_image.clear()
            return

        self._cuda_image.set_tensor(mask_tensor)
        self._cuda_image.update()


        self.process_scale: float = 2.0

        if self._cuda_image.allocated:
            w = int(self._cuda_image.width * self.process_scale)
            h = int(self._cuda_image.height * self.process_scale)
            if not self._fbo.allocated or self._fbo.width != w or self._fbo.height != h:
                self._fbo.allocate(w, h, self._cuda_image.internal_format)

            LayerBase.setView(self._fbo.width, self._fbo.height)
            self._fbo.clear()
            # glDisable(GL_BLEND)
            glColor4f(1.0, 1.0, 1.0, 1.0)

            self._fbo.begin()
            self._cuda_image.draw(0, 0, w, h)
            self._fbo.end()

            for i in range(1):
                self._fbo.swap()
                CamMaskRenderer._dilate_shader.use(self._fbo.fbo_id, self._fbo.back_tex_id, 1.0)




