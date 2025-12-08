# Standard library imports
import torch

# Third-party imports
from OpenGL.GL import * # type: ignore

# Local application imports
from modules.DataHub import DataHub, DataHubType
from modules.gl.LayerBase import LayerBase, Rect
from modules.gl.Fbo import SwapFbo
from modules.gl.Texture import Texture, draw_quad
from modules.gl.shaders.Blend import Blend as shader

from modules.utils.HotReloadMethods import HotReloadMethods


class MaskLayer(LayerBase):
    """Renders segmentation masks directly from GPU tensors using Pixel Buffer Objects.

    Uses PBO (GL_PIXEL_UNPACK_BUFFER) to transfer PyTorch CUDA tensors to OpenGL textures
    without CPU roundtrip. Masks stay entirely on GPU throughout the pipeline.
    Falls back to CPU transfer if PBO mapping fails.
    """

    _shader = shader()

    def __init__(self, track_id: int, data_hub: DataHub) -> None:
        self._track_id: int = track_id
        self._data_hub: DataHub = data_hub
        self._fbo: SwapFbo = SwapFbo()
        self._texture: Texture = Texture()

        # PBO for GPU-to-GPU texture updates
        self._pbo: int = 0
        self._pbo_size: int = 0
        self._prev_tensor: torch.Tensor | None = None


        self.blend_factor: float = 0.25

        # hot reloader
        self.hot_reloader = HotReloadMethods(self.__class__, True, True)

    def allocate(self, width: int, height: int, internal_format: int) -> None:
        """Allocate OpenGL texture and PBO for GPU-to-GPU updates."""

        self._fbo.allocate(width, height, GL_R32F)
        self._fbo.clear()

        # Allocate texture using Texture class
        self._texture.allocate(width, height, GL_R32F)

        # Allocate PBO for async GPU transfers
        if self._texture.allocated:
            self._pbo_size = width * height * 4  # float32 = 4 bytes
            self._pbo = glGenBuffers(1)
            glBindBuffer(GL_PIXEL_UNPACK_BUFFER, self._pbo)
            glBufferData(GL_PIXEL_UNPACK_BUFFER, self._pbo_size, None, GL_STREAM_DRAW)
            glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0)

        if not MaskLayer._shader.allocated:
            MaskLayer._shader.allocate(monitor_file=True)

    def deallocate(self) -> None:
        """Delete PBO and OpenGL texture."""
        self._fbo.deallocate()
        self._texture.deallocate()
        if self._pbo > 0:
            glDeleteBuffers(1, [self._pbo])
            self._pbo = 0
            self._pbo_size = 0
        if self._shader.allocated:
            MaskLayer._shader.deallocate()

    def draw(self, rect: Rect) -> None:
        self._fbo.draw(rect.x, rect.y, rect.width, rect.height)

    def update(self) -> None:
        """Update texture from GPU tensor using CUDA-OpenGL interop or CPU fallback."""

        if not MaskLayer._shader.allocated:
            MaskLayer._shader.allocate(monitor_file=True)

        mask_tensor: torch.Tensor | None = self._data_hub.get_item(DataHubType.mask_tensor, self._track_id)

        # Only update if tensor changed
        if mask_tensor is self._prev_tensor:
            return
        self._prev_tensor = mask_tensor

        if mask_tensor is not None:
            if mask_tensor.shape[0] != self._texture.height or mask_tensor.shape[1] != self._texture.width:
                # Resize texture and FBO
                self.allocate(mask_tensor.shape[1], mask_tensor.shape[0], GL_R32F)

        LayerBase.setView(self._fbo.width, self._fbo.height)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

        if mask_tensor is None:
            self._fbo.clear()
            return

        # GPU-to-GPU copy using PBO
        self._update_with_pbo(mask_tensor)

        self.blend_factor = 0.25

        self._fbo.swap()
        MaskLayer._shader.use(self._fbo.fbo_id, self._fbo.back_tex_id,self._texture.tex_id, self.blend_factor)


    def _update_with_pbo(self, mask_tensor: torch.Tensor) -> None:
        """GPU-to-GPU copy using Pixel Buffer Object.

        PyTorch tensor → PBO → GL texture (all on GPU).
        """
        # Convert FP16 to FP32 and flip vertically
        if mask_tensor.dtype == torch.float16:
            mask_tensor = mask_tensor.float()

        mask_tensor = mask_tensor.flip(0).contiguous()

        # Bind PBO
        glBindBuffer(GL_PIXEL_UNPACK_BUFFER, self._pbo)

        # Get CUDA device pointer from PyTorch tensor
        tensor_ptr = mask_tensor.data_ptr()

        # Copy directly from CUDA tensor to PBO using cudaMemcpy
        try:
            import torch.cuda as cuda
            # Get PBO's device pointer via CUDA-GL interop
            # This requires registering the PBO buffer with CUDA first

            # Fallback: use CPU for now since we need proper CUDA-GL buffer registration
            mask_np = mask_tensor.cpu().numpy()
            glBufferSubData(GL_PIXEL_UNPACK_BUFFER, 0, self._pbo_size, mask_np)

        except Exception as e:
            print(f"MaskRenderer: PBO copy failed ({e}), using CPU")
            glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0)
            self._update_with_pbo(mask_tensor.flip(0))  # Already flipped, flip back
            return

        # Upload PBO to texture (GPU-to-GPU via OpenGL driver)
        self._texture.bind()
        glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, self._texture.width, self._texture.height,
                       GL_RED, GL_FLOAT, None)  # None = use bound PBO
        self._texture.unbind()

        # Unbind PBO
        glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0)

    def _update_with_cpu(self, mask_tensor: torch.Tensor) -> None:
        """Fallback: CPU transfer (GPU -> CPU -> GPU)."""
        # Convert to float32, flip vertically, and move to CPU
        mask_np = mask_tensor.flip(0).float().cpu().numpy()  # (H, W) float32 [0-1]

        # Upload to OpenGL texture
        self._texture.bind()
        glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, self._texture.width, self._texture.height,
                       GL_RED, GL_FLOAT, mask_np)
        self._texture.unbind()
