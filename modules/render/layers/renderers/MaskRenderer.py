# Standard library imports

# Third-party imports
from cuda.bindings import runtime # type: ignore
import numpy as np
from OpenGL.GL import * # type: ignore
import torch

# Local application imports
from modules.DataHub import DataHub, DataHubType
from modules.gl.LayerBase import LayerBase, Rect
from modules.gl.Texture import Texture

from modules.utils.HotReloadMethods import HotReloadMethods


class MaskRenderer(LayerBase):

    def __init__(self, track_id: int, data_hub: DataHub) -> None:
        self._track_id: int = track_id
        self._data_hub: DataHub = data_hub
        self._texture: Texture = Texture()

        # PBO for GPU-to-GPU texture updates
        self._pbo: int = 0
        self._pbo_size: int = 0
        self._prev_tensor: torch.Tensor | None = None

        # CUDA-OpenGL interop
        self._cuda_gl_resource = None

        # hot reloader
        self.hot_reloader = HotReloadMethods(self.__class__, True, True)

    def allocate(self, width: int, height: int, internal_format: int) -> None:
        self._texture.allocate(width, height, GL_R32F)

        # Allocate new PBO for async GPU transfers
        self._pbo = glGenBuffers(1)
        self._pbo_size = width * height * 4  # float32 = 4 bytes
        glBindBuffer(GL_PIXEL_UNPACK_BUFFER, self._pbo)
        glBufferData(GL_PIXEL_UNPACK_BUFFER, self._pbo_size, None, GL_STREAM_DRAW)
        glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0)

        # Register PBO with CUDA for interop
        try:
            err, self._cuda_gl_resource = runtime.cudaGraphicsGLRegisterBuffer(
                self._pbo,
                runtime.cudaGraphicsRegisterFlags.cudaGraphicsRegisterFlagsWriteDiscard
            )
            if err != runtime.cudaError_t.cudaSuccess:
                print(f"MaskRenderer: CUDA-GL interop registration failed (error {err}), using CPU fallback")
                self._cuda_gl_resource = None
        except Exception as e:
            print(f"MaskRenderer: CUDA-GL interop registration failed ({e}), using CPU fallback")
            self._cuda_gl_resource = None

    def deallocate(self) -> None:
        self._texture.deallocate()

        if self._pbo > 0:
            glDeleteBuffers(1, [self._pbo])
            self._pbo = 0
            self._pbo_size = 0

        if self._cuda_gl_resource is not None:
            runtime.cudaGraphicsUnregisterResource(self._cuda_gl_resource)
            self._cuda_gl_resource = None

    def draw(self, rect: Rect) -> None:
        self._texture.draw(rect.x, rect.y, rect.width, rect.height)

    def get_texture_id(self) -> int:
        return self._texture.tex_id

    def update(self) -> None:

        mask_tensor: torch.Tensor | None = self._data_hub.get_item(DataHubType.mask_tensor, self._track_id)

        # Only update if tensor changed
        if mask_tensor is self._prev_tensor:
            return
        self._prev_tensor = mask_tensor

        if mask_tensor is not None:
            if mask_tensor.shape[0] != self._texture.height or mask_tensor.shape[1] != self._texture.width:
                # Resize texture and FBO
                self.deallocate()
                self.allocate(mask_tensor.shape[1], mask_tensor.shape[0], GL_R32F)

        if mask_tensor is None:
            glClearTexImage(self._texture.tex_id, 0, GL_RED, GL_FLOAT, None)
            return

        if self._cuda_gl_resource is None:
            self._update_with_cpu(mask_tensor)
        else:
            self._update_with_gpu(mask_tensor)

    def _update_with_gpu(self, mask_tensor: torch.Tensor) -> None:
        """GPU-to-GPU copy using CUDA-OpenGL interop.

        PyTorch CUDA tensor → PBO (device-to-device) → GL texture (all on GPU).
        """
        # Convert FP16 to FP32
        if mask_tensor.dtype == torch.float16:
            mask_tensor = mask_tensor.float()

        # Flip vertically and ensure contiguous memory for GPU copy
        mask_tensor = mask_tensor.flip(0).contiguous()
        # Map graphics resource to get CUDA device pointer
        (err,) = runtime.cudaGraphicsMapResources(1, self._cuda_gl_resource, torch.cuda.current_stream().cuda_stream)
        if err != runtime.cudaError_t.cudaSuccess:
            raise RuntimeError(f"Failed to map graphics resource: {err}")

        # Get mapped device pointer and size
        err, pbo_dev_ptr, pbo_size = runtime.cudaGraphicsResourceGetMappedPointer(self._cuda_gl_resource)
        if err != runtime.cudaError_t.cudaSuccess:
            raise RuntimeError(f"Failed to get mapped pointer: {err}")

        # Get PyTorch tensor's CUDA device pointer
        tensor_ptr = mask_tensor.data_ptr()

        # Copy directly from tensor to PBO (device-to-device, no CPU involved)
        (err,) = runtime.cudaMemcpy(pbo_dev_ptr, tensor_ptr, self._pbo_size, runtime.cudaMemcpyKind.cudaMemcpyDeviceToDevice)
        if err != runtime.cudaError_t.cudaSuccess:
            raise RuntimeError(f"Failed to copy to PBO: {err}")

        # Unmap graphics resource
        runtime.cudaGraphicsUnmapResources(1, self._cuda_gl_resource, torch.cuda.current_stream().cuda_stream)

        # Upload PBO to texture (GPU-to-GPU via OpenGL driver)
        glBindBuffer(GL_PIXEL_UNPACK_BUFFER, self._pbo)
        self._texture.bind()
        glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, self._texture.width, self._texture.height, GL_RED, GL_FLOAT, None)  # None = use bound PBO
        self._texture.unbind()
        glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0)

    def _update_with_cpu(self, mask_tensor: torch.Tensor) -> None:
        """Fallback: CPU transfer (GPU -> CPU -> GPU)."""
        # Flip vertically, convert to float32, and move to CPU
        mask_np = mask_tensor.flip(0).float().cpu().numpy()  # (H, W) float32 [0-1]

        # Upload to OpenGL texture
        self._texture.bind()
        glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, self._texture.width, self._texture.height, GL_RED, GL_FLOAT, mask_np)
        self._texture.unbind()
