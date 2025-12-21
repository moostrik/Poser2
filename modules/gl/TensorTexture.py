# Standard library imports
from threading import Lock

import numpy as np

# Third-party imports
import torch
from cuda.bindings import runtime # type: ignore
from OpenGL.GL import * # type: ignore

# Local application imports
from modules.gl.Texture import Texture, draw_quad


class TensorTexture(Texture):
    """GPU texture updated directly from CUDA tensors using CUDA-OpenGL interop.

    Provides zero-copy GPU-to-GPU transfer from PyTorch CUDA tensors to OpenGL textures
    via Pixel Buffer Objects (PBO) and CUDA graphics resource registration.
    Falls back to CPU transfer if CUDA-OpenGL interop is unavailable.
    """

    def __init__(self) -> None:
        super().__init__()
        self._tensor: torch.Tensor | None = None
        self._needs_update: bool = False
        self._mutex: Lock = Lock()

        # PBO for GPU-to-GPU texture updates
        self._pbo: int = 0
        self._pbo_size: int = 0
        self._channels: int = 1

        # CUDA-OpenGL interop
        self._cuda_gl_resource = None

    def set_tensor(self, tensor: torch.Tensor) -> None:
        """Set a new CUDA tensor to be uploaded to the texture.

        Args:
            tensor: PyTorch tensor on CUDA device.
                   Shape: (H, W) for 1-channel, (C, H, W) or (H, W, C) for multi-channel.
                   Supported dtypes: float16, float32, uint8.
        """
        with self._mutex:
            self._tensor = tensor
            self._needs_update = True

    def update(self) -> None:
        """Upload pending tensor to GPU texture if needed.

        Checks for size changes and reallocates if necessary.
        Uses CUDA-OpenGL interop for GPU-to-GPU transfer when available,
        falls back to CPU transfer otherwise.
        """
        tensor: torch.Tensor | None = None
        needs_update: bool = False

        with self._mutex:
            tensor = self._tensor
            needs_update = self._needs_update
            self._needs_update = False

        if not needs_update or tensor is None:
            return

        try:
            # Detect tensor shape and channels
            if len(tensor.shape) == 2:  # (H, W)
                height, width = tensor.shape
                channels = 1
            elif len(tensor.shape) == 3:
                if tensor.shape[0] <= 4:  # (C, H, W) format
                    channels, height, width = tensor.shape
                else:  # (H, W, C) format
                    height, width, channels = tensor.shape
            else:
                raise ValueError(f"Unsupported tensor shape: {tensor.shape}")

            # Check if we need to reallocate
            if width != self.width or height != self.height or channels != self._channels or not self.allocated:
                self._reallocate(width, height, channels)

            # Upload tensor to texture
            if self._cuda_gl_resource is None:
                self._update_with_cpu(tensor)
            else:
                self._update_with_pbo(tensor)
        except Exception as e:
            print(f"TensorTexture.update() error: {e}, tensor shape: {tensor.shape if tensor is not None else 'None'}")
            import traceback
            traceback.print_exc()
        finally:
            # Ensure clean OpenGL state
            glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0)
            glBindTexture(GL_TEXTURE_2D, 0)

    def _reallocate(self, width: int, height: int, channels: int) -> None:
        """Reallocate texture and PBO with new dimensions and channel count."""
        # Deallocate old resources
        self._deallocate_pbo()
        if self.allocated:
            super().deallocate()

        self._channels = channels

        # Select internal format based on channel count
        format_map = {
            1: GL_R32F,
            2: GL_RG32F,
            3: GL_RGB32F,
            4: GL_RGBA32F
        }
        internal_format = format_map.get(channels, GL_R32F)

        # Allocate texture (always float32 for flexibility)
        super().allocate(width, height, internal_format)

        # Allocate PBO
        self._pbo = glGenBuffers(1)
        self._pbo_size = width * height * channels * 4  # channels * float32 (4 bytes)
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
                print(f"CudaImage: CUDA-GL interop registration failed (error {err}), using CPU fallback")
                self._cuda_gl_resource = None
        except Exception as e:
            print(f"CudaImage: CUDA-GL interop registration failed ({e}), using CPU fallback")
            self._cuda_gl_resource = None

    def _deallocate_pbo(self) -> None:
        """Deallocate PBO and unregister CUDA resource."""
        if self._pbo > 0:
            glDeleteBuffers(1, [self._pbo])
            self._pbo = 0
            self._pbo_size = 0

        if self._cuda_gl_resource is not None:
            runtime.cudaGraphicsUnregisterResource(self._cuda_gl_resource)
            self._cuda_gl_resource = None

    def deallocate(self) -> None:
        """Deallocate all GPU resources."""
        self._deallocate_pbo()
        super().deallocate()

    def _update_with_pbo(self, tensor: torch.Tensor) -> None:
        """GPU-to-GPU copy using CUDA-OpenGL interop.

        PyTorch CUDA tensor → PBO (device-to-device) → GL texture (all on GPU).
        """
        # Convert to FP32
        if tensor.dtype == torch.float16:
            tensor = tensor.float()
        elif tensor.dtype == torch.uint8:
            tensor = tensor.float() / 255.0

        # Permute multi-channel tensors from (C, H, W) to (H, W, C)
        if len(tensor.shape) == 3 and tensor.shape[0] <= 4:
            tensor = tensor.permute(1, 2, 0)

        # Ensure contiguous memory for GPU copy
        tensor = tensor.contiguous()

        # Map graphics resource to get CUDA device pointer
        (err,) = runtime.cudaGraphicsMapResources(1, self._cuda_gl_resource, torch.cuda.current_stream().cuda_stream)
        if err != runtime.cudaError_t.cudaSuccess:
            raise RuntimeError(f"Failed to map graphics resource: {err}")

        # Get mapped device pointer
        err, pbo_dev_ptr, pbo_size = runtime.cudaGraphicsResourceGetMappedPointer(self._cuda_gl_resource)
        if err != runtime.cudaError_t.cudaSuccess:
            raise RuntimeError(f"Failed to get mapped pointer: {err}")

        # Get PyTorch tensor's CUDA device pointer
        tensor_ptr = tensor.data_ptr()

        # Copy directly from tensor to PBO (device-to-device, no CPU involved)
        (err,) = runtime.cudaMemcpy(pbo_dev_ptr, tensor_ptr, self._pbo_size, runtime.cudaMemcpyKind.cudaMemcpyDeviceToDevice)
        if err != runtime.cudaError_t.cudaSuccess:
            raise RuntimeError(f"Failed to copy to PBO: {err}")

        # Unmap graphics resource
        (err,) = runtime.cudaGraphicsUnmapResources(1, self._cuda_gl_resource, torch.cuda.current_stream().cuda_stream)
        if err != runtime.cudaError_t.cudaSuccess:
            raise RuntimeError(f"Failed to unmap graphics resource: {err}")

        # Select upload format based on channel count
        format_map = {
            1: GL_RED,
            2: GL_RG,
            3: GL_RGB,
            4: GL_RGBA
        }
        upload_format = format_map.get(self._channels, GL_RED)

        # Upload PBO to texture (GPU-to-GPU via OpenGL driver)
        glBindBuffer(GL_PIXEL_UNPACK_BUFFER, self._pbo)
        self.bind()
        glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, self.width, self.height, upload_format, GL_FLOAT, None)

        # Check for OpenGL errors
        err = glGetError()
        if err != 0:
            print(f"TensorTexture: OpenGL error after glTexSubImage2D: {err} (channels={self._channels}, format={upload_format})")

        self.unbind()
        glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0)

    def _update_with_cpu(self, tensor: torch.Tensor) -> None:
        """Fallback: CPU transfer (GPU → CPU → GPU)."""
        # Permute multi-channel tensors from (C, H, W) to (H, W, C)
        if len(tensor.shape) == 3 and tensor.shape[0] <= 4:
            tensor = tensor.permute(1, 2, 0)

        # Convert to float32 and move to CPU
        tensor_np = tensor.float().cpu().numpy()  # (H, W) or (H, W, C)

        # Select upload format based on channel count
        format_map = {
            1: GL_RED,
            2: GL_RG,
            3: GL_RGB,
            4: GL_RGBA
        }
        upload_format = format_map.get(self._channels, GL_RED)

        # Upload to OpenGL texture
        self.bind()
        glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, self.width, self.height, upload_format, GL_FLOAT, tensor_np)
        self.unbind()

    def draw(self, x, y, w, h) -> None:
        """Draw the texture with vertical flip (matching Image behavior)."""
        self.bind()
        draw_quad(x, y, w, h)
        self.unbind()

    def clear(self) -> None:
        """Clear texture to black (all zeros)."""
        if self.allocated and self.tex_id > 0:
            try:
                format_map = {
                    1: GL_RED,
                    2: GL_RG,
                    3: GL_RGB,
                    4: GL_RGBA
                }
                clear_format = format_map.get(self._channels, GL_RED)

                # Bind texture and clear with zeros
                self.bind()
                zeros = np.zeros((self.height, self.width, self._channels), dtype=np.float32)
                glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, self.width, self.height, clear_format, GL_FLOAT, zeros)
                self.unbind()
            except Exception as e:
                print(f"TensorTexture.clear() error: {e}")
