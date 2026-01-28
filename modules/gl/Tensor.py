# Standard library imports
from threading import Lock

# Third-party imports
import numpy as np
import torch
from cuda.bindings import runtime # type: ignore
from OpenGL.GL import * # type: ignore

# Local application imports
from modules.gl.Texture import Texture


def get_channel_count_from_format(internal_format) -> int:
    """Get number of channels from OpenGL internal format.

    Args:
        internal_format: OpenGL internal format constant

    Returns:
        Number of channels (1-4)
    """
    if internal_format in (GL_R8, GL_R16F, GL_R32F):
        return 1
    elif internal_format in (GL_RG8, GL_RG16F, GL_RG32F):
        return 2
    elif internal_format in (GL_RGB, GL_RGB8, GL_RGB16F, GL_RGB32F):
        return 3
    elif internal_format in (GL_RGBA, GL_RGBA8, GL_RGBA16F, GL_RGBA32F):
        return 4
    return 1  # fallback

def get_bytes_per_channel(internal_format) -> int:
    """Get bytes per channel from OpenGL internal format.

    Args:
        internal_format: OpenGL internal format constant

    Returns:
        Bytes per channel (1 for uint8, 2 for FP16, 4 for FP32)
    """
    if internal_format in (GL_R8, GL_RG8, GL_RGB8, GL_RGBA8):
        return 1  # uint8
    elif internal_format in (GL_R16F, GL_RG16F, GL_RGB16F, GL_RGBA16F):
        return 2  # FP16
    elif internal_format in (GL_R32F, GL_RG32F, GL_RGB32F, GL_RGBA32F):
        return 4  # FP32
    return 4  # fallback to FP32

def get_gl_type_from_format(internal_format) -> int:
    """Get OpenGL type constant from internal format.

    Args:
        internal_format: OpenGL internal format constant

    Returns:
        GL_UNSIGNED_BYTE, GL_HALF_FLOAT, or GL_FLOAT
    """
    if internal_format in (GL_R8, GL_RG8, GL_RGB8, GL_RGBA8):
        return GL_UNSIGNED_BYTE
    elif internal_format in (GL_R16F, GL_RG16F, GL_RGB16F, GL_RGBA16F):
        return GL_HALF_FLOAT
    return GL_FLOAT

def infer_internal_format(tensor: torch.Tensor) -> int:
    """Infer OpenGL internal format from tensor shape and dtype.

    Args:
        tensor: PyTorch tensor (H, W) or (C, H, W) or (H, W, C)

    Returns:
        OpenGL internal format constant
    """
    # Detect channel count
    if len(tensor.shape) == 2:
        channels = 1
    elif len(tensor.shape) == 3:
        if tensor.shape[0] <= 4:  # (C, H, W)
            channels = tensor.shape[0]
        else:  # (H, W, C)
            channels = tensor.shape[2]
    else:
        channels = 1

    # Determine precision from dtype
    if tensor.dtype == torch.uint8:
        format_map = {1: GL_R8, 2: GL_RG8, 3: GL_RGB8, 4: GL_RGBA8}
    elif tensor.dtype == torch.float16:
        format_map = {1: GL_R16F, 2: GL_RG16F, 3: GL_RGB16F, 4: GL_RGBA16F}
    else:  # torch.float32 or other
        format_map = {1: GL_R32F, 2: GL_RG32F, 3: GL_RGB32F, 4: GL_RGBA32F}

    return format_map.get(channels, GL_R32F)


class Tensor(Texture):
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

    def allocate(self, width: int, height: int, internal_format,
                 wrap_s: int = GL_CLAMP_TO_EDGE,
                 wrap_t: int = GL_CLAMP_TO_EDGE,
                 min_filter: int = GL_LINEAR,
                 mag_filter: int = GL_LINEAR) -> None:
        """Allocate texture and PBO for CUDA-OpenGL interop.

        Args:
            width: Texture width in pixels
            height: Texture height in pixels
            internal_format: OpenGL internal format (GL_R8, GL_R16F, GL_R32F, GL_RGB8, GL_RGB16F, GL_RGB32F, etc.)
            wrap_s: Horizontal wrap mode (default: GL_CLAMP_TO_EDGE)
            wrap_t: Vertical wrap mode (default: GL_CLAMP_TO_EDGE)
            min_filter: Minification filter (default: GL_LINEAR)
            mag_filter: Magnification filter (default: GL_LINEAR)
        """
        # Allocate texture via parent class
        super().allocate(width, height, internal_format, wrap_s, wrap_t, min_filter, mag_filter)
        if not self.allocated:
            return

        # Get channel count and bytes per channel
        channels = get_channel_count_from_format(internal_format)
        bytes_per_channel = get_bytes_per_channel(internal_format)

        # Allocate PBO
        self._pbo = glGenBuffers(1)
        self._pbo_size = width * height * channels * bytes_per_channel
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
                print(f"TensorTexture: CUDA-GL interop registration failed (error {err}), using CPU fallback")
                self._cuda_gl_resource = None
        except Exception as e:
            print(f"TensorTexture: CUDA-GL interop registration failed ({e}), using CPU fallback")
            self._cuda_gl_resource = None

    def deallocate(self) -> None:
        """Deallocate texture and PBO resources."""
        # Clean up PBO
        if self._pbo > 0:
            glDeleteBuffers(1, [self._pbo])
            self._pbo = 0
            self._pbo_size = 0

        # Unregister CUDA resource
        if self._cuda_gl_resource is not None:
            runtime.cudaGraphicsUnregisterResource(self._cuda_gl_resource)
            self._cuda_gl_resource = None

        super().deallocate()

    def update(self) -> None:
        """Upload pending tensor to GPU texture.

        Auto-reallocates if tensor dimensions or format change.
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
            # Detect tensor dimensions
            if len(tensor.shape) == 2:  # (H, W)
                height, width = tensor.shape
            elif len(tensor.shape) == 3:
                if tensor.shape[0] <= 4:  # (C, H, W)
                    channels, height, width = tensor.shape
                else:  # (H, W, C)
                    height, width, channels = tensor.shape
            else:
                raise ValueError(f"Unsupported tensor shape: {tensor.shape}")

            # Infer internal format from tensor
            internal_format = infer_internal_format(tensor)

            # Reallocate if dimensions or format changed
            if width != self.width or height != self.height or internal_format != self.internal_format or not self.allocated:
                if self.allocated:
                    self.deallocate()
                self.allocate(width, height, internal_format)

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

    def _update_with_pbo(self, tensor: torch.Tensor) -> None:
        """GPU-to-GPU copy using CUDA-OpenGL interop.

        PyTorch CUDA tensor → PBO (device-to-device) → GL texture (all on GPU).
        Preserves uint8, FP16, or FP32 precision based on internal format.
        """
        # Convert to appropriate dtype based on internal format
        bytes_per_channel = get_bytes_per_channel(self.internal_format)

        if bytes_per_channel == 1:  # uint8
            target_dtype = torch.uint8
            if tensor.dtype in (torch.float16, torch.float32):
                # Convert float [0.0, 1.0] to uint8 [0, 255]
                tensor = (tensor.clamp(0.0, 1.0) * 255.0).to(torch.uint8)
            elif tensor.dtype != torch.uint8:
                tensor = tensor.to(torch.uint8)
        elif bytes_per_channel == 2:  # FP16
            target_dtype = torch.float16
            if tensor.dtype == torch.uint8:
                tensor = tensor.float() / 255.0
                tensor = tensor.half()
            elif tensor.dtype != torch.float16:
                tensor = tensor.half()
        else:  # FP32
            target_dtype = torch.float32
            if tensor.dtype == torch.uint8:
                tensor = tensor.float() / 255.0
            elif tensor.dtype != torch.float32:
                tensor = tensor.float()

        # Permute multi-channel tensors from (C, H, W) to (H, W, C)
        if len(tensor.shape) == 3 and tensor.shape[0] <= 4:
            tensor = tensor.permute(1, 2, 0)

        # Flip vertically to match OpenGL/FBO/Image convention
        tensor = torch.flip(tensor, [0])

        # Ensure contiguous memory for GPU copy
        tensor = tensor.contiguous()

        # Synchronize to ensure all prior CUDA operations are complete
        # This is critical when tensor comes from TensorRT/CuPy on different streams
        torch.cuda.synchronize()

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

        # Get channel count and GL type
        channels = get_channel_count_from_format(self.internal_format)
        gl_type = get_gl_type_from_format(self.internal_format)

        # Select upload format based on channel count
        format_map = {1: GL_RED, 2: GL_RG, 3: GL_RGB, 4: GL_RGBA}
        upload_format = format_map.get(channels, GL_RED)

        # Upload PBO to texture (GPU-to-GPU via OpenGL driver)
        glBindBuffer(GL_PIXEL_UNPACK_BUFFER, self._pbo)
        self.bind()
        glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, self.width, self.height, upload_format, gl_type, None)

        # Check for OpenGL errors
        err = glGetError()
        if err != 0:
            type_name = 'uint8' if gl_type == GL_UNSIGNED_BYTE else ('FP16' if gl_type == GL_HALF_FLOAT else 'FP32')
            print(f"TensorTexture: OpenGL error after glTexSubImage2D: {err} (channels={channels}, format={upload_format}, type={type_name})")

        self.unbind()
        glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0)

    def _update_with_cpu(self, tensor: torch.Tensor) -> None:
        """Fallback: CPU transfer (GPU → CPU → GPU).

        Preserves uint8, FP16, or FP32 precision based on internal format.
        """
        # Permute multi-channel tensors from (C, H, W) to (H, W, C)
        if len(tensor.shape) == 3 and tensor.shape[0] <= 4:
            tensor = tensor.permute(1, 2, 0)

        # Flip vertically to match OpenGL/FBO/Image convention
        # After permute, shape is (H, W) or (H, W, C), so flip dimension [0] (height)
        tensor = torch.flip(tensor, [0])

        # Convert to appropriate dtype and move to CPU
        bytes_per_channel = get_bytes_per_channel(self.internal_format)

        if bytes_per_channel == 1:  # uint8
            if tensor.dtype in (torch.float16, torch.float32):
                # Convert float [0.0, 1.0] to uint8 [0, 255]
                tensor_np = (tensor.clamp(0.0, 1.0) * 255.0).to(torch.uint8).cpu().numpy()
            else:
                tensor_np = tensor.to(torch.uint8).cpu().numpy()
        elif bytes_per_channel == 2:  # FP16
            if tensor.dtype == torch.uint8:
                tensor_np = (tensor.float() / 255.0).half().cpu().numpy()
            else:
                tensor_np = tensor.half().cpu().numpy()
        else:  # FP32
            if tensor.dtype == torch.uint8:
                tensor_np = (tensor.float() / 255.0).cpu().numpy()
            else:
                tensor_np = tensor.float().cpu().numpy()

        # Get channel count and GL type
        channels = get_channel_count_from_format(self.internal_format)
        gl_type = get_gl_type_from_format(self.internal_format)

        # Select upload format based on channel count
        format_map = {1: GL_RED, 2: GL_RG, 3: GL_RGB, 4: GL_RGBA}
        upload_format = format_map.get(channels, GL_RED)

        # Upload to OpenGL texture
        self.bind()
        glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, self.width, self.height, upload_format, gl_type, tensor_np)
        self.unbind()

