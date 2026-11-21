# Standard library imports
import logging
from threading import Lock
from typing import TYPE_CHECKING

logger = logging.getLogger(__name__)

# Third-party imports
import numpy as np
try:
    import torch
    from . import cudart   # binds torch's CUDA runtime; fails without a CUDA torch
    _GPU_OK = True
except (ImportError, OSError):
    _GPU_OK = False

if TYPE_CHECKING:
    import torch

from OpenGL.GL import Constant, GL_CLAMP_TO_EDGE, GL_LINEAR, GL_PIXEL_UNPACK_BUFFER, GL_R16F, GL_R32F, GL_R8, GL_RED, GL_RG, GL_RG16F, GL_RG32F, GL_RG8, GL_RGB, GL_RGB16F, GL_RGB32F, GL_RGB8, GL_RGBA, GL_RGBA16F, GL_RGBA32F, GL_RGBA8, GL_STREAM_DRAW, glBindBuffer, glBufferData, glDeleteBuffers, glGenBuffers, glTextureSubImage2D

# Local application imports
from .texture import Texture
from .texture_format import bytes_per_channel as format_bytes_per_channel, channel_count, tensor_type


def infer_internal_format(tensor: "torch.Tensor") -> Constant:
    """Infer OpenGL internal format from tensor shape and dtype.

    Args:
        tensor: PyTorch tensor (H, W) or (C, H, W) or (H, W, C)

    Returns:
        OpenGL internal format constant
    """
    channels: int = interleaved(tensor).shape[2] if len(tensor.shape) == 3 else 1

    # Determine precision from dtype
    if tensor.dtype == torch.uint8:
        format_map = {1: GL_R8, 2: GL_RG8, 3: GL_RGB8, 4: GL_RGBA8}
    elif tensor.dtype == torch.float16:
        format_map = {1: GL_R16F, 2: GL_RG16F, 3: GL_RGB16F, 4: GL_RGBA16F}
    else:  # torch.float32 or other
        format_map = {1: GL_R32F, 2: GL_RG32F, 3: GL_RGB32F, 4: GL_RGBA32F}

    return format_map.get(channels, GL_R32F)


def interleaved(tensor: "torch.Tensor") -> "torch.Tensor":
    """The tensor with its channels last, as OpenGL takes them: (H, W) and (H, W, C) as they are, (C, H, W), C up to
    4, permuted to (H, W, C)."""
    if len(tensor.shape) == 3 and tensor.shape[0] <= 4:
        return tensor.permute(1, 2, 0)
    return tensor


class Tensor(Texture):
    """GPU texture updated directly from CUDA tensors using CUDA-OpenGL interop.

    Provides zero-copy GPU-to-GPU transfer from PyTorch CUDA tensors to OpenGL textures
    via Pixel Buffer Objects (PBO) and CUDA graphics resource registration.
    Falls back to CPU transfer if CUDA-OpenGL interop is unavailable.
    """

    def __init__(self,
                 interpolation: int = GL_LINEAR,
                 wrap: int = GL_CLAMP_TO_EDGE,
                 border_color: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 1.0)) -> None:
        """Initialize Tensor texture with configuration.

        Args:
            interpolation: Filter mode for min/mag (GL_LINEAR or GL_NEAREST)
            wrap: Wrap mode for both axes (GL_CLAMP_TO_EDGE, GL_REPEAT, GL_MIRRORED_REPEAT, GL_CLAMP_TO_BORDER)
            border_color: RGBA color when wrap is GL_CLAMP_TO_BORDER
        """
        super().__init__(interpolation, wrap, border_color)
        self._tensor: torch.Tensor | None = None
        self._needs_update: bool = False
        self._mutex: Lock = Lock()

        # PBO for GPU-to-GPU texture updates
        self._pbo: int = 0
        self._pbo_size: int = 0

        # CUDA-OpenGL interop
        self._cuda_gl_resource = None

    def set_tensor(self, tensor: "torch.Tensor") -> None:
        """Set a new CUDA tensor to be uploaded to the texture.

        Args:
            tensor: PyTorch tensor on CUDA device.
                   Shape: (H, W) for 1-channel, (C, H, W) or (H, W, C) for multi-channel.
                   Supported dtypes: float16, float32, uint8.
        """
        if not _GPU_OK:
            return
        with self._mutex:
            self._tensor = tensor
            self._needs_update = True

    def allocate(self, width: int, height: int, internal_format: Constant | int) -> None:
        """Allocate texture and PBO for CUDA-OpenGL interop.

        Args:
            width: Texture width in pixels
            height: Texture height in pixels
            internal_format: OpenGL internal format (GL_R8, GL_R16F, GL_R32F, GL_RGB8, GL_RGB16F, GL_RGB32F, etc.)
        """
        # Allocate texture via parent class
        super().allocate(width, height, internal_format)
        if not self.allocated:
            return

        # Get channel count and bytes per channel
        channels = channel_count(internal_format)
        bytes_per_channel = format_bytes_per_channel(internal_format)

        # Allocate PBO
        self._pbo = glGenBuffers(1)
        self._pbo_size = width * height * channels * bytes_per_channel
        glBindBuffer(GL_PIXEL_UNPACK_BUFFER, self._pbo)
        glBufferData(GL_PIXEL_UNPACK_BUFFER, self._pbo_size, None, GL_STREAM_DRAW)
        glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0)

        # Register PBO with CUDA for interop
        if _GPU_OK:
            try:
                self._cuda_gl_resource = cudart.register_gl_buffer(self._pbo)
            except Exception as e:
                logger.warning("CUDA-GL interop registration failed (%s), using CPU fallback", e)
                self._cuda_gl_resource = None

    def deallocate(self) -> None:
        """Deallocate texture and PBO resources."""
        # Clean up PBO
        if self._pbo > 0:
            glDeleteBuffers(1, [self._pbo])
            self._pbo = 0
            self._pbo_size = 0

        # Unregister CUDA resource
        if _GPU_OK and self._cuda_gl_resource is not None:
            try:
                cudart.unregister_resource(self._cuda_gl_resource)
            except cudart.CudaError as e:
                logger.warning("CUDA-GL interop unregister failed: %s", e)
            self._cuda_gl_resource = None

        super().deallocate()

    def update(self) -> None:
        """Upload pending tensor to GPU texture.

        Auto-reallocates if tensor dimensions or format change.
        Uses CUDA-OpenGL interop for GPU-to-GPU transfer when available,
        falls back to CPU transfer otherwise.
        """
        if not _GPU_OK:
            return
        tensor: torch.Tensor | None = None
        needs_update: bool = False

        with self._mutex:
            tensor = self._tensor
            needs_update = self._needs_update
            self._needs_update = False

        if not needs_update or tensor is None:
            return

        try:
            if len(tensor.shape) not in (2, 3):
                raise ValueError(f"Unsupported tensor shape: {tensor.shape}")
            height, width = interleaved(tensor).shape[:2]

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
        except Exception:
            logger.exception("update() error, tensor shape: %s", tensor.shape if tensor is not None else 'None')
            glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0)     # an upload that raised leaves no PBO bound

    def _update_with_pbo(self, tensor: "torch.Tensor") -> None:
        """GPU-to-GPU copy using CUDA-OpenGL interop.

        PyTorch CUDA tensor → PBO (device-to-device) → GL texture (all on GPU).
        Preserves uint8, FP16, or FP32 precision based on internal format.
        """
        # Convert to appropriate dtype based on internal format
        bytes_per_channel = format_bytes_per_channel(self.internal_format)

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

        # Channels last, then flipped vertically to match OpenGL/FBO/Image convention
        tensor = torch.flip(interleaved(tensor), [0])

        # Ensure contiguous memory for GPU copy
        tensor = tensor.contiguous()

        # Map the PBO on the current stream, queue the copy of the tensor into it there, unmap: the copy runs after
        # the conversions above on the same stream, and the unmap has it done before OpenGL reads the PBO, so the
        # main thread queues it and goes on. A tensor written on another stream is waited for by its reader
        # (`stream.wait_event`) before it gets here.
        assert self._cuda_gl_resource is not None
        stream = torch.cuda.current_stream().cuda_stream
        cudart.map_resource(self._cuda_gl_resource, stream)
        try:
            pbo_dev_ptr, _ = cudart.mapped_pointer(self._cuda_gl_resource)
            cudart.memcpy_device_to_device(pbo_dev_ptr, tensor.data_ptr(), self._pbo_size, stream)
        finally:
            cudart.unmap_resource(self._cuda_gl_resource, stream)

        # Get channel count and GL type
        channels = channel_count(self.internal_format)
        gl_type = tensor_type(self.internal_format)

        # Select upload format based on channel count
        format_map = {1: GL_RED, 2: GL_RG, 3: GL_RGB, 4: GL_RGBA}
        upload_format = format_map.get(channels, GL_RED)

        # Upload PBO to texture (GPU-to-GPU via OpenGL driver); an error in it is logged at the call by GL's debug
        # output, behind the window manager's `gl_debug`
        glBindBuffer(GL_PIXEL_UNPACK_BUFFER, self._pbo)
        glTextureSubImage2D(self.tex_id, 0, 0, 0, self.width, self.height, upload_format, gl_type, None)
        glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0)

    def _update_with_cpu(self, tensor: "torch.Tensor") -> None:
        """Fallback: CPU transfer (GPU → CPU → GPU).

        Preserves uint8, FP16, or FP32 precision based on internal format.
        """
        # Channels last, then flipped vertically to match OpenGL/FBO/Image convention
        tensor = torch.flip(interleaved(tensor), [0])

        # Convert to appropriate dtype and move to CPU
        bytes_per_channel = format_bytes_per_channel(self.internal_format)

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
        channels = channel_count(self.internal_format)
        gl_type = tensor_type(self.internal_format)

        # Select upload format based on channel count
        format_map = {1: GL_RED, 2: GL_RG, 3: GL_RGB, 4: GL_RGBA}
        upload_format = format_map.get(channels, GL_RED)

        # Upload to OpenGL texture; glTextureSubImage2D takes only a contiguous array, a strided one is copied once
        glTextureSubImage2D(self.tex_id, 0, 0, 0, self.width, self.height, upload_format, gl_type,
                            np.ascontiguousarray(tensor_np))

