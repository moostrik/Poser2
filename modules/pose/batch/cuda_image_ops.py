"""CUDA kernels for fast GPU image processing.

Provides:
- Bilinear interpolation resize using CuPy RawKernel for maximum performance
- BGR to RGB channel conversion (in-place)

Supports both batched operations (for TRT inference classes) and single-image operations
(for GPUCropProcessor which has variable input sizes).
"""

import numpy as np
import cupy as cp


# CUDA kernel for fast BGR->RGB conversion (swaps channels 0 and 2 in-place)
_BGR_TO_RGB_KERNEL = cp.RawKernel(r'''
extern "C" __global__
void bgr_to_rgb_inplace(
    unsigned char* __restrict__ img,
    const int height, const int width
) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) return;

    const int idx = (y * width + x) * 3;

    // Swap B and R channels
    const unsigned char tmp = img[idx];      // B
    img[idx] = img[idx + 2];                 // B <- R
    img[idx + 2] = tmp;                      // R <- B
}
''', 'bgr_to_rgb_inplace')


# CUDA kernel for fast bilinear resize - handles both batched and single images
_BILINEAR_RESIZE_KERNEL = cp.RawKernel(r'''
extern "C" __global__
void bilinear_resize_batch(
    const unsigned char* __restrict__ src,
    unsigned char* __restrict__ dst,
    const int batch_size,
    const int src_h, const int src_w,
    const int dst_h, const int dst_w
) {
    // Each thread handles one output pixel (x, y) across batch items
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    const int b = blockIdx.z;  // batch index

    if (x >= dst_w || y >= dst_h || b >= batch_size) return;

    // Scale factors
    const float scale_x = (float)src_w / (float)dst_w;
    const float scale_y = (float)src_h / (float)dst_h;

    // Map to source coordinates
    const float src_x = x * scale_x;
    const float src_y = y * scale_y;

    // Integer coordinates and fractions
    const int x0 = (int)src_x;
    const int y0 = (int)src_y;
    const int x1 = min(x0 + 1, src_w - 1);
    const int y1 = min(y0 + 1, src_h - 1);

    const float fx = src_x - x0;
    const float fy = src_y - y0;

    // Source and destination strides
    const int src_stride = src_h * src_w * 3;
    const int dst_stride = dst_h * dst_w * 3;

    // Base pointers for this batch item
    const unsigned char* src_batch = src + b * src_stride;
    unsigned char* dst_batch = dst + b * dst_stride;

    // Bilinear interpolation for each channel
    #pragma unroll
    for (int c = 0; c < 3; c++) {
        const float tl = src_batch[(y0 * src_w + x0) * 3 + c];
        const float tr = src_batch[(y0 * src_w + x1) * 3 + c];
        const float bl = src_batch[(y1 * src_w + x0) * 3 + c];
        const float br = src_batch[(y1 * src_w + x1) * 3 + c];

        const float top = tl * (1.0f - fx) + tr * fx;
        const float bot = bl * (1.0f - fx) + br * fx;
        const float val = top * (1.0f - fy) + bot * fy;

        dst_batch[(y * dst_w + x) * 3 + c] = (unsigned char)(val + 0.5f);
    }
}
''', 'bilinear_resize_batch')


def batched_bilinear_resize(
    batch_src: cp.ndarray,
    dst_h: int,
    dst_w: int,
    stream: cp.cuda.Stream | None = None
) -> cp.ndarray:
    """Resize batch of images using CUDA kernel.

    Args:
        batch_src: (B, H, W, 3) uint8 source images
        dst_h: Target height
        dst_w: Target width
        stream: Optional CUDA stream (if None, uses current stream context)

    Returns:
        (B, dst_h, dst_w, 3) uint8 resized images
    """
    batch_size, src_h, src_w = batch_src.shape[:3]

    # Ensure contiguous for kernel
    if not batch_src.flags.c_contiguous:
        batch_src = cp.ascontiguousarray(batch_src)

    # Allocate output
    batch_dst = cp.empty((batch_size, dst_h, dst_w, 3), dtype=cp.uint8)

    # Launch kernel
    block = (16, 16, 1)
    grid = (
        (dst_w + block[0] - 1) // block[0],
        (dst_h + block[1] - 1) // block[1],
        batch_size
    )

    # Launch on provided stream or current stream context
    _BILINEAR_RESIZE_KERNEL(
        grid, block,
        (batch_src, batch_dst,
         np.int32(batch_size),
         np.int32(src_h), np.int32(src_w),
         np.int32(dst_h), np.int32(dst_w)),
        stream=stream
    )

    return batch_dst


def batched_bilinear_resize_inplace(
    batch_src: cp.ndarray,
    batch_dst: cp.ndarray,
    stream: cp.cuda.Stream | None = None
) -> None:
    """Resize batch of images into pre-allocated buffer using CUDA kernel.

    Avoids allocation overhead when destination buffer is reused.

    Args:
        batch_src: (B, H, W, 3) uint8 source images
        batch_dst: (B, dst_h, dst_w, 3) uint8 pre-allocated destination
        stream: Optional CUDA stream
    """
    batch_size, src_h, src_w = batch_src.shape[:3]
    dst_h, dst_w = batch_dst.shape[1:3]

    # Ensure contiguous for kernel
    if not batch_src.flags.c_contiguous:
        batch_src = cp.ascontiguousarray(batch_src)

    # Launch kernel
    block = (16, 16, 1)
    grid = (
        (dst_w + block[0] - 1) // block[0],
        (dst_h + block[1] - 1) // block[1],
        batch_size
    )

    _BILINEAR_RESIZE_KERNEL(
        grid, block,
        (batch_src, batch_dst,
         np.int32(batch_size),
         np.int32(src_h), np.int32(src_w),
         np.int32(dst_h), np.int32(dst_w)),
        stream=stream
    )


def bilinear_resize_single(
    src: cp.ndarray,
    dst_h: int,
    dst_w: int,
    stream: cp.cuda.Stream | None = None
) -> cp.ndarray:
    """Fast single-image bilinear resize using CUDA kernel.

    Wraps the batched kernel with batch_size=1.

    Args:
        src: Source image (H, W, 3) uint8 on GPU
        dst_h: Target height
        dst_w: Target width
        stream: Optional CUDA stream (uses default if None)

    Returns:
        Resized image (dst_h, dst_w, 3) uint8 on GPU
    """
    src_h, src_w = src.shape[:2]

    # Ensure contiguous
    if not src.flags.c_contiguous:
        src = cp.ascontiguousarray(src)

    # Allocate output
    dst = cp.empty((dst_h, dst_w, 3), dtype=cp.uint8)

    # Launch kernel with batch_size=1
    block = (16, 16, 1)
    grid = (
        (dst_w + block[0] - 1) // block[0],
        (dst_h + block[1] - 1) // block[1],
        1
    )

    _BILINEAR_RESIZE_KERNEL(
        grid, block,
        (src, dst,
         np.int32(1),
         np.int32(src_h), np.int32(src_w),
         np.int32(dst_h), np.int32(dst_w)),
        stream=stream
    )

    return dst


def bilinear_resize_inplace(
    src: cp.ndarray,
    dst: cp.ndarray,
    stream: cp.cuda.Stream | None = None
) -> None:
    """Fast single-image bilinear resize into pre-allocated buffer.

    Useful when destination buffer is reused across frames.

    Args:
        src: Source image (H, W, 3) uint8 on GPU
        dst: Destination buffer (dst_h, dst_w, 3) uint8 on GPU (will be overwritten)
        stream: Optional CUDA stream (uses default if None)
    """
    src_h, src_w = src.shape[:2]
    dst_h, dst_w = dst.shape[:2]

    # Ensure contiguous
    if not src.flags.c_contiguous:
        src = cp.ascontiguousarray(src)

    # Launch kernel with batch_size=1
    block = (16, 16, 1)
    grid = (
        (dst_w + block[0] - 1) // block[0],
        (dst_h + block[1] - 1) // block[1],
        1
    )

    _BILINEAR_RESIZE_KERNEL(
        grid, block,
        (src, dst,
         np.int32(1),
         np.int32(src_h), np.int32(src_w),
         np.int32(dst_h), np.int32(dst_w)),
        stream=stream
    )


def bgr_to_rgb_inplace(
    img: cp.ndarray,
    stream: cp.cuda.Stream | None = None
) -> None:
    """Convert BGR image to RGB in-place on GPU.

    Swaps the B and R channels directly in memory, avoiding any copies.
    Much faster than uploading a non-contiguous BGR→RGB view from CPU.

    Args:
        img: Image (H, W, 3) uint8 on GPU - modified in-place
        stream: Optional CUDA stream
    """
    height, width = img.shape[:2]

    block = (16, 16, 1)
    grid = (
        (width + block[0] - 1) // block[0],
        (height + block[1] - 1) // block[1],
        1
    )

    _BGR_TO_RGB_KERNEL(
        grid, block,
        (img, np.int32(height), np.int32(width)),
        stream=stream
    )


# CUDA kernel for fused normalize + HWC->CHW conversion (FP16 output)
_NORMALIZE_HWC_TO_CHW_FP16_KERNEL = cp.RawKernel(r'''
#include <cuda_fp16.h>

extern "C" __global__
void normalize_hwc_to_chw_fp16(
    const unsigned char* __restrict__ src,
    __half* __restrict__ dst,
    const int batch_size,
    const int height,
    const int width
) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    const int b = blockIdx.z;

    if (x >= width || y >= height || b >= batch_size) return;

    const int src_idx = (b * height * width + y * width + x) * 3;
    const int hw = height * width;
    const int dst_base = b * 3 * hw + y * width + x;

    // Read RGB, normalize to [0,1], write as CHW
    dst[dst_base + 0 * hw] = __float2half(src[src_idx + 0] / 255.0f);  // R
    dst[dst_base + 1 * hw] = __float2half(src[src_idx + 1] / 255.0f);  // G
    dst[dst_base + 2 * hw] = __float2half(src[src_idx + 2] / 255.0f);  // B
}
''', 'normalize_hwc_to_chw_fp16')


# CUDA kernel for fused normalize + HWC->CHW conversion (FP32 output)
_NORMALIZE_HWC_TO_CHW_FP32_KERNEL = cp.RawKernel(r'''
extern "C" __global__
void normalize_hwc_to_chw_fp32(
    const unsigned char* __restrict__ src,
    float* __restrict__ dst,
    const int batch_size,
    const int height,
    const int width
) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    const int b = blockIdx.z;

    if (x >= width || y >= height || b >= batch_size) return;

    const int src_idx = (b * height * width + y * width + x) * 3;
    const int hw = height * width;
    const int dst_base = b * 3 * hw + y * width + x;

    // Read RGB, normalize to [0,1], write as CHW
    dst[dst_base + 0 * hw] = src[src_idx + 0] / 255.0f;  // R
    dst[dst_base + 1 * hw] = src[src_idx + 1] / 255.0f;  // G
    dst[dst_base + 2 * hw] = src[src_idx + 2] / 255.0f;  // B
}
''', 'normalize_hwc_to_chw_fp32')


def normalize_hwc_to_chw_inplace(
    batch_src: cp.ndarray,
    batch_dst: cp.ndarray,
    stream: cp.cuda.Stream | None = None
) -> None:
    """Convert uint8 HWC batch to float CHW batch with normalization.

    Fused kernel: reads (B, H, W, 3) uint8, writes (B, 3, H, W) float16/float32.
    Normalizes to [0, 1] range. Eliminates intermediate allocations.

    Args:
        batch_src: (B, H, W, 3) uint8 source
        batch_dst: (B, 3, H, W) float16 or float32 destination (preallocated)
        stream: Optional CUDA stream
    """
    batch_size, height, width = batch_src.shape[:3]

    # Ensure contiguous for kernel
    if not batch_src.flags.c_contiguous:
        batch_src = cp.ascontiguousarray(batch_src)

    block = (16, 16, 1)
    grid = (
        (width + block[0] - 1) // block[0],
        (height + block[1] - 1) // block[1],
        batch_size
    )

    # Select kernel based on output dtype
    if batch_dst.dtype == cp.float16:
        _NORMALIZE_HWC_TO_CHW_FP16_KERNEL(
            grid, block,
            (batch_src, batch_dst,
             np.int32(batch_size), np.int32(height), np.int32(width)),
            stream=stream
        )
    else:
        _NORMALIZE_HWC_TO_CHW_FP32_KERNEL(
            grid, block,
            (batch_src, batch_dst,
             np.int32(batch_size), np.int32(height), np.int32(width)),
            stream=stream
        )


# ============================================================================
# ImageNet Normalization + HWC->CHW Kernel (for Detection)
# ============================================================================

_IMAGENET_NORMALIZE_HWC_TO_CHW_FP16_KERNEL = cp.RawKernel(r'''
#include <cuda_fp16.h>
extern "C" __global__
void imagenet_normalize_hwc_to_chw_fp16(
    const unsigned char* __restrict__ src,
    __half* __restrict__ dst,
    const int batch_size,
    const int height,
    const int width,
    const float mean_r, const float mean_g, const float mean_b,
    const float std_r, const float std_g, const float std_b
) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    const int b = blockIdx.z;

    if (x >= width || y >= height || b >= batch_size) return;

    const int src_idx = (b * height * width + y * width + x) * 3;
    const int hw = height * width;
    const int dst_base = b * 3 * hw + y * width + x;

    // Read RGB, apply ImageNet normalization, write as CHW
    float r = (float)src[src_idx + 0];
    float g = (float)src[src_idx + 1];
    float b_val = (float)src[src_idx + 2];

    dst[dst_base + 0 * hw] = __float2half((r - mean_r) / std_r);
    dst[dst_base + 1 * hw] = __float2half((g - mean_g) / std_g);
    dst[dst_base + 2 * hw] = __float2half((b_val - mean_b) / std_b);
}
''', 'imagenet_normalize_hwc_to_chw_fp16')


_IMAGENET_NORMALIZE_HWC_TO_CHW_FP32_KERNEL = cp.RawKernel(r'''
extern "C" __global__
void imagenet_normalize_hwc_to_chw_fp32(
    const unsigned char* __restrict__ src,
    float* __restrict__ dst,
    const int batch_size,
    const int height,
    const int width,
    const float mean_r, const float mean_g, const float mean_b,
    const float std_r, const float std_g, const float std_b
) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    const int b = blockIdx.z;

    if (x >= width || y >= height || b >= batch_size) return;

    const int src_idx = (b * height * width + y * width + x) * 3;
    const int hw = height * width;
    const int dst_base = b * 3 * hw + y * width + x;

    // Read RGB, apply ImageNet normalization, write as CHW
    float r = (float)src[src_idx + 0];
    float g = (float)src[src_idx + 1];
    float b_val = (float)src[src_idx + 2];

    dst[dst_base + 0 * hw] = (r - mean_r) / std_r;
    dst[dst_base + 1 * hw] = (g - mean_g) / std_g;
    dst[dst_base + 2 * hw] = (b_val - mean_b) / std_b;
}
''', 'imagenet_normalize_hwc_to_chw_fp32')


def imagenet_normalize_hwc_to_chw_inplace(
    batch_src: cp.ndarray,
    batch_dst: cp.ndarray,
    mean: tuple[float, float, float] = (123.675, 116.28, 103.53),
    std: tuple[float, float, float] = (58.395, 57.12, 57.375),
    stream: cp.cuda.Stream | None = None
) -> None:
    """Convert uint8 HWC batch to float CHW batch with ImageNet normalization.

    Fused kernel: reads (B, H, W, 3) uint8, applies (x - mean) / std per channel,
    writes (B, 3, H, W) float16/float32.

    Args:
        batch_src: (B, H, W, 3) uint8 source (RGB order)
        batch_dst: (B, 3, H, W) float16/float32 destination (preallocated)
        mean: Per-channel mean (R, G, B) - default ImageNet values
        std: Per-channel std (R, G, B) - default ImageNet values
        stream: Optional CUDA stream
    """
    batch_size, height, width = batch_src.shape[:3]

    # Ensure contiguous for kernel
    if not batch_src.flags.c_contiguous:
        batch_src = cp.ascontiguousarray(batch_src)

    block = (16, 16, 1)
    grid = (
        (width + block[0] - 1) // block[0],
        (height + block[1] - 1) // block[1],
        batch_size
    )

    args = (
        batch_src, batch_dst,
        np.int32(batch_size), np.int32(height), np.int32(width),
        np.float32(mean[0]), np.float32(mean[1]), np.float32(mean[2]),
        np.float32(std[0]), np.float32(std[1]), np.float32(std[2])
    )

    if batch_dst.dtype == cp.float16:
        _IMAGENET_NORMALIZE_HWC_TO_CHW_FP16_KERNEL(grid, block, args, stream=stream)
    else:
        _IMAGENET_NORMALIZE_HWC_TO_CHW_FP32_KERNEL(grid, block, args, stream=stream)
