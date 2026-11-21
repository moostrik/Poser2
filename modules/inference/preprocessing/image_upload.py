"""ImageUpload: camera color frames onto the GPU, the work both uploaders share.

One frame, or a stack of frames of one size, layout and turn, goes through a pinned staging buffer to the GPU, is
converted there (NV12, BGR or mono) to float16 RGB CHW in [0, 1], straightened by its camera's homography, then
turned and flipped as its frame says (`CameraFrame`'s order: the turn, `flip_v`, `flip_h`), and comes out with one
event recorded after it on the caller's stream; a turned frame comes out with its width and height swapped. A stack
is one copy and one CUDA kernel compiled at runtime (`SOURCE`, `cuda_kernels`), which does it all per output pixel.
Where the kernel cannot be compiled (logged once), or with `kernel` off, the same runs as torch ops: the conversion
and the straightening over the whole stack (the cameras' grids stacked), the turn and the flips per frame; they are
the reference `scripts/gpu_input_timing.py` holds the kernel to. Neither the caller's thread nor this waits for the
GPU while it has room: a frame goes through its cameras' ring of staging buffers (`staging_ring.py`). Not
thread-safe: one uploader's thread uses it."""

import ctypes
import logging

import numpy as np
import torch
import torch.nn.functional as F

from modules.camera import CameraFrame

from .cuda_kernels import CudaKernelError, CudaKernels
from .staging_ring import StagingRing

logger = logging.getLogger(__name__)

MAX_STACK: int = 8          # the frames one kernel launch takes
BLOCK: tuple[int, int] = (16, 16)
NV12, BGR, MONO = 0, 1, 2   # the layouts the kernel reads

# One thread per output pixel of one frame of the stack (rows of all frames one after the other): the flips and the
# turn, undone in reverse, pick the straightened pixel, the straightening the place in the camera frame it came from
# (`warp_grid`'s, sampled bilinearly with zeros outside, as `grid_sample` does), and each of the four pixels around it
# is converted from the layout to RGB and rounded to float16, as the torch path converts before it straightens.
# Out: float16 RGB (N, 3, H, W), H and W swapped when the stack is turned.
SOURCE: str = r"""
#define MAX_STACK 8
struct Stack {
    float inverse[MAX_STACK][9];    // row-major 3x3: straightened uv to the camera frame's uv
    int straighten[MAX_STACK], flip_h[MAX_STACK], flip_v[MAX_STACK];
    int count, width, height, layout;   // the frames' own size; layout: 0 NV12, 1 BGR, 2 mono
    int rotate90;                       // every frame of the stack turned: out is height wide, width high
};

__device__ float rounded(float value) {     // to float16 and back
    unsigned short bits;
    asm("cvt.rn.f16.f32 %0, %1;" : "=h"(bits) : "f"(value));
    float back;
    asm("cvt.f32.f16 %0, %1;" : "=f"(back) : "h"(bits));
    return back;
}

__device__ void pixel(const unsigned char* frame, int x, int y, const Stack& s, float rgb[3]) {
    int w = s.width, h = s.height;
    if (s.layout == 0) {
        float luma = ((float)frame[y * w + x] - 16.0f) * 1.164383f;
        const unsigned char* uv = frame + (h + y / 2) * w + (x & ~1);
        float u = (float)uv[0] - 128.0f, v = (float)uv[1] - 128.0f;
        rgb[0] = fminf(fmaxf(luma + 1.596027f * v, 0.0f), 255.0f) / 255.0f;
        rgb[1] = fminf(fmaxf(luma - 0.391762f * u - 0.812968f * v, 0.0f), 255.0f) / 255.0f;
        rgb[2] = fminf(fmaxf(luma + 2.017232f * u, 0.0f), 255.0f) / 255.0f;
    } else if (s.layout == 1) {
        const unsigned char* bgr = frame + (y * w + x) * 3;
        rgb[0] = (float)bgr[2] * (1.0f / 255.0f);
        rgb[1] = (float)bgr[1] * (1.0f / 255.0f);
        rgb[2] = (float)bgr[0] * (1.0f / 255.0f);
    } else {
        rgb[0] = rgb[1] = rgb[2] = (float)frame[y * w + x] * (1.0f / 255.0f);
    }
    for (int c = 0; c < 3; ++c) rgb[c] = rounded(rgb[c]);
}

extern "C" __global__ void upload(const unsigned char* frames, unsigned short* out, Stack s) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int w = s.width, h = s.height;
    int ow = s.rotate90 ? h : w, oh = s.rotate90 ? w : h;      // the output's size
    if (x >= ow || row >= oh * s.count) return;
    int n = row / oh, y = row - n * oh;
    long frame_bytes = s.layout == 0 ? (long)w * h * 3 / 2 : s.layout == 1 ? (long)w * h * 3 : (long)w * h;
    const unsigned char* frame = frames + n * frame_bytes;
    int xf = s.flip_h[n] ? ow - 1 - x : x;      // the mirror undone
    int yf = s.flip_v[n] ? oh - 1 - y : y;      // the vertical flip undone
    int xs = s.rotate90 ? yf : xf;              // the turn undone: the straightened pixel this one shows
    int ys = s.rotate90 ? xf : yf;
    float rgb[3] = {0.0f, 0.0f, 0.0f};
    if (!s.straighten[n]) {
        pixel(frame, xs, ys, s, rgb);
    } else {
        const float* m = s.inverse[n];
        float u = (xs + 0.5f) / w, v = (ys + 0.5f) / h;
        float sz = m[6] * u + m[7] * v + m[8];
        if (sz > 0.0f) {
            float ix = (m[0] * u + m[1] * v + m[2]) / sz * w - 0.5f;
            float iy = (m[3] * u + m[4] * v + m[5]) / sz * h - 0.5f;
            float fx = floorf(ix), fy = floorf(iy);
            int x0 = (int)fx, y0 = (int)fy;
            float ax = ix - fx, ay = iy - fy;
            float weights[4] = {(1.0f - ax) * (1.0f - ay), ax * (1.0f - ay), (1.0f - ax) * ay, ax * ay};
            for (int k = 0; k < 4; ++k) {
                int tx = x0 + (k & 1), ty = y0 + (k >> 1);
                if (tx < 0 || ty < 0 || tx >= w || ty >= h) continue;
                float tap[3];
                pixel(frame, tx, ty, s, tap);
                for (int c = 0; c < 3; ++c) rgb[c] += weights[k] * tap[c];
            }
        }
    }
    for (int c = 0; c < 3; ++c) {
        unsigned short bits;
        asm("cvt.rn.f16.f32 %0, %1;" : "=h"(bits) : "f"(rgb[c]));
        out[((long)(n * 3 + c) * oh + y) * ow + x] = bits;
    }
}
"""


class _Stack(ctypes.Structure):
    """`SOURCE`'s `Stack`, passed to the kernel by value."""
    _fields_ = [('inverse', (ctypes.c_float * 9) * MAX_STACK), ('straighten', ctypes.c_int * MAX_STACK),
                ('flip_h', ctypes.c_int * MAX_STACK), ('flip_v', ctypes.c_int * MAX_STACK), ('count', ctypes.c_int),
                ('width', ctypes.c_int), ('height', ctypes.c_int), ('layout', ctypes.c_int), ('rotate90', ctypes.c_int)]


def nv12_to_rgb(y: torch.Tensor, uv: torch.Tensor) -> torch.Tensor:
    """RGB float32 in [0, 1], channels before the rows, from NV12's planes, uint8: Y (…, H, W) and interleaved UV
    (…, H / 2, W), any leading dimensions (a stack) kept: (…, 3, H, W); BT.601 limited range, the chroma repeated by a
    broadcast view."""
    lead: tuple[int, ...] = tuple(uv.shape[:-2])
    half: torch.Tensor = uv.reshape(*lead, uv.shape[-2], -1, 2)
    rows, columns = half.shape[-3], half.shape[-2]
    chroma: torch.Tensor = (half[..., :, None, :, None, :].expand(*lead, rows, 2, columns, 2, 2)
                            .reshape(*lead, rows * 2, columns * 2, 2).float() - 128.0)
    u, v = chroma[..., 0], chroma[..., 1]
    luma: torch.Tensor = (y.float() - 16.0) * 1.164383
    rgb: torch.Tensor = torch.stack((luma + 1.596027 * v, luma - 0.391762 * u - 0.812968 * v, luma + 2.017232 * u), dim=-3)
    return rgb.clamp_(0.0, 255.0).div_(255.0)


def warp_grid(homography: torch.Tensor, height: int, width: int) -> torch.Tensor:
    """The `grid_sample` grid (1, height, width, 2) that draws a frame through `homography`, uv to
    uv, on the homography's device: each output pixel looks up where it came from. A pixel from
    behind the camera gets a place outside the frame, which samples to nothing."""
    inverse: torch.Tensor = torch.linalg.inv(homography)
    device: torch.device = homography.device
    u: torch.Tensor = (torch.arange(width, device=device, dtype=torch.float32) + 0.5) / width
    v: torch.Tensor = (torch.arange(height, device=device, dtype=torch.float32) + 0.5) / height
    grid_v, grid_u = torch.meshgrid(v, u, indexing='ij')
    source: torch.Tensor = inverse @ torch.stack((grid_u.ravel(), grid_v.ravel(), torch.ones_like(grid_u).ravel()))
    uv: torch.Tensor = (source[:2] / source[2]).T.reshape(height, width, 2) * 2.0 - 1.0
    uv[(source[2] <= 0.0).reshape(height, width)] = 2.0
    return uv.unsqueeze(0)


def upright(image: torch.Tensor, frame: CameraFrame) -> torch.Tensor:
    """A straightened (C, H, W) image turned and flipped as its frame says, in `CameraFrame`'s order; contiguous."""
    if frame.rotate90:
        image = image.transpose(-1, -2)
    dims: list[int] = ([-2] if frame.flip_v else []) + ([-1] if frame.flip_h else [])
    if dims:
        return image.flip(dims)
    return image.contiguous()


class ImageUpload:
    """See the module docstring."""

    def __init__(self) -> None:
        self.staging: StagingRing = StagingRing()      # each stack's pinned buffers, by its cameras
        self._warp_grids: dict[int, tuple[np.ndarray, torch.Tensor]] = {}      # each camera's homography and its grid
        self._kernels: CudaKernels = CudaKernels(SOURCE, ('upload',), 'ImageUpload')
        self._kernel_failed: bool = False
        self._inverses: dict[int, tuple[np.ndarray, list[float]]] = {}       # each camera's homography and its inverse

    def upload(self, frames: list[CameraFrame], stream: torch.cuda.Stream,
               kernel: bool = True) -> tuple[list[torch.Tensor], torch.cuda.Event]:
        """`frames`, color frames of one size, layout and turn, onto the GPU on `stream`: each one's image, float16 RGB
        CHW in [0, 1], straightened, turned and flipped, in the frames' order, and the event recorded after them all.
        With `kernel`, as one CUDA kernel; without, or where it cannot be compiled, as torch ops."""
        assert len({frame.rotate90 for frame in frames}) == 1, 'a stack holds turned or unturned frames, not both'
        first: np.ndarray = frames[0].image
        cam_ids: tuple[int, ...] = tuple(frame.cam_id for frame in frames)
        staging: torch.Tensor = self.staging.buffer(cam_ids, (len(frames), *first.shape), torch.from_numpy(first[:0]).dtype)
        for index, frame in enumerate(frames):
            staging[index].copy_(torch.from_numpy(frame.image))
        with torch.cuda.stream(stream):
            gpu: torch.Tensor = staging.cuda(non_blocking=True)
            images: list[torch.Tensor] | None = None
            if kernel and not self._kernel_failed and len(frames) <= MAX_STACK and gpu.dtype == torch.uint8:
                try:
                    images = self._upload_kernel(gpu, frames, stream)
                except (CudaKernelError, OSError):
                    logger.exception("ImageUpload: the CUDA kernel failed; the color is uploaded by torch ops from now on")
                    self._kernel_failed = True
            if images is None:
                images = self._upload_torch(gpu, frames)
            ready: torch.cuda.Event = torch.cuda.Event()
            ready.record(stream)
        self.staging.queued(cam_ids, ready)
        return images, ready

    def _upload_kernel(self, gpu: torch.Tensor, frames: list[CameraFrame], stream: torch.cuda.Stream) -> list[torch.Tensor]:
        """The stack, all turned or none, converted, straightened, turned and flipped by `SOURCE`'s one kernel."""
        first: CameraFrame = frames[0]
        layout: int = NV12 if first.nv12 else MONO if gpu.ndim == 3 else BGR
        height: int = gpu.shape[1] * 2 // 3 if layout == NV12 else gpu.shape[1]
        width: int = gpu.shape[2]
        stack = _Stack()
        stack.count, stack.width, stack.height, stack.layout = len(frames), width, height, layout
        stack.rotate90 = int(first.rotate90)
        for index, frame in enumerate(frames):
            stack.flip_h[index] = int(frame.flip_h)
            stack.flip_v[index] = int(frame.flip_v)
            if frame.homography is not None:
                stack.straighten[index] = 1
                for i, value in enumerate(self._inverse(frame)):
                    stack.inverse[index][i] = value
        out_height, out_width = (width, height) if first.rotate90 else (height, width)
        out: torch.Tensor = torch.empty((len(frames), 3, out_height, out_width), dtype=torch.float16, device='cuda')
        grid: tuple[int, int] = ((out_width + BLOCK[0] - 1) // BLOCK[0],
                                 (out_height * len(frames) + BLOCK[1] - 1) // BLOCK[1])
        self._kernels.launch('upload', grid, BLOCK, [ctypes.c_void_p(gpu.data_ptr()), ctypes.c_void_p(out.data_ptr()), stack], stream)
        return [out[index] for index in range(len(frames))]

    def _inverse(self, frame: CameraFrame) -> list[float]:
        """The frame's camera's homography inverted, row-major, made once per homography."""
        assert frame.homography is not None
        cached: tuple[np.ndarray, list[float]] | None = self._inverses.get(frame.cam_id)
        if cached is None or cached[0] is not frame.homography:
            cached = (frame.homography, np.linalg.inv(np.asarray(frame.homography, dtype=np.float64)).astype(np.float32).ravel().tolist())
            self._inverses[frame.cam_id] = cached
        return cached[1]

    def _upload_torch(self, gpu: torch.Tensor, frames: list[CameraFrame]) -> list[torch.Tensor]:
        """The stack converted, straightened, turned and flipped by torch ops."""
        rgb: torch.Tensor = self._converted(gpu, frames)
        straightened: list[int] = [index for index, frame in enumerate(frames) if frame.homography is not None]
        if len(straightened) == len(frames):        # every camera (the rig's case): the whole stack, no gather
            rgb = F.grid_sample(rgb.float(), self._grids(frames, rgb.shape[2], rgb.shape[3]),
                                align_corners=False).to(dtype=torch.float16)
        elif straightened:
            selected: list[CameraFrame] = [frames[i] for i in straightened]
            warped: torch.Tensor = F.grid_sample(rgb[straightened].float(), self._grids(selected, rgb.shape[2], rgb.shape[3]),
                                                 align_corners=False).to(dtype=torch.float16)
            rgb[straightened] = warped
        return [upright(rgb[index], frame) for index, frame in enumerate(frames)]

    def _converted(self, gpu: torch.Tensor, frames: list[CameraFrame]) -> torch.Tensor:
        """The stack (N, …) as uploaded, to float16 RGB (N, 3, H, W) in [0, 1]."""
        if frames[0].nv12:
            rows: int = gpu.shape[1] * 2 // 3
            return nv12_to_rgb(gpu[:, :rows], gpu[:, rows:]).to(dtype=torch.float16)
        if gpu.ndim == 3:       # mono (N, H, W)
            return gpu.unsqueeze(1).to(dtype=torch.float16).mul_(1.0 / 255.0).expand(-1, 3, -1, -1).contiguous()
        return gpu.permute(0, 3, 1, 2).flip(1).to(dtype=torch.float16).mul_(1.0 / 255.0)       # BGR HWC to RGB CHW

    def _grids(self, frames: list[CameraFrame], height: int, width: int) -> torch.Tensor:
        """The frames' straightening grids (N, H, W, 2); one frame's is its cached grid itself, not a copy."""
        grids: list[torch.Tensor] = [self._grid(frame, height, width) for frame in frames]
        return grids[0] if len(grids) == 1 else torch.cat(grids)

    def _grid(self, frame: CameraFrame, height: int, width: int) -> torch.Tensor:
        """The frame's camera's straightening grid (1, H, W, 2), made once per homography."""
        assert frame.homography is not None
        cached: tuple[np.ndarray, torch.Tensor] | None = self._warp_grids.get(frame.cam_id)
        if cached is None or cached[0] is not frame.homography:
            matrix: torch.Tensor = torch.from_numpy(frame.homography).to('cuda', torch.float32)
            cached = (frame.homography, warp_grid(matrix, height, width))
            self._warp_grids[frame.cam_id] = cached
        return cached[1]
