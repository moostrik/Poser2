"""What the GPU spends per frame on the color upload and the pose crops, step by step, with the GPU otherwise idle.

Takes a recording's color frame (and an NV12 frame of the same size, as the live camera delivers), runs each step of
`Uploader._upload` and of the crop path (`CropExtractor._gpu_crop_resize`, then the pose runner's batch, resize and
normalization) on one stream, and times each with CUDA events over many runs. Also counts the kernels each step
launches (torch's profiler).

    python scripts/gpu_input_timing.py [RECORDING] [--runs 200]
"""
import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from modules.camera import FrameType, TiltCorrection, TiltCorrectionSettings, read_metadata  # noqa: E402
from modules.inference.uploader import nv12_to_rgb, warp_grid  # noqa: E402
from modules.video import StreamReader  # noqa: E402

RECORDINGS = Path('apps/deep_flow/data/recordings')
CROP_SIZE = (768, 1024)         # the preset's crop (width, height)
MODEL_SIZE = (192, 256)         # the pose model's input, STANDARD


def color_frame(chunk: Path) -> np.ndarray:
    found: list[np.ndarray] = []
    reader = StreamReader(0, FrameType.VIDEO, lambda _c, _t, frame, _chunk, _i: found.append(frame.copy()) if not found else None,
                          True, fps=1000.0, stream=0)
    reader.load(str(chunk), 0, 0, 1)
    while not reader.is_loaded():
        time.sleep(0.01)
    reader.play()
    while reader.is_playing():
        time.sleep(0.01)
    reader.stop()
    reader.join()
    return found[0]


def timed(name: str, step, runs: int) -> float:
    """The step's mean GPU time (ms) over `runs`, and its kernel count."""
    stream = torch.cuda.current_stream()
    for _ in range(10):
        step()
    torch.cuda.synchronize()
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    start.record(stream)
    for _ in range(runs):
        step()
    end.record(stream)
    torch.cuda.synchronize()
    ms = start.elapsed_time(end) / runs
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA]) as profiler:
        step()
        torch.cuda.synchronize()
    kernels = sum(1 for e in profiler.events() if e.device_type == torch.autograd.DeviceType.CUDA)
    print(f'  {name:44} {ms:6.3f} ms  {kernels:3d} GPU ops')
    return ms


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('recording', nargs='?', default='20261111-123241')
    parser.add_argument('--runs', type=int, default=200)
    args = parser.parse_args()
    chunk = RECORDINGS / args.recording / '0_000.mkv'
    metadata = read_metadata(chunk)
    info = metadata.info
    assert info is not None
    bgr = color_frame(chunk)
    height, width = bgr.shape[:2]
    settings = TiltCorrectionSettings()
    settings.initialize()
    settings.enabled = True
    correction = TiltCorrection(settings)
    readings = metadata.frames[0].readings if metadata.frames else None
    correction.update(readings.tilt if readings is not None else 0.0, info.intrinsics, (width, height))
    homography = correction.homography
    print(f'{args.recording}: color {width} x {height}, homography {"yes" if homography is not None else "none"}, {args.runs} runs each')

    pinned_bgr = torch.from_numpy(bgr).pin_memory()
    nv12 = np.random.default_rng(0).integers(0, 255, (height * 3 // 2, width), dtype=np.uint8)
    pinned_nv12 = torch.from_numpy(nv12).pin_memory()
    gpu_bgr = pinned_bgr.cuda()
    gpu_nv12 = pinned_nv12.cuda()
    grid = warp_grid(torch.from_numpy(homography).cuda().float(), height, width) if homography is not None else None

    def convert_bgr() -> torch.Tensor:
        return gpu_bgr.permute(2, 0, 1).flip(0).to(dtype=torch.float16).mul_(1.0 / 255.0)

    def convert_nv12() -> torch.Tensor:
        return nv12_to_rgb(gpu_nv12[:height], gpu_nv12[height:]).to(dtype=torch.float16)

    rgb = convert_bgr()

    def straighten() -> torch.Tensor:
        return F.grid_sample(rgb.unsqueeze(0).float(), grid, align_corners=False)[0].to(dtype=torch.float16) if grid is not None else rgb

    def mirror() -> torch.Tensor:
        return rgb.flip(-1)

    print('color upload')
    total = timed('copy from pinned memory (BGR)', lambda: pinned_bgr.cuda(non_blocking=True), args.runs)
    total += timed('BGR to RGB half', convert_bgr, args.runs)
    total += timed('straighten (grid_sample)', straighten, args.runs)
    total += timed('mirror', mirror, args.runs)
    print(f'  {"recordings (BGR), all":44} {total:6.3f} ms')
    live = timed('copy from pinned memory (NV12)', lambda: pinned_nv12.cuda(non_blocking=True), args.runs)
    live += timed('NV12 to RGB half', convert_nv12, args.runs)
    print(f'  {"live (NV12), with straighten and mirror":44} {live + total - timed("(BGR copy and convert, again)", lambda: (pinned_bgr.cuda(non_blocking=True), convert_bgr()), args.runs):6.3f} ms')

    # A player about a third of the frame wide and most of its height, padded as the preset pads
    roi = (width * 0.30, height * 0.05, width * 0.40, height * 0.90)
    crop_w, crop_h = CROP_SIZE
    scale = max(crop_w / roi[2], crop_h / roi[3])
    cut_w, cut_h = crop_w / scale, crop_h / scale
    x1, y1 = int(roi[0] + (roi[2] - cut_w) / 2), int(roi[1] + (roi[3] - cut_h) / 2)
    x2, y2 = int(x1 + cut_w), int(y1 + cut_h)
    mean = torch.tensor([0.485, 0.456, 0.406], device='cuda', dtype=torch.float16).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device='cuda', dtype=torch.float16).view(1, 3, 1, 1)

    def crop() -> torch.Tensor:
        cut = rgb[:, max(0, y1):min(height, y2), max(0, x1):min(width, x2)]
        return F.interpolate(cut.unsqueeze(0), size=(crop_h, crop_w), mode='bilinear', align_corners=False, antialias=True).squeeze(0)

    crops = [crop()]

    def model_input() -> torch.Tensor:
        batch = torch.stack(crops, dim=0).to(torch.float16)
        batch = F.interpolate(batch, size=(MODEL_SIZE[1], MODEL_SIZE[0]), mode='bilinear', align_corners=False)
        return (batch - mean) / std

    print(f'pose crops, one player, cut {x2 - x1} x {y2 - y1} to {crop_w} x {crop_h}, the model at {MODEL_SIZE[0]} x {MODEL_SIZE[1]}')
    crop_ms = timed('crop and resize (CropExtractor)', crop, args.runs)
    crop_ms += timed('batch, resize, normalize (pose runner)', model_input, args.runs)
    print(f'  {"one player, all":44} {crop_ms:6.3f} ms')
    return 0


if __name__ == '__main__':
    sys.exit(main())
