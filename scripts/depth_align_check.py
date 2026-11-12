"""Does the torch depth alignment (`modules.inference.depth_align`) give what the GL `DepthAlign` gives?

Takes depth frames of a recording with their recorded alignment, mirrored as the replay mirrors it, runs each through
both, level and straightened by the recorded tilt (and once by a steeper one), and compares them pixel for pixel; then
times both on the GPU. Expected: equal to the float's last bits (fused multiply-adds round differently), and a handful
of border pixels taking a neighbour's depth where that rounding moves a landing point across a pixel edge.

    python scripts/depth_align_check.py [RECORDING] [--frames 40]
"""
import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np  # noqa: E402
import torch  # noqa: E402
from OpenGL.GL import GL_NEAREST, GL_R32F, GL_R32UI, GL_RED, GL_FLOAT, GL_TEXTURE_2D, glBindTexture, glFinish, glGetTexImage  # noqa: E402

import monitor_harness as mh  # noqa: E402
from modules.camera import (DepthAlignment, FrameType, TiltCorrection, TiltCorrectionSettings, aligned_depth_size, depth_projection,  # noqa: E402
                            mirrored_alignment, mirrored_intrinsics, read_metadata)
from modules.gl import Image, Texture  # noqa: E402
from modules.inference.depth_align import DepthAligner, depth_metres  # noqa: E402
from modules.render.shaders import DepthAlign  # noqa: E402
from modules.video import StreamReader  # noqa: E402

RECORDINGS = Path('apps/deep_flow/data/recordings')


def depth_frames(chunk: Path, count: int) -> list[np.ndarray]:
    frames: list[np.ndarray] = []
    reader = StreamReader(0, FrameType.DEPTH, lambda _c, _t, frame, _chunk, _i: frames.append(frame.copy()) if len(frames) < count else None,
                          False, fps=1000.0, stream=1, depth=True)
    reader.load(str(chunk), 0, 0, count)
    while not reader.is_loaded():
        time.sleep(0.01)
    reader.play()
    while reader.is_playing():
        time.sleep(0.01)
    reader.stop()
    reader.join()
    return frames


def gl_align(shader: DepthAlign, source: Image, nearest: Texture, aligned: Texture, depth: np.ndarray, to_ray: np.ndarray, to_uv: np.ndarray,
             homography: np.ndarray | None) -> np.ndarray:
    source.set_image(depth)
    source.update()
    shader.use(source, nearest, aligned, to_ray, to_uv, homography, False)
    glFinish()
    glBindTexture(GL_TEXTURE_2D, aligned.tex_id)
    data = glGetTexImage(GL_TEXTURE_2D, 0, GL_RED, GL_FLOAT)
    glBindTexture(GL_TEXTURE_2D, 0)
    return np.frombuffer(data, np.float32).reshape(aligned.height, aligned.width)[::-1]     # texture rows run up


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('recording', nargs='?', default='20261111-123241')
    parser.add_argument('--frames', type=int, default=40)
    args = parser.parse_args()
    chunk = RECORDINGS / args.recording / '0_000.mkv'
    metadata = read_metadata(chunk)
    info = metadata.info
    assert info is not None and info.depth_alignment is not None, 'the recording has no depth alignment'
    frames = depth_frames(chunk, args.frames)
    depth_size = (frames[0].shape[1], frames[0].shape[0])
    alignment: DepthAlignment = mirrored_alignment(info.depth_alignment, depth_size[0]) if info.flip_h else info.depth_alignment
    intrinsics = mirrored_intrinsics(info.intrinsics, info.width) if info.flip_h else info.intrinsics
    to_ray, to_uv = depth_projection(alignment, depth_size)
    size = aligned_depth_size(depth_size, alignment)

    window = mh.context()
    shader = DepthAlign()
    shader.allocate()
    source = Image(interpolation=GL_NEAREST)
    nearest = Texture(interpolation=GL_NEAREST)
    nearest.allocate(size[0], size[1], GL_R32UI)
    DepthAlign.clear(nearest)
    aligned = Texture(interpolation=GL_NEAREST)
    aligned.allocate(size[0], size[1], GL_R32F)

    aligner = DepthAligner()
    tilt_settings = TiltCorrectionSettings()
    tilt_settings.initialize()
    tilt_settings.enabled = True
    print(f'{args.recording}: depth {depth_size}, aligned {size}, {len(frames)} frames, flip_h {info.flip_h}')
    worst = 0

    def ours(path: str, frame: np.ndarray, homography: np.ndarray | None) -> np.ndarray:
        if path == 'kernels':
            return aligner.align_staged(torch.from_numpy(frame).pin_memory(), to_ray, to_uv, homography, False, size).cpu().numpy()
        return aligner.align(depth_metres(torch.from_numpy(frame.astype(np.int32)).cuda()), to_ray, to_uv, homography, False, size).cpu().numpy()

    for path in ('torch', 'kernels'):
        for label, extra in (('level', None), ('recorded tilt', 0.0), ('tilt + 15 deg', 15.0)):
            differing, rounding, largest, landed = 0, 0.0, 0.0, 0
            for i, frame in enumerate(frames):
                homography: np.ndarray | None = None
                if extra is not None:
                    tilt = metadata.readings[i].tilt if i < len(metadata.readings) else 0.0
                    correction = TiltCorrection(tilt_settings)
                    correction.update(tilt + extra, intrinsics, (info.width, info.height))
                    homography = correction.homography
                metres = frame.astype(np.float32) / 1000.0
                want = gl_align(shader, source, nearest, aligned, metres, to_ray, to_uv, homography)
                got = ours(path, frame, homography)
                diff = np.abs(got - want)
                border = diff > 0.001                   # a neighbour's depth, not a rounding
                differing += int(border.sum())
                both = (want > 0.0) & (got > 0.0) & ~border
                if both.any():
                    rounding = max(rounding, float((diff[both] / want[both]).max()))
                largest = max(largest, float(diff.max()))
                landed += int((want > 0.0).sum())
            print(f'{path:7} {label:14}: {differing} of {landed} pixels with depth differ by over 1 mm over {len(frames)} frames '
                  f'(largest {largest:.4f} m); the rest within {rounding:.1e} of the depth')
            worst = max(worst, differing)

    # GPU time per frame, each alone: torch from metres on the GPU; the kernels from the pinned millimetres, the copy included
    metres_np = frames[0].astype(np.float32) / 1000.0
    tensor = depth_metres(torch.from_numpy(frames[0].astype(np.int32)).cuda())
    staged = torch.from_numpy(frames[0]).pin_memory()
    timings: dict[str, float] = {}
    for path, run in (('torch', lambda: aligner.align(tensor, to_ray, to_uv, None, False, size)),
                      ('kernels', lambda: aligner.align_staged(staged, to_ray, to_uv, None, False, size))):
        for _ in range(10):
            run()
        torch.cuda.synchronize()
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(200):
            run()
        end.record()
        torch.cuda.synchronize()
        timings[path] = start.elapsed_time(end) / 200
    for _ in range(10):
        gl_align(shader, source, nearest, aligned, metres_np, to_ray, to_uv, None)
    t0 = time.perf_counter()
    for _ in range(200):
        shader.use(source, nearest, aligned, to_ray, to_uv, None, False)
    glFinish()
    gl_ms = (time.perf_counter() - t0) * 1000.0 / 200
    print(f"per frame: torch {timings['torch']:.3f} ms, kernels {timings['kernels']:.3f} ms (GPU, with the copy), GL {gl_ms:.3f} ms (wall, the shader alone)")
    return 1 if worst > 5 * args.frames else 0      # more than a few border pixels a frame is a real difference


if __name__ == '__main__':
    sys.exit(main())
