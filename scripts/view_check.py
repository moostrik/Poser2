"""Does the view stage (`ViewLayer`) make what it should from recorded captures?

Plays a recording's color and depth frames through the `Uploader` (the color straightened, the depth aligned, as the
app does), takes them into GL and runs the view stage at the preset's settings. Checks, per frame:
- the depth read into the view against `edge_aware_depth`, the numpy reading of the same rules: pixels off by over
  1 mm (expected: none, a float rounding at the tolerance's edge at most);
- the mask against the depth mask it is made from: their overlap (intersection over union; the guided filter
  only moves the edge, by a few view pixels, so near 1) and the share of the view that is the player.
Saves the last frame's image, depth mask, mask and foreground side by side as a PNG.

    python scripts/view_check.py [RECORDING] [--frames 40] [--png PATH]
"""
import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import cv2  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from OpenGL.GL import GL_CLAMP_TO_BORDER, GL_FLOAT, GL_NEAREST, GL_RED, GL_RGBA, GL_TEXTURE_2D, glBindTexture, glFinish, glGetTexImage  # noqa: E402

import monitor_harness as mh  # noqa: E402
from modules.camera import CameraFrame, FrameType, read_metadata  # noqa: E402
from modules.flow import DepthVelocitySettings, edge_aware_depth  # noqa: E402
from modules.gl import Tensor, Texture  # noqa: E402
from modules.inference import GpuDepth, GpuImage, Uploader, UploaderSettings  # noqa: E402
from modules.render.layers import ViewInput, ViewLayer, ViewLayerSettings  # noqa: E402
from modules.video import StreamReader  # noqa: E402

RECORDINGS = Path('apps/deep_flow/data/recordings')
PRESET = Path('apps/deep_flow/data/settings/settings.json')


def frames(chunk: Path, count: int, depth: bool) -> list[np.ndarray]:
    found: list[np.ndarray] = []
    kind = FrameType.DEPTH if depth else FrameType.VIDEO
    reader = StreamReader(0, kind, lambda _c, _t, frame, _chunk, _i: found.append(frame.copy()) if len(found) < count else None,
                          True, fps=1000.0, stream=1 if depth else 0, depth=depth)
    reader.load(str(chunk), 0, 0, count)
    while not reader.is_loaded():
        time.sleep(0.01)
    reader.play()
    while reader.is_playing():
        time.sleep(0.01)
    reader.stop()
    reader.join()
    return found


def read(texture: Texture, channels: int = 1) -> np.ndarray:
    """A texture's pixels, rows down."""
    glFinish()
    glBindTexture(GL_TEXTURE_2D, texture.tex_id)
    data = glGetTexImage(GL_TEXTURE_2D, 0, GL_RED if channels == 1 else GL_RGBA, GL_FLOAT)
    glBindTexture(GL_TEXTURE_2D, 0)
    shape = (texture.height, texture.width) if channels == 1 else (texture.height, texture.width, 4)
    return np.frombuffer(data, np.float32).reshape(shape)[::-1]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('recording', nargs='?', default='20261111-123241')
    parser.add_argument('--frames', type=int, default=40)
    parser.add_argument('--png', default='')
    args = parser.parse_args()
    chunk = RECORDINGS / args.recording / '0_000.mkv'
    info = read_metadata(chunk).info
    assert info is not None, 'the recording has no metadata'
    colors = frames(chunk, args.frames, depth=False)
    depths = frames(chunk, args.frames, depth=True)
    count = min(len(colors), len(depths))
    alignment = info.depth_alignment

    preset = json.loads(PRESET.read_text(encoding='utf-8'))['render']
    window = mh.context()
    settings = ViewLayerSettings()
    settings.initialize()
    for name, value in preset['view'].items():
        setattr(settings, name, value)
    depth_settings = DepthVelocitySettings()
    depth_settings.initialize()
    depth_settings.tolerance = preset['motion']['z']['tolerance']
    view = ViewLayer(settings, depth_settings, json.loads(PRESET.read_text(encoding='utf-8'))['camera']['fov'])
    view.allocate(preset['motion']['width'], preset['motion']['height'])

    uploader = Uploader(UploaderSettings())
    images: list[GpuImage] = []
    gpu_depths: list[GpuDepth] = []
    uploader.add_gpu_image_callback(images.append)
    uploader.add_gpu_depth_callback(gpu_depths.append)
    uploader.start()
    image_texture = Tensor()
    depth_texture = Tensor(interpolation=GL_NEAREST, wrap=GL_CLAMP_TO_BORDER, border_color=(0.0, 0.0, 0.0, 0.0))

    print(f'{args.recording}: {count} captures, view {view.output.image.width} x {view.output.image.height}, '
          f'tolerance {depth_settings.tolerance} m, near {settings.near} far {settings.far}')
    off_total, pixels, overlaps, shares = 0, 0, [], []
    last: dict[str, np.ndarray] = {}
    for i in range(count):
        t = float(i)
        uploader.submit_frame(CameraFrame(0, FrameType.VIDEO, colors[i], t))
        uploader.submit_frame(CameraFrame(0, FrameType.DEPTH, depths[i], t, alignment=alignment))
        end = time.perf_counter() + 5.0
        while (len(images) <= i or len(gpu_depths) <= i) and time.perf_counter() < end:
            time.sleep(0.002)
        image, depth = images[i], gpu_depths[i]
        torch.cuda.synchronize()
        image_texture.set_tensor(image.image)
        image_texture.update()
        depth_texture.set_tensor(depth.depth)
        depth_texture.update()
        view.set_input(ViewInput(image_texture, depth_texture, t))
        view.update()
        output = view.output

        got = read(output.depth.texture)
        height, width = got.shape
        w = view._window        # the window the view was cut with, uv rows down
        u = w.x + (np.arange(width) + 0.5) / width * w.width
        v = w.y + (np.arange(height) + 0.5) / height * w.height
        uu, vv = np.meshgrid(u, v)
        want = edge_aware_depth(depth.depth.cpu().numpy(), uu, vv, depth_settings.tolerance)
        off_total += int((np.abs(got - want) > 0.001).sum())
        pixels += got.size

        mask = read(output.mask) > 0.5
        depth_mask = read(output.depth_mask) > 0.5
        union = int((mask | depth_mask).sum())
        if union:
            overlaps.append(int((mask & depth_mask).sum()) / union)
        shares.append(float(mask.mean()))
        if i == count - 1:
            last = {'image': read(output.image, 4), 'depth_mask': read(output.depth_mask), 'mask': read(output.mask),
                    'foreground': read(output.foreground, 4)}

    uploader.stop()
    print(f'depth in the view: {off_total} of {pixels} pixels off the reference by over 1 mm')
    if overlaps:
        print(f'mask vs depth mask: overlap mean {np.mean(overlaps):.3f}, min {np.min(overlaps):.3f}; '
              f'the player {np.mean(shares) * 100:.1f}% of the view')
    if args.png and last:
        def bgr(picture: np.ndarray) -> np.ndarray:
            if picture.ndim == 2:
                picture = np.repeat(picture[..., None], 3, axis=2)
            return (np.clip(picture[..., :3][..., ::-1], 0.0, 1.0) * 255.0).astype(np.uint8)
        cv2.imwrite(args.png, np.hstack([bgr(last[k]) for k in ('image', 'depth_mask', 'mask', 'foreground')]))
        print(f'saved {args.png}')
    return 1 if off_total > pixels * 1e-4 else 0


if __name__ == '__main__':
    sys.exit(main())
