"""Does the view stage (`ViewLayer`) make what it should from recorded captures?

Plays a recording's color and depth frames through the `SerialUploader` (the color straightened, the depth aligned, as the
app does), takes them into GL and runs the view stage at the preset's settings. Checks, per frame:
- the crop's depth against its numpy reading (`depth_crop`): pixels off by over 1 mm (expected: none);
- the read against its numpy reading (`masked_depth`) on the GPU's mask: pixels off by over 1 mm (expected: none,
  a float rounding at the edge's jump at most);
- the fill against its numpy reading (`push_pull`) on the GPU's read and mask: pixels off by over 1 mm;
- the share of the mask without depth, in the read and in the depth (0 with the fill), and the share of the view
  that is the player.
Saves the last frame's image, the crop's depth, the mask, the read, the depth and the graded image side by side as a
PNG, the depths from the box's front (white) to its back (black).

With `--time`, then each of the view's passes on the last capture, the median of 40 runs, and the whole view.

    python apps/deep_flow/scripts/view_check.py [RECORDING] [--frames 40] [--png PATH] [--time]
"""
import OpenGL
OpenGL.ERROR_CHECKING = False  # as launcher.py runs the app; a glGetError per call would time the CPU
import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / 'scripts'))
import cv2  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from OpenGL.GL import GL_CLAMP_TO_BORDER, GL_FLOAT, GL_NEAREST, GL_RED, GL_RGBA, GL_TEXTURE_2D, glBindTexture, glFinish, glGetTexImage  # noqa: E402

import monitor_harness as mh  # noqa: E402
from modules.camera import CameraFrame, FrameType  # noqa: E402
from modules.recording import read_metadata  # noqa: E402
from modules.flow import depth_crop, masked_depth, push_pull  # noqa: E402
from modules.gl import Style, Tensor, Texture  # noqa: E402
from modules.inference import GpuDepth, GpuImage, SerialUploader, SerialUploaderSettings  # noqa: E402
from apps.deep_flow.render.layers import EDGE, ViewInput, ViewLayer  # noqa: E402
from apps.deep_flow.settings import ViewLayerSettings  # noqa: E402
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


def time_view(view: ViewLayer, image: Texture, depth: Texture) -> None:
    """Each of the view's passes on the last capture, the median of 40 runs (ms), as `ViewLayer._update` runs them;
    then the whole view made again."""
    roi = view._window
    crop, mask, view_depth = view._crop, view._mask, view._depth

    def whole() -> None:
        view._made = None
        view.update()

    Style.push_style()
    Style.set_blend_mode(Style.BlendMode.DISABLED)
    rows = [('crop, image and depth', lambda: crop.crop(image, depth, roi)),
            ('mask, the guide', lambda: mask.guide(crop.image, crop.depth, roi, view._grid_size[1])),
            ('depth, read and fill', lambda: view_depth.read(depth, roi, mask.mask)),
            ('depth, the fill', lambda: view_depth._fill.use(view_depth._read, mask.mask, view_depth._depth)),
            ('color grade', lambda: view._color_grade.process(crop.image, mask.mask)),
            ('the view', whole)]
    for name, work in rows:
        for _ in range(3):
            work()
        glFinish()
        print(f'{name:<24} {mh.gpu_ms(work, 40):7.3f} ms', flush=True)
    Style.pop_style()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('recording', nargs='?', default='20261111-123241')
    parser.add_argument('--frames', type=int, default=40)
    parser.add_argument('--png', default='')
    parser.add_argument('--time', action='store_true', help='also time each pass on the last capture')
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
    settings.update_from_dict(preset['view'])
    d = settings.depth
    band = settings.crop
    view = ViewLayer(settings, json.loads(PRESET.read_text(encoding='utf-8'))['camera']['fov'])
    view.allocate(preset['view']['width'], preset['view']['height'])

    uploader = SerialUploader(SerialUploaderSettings())
    images: list[GpuImage] = []
    gpu_depths: list[GpuDepth] = []
    uploader.add_gpu_image_callback(images.append)
    uploader.add_gpu_depth_callback(gpu_depths.append)
    uploader.start()
    image_texture = Tensor()
    depth_texture = Tensor(interpolation=GL_NEAREST, wrap=GL_CLAMP_TO_BORDER, border_color=(0.0, 0.0, 0.0, 0.0))

    print(f'{args.recording}: {count} captures, view {view.output.image.width} x {view.output.image.height}, '
          f'edge {EDGE} m, near {band.near} far {band.far}, fill {d.fill_enabled}')
    crop_off, read_off, fill_off, pixels = 0, 0, 0, 0
    unread, unfilled, shares = [], [], []
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

        cropped = read(view._crop.depth)
        mask = read(output.mask)
        got_read = read(view._depth._read)
        height, width = got_read.shape
        w = view._window        # the box's sides the view was cut with, uv rows down
        u = w.x + (np.arange(width) + 0.5) / width * w.width
        v = w.y + (np.arange(height) + 0.5) / height * w.height
        uu, vv = np.meshgrid(u, v)
        grid = depth.depth.cpu().numpy()
        crop_off += int((np.abs(cropped - depth_crop(grid, uu, vv, band.near, band.far)) > 0.001).sum())
        read_off += int((np.abs(got_read - masked_depth(grid, uu, vv, EDGE, mask, band.near, band.far)) > 0.001).sum())
        got = read(output.depth.texture)
        if d.fill_enabled:
            fill_off += int((np.abs(got - push_pull(got_read.astype(np.float64), mask)) > 0.001).sum())
        pixels += got.size

        inside = mask > 0.0
        if inside.any():
            unread.append(float((inside & (got_read <= 0.0)).sum()) / float(inside.sum()))
            unfilled.append(float((inside & (got <= 0.0)).sum()) / float(inside.sum()))
        shares.append(float((mask > 0.5).mean()))
        if i == count - 1:
            def near_white(metres: np.ndarray) -> np.ndarray:
                return np.where(metres > 0.0, 1.0 - (metres - band.near) / max(band.far - band.near, 1e-6), 0.0)
            last = {'image': read(output.image, 4), 'crop depth': near_white(cropped), 'mask': mask,
                    'read': near_white(got_read), 'depth': near_white(got), 'graded_image': read(output.graded_image, 4)}

    uploader.stop()
    if args.time:
        time_view(view, image_texture, depth_texture)
    print(f'the crop\'s depth: {crop_off} of {pixels} pixels off the reference')
    print(f'the read: {read_off} of {pixels} pixels off the reference')
    if d.fill_enabled:
        print(f'the fill: {fill_off} of {pixels} pixels off the reference')
    if unread:
        print(f'the mask without depth: {np.mean(unread) * 100:.2f}% in the read, {np.mean(unfilled) * 100:.2f}% in the depth; '
              f'the player {np.mean(shares) * 100:.1f}% of the view')
    if args.png and last:
        def bgr(picture: np.ndarray) -> np.ndarray:
            if picture.ndim == 2:
                picture = np.repeat(picture[..., None], 3, axis=2)
            return (np.clip(picture[..., :3][..., ::-1], 0.0, 1.0) * 255.0).astype(np.uint8)
        cv2.imwrite(args.png, np.hstack([bgr(last[k]) for k in ('image', 'crop depth', 'mask', 'read', 'depth', 'graded_image')]))
        print(f'saved {args.png}')
    return 1 if max(crop_off, read_off, fill_off) > pixels * 1e-4 else 0


if __name__ == '__main__':
    sys.exit(main())
