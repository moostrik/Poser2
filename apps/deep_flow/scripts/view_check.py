"""Does the view stage (`ViewLayer`) make what it should from recorded captures?

Plays a recording's color and depth frames through the `SerialUploader` (the color straightened, the depth aligned, as the
app does), takes them into GL and runs the view stage at the preset's settings. Checks, per frame:
- the depth read into the view against its numpy reading (`edge_aware_depth`, `mask_sided_depth`, as `--read`
  says): pixels off by over 1 mm (expected: none, a float rounding at the tolerance's edge at most); MASK's ties,
  two depth pixels on the mask's side equally near, counted apart;
- the mask against the depth mask it is made from: their overlap (intersection over union; the guided filter
  only moves the edge, by a few view pixels, so near 1) and the share of the view that is the player.
Saves the last frame's image, depth mask, mask and foreground side by side as a PNG.

With `--time`, then each of the view's passes on the last capture, the median of 40 runs, and the whole view.

    python apps/deep_flow/scripts/view_check.py [RECORDING] [--frames 40] [--png PATH] [--read surface|mask] [--time]
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
from modules.flow import DepthRead, edge_aware_depth, mask_sided_depth  # noqa: E402
from modules.gl import Style, Tensor, Texture  # noqa: E402
from modules.inference import GpuDepth, GpuImage, SerialUploader, SerialUploaderSettings  # noqa: E402
from apps.deep_flow.render.layers import ViewInput, ViewLayer  # noqa: E402
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
    then the whole view made again, and with MASK's second read."""
    s = view.settings
    roi = view._window

    def image_pass() -> None:
        view._image.begin()
        view._roi_shader.use(image, roi)
        view._image.end()

    def depth_mask_pass() -> None:
        view._depth_mask.begin()
        view._depth_mask_shader.use(view._depth, s.depth.near, s.depth.far)
        view._depth_mask.end()

    def threshold_pass() -> None:
        view._mask.begin()
        view._threshold_shader.use(view._guided)
        view._mask.end()

    def foreground_pass() -> None:
        view._foreground.begin()
        view._foreground_shader.use(view._image, view._mask)
        view._foreground.end()

    def whole(read: DepthRead) -> None:
        settings_read = s.depth.depth_read
        s.depth.depth_read = read
        view._made = None
        view.update()
        s.depth.depth_read = settings_read

    Style.push_style()
    Style.set_blend_mode(Style.BlendMode.DISABLED)
    rows = [('image into the view', image_pass),
            ('depth read, SURFACE', lambda: view._read_depth(depth, roi, DepthRead.SURFACE, s)),
            ('depth mask', depth_mask_pass),
            ('guided filter', lambda: view._guide_edge(s)),
            ('mask threshold', threshold_pass),
            ('depth read, MASK', lambda: view._read_depth(depth, roi, DepthRead.MASK, s)),
            ('foreground', foreground_pass)]
    for name, work in rows:
        for _ in range(3):
            work()
        glFinish()
        print(f'{name:<24} {mh.gpu_ms(work, 40):7.3f} ms', flush=True)
    Style.pop_style()
    for read_mode in (DepthRead.SURFACE, DepthRead.MASK):
        work = lambda read_mode=read_mode: whole(read_mode)
        for _ in range(3):
            work()
        glFinish()
        print(f'{"the view, " + read_mode.name:<24} {mh.gpu_ms(work, 40):7.3f} ms', flush=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('recording', nargs='?', default='20261111-123241')
    parser.add_argument('--frames', type=int, default=40)
    parser.add_argument('--png', default='')
    parser.add_argument('--read', default='', help='surface or mask; the preset\'s without')
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
    if args.read:
        d.depth_read = DepthRead[args.read.upper()]
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
          f'tolerance {d.tolerance} m, near {d.near} far {d.far}')
    off_total, ties, pixels, overlaps, shares = 0, 0, 0, [], []
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
        grid = depth.depth.cpu().numpy()
        if d.depth_read == DepthRead.MASK:
            want = mask_sided_depth(grid, uu, vv, d.tolerance, read(output.mask) > 0.5, d.near, d.far,
                                    settings.mask.guided_radius)
        else:
            want = edge_aware_depth(grid, uu, vv, d.tolerance)
        off = np.abs(got - want) > 0.001
        if d.depth_read == DepthRead.MASK:
            # two depth pixels on the pixel's side as near as each other: either is the rule's, float decides
            def sided(values: np.ndarray) -> np.ndarray:
                return ((values > 0.0) & (values >= d.near) & (values <= d.far)) == (read(output.mask) > 0.5)
            tie = off & sided(got) & sided(want)
            ties += int(tie.sum())
            off &= ~tie
        off_total += int(off.sum())
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
    if args.time:
        time_view(view, image_texture, depth_texture)
    print(f'depth in the view: {off_total} of {pixels} pixels off the reference'
          + (f', {ties} more a tie between two on the mask\'s side' if d.depth_read == DepthRead.MASK else ''))
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
