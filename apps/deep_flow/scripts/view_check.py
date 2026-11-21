"""What do the view stage (`ViewLayer`) and the color stage after it (`ColorLayer`) make from recorded captures, and
how long do their passes take?

Plays a recording's color and depth frames through the `SerialUploader` (the color straightened, the depth aligned, as the
app does, with the depth's confidence where the chunk has it), takes them into GL and runs the view stage and the color
stage at the flow file's settings. Prints the share of the mask without depth, upscaled and filled (0), the share of
the view that is the player, and the confidence in green: the upscale's where the fill kept the depth, 0 where it
filled (1 where the recording has none). Saves the last frame's
image, the crop's depth, the mask, the upscaled depth, the filled depth and the color side by side as a PNG, the depths
from the box's front (white) to its back (black).

With `--time`, then each of the view's passes on the last capture, the median of 40 runs, and the whole view. With
`--npy`, the last capture's color saved as float RGBA; with `--compare`, its largest difference from such a file.

    python apps/deep_flow/scripts/view_check.py [RECORDING] [--frames 40] [--png PATH] [--time] [--npy PATH] [--compare PATH]
"""
import OpenGL
OpenGL.ERROR_CHECKING = False  # as launcher.py runs the app; a glGetError per call would time the CPU
import argparse
import json
import sys
import time
from pathlib import Path
from typing import cast

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
import cv2  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from OpenGL.GL import GL_CLAMP_TO_BORDER, GL_FLOAT, GL_NEAREST, GL_PACK_ALIGNMENT, GL_RED, GL_RGBA, GL_TEXTURE_2D, glBindTexture, glFinish, glGetTexImage, glPixelStorei  # noqa: E402

from scripts import monitor_harness as mh  # noqa: E402
from modules.camera import CameraFrame, FrameType  # noqa: E402
from modules.recording import ChunkStream, read_metadata  # noqa: E402
from modules.gl import BlendState, Tensor, Texture  # noqa: E402
from modules.inference import GpuDepth, GpuImage, SerialUploader, SerialUploaderSettings  # noqa: E402
from apps.deep_flow.render.layers import EDGE, ViewInput, ViewLayer  # noqa: E402
from apps.deep_flow.settings import ViewLayerSettings  # noqa: E402
from modules.render import ColorInput, ColorLayer, ColorLayerSettings  # noqa: E402
from modules.video import StreamReader  # noqa: E402

RECORDINGS = Path('apps/deep_flow/data/recordings')
SETTINGS_FILE = Path('apps/deep_flow/data/settings/flow/settings.json')
SETUP = Path('apps/deep_flow/data/settings/setup/studio.json')


def frames(chunk: Path, count: int, stream: ChunkStream) -> list[np.ndarray]:
    """The chunk's first `count` frames of `stream`: color as BGR, depth as millimeters, the confidence grey."""
    found: list[np.ndarray] = []
    reader = StreamReader(0, stream, lambda _c, _t, frame, _chunk, _i: found.append(frame.copy()) if len(found) < count else None,
                          stream == ChunkStream.VIDEO, fps=1000.0, stream=int(stream), depth=stream == ChunkStream.DEPTH)
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
    glPixelStorei(GL_PACK_ALIGNMENT, 1)        # the rows packed, as numpy holds them
    data = cast(bytes, glGetTexImage(GL_TEXTURE_2D, 0, GL_RED if channels == 1 else GL_RGBA, GL_FLOAT))
    glBindTexture(GL_TEXTURE_2D, 0)
    shape = (texture.height, texture.width) if channels == 1 else (texture.height, texture.width, 4)
    return np.frombuffer(data, np.float32).reshape(shape)[::-1]


def time_view(view: ViewLayer, color_stage: ColorLayer, image: Texture, depth: Texture) -> None:
    """Each of the view's passes on the last capture, the median of 40 runs (ms), as `ViewLayer._update` runs them,
    and the color stage's; then the whole view made again."""
    roi = view._window
    crop, mask, view_depth, color = view._crop, view._mask, view._depth, color_stage._color

    def whole() -> None:
        view._made = None
        view.update()

    BlendState.push()
    BlendState.set_blend_mode(BlendState.BlendMode.DISABLED)
    rows = [('crop, image and depth', lambda: crop.crop(image, depth, roi)),
            ('mask, the guide', lambda: mask.guide(crop.image, crop.depth, roi, view._grid_size[1])),
            ('depth, upscale and fill', lambda: view_depth.upscale(depth, roi, mask.mask)),
            ('depth, the fill', lambda: view_depth._fill.use(view_depth._upscaled, mask.mask, view_depth._filled)),
            ('color', lambda: color.process(crop.image, mask.mask)),
            ('the view', whole)]
    for name, work in rows:
        for _ in range(3):
            work()
        glFinish()
        print(f'{name:<24} {mh.gpu_ms(work, 40):7.3f} ms', flush=True)
    BlendState.pop()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('recording', nargs='?', default='20261111-123241')
    parser.add_argument('--frames', type=int, default=40)
    parser.add_argument('--png', default='')
    parser.add_argument('--time', action='store_true', help='also time each pass on the last capture')
    parser.add_argument('--erode', type=int, help="the color's erode, where not the settings file's")
    parser.add_argument('--feather', type=float, help="the color's feather, where not the settings file's")
    parser.add_argument('--npy', default='', help="save the last capture's color, float RGBA, as .npy")
    parser.add_argument('--compare', default='', help="a .npy of the color to compare the last capture's with (largest difference)")
    args = parser.parse_args()
    chunk = RECORDINGS / args.recording / '0_000.mkv'
    info = read_metadata(chunk).info
    assert info is not None, 'the recording has no metadata'
    colors = frames(chunk, args.frames, ChunkStream.VIDEO)
    depths = frames(chunk, args.frames, ChunkStream.DEPTH)
    confidences: list[np.ndarray | None] = list(frames(chunk, args.frames, ChunkStream.CONFIDENCE)) if info.confidence else [None] * len(depths)
    count = min(len(colors), len(depths), len(confidences))
    alignment = info.depth_alignment

    settings_file = json.loads(SETTINGS_FILE.read_text(encoding='utf-8'))
    mh.context()
    settings = ViewLayerSettings()
    settings.initialize()
    settings.update_from_dict(settings_file['view'])
    color_settings = ColorLayerSettings()
    color_settings.initialize()
    color_settings.update_from_dict(settings_file['color'])
    if args.erode is not None:
        color_settings.erode = args.erode
    if args.feather is not None:
        color_settings.feather = args.feather
    band = settings.crop
    view = ViewLayer(settings, json.loads(SETUP.read_text(encoding='utf-8'))['setup']['camera']['fov'])
    view.allocate(settings_file['view']['width'], settings_file['view']['height'])
    color_stage = ColorLayer(color_settings)
    color_stage.allocate(settings_file['view']['width'], settings_file['view']['height'])

    uploader = SerialUploader(SerialUploaderSettings())
    images: list[GpuImage] = []
    gpu_depths: list[GpuDepth] = []
    uploader.add_gpu_image_callback(images.append)
    uploader.add_gpu_depth_callback(gpu_depths.append)
    uploader.start()
    image_texture = Tensor()
    depth_texture = Tensor(interpolation=GL_NEAREST, wrap=GL_CLAMP_TO_BORDER, border_color=(0.0, 0.0, 0.0, 0.0))

    print(f'{args.recording}: {count} captures, view {view.output.image.width} x {view.output.image.height}, '
          f'edge {EDGE} m, near {band.near} far {band.far}')
    empty_upscaled, empty_filled, shares = [], [], []
    green_kept, green_made, green_mean = [], [], []
    last: dict[str, np.ndarray] = {}
    for i in range(count):
        t = float(i)
        uploader.submit_frame(CameraFrame(0, FrameType.VIDEO, colors[i], t))
        uploader.submit_frame(CameraFrame(0, FrameType.DEPTH, depths[i], t, alignment=alignment, confidence=confidences[i]))
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
        color_stage.set_input(ColorInput(output.image, output.mask, view.timestamp, view.made))
        color_stage.update()

        mask = read(output.mask)
        upscaled = read(view._depth._upscaled)
        filled = read(output.depth.texture)
        inside = mask > 0.0
        if inside.any():
            empty_upscaled.append(float((inside & (upscaled <= 0.0)).sum()) / float(inside.sum()))
            empty_filled.append(float((inside & (filled <= 0.0)).sum()) / float(inside.sum()))
        # The confidence in green: the camera's where the upscale kept a depth, 0 where the fill made it
        upscaled_green, filled_green = read(view._depth._upscaled, 4)[..., 1], read(output.depth.texture, 4)[..., 1]
        kept, made = upscaled > 0.0, (filled > 0.0) & (upscaled <= 0.0)
        green_kept.append(float(np.abs(filled_green[kept] - upscaled_green[kept]).max(initial=0.0)))
        green_made.append(float(np.abs(filled_green[made]).max(initial=0.0)))
        green_mean.append(float(filled_green[kept].mean()) if kept.any() else float('nan'))
        shares.append(float((mask > 0.5).mean()))
        if i == count - 1:
            def near_white(metres: np.ndarray) -> np.ndarray:
                return np.where(metres > 0.0, 1.0 - (metres - band.near) / max(band.far - band.near, 1e-6), 0.0)
            last = {'image': read(output.image, 4), 'crop depth': near_white(read(view._crop.depth)), 'mask': mask,
                    'upscaled': near_white(upscaled), 'filled': near_white(filled), 'color': read(color_stage.output.color, 4)}

    uploader.stop()
    if args.npy and last:
        np.save(args.npy, last['color'])
        print(f'saved {args.npy}')
    if args.compare and last:
        reference = np.load(args.compare)
        difference = np.abs(last['color'].astype(np.float64) - reference.astype(np.float64))
        print(f'the color against {args.compare}: largest difference {difference.max() * 255.0:.3f} of 255, '
              f'mean {difference.mean() * 255.0:.4f}')
    if args.time:
        time_view(view, color_stage, image_texture, depth_texture)
    if empty_upscaled:
        print(f'the mask without depth: {np.mean(empty_upscaled) * 100:.2f}% upscaled, {np.mean(empty_filled) * 100:.2f}% filled; '
              f'the player {np.mean(shares) * 100:.1f}% of the view')
    if green_kept:
        print(f'the confidence ({"recorded" if info.confidence else "none recorded: 1"}): where kept off the upscale by at most '
              f'{max(green_kept):.6f}, mean {np.nanmean(green_mean):.3f}; where filled at most {max(green_made):.6f}')
    if args.png and last:
        def bgr(picture: np.ndarray) -> np.ndarray:
            if picture.ndim == 2:
                picture = np.repeat(picture[..., None], 3, axis=2)
            return (np.clip(picture[..., :3][..., ::-1], 0.0, 1.0) * 255.0).astype(np.uint8)
        cv2.imwrite(args.png, np.hstack([bgr(last[k]) for k in ('image', 'crop depth', 'mask', 'upscaled', 'filled', 'color')]))
        print(f'saved {args.png}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
