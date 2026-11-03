"""Play saved frames through the depth track's shaders, A beside B, and measure both.

    python scripts/depth_track_compare.py FRAMES [--a NAME=VALUE ...] [--b NAME=VALUE ...] [--fast] [--verify]

FRAMES is what depth_track_capture saved from one loop of a recording (temp/depth_track/RECORDING).
No app and no camera: a window gives the GL context, and two `DepthVelocity`, A and B, run the
saved calls as the app made them; `--a` and `--b` override any of their arguments (the `z_`
settings without their prefix: time, threshold, tolerance, dead_zone, catch_up_time, window, and
motion_floor, the xy_threshold). The window plays the loop at its recorded rate, again and again,
A on the left, B on the right:

    1  z of the track, red toward the camera, green away     space  pause
    2  z after the threshold, as the fluid gets it           ← →    a frame back or on, paused
    3  the depth                                              + -    brighter, darker
                                                              esc    quit

Each loop prints, for A and B, z after the threshold in the still part (1–5 s): at the edge (0–6 px
inside the mask), inside (12 px and more) and on the table (outside, 16 px and more); and how the
body's z follows its real depth change in the slow (5–10 s) and fast (10 s on) parts. `--fast` runs
one loop without the window, as fast as it goes, and prints the same; `--verify` compares A's first
frames with the numpy reference instead.
"""

import ctypes
import json
import sys
import time
from argparse import ArgumentParser
from pathlib import Path

import cv2
import glfw
import numpy as np
from OpenGL.GL import (GL_ARRAY_BUFFER, GL_COLOR_BUFFER_BIT, GL_FALSE, GL_FLOAT, GL_FRAMEBUFFER, GL_R32F, GL_RED, GL_RG, GL_RG32F, GL_RGBA32F,
                       GL_RGB, GL_STATIC_DRAW, GL_TEXTURE_2D, GL_UNSIGNED_BYTE, glBindBuffer, glBindFramebuffer, glBindTexture, glBindVertexArray, glBufferData,
                       glClear, glClearColor, glEnableVertexAttribArray, glFinish, glGenBuffers, glGenVertexArrays, glTexImage2D,
                       glReadPixels, glVertexAttribPointer, glViewport)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from modules.gl import Blit, Fbo, Texture  # noqa: E402
from modules.flow import DepthVelocity, FlowUtil, SignedField, VisualizationFieldSettings, filter_depth_steps, track_depth  # noqa: E402

SEGMENTS: dict[str, tuple[float, float]] = {'still': (1.0, 5.0), 'slow': (5.0, 10.0), 'fast': (10.0, 1e9)}
VIEWS: dict[int, str] = {1: 'z of the track', 2: 'z after the threshold', 3: 'depth'}
ARGUMENTS: tuple[str, ...] = ('time', 'threshold', 'tolerance', 'dead_zone', 'catch_up_time', 'motion_floor', 'window')


def context(visible: bool, width: int, height: int):
    """A GL 4.6 window, current, with the fullscreen quad bound as the app has it."""
    if not glfw.init():
        raise RuntimeError('GLFW did not start')
    glfw.window_hint(glfw.VISIBLE, glfw.TRUE if visible else glfw.FALSE)
    glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 4)
    glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 6)
    glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)
    glfw.window_hint(glfw.RESIZABLE, glfw.FALSE)
    window = glfw.create_window(width, height, 'depth track', None, None)
    if not window:
        raise RuntimeError('no GL window')
    glfw.make_context_current(window)
    glfw.swap_interval(1)
    vertices = np.array([-1, -1, 0, 0, 1, -1, 1, 0, 1, 1, 1, 1, -1, 1, 0, 1], dtype=np.float32)
    glBindBuffer(GL_ARRAY_BUFFER, glGenBuffers(1))
    glBufferData(GL_ARRAY_BUFFER, vertices.nbytes, vertices, GL_STATIC_DRAW)
    glBindVertexArray(glGenVertexArrays(1))
    glEnableVertexAttribArray(0)
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 16, ctypes.c_void_p(0))
    glEnableVertexAttribArray(1)
    glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 16, ctypes.c_void_p(8))
    return window


def upload(texture: Texture, array: np.ndarray, internal_format: int) -> None:
    h, w = array.shape[:2]
    if not texture.allocated or texture.width != w or texture.height != h:
        texture.allocate(w, h, internal_format)
    glBindTexture(GL_TEXTURE_2D, texture.tex_id)
    glTexImage2D(GL_TEXTURE_2D, 0, internal_format, w, h, 0, GL_RED if internal_format == GL_R32F else GL_RG, GL_FLOAT,
                 np.ascontiguousarray(array, dtype=np.float32))
    glBindTexture(GL_TEXTURE_2D, 0)


_READ_BACK: dict[tuple[int, int], Fbo] = {}


def read_back(texture: Texture) -> np.ndarray:
    """A texture's RGBA as float32 (H, W, 4), through a float32 copy: read_to_numpy gives zeros for R16F."""
    key = (texture.width, texture.height)
    if key not in _READ_BACK:
        _READ_BACK[key] = Fbo()
        _READ_BACK[key].allocate(texture.width, texture.height, GL_RGBA32F)
    FlowUtil.blit(_READ_BACK[key], texture)
    pixels = _READ_BACK[key].read_to_numpy(flip=False)
    assert pixels is not None
    return pixels


def overrides(items: list[str]) -> dict[str, float]:
    result: dict[str, float] = {}
    for item in items:
        name, value = item.split('=', 1)
        if name not in ARGUMENTS:
            raise SystemExit(f'unknown argument {name}: one of {", ".join(ARGUMENTS)}')
        result[name] = float(value)
    return result


def arguments(call: dict, changed: dict[str, float]) -> dict:
    a = {name: call.get(name, default) for name, default in (('time', 50.0), ('threshold', 0.0), ('tolerance', 0.1), ('dead_zone', 0.0),
                                                             ('catch_up_time', 0.0), ('motion_floor', 0.5), ('window', 1))}
    a.update(changed)
    a['window'] = int(a['window'])
    return a


class Frames:
    """The saved calls, read as they are needed."""

    def __init__(self, folder: Path) -> None:
        self.folder = folder
        self.calls: list[dict] = json.loads((folder / 'calls.json').read_text())
        self.width, self.height = self.calls[0]['width'], self.calls[0]['height']
        t = np.array([c['t'] for c in self.calls])
        self.t: np.ndarray = t - t[0]

    def __len__(self) -> int:
        return len(self.calls)

    def load(self, k: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        return (np.load(self.folder / f'depth_{k:04d}.npy').astype(np.float32), np.load(self.folder / f'moved_{k:04d}.npy').astype(np.float32),
                np.load(self.folder / f'mask_{k:04d}.npy'))

    def segment(self, k: int) -> str:
        return next((name for name, (lo, hi) in SEGMENTS.items() if lo <= self.t[k] < hi), '')


class Measures:
    """One loop's measures of one track."""

    def __init__(self) -> None:
        self.noise: dict[str, list[np.ndarray]] = {'edge': [], 'inside': [], 'table': []}
        self.body_z: list[float] = []
        self.body_depth: list[float] = []
        self.times: list[float] = []
        self.restarts: dict[str, list[float]] = {name: [] for name in SEGMENTS}
        self.milliseconds: list[float] = []
        self._before: np.ndarray | None = None

    def add(self, t: float, segment: str, track: np.ndarray, out: np.ndarray, depth: np.ndarray, mask: np.ndarray, width: int,
            milliseconds: float) -> None:
        inside = cv2.distanceTransform(mask.astype(np.uint8), cv2.DIST_L2, 3)
        outside = cv2.distanceTransform((~mask).astype(np.uint8), cv2.DIST_L2, 3)
        core = mask & (depth > 0) & (inside > 12)
        near = float(np.median(depth[core])) if core.any() else np.nan
        if self._before is not None and segment:
            held = core & (self._before[..., 1] > 0)
            if held.any():                        # a track there the frame before that now has no rate: it started over
                self.restarts[segment].append(float(np.mean((track[..., 2] == 0.0)[held] & (track[..., 1][held] > 0))))
        self._before = track
        self.milliseconds.append(milliseconds)
        self.times.append(t)
        self.body_depth.append(near)
        self.body_z.append(float(np.mean(track[..., 0][core])) * width if core.any() else np.nan)
        if segment == 'still':
            z = np.abs(out) * width
            self.noise['edge'].append(z[mask & (depth > 0) & (inside <= 6)])
            self.noise['inside'].append(z[core])
            self.noise['table'].append(z[(~mask) & (depth > 0) & (outside > 16) & (depth < near + 0.8)])

    def report(self, span: float, width: int) -> str:
        lines: list[str] = []
        for name, parts in self.noise.items():
            v = np.concatenate(parts) if parts else np.zeros(0)
            lines.append(f'still {name} z>0 {np.mean(v > 0) if v.size else 0:.4f} p99 {np.percentile(v, 99) if v.size else 0:.2f}')
        d, t = np.array(self.body_depth), np.array(self.times)
        real = np.zeros_like(d); real[1:] = -(d[1:] - d[:-1]) / (d[:-1] * span) * width
        kernel = np.ones(5) / 5
        for name in ('slow', 'fast'):
            lo, hi = SEGMENTS[name]
            sel = (t >= lo) & (t < hi)
            if sel.sum() < 10:
                continue
            r = np.convolve(np.nan_to_num(real), kernel, 'same')[sel]
            z = np.convolve(np.nan_to_num(np.array(self.body_z)), kernel, 'same')[sel]
            lines.append(f'{name} z/real {np.dot(z, r) / max(np.dot(r, r), 1e-12):.2f} (corr {np.corrcoef(z, r)[0, 1]:.2f})')
        lines.append('restarts on the body per frame ' + ', '.join(f'{name} {np.mean(v) * 100:.2f}%' for name, v in self.restarts.items() if v))
        lines.append(f'{np.median(self.milliseconds):.2f} ms per frame')
        return ' | '.join(lines)


class Side:
    """One of A and B: its arguments, its track, what it shows and what it measured."""

    def __init__(self, name: str, changed: dict[str, float], width: int, height: int, show: bool) -> None:
        self.name, self.changed = name, changed
        self.track = DepthVelocity()
        self.track.allocate(width, height)
        self.view_settings = VisualizationFieldSettings()
        self.field = SignedField(self.view_settings) if show else None
        if self.field is not None:
            self.field.allocate(width, height)
        self.measures = Measures()

    def step(self, frames: Frames, k: int, depth_texture: Texture, moved_texture: Texture) -> dict:
        a = arguments(frames.calls[k], self.changed)
        c = frames.calls[k]
        self.track.update_depth(depth_texture, c['delta_time'], c['view_span'], tuple(c['depth_step']), a['window'], a['time'], a['threshold'],
                                a['tolerance'], a['dead_zone'], a['catch_up_time'], a['motion_floor'], moved_texture)
        return a

    def show(self, view: int, depth_texture: Texture, scale: float, width: int) -> Texture | None:
        if self.field is None:
            return None
        if view == 3:
            self.view_settings.scalar_scale = scale / 2.0
            self.field.set(depth_texture, depth=False, channel=0, unit=1.0)        # 2 m full
        else:
            self.view_settings.scalar_scale = scale / 2.0
            self.field.set(self.track.difference if view == 1 else self.track.depth_velocity, depth=True, channel=0, unit=float(width))  # 2 px per frame full
        self.field.update()
        return self.field.texture


def verify(frames: Frames, changed: dict[str, float]) -> int:
    """A's first frames on the GPU against the numpy reference."""
    context(False, 64, 64)
    w, h = frames.width, frames.height
    side = Side('A', changed, w, h, False)
    depth_texture, moved_texture = Texture(), Texture()
    state, catch_up = np.zeros((h, w, 4)), np.zeros((h, w, 2))
    for k in range(min(40, len(frames))):
        depth, moved, _ = frames.load(k)
        upload(depth_texture, depth, GL_R32F)
        upload(moved_texture, moved, GL_RG32F)
        a = side.step(frames, k, depth_texture, moved_texture)
        gpu = read_back(side.track.difference)
        assert gpu is not None
        c = frames.calls[k]
        step = (c['depth_step'][0] * w, c['depth_step'][1] * h) if c['depth_step'][0] > 0 else (1.0, 1.0)
        state, catch_up = track_depth(filter_depth_steps(depth.astype(np.float64), step, a['tolerance']), state, catch_up,
                                      moved.astype(np.float64) * np.array([w, h]), c['delta_time'], a['window'], step, a['tolerance'],
                                      a['time'], c['view_span'], a['dead_zone'], a['catch_up_time'], a['motion_floor'])
        held = (gpu[..., 1] > 0) | (state[..., 1] > 0)
        dz = np.abs(state[..., 0] - gpu[..., 0])[held] * w
        if k % 5 == 0 or k == 39:
            print(f'frame {k:2d}: z differs p50 {np.median(dz):.1e} p99.9 {np.percentile(dz, 99.9):.1e} px per frame | '
                  f'z nonzero GPU {np.mean(gpu[..., 0][held] != 0):.4f} numpy {np.mean(state[..., 0][held] != 0):.4f}')
    glfw.terminate()
    return 0


def main() -> int:
    assert __doc__ is not None
    parser = ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('frames', type=Path)
    parser.add_argument('--a', action='append', default=[], metavar='NAME=VALUE', help="override one of A's arguments")
    parser.add_argument('--b', action='append', default=[], metavar='NAME=VALUE', help="override one of B's arguments")
    parser.add_argument('--fast', action='store_true', help='one loop without the window, then the measures')
    parser.add_argument('--verify', action='store_true', help="A's first frames against the numpy reference")
    parser.add_argument('--scale', type=float, default=0.9, help='window size per pixel of the frames (default 0.9)')
    parser.add_argument('--loops', type=int, default=0, help='stop after this many loops (default: until esc)')
    parser.add_argument('--view', type=int, choices=sorted(VIEWS), default=1, help='the view to start in (default 1)')
    parser.add_argument('--snapshot', type=Path, help='save the window halfway through the first loop, as an image')
    args = parser.parse_args()
    frames = Frames(args.frames)
    if args.verify:
        return verify(frames, overrides(args.a))

    w, h = frames.width, frames.height
    panel_w, panel_h = int(w * args.scale), int(h * args.scale)
    window = context(not args.fast, panel_w * 2, panel_h)
    sides = [Side('A', overrides(args.a), w, h, not args.fast), Side('B', overrides(args.b), w, h, not args.fast)]
    depth_texture, moved_texture = Texture(), Texture()
    span = float(np.median([c['view_span'] for c in frames.calls]))
    controls = {'view': args.view, 'paused': False, 'step': 0, 'scale': 1.0}

    def on_key(_window, key, _scancode, action, _mods) -> None:
        if action not in (glfw.PRESS, glfw.REPEAT):
            return
        if key in (glfw.KEY_1, glfw.KEY_2, glfw.KEY_3):
            controls['view'] = key - glfw.KEY_0
        elif key == glfw.KEY_SPACE:
            controls['paused'] = not controls['paused']
        elif key == glfw.KEY_RIGHT:
            controls['step'] = 1
        elif key == glfw.KEY_LEFT:
            controls['step'] = -1
        elif key in (glfw.KEY_EQUAL, glfw.KEY_KP_ADD):
            controls['scale'] *= 1.5
        elif key in (glfw.KEY_MINUS, glfw.KEY_KP_SUBTRACT):
            controls['scale'] /= 1.5
        elif key == glfw.KEY_ESCAPE:
            glfw.set_window_should_close(window, True)

    glfw.set_key_callback(window, on_key)
    for side in sides:
        print(f'{side.name}: {side.changed or "as recorded"}')

    loop, k, started = 1, 0, time.perf_counter()
    while not glfw.window_should_close(window):
        if k >= len(frames):                      # the loop's end: its measures, and from no track again
            for side in sides:
                print(f'loop {loop} {side.name}: {side.measures.report(span, w)}', flush=True)
                side.track.reset()
                side.measures = Measures()
            if args.fast or (args.loops and loop >= args.loops):
                break
            loop, k, started = loop + 1, 0, time.perf_counter()
            continue
        if not args.fast and controls['paused'] and controls['step'] == 0:
            glfw.wait_events_timeout(0.05)
            continue
        if not args.fast and not controls['paused']:
            wait = started + frames.t[k] - time.perf_counter()
            if wait > 0:
                glfw.wait_events_timeout(wait)
                continue
        if controls['step'] < 0:                  # a frame back: the track follows from no track again up to it
            target = max(k - 2, 0)
            for side in sides:
                side.track.reset()
                side.measures = Measures()
            k = 0
            while k < target:
                depth, moved, mask = frames.load(k)
                upload(depth_texture, depth, GL_R32F); upload(moved_texture, moved, GL_RG32F)
                for side in sides:
                    side.step(frames, k, depth_texture, moved_texture)
                k += 1
        controls['step'] = 0

        depth, moved, mask = frames.load(k)
        upload(depth_texture, depth, GL_R32F)
        upload(moved_texture, moved, GL_RG32F)
        segment = frames.segment(k)
        for side in sides:
            glFinish()
            began = time.perf_counter()
            side.step(frames, k, depth_texture, moved_texture)
            glFinish()                            # the track's own time on the GPU, before its read back
            milliseconds = (time.perf_counter() - began) * 1000.0
            track = read_back(side.track.difference)
            out = read_back(side.track.depth_velocity)[..., 0]

            side.measures.add(float(frames.t[k]), segment, track, out, depth, mask, w, milliseconds)

        if not args.fast:
            shown = [side.show(controls['view'], depth_texture, controls['scale'], w) for side in sides]
            glBindFramebuffer(GL_FRAMEBUFFER, 0)
            glClearColor(0.0, 0.0, 0.0, 1.0)
            glClear(GL_COLOR_BUFFER_BIT)
            for i, texture in enumerate(shown):
                if texture is not None:
                    glViewport(i * panel_w, 0, panel_w, panel_h)
                    Blit.use(texture)
            if args.snapshot is not None and k == len(frames) // 2:
                pixels = np.frombuffer(glReadPixels(0, 0, panel_w * 2, panel_h, GL_RGB, GL_UNSIGNED_BYTE), np.uint8).reshape(panel_h, panel_w * 2, 3)
                cv2.imwrite(str(args.snapshot), np.flipud(pixels)[..., ::-1])
            glfw.swap_buffers(window)
            glfw.set_window_title(window, f'A | B   {VIEWS[controls["view"]]}   frame {k + 1}/{len(frames)}  {frames.t[k]:5.2f} s  {segment}'
                                          f'{"   paused" if controls["paused"] else ""}   loop {loop}')
            glfw.poll_events()
        k += 1
    glfw.terminate()
    return 0


if __name__ == '__main__':
    sys.exit(main())
