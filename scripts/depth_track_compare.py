"""Play saved frames through the Lucas–Kanade flow's shaders, A beside B, and measure both.

    python scripts/depth_track_compare.py FRAMES [--a NAME=VALUE ...] [--b NAME=VALUE ...] [--fast] [--verify]

FRAMES is what depth_track_capture saved from one loop of a recording (temp/depth_track/RECORDING).
No app and no camera: a window gives the GL context, and two `LucasKanadeFlow`, A and B, measure the
saved frames with the settings they were recorded with; `--a` and `--b` override any of them by its
path in the panel (lucas_kanade.levels, lucas_kanade.threshold, depth_track.catch_up, z, ...). The window plays the loop at its
recorded rate, again and again, A on the left, B on the right:

    1  z of the track, red toward the camera, green away     space  pause
    2  z after the threshold, as the fluid gets it           ← →    a frame back or on, paused
    3  the depth                                              + -    brighter, darker
    4  the velocity's direction (x/y)                         esc    quit
    5  the velocity as arrows

The loop splits into a still part and moves, each a slow half then a fast one: by `segments.json`
beside the frames ({"still": [from, to], "moves": [[from, fast from, to], ...]}, seconds) where there
is one, else by the frames' motion (the mean luminance change in the mask): the moves are where it
rises above the still part's for a while, and a move's fast half is where it steps up most.

Each loop prints, for A and B, in the still part: x/y and z as the fluid gets them, at the edge (0–6
px inside the mask) and inside (12 px and more), and z on the table (outside, 16 px and more); per
move, the body's x/y and z in its slow and fast half and fast over slow, how its z follows its real
depth change, and the frames x/y takes to stop after it; the restarts of the depth track, and the
flow's time on the GPU per frame. `--fast` runs one loop without the window, as fast as it goes;
`--verify` compares A's first frames with the numpy references instead.
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
from OpenGL.GL import (GL_ARRAY_BUFFER, GL_COLOR_BUFFER_BIT, GL_FALSE, GL_FLOAT, GL_FRAMEBUFFER, GL_R32F, GL_RED, GL_RGB, GL_RGBA32F,
                       GL_STATIC_DRAW, GL_TEXTURE_2D, GL_UNSIGNED_BYTE, glBindBuffer, glBindFramebuffer, glBindTexture, glBindVertexArray,
                       glBufferData, glClear, glClearColor, glEnableVertexAttribArray, glFinish, glGenBuffers, glGenVertexArrays,
                       glReadPixels, glTexImage2D, glVertexAttribPointer, glViewport)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from modules.gl import Blit, Fbo, Texture  # noqa: E402
from modules.flow import (FlowUtil, LucasKanadeFlow, LucasKanadeFlowSettings, Patch, SignedField, VelocityField,  # noqa: E402
                          VisualizationFieldSettings, filter_depth_steps, guide_depth, lucas_kanade, track_depth)

WARM_UP: float = 1.0          # s at the start the flows settle in, measured in no part
SHORTEST_PAUSE: float = 1.0   # s of stillness that end a move
RENAMED: dict[str, str] = {'z_time': 'z_smooth', 'z_catch_up_time': 'z_catch_up', 'xy_follow_z_time': 'xy_smooth', 'xy_follow_z': 'xy_smooth', 'xy_regularizer': 'xy_damping',
                           'strength_x': 'x_gain', 'strength_y': 'y_gain', 'strength_z': 'z_gain'}   # in frames saved before
SWITCHED: dict[str, str] = {'x_gain': 'x', 'y_gain': 'y', 'z_gain': 'z'}   # in frames saved with gains: a gain of 0 is the axis off
GROUPED: dict[str, str] = {'xy_levels': 'lucas_kanade.levels', 'xy_window': 'lucas_kanade.window', 'xy_damping': 'lucas_kanade.damping',
                           'xy_same_surface': 'lucas_kanade.same_surface', 'xy_threshold': 'lucas_kanade.threshold', 'xy_median': 'lucas_kanade.median',
                           'z_edge': 'depth_track.edge', 'z_window': 'depth_track.window', 'z_tolerance': 'depth_track.tolerance',
                           'z_smooth': 'depth_track.smooth', 'xy_smooth': 'depth_track.xy_smooth',
                           'z_catch_up': 'depth_track.catch_up', 'z_dead_zone': 'depth_track.dead_zone', 'z_threshold': 'depth_track.threshold'}   # in frames saved flat, before the groups
MOVED: float = 0.05          # luminance change of a pixel that moves
MOVED_FRAMES: int = 5         # frames it changes that much in, over the loop
VIEWS: dict[int, str] = {1: 'z of the track', 2: 'z after the threshold', 3: 'depth', 4: 'velocity direction', 5: 'velocity arrows'}


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


def upload(texture: Texture, array: np.ndarray) -> None:
    """One channel, float32, into an R32F texture."""
    h, w = array.shape[:2]
    if not texture.allocated or texture.width != w or texture.height != h:
        texture.allocate(w, h, GL_R32F)
    glBindTexture(GL_TEXTURE_2D, texture.tex_id)
    glTexImage2D(GL_TEXTURE_2D, 0, GL_R32F, w, h, 0, GL_RED, GL_FLOAT, np.ascontiguousarray(array, dtype=np.float32))
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


def recorded_settings(recorded: dict) -> dict:
    """The settings a frame was saved with, as the settings' dict: saved flat before the groups, nested since."""
    result: dict = {}
    for key, value in recorded.items():
        key = RENAMED.get(key, key)
        if key in SWITCHED:
            key, value = SWITCHED[key], value != 0.0
        *groups, name = GROUPED.get(key, key).split('.')
        target: dict = result
        for group in groups:
            target = target.setdefault(group, {})
        target[name] = value
    return result


def setting_owner(settings: LucasKanadeFlowSettings, path: str) -> tuple[object, str]:
    """The settings object that holds the setting at `path` (lucas_kanade.levels) and its name there."""
    *groups, name = path.split('.')
    owner: object = settings
    for group in groups:
        owner = getattr(owner, group, None)
        if owner is None:
            raise SystemExit(f'unknown settings group {group} in {path}')
    return owner, name


def overrides(items: list[str]) -> dict[str, object]:
    defaults = LucasKanadeFlowSettings()
    result: dict[str, object] = {}
    for item in items:
        name, value = item.split('=', 1)
        owner, field = setting_owner(defaults, name)
        fields = getattr(owner, 'fields')
        if field not in fields:
            raise SystemExit(f'unknown setting {name}: one of {", ".join(fields)}')
        kind = type(getattr(owner, field))
        if issubclass(kind, Patch):                              # by name (SEVEN) or radius (3)
            result[name] = Patch[value.upper()] if value.isalpha() else Patch(int(value))
        else:
            result[name] = value.lower() in ('1', 'true', 'on', 'yes') if kind is bool else kind(float(value)) if kind is int else kind(value)
    return result


class Frames:
    """The saved frames, read as they are needed."""

    def __init__(self, folder: Path) -> None:
        self.folder = folder
        if not (folder / 'luminance_0000.npy').exists():
            raise SystemExit(f'{folder} has no luminance: capture it again with depth_track_capture')
        self.calls: list[dict] = json.loads((folder / 'calls.json').read_text())
        self.width, self.height = self.calls[0]['width'], self.calls[0]['height']
        t = np.array([c['t'] for c in self.calls])
        self.t: np.ndarray = t - t[0]
        motion = self.motion()
        given = folder / 'segments.json'
        if given.exists():
            parts = json.loads(given.read_text())
            self.still: tuple[float, float] = tuple(parts['still'])
            self.moves: list[tuple[float, float, float]] = [tuple(move) for move in parts['moves']]
        else:
            self.still, self.moves = split_moves(self.t, motion)

    def __len__(self) -> int:
        return len(self.calls)

    def load(self, k: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        return (np.load(self.folder / f'luminance_{k:04d}.npy').astype(np.float32), np.load(self.folder / f'depth_{k:04d}.npy').astype(np.float32),
                np.load(self.folder / f'mask_{k:04d}.npy'))

    def motion(self) -> np.ndarray:
        """Per frame, the mean luminance change in the mask since the frame before (0 to 1). Also
        `moved`: the pixels whose luminance changes by more than MOVED in MOVED_FRAMES frames or more,
        so a still background in the mask can be told from the player."""
        motion, before = np.zeros(len(self)), None
        changes = np.zeros((self.height, self.width), np.int32)
        for k in range(len(self)):
            luminance, mask = np.load(self.folder / f'luminance_{k:04d}.npy').astype(np.float32), np.load(self.folder / f'mask_{k:04d}.npy')
            if before is not None:
                change = np.abs(luminance - before)
                changes += change > MOVED
                if mask.any():
                    motion[k] = float(np.mean(change[mask]))
            before = luminance
        self.moved: np.ndarray = changes >= MOVED_FRAMES
        return motion

    def segment(self, k: int) -> str:
        """'still', 'move N slow' or 'move N fast' (N from 1), or '' between them."""
        t = self.t[k]
        if self.still[0] <= t < self.still[1]:
            return 'still'
        for n, (begin, fast, end) in enumerate(self.moves, 1):
            if begin <= t < end:
                return f'move {n} {"slow" if t < fast else "fast"}'
        return ''

    def describe(self) -> str:
        return (f'still {self.still[0]:.1f}-{self.still[1]:.1f} s; '
                + '; '.join(f'move {n} {b:.1f}-{e:.1f} s, fast from {f:.1f}' for n, (b, f, e) in enumerate(self.moves, 1)))


def split_moves(t: np.ndarray, motion: np.ndarray) -> tuple[tuple[float, float], list[tuple[float, float, float]]]:
    """The still part and the moves (from, fast from, to; s) of a loop, from its motion per frame.

    A move is where the motion, over 5 frames, is past the still part's (its first 2 s after the warm
    up) by a tenth of the way to the loudest; pauses shorter than SHORTEST_PAUSE don't end one. Its
    fast half begins where the motion steps up most: the split with the least spread on both sides."""
    smooth = np.convolve(motion, np.ones(5) / 5, 'same')
    calm = float(np.median(smooth[(t >= WARM_UP) & (t < WARM_UP + 2.0)]))
    moving = smooth > calm + 0.1 * (float(np.percentile(smooth, 95)) - calm)
    moving[t < WARM_UP] = False
    edges = np.flatnonzero(np.diff(np.concatenate(([0], moving.astype(int), [0]))))
    runs = [[int(a), int(b)] for a, b in zip(edges[::2], edges[1::2])]     # frames [a, b)
    joined: list[list[int]] = []
    for run in runs:
        if joined and t[run[0]] - t[joined[-1][1] - 1] < SHORTEST_PAUSE:
            joined[-1][1] = run[1]
        else:
            joined.append(run)
    joined = [run for run in joined if t[run[1] - 1] - t[run[0]] >= 0.3]
    moves: list[tuple[float, float, float]] = []
    for a, b in joined:
        part = smooth[a:b]
        splits = range(3, len(part) - 2)
        spread = [float(np.var(part[:i]) * i + np.var(part[i:]) * (len(part) - i)) for i in splits]
        fast = a + (splits[int(np.argmin(spread))] if spread else len(part) // 2)
        moves.append((float(t[a]), float(t[fast]), float(t[b - 1]) + 1e-6))
    still_end = moves[0][0] if moves else float(t[-1])
    return (WARM_UP, still_end), moves


class Measures:
    """One loop's measures of one flow."""

    def __init__(self) -> None:
        self.still: dict[str, list[np.ndarray]] = {'xy edge': [], 'xy inside': [], 'z edge': [], 'z inside': [], 'z table': []}
        self.segments: list[str] = []
        self.body_xy: list[float] = []            # px per frame, as the fluid gets it
        self.moving_xy: list[float] = []          # the same on the body's pixels that move in the loop
        self.body_z: list[float] = []             # the track's z, px per frame
        self.moving_z: list[float] = []
        self.mask_share: list[float] = []         # of the image, in the still part
        self.static_share: list[float] = []       # of the mask that never moves, in the still part
        self.body_depth: list[float] = []
        self.times: list[float] = []
        self.restarts: dict[str, list[float]] = {'still': [], 'moving': []}
        self.milliseconds: list[float] = []
        self._before: np.ndarray | None = None

    def add(self, t: float, segment: str, xy: np.ndarray, track: np.ndarray, z: np.ndarray, depth: np.ndarray, mask: np.ndarray,
            moved: np.ndarray, milliseconds: float) -> None:
        """`xy` and `z` are as the fluid gets them, in px per frame; `track` is the depth track's state;
        `moved` the pixels that move in the loop."""
        width = depth.shape[1]
        inside = cv2.distanceTransform(mask.astype(np.uint8), cv2.DIST_L2, 3)
        outside = cv2.distanceTransform((~mask).astype(np.uint8), cv2.DIST_L2, 3)
        core = mask & (depth > 0) & (inside > 12)
        edge = mask & (depth > 0) & (inside <= 6)
        near = float(np.median(depth[core])) if core.any() else np.nan
        if self._before is not None and segment:
            held = core & (self._before[..., 1] > 0)
            if held.any():                        # a track there the frame before that now has no rate: it started over
                restarted = float(np.mean((track[..., 2] == 0.0)[held] & (track[..., 1][held] > 0)))
                self.restarts['still' if segment == 'still' else 'moving'].append(restarted)
        self._before = track
        self.milliseconds.append(milliseconds)
        self.times.append(t)
        self.segments.append(segment)
        self.body_depth.append(near)
        self.body_xy.append(float(np.mean(xy[core])) if core.any() else 0.0)
        self.body_z.append(float(np.mean(track[..., 0][core])) * width if core.any() else np.nan)
        moving = core & moved
        self.moving_xy.append(float(np.mean(xy[moving])) if moving.any() else 0.0)
        self.moving_z.append(float(np.mean(track[..., 0][moving])) * width if moving.any() else np.nan)
        if segment == 'still':
            self.mask_share.append(float(np.mean(mask)))
            self.static_share.append(float(np.mean(~moved[mask])) if mask.any() else 0.0)
            self.still['xy edge'].append(xy[edge]); self.still['xy inside'].append(xy[core])
            self.still['z edge'].append(z[edge]); self.still['z inside'].append(z[core])
            self.still['z table'].append(z[(~mask) & (depth > 0) & (outside > 16) & (depth < near + 0.8)])

    def report(self, span: float, width: int, moves: int) -> str:
        lines: list[str] = []
        for name, parts in self.still.items():
            v = np.concatenate(parts) if parts else np.zeros(0)
            lines.append(f'still {name} >0 {np.mean(v > 0) if v.size else 0:.4f} p99 {np.percentile(v, 99) if v.size else 0:.2f}')
        d, segments = np.array(self.body_depth), np.array(self.segments)
        real = np.zeros_like(d); real[1:] = -(d[1:] - d[:-1]) / (d[:-1] * span) * width
        kernel = np.ones(5) / 5
        real = np.convolve(np.nan_to_num(real), kernel, 'same')
        body_z = np.convolve(np.nan_to_num(np.array(self.body_z)), kernel, 'same')
        body_xy, moving_xy = np.array(self.body_xy), np.array(self.moving_xy)
        moving_z = np.convolve(np.nan_to_num(np.array(self.moving_z)), kernel, 'same')
        lines.append(f'mask {np.mean(self.mask_share) * 100:.0f}% of the image, {np.mean(self.static_share) * 100:.0f}% of it never moves')

        def halves(series: np.ndarray, slow: np.ndarray, fast: np.ndarray) -> str:
            a, b = float(np.mean(np.abs(series[slow]))), float(np.mean(np.abs(series[fast])))
            return f'{a:.2f} -> {b:.2f} x{b / max(a, 1e-6):.1f}'

        for n in range(1, moves + 1):
            slow, fast = segments == f'move {n} slow', segments == f'move {n} fast'
            if fast.sum() < 3:
                continue
            move = slow | fast
            after = np.flatnonzero(move)[-1] + 1
            calm = body_xy[after:after + 60] <= 0.1 * float(np.max(body_xy[move]))
            stop = f'xy stops {int(np.argmax(calm))} frames after' if calm.any() else 'xy not stopped in 60 frames'
            if slow.sum() < 3:                    # a fast move alone, as before a freeze
                lines.append(f'move {n}: fast xy {np.mean(body_xy[fast]):.2f} | {stop}')
                continue
            r, z = real[move], body_z[move]
            lines.append(f'move {n} slow -> fast: xy {halves(body_xy, slow, fast)}, moving {halves(moving_xy, slow, fast)} | '
                         f'z {halves(body_z, slow, fast)}, moving {halves(moving_z, slow, fast)}, real {halves(real, slow, fast)} | '
                         f'z/real {np.dot(z, r) / max(np.dot(r, r), 1e-12):.2f} (corr {np.corrcoef(z, r)[0, 1]:.2f}) | {stop}')
        lines.append('restarts ' + ', '.join(f'{name} {np.mean(v) * 100:.2f}%' for name, v in self.restarts.items() if v))
        lines.append(f'{np.median(self.milliseconds):.2f} ms per frame')
        return '\n    '.join([' | '.join(lines[:5])] + lines[5:])


class Side:
    """One of A and B: its settings, its flow, what it shows and what it measured."""

    def __init__(self, name: str, recorded: dict, changed: dict[str, object], width: int, height: int, show: bool) -> None:
        self.name, self.changed = name, changed
        self.settings = LucasKanadeFlowSettings()
        self.settings.update_from_dict(recorded_settings(recorded))
        for path, value in changed.items():
            owner, name = setting_owner(self.settings, path)
            setattr(owner, name, value)
        self.flow = LucasKanadeFlow(self.settings)
        self.flow.allocate(width, height)
        self.view_settings = VisualizationFieldSettings()
        self.signed = SignedField(self.view_settings) if show else None
        self.velocity = VelocityField(VisualizationFieldSettings()) if show else None
        if self.signed is not None and self.velocity is not None:
            self.signed.allocate(width, height)
            self.velocity.allocate(width, height)
        self.measures = Measures()

    def step(self, frames: Frames, k: int, luminance: Texture, depth: Texture, mask: Texture) -> None:
        c = frames.calls[k]
        self.flow.set_input(luminance)
        self.flow.set_mask(mask)
        self.flow.update(depth, c['delta_time'], c['view_span'], tuple(c['depth_step']))

    def xy_px(self, velocity: np.ndarray, delta_time: float, width: int) -> np.ndarray:
        """The velocity's x/y (image widths per second) as px per frame."""
        return np.hypot(velocity[..., 0], velocity[..., 1]) * delta_time * width

    def show(self, view: int, depth: Texture, scale: float, width: int) -> Texture | None:
        if self.signed is None or self.velocity is None:
            return None
        if view >= 4:
            self.velocity.settings.toggle_scalar = view == 4
            self.velocity.settings.scale = scale
            self.velocity.set(self.flow.velocity)
            self.velocity.update()
            return self.velocity.texture
        self.view_settings.scalar_scale = scale / 2.0
        if view == 3:
            self.signed.set(depth, depth=False, channel=0, unit=1.0)               # 2 m full
        else:
            tracked = self.flow.depth_velocity
            self.signed.set(tracked.difference if view == 1 else tracked.depth_velocity, depth=True, channel=0, unit=float(width))  # 2 px per frame full
        self.signed.update()
        return self.signed.texture


def verify(frames: Frames, changed: dict[str, object], frames_to_check: int) -> int:
    """A's first frames on the GPU against the numpy references: the x/y solve, and the depth track carried by it."""
    context(False, 64, 64)
    w, h = frames.width, frames.height
    side = Side('A', frames.calls[0]['settings'], changed, w, h, False)
    xy, z = side.settings.lucas_kanade, side.settings.depth_track
    luminance, depth_texture, mask_texture = Texture(), Texture(), Texture()
    state, catch_up, previous = np.zeros((h, w, 4)), np.zeros((h, w, 2)), None
    last = min(frames_to_check, len(frames)) - 1
    for k in range(last + 1):
        lum, depth, mask = frames.load(k)
        upload(luminance, lum); upload(depth_texture, depth); upload(mask_texture, mask.astype(np.float32))
        side.step(frames, k, luminance, depth_texture, mask_texture)
        if previous is None:
            previous = lum
            continue
        solve = read_back(side.flow._solve.texture)[..., :2] * np.array([w, h])                  # px per frame
        now, then = (np.round(f * 255.0).astype(np.float64) / 255.0 for f in (lum, previous))    # as the R8 frames hold them
        reference = lucas_kanade(then, now, xy.levels, xy.window, xy.damping, xy.median,
                                 depth.astype(np.float64) if xy.same_surface else None, z.tolerance)
        previous = lum
        c = frames.calls[k]
        step = (c['depth_step'][0] * w, c['depth_step'][1] * h) if c['depth_step'][0] > 0 else (1.0, 1.0)
        measured = guide_depth(depth.astype(np.float64), now) if z.guide else depth.astype(np.float64)
        state, catch_up = track_depth(filter_depth_steps(measured, step, z.edge), state, catch_up, solve.astype(np.float64),
                                      c['delta_time'], z.window, step, z.tolerance, z.smooth, c['view_span'], z.dead_zone,
                                      z.catch_up, xy.threshold)
        gpu = read_back(side.flow.depth_velocity.difference)
        if k % 5 == 0 or k == last:
            dxy = np.linalg.norm(solve - reference, axis=-1)
            held = (gpu[..., 1] > 0) | (state[..., 1] > 0)
            dz = np.abs(state[..., 0] - gpu[..., 0])[held] * w
            print(f'frame {k:2d}: x/y solve differs p50 {np.median(dxy):.3f} p99 {np.percentile(dxy, 99):.3f} px | '
                  f'z differs p50 {np.median(dz):.1e} p99.9 {np.percentile(dz, 99.9):.1e} px per frame')
    glfw.terminate()
    return 0


def main() -> int:
    assert __doc__ is not None
    parser = ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('frames', type=Path)
    parser.add_argument('--a', action='append', default=[], metavar='NAME=VALUE', help="override one of A's settings")
    parser.add_argument('--b', action='append', default=[], metavar='NAME=VALUE', help="override one of B's settings")
    parser.add_argument('--fast', action='store_true', help='one loop without the window, then the measures')
    parser.add_argument('--verify', action='store_true', help="A's first frames against the numpy references")
    parser.add_argument('--verify-frames', type=int, default=30, help='how many frames --verify checks (default 30; the numpy reference is slow)')
    parser.add_argument('--scale', type=float, default=0.9, help='window size per pixel of the frames (default 0.9)')
    parser.add_argument('--loops', type=int, default=0, help='stop after this many loops (default: until esc)')
    parser.add_argument('--view', type=int, choices=sorted(VIEWS), default=1, help='the view to start in (default 1)')
    parser.add_argument('--snapshot', type=Path, help='save the window halfway through the first loop, as an image')
    args = parser.parse_args()
    frames = Frames(args.frames)
    if args.verify:
        return verify(frames, overrides(args.a), args.verify_frames)

    w, h = frames.width, frames.height
    panel_w, panel_h = int(w * args.scale), int(h * args.scale)
    window = context(not args.fast, panel_w * 2, panel_h)
    recorded = frames.calls[0]['settings']
    sides = [Side('A', recorded, overrides(args.a), w, h, not args.fast), Side('B', recorded, overrides(args.b), w, h, not args.fast)]
    luminance, depth_texture, mask_texture = Texture(), Texture(), Texture()
    span = float(np.median([c['view_span'] for c in frames.calls]))
    controls = {'view': args.view, 'paused': False, 'step': 0, 'scale': 1.0}

    def on_key(_window, key, _scancode, action, _mods) -> None:
        if action not in (glfw.PRESS, glfw.REPEAT):
            return
        if glfw.KEY_1 <= key <= glfw.KEY_5:
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
    print(frames.describe())
    for side in sides:
        print(f'{side.name}: {side.changed or "as recorded"}')

    def feed(k: int) -> tuple[np.ndarray, np.ndarray]:
        lum, depth, mask = frames.load(k)
        upload(luminance, lum); upload(depth_texture, depth); upload(mask_texture, mask.astype(np.float32))
        return depth, mask

    loop, k, started = 1, 0, time.perf_counter()
    while not glfw.window_should_close(window):
        if k >= len(frames):                      # the loop's end: its measures, and from no motion again
            for side in sides:
                print(f'loop {loop} {side.name}: {side.measures.report(span, w, len(frames.moves))}', flush=True)
                side.flow.reset()
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
        if controls['step'] < 0:                  # a frame back: the flows measure again from no motion up to it
            target = max(k - 2, 0)
            for side in sides:
                side.flow.reset()
                side.measures = Measures()
            for j in range(target):
                feed(j)
                for side in sides:
                    side.step(frames, j, luminance, depth_texture, mask_texture)
            k = target
        controls['step'] = 0

        depth, mask = feed(k)
        segment = frames.segment(k)
        for side in sides:
            glFinish()
            began = time.perf_counter()
            side.step(frames, k, luminance, depth_texture, mask_texture)
            glFinish()                            # the flow's own time on the GPU, before its read back
            milliseconds = (time.perf_counter() - began) * 1000.0
            xy = side.xy_px(read_back(side.flow.velocity), frames.calls[k]['delta_time'], w)
            track = read_back(side.flow.depth_velocity.difference)
            z = np.abs(read_back(side.flow.depth_velocity.depth_velocity)[..., 0]) * w
            if side.settings.use_mask:
                z = z * mask
            side.measures.add(float(frames.t[k]), segment, xy, track, z, depth, mask, frames.moved, milliseconds)

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
