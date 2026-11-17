"""Play saved frames through the Lucas–Kanade flow's shaders, A beside B, and measure both.

    python scripts/depth_velocity_compare.py FRAMES [--a NAME=VALUE ...] [--b NAME=VALUE ...] [--fast] [--verify]

FRAMES is what depth_velocity_capture saved from one loop of a recording (temp/depth_velocity/RECORDING).
No app and no camera: a window gives the GL context, and two `LucasKanadeFlow3D`, A and B, measure the
saved frames with the settings they were recorded with; `--a` and `--b` override any of them by its
path in the panel (xy.levels, xy.threshold, z.xy_halving, z_gain, ...). The window plays the loop at its
recorded rate, again and again, A on the left, B on the right:

    1  z of the track, red toward the camera, green away     space  pause
    2  z after the threshold, as the fluid gets it           ← →    a frame back or on, paused
    3  the depth                                              + -    brighter, darker
    4  the velocity's direction (x/y)                         esc    quit
    5  the velocity as arrows

The loop splits into a still part and moves, each a slow half then a fast one: by `segments.json`
beside the frames ({"still": [from, to], "moves": [[from, fast from, to], ...]}, seconds; a move as
[from, slow to, fast from, to] leaves out the time between) where there is one, else by the frames' motion (the mean luminance change in the mask): the moves are where it
rises above the still part's for a while, and a move's fast half is where it steps up most.

Each loop prints, for A and B, in the still part: x/y and z as the fluid gets them, at the edge (0–6
px inside the mask) and inside (12 px and more), and z on the table (outside, 16 px and more); per
move, the body's x/y and z in its slow and fast half and fast over slow, how its z follows its real
depth change, and the frames x/y takes to stop after it; the restarts of the depth velocity, and the
flow's time on the GPU per frame; then the one results table's row (`ROW`), with the share of the mask
without a z position and the mask's fit to the image's edges. `--fast` runs one loop without the window, as fast as it goes;
with `--fast`, several captures, `--side NAME:PATH=VALUE,...` or `--out FILE` measure every side on every capture
and print one row a side, x/y beside z: the mean over the captures and its spread, the ms each side comes after A
over the moves, and with `--out` the rows and their stamp (date, commit, captures, each side's settings) added
to that file. A capture's stretches come from `segments.json` in its recording's folder, else beside the capture.
`--verify` compares A's first frames of the x/y solve with its numpy reference instead.
"""

import json
import subprocess
import sys
import time
from argparse import ArgumentParser
from pathlib import Path

import cv2
import glfw
import numpy as np
from OpenGL.GL import (GL_COLOR_BUFFER_BIT, GL_FLOAT, GL_FRAMEBUFFER, GL_R32F, GL_RED, GL_RGB, GL_RGBA32F,
                       GL_TEXTURE_2D, GL_UNSIGNED_BYTE, glBindFramebuffer, glBindTexture,
                       glClear, glClearColor, glFinish, glReadPixels, glTexImage2D, glViewport)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from modules.gl import Context, Fbo, Texture  # noqa: E402
from modules.flow import (FlowUtil, LucasKanadeFlow3D, LucasKanadeFlow3DSettings, Patch, Visualizer,  # noqa: E402
                          lucas_kanade)

WARM_UP: float = 1.0          # s at the start the flows settle in, measured in no part
SHORTEST_PAUSE: float = 1.0   # s of stillness that end a move
RENAMED: dict[str, str] = {'z_time': 'z_smooth', 'z_catch_up_time': 'z_catch_up', 'xy_follow_z_time': 'xy_smooth', 'xy_follow_z': 'xy_smooth', 'xy_regularizer': 'xy_damping',
                           'strength_x': 'x_gain', 'strength_y': 'y_gain', 'strength_z': 'z_gain',
                           'lucas_kanade': 'xy', 'depth_track': 'z', 'solve': 'xy', 'depth_velocity': 'z'}   # in frames saved before
SWITCHED: dict[str, str] = {'x': 'x_gain', 'y': 'y_gain', 'z': 'z_gain'}   # in frames saved with axis switches: off is a gain of 0
GROUPED: dict[str, str] = {'xy_levels': 'xy.levels', 'xy_window': 'xy.window', 'xy_damping': 'xy.damping',
                           'xy_same_surface': 'xy.same_surface', 'xy_threshold': 'xy.threshold',
                           'z_edge': 'z.edge', 'z_window': 'z.window', 'z_tolerance': 'z.tolerance',
                           'z_smooth': 'z.smooth', 'xy_smooth': 'xy.z_smooth',
                           'z_catch_up': 'z.catch_up', 'z_dead_zone': 'z.dead_zone', 'z_threshold': 'z.threshold'}   # in frames saved flat, before the groups
MOVED: float = 0.05          # luminance change of a pixel that moves
MOVED_FRAMES: int = 5         # frames it changes that much in, over the loop
RECORDINGS: Path = Path(__file__).resolve().parent.parent / 'apps' / 'deep_flow' / 'data' / 'recordings'
LAG_REACH: int = 12           # frames a side may lag or lead the first, searched
VIEWS: dict[int, str] = {1: 'z of the track', 2: 'z after the threshold', 3: 'depth', 4: 'velocity colormap', 5: 'velocity arrows'}


def context(visible: bool, width: int, height: int):
    """A GL 4.6 window, current, with the fullscreen quad bound as the app has it."""
    if not glfw.init():
        raise RuntimeError('GLFW did not start')
    glfw.window_hint(glfw.VISIBLE, glfw.TRUE if visible else glfw.FALSE)
    glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 4)
    glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 6)
    glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)
    glfw.window_hint(glfw.RESIZABLE, glfw.FALSE)
    window = glfw.create_window(width, height, 'depth velocity', None, None)
    if not window:
        raise RuntimeError('no GL window')
    glfw.make_context_current(window)
    glfw.swap_interval(1)
    Context().allocate()
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
    z_group = recorded.get('z')
    if isinstance(z_group, dict) and 'xy_smooth' in z_group:          # nested, before it moved to x/y as z_smooth
        recorded = {**recorded, 'z': {k: v for k, v in z_group.items() if k != 'xy_smooth'},
                    'xy': {**recorded.get('xy', {}), 'z_smooth': z_group['xy_smooth']}}
    for key, value in recorded.items():
        key = RENAMED.get(key, key)
        if key in SWITCHED and isinstance(value, bool):     # the old axis switches; a "z" group is a dict
            key, value = SWITCHED[key], 1.0 if value else 0.0
        *groups, name = GROUPED.get(key, key).split('.')
        target: dict = result
        for group in groups:
            target = target.setdefault(group, {})
        target[name] = value
    return result


def setting_owner(settings: LucasKanadeFlow3DSettings, path: str) -> tuple[object, str]:
    """The settings object that holds the setting at `path` (xy.levels) and its name there."""
    *groups, name = path.split('.')
    owner: object = settings
    for group in groups:
        owner = getattr(owner, group, None)
        if owner is None:
            raise SystemExit(f'unknown settings group {group} in {path}')
    return owner, name


def overrides(items: list[str]) -> dict[str, object]:
    defaults = LucasKanadeFlow3DSettings()
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
            raise SystemExit(f'{folder} has no luminance: capture it again with depth_velocity_capture')
        self.calls: list[dict] = json.loads((folder / 'calls.json').read_text())
        self.width, self.height = self.calls[0]['width'], self.calls[0]['height']
        t = np.array([c['t'] for c in self.calls])
        self.t: np.ndarray = t - t[0]
        stamp = folder / 'capture.json'
        self.capture: dict = json.loads(stamp.read_text()) if stamp.exists() else {}
        motion = self.motion()
        given = folder / 'segments.json'                 # beside the capture, as before
        if self.capture.get('recording'):                # the recording's own, written once for all its captures
            own = RECORDINGS / self.capture['recording'] / 'segments.json'
            given = own if own.exists() else given
        if given.exists():
            parts = json.loads(given.read_text())
            self.still: tuple[float, float] = tuple(parts['still'])
            self.moves: list[tuple[float, ...]] = [tuple(move) for move in parts['moves']]
        else:
            self.still, self.moves = split_moves(self.t, motion)

    def __len__(self) -> int:
        return len(self.calls)

    def load(self, k: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """The luminance, the depth and the mask, 0 to 1 (a mask saved as a bool before is 0 or 1). A capture made
        with a depth mask, which gated z, has its depth 0 where that mask is: a pixel without depth has no z and
        no z position, as the gate gave it, and its x/y as before."""
        depth: np.ndarray = np.load(self.folder / f'depth_{k:04d}.npy').astype(np.float32)
        depth_mask: Path = self.folder / f'depth_mask_{k:04d}.npy'
        if depth_mask.exists():
            depth = np.where(np.load(depth_mask), depth, 0.0).astype(np.float32)
        return (np.load(self.folder / f'luminance_{k:04d}.npy').astype(np.float32), depth,
                np.load(self.folder / f'mask_{k:04d}.npy').astype(np.float32))

    def motion(self) -> np.ndarray:
        """Per frame, the mean luminance change in the mask since the frame before (0 to 1). Also
        `moved`: the pixels whose luminance changes by more than MOVED in MOVED_FRAMES frames or more,
        so a still background in the mask can be told from the player."""
        motion, before = np.zeros(len(self)), None
        changes = np.zeros((self.height, self.width), np.int32)
        for k in range(len(self)):
            luminance, mask = np.load(self.folder / f'luminance_{k:04d}.npy').astype(np.float32), np.load(self.folder / f'mask_{k:04d}.npy') > 0.5
            if before is not None:
                change = np.abs(luminance - before)
                changes += change > MOVED
                if mask.any():
                    motion[k] = float(np.mean(change[mask]))
            before = luminance
        self.moved: np.ndarray = changes >= MOVED_FRAMES
        return motion

    def segment(self, k: int) -> str:
        """'still', 'move N slow' or 'move N fast' (N from 1), or '' between them. A move given with four
        moments (from, slow to, fast from, to) leaves out the time between its halves."""
        t = self.t[k]
        if self.still[0] <= t < self.still[1]:
            return 'still'
        for n, move in enumerate(self.moves, 1):
            begin, fast, end = move[0], move[-2], move[-1]
            slow_end = move[1] if len(move) == 4 else fast
            if begin <= t < end:
                return f'move {n} slow' if t < slow_end else f'move {n} fast' if t >= fast else ''
        return ''

    def describe(self) -> str:
        return (f'still {self.still[0]:.1f}-{self.still[1]:.1f} s; '
                + '; '.join(f'move {n} {m[0]:.1f}-{m[-1]:.1f} s, fast from {m[-2]:.1f}' for n, m in enumerate(self.moves, 1)))


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


def frame_regions(segment: str, depth: np.ndarray, mask: np.ndarray, luminance: np.ndarray | None) -> dict:
    """A frame's regions, the same for every flow that measures it: `core` (the mask's pixels with depth, 12 px
    and more inside), `edge` (0 to 6 px inside), `table` (outside, 16 px and more, no further than 0.8 m behind
    the body), `near` (the core's median depth) and `fit` (the image's brightness gradient along the mask's edge,
    in the still part, else None)."""
    fit: float | None = None
    if segment == 'still' and luminance is not None and mask.any():
        cut = mask.astype(np.uint8)
        edge_of_cut = (cut - cv2.erode(cut, np.ones((3, 3), np.uint8))).astype(bool)
        gradient = np.hypot(cv2.Sobel(luminance, cv2.CV_32F, 1, 0, ksize=3), cv2.Sobel(luminance, cv2.CV_32F, 0, 1, ksize=3))
        if edge_of_cut.any():
            fit = float(np.mean(gradient[edge_of_cut]))
    inside = cv2.distanceTransform(mask.astype(np.uint8), cv2.DIST_L2, 3)
    core = mask & (depth > 0) & (inside > 12)
    edge = mask & (depth > 0) & (inside <= 6)
    near = float(np.median(depth[core])) if core.any() else np.nan
    table = np.zeros_like(mask)
    if segment == 'still':
        outside = cv2.distanceTransform((~mask).astype(np.uint8), cv2.DIST_L2, 3)
        table = (~mask) & (depth > 0) & (outside > 16) & (depth < near + 0.8)
    return dict(core=core, edge=edge, near=near, table=table, fit=fit)


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
        self.fluid_z: list[float] = []           # z as the fluid gets it, inside, px per frame
        self.signed_z: list[float] = []          # the same with its sign, over the pixels that move in the loop: the pixels' noise cancels, the moving part's z stays
        self.without_z: list[float] = []         # the share of the mask without a z position, in the parts measured
        self.fit: list[float] = []               # the image's brightness gradient along the mask's edge, in the still part
        self._before: np.ndarray | None = None

    def add(self, t: float, segment: str, xy: np.ndarray, track: np.ndarray, z: np.ndarray, depth: np.ndarray, mask: np.ndarray,
            moved: np.ndarray, milliseconds: float, weight: np.ndarray | None = None,
            luminance: np.ndarray | None = None, signed_z: np.ndarray | None = None, regions: dict | None = None) -> None:
        """`xy` and `z` are as the fluid gets them, in px per frame; `track` is the depth velocity's state; `mask`
        the player's pixels (the mask above 0.5); `moved` the pixels that move in the loop; `weight` the z
        position's, 0 where the motion has no depth for a pixel; `luminance` for the mask's fit to the image's
        edges."""
        if segment and weight is not None and mask.any():
            self.without_z.append(float(np.mean(weight[mask] <= 0.0)))
        if regions is None:
            regions = frame_regions(segment, depth, mask, luminance)
        if regions['fit'] is not None:
            self.fit.append(regions['fit'])
        width = depth.shape[1]
        core, edge, near, table = regions['core'], regions['edge'], regions['near'], regions['table']
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
        self.fluid_z.append(float(np.mean(z[core])) if core.any() else 0.0)
        self.signed_z.append(float(np.mean(signed_z[core & moved])) if signed_z is not None and (core & moved).any() else 0.0)
        moving = core & moved
        self.moving_xy.append(float(np.mean(xy[moving])) if moving.any() else 0.0)
        self.moving_z.append(float(np.mean(track[..., 0][moving])) * width if moving.any() else np.nan)
        if segment == 'still':
            self.mask_share.append(float(np.mean(mask)))
            self.static_share.append(float(np.mean(~moved[mask])) if mask.any() else 0.0)
            self.still['xy edge'].append(xy[edge]); self.still['xy inside'].append(xy[core])
            self.still['z edge'].append(z[edge]); self.still['z inside'].append(z[core])
            self.still['z table'].append(z[table])

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
        lines.append(self.row(segments))
        return '\n    '.join([' | '.join(lines[:5])] + lines[5:])

    def values(self, span: float, width: int) -> dict[str, float]:
        """The row's numbers, x/y beside z, all in px per frame as the fluid gets them: the still part's noise
        (p99) inside and at the edge, the mean inside in the slow and the fast halves, fast over noise; how the
        track's z follows the real depth change over the moves (its share of it and their correlation); the
        share of the mask without a z position, the mask's fit, and the flow's ms per frame. NaN where the
        loop has no such part."""
        def p99(name: str) -> float:
            parts = self.still[name]
            found = np.concatenate(parts) if parts else np.zeros(0)
            return float(np.percentile(found, 99)) if found.size else float('nan')

        def mean(series: list[float], where: np.ndarray) -> float:
            return float(np.mean(np.array(series)[where])) if where.any() else float('nan')

        segments = np.array(self.segments)
        slow = np.array([s.endswith('slow') for s in segments])
        fast = np.array([s.endswith('fast') for s in segments])
        result: dict[str, float] = {}
        for axis, series in (('xy', self.body_xy), ('z', self.fluid_z)):
            noise, fast_mean = p99(f'{axis} inside'), mean(series, fast)
            result[f'{axis} noise'] = noise
            result[f'{axis} noise edge'] = p99(f'{axis} edge')
            result[f'{axis} slow'] = mean(series, slow)
            result[f'{axis} fast'] = fast_mean
            result[f'{axis} fast/noise'] = fast_mean / noise if noise > 0 else float('nan')
        still = segments == 'still'
        part = np.abs(np.array(self.signed_z))            # the moving part's z: the mean of its pixels, signed
        result['z part still'] = float(np.percentile(part[still], 99)) if still.any() else float('nan')
        result['z part fast'] = mean(list(part), fast)
        d = np.array(self.body_depth)
        real = np.zeros_like(d); real[1:] = -(d[1:] - d[:-1]) / (d[:-1] * span) * width
        kernel = np.ones(5) / 5
        real = np.convolve(np.nan_to_num(real), kernel, 'same')
        body_z = np.convolve(np.nan_to_num(np.array(self.body_z)), kernel, 'same')
        moving = slow | fast
        if moving.sum() >= 3 and float(np.dot(real[moving], real[moving])) > 0:
            result['z/real'] = float(np.dot(body_z[moving], real[moving]) / np.dot(real[moving], real[moving]))
            result['z corr'] = float(np.corrcoef(body_z[moving], real[moving])[0, 1])
        else:
            result['z/real'] = result['z corr'] = float('nan')
        for name, shares in self.restarts.items():
            result[f'restarts {name}'] = float(np.mean(shares)) * 100.0 if shares else float('nan')     # % of the core's tracks
        result['no z position'] = float(np.mean(self.without_z)) if self.without_z else float('nan')
        result['fit'] = float(np.mean(self.fit)) if self.fit else float('nan')
        result['ms'] = float(np.median(self.milliseconds))
        return result

    def lag(self, first: 'Measures') -> dict[str, float]:
        """This flow's x/y and z beside `first`'s over the moves, on the pixels that move in the loop (x/y their
        mean length, z their mean with its sign): the share of `first`'s it keeps (`kept`, 1 the same), and the
        ms it comes after it (`lag`): the shift at which the two series fit best, between frames by the parabola
        through the best shift and its neighbors."""
        moving = np.array([bool(s) and s != 'still' for s in self.segments])
        times = np.array(self.times)
        interval = float(np.median(np.diff(times))) * 1000.0 if len(times) > 1 else 0.0
        result: dict[str, float] = {}
        for axis, mine, theirs in (('xy', self.moving_xy, first.moving_xy), ('z', self.signed_z, first.signed_z)):
            a, b = np.nan_to_num(np.array(theirs)), np.nan_to_num(np.array(mine))
            result[f'{axis} lag'] = result[f'{axis} kept'] = float('nan')
            if moving.sum() < 2 * LAG_REACH or len(a) != len(b):
                continue
            power = float(np.dot(a[moving], a[moving]))
            result[f'{axis} kept'] = float(np.dot(a[moving], b[moving])) / power if power > 0 else float('nan')
            a, b = a - a[moving].mean(), b - b[moving].mean()
            index = np.flatnonzero(moving)
            fits: list[float] = []
            for shift in range(-LAG_REACH, LAG_REACH + 1):
                at = index + shift
                held = (at >= 0) & (at < len(b))
                ours, theirs_now = b[at[held]], a[index[held]]
                size = float(np.linalg.norm(ours) * np.linalg.norm(theirs_now))
                fits.append(float(np.dot(theirs_now, ours)) / size if size > 0 else 0.0)
            best = int(np.argmax(fits))
            between = 0.0
            if 0 < best < len(fits) - 1:
                left, top, right = fits[best - 1], fits[best], fits[best + 1]
                bend = left - 2.0 * top + right
                between = 0.5 * (left - right) / bend if bend != 0.0 else 0.0
            result[f'{axis} lag'] = (best - LAG_REACH + between) * interval
        return result

    def row(self, segments: np.ndarray) -> str:
        """The one results table's row: the share of the mask without a z position, the mask's fit, z's
        noise inside and at the edge (the still part's p99), z inside in the slow and the fast halves (their
        mean), fast over noise, x/y at the edge (p99), and the flow's ms per frame."""
        def p99(name: str) -> float:
            parts = self.still[name]
            values = np.concatenate(parts) if parts else np.zeros(0)
            return float(np.percentile(values, 99)) if values.size else 0.0

        fluid_z = np.array(self.fluid_z)
        slow = np.array([s.endswith('slow') for s in segments])
        fast = np.array([s.endswith('fast') for s in segments])
        slow_z = float(np.mean(fluid_z[slow])) if slow.any() else 0.0
        fast_z = float(np.mean(fluid_z[fast])) if fast.any() else 0.0
        noise = p99('z inside')
        without = f'{np.mean(self.without_z) * 100:.1f}%' if self.without_z else 'n/a'
        fit = f'{np.mean(self.fit):.4f}' if self.fit else 'n/a'
        return (f'ROW no z position {without} | fit {fit} | z noise inside / edge {noise:.3f} / {p99("z edge"):.3f} | '
                f'slow {slow_z:.3f} | fast {fast_z:.3f} | fast / noise {fast_z / max(noise, 1e-9):.3f} | '
                f'x/y at edge {p99("xy edge"):.3f} | {np.median(self.milliseconds):.2f} ms')


class Side:
    """One of A and B: its settings, its flow, what it shows and what it measured."""

    def __init__(self, name: str, recorded: dict, changed: dict[str, object], width: int, height: int, show: bool) -> None:
        self.name, self.changed = name, changed
        self.settings = LucasKanadeFlow3DSettings()
        self.settings.update_from_dict(recorded_settings(recorded))
        for path, value in changed.items():
            owner, name = setting_owner(self.settings, path)
            setattr(owner, name, value)
        self.flow = LucasKanadeFlow3D(self.settings)
        self.flow.allocate(width, height)
        self.visualizer = Visualizer() if show else None
        if self.visualizer is not None:
            self.visualizer.allocate()
            self.visualizer.resize(width, height)
        self.measures = Measures()

    def step(self, frames: Frames, k: int, luminance: Texture, depth: Texture, mask: Texture) -> None:
        c = frames.calls[k]
        self.flow.set_input(luminance)
        self.flow.set_mask(mask)
        self.flow.update(depth, c['delta_time'], c['view_span'], tuple(c['depth_step']),
                         depth_range=tuple(c.get('depth_range', (0.0, 0.0))))

    def z_px(self, velocity: np.ndarray, delta_time: float, width: int) -> np.ndarray:
        """The velocity's z (image widths per second, times its gain, weighted by the mask) as
        px per frame: z as the fluid gets it."""
        s = self.settings
        return np.abs(velocity[..., 2] / (s.z_gain if s.z_gain else 1.0)) * delta_time * width

    def xy_px(self, velocity: np.ndarray, delta_time: float, width: int) -> np.ndarray:
        """The velocity's x/y (image widths per second, times the gains) as px per frame."""
        s = self.settings
        vx = velocity[..., 0] / (s.x_gain if s.x_gain else 1.0)
        vy = velocity[..., 1] / (s.y_gain if s.y_gain else 1.0)
        return np.hypot(vx, vy) * delta_time * width

    def show(self, view: int, depth: Texture, scale: float, width: int) -> Texture | None:
        if self.visualizer is None:
            return None
        if view == 5:
            self.visualizer.process_glyphs(self.flow.velocity, scale, 20.0)
        elif view == 4:
            self.visualizer.process_colormap(self.flow.velocity, scale)
        elif view == 3:
            self.visualizer.process_colormap(depth, scale / 2.0, 0)                  # 2 m full
        else:
            tracked = self.flow.depth_velocity
            self.visualizer.process_colormap(tracked.difference if view == 1 else tracked.depth_velocity, scale / 2.0 * width, 0)  # 2 px per frame full
        return self.visualizer.texture


def verify(frames: Frames, changed: dict[str, object], frames_to_check: int) -> int:
    """A's first frames on the GPU against the numpy reference of the x/y solve."""
    context(False, 64, 64)
    w, h = frames.width, frames.height
    side = Side('A', frames.calls[0]['settings'], changed, w, h, False)
    xy = side.settings.xy
    luminance, depth_texture, mask_texture = Texture(), Texture(), Texture()
    previous = None
    last = min(frames_to_check, len(frames)) - 1
    for k in range(last + 1):
        lum, depth, mask = frames.load(k)
        upload(luminance, lum); upload(depth_texture, depth); upload(mask_texture, mask)
        side.step(frames, k, luminance, depth_texture, mask_texture)
        if previous is None:
            previous = lum
            continue
        solve = read_back(side.flow._solve)[..., :2] * np.array([w, h])                  # px per frame
        now, then = (np.round(f * 255.0).astype(np.float64) / 255.0 for f in (lum, previous))    # as the R8 frames hold them
        reference = lucas_kanade(then, now, xy.levels, xy.window, xy.damping)
        previous = lum
        if k % 5 == 0 or k == last:
            dxy = np.linalg.norm(solve - reference, axis=-1)
            print(f'frame {k:2d}: x/y solve differs p50 {np.median(dxy):.3f} p99 {np.percentile(dxy, 99):.3f} px')
    glfw.terminate()
    return 0


COLUMNS: tuple[str, ...] = ('xy noise', 'xy slow', 'xy fast', 'xy fast/noise', 'xy kept', 'xy lag', 'xy noise edge',
                            'z noise', 'z slow', 'z fast', 'z fast/noise', 'z kept', 'z lag', 'z noise edge',
                            'z part still', 'z part fast', 'z/real', 'z corr', 'restarts still', 'restarts moving',
                            'no z position', 'fit', 'ms')


def commit() -> str:
    """The checkout's commit, short, with a + where files differ from it."""
    def git(*words: str) -> str:
        return subprocess.run(['git', *words], capture_output=True, text=True, cwd=Path(__file__).resolve().parent.parent).stdout.strip()
    return git('rev-parse', '--short', 'HEAD') + ('+' if git('status', '--porcelain', '--untracked-files=no') else '')


def measure(captures: list[Path], specs: list[tuple[str, dict[str, object]]], out: Path | None, label: str) -> int:
    """Every side on every capture, one loop each without the window: per side the row's mean over the captures
    and its spread (highest minus lowest), the lag behind the first side, and the whole with its stamp added to
    `out` as one line of JSON."""
    context(False, 64, 64)
    luminance, depth_texture, mask_texture = Texture(), Texture(), Texture()
    per_side: dict[str, list[dict[str, float]]] = {name: [] for name, _ in specs}
    settings: dict[str, dict] = {}
    stamps: list[dict] = []
    described: str = ''
    for folder in captures:
        frames = Frames(folder)
        described = frames.describe()
        w, h = frames.width, frames.height
        span = float(np.median([c['view_span'] for c in frames.calls]))
        sides = [Side(name, frames.calls[0]['settings'], changed, w, h, False) for name, changed in specs]
        for k in range(len(frames)):
            lum, depth, mask = frames.load(k)
            upload(luminance, lum); upload(depth_texture, depth); upload(mask_texture, mask)
            segment = frames.segment(k)
            regions = frame_regions(segment, depth, mask > 0.5, lum)
            for side in sides:
                glFinish()
                began = time.perf_counter()
                side.step(frames, k, luminance, depth_texture, mask_texture)
                glFinish()
                milliseconds = (time.perf_counter() - began) * 1000.0
                velocity = read_back(side.flow.velocity)
                delta_time = frames.calls[k]['delta_time']
                track = read_back(side.flow.depth_velocity.difference)
                weight = read_back(side.flow.z_position)[..., 1] if side is sides[0] and 'depth_range' in frames.calls[k] else None   # the view's, the same for every side
                gain = side.settings.z_gain if side.settings.z_gain else 1.0
                side.measures.add(float(frames.t[k]), segment, side.xy_px(velocity, delta_time, w), track,
                                  side.z_px(velocity, delta_time, w), depth, mask > 0.5, frames.moved, milliseconds, weight, lum,
                                  velocity[..., 2] / gain * delta_time * w, regions)
        for side in sides:
            found = side.measures.values(span, w) | side.measures.lag(sides[0].measures)
            per_side[side.name].append(found)
            settings[side.name] = side.settings.to_dict()
            side.flow.deallocate()
        stamps.append({'capture': folder.name} | frames.capture)
        print(f'{folder.name}: {len(frames)} frames, {described}', flush=True)

    result: dict = {'kind': 'result', 'label': label, 'date': time.strftime('%Y-%m-%d %H:%M'), 'commit': commit(),
                    'captures': stamps, 'stretches': described, 'sides': {}}
    print(f'{"side":<34}' + ''.join(f'{column:>15}' for column in COLUMNS))
    for name, changed in specs:
        table = np.array([[found.get(column, np.nan) for column in COLUMNS] for found in per_side[name]])
        with np.errstate(all='ignore'):
            mean, spread = np.nanmean(table, axis=0), np.nanmax(table, axis=0) - np.nanmin(table, axis=0)
        result['sides'][name] = {'changed': {path: str(value) for path, value in changed.items()}, 'settings': settings[name],
                                 'mean': dict(zip(COLUMNS, mean.tolist())), 'spread': dict(zip(COLUMNS, spread.tolist())),
                                 'captures': per_side[name]}
        print(f'{name:<34}' + ''.join(f'{m:>9.3f}~{s:<5.2f}' for m, s in zip(mean, spread)), flush=True)
    if out is not None:
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open('a', encoding='utf-8') as file:
            file.write(json.dumps(result, default=str) + '\n')
    glfw.terminate()
    return 0


def main() -> int:
    assert __doc__ is not None
    parser = ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('frames', type=Path, nargs='+', help='a capture; several with --fast, measured as one row')
    parser.add_argument('--side', action='append', default=[], metavar='NAME:PATH=VALUE,...',
                        help='with --fast, a further side by its name and its settings; the lag is behind A')
    parser.add_argument('--out', type=Path, help='with --fast, the file each call adds its rows and their stamp to, as a line of JSON')
    parser.add_argument('--label', default='', help='with --out, what this call measures')
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
    if args.fast and (len(args.frames) > 1 or args.side or args.out):
        specs: list[tuple[str, dict[str, object]]] = [('A', overrides(args.a))]
        if args.b:
            specs.append(('B', overrides(args.b)))
        for item in args.side:
            name, _, changes = item.partition(':')
            specs.append((name, overrides([change for change in changes.split(',') if change])))
        return measure(args.frames, specs, args.out, args.label)
    frames = Frames(args.frames[0])
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

    def feed(k: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        lum, depth, mask = frames.load(k)
        upload(luminance, lum); upload(depth_texture, depth); upload(mask_texture, mask)
        return lum, depth, mask > 0.5

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

        lum, depth, mask = feed(k)
        segment = frames.segment(k)
        for side in sides:
            glFinish()
            began = time.perf_counter()
            side.step(frames, k, luminance, depth_texture, mask_texture)
            glFinish()                            # the flow's own time on the GPU, before its read back
            milliseconds = (time.perf_counter() - began) * 1000.0
            velocity = read_back(side.flow.velocity)
            xy = side.xy_px(velocity, frames.calls[k]['delta_time'], w)
            z = side.z_px(velocity, frames.calls[k]['delta_time'], w)       # as the fluid gets it: weighted by the mask
            track = read_back(side.flow.depth_velocity.difference)
            weight = read_back(side.flow.z_position)[..., 1] if 'depth_range' in frames.calls[k] else None
            side.measures.add(float(frames.t[k]), segment, xy, track, z, depth, mask, frames.moved, milliseconds,
                              weight, lum)

        if not args.fast:
            shown = [side.show(controls['view'], depth_texture, controls['scale'], w) for side in sides]
            glBindFramebuffer(GL_FRAMEBUFFER, 0)
            glClearColor(0.0, 0.0, 0.0, 1.0)
            glClear(GL_COLOR_BUFFER_BIT)
            for i, texture in enumerate(shown):
                if texture is not None:
                    glViewport(i * panel_w, 0, panel_w, panel_h)
                    texture.draw()
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
